import { mat4, vec3, vec4 } from 'gl-matrix';
import initShaderCode from './shaders/init.wgsl?raw';
import updateShaderCode from './shaders/update.wgsl?raw';
import renderShaderCode from './shaders/render.wgsl?raw';
import fadeShaderCode from './shaders/fade.wgsl?raw';
import bloomShaderCode from './shaders/bloom.wgsl?raw';
import compositeShaderCode from './shaders/composite.wgsl?raw';

const PARTICLE_COUNT = 1_000_000;

// Live-tunable settings. The UI panel mutates this object in place; the
// frame loop reads from it every tick, so changes apply on the next frame.
const config = {
  gravity: 2.0,        // attraction strength toward the cursor
  drag: 0.6,           // velocity damping per second (exponential)
  trailDecay: 0.93,    // motion-trail blend constant (higher = longer trails)
  boomPeak: 80,        // initial repulsion strength of each black-hole evaporation
  boomTrigger: 6,      // seconds of "pressure" required before a boom fires
  particleSize: 0.008, // sprite half-extent in NDC (≈ 8 px on a 1080p canvas)
  particleGlow: 0.35,  // peak alpha at each sprite's center
  bloom: 1.0,          // bloom contribution multiplier (0 = off, 1 = standard, 3 = blown out)
};

// Shareable-URL encoding: compact `key=value` pairs in the URL hash.
// Keys are 1-2 chars to keep the URL short.
const SHARE_KEYS: ReadonlyArray<readonly [string, keyof typeof config]> = [
  ['g',  'gravity'],
  ['d',  'drag'],
  ['t',  'trailDecay'],
  ['ps', 'particleSize'],
  ['gl', 'particleGlow'],
  ['b',  'bloom'],
  ['bp', 'boomPeak'],
  ['bt', 'boomTrigger'],
];

function configToHash(): string {
  const parts: string[] = [];
  for (const [short, full] of SHARE_KEYS) {
    // Trim trailing zeros for readability: 0.93000 → 0.93
    const trimmed = parseFloat(config[full].toFixed(4)).toString();
    parts.push(`${short}=${trimmed}`);
  }
  return parts.join('&');
}

// Read the URL hash on load and overwrite matching config keys.
{
  const hash = window.location.hash.slice(1);
  if (hash) {
    const params = new URLSearchParams(hash);
    for (const [short, full] of SHARE_KEYS) {
      const v = params.get(short);
      if (v !== null) {
        const n = parseFloat(v);
        if (Number.isFinite(n)) config[full] = n;
      }
    }
  }
}

// Hard-coded — these aren't worth surfacing in the UI.
const SLOW_FPS   = 45;  // FPS threshold; below this, pressure accumulates faster
const BOOM_DECAY = 5;   // exponential decay rate of the boom impulse per second

// Each particle is two vec4<f32> = 32 bytes. Must match WGSL struct.
const PARTICLE_BYTES = 32;
const WORKGROUP_SIZE = 64;

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const fallback = document.getElementById('fallback') as HTMLDivElement;
const stats = document.getElementById('stats') as HTMLDivElement;

function fail(msg: string): never {
  fallback.textContent = msg;
  fallback.classList.add('show');
  throw new Error(msg);
}

async function main() {
  if (!navigator.gpu) {
    fail('WebGPU not supported in this browser. Try Chrome, Edge, or Safari 17+.');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fail('No GPU adapter available.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));

  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();

  // Internal pipeline runs in HDR. Trail and bloom textures use 16-bit
  // float so additive blending can push values above 1.0; ACES in the
  // composite pass then maps everything back to the canvas's 8-bit range.
  const HDR_FORMAT: GPUTextureFormat = 'rgba16float';

  // Persistent texture that accumulates particle draws across frames.
  // The composite pass samples this and writes to the swapchain with a
  // vignette, so the trail needs TEXTURE_BINDING usage.
  let trailTexture!: GPUTexture;
  // Half-res ping-pong textures for the separable gaussian bloom.
  let bloomTextureA!: GPUTexture;
  let bloomTextureB!: GPUTexture;
  let compositeBindGroup: GPUBindGroup | null = null;
  let bloomBindGroupH: GPUBindGroup | null = null;
  let bloomBindGroupV: GPUBindGroup | null = null;
  let trailNeedsInit = true;

  function configureCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.configure({ device, format, alphaMode: 'premultiplied' });
    if (trailTexture) trailTexture.destroy();
    trailTexture = device.createTexture({
      label: 'trail',
      size: { width: canvas.width, height: canvas.height },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Bloom textures are half-res — bloom is intrinsically blurry, so
    // sub-sampling is invisible and saves 4× the fragment work.
    const bw = Math.max(1, Math.floor(canvas.width / 2));
    const bh = Math.max(1, Math.floor(canvas.height / 2));
    if (bloomTextureA) bloomTextureA.destroy();
    if (bloomTextureB) bloomTextureB.destroy();
    bloomTextureA = device.createTexture({
      label: 'bloom-a',
      size: { width: bw, height: bh },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    bloomTextureB = device.createTexture({
      label: 'bloom-b',
      size: { width: bw, height: bh },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Bind groups reference these textures, so all of them are stale.
    compositeBindGroup = null;
    bloomBindGroupH = null;
    bloomBindGroupV = null;
    trailNeedsInit = true;
  }
  configureCanvas();
  window.addEventListener('resize', configureCanvas);

  // Mouse cursor in normalized device coords [-1, 1]. y is flipped so up is +y.
  const mouseNdc: [number, number] = [0, 0];

  // Manual orbit camera. Drag to rotate, scroll to zoom.
  const camera = {
    yaw: 0.4,
    pitch: 0.25,
    radius: 3,
  };
  let dragging = false;
  let lastDragX = 0;
  let lastDragY = 0;

  // Persistent gravity wells dropped via right-click. Up to MAX_WELLS.
  const MAX_WELLS = 8;
  const wells: Array<[number, number, number]> = [];

  canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) {
      dragging = true;
      lastDragX = e.clientX;
      lastDragY = e.clientY;
      canvas.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mouseup', (e) => {
    if (e.button === 0 && dragging) {
      dragging = false;
      canvas.style.cursor = '';
    }
  });
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    if (dragging) {
      const dx = e.clientX - lastDragX;
      const dy = e.clientY - lastDragY;
      camera.yaw -= dx * 0.005;
      camera.pitch = Math.max(-1.4, Math.min(1.4, camera.pitch + dy * 0.005));
      lastDragX = e.clientX;
      lastDragY = e.clientY;
    }
    // Cursor NDC always updates so the gravity well still tracks the mouse,
    // even while orbiting.
    mouseNdc[0] =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouseNdc[1] = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
    camera.radius = Math.max(1.2, Math.min(8, camera.radius * factor));
  }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (wells.length < MAX_WELLS) {
      wells.push([mouseWorld[0], mouseWorld[1], mouseWorld[2]]);
    }
  });

  // ── Touch input ────────────────────────────────────────────────────
  // 1 finger = cursor (gravity well), 2 fingers = orbit + pinch zoom,
  // double-tap = drop persistent gravity well.
  let touchMidX = 0;
  let touchMidY = 0;
  let lastPinchDist = 0;
  let touchStartTime = 0;
  let touchStartX = 0;
  let touchStartY = 0;
  let lastTapTime = 0;

  function setCursorFromClient(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    mouseNdc[0] =  ((clientX - rect.left) / rect.width)  * 2 - 1;
    mouseNdc[1] = -(((clientY - rect.top) / rect.height) * 2 - 1);
  }

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (e.touches.length === 1) {
      const t = e.touches[0];
      touchStartTime = performance.now();
      touchStartX = t.clientX;
      touchStartY = t.clientY;
      setCursorFromClient(t.clientX, t.clientY);
    } else if (e.touches.length === 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      touchMidX = (a.clientX + b.clientX) / 2;
      touchMidY = (a.clientY + b.clientY) / 2;
      const dx = b.clientX - a.clientX;
      const dy = b.clientY - a.clientY;
      lastPinchDist = Math.sqrt(dx * dx + dy * dy);
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1) {
      const t = e.touches[0];
      setCursorFromClient(t.clientX, t.clientY);
    } else if (e.touches.length === 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      // Pinch: relative distance change → camera radius
      const dx = b.clientX - a.clientX;
      const dy = b.clientY - a.clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (lastPinchDist > 0 && dist > 0) {
        const factor = lastPinchDist / dist;
        camera.radius = Math.max(1.2, Math.min(8, camera.radius * factor));
      }
      lastPinchDist = dist;
      // Orbit: midpoint motion → yaw/pitch
      const mx = (a.clientX + b.clientX) / 2;
      const my = (a.clientY + b.clientY) / 2;
      camera.yaw -= (mx - touchMidX) * 0.005;
      camera.pitch = Math.max(-1.4, Math.min(1.4, camera.pitch + (my - touchMidY) * 0.005));
      touchMidX = mx;
      touchMidY = my;
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    if (e.touches.length === 0) {
      // Final touch lifted — check whether this was a tap (short, no movement)
      const ct = e.changedTouches[0];
      if (ct) {
        const duration = performance.now() - touchStartTime;
        const dx = ct.clientX - touchStartX;
        const dy = ct.clientY - touchStartY;
        const moved = Math.sqrt(dx * dx + dy * dy);
        if (duration < 300 && moved < 10) {
          const now = performance.now();
          if (now - lastTapTime < 300) {
            // Double-tap: drop a persistent well at the cursor position
            if (wells.length < MAX_WELLS) {
              wells.push([mouseWorld[0], mouseWorld[1], mouseWorld[2]]);
            }
            lastTapTime = 0;
          } else {
            lastTapTime = now;
          }
        }
      }
      lastPinchDist = 0;
    } else if (e.touches.length === 1) {
      // Lifted from 2 to 1 — reset pinch state so we don't see a huge delta
      lastPinchDist = 0;
    }
  }, { passive: false });

  // H key toggles the config panel — useful for clean screenshots and
  // mobile real estate.
  window.addEventListener('keydown', (e) => {
    if (e.key === 'h' || e.key === 'H') {
      const cfg = document.getElementById('config');
      if (cfg) cfg.style.display = cfg.style.display === 'none' ? '' : 'none';
    }
  });

  // Click the panel header to collapse/expand. Default to collapsed on
  // narrow viewports so the panel doesn't dominate phone screens.
  const configEl = document.getElementById('config');
  const configHeader = configEl?.querySelector('h3');
  configHeader?.addEventListener('click', () => {
    configEl?.classList.toggle('collapsed');
  });
  if (window.matchMedia('(max-width: 600px)').matches) {
    configEl?.classList.add('collapsed');
  }

  // ── Particle storage. Lives entirely on the GPU. ───────────────────
  const particleBuffer = device.createBuffer({
    label: 'particles',
    size: PARTICLE_COUNT * PARTICLE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
  });

  // Update uniforms: { dt, gravity, drag, boom_strength, mouse_world: vec4 } = 32 bytes
  const updateUniforms = device.createBuffer({
    label: 'update-uniforms',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Wells uniform: { count: u32, _pad×3, vec4<f32>×8 } = 144 bytes.
  // Two TypedArray views over the same ArrayBuffer let us write a u32
  // header alongside f32 data without extra encoding.
  const wellsBytes = new ArrayBuffer(16 + 8 * 16);
  const wellsCountView = new Uint32Array(wellsBytes, 0, 4);
  const wellsDataView  = new Float32Array(wellsBytes, 16, 8 * 4);
  const wellsBuffer = device.createBuffer({
    label: 'wells',
    size: wellsBytes.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // View-projection matrix: mat4x4<f32> = 64 bytes
  const vpUniforms = device.createBuffer({
    label: 'view-proj',
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Render config uniforms: { particle_size, particle_glow, _pad, _pad } = 16 bytes
  const renderConfigUniforms = device.createBuffer({
    label: 'render-config',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const renderConfigArray = new Float32Array(4);

  // Blur uniforms — one per direction. Could be merged with dynamicOffset
  // but two 16-byte buffers are simpler.
  const blurUniformsH = device.createBuffer({
    label: 'blur-h',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const blurUniformsV = device.createBuffer({
    label: 'blur-v',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const blurArrayH = new Float32Array(4);
  const blurArrayV = new Float32Array(4);

  // Composite uniforms: bloom strength + padding.
  const compositeUniforms = device.createBuffer({
    label: 'composite-uniforms',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const compositeArray = new Float32Array(4);

  // ── Init pipeline: one-shot seed of positions/velocities ───────────
  const initPipeline = device.createComputePipeline({
    label: 'init',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: initShaderCode }),
      entryPoint: 'main',
    },
  });
  const initBindGroup = device.createBindGroup({
    layout: initPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: particleBuffer } }],
  });
  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(initPipeline);
    pass.setBindGroup(0, initBindGroup);
    pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ── Update pipeline: runs every frame ──────────────────────────────
  const updatePipeline = device.createComputePipeline({
    label: 'update',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: updateShaderCode }),
      entryPoint: 'main',
    },
  });
  const updateBindGroup = device.createBindGroup({
    layout: updatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: updateUniforms } },
      { binding: 2, resource: { buffer: wellsBuffer } },
    ],
  });

  // ── Render pipeline: draws particles as soft additive sprites ─────
  // Six vertices per particle (two triangles forming a billboarded quad).
  const renderModule = device.createShaderModule({ code: renderShaderCode });
  const renderPipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: {
      module: renderModule,
      entryPoint: 'fs',
      targets: [
        {
          format: HDR_FORMAT,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: vpUniforms } },
      // Same uniform buffer the compute pass reads — the renderer pulls
      // the current cycle's color phase and the live boom strength out of it.
      { binding: 2, resource: { buffer: updateUniforms } },
      { binding: 3, resource: { buffer: renderConfigUniforms } },
    ],
  });

  // ── Fade pipeline: scales the trail texture by a blend constant ───
  const fadeModule = device.createShaderModule({ code: fadeShaderCode });
  const fadePipeline = device.createRenderPipeline({
    label: 'fade',
    layout: 'auto',
    vertex: { module: fadeModule, entryPoint: 'vs' },
    fragment: {
      module: fadeModule,
      entryPoint: 'fs',
      targets: [
        {
          format: HDR_FORMAT,
          // dst.rgb = 0 * src.rgb + constant.rgb * dst.rgb  → fades existing pixels
          // dst.a   = 0 * src.a   + 1        * dst.a       → leaves alpha alone
          blend: {
            color: { srcFactor: 'zero', dstFactor: 'constant', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one',      operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
  });

  // ── Bloom pipeline: separable 1D gaussian blur, used twice per frame
  const bloomModule = device.createShaderModule({ code: bloomShaderCode });
  const blurPipeline = device.createRenderPipeline({
    label: 'blur',
    layout: 'auto',
    vertex: { module: bloomModule, entryPoint: 'vs' },
    fragment: {
      module: bloomModule,
      entryPoint: 'fs',
      targets: [{ format: HDR_FORMAT }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // ── Composite pipeline: samples trail + bloom, vignette, → swapchain
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });
  const compositeModule = device.createShaderModule({ code: compositeShaderCode });
  const compositePipeline = device.createRenderPipeline({
    label: 'composite',
    layout: 'auto',
    vertex: { module: compositeModule, entryPoint: 'vs' },
    fragment: {
      module: compositeModule,
      entryPoint: 'fs',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // ── Camera: orbit + cursor unprojection via inverse view ──────────
  const proj = mat4.create();
  const view = mat4.create();
  const viewInv = mat4.create();
  const vp = mat4.create();
  const camPos = vec3.create();
  const cursorView = vec4.create();
  const cursorWorld = vec4.create();
  const mouseWorld = vec3.create();

  // Reused upload buffer for update uniforms. 8 floats = 32 bytes.
  const uniformsArray = new Float32Array(8);

  function updateCamera() {
    const aspect = canvas.width / canvas.height || 1;
    const fovY = Math.PI / 3;
    const r = camera.radius;

    // Spherical → Cartesian. yaw rotates in the XZ plane (around world Y),
    // pitch lifts the camera vertically.
    vec3.set(camPos,
      r * Math.cos(camera.pitch) * Math.sin(camera.yaw),
      r * Math.sin(camera.pitch),
      r * Math.cos(camera.pitch) * Math.cos(camera.yaw),
    );
    mat4.perspective(proj, fovY, aspect, 0.1, 100);
    mat4.lookAt(view, camPos as Float32Array, [0, 0, 0], [0, 1, 0]);
    mat4.multiply(vp, proj, view);
    device.queue.writeBuffer(vpUniforms, 0, vp as Float32Array<ArrayBuffer>);

    // Place the cursor in view space (camera at origin, +y up, looking
    // down -z), then transform through the inverse view matrix to land
    // it on the focal plane in world space. Reusing the rendering view
    // matrix means basis signs can't drift from what the camera sees.
    mat4.invert(viewInv, view);
    const halfH = Math.tan(fovY / 2) * r;
    const halfW = halfH * aspect;
    vec4.set(cursorView, mouseNdc[0] * halfW, mouseNdc[1] * halfH, -r, 1);
    vec4.transformMat4(cursorWorld, cursorView, viewInv);
    vec3.set(mouseWorld, cursorWorld[0], cursorWorld[1], cursorWorld[2]);
  }

  // ── Config panel wiring ───────────────────────────────────────────
  // Each slider registers a sync function. When a preset mutates `config`,
  // we walk these to push the new values back into the DOM.
  const sliderSyncs: Array<() => void> = [];
  function bindSlider(
    id: string,
    get: () => number,
    set: (v: number) => void,
    fmt: (v: number) => string,
  ) {
    const input = document.getElementById(id) as HTMLInputElement;
    const valueLabel = document.getElementById(id + '-val') as HTMLSpanElement;
    function sync() {
      const v = get();
      input.value = String(v);
      valueLabel.textContent = fmt(v);
    }
    sync();
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      set(v);
      valueLabel.textContent = fmt(v);
    });
    sliderSyncs.push(sync);
  }
  bindSlider('cfg-gravity',      () => config.gravity,      v => { config.gravity      = v; }, v => v.toFixed(2));
  bindSlider('cfg-drag',         () => config.drag,         v => { config.drag         = v; }, v => v.toFixed(2));
  bindSlider('cfg-trail',        () => config.trailDecay,   v => { config.trailDecay   = v; }, v => v.toFixed(2));
  bindSlider('cfg-particlesize', () => config.particleSize, v => { config.particleSize = v; }, v => v.toFixed(3));
  bindSlider('cfg-glow',         () => config.particleGlow, v => { config.particleGlow = v; }, v => v.toFixed(2));
  bindSlider('cfg-bloom',        () => config.bloom,        v => { config.bloom        = v; }, v => v.toFixed(2));
  bindSlider('cfg-boompeak',     () => config.boomPeak,     v => { config.boomPeak     = v; }, v => v.toFixed(0));
  bindSlider('cfg-boomtrigger',  () => config.boomTrigger,  v => { config.boomTrigger  = v; }, v => v.toFixed(1));

  // ── Presets: snap all sliders at once ─────────────────────────────
  const presets: Record<string, Partial<typeof config>> = {
    dust: {
      gravity: 0.5, drag: 0.2, trailDecay: 0.97,
      particleSize: 0.014, particleGlow: 0.5, bloom: 1.5,
      boomPeak: 50, boomTrigger: 10,
    },
    galaxy: {
      gravity: 1.8, drag: 0.05, trailDecay: 0.96,
      particleSize: 0.008, particleGlow: 0.4, bloom: 1.0,
      boomPeak: 80, boomTrigger: 6,
    },
    nova: {
      gravity: 3.5, drag: 0.4, trailDecay: 0.95,
      particleSize: 0.014, particleGlow: 0.7, bloom: 2.5,
      boomPeak: 180, boomTrigger: 3,
    },
    chaos: {
      gravity: 5.0, drag: 0.05, trailDecay: 0.92,
      particleSize: 0.006, particleGlow: 0.4, bloom: 1.2,
      boomPeak: 120, boomTrigger: 2.5,
    },
    calm: {
      gravity: 0.8, drag: 1.0, trailDecay: 0.85,
      particleSize: 0.008, particleGlow: 0.25, bloom: 0.6,
      boomPeak: 50, boomTrigger: 18,
    },
  };
  document.querySelectorAll<HTMLButtonElement>('.cfg-preset').forEach((btn) => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.preset;
      if (!name) return;
      const p = presets[name];
      if (!p) return;
      Object.assign(config, p);
      for (const sync of sliderSyncs) sync();
    });
  });

  // Restart button: re-run the init compute pass to reseed all particle
  // positions and velocities, and clear the trail texture for a fresh start.
  document.getElementById('cfg-restart')!.addEventListener('click', () => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(initPipeline);
    pass.setBindGroup(0, initBindGroup);
    pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([enc.finish()]);
    trailNeedsInit = true;
    boomStrength = 0;
    pressureTime = 0;
  });

  // Clear wells button: remove all persistent gravity wells.
  document.getElementById('cfg-clear-wells')!.addEventListener('click', () => {
    wells.length = 0;
  });

  // Screenshot button: captures the canvas as PNG. Flag is consumed in
  // the frame loop, where toBlob() runs immediately after submit() —
  // catching the canvas's drawing buffer before the browser presents it.
  let pendingScreenshot = false;
  document.getElementById('cfg-screenshot')!.addEventListener('click', () => {
    pendingScreenshot = true;
  });

  // Share button: encode current slider state into the URL hash and
  // copy the shareable link to the clipboard.
  document.getElementById('cfg-share')!.addEventListener('click', async () => {
    const hash = configToHash();
    history.replaceState(null, '', '#' + hash);
    const url = `${window.location.origin}${window.location.pathname}#${hash}`;
    const btn = document.getElementById('cfg-share') as HTMLButtonElement;
    const originalText = btn.textContent;
    try {
      await navigator.clipboard.writeText(url);
      btn.textContent = 'copied!';
    } catch {
      // Clipboard API needs HTTPS/localhost. Fallback: surface the URL.
      window.prompt('copy this share url:', url);
    }
    setTimeout(() => { btn.textContent = originalText; }, 1200);
  });

  // ── Frame loop ─────────────────────────────────────────────────────
  let lastTime = performance.now();
  let frames = 0;
  let fpsAccum = 0;
  let smoothedDt = 1 / 60;
  let pressureTime = 0;
  let boomStrength = 0;
  let boomPhase = 0;

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 1 / 30);
    lastTime = now;
    const t = now / 1000;

    // Black-hole pressure: rises faster when the framerate is suffering,
    // fires the boom when sustained. The boom decays exponentially so the
    // outward kick dies before particles fly off-screen.
    smoothedDt = smoothedDt * 0.9 + dt * 0.1;
    const isSlow = smoothedDt > 1 / SLOW_FPS;
    pressureTime += dt * (isSlow ? 3 : 1);
    if (pressureTime > config.boomTrigger) {
      boomStrength = config.boomPeak;
      // Fresh seed per boom — held constant for this entire decay so
      // particles get a stable random jitter, not per-frame noise.
      boomPhase = Math.random() * 1000;
      pressureTime = 0;
    }
    boomStrength *= Math.exp(-BOOM_DECAY * dt);
    if (boomStrength < 0.1) boomStrength = 0;

    updateCamera();
    uniformsArray[0] = dt;
    uniformsArray[1] = config.gravity;
    uniformsArray[2] = config.drag;
    uniformsArray[3] = boomStrength;
    uniformsArray[4] = mouseWorld[0];
    uniformsArray[5] = mouseWorld[1];
    uniformsArray[6] = mouseWorld[2];
    uniformsArray[7] = boomPhase;
    device.queue.writeBuffer(updateUniforms, 0, uniformsArray);

    renderConfigArray[0] = config.particleSize;
    renderConfigArray[1] = config.particleGlow;
    device.queue.writeBuffer(renderConfigUniforms, 0, renderConfigArray);

    // Blur step is 1.5 source pixels per tap; 9 taps gives ±6 px radius.
    // We compute in source-uv space, so divide by source dimensions:
    // H blur source is the full-res trail, V blur source is half-res bloomA.
    const STEP = 1.5;
    blurArrayH[0] = STEP / canvas.width;
    blurArrayH[1] = 0;
    blurArrayH[2] = 1.0;
    device.queue.writeBuffer(blurUniformsH, 0, blurArrayH);

    blurArrayV[0] = 0;
    blurArrayV[1] = STEP / Math.max(1, Math.floor(canvas.height / 2));
    blurArrayV[2] = 1.0;
    device.queue.writeBuffer(blurUniformsV, 0, blurArrayV);

    compositeArray[0] = config.bloom;
    device.queue.writeBuffer(compositeUniforms, 0, compositeArray);

    // Pack persistent wells into the uniform buffer. count + xyz per well.
    wellsCountView[0] = wells.length;
    for (let i = 0; i < wells.length; i++) {
      const base = i * 4;
      wellsDataView[base + 0] = wells[i][0];
      wellsDataView[base + 1] = wells[i][1];
      wellsDataView[base + 2] = wells[i][2];
      wellsDataView[base + 3] = 0;
    }
    device.queue.writeBuffer(wellsBuffer, 0, wellsBytes);

    // Lazy-create bind groups whenever they've been invalidated by a
    // texture resize.
    if (!bloomBindGroupH) {
      bloomBindGroupH = device.createBindGroup({
        layout: blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: trailTexture.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: blurUniformsH } },
        ],
      });
    }
    if (!bloomBindGroupV) {
      bloomBindGroupV = device.createBindGroup({
        layout: blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: bloomTextureA.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: blurUniformsV } },
        ],
      });
    }
    if (!compositeBindGroup) {
      compositeBindGroup = device.createBindGroup({
        layout: compositePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: trailTexture.createView() },
          { binding: 1, resource: bloomTextureB.createView() },
          { binding: 2, resource: sampler },
          { binding: 3, resource: { buffer: compositeUniforms } },
        ],
      });
    }

    const enc = device.createCommandEncoder();

    // Compute pass: integrate physics on the GPU.
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(updatePipeline);
      pass.setBindGroup(0, updateBindGroup);
      pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
      pass.end();
    }

    // Initialize the trail texture to black on the first frame after
    // creation/resize. After this, every frame just loads + fades + adds.
    if (trailNeedsInit) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: trailTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
      trailNeedsInit = false;
    }

    // Fade pass: dim every pixel of the trail texture by a constant.
    // This is what makes motion trails decay rather than persist forever.
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: trailTexture.createView(),
          loadOp: 'load',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(fadePipeline);
      const k = config.trailDecay;
      pass.setBlendConstant({ r: k, g: k, b: k, a: 1.0 });
      pass.draw(3);
      pass.end();
    }

    // Particle pass: draw billboarded sprite quads (6 verts per particle)
    // additively over the faded trail.
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: trailTexture.createView(),
          loadOp: 'load',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(renderPipeline);
      pass.setBindGroup(0, renderBindGroup);
      pass.draw(PARTICLE_COUNT * 6);
      pass.end();
    }

    // Bloom H pass: sample trail, blur horizontally, write to bloom-a
    // (which is half-res — the linear sampler does the downsampling).
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: bloomTextureA.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(blurPipeline);
      pass.setBindGroup(0, bloomBindGroupH);
      pass.draw(3);
      pass.end();
    }

    // Bloom V pass: blur bloom-a vertically into bloom-b. Together with
    // the H pass, this is a full 2D gaussian via separability.
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: bloomTextureB.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(blurPipeline);
      pass.setBindGroup(0, bloomBindGroupV);
      pass.draw(3);
      pass.end();
    }

    // Composite pass: sample trail + blurred bloom, vignette, write to
    // the swapchain.
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(compositePipeline);
      pass.setBindGroup(0, compositeBindGroup);
      pass.draw(3);
      pass.end();
    }

    device.queue.submit([enc.finish()]);

    // Screenshot: capture immediately after submit, before the browser
    // presents the canvas. toBlob is async; the snapshot is taken
    // synchronously, encoding happens in the background.
    if (pendingScreenshot) {
      pendingScreenshot = false;
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = `aupa-${Date.now()}.png`;
        link.href = url;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      }, 'image/png');
    }

    // FPS readout
    frames += 1;
    fpsAccum += dt;
    if (fpsAccum >= 0.5) {
      const fps = Math.round(frames / fpsAccum);
      stats.textContent = `${PARTICLE_COUNT.toLocaleString()} particles · ${fps} fps`;
      frames = 0;
      fpsAccum = 0;
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  fail(String(e?.message ?? e));
});
