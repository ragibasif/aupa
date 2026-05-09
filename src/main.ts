import { mat4, vec3, vec4 } from 'gl-matrix';
import initShaderCode from './shaders/init.wgsl?raw';
import updateShaderCode from './shaders/update.wgsl?raw';
import renderShaderCode from './shaders/render.wgsl?raw';
import fadeShaderCode from './shaders/fade.wgsl?raw';

const PARTICLE_COUNT = 1_000_000;

// Live-tunable settings. The UI panel mutates this object in place; the
// frame loop reads from it every tick, so changes apply on the next frame.
const config = {
  gravity: 2.0,       // attraction strength toward the cursor
  drag: 0.6,          // velocity damping per second (exponential)
  trailDecay: 0.93,   // motion-trail blend constant (higher = longer trails)
  boomPeak: 80,       // initial repulsion strength of each black-hole evaporation
  boomTrigger: 6,     // seconds of "pressure" required before a boom fires
};

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

  // Persistent texture that accumulates particle draws across frames —
  // the fade-and-redraw cycle on this is what produces motion trails.
  // Recreated on every resize.
  let trailTexture!: GPUTexture;
  let trailNeedsInit = true;

  function configureCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.configure({
      device,
      format,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    if (trailTexture) trailTexture.destroy();
    trailTexture = device.createTexture({
      label: 'trail',
      size: { width: canvas.width, height: canvas.height },
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    trailNeedsInit = true;
  }
  configureCanvas();
  window.addEventListener('resize', configureCanvas);

  // Mouse cursor in normalized device coords [-1, 1]. y is flipped so up is +y.
  const mouseNdc: [number, number] = [0, 0];
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseNdc[0] =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouseNdc[1] = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  });

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

  // View-projection matrix: mat4x4<f32> = 64 bytes
  const vpUniforms = device.createBuffer({
    label: 'view-proj',
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

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
    ],
  });

  // ── Render pipeline: draws particles as additive points ────────────
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
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'point-list' },
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: vpUniforms } },
      // Same uniform buffer the compute pass reads — the renderer pulls
      // the current cycle's color phase and the live boom strength out of it.
      { binding: 2, resource: { buffer: updateUniforms } },
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
          format,
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

  function updateCamera(t: number) {
    const aspect = canvas.width / canvas.height || 1;
    const fovY = Math.PI / 3;
    const r = 3.0;

    vec3.set(camPos, r * Math.sin(t * 0.2), 0.6, r * Math.cos(t * 0.2));
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
  function bindSlider(
    id: string,
    set: (v: number) => void,
    fmt: (v: number) => string,
  ) {
    const input = document.getElementById(id) as HTMLInputElement;
    const valueLabel = document.getElementById(id + '-val') as HTMLSpanElement;
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      set(v);
      valueLabel.textContent = fmt(v);
    });
  }
  bindSlider('cfg-gravity',     v => { config.gravity     = v; }, v => v.toFixed(2));
  bindSlider('cfg-drag',        v => { config.drag        = v; }, v => v.toFixed(2));
  bindSlider('cfg-trail',       v => { config.trailDecay  = v; }, v => v.toFixed(2));
  bindSlider('cfg-boompeak',    v => { config.boomPeak    = v; }, v => v.toFixed(0));
  bindSlider('cfg-boomtrigger', v => { config.boomTrigger = v; }, v => v.toFixed(1));

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

    updateCamera(t);
    uniformsArray[0] = dt;
    uniformsArray[1] = config.gravity;
    uniformsArray[2] = config.drag;
    uniformsArray[3] = boomStrength;
    uniformsArray[4] = mouseWorld[0];
    uniformsArray[5] = mouseWorld[1];
    uniformsArray[6] = mouseWorld[2];
    uniformsArray[7] = boomPhase;
    device.queue.writeBuffer(updateUniforms, 0, uniformsArray);

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

    // Particle pass: draw all particles additively over the faded trail.
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
      pass.draw(PARTICLE_COUNT);
      pass.end();
    }

    // Present: copy the accumulated trail texture into this frame's
    // swapchain image. (Could also be a fullscreen-quad sample for
    // post-processing; copy is fine when no further work is needed.)
    enc.copyTextureToTexture(
      { texture: trailTexture },
      { texture: ctx.getCurrentTexture() },
      { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
    );

    device.queue.submit([enc.finish()]);

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
