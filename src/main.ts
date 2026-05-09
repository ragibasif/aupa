import { mat4 } from 'gl-matrix';
import initShaderCode from './shaders/init.wgsl?raw';
import updateShaderCode from './shaders/update.wgsl?raw';
import renderShaderCode from './shaders/render.wgsl?raw';

const PARTICLE_COUNT = 1_000_000;

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

  function configureCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.configure({ device, format, alphaMode: 'premultiplied' });
  }
  configureCanvas();
  window.addEventListener('resize', configureCanvas);

  // ── Particle storage. Lives entirely on the GPU. ───────────────────
  const particleBuffer = device.createBuffer({
    label: 'particles',
    size: PARTICLE_COUNT * PARTICLE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
  });

  // Update uniforms: { dt, gravity, drag, _pad, mouse_world: vec4 } = 32 bytes
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
    ],
  });

  // ── Camera: simple orbit around the origin ─────────────────────────
  const proj = mat4.create();
  const view = mat4.create();
  const vp = mat4.create();
  function updateViewProj(t: number) {
    const aspect = canvas.width / canvas.height || 1;
    mat4.perspective(proj, Math.PI / 3, aspect, 0.1, 100);
    const r = 3.0;
    mat4.lookAt(
      view,
      [r * Math.sin(t * 0.2), 0.6, r * Math.cos(t * 0.2)],
      [0, 0, 0],
      [0, 1, 0]
    );
    mat4.multiply(vp, proj, view);
    // gl-matrix's mat4 has a loose ArrayBufferLike type; WebGPU wants a non-shared
    // ArrayBuffer-backed view. The underlying buffer is always plain ArrayBuffer.
    device.queue.writeBuffer(vpUniforms, 0, vp as Float32Array<ArrayBuffer>);
  }

  // ── Frame loop ─────────────────────────────────────────────────────
  let lastTime = performance.now();
  let frames = 0;
  let fpsAccum = 0;

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 1 / 30);
    lastTime = now;
    const t = now / 1000;

    // Update uniforms. mouse_world stays (0,0,0,0) until Sunday.
    device.queue.writeBuffer(
      updateUniforms,
      0,
      new Float32Array([dt, /*gravity*/ 0, /*drag*/ 0, /*_pad*/ 0,
                        /*mouse_world*/ 0, 0, 0, 0])
    );
    updateViewProj(t);

    const enc = device.createCommandEncoder();

    // Compute pass: integrate physics
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(updatePipeline);
      pass.setBindGroup(0, updateBindGroup);
      pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
      pass.end();
    }

    // Render pass: draw points
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [
          {
            view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(renderPipeline);
      pass.setBindGroup(0, renderBindGroup);
      pass.draw(PARTICLE_COUNT);
      pass.end();
    }

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
