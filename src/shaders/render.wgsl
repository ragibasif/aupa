struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

struct ViewProj {
  mat: mat4x4<f32>,
};

// Same layout as Uniforms in update.wgsl — same buffer is bound here.
struct CycleUniforms {
  dt: f32,
  gravity: f32,
  drag: f32,
  boom_strength: f32,
  mouse_world: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> vp: ViewProj;
@group(0) @binding(2) var<uniform> u: CycleUniforms;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) speed: f32,
  @location(1) particle_hue: f32,
  @location(2) phase: f32,
  @location(3) boom: f32,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = particles[i];
  var o: VOut;
  o.clip = vp.mat * vec4<f32>(p.pos.xyz, 1.0);
  o.speed = length(p.vel.xyz);
  // Per-particle pseudo-random hue in [0, 1). Fixed for this particle's
  // identity — every particle wears its own color.
  o.particle_hue = fract(sin(f32(i) * 0.08731) * 43758.5453);
  o.phase = u.mouse_world.w;
  o.boom = u.boom_strength;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // Each particle's hue = its own offset + a cycle-wide rotation. Cycle
  // phase shifts everyone uniformly; the random per-particle base means
  // every cluster looks like a multi-colored nebula.
  let hue = (in.particle_hue + in.phase * 0.05) * 6.283185;
  let color = 0.5 + 0.5 * vec3<f32>(
    sin(hue),
    sin(hue + 2.094),
    sin(hue + 4.189),
  );

  // Speed → brightness boost. Fast particles glow brighter than the
  // slow ambient cluster.
  let brightness = 0.55 + 0.45 * clamp(in.speed * 2.0, 0.0, 1.0);

  // Big Bang flash: while boom_strength is high, blend toward a
  // cycle-tinted near-white. Each boom's flash is a slightly different
  // shade — orange-white this cycle, blue-white next.
  let flash = vec3<f32>(
    0.9  + 0.1 * sin(in.phase * 0.13),
    0.85 + 0.15 * sin(in.phase * 0.17 + 1.0),
    0.7  + 0.3 * sin(in.phase * 0.19 + 2.0),
  );
  let flash_amt = clamp(in.boom * 0.015, 0.0, 1.0);
  let final_color = mix(color * brightness, flash, flash_amt);

  // Premultiplied alpha for additive blending.
  return vec4<f32>(final_color * 0.35, 0.35);
}
