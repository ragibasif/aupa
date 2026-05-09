struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

struct ViewProj {
  mat: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> vp: ViewProj;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) speed: f32,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = particles[i];
  var o: VOut;
  o.clip = vp.mat * vec4<f32>(p.pos.xyz, 1.0);
  o.speed = length(p.vel.xyz);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // speed → color: cool blue for slow, hot near white for fast
  let t = clamp(in.speed * 8.0, 0.0, 1.0);
  let cool = vec3<f32>(0.20, 0.45, 1.00);
  let hot  = vec3<f32>(1.00, 0.85, 0.55);
  let c = mix(cool, hot, t);
  // additive blending: low alpha so overlapping particles bloom
  return vec4<f32>(c * 0.35, 0.35);
}
