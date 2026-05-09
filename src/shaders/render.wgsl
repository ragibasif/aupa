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

// Render-only knobs. Lets us tune sprite size and brightness without
// recompiling the shader.
struct RenderConfig {
  particle_size: f32,   // half-extent in NDC; 0.008 ≈ 8 px on a 1080p canvas
  particle_glow: f32,   // peak alpha of each particle's center
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> vp: ViewProj;
@group(0) @binding(2) var<uniform> u: CycleUniforms;
@group(0) @binding(3) var<uniform> rc: RenderConfig;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) speed: f32,
  @location(1) particle_hue: f32,
  @location(2) phase: f32,
  @location(3) boom: f32,
  @location(4) uv: vec2<f32>,  // [-1, 1] across each sprite, for radial alpha
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  // Each particle is a billboarded quad: two triangles, six vertices.
  let particle_id = vid / 6u;
  let corner_id = vid % 6u;

  let p = particles[particle_id];
  let center_clip = vp.mat * vec4<f32>(p.pos.xyz, 1.0);

  // Six corners forming two CCW triangles.
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  let corner = corners[corner_id];

  // Multiply the offset by w so it cancels with the perspective divide.
  // Result: every sprite is the same screen-space size regardless of depth.
  let offset = corner * rc.particle_size * center_clip.w;

  var o: VOut;
  o.clip = vec4<f32>(
    center_clip.x + offset.x,
    center_clip.y + offset.y,
    center_clip.z,
    center_clip.w,
  );
  o.speed = length(p.vel.xyz);
  o.particle_hue = fract(sin(f32(particle_id) * 0.08731) * 43758.5453);
  o.phase = u.mouse_world.w;
  o.boom = u.boom_strength;
  o.uv = corner;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // Discard pixels outside the inscribed circle so each sprite is a soft
  // disc rather than a square. Saves blend ops in the corners.
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }

  // Gaussian-ish radial falloff — bright center, smooth fade to edge.
  let falloff = exp(-r2 * 3.5);

  // Per-particle hue + cycle-wide rotation: rainbow nebula.
  let hue = (in.particle_hue + in.phase * 0.05) * 6.283185;
  let color = 0.5 + 0.5 * vec3<f32>(
    sin(hue),
    sin(hue + 2.094),
    sin(hue + 4.189),
  );

  // Speed → brightness boost. Fast particles glow brighter.
  let brightness = 0.55 + 0.45 * clamp(in.speed * 2.0, 0.0, 1.0);

  // Big Bang flash, cycle-tinted.
  let flash = vec3<f32>(
    0.9  + 0.1 * sin(in.phase * 0.13),
    0.85 + 0.15 * sin(in.phase * 0.17 + 1.0),
    0.7  + 0.3 * sin(in.phase * 0.19 + 2.0),
  );
  let flash_amt = clamp(in.boom * 0.015, 0.0, 1.0);
  let final_color = mix(color * brightness, flash, flash_amt);

  let a = falloff * rc.particle_glow;
  return vec4<f32>(final_color * a, a);
}
