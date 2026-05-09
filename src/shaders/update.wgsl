struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

struct Uniforms {
  // dt            = frame delta seconds
  // gravity       = attraction strength toward mouse
  // drag          = velocity damping per second
  // boom_strength = repulsion strength; when > gravity, particles explode
  //                 outward from the cursor. CPU drives this when sustained
  //                 clustering tanks the framerate.
  // mouse_world.xyz = cursor projected onto the focal plane
  // mouse_world.w   = boom phase seed; held constant per boom event,
  //                 randomized at each new boom so successive explosions
  //                 scatter in different patterns.
  dt: f32,
  gravity: f32,
  drag: f32,
  boom_strength: f32,
  mouse_world: vec4<f32>,
};

// Persistent gravity wells dropped by right-clicking. count is u32; data
// is a fixed-size array because uniform buffers must have static layouts.
struct Wells {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  data: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<uniform> wells: Wells;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }

  var p = particles[i];

  // Softened-gravity attraction toward the cursor, plus optional outward
  // boom force. Net coefficient = gravity − boom_strength.
  let to_mouse = u.mouse_world.xyz - p.pos.xyz;
  let r2 = dot(to_mouse, to_mouse) + 0.01;
  let dir = to_mouse * inverseSqrt(r2);

  // Per-particle pseudo-random unit-ish vector, parameterized by a phase
  // that the CPU re-rolls on every boom. Same particle gets the same
  // jitter for the duration of one boom (no flicker), but a different
  // jitter on the next one (no two explosions look alike).
  let phase = u.mouse_world.w;
  let f_i = f32(i);
  let jitter = vec3<f32>(
    sin(f_i * 12.9898 + phase * 1.7),
    sin(f_i * 78.2330 + phase * 2.3),
    sin(f_i * 37.7190 + phase * 3.1),
  );

  // Blend jitter into the radial direction *only while the boom is active*.
  // When boom_strength is 0 the perturbation vanishes and gravity stays
  // perfectly radial — normal-play dynamics are untouched.
  let perturb = clamp(u.boom_strength * 0.01, 0.0, 0.5);
  let dir_jitter = normalize(dir + jitter * perturb);

  // Cap the coefficient. Without this, particles at r² ≈ 0.01 during a
  // boom get acceleration ~7800 — well past escape velocity. The clamp
  // leaves normal-play dynamics untouched (gravity coefficient stays
  // far below 30) but defangs the singularity.
  let coeff = clamp((u.gravity - u.boom_strength) / r2, -30.0, 30.0);
  var accel_main = dir_jitter * coeff;

  // Sum forces from each persistent well. Same softened-gravity law as
  // the cursor, but with no boom term — wells are stable attractors.
  for (var k = 0u; k < wells.count; k = k + 1u) {
    let to_well = wells.data[k].xyz - p.pos.xyz;
    let r2_w = dot(to_well, to_well) + 0.01;
    let dir_w = to_well * inverseSqrt(r2_w);
    let coeff_w = clamp(u.gravity / r2_w, 0.0, 30.0);
    accel_main = accel_main + dir_w * coeff_w;
  }

  // Soft bounding sphere. Beyond `BOUND` units from the origin, a linear
  // restoring force kicks in — particles can be kicked outward, but the
  // farther they go the harder they're pulled back. Without this, 1/r²
  // gravity is too weak at distance for them to ever return.
  const BOUND = 5.0;
  const BOUND_K = 4.0;
  let pos_len = length(p.pos.xyz) + 1e-4;
  let overshoot = max(0.0, pos_len - BOUND);
  let accel_bound = -p.pos.xyz * (overshoot * BOUND_K / pos_len);

  let accel = accel_main + accel_bound;

  // Exponential drag — analytically correct for arbitrary dt and drag.
  let new_vel = p.vel.xyz * exp(-u.drag * u.dt) + accel * u.dt;

  // Symplectic Euler: integrate position with the *new* velocity.
  // Conserves energy much better than standard Euler — orbits stay stable
  // instead of slowly spiralling out.
  let new_pos = p.pos.xyz + new_vel * u.dt;

  particles[i].pos = vec4<f32>(new_pos, 1.0);
  particles[i].vel = vec4<f32>(new_vel, 0.0);
}
