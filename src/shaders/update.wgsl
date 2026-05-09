struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

struct Uniforms {
  // dt = frame delta seconds
  // mouse_world = cursor projected into world space (Sunday hookup)
  // gravity = attraction strength toward mouse
  // drag = velocity damping per second
  dt: f32,
  gravity: f32,
  drag: f32,
  _pad: f32,
  mouse_world: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }

  var p = particles[i];

  // ── Saturday: pure ballistic drift ────────────────────────────────
  // Velocity is integrated into position. No forces yet.
  // ──────────────────────────────────────────────────────────────────
  //
  // ── Sunday TODO (your contribution) ───────────────────────────────
  // Replace the body below with a force model. Suggested shape:
  //   1. let to_mouse = u.mouse_world.xyz - p.pos.xyz;
  //   2. compute distance, clamp to avoid singularity at the cursor
  //   3. accumulate acceleration toward mouse scaled by u.gravity
  //   4. apply drag: vel *= exp(-u.drag * u.dt)
  //   5. integrate: vel += accel * dt; pos += vel * dt
  // ──────────────────────────────────────────────────────────────────
  p.pos = p.pos + p.vel * u.dt;

  particles[i] = p;
}
