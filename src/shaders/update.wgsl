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

  // Softened-gravity attraction toward the cursor.
  // r2 = |to_mouse|² + ε. The ε term is the Plummer softening length —
  // it caps the force as a particle approaches the cursor, so nothing
  // accelerates to infinity at the singularity. Try halving it for
  // sharper slingshots, or doubling for a gentler swarm.
  let to_mouse = u.mouse_world.xyz - p.pos.xyz;
  let r2 = dot(to_mouse, to_mouse) + 0.01;
  let dir = to_mouse * inverseSqrt(r2);
  let accel = dir * (u.gravity / r2);

  // Exponential drag — analytically correct for arbitrary dt and drag.
  let new_vel = p.vel.xyz * exp(-u.drag * u.dt) + accel * u.dt;

  // Symplectic Euler: integrate position with the *new* velocity.
  // Conserves energy much better than standard Euler — orbits stay stable
  // instead of slowly spiralling out.
  let new_pos = p.pos.xyz + new_vel * u.dt;

  particles[i].pos = vec4<f32>(new_pos, 1.0);
  particles[i].vel = vec4<f32>(new_vel, 0.0);
}
