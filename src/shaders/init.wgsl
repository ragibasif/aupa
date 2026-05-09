struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

// PCG-style hash → uniform [0, 1)
fn hash(n: u32) -> f32 {
  var x = n;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }

  let r1 = hash(i * 7u + 1u);
  let r2 = hash(i * 7u + 2u);
  let r3 = hash(i * 7u + 3u);

  // Uniform sphere distribution, radius 1
  let theta = r1 * 6.2831853;
  let phi   = acos(2.0 * r2 - 1.0);
  let radius = pow(r3, 0.3333333);

  particles[i].pos = vec4<f32>(
    radius * sin(phi) * cos(theta),
    radius * sin(phi) * sin(theta),
    radius * cos(phi),
    1.0
  );

  // Small tangential-ish initial velocity
  particles[i].vel = vec4<f32>(
    (hash(i * 7u + 4u) - 0.5) * 0.05,
    (hash(i * 7u + 5u) - 0.5) * 0.05,
    (hash(i * 7u + 6u) - 0.5) * 0.05,
    0.0
  );
}
