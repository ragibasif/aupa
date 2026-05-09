// Fullscreen-triangle fade pass. Run once per frame on the trail texture
// before drawing new particles, with a blend mode that scales the
// existing pixels by a constant — roll-your-own motion blur.

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  // Three verts cover the [-1,1]² NDC quad after clipping.
  let xy = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(xy[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  // Output is multiplied by zero in the blend equation. Only the blend
  // constant (set via pass.setBlendConstant on the CPU side) matters.
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
