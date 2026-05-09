// Final compositing pass. Samples the trail texture and the blurred
// bloom texture, sums them, applies vignette, outputs to the swapchain.

struct CompositeUniforms {
  bloom_strength: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var trail: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: CompositeUniforms;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  let xy = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let p = xy[vid];
  var o: VOut;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  // y-flip: textures are top-down, NDC is bottom-up.
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return o;
}

// ACES filmic tonemap (Krzysztof Narkowicz fast approximation).
// Maps potentially-HDR linear colors to a film-like [0,1] range with a
// soft-shoulder highlight rolloff and gentle toe in the shadows.
fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
               vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let trail_color = textureSample(trail, samp, in.uv).rgb;
  let bloom_color = textureSample(bloom, samp, in.uv).rgb;

  // Sum in HDR — values can exceed 1.0 here because trail and bloom
  // textures are RGBA16Float.
  let combined = trail_color + bloom_color * u.bloom_strength;
  let tonemapped = aces(combined);

  // Vignette: smooth darkening from a comfortable interior radius
  // outward to the corners. Applied AFTER tonemap so it isn't
  // pulled back up by the curve.
  let centered = in.uv - 0.5;
  let dist = length(centered) * 1.41;
  let vignette = 1.0 - smoothstep(0.55, 1.1, dist);

  return vec4<f32>(tonemapped * vignette, 1.0);
}
