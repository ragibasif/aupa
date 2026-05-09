// Separable 1D gaussian blur. Run twice — once horizontally, once
// vertically — to compose a full 2D blur. The direction uniform is the
// per-tap UV offset; CPU sets it differently for the H vs V pass.

struct BlurUniforms {
  direction: vec2<f32>,  // per-tap UV offset in source space
  intensity: f32,        // multiplier on output (1.0 = identity)
  _pad: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: BlurUniforms;

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
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return o;
}

// 9-tap gaussian, σ ≈ 2 pixels. Weights pre-normalized to sum to 1.
const W0: f32 = 0.20236;
const W1: f32 = 0.17969;
const W2: f32 = 0.12698;
const W3: f32 = 0.07127;
const W4: f32 = 0.03167;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let off = u.direction;
  var c = textureSample(src, samp, in.uv).rgb * W0;
  c += (textureSample(src, samp, in.uv + off * 1.0).rgb +
        textureSample(src, samp, in.uv - off * 1.0).rgb) * W1;
  c += (textureSample(src, samp, in.uv + off * 2.0).rgb +
        textureSample(src, samp, in.uv - off * 2.0).rgb) * W2;
  c += (textureSample(src, samp, in.uv + off * 3.0).rgb +
        textureSample(src, samp, in.uv - off * 3.0).rgb) * W3;
  c += (textureSample(src, samp, in.uv + off * 4.0).rgb +
        textureSample(src, samp, in.uv - off * 4.0).rgb) * W4;
  return vec4<f32>(c * u.intensity, 1.0);
}
