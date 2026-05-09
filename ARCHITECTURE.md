# Architecture

Detailed breakdown of the rendering pipeline, data flow, and per-component responsibilities.

## High-level overview

aupa is a WebGPU compute + render pipeline. Particle state lives entirely in GPU memory; the CPU's responsibilities are uniform writes, camera math, and event handling. Each frame executes 5–6 GPU passes:

```
[uniforms]──┐
            ↓
   ┌──→ [compute] ───→ [particle buffer]
   │                          │
   │      ┌───────────────────┘
   │      ↓
   │  [fade] ────→ [trail (HDR)] ◀─┐
   │                  │             │
   │      ┌───────────┘             │
   │      ↓                         │
   │  [particles ⊕]──→ [trail] ─────┤
   │                                │
   │      ┌─────────────────────────┘
   │      ↓
   │  [bloom H]─→ [bloom-a (½ res)]
   │                  │
   │      ┌───────────┘
   │      ↓
   │  [bloom V]─→ [bloom-b (½ res)]
   │                          │
   │      ┌───────────────────┘
   │      ↓                          ┌─→ [trail]
   └─→ [composite (ACES + vignette)] ┴─→ swapchain
```

## Particle storage

Each particle is two `vec4<f32>`:

| Field | Bytes | Purpose |
|---|---|---|
| `pos.xyz` | 12 | world position |
| `pos.w` | 4 | reserved (padding for vec4 alignment) |
| `vel.xyz` | 12 | velocity |
| `vel.w` | 4 | reserved |

**32 bytes per particle.** At 1M particles, the buffer is **32 MB**. Allocated once with `STORAGE | VERTEX` usage. The buffer is bound as a `read_write` storage buffer in the compute shader and as a `read` storage buffer in the vertex shader — same buffer, different bindings.

WGSL alignment: `vec3` in arrays gets padded to `vec4` regardless, so `vec4` is used explicitly. The wasted `.w` slot is a documentation choice — naming the unused byte is cheaper than debugging an alignment mismatch later.

## Compute pipeline

### Init shader (`init.wgsl`)

Run once on startup and on **restart universe**. Seeds positions on a uniform sphere using `r = pow(rand, 1/3)` for proper volume distribution. PCG-style integer hash gives uniform `[0, 1)` randoms.

### Update shader (`update.wgsl`)

Run every frame. Per particle:

1. **Cursor force** — softened-gravity `1/(r² + ε)` toward `mouse_world.xyz`. The ε term (Plummer softening, named after a 1911 stellar dynamics paper) caps the peak force as a particle approaches the cursor, avoiding the singularity.
2. **Boom term** — net coefficient `= gravity − boom_strength`. When `boom_strength > gravity`, the sign flips and particles closest to the cursor (highest `1/r²`) get the strongest *outward* kick. Black-hole evaporation, exactly.
3. **Per-particle jitter** — a pseudo-random direction (three offset sines indexed by particle ID and a per-cycle phase from `mouse_world.w`) is blended into the radial direction during booms. Each boom cycle gets a fresh phase, so successive explosions scatter in distinct patterns.
4. **Coefficient cap** — the force coefficient is clamped to `±30`. Without this, particles at `r² ≈ 0.01` during a boom get acceleration ~7800 — well past escape velocity. The clamp leaves normal-play dynamics untouched (gravity coefficient stays far below 30) but defangs the singularity.
5. **Persistent wells** — iterates a uniform `array<vec4<f32>, 8>` of well positions, accumulating attractive force from each. Uniform-array iteration is cheap on the GPU (fixed loop, no divergence).
6. **Bounding sphere** — beyond `r = 5` from origin, a linear restoring force pulls particles back. Required because `1/r²` gravity at distance is too weak to bring particles home after a boom.
7. **Drag** — exponential damping: `vel *= exp(-drag * dt)`. Analytically correct for any `dt` and `drag`.
8. **Symplectic Euler integration** — update velocity first, then position with the *new* velocity. Conserves a discrete approximation of energy; orbits stay bounded for thousands of steps instead of slowly spiralling.

## Render pipeline

### Particle shader (`render.wgsl`)

Drawn as `triangle-list` topology, **6 vertices per particle**. Vertex shader uses `vertex_index / 6` for particle ID and `vertex_index % 6` for which corner of the billboard quad. The corner offset is multiplied by clip-space `w` so it cancels with the perspective divide downstream — sprites are screen-aligned at any depth without instanced rendering or vertex buffers.

Fragment shader:

- **Discard** pixels outside the inscribed circle (avoids square corners; saves blend ops).
- **Gaussian alpha falloff**: `exp(-r² * 3.5)`.
- **Per-particle hue** from `fract(sin(particle_id * 0.08731) * 43758.5453)` — the classic GLSL pseudo-random hash.
- **Cycle hue offset** from `mouse_world.w * 0.05` — every cycle's universe rotates the whole rainbow.
- **Speed-based brightness** boost.
- **Big Bang flash** blends toward bright cycle-tinted near-white during high `boom_strength`.
- **Premultiplied alpha output** for additive blending.

The fragment shader writes to RGBA16Float (HDR), so accumulated bright spots can exceed `1.0` — color is preserved instead of clipping to white.

## Post-processing pipeline

### Trail texture

Persistent canvas-sized **RGBA16Float** texture. Survives across frames. Recreated only on resize. Acts as the canvas for the feedback-render pattern that produces motion trails.

### Fade pass (`fade.wgsl`)

Fullscreen-triangle pass with blend mode:

```
color: srcFactor=zero, dstFactor=constant, op=add
alpha: srcFactor=zero, dstFactor=one,      op=add
```

The blend constant (`pass.setBlendConstant({...})`, set per frame from `config.trailDecay`) multiplies every existing pixel by ~0.93. Fragment shader output is irrelevant since `srcFactor=zero` zeros it. Alpha is left alone so the texture stays valid for sampling.

### Bloom passes (`bloom.wgsl`)

Two passes, separable Gaussian:

| | Source | Destination |
|---|---|---|
| H blur | trail (full-res) | bloom-a (½ res) |
| V blur | bloom-a (½ res) | bloom-b (½ res) |

Both use a 9-tap kernel with `σ ≈ 2` weights (weights 0.20236, 0.17969, 0.12698, 0.07127, 0.03167 — pre-normalized, summing to 1). The `direction` uniform encodes per-tap UV offset; CPU sets it per pass.

Half-resolution output saves **4×** the fragment work — bloom is intrinsically blurry, so subsampling is invisible.

### Composite pass (`composite.wgsl`)

Final pass writing to the swapchain. Per fragment:

1. Sample **trail** (linear filter, full-res).
2. Sample **bloom-b** (linear filter, upsampled from half-res).
3. `combined = trail + bloom * bloom_strength` — added in HDR.
4. **ACES filmic tonemap** (Krzysztof Narkowicz fast approximation):

   ```
   aces(x) = clamp((x * (2.51*x + 0.03)) /
                   (x * (2.43*x + 0.59) + 0.14),
                   0, 1)
   ```

   Five magic constants, one rational expression. Industry-standard since ~2015.
5. **Vignette** with smoothstep edge.
6. Output in RGBA8 (the swapchain's preferred format).

The HDR→LDR transition happens at the ACES step. Inputs to ACES can exceed `1.0`; output is clamped to `[0, 1]`.

## Camera & cursor unprojection

Camera state is spherical (`yaw`, `pitch`, `radius`). Each frame:

```
camPos = (r·cos(pitch)·sin(yaw),
          r·sin(pitch),
          r·cos(pitch)·cos(yaw))
```

The view matrix is built with `mat4.lookAt(view, camPos, [0,0,0], [0,1,0])`. To unproject the cursor into world space (so dropped wells land where the user clicked), we:

1. Place the cursor at `(mouseNdc.x * halfW, mouseNdc.y * halfH, -r, 1)` in **view space** — at depth `r`, in front of the camera, sized to fill the focal plane.
2. Transform through the **inverse view matrix** to get world coordinates.

Reusing the rendering view matrix means basis signs can never drift from what the camera sees — there's exactly one source of truth.

## Configuration & sharing

The `config` object is a plain JavaScript object holding all live-tunable values. Sliders mutate it in place; the frame loop reads from it each tick. No observers, no events, no reactive framework — just polling.

URL-hash encoding uses compact 1–2 char keys (`g`, `d`, `t`, `ps`, `gl`, `b`, `bp`, `bt`). On page load, `URLSearchParams` parses the hash and overwrites matching config keys. On share-button click, the hash is regenerated from current config and copied to clipboard via `navigator.clipboard.writeText`.

`history.replaceState` is used (instead of `pushState`) to avoid polluting the browser history with every share click.

## Per-frame timing budget

Approximate ms/frame on a discrete GPU at 1080p, 1M particles:

| Pass | ms |
|---|---|
| Compute (1M particles, simple force law) | ~0.4 |
| Fade (full-screen, single quad, no kernel) | ~0.1 |
| Particle render (1M sprites × ~12 fragments) | ~5–8 |
| Bloom H + V (2× 9-tap Gaussian, half-res) | ~0.4 |
| Composite (ACES + vignette + 2 samples) | ~0.3 |
| **Total** | **~7–9 ms** |

The bottleneck is overwhelmingly the particle render's fragment shader — every other pass is sub-millisecond. Scaling sprite size ↑ or particle count ↑ hits this first.

## Files

```
src/
├── main.ts                  WebGPU setup, frame loop, event handlers, config
└── shaders/
    ├── init.wgsl            One-shot particle seeding
    ├── update.wgsl          Per-frame physics integration
    ├── render.wgsl          Sprite rendering
    ├── fade.wgsl            Trail fade
    ├── bloom.wgsl           Separable Gaussian blur (H or V)
    └── composite.wgsl       Final tonemap + vignette
index.html                   Markup, CSS, configurator panel
public/og.png                Social-share preview image (user-supplied)
```
