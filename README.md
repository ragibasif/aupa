# aupa

> One million GPU-driven particles, cyclical Big Bangs, real-time multi-attractor physics. Built from scratch with WebGPU.

A 3D particle visualizer where particles attract toward your cursor under softened-gravity physics, periodically erupt in randomized "Big Bang" explosions when fragment overdraw threatens framerate, and rebuild new universes with fresh color palettes — repeating the cycle indefinitely.

## Features

- **1M GPU compute particles** — physics integration runs entirely on the GPU via WebGPU compute pipelines; particle state never touches the CPU
- **Cyclical Big Bang dynamics** — adaptive FPS-driven trigger fires outward-impulse cycles, randomizes seed + scatter pattern + color palette on each cycle, then particles rebuild a new "universe" under gravity
- **Multi-attractor physics** — cursor pulls in real time + up to 8 persistent gravity wells (right-click on desktop, double-tap on touch)
- **HDR post-processing pipeline** — RGBA16Float trail texture, half-resolution separable Gaussian bloom (9-tap kernel, σ ≈ 2), ACES filmic tonemap, vignette
- **Soft sprite particles** — billboarded quads with Gaussian alpha falloff, per-particle pseudo-random hue, additive HDR blending
- **Motion trails** — feedback-render pattern with constant-blend fade pass for variable-length comet tails (~0.5s to ~3s)
- **Live configurator** — 8 sliders + 5 cinematic presets (dust / galaxy / nova / chaos / calm) + URL-encoded share links
- **Cross-platform input** — mouse, touch (1-finger cursor / 2-finger orbit+pinch / double-tap wells), keyboard shortcuts
- **~13KB gzipped** — zero runtime UI library, only `gl-matrix` as a third-party dependency

## Quick start

```bash
npm install
npm run dev   # http://localhost:5173
```

For production:

```bash
npm run build       # outputs to dist/
npm run preview     # preview the production build locally
```

## Controls

| Action | Desktop | Touch |
|---|---|---|
| Attract particles | mouse cursor (no click) | one finger |
| Orbit camera | left-click + drag | two-finger drag |
| Zoom | scroll wheel | two-finger pinch |
| Drop persistent gravity well | right-click | double-tap |
| Hide / show UI | press `h` | tap `controls` header to collapse |

## Tech stack

- **WebGPU** — compute pipelines, render pipelines, multi-pass post-processing
- **WGSL** — all shaders (init, update, render, fade, bloom, composite)
- **TypeScript** — strict mode, zero `any` in source code
- **Vite** — dev server with HMR for both TS and WGSL via `?raw` imports
- **gl-matrix** — only runtime dependency (mat4 / vec3 / vec4 helpers)

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the detailed pipeline. In short, each frame runs:

1. **Compute** — softened-gravity physics integration for all 1M particles in parallel
2. **Trail fade** — multiply persistent trail texture by configurable constant via blend op
3. **Particle render** — draw billboarded sprite quads additively over the faded trail
4. **Bloom H + V** — separable 1D Gaussian blur at half-res, ping-pong textures
5. **Composite** — sample trail + bloom, ACES tonemap, vignette, write to swapchain

Particle state lives entirely on the GPU. The CPU's role is uniform updates, camera math, and event handling.

## Tunable parameters

The config panel exposes 8 live-tunable parameters:

| | |
|---|---|
| **physics** | `gravity`, `drag`, `boomPeak`, `boomTrigger` |
| **visuals** | `trailDecay`, `particleSize`, `particleGlow`, `bloom` |

Plus 5 curated presets that snap every slider at once, a URL-encoded share button, and a "save screenshot" button.

Shareable URL format — anyone opening this URL gets that exact configuration:

```
https://your-deploy.vercel.app/#g=2&d=0.6&t=0.93&ps=0.008&gl=0.35&b=1&bp=80&bt=6
```

## Deploy

```bash
npx vercel
```

Vercel auto-detects the Vite config — no setup required. After deploy, drop a representative 1200×630 screenshot at `public/og.png` for social-share previews on Twitter / Slack / iMessage.

## Browser support

| Browser | Status |
|---|---|
| Chrome 121+ | full support |
| Edge 121+ | full support |
| Safari 17+ (macOS / iOS) | full support |
| Firefox | not yet (WebGPU still flagged in stable) |

A non-WebGPU fallback shows a friendly error with browser recommendations.

## File layout

```
src/
├── main.ts                  WebGPU setup, frame loop, event handlers, config
└── shaders/
    ├── init.wgsl            One-shot particle seeding (uniform sphere distribution)
    ├── update.wgsl          Per-frame physics: gravity + boom + drag + bounds
    ├── render.wgsl          Sprite billboard + per-particle color
    ├── fade.wgsl            Trail-texture fade (blend constant only)
    ├── bloom.wgsl           Separable Gaussian blur (H or V via uniform)
    └── composite.wgsl       Trail + bloom + ACES + vignette → swapchain
index.html                   Markup, CSS, configurator panel
public/og.png                (you add this) Social-share preview image
```
