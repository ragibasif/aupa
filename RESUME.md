# Resume Bullet Points — aupa

ATS-friendly resume bullets for the aupa WebGPU particle visualizer project.

## Detailed bullets (technical depth, one-liner each)

- Built a WebGPU 3D particle visualizer rendering 1,000,000 GPU-resident particles at 60 fps using parallel compute pipelines for physics integration and a multi-pass post-processing chain (motion trails, half-resolution separable Gaussian bloom, ACES filmic tonemap, vignette).

- Designed an adaptive performance-recovery system that detects sustained framerate drops via exponentially-smoothed dt sampling and fires randomized outward-impulse cycles to scatter dense particle clusters, keeping perceived FPS stable under heavy fragment overdraw.

- Implemented a softened-gravity (1/(r²+ε)) physics integrator with symplectic Euler in WGSL compute shaders, including coefficient clamping for numerical stability, a soft bounding sphere to prevent particle escape, and force summation across 8 persistent gravity wells plus a real-time cursor.

- Architected an HDR rendering pipeline using RGBA16Float ping-pong textures so additive blending exceeds [0,1] before tonemap, producing cinematic highlight rolloff and color preservation instead of hard white-clipping.

- Engineered cross-platform input handling (mouse drag-to-orbit, two-finger pinch-zoom + midpoint-orbit, double-tap drop with timing/movement thresholds) without external libraries; works on desktop, iOS Safari, and Android Chrome.

- Shipped a 13 KB gzipped production bundle (TypeScript strict mode + WGSL, zero UI framework) with live-tunable configurator, URL-encoded shareable presets, canvas-to-blob screenshot export, and a mobile-responsive collapsible panel.

## Compact bullets (for tighter resume formats)

- Built WebGPU 3D particle visualizer rendering 1M GPU-resident particles at 60fps using compute pipelines and a 5-pass HDR post-processing chain.

- Implemented physics integrator (softened gravity, symplectic Euler, multi-attractor force summation, bounded universe) entirely in WGSL compute shaders.

- Architected HDR rendering pipeline with RGBA16Float ping-pong textures for cinematic highlight rolloff via ACES filmic tonemap.

- Designed adaptive performance-recovery system that detects framerate pressure via smoothed-dt and triggers reset cycles to maintain 60fps under heavy GPU contention.

- Engineered unified mouse + touch + keyboard input (orbit, pinch-zoom, double-tap) without external libraries.

- Shipped 13 KB gzipped TypeScript + WGSL bundle with URL-shareable presets, screenshot export, and zero-config Vercel deploy.

## Tech-stack tags

`WebGPU` · `WGSL` · `TypeScript` · `Vite` · `GPU compute shaders` · `HDR rendering` · `ACES tonemapping` · `Separable Gaussian bloom` · `Symplectic Euler integration`

## One-line summary (for resume header or portfolio caption)

> 1,000,000-particle WebGPU visualizer with cyclical Big Bang dynamics, real-time multi-attractor physics, and a full HDR + bloom + ACES post-processing pipeline — 13 KB gzipped, zero graphics dependencies.

## Quantifiable facts (for inline use)

- **1,000,000** particles rendered per frame
- **60 fps** target framerate sustained
- **13 KB** gzipped production bundle
- **5-pass** post-processing pipeline (fade, particles, bloom H, bloom V, composite)
- **8** persistent gravity wells supported per scene
- **9-tap** Gaussian kernel for bloom (separable, σ ≈ 2)
- **0** UI framework dependencies
- **1** runtime dependency (`gl-matrix`, ~6 KB)
- **6** WGSL shader files (init, update, render, fade, bloom, composite)
