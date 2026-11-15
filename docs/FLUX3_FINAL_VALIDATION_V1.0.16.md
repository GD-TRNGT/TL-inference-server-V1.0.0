# FLUX.3 Image Klein v1.0.16 — Final Galaxy Validation

## Scope

- Image: `galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16`
- Verified image digest: `sha256:04d7aa72a3d8e1e7a8a5af2c822e611bb477d3cd85a1add1222ba81fd02455fb`
- Model: FLUX.3 Image Klein **Distilled**
- Test date: 2026-11-15 (Galaxy03)
- Request: 1024×1024, 16 inference steps, seed 42, prompt `a black bird`, no reference image, JPEG/base64 response.
- Power cap: 160 W per chip.
- Core grid: `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=11,9`, which is the physical **12×10** grid.
- Prompt Upsampler: disabled.

## Method

Each profile was reset and started from the same v1.0.16 candidate image. Timing begins only after the log reports:

```text
All workers ready (model readiness gated by per-runner warmup)
```

A non-measured prime request is then sent.  The measured pass sends one request per worker simultaneously, so the figures below exclude worker queueing and first-run compilation.

## Results

| Devices per worker | Profile / mesh | Worker count | Successful requests | HTTP mean | HTTP range | Mean denoise | Mean VAE decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | legacy-single / 1×1 | 32 | 32 / 32 | 11.366s | 11.153–11.561s | 9.638s | 1.230s |
| 2 | SP2 / 2×1 | 16 | 16 / 16 | 6.293s | 6.202–6.397s | 5.385s | 0.657s |
| 4 | SP4 / 4×1 Ring | 8 | 8 / 8 | 3.345s | 3.299–3.382s | 2.829s | 0.344s |
| 8 | SP8 / 8×1 Ring | 4 | 4 / 4 | 2.618s | 2.601–2.630s | 1.849s | 0.655s |
| 32 | TP4×SP8 / 4×8 Ring | 1 | 3 / 3 | 2.440s | 2.432–2.452s | 1.416s | 0.914s |

The 32-chip worker explicitly created a 32-device mesh and logged:

```text
Created mesh device with 32 devices
loading upstream FLUX.3 Klein model=distilled, TP4xSP8 mesh=(4, 8), topology=Ring
```

## Startup behavior

- 1-card: 32 workers. Worker 0 bootstraps the shared cache before the remaining 31 workers fan out.
- 2-card: 16 SP2 workers.
- 4-card: 8 SP4 workers.
- 8-card: 4 SP8 workers, launched at the configured fixed 10-second interval; a later worker does not wait for the previous worker to become ready.
- 32-card: one TP4×SP8 worker using the full Galaxy 4×8 dual-Ring descriptor.

## Fix included in this version

The v1.0.16 image contains `bh_galaxy_flux3_sp8tp4_ring.textproto`. The runner replaces any inherited per-tray SP4 descriptor for `FLUX3_EXECUTION_PROFILE=sp8tp4` with this full-Galaxy descriptor. This prevents the prior failure where TT-Metal discovered only a 4×1 system mesh and rejected the requested 4×8 mesh.

Raw artifacts are retained on Galaxy03 under:

```text
/local/lnn/fluxcache/flux3-v116-final/
/local/lnn/fluxcache/flux3-v116-final/sp8tp4/
```
