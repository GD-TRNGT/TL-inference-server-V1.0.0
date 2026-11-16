# FLUX.3 Klein Galaxy Parent/Child Mesh Validation

## Scope

The FLUX.3 Image Klein runner supports an opt-in Galaxy parent mesh followed
by a topology-valid child mesh. The legacy direct per-worker path remains
available with:

```bash
FLUX3_PARENT_MESH_MODE=direct
```

The implementation is based on the current `tt-metal` multi-mesh API,
`MeshDevice.create_submeshes(...)`, and keeps the existing SP/TP model
parallel semantics on the selected child mesh.

| Profile | Parent mesh | Child mesh | Child devices |
| --- | ---: | ---: | ---: |
| `sp4` | `4x8` | `4x1` | 4 |
| `sp8` | `8x4` | `8x1` | 8 |
| `tp8ring` | `4x8` | `1x8` | 8 |

Parent mode uses one process owning the complete 32-chip Galaxy group and
selects child index `0`. The scheduler adds a `::submesh=N` suffix to the
worker identity; runtime setup strips that suffix when building the device
environment and exports `FLUX3_SUBMESH_INDEX`.

## Runtime safeguards

TP8 Ring uses `all_gather_minimal_matmul_async`. For `K=3072` and TP8, each
device receives 12 K tiles. Ring requires `K_tiles_per_device % K_block_size ==
0`; the launch-boundary guard changes an invalid `K_block_size=8` to the
largest valid divisor, `6`. This prevents the previous fatal error:

```text
K_block_size (8) must evenly divide the number of K tiles per device (12)
```

All measurements use:

```bash
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

Requests were sent synchronously to `POST /v1/images/generations` with
1024x1024, 16 steps, seed 42, no reference image, and the distilled model.

## Synchronous validation results

| Profile | Parent/child placement | HTTP | process_request | denoise | VAE decode | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sp4` | `4x8 -> 4x1` | 3.720s | 3.684s | 2.855s | 0.343s | HTTP 200, valid PNG |
| `sp8` | `8x4 -> 8x1` | 2.961s | 2.922s | 1.831s | 0.601s | HTTP 200, valid PNG |
| `tp8ring` | `4x8 -> 1x8` | 3.985s | 3.943s | — | — | HTTP 200, valid PNG |

The TP8 request completed after the Ring K-block fix and no new Ring
`K_block_size` fatal occurred. The earlier fatal lines in the container log
belong to the pre-fix attempt.

## Source versions

```text
TL-inference-server commit: 55efc927
tt-metal commit:            fe56709ca9
tag:                         flux3-parentmesh-tp8ring-v0.1.0
```

The current `tt-metal` main reference inspected on Galaxy03 was
`d69396a4d5d2c4432ce577de260dfa9f7e84935f`.

The formal Docker image was intentionally not rebuilt in this validation
pass; the running tests used the existing image with the checked-out server
and temporary in-container `tt-metal` source injection for the TP8 fix.
