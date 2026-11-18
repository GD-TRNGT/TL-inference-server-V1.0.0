# FLUX.3 Klein 父 Mesh / 子 Mesh 回归记录（2026-11-18）

本轮使用现有镜像 `galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.27`，不构建新镜像。

固定条件：

- Galaxy03，`TT_MM_THROTTLE_PERF=0`
- `TT_METAL_TDP_LIMIT_WATTS=160`
- `FLUX3_CORE_GRID_OVERRIDE=12x10`
- 同步串行请求，`1024x1024`、16 steps、seed 42
- `response_format=file`、`image_return_format=JPEG`
- API key：`turing123`
- 缓存：容器 `/local/ttcache`，宿主 `/local/ttcache/flux-3-image-klein`

## 回归结果

本表中的 `Parent Fabric` 是父 Mesh 做设备放置时使用的 Fabric；`Child topology` 是模型子 Mesh 实际加载时记录的通信拓扑。两者分别记录，避免把父 Mesh 放置方式和模型通信方式混为一谈。

| Profile | 父 Mesh | 子 Mesh | Parent Fabric | Child topology | HTTP | Denoise | VAE decode | JPEG | 状态 |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| `sp2` | 4x8 | 2x1 | 1D Ring | Linear（启动日志） | 6.329s | 5.544s | 0.658s | 成功 | HTTP 200 |
| `tp2` | 4x8 | 2x1 | 1D Linear | Linear | 4.289s | 3.498s | 0.658s | 成功 | HTTP 200 |
| `sp4` | 4x8 | 4x1 | 1D Ring | Ring | 3.327s | 2.857s | 0.343s | 成功 | HTTP 200 |
| `tp4` | 4x8 | 4x1 | 1D Ring | 未单独打印（TP-only） | 4.030s | — | — | 成功 | HTTP 200 |
| `tp4axis1` | 4x8 | 1x4 | 1D Ring | 未单独打印（TP-only） | 5.045s | — | — | 成功 | HTTP 200 |
| `tp2sp2` | 8x4 | 2x2 | 2D | Linear | 4.298s | 3.518s | 0.658s | 成功 | HTTP 200 |
| `sp8` | 8x4 | 8x1 | 1D Ring | Ring | 2.580s | 1.832s | 0.624s | 成功 | HTTP 200 |
| `tp8ring` | 4x8 | 1x8 | 1D Ring | Ring | 3.464s | — | — | 成功 | HTTP 200 |
| `tp2sp4` | 8x4 | 2x4 | 2D | Linear | 2.773s | 2.306s | 0.344s | 成功 | HTTP 200 |
| `tp4sp2` | 8x4 | 4x2 | 2D | Linear | 3.349s | 2.565s | 0.657s | 成功 | HTTP 200 |
| `sp8tp4` | — | 4x8 | SP Ring + TP reduce | — | — | — | — | — | 保持 direct 全 Galaxy，不做父子拆分 |

本轮没有调试或回归 `21:9`、`9:21`。

原始远端记录：`/local/ttcache/flux3_parent_submesh_tests.log`；结果目录：`/local/ttcache/flux3-parent-submesh-v124/`。

## 代码修正

本轮将 `sp2`、`tp2` 纳入 parent profile 表，并让 parent Fabric 跟随 profile：

- `sp2`：`4x8 -> 2x1`，parent Ring；
- `tp2`：`4x8 -> 2x1`，parent Linear；
- 2D profiles：使用 `FABRIC_2D`，不再继承 parent Ring 默认值。

本轮所有列出的父子 Mesh profile 均已完成同步真机回归；TP-only profile 的阶段计时仍受当前日志实现限制，因此只记录 HTTP 总耗时。文档随后同步到远端仓库。
