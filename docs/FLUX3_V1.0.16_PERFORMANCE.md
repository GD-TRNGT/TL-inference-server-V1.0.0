# FLUX.3 Image Klein v1.0.16 — Galaxy 性能记录

测试条件：Distilled、16 steps、seed 42、无参考图、`a black bird`、JPEG/Base64。1024×1024/2048×2048 profile 完成对应分辨率的 warmup 后先发一次不计时 prime，再按 worker 数并发发送请求；HTTP 时间不包含请求排队与首次编译。SP8 4096×4096 使用同一已完整 warmup 的服务：第一笔同步请求作为不计时 prime，随后串行发送四次正式请求。

| 每 worker 卡数 | Profile | 分辨率 | Worker 数 | 成功 | 平均 HTTP | HTTP 范围 | 平均去噪 | 平均 VAE decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | legacy-single / 1×1 | 1024×1024 | 32 | 32/32 | 11.366s | 11.153–11.561s | 9.638s | 1.230s |
| 2 | SP2 / 2×1 | 1024×1024 | 16 | 16/16 | 6.293s | 6.202–6.397s | 5.385s | 0.657s |
| 4 | SP4 / 4×1 Ring | 1024×1024 | 8 | 8/8 | 3.345s | 3.299–3.382s | 2.829s | 0.344s |
| 4 | SP4 / 4×1 Ring | 2048×2048 | 8 | 8/8 | 13.609s | 13.460–13.746s | 11.750s | 1.313s |
| 8 | SP8 / 8×1 Ring | 1024×1024 | 4 | 4/4 | 2.618s | 2.601–2.630s | 1.849s | 0.655s |
| 8 | SP8 / 8×1 Ring | 2048×2048 | 4 | 4/4 | 8.477s | 8.427–8.538s | 7.159s | 1.049s |
| 8 | SP8 / 8×1 Ring（同步串行，prime 排除） | 4096×4096 | 4 | 4/4 | 40.829s | 40.715–41.069s | 37.175s | 3.009s |
| 32 | TP4×SP8 / 4×8 Ring | 1024×1024 | 1 | 3/3 | 2.440s | 2.432–2.452s | 1.416s | 0.914s |

- 所有 FLUX.3 profile 使用物理 **12×10** core grid（环境端点值为 `11,9`）。
- SP8 的 4 个 worker 使用固定约 10 秒错峰启动，不等待前一 worker ready。
- TP4×SP8 已真机确认创建 32-device、4×8 Ring mesh。
- 2048×2048 补测确认：SP8 的 VAE decode 为 `1.049s`，快于 SP4 的 `1.313s`；相较 1024×1024，H8 的固定通信/分片成本已被更大的卷积工作量摊薄。
- SP8 4096×4096 补测：完成 4096-only warmup 后，以同步 API 串行发送请求，每次完整响应后才发送下一次。第一笔 HTTP `115.107s` 为稳定化 prime，不纳入统计；后续四笔 HTTP 为 `40.715s`、`40.791s`、`41.069s`、`40.741s`，对应表中均值和范围。
- 同一服务的并发诊断及串行 prime 曾在设备组 `(24,25,26,27,19,18,17,16)` 上出现慢路径；该现象保留在原始证据中供后续调优，但不进入已稳定的同步主统计。
- 4096 的 Ring SDPA 当前没有 token 长度 `65,616` 的专用 chunk 表，运行使用默认 `(128, 256)`；该状态可能是上述 SP8 4096 性能不均衡的相关因素，尚未证明为唯一根因。
- 原始请求结果和服务日志：1024×1024 在 `/local/lnn/fluxcache/flux3-v116-final/`；2048×2048 补测在 `/local/lnn/fluxcache/flux3-v116-2048/`；SP8 4096×4096 同步主测在 `/local/lnn/fluxcache/flux3-v116-4096-sync/`，其中 `sync_measured_timings.log` 和 `sync_request_{2,3,4,5}.http_s.txt` 是主统计证据；先前的并发诊断在 `/local/lnn/fluxcache/flux3-v116-4096/`。

## 2026-11-15 从头重建镜像复测

镜像 `galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16` 从当前 `/local/lnn/TL-inference-server` 和 `/local/lnn/tt-metal` 源码完整重建，得到镜像 ID `sha256:d08eed45d56c09dbb1a4a61afa0a68bbcfe6c210612061f1a6020052c1acd8b9`。对该新镜像单独创建缓存、完成 SP4 warmup 后进行复测：Distilled、1024×1024、16 steps、seed 42 起递增、无参考图、8 worker 并发、prime 排除。

| 每 worker 卡数 | Profile | 分辨率 | Worker 数 | 成功 | 平均 HTTP | HTTP 范围 | 平均去噪 | 平均 VAE decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | SP4 / 4×1 Ring | 1024×1024 | 8 | 8/8 | 3.342s | 3.308–3.383s | 2.833s | 0.342s |

原始证据：`/local/lnn/fluxcache/v1016-rebuild-sp4/benchmark/`。其中 `prime.http_s` 是不计时 prime，`request_0.http_s` 至 `request_7.http_s` 是正式 HTTP 结果，`measured_device_timings.log` 是对应的设备阶段时间。
