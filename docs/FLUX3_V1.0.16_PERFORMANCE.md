# FLUX.3 Image Klein v1.0.16 — Galaxy 性能记录

测试条件：Distilled、1024×1024、16 steps、seed 42、无参考图、`a black bird`、JPEG/Base64。每个 profile 完成 warmup 后先发一次不计时 prime，再按 worker 数并发发送请求；HTTP 时间不包含请求排队与首次编译。

| 每 worker 卡数 | Profile | Worker 数 | 成功 | 平均 HTTP | HTTP 范围 | 平均去噪 | 平均 VAE decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | legacy-single / 1×1 | 32 | 32/32 | 11.366s | 11.153–11.561s | 9.638s | 1.230s |
| 2 | SP2 / 2×1 | 16 | 16/16 | 6.293s | 6.202–6.397s | 5.385s | 0.657s |
| 4 | SP4 / 4×1 Ring | 8 | 8/8 | 3.345s | 3.299–3.382s | 2.829s | 0.344s |
| 8 | SP8 / 8×1 Ring | 4 | 4/4 | 2.618s | 2.601–2.630s | 1.849s | 0.655s |
| 32 | TP4×SP8 / 4×8 Ring | 1 | 3/3 | 2.440s | 2.432–2.452s | 1.416s | 0.914s |

- 所有 FLUX.3 profile 使用物理 **12×10** core grid（环境端点值为 `11,9`）。
- SP8 的 4 个 worker 使用固定约 10 秒错峰启动，不等待前一 worker ready。
- TP4×SP8 已真机确认创建 32-device、4×8 Ring mesh。
- 原始请求结果和服务日志：`/local/lnn/fluxcache/flux3-v116-final/`。
