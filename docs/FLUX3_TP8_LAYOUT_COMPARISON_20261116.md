# FLUX.3 Klein 纯 TP8 布局对比

测试日期：2026-11-16

## 固定测试条件

- Galaxy03，Blackhole，8 卡、1 worker
- Distilled 模型
- 同步 `POST /v1/images/generations`
- prompt：`a black bird`
- `seed=42`，`num_inference_steps=16`
- 无参考图
- `TT_MM_THROTTLE_PERF=0`
- `TT_METAL_TDP_LIMIT_WATTS=160`
- core grid：`12x10`
- 首次 warmup 和首次尺寸 kernel 编译不计入热态性能
- 每次请求等待 HTTP 200 返回后再记录；没有使用异步生成接口，也没有把排队时间作为单请求结果

## HTTP 热态结果

| Profile | Mesh | 拓扑 | 1024×1024 HTTP | 2048×2048 HTTP | 说明 |
|---|---:|---|---:|---:|---|
| `tp8` | 1×8 | Linear | 4.893–4.907 s，均值 4.899 s | 约 20.60 s | 纯 TP8，线性 Fabric |
| `tp8ring` | 1×8 | Ring | 3.623–3.694 s，均值 3.651 s | 15.674–15.742 s，均值 15.708 s | 纯 TP8，Ring |
| `tp8ring8x1` | 8×1 | Ring | 3.630–3.690 s，均值 3.650 s | 15.678–15.699 s，均值 15.688 s | 纯 TP8，转置逻辑 mesh，Ring |

首次 2048×2048 请求会触发尺寸相关 kernel 编译：`1×8 Ring` 约 166.5 s，`8×1 Ring` 约 166.3 s。这部分不是稳定推理时间，不能与热态请求比较。

## 设备阶段结果

| Profile | 分辨率 | Denoise | VAE decode |
|---|---:|---:|---:|
| `tp8` | 1024×1024 | 约 3.65 s | 约 0.857 s |
| `tp8` | 2048×2048 | 约 15.78 s | 约 3.47 s |
| `tp8ring` | 1024×1024 | 2.530 s | 0.715 s |
| `tp8ring` | 2048×2048 | 11.408 s | 2.914 s |
| `tp8ring8x1` | 1024×1024 | 2.531 s | 0.715 s |
| `tp8ring8x1` | 2048×2048 | 11.416 s | 2.915 s |

## 结论

1. Ring 相比 1×8 Linear 明显更快：1024×1024 HTTP 约降低 25.5%，2048×2048 热态 HTTP 约降低 24%。
2. `1×8 Ring` 与 `8×1 Ring` 的差异小于测试波动：1024×1024 基本相同；2048×2048 的均值差约 0.02 s。
3. 推荐纯 TP8 使用显式 profile `FLUX3_EXECUTION_PROFILE=tp8ring`。它保留 1×8 Ring 的稳定路径，不需要把逻辑 mesh 转置到 8×1。
4. `FLUX3_EXECUTION_PROFILE=tp8ring8x1` 保留为可复现的实验 profile。它使用独立 descriptor，日志应显示 `mesh=(8, 1)`、`TP8xSP1` 和 `FABRIC_1D_RING`，不会被自动识别成 SP8。

## 启动示例

```bash
FLUX3_EXECUTION_PROFILE=tp8ring
FLUX3_TP_SIZE=8
FLUX3_TP_MESH=1x8
FLUX3_PARALLELISM=tp8ring
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

转置 Ring 实验只需将 profile、mesh 和 parallelism 同时改为：

```bash
FLUX3_EXECUTION_PROFILE=tp8ring8x1
FLUX3_TP_SIZE=8
FLUX3_TP_MESH=8x1
FLUX3_PARALLELISM=tp8ring8x1
```
