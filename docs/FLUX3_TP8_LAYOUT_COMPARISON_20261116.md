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
| `tp8` | 1×8 | Linear | 4.893–4.907 s，均值 4.899 s | 约 20.60 s | 历史对比：纯 TP8，线性 Fabric；已不再作为受支持入口 |
| `tp8ring` | 1×8 | Ring | 3.623–3.694 s，均值 3.651 s | 15.674–15.742 s，均值 15.708 s | 当前唯一受支持的纯 TP8 入口 |
| `tp8ring8x1` | 8×1 | Ring | 3.630–3.690 s，均值 3.650 s | 15.678–15.699 s，均值 15.688 s | 历史转置实验；已不再作为受支持入口 |

首次 2048×2048 请求会触发尺寸相关 kernel 编译：`1×8 Ring` 约 166.5 s，`8×1 Ring` 约 166.3 s。这部分不是稳定推理时间，不能与热态请求比较。

## 设备阶段结果

| Profile | 分辨率 | Denoise | VAE decode |
|---|---:|---:|---:|
| `tp8ring` | 1024×1024 | 2.530 s | 0.715 s |
| `tp8ring` | 2048×2048 | 11.408 s | 2.914 s |
| `tp8`（历史） | 1024×1024 | 约 3.65 s | 约 0.857 s |
| `tp8`（历史） | 2048×2048 | 约 15.78 s | 约 3.47 s |
| `tp8ring8x1`（历史） | 1024×1024 | 2.531 s | 0.715 s |
| `tp8ring8x1`（历史） | 2048×2048 | 11.416 s | 2.915 s |

## 结论

1. Ring 相比 1×8 Linear 明显更快：1024×1024 HTTP 约降低 25.5%，2048×2048 热态 HTTP 约降低 24%。
2. `1×8 Ring` 与 `8×1 Ring` 的差异小于测试波动：1024×1024 基本相同；2048×2048 的均值差约 0.02 s。
3. 纯 TP8 现在只保留显式 profile `FLUX3_EXECUTION_PROFILE=tp8ring`。它使用 1×8 Ring 的稳定路径，不需要把逻辑 mesh 转置到 8×1；`tp8` 和 `tp8ring8x1` 仅作为历史测试记录。
4. 1 worker、纯 TP4 的最优路径继续保留，入口为 `FLUX3_EXECUTION_PROFILE=tp4`，对应 mesh `4×1`；本次收敛纯 TP8 入口不会删除或改变 TP4 逻辑。

## 启动示例

```bash
FLUX3_EXECUTION_PROFILE=tp8ring
FLUX3_TP_SIZE=8
FLUX3_TP_MESH=1x8
FLUX3_PARALLELISM=tp8ring
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

纯 TP4 入口仍为：

```bash
FLUX3_EXECUTION_PROFILE=tp4
FLUX3_TP_SIZE=4
FLUX3_TP_MESH=4x1
FLUX3_PARALLELISM=tp
```
