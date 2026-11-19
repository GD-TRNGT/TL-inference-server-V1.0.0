# FLUX.3 Klein SP8 VAE decoder 调度结论

更新时间：2026-11-14

## 1. 结论

Galaxy03 的 `TP1×SP8` Ring worker 将 VAE decoder 的图像高度切成 8 份。对 `4096×4096`、Distilled、16 steps、12×10 core grid、160 W 的完整 warmup 后请求，推荐默认调度为：

```text
act_block_h=64
double_buffer=true
```

`double_buffer=true` 是收益的主要来源。默认请求 `act_block_h=64`；在一部分 SP8 decoder 卷积层中，64 不能满足 TTNN 的 tile/每核心输出高度约束，运行时会自动回退到 32。该“64 优先、逐层回退”的路径已完整 warmup 并跑通，正式 decode 为 3.054s；默认保留这一最快实测结果，同时仍允许通过环境变量做诊断覆盖。

小 worker 的边界单独处理：4 卡及以下不预热 `4096×4096`，并在 API 入队前以 HTTP 422 拒绝该尺寸；这些 worker 只预热 `1024×1024` 和 `2048×2048`。8 卡及以上继续保留 `4096×4096` warmup 和已验证的 SP8/TP8 调度。

## 2. 实测条件和结果

固定条件：Galaxy03 的 8×1 physical Ring `(0,1,2,3,11,10,9,8)`、`FLUX3_EXECUTION_PROFILE=sp8`、`FLUX3_CORE_GRID_OVERRIDE=12x10`、`TT_METAL_TDP_LIMIT_WATTS=160`、`TT_MM_THROTTLE_PERF=0`、Distilled、`a black bird`、seed 42、无参考图、4096×4096、16 steps。每组先完成服务完整 warmup（包括无参考图和一张参考图 layout），再发送一条正式 API 请求。

| SP8 decoder 参数 | HTTP | denoise | VAE decode | 相对旧默认 decode |
| --- | ---: | ---: | ---: | ---: |
| 历史默认 `32 / false` | 48.136s | 36.986s | 3.541s | — |
| **`64 / true`（默认）** | **47.806s** | **37.070s** | **3.054s** | **快 13.7%** |
| `64 / false` | 47.971s | 36.862s | 3.466s | 快 2.1% |
| `32 / true` | 47.824s | 37.035s | 3.056s | 快 13.7% |

所有正式请求输出相同的 4096×4096 PNG：

```text
SHA256 5fba1e46f5c466f193a6f3dafc589f640a87f3777b39eab636a53cbb942b8cbe
```

因此优化不改变模型输出；HTTP 的剩余主要时间来自 DiT denoise、PNG 编码与服务路径，VAE 调度只直接缩短 decode 段。

## 3. 为什么 64 会回退为 32

TTNN Conv2d 的 `act_block_h_override` 必须满足 tile 高度和本层 per-core 输出高度的约束。SP8 H-shard 后，某些 decoder 层会出现例如 `359` 或 `35` tiles/core；64 对这些 shape 不是合法覆盖值。运行时会记录：

```text
act_block_h_override 64 is not a valid override ... instead 32 was selected
```

这使 `64 / true` 成为“能用 64 的层使用 64、其余层回退 32”的混合方案。其 decode 为 3.054 秒，较 `32 / true` 的 3.056 秒略快；差异约 0.001 秒，接近测量噪声，但正式默认按已验证的 `64 / true` 保留。

## 4. 各 worker 尺寸的一致策略

这里的“一致”是指所有正式 profile 都启用 double buffer；activation block 的高度由 VAE H 分片和 L1 容量决定。

| 服务 profile | 每 worker 芯片数 | Galaxy worker 数 | DiT 并行 | VAE decoder H 分片 | 默认 decoder 调度 |
| --- | ---: | ---: | --- | --- | --- |
| `legacy-single` | 1 | 32 | TP1 × SP1 | H1 | `64 / true` |
| `sp2` | 2 | 16 | TP1 × SP2 | H2 | `64 / true` |
| `sp4` | 4 | 8 | TP1 × SP4 Ring | H4 | `64 / true`（不预热/不支持 4096×4096） |
| `sp8` | 8 | 4 | TP1 × SP8 Ring | H8 | **`64 / true`，不合法层自动回退 32** |

单卡、SP2、SP4 的 H shard 较大，常规 1024/2048 请求使用 `64 / true`。4 卡及以下不把 4096×4096 放入 warmup，也不接受该请求，从而避免在小 worker 上为不可支持的 VAE geometry 分配 L1。SP8 的每卡图像行数更少、4096 宽 halo convolution 工作集更紧张，因此保持 64 优先但由 TTNN 对不合法层选择 32；8 卡及以上保留 4096 路径。

## 5. 实现与覆盖方式

默认策略由 `tt-metal/models/tt_dit/pipelines/flux3_klein/vae_flux3_klein.py` 的 `default_decoder_conv_schedule()` 集中定义：

```python
H1/H2/H4/H8 -> (64, True)
H8 中不合法的 Conv2d shape 由 TTNN 自动回退为 32
```

仅用于诊断或专项调参时，可以在启动容器时显式覆盖：

```bash
-e FLUX3_VAE_ACT_BLOCK_H=32 \
-e FLUX3_VAE_DOUBLE_BUFFER=true
```

覆盖值必须满足 `act_block_h > 0` 且是 32 的倍数；生产默认不需要设置这两个变量。

## 6. 验证边界

本结论验证的是单个 SP8 worker 的 4096×4096、16-step 热态请求。它不改变 DiT 的 TP/SP 划分、Ring descriptor、VAE 权重、输出格式或 API 契约。服务仍要求在旧 SP8 worker 完全停止并完成目标芯片 reset 后，才重新打开同一组设备进行冷启动验证。
