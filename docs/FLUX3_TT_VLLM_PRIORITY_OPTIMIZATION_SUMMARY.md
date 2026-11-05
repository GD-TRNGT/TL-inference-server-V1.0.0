# FLUX.3 TT 按 vLLM-Omni 优先级实施的性能优化总结

更新时间：2026-11-05  
目标机器：`10.12.11.2 / my01-galaxy02`  
开发分支：`flux-3-image-klein`

## 1. 优化目标

本轮优化参考 vLLM-Omni 的模型生命周期和采样执行方式，按照以下优先级改造 TT FLUX.3：

1. 每个 worker 常驻 Prompt Encoder、Transformer 和 VAE Decoder。
2. 缓存相同步数使用的 Scheduler/Flow Schedule。
3. 初始 latent 只从 Host 上传 TT 设备一次。
4. CFG、Euler update 和整个 denoise latent 尽可能保留在 TT 设备端。
5. 最终 latent 直接交给 TT VAE Decoder，减少中间 Host/Device 往返。
6. 设备内存不足时能够切回旧的逐请求加载方式。

本轮没有修改模型权重，也没有把模型复制进 Docker image。Base 和 Distilled 模型仍通过宿主机目录只读挂载：

```text
/local/models/flux-3-image-klein-base
/local/models/flux-3-image-klein
```

## 2. 优化前的现状

### 2.1 模型组件按请求创建和加载

优化前，`TTFlux3ImageKleinRunner.run()` 调用 pipeline 时没有传入已经创建的模型组件。

因此每个 HTTP 请求进入 `Flux3T2IPipeline.run()` 后，都可能重新执行：

```text
创建并加载 Qwen3-VL Prompt Encoder
  -> 创建并加载 FLUX.3 Transformer
  -> 创建并加载 VAE Decoder
  -> 执行本次推理
```

即使权重转换 cache 已存在，仍会产生 cache 读取、TTNN 参数对象创建、设备权重装载和 Python 对象初始化开销。

在 32 个 worker 并发时，这些重复工作还会放大主机 CPU、内存和文件 I/O 竞争。

### 2.2 denoise loop 每一步经过 Host

优化前的多步去噪流程可以简化为：

```text
Host image_tokens
  -> 每一步重新 H2D
  -> TT Transformer forward
  -> 每一步将 model_output D2H
  -> Host 上执行 CFG
  -> Host 上执行 Euler update
  -> 得到下一步 Host image_tokens
  -> 下一步再次 H2D
```

如果请求使用 16 步，就发生约 16 轮 latent/output 往返；Base 使用 50 步时，往返次数更多。

Base 还需要 conditional 和 unconditional 两次 Transformer forward，再在 Host 上合成 CFG 输出。

### 2.3 Schedule 每次请求重新生成

即使连续请求使用相同的 `num_inference_steps`，Flow Schedule 的 timesteps 和 sigmas 仍会在每次请求中重新创建。

该开销相对 Transformer 较小，但属于可以直接消除的重复计算。

### 2.4 优化前性能基线

Distilled 模型参数：

```text
模型：/local/models/flux-3-image-klein
分辨率：1024×1024
步数：16
worker：32
请求：200 个客户端同时提交
```

优化前结果：

| 指标 | 优化前 |
|---|---:|
| 成功请求 | 200/200 |
| 总墙钟时间 | 132.499 s |
| 整机吞吐 | 1.5094 images/s |
| 平均响应时间 | 73.215 s |
| P50 | 78.752 s |
| P95 | 121.740 s |
| 单请求历史热态耗时 | 约 17.45 s |

## 3. 优化方案总览

优化后的主执行链路为：

```text
worker 初始化
  -> 加载一次 Prompt Encoder
  -> 加载一次 Transformer
  -> 加载一次 VAE Decoder
  -> 三个组件在 worker 生命周期内常驻

请求进入
  -> Prompt Encoder on TT
  -> Host 创建初始 noise/latent
  -> 一次 H2D，得到 image_tt
  -> 多步 denoise 全程保留 image_tt
       -> Transformer forward on TT
       -> CFG on TT（Base）
       -> Euler update on TT
  -> 最终 image_tt 直接进入 VAE
  -> VAE decode on TT
  -> 最终图片一次 D2H
  -> PNG/JPEG 编码和 HTTP 返回
```

## 4. 第一步：让三个模型组件按 worker 常驻

### 4.1 修改文件

```text
/local/lnn/TL-inference-server/
  tt-media-server/tt_model_runners/dit_runners.py
```

### 4.2 增加常驻对象

在 `TTFlux3ImageKleinRunner.__init__()` 中增加：

```python
self._prompt_encoder = None
self._transformer = None
self._vae_decoder = None
self._component_residency = os.environ.get(
    "FLUX3_COMPONENT_RESIDENCY", "resident"
).strip().lower()
```

### 4.3 worker 初始化时加载

`create_pipeline()` 在 worker 初始化并打开 TT 设备后执行一次：

```python
self._prompt_encoder = pipeline.create_prompt_encoder(...)

self._transformer = pipeline.create_transformer(...)
pipeline.load_transformer_weights(self._transformer)

self._vae_decoder = pipeline.create_vae_decoder(...)
pipeline.load_vae_weights(self._vae_decoder)
```

加载完成后日志输出：

```text
Device N: all FLUX.3 components are resident
```

32-worker 服务启动时，worker 0～31 均完成三个组件的常驻加载。

### 4.4 请求复用常驻对象

`TTFlux3ImageKleinRunner.run()` 调用 pipeline 时显式传入：

```python
pipeline.run(
    ...,
    prompt_encoder=self._prompt_encoder,
    transformer=self._transformer,
    vae_decoder=self._vae_decoder,
)
```

请求期间不再重新创建和加载三个模型组件。

从第一张优化后请求开始统计服务日志：

```text
组件/cache 重复加载事件：0
worker error_count：0
```

## 5. 第二步：保留逐请求加载回退模式

常驻模式是默认方式：

```bash
-e FLUX3_COMPONENT_RESIDENCY=resident
```

如果未来某个设备配置无法同时容纳 Prompt Encoder、Transformer 和 VAE，可以切回原来的逐请求加载方式：

```bash
-e FLUX3_COMPONENT_RESIDENCY=per_request
```

两种模式的区别：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `resident` | worker 初始化时加载一次，请求之间复用 | 默认模式，追求低延迟和高吞吐 |
| `per_request` | 每个请求按需创建和加载组件 | 设备内存无法容纳全部组件时回退 |

回退模式复用同一套 pipeline，不维护第二份模型实现。

Dockerfile 中增加默认值：

```dockerfile
ENV FLUX3_COMPONENT_RESIDENCY=resident
```

文件位置：

```text
/local/lnn/TL-inference-server/tt-media-server/Dockerfile
```

## 6. 第三步：缓存 Flow Schedule

### 6.1 修改文件

```text
/local/lnn/tt-metal/
  models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

### 6.2 实现方式

在 `Flux3T2IPipeline` 中增加：

```python
self._schedule_cache: dict[int, Flux3FlowSchedule] = {}
```

`prepare_schedule()` 按 `request.steps` 查找缓存：

```text
16 steps -> 复用同一套 timesteps/sigmas
50 steps -> 复用另一套 timesteps/sigmas
```

只有首次出现新步数时才调用 `make_flow_schedule()`。

## 7. 第四步：将 denoise latent 保留在 TT 设备端

### 7.1 初始 latent 只上传一次

优化前在 denoise loop 内每一步执行 Host 到 TT 的转换。

优化后在进入循环前执行一次：

```python
image_tokens = self.initialize_image_tokens(request)
image_tt = from_torch(
    image_tokens.to(torch.bfloat16),
    device=device,
    dtype=ttnn.bfloat16,
)
```

整个 loop 始终使用同一个设备端 `image_tt`。

### 7.2 Transformer 输出不再逐步传回 Host

`predict()` 直接返回 TT tensor：

```python
return transformer.forward(...)
```

不再在每个 step 调用 `_to_host_tensor()`。

## 8. 第五步：在 TT 设备端执行 CFG

Base 模型使用 conditional/unconditional 双分支 CFG。

优化前：

```python
model_output = unconditional_output + guidance * (
    conditional_output - unconditional_output
)
```

这段计算在 Host tensor 上完成。

优化后使用 TTNN：

```python
model_output = ttnn.subtract(
    conditional_output,
    unconditional_output,
)
ttnn.multiply_(model_output, float(request.guidance))
ttnn.add_(model_output, unconditional_output)
```

Distilled 的 flow-map 路径不需要双分支 CFG，直接使用 conditional output。

## 9. 第六步：在 TT 设备端执行 Euler update

Euler update 的作用是根据当前模型预测和相邻两个 sigma 更新 latent：

```text
latent_next = latent_current
              + model_output × (sigma_next - sigma_current)
```

优化后使用 TTNN 原地运算：

```python
sigma_delta = float(
    (schedule.sigmas[step + 1] - schedule.sigmas[step]).item()
)
ttnn.multiply_(model_output, sigma_delta)
ttnn.add_(image_tt, model_output)
```

这样 `image_tt` 在第 0 步到最后一步之间一直位于 TT 设备。

## 10. 第七步：最终 latent 直接进入 VAE

优化前最后一步先把 latent 传回 Host，再重新上传给 VAE。

优化后直接调用：

```python
tt_latents = vae_decoder.preprocess_and_unpatchify(
    image_tt,
    height=latent_height,
    width=latent_width,
)
decoded_tt = vae_decoder.forward(tt_latents)
```

仅在 VAE 输出最终图片时执行设备到 Host 的复制。

## 11. 修改文件和提交

### 11.1 TL-inference-server

| 文件 | 修改内容 |
|---|---|
| `tt-media-server/tt_model_runners/dit_runners.py` | 三组件常驻、请求复用、环境变量和回退模式 |
| `tt-media-server/Dockerfile` | 默认设置 `FLUX3_COMPONENT_RESIDENCY=resident` |

功能提交：

```text
e46186a9160d48344ec1f706f6ff4efd6f6c53ba
Keep FLUX.3 pipeline components resident per worker
```

### 11.2 tt-metal

| 文件 | 修改内容 |
|---|---|
| `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | Prompt Encoder 创建入口、schedule 缓存、设备端 CFG/Euler/latent、VAE 直连 |

功能提交：

```text
3ff5dc122dae38d173870cf5525e9ed96d877c33
Optimize FLUX.3 device-resident denoise loop
```

## 12. 验证方法

### 12.1 基础正确性

验证内容包括：

- Python 语法检查通过；
- `git diff --check` 通过；
- 32 个 worker 全部完成常驻加载和 warmup；
- `model_ready=true`；
- `canary_state=healthy`；
- 所有 worker `error_count=0`；
- 请求期间没有重复的模型/cache 加载事件；
- 返回文件确认为有效 1024×1024 PNG；
- `a black bird` 生成图片与提示词一致。

### 12.2 Distilled 性能测试

```text
模型：flux-3-image-klein
分辨率：1024×1024
步数：16
seed：42 或按请求递增
```

测试项目：

1. 单请求连续 4 次，丢弃第一次，取后 3 次平均。
2. 32 请求、32 并发。
3. 200 请求、最多 32 个在途请求。
4. 200 请求、200 个客户端同时提交。

### 12.3 Base 性能测试

```text
模型：flux-3-image-klein-base
分辨率：1024×1024
步数：50
CFG：3.5
```

测试项目：

1. 单 worker 验证设备端 CFG 和 Euler。
2. 32 请求、32 并发，验证 worker 0～31 全覆盖。
3. 200 请求、200 个客户端同时提交。

## 13. 优化后的 Distilled 结果

### 13.1 单请求

| 次数 | 耗时 |
|---:|---:|
| 1 | 11.568 s |
| 2 | 11.522 s |
| 3 | 11.546 s |
| 4 | 11.555 s |
| 后 3 次平均 | **11.541 s** |

### 13.2 并发结果

| 场景 | 成功 | 总墙钟 | 吞吐 | 平均延迟 | P95 |
|---|---:|---:|---:|---:|---:|
| 32 请求 / 32 并发 | 32/32 | 12.123 s | 2.6397 images/s | 11.923 s | 12.093 s |
| 200 请求 / 32 并发 | 200/200 | 81.209 s | 2.4628 images/s | 11.626 s | 11.995 s |
| 200 请求 / 200 同时提交 | 200/200 | 77.820 s | 2.5700 images/s | 40.971 s | 67.038 s |

### 13.3 优化前后对比

采用相同的 Distilled、1024×1024、16 步、32 worker、200 请求同时提交口径：

| 指标 | 优化前 | 优化后 | 改善 |
|---|---:|---:|---:|
| 总墙钟时间 | 132.499 s | 77.820 s | 降低 **41.3%** |
| 整机吞吐 | 1.5094 images/s | 2.5700 images/s | 提升 **70.3%** |
| 平均响应时间 | 73.215 s | 40.971 s | 降低 **44.0%** |
| P95 | 121.740 s | 67.038 s | 降低 **44.9%** |
| 单请求热态耗时 | 约 17.45 s | 11.541 s | 降低约 **33.9%** |

## 14. 优化后的 Base 结果

### 14.1 单 worker CFG 路径

| 模型 | steps | CFG | 耗时 | 输出 |
|---|---:|---:|---:|---|
| Base | 50 | 3.5 | 62.395 s | HTTP 200，有效 1024×1024 PNG，语义正确 |

### 14.2 32-worker 全覆盖

| 指标 | 结果 |
|---|---:|
| 请求 | 32 |
| 成功/失败 | 32/0 |
| worker 覆盖 | 0～31 每个恰好处理 1 个 |
| 总墙钟 | 62.790 s |
| 吞吐 | 0.5096 images/s |
| 平均延迟 | 62.520 s |
| P95 | 62.729 s |

### 14.3 Base 200 请求

| 指标 | 结果 |
|---|---:|
| 请求 | 200 个同时提交 |
| 成功/失败 | 200/0 |
| 有效 PNG | 200 |
| 总墙钟 | 433.557 s |
| 吞吐 | 0.4613 images/s |
| 平均响应 | 225.999 s |
| P50 | 247.863 s |
| P95 | 373.659 s |
| 最大响应 | 433.487 s |
| worker 分配 | 每个 worker 处理 6～7 个 |
| worker 错误 | 0 |

Base 比 Distilled 慢是预期结果：

- 步数由 16 增加到 50；
- Base 每一步执行 conditional 和 unconditional 两次 Transformer forward；
- Distilled 使用 flow-map，不执行双分支 CFG。

## 15. 优化后的现状

当前已经实现：

- Prompt Encoder、Transformer、VAE 按 worker 常驻；
- 32 个 worker 可以各自在一个芯片上独立常驻完整 pipeline；
- 相同步数的 Flow Schedule 可以复用；
- 初始 latent 只上传设备一次；
- Base CFG 在 TT 设备端完成；
- Euler update 在 TT 设备端完成；
- denoise latent 全程保留在 TT 设备端；
- 最终 latent 直接进入 TT VAE；
- 支持 `per_request` 低内存回退模式；
- Base 和 Distilled 两套模型均完成真实图片验证；
- Base 和 Distilled 均完成 32-worker 与 200 请求验证；
- 当前恢复运行的是 Distilled 32-worker 服务，32/32 ready、队列为 0、canary healthy、0 worker 错误。

## 16. 当前仍未完成的优化

### 16.1 精度

当前 TT 设备端 latent accumulator 使用 BF16。图片语义验证已经通过，但还没有完成逐步 Golden Data 数值对齐。

后续应对比：

- Prompt embedding；
- 初始 latent；
- 指定 denoise step 的 latent；
- 关键 Transformer block 输入输出；
- VAE 输入输出；
- BF16 accumulator 与设备端 FP32 accumulator。

如果发现 BF16 长步数累积误差，应优先尝试 TT 设备端 FP32 accumulator，不应直接恢复每一步 D2H/H2D。

### 16.2 性能

本轮主要消除了生命周期和 Host/Device 往返开销，尚未完成：

- matmul blocking 参数专项优化；
- layout 转换数量和耗时分析；
- TT trace/capture；
- Prompt Encoder、每步 Transformer、VAE、最终 D2H、PNG/JPEG 编码的独立 profiling；
- 峰值内存和功耗分析；
- Base 50-step CFG 的进一步加速；
- 32-worker CPU、内存带宽和图片编码并发优化。

### 16.3 工程

- 32 个 worker 首次共同创建相同 cache 时的跨进程锁；
- cache 临时目录写入和原子发布；
- 正式重建包含当前最新源码的交付镜像；
- 服务重启后的异步 image job 恢复。

## 17. 结果和日志位置

远端：

```text
/local/fluxcache/benchmarks/flux3-resident-device-euler.png
/local/fluxcache/benchmarks/flux3-resident-device-euler-32x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x200.json
/local/fluxcache/benchmarks/flux3-base-resident-device-cfg-euler.png
/local/fluxcache/benchmarks/flux3-base-32worker-20261105/
/local/fluxcache/benchmarks/flux3-base-200requests-20261105/
```

本地：

```text
outputs/flux3-resident-device-euler.png
outputs/flux3-base-resident-device-cfg-euler.png
outputs/flux3-base-32worker-20261105/
outputs/flux3-base-200requests-20261105/
```

## 18. 最终结论

本轮优化不是简单调整一个参数，而是改变了 FLUX.3 在 TT 上的模型生命周期和采样数据流：

```text
逐请求加载组件
  -> 按 worker 常驻组件

每个 denoise step 在 Host/TT 之间往返
  -> latent、CFG、Euler 全程留在 TT
```

在 Distilled 200 请求同口径测试中，整机吞吐提升 70.3%，总墙钟时间降低 41.3%。Base 的设备端 CFG 和 Euler 路径也已经在单 worker、32-worker 和 200 请求场景中真实验证完成。

因此，“TT 按 vLLM-Omni 相同优先级实现”的第一阶段目标已经完成。下一阶段应从算子 blocking/layout、trace、设备端 FP32 累积精度和分阶段 profiling 继续优化。
