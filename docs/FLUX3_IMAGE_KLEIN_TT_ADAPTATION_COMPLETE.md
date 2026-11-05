# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 上的完整适配记录

更新时间：2026-11-05

## 1. 当前结论

当前已经完成并在 10.12.11.2 的 Blackhole Galaxy 上验证：

- FLUX.3 Image Klein 模型目录和配置校验；
- TTNN Transformer 结构，包括 5 个 text mode block、5 个 image mode block、5 个 image-condition mode block 和 28 个 single block；
- DiT、Qwen3-VL Text Encoder、VAE Decoder 的权重读取和 TTNN 缓存；
- 单芯片 `(1,1)` mesh 上的真实 Transformer 执行；
- 多步 denoise、Qwen3-VL prompt 编码、VAE decode 和 PNG 输出；
- 每个 worker 常驻 Qwen3-VL prompt encoder、FLUX.3 Transformer 和 VAE decoder，请求之间不再重复装载权重；
- flow schedule 按步数缓存，初始 latent 只上传设备一次，CFG、Euler update 和整个 denoise latent 均保留在 TT 设备端；
- 可用 `FLUX3_COMPONENT_RESIDENCY=per_request` 回退到原逐请求加载方式，默认值为 `resident`；
- `tt-media-server` 的 FLUX.3 runner 和 `/v1/images/generations` API；
- 32 个单芯片 worker 启动、warmup 和真实请求；
- 64 个并发 T2I 历史回归请求，64/64 成功；
- 2026-11-05 完成 200 个同时提交的 T2I 请求测试，200/200 成功，worker 0～31 全部参与且每个处理 6～7 个请求；
- 2026-11-05 完成组件常驻和设备端 denoise 优化后，同口径 200 个同时提交请求总耗时从 132.499 秒降至 77.820 秒，吞吐从 1.5094 提升至 2.5700 images/s；
- Base 与 Distilled 两套 `/local/models` 权重的自动识别、独立缓存和单芯片 T2I；
- 修复 FLUX.3 checkpoint 与 TTNN fused SwiGLU 的权重半区顺序不一致问题；
- 使用 `a black bird` 分别完成 Base 50-step 和 Distilled 16-step 的真实语义验证。

当前明确暂缓或未完成：

- 2 芯片和 4 芯片 mesh 的权重分片、通信和性能验证；
- 与 PyTorch/vLLM/官方参考实现的逐层精度对齐；
- 更深层的 kernel blocking、layout、trace、峰值内存和功耗专项优化；
- 服务重启后的历史 image job 恢复和独立图片下载接口。
- 32 worker 首次同时创建同一模型转换缓存时的跨进程锁和原子发布。

已完成的图片异步任务 API、任务状态管理和 cancel 语义见第 20 节。本文件只描述真实代码、真实修改、真实验证和当前状态，不把早期失败结论当作当前状态。

## 2. 实现环境和代码边界

### 2.1 机器、模型和缓存

```text
机器：10.12.11.2
硬件：Blackhole Galaxy，32 个芯片
模型（Base）：/local/models/flux-3-image-klein-base
模型（Distilled）：/local/models/flux-3-image-klein
持久化缓存：/local/lnn/fluxcache
服务容器：flux3-media-server-test
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v5
分支：flux-3-image-klein
```

当前容器使用 v5 镜像作为运行环境，同时把 11.2 上最新的 `TL-inference-server/tt-media-server` 和 `tt-metal/models` bind mount 进容器。因此当前运行代码对应本地仓库最新提交；**v5 镜像本体单独启动时不包含 2026-11-04 之后的全部源码修改**，后续正式交付仍应基于当前两个仓库重建新镜像。

模型权重没有复制进 Docker image，而是运行时只读挂载：

```text
/local/models
  -> /local/models:ro
```

缓存目录保存：

```text
/local/lnn/fluxcache/tt_dit_cache       TT-DiT 转换后的权重
/local/lnn/fluxcache/tt-metal-cache     TTNN/JIT/kernel cache
/local/lnn/fluxcache/hf                 tokenizer、HF cache
/local/lnn/fluxcache/torch              PyTorch cache
/local/lnn/fluxcache/xdg                XDG 临时缓存
```

### 2.2 两个仓库的职责

| 仓库 | 负责内容 |
|---|---|
| `/local/lnn/tt-metal` | TTNN Transformer、Qwen3-VL encoder、VAE、权重映射、设备执行、kernel 和缓存 |
| `/local/lnn/TL-inference-server` | Dockerfile、模型注册、runner、worker、队列、设备生命周期和 HTTP API |

FLUX.3 适配不是只修改 server 仓库。server 负责把请求送进 pipeline，真正的模型结构和 TTNN 算子在 tt-metal 仓库中。

## 3. 完整实现流程总览

实际实现顺序如下：

```text
模型目录校验
  -> FLUX.3 TTNN Transformer 结构
  -> DiT 权重映射
  -> Qwen3-VL Text Encoder
  -> VAE Decoder
  -> latent / patchify / RoPE / scheduler / Euler
  -> Flux3T2IPipeline
  -> TTNN cache 和 Docker 镜像
  -> tt-media-server runner/API 注册
  -> 32 个单芯片 worker warmup
  -> 单请求 HTTP T2I
  -> 64 个并发 HTTP T2I
  -> 图片异步任务、轮询和 cancel
  -> 三类模型组件按 worker 常驻
  -> schedule 缓存和设备端 CFG / Euler / latent
  -> 32 并发与 200 同时提交性能回归
```

以下各节按这个真实顺序说明每一步修改了什么、在哪个文件、如何验证以及结果是什么。

## 4. 第一步：确认模型资产和实现边界

### 4.1 修改和实现内容

没有修改模型权重。首先确认模型目录包含 `config.json`、权重文件、tokenizer/processor 相关文件，并根据 `config.json` 确定 Transformer、Qwen3-VL 和 VAE 的结构参数。

### 4.2 代码位置

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_config.py
```

关键实现：

```text
Flux3Config.from_checkpoint()
validate_checkpoint()
```

`Flux3T2IPipeline.__init__()` 初始化时调用 checkpoint 校验，因此后续结构、权重和 pipeline 都基于同一个模型目录。

状态：已实现并验证。

## 5. 第二步：建立 FLUX.3 TTNN Transformer 结构

### 5.1 修改文件

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
/local/lnn/tt-metal/models/tt_dit/blocks/attention_opt.py
```

### 5.2 具体实现

`transformer_flux3.py` 中实现：

```text
Flux3TimeIn
Flux3ModeBlock
Flux3SingleBlock
Flux3Transformer
```

`Flux3Transformer.__init__()` 创建完整模块；`Flux3Transformer.forward()` 按 FLUX.3 的 joint token、text token、image token 和 image-condition token 组织输入，并依次执行 mode block、single block、final modulation 和 output projection。

结构数量：

```text
5 个 text mode block
5 个 image mode block
5 个 image-condition mode block
28 个 single block
```

这不是 5+28 个总 block；mode block 按三类输入分别实例化。

`attention_opt.py` 调整了单芯片 Blackhole attention/layout，使 FLUX.3 的 Q/K/V、RoPE 和 joint attention 可以在 `(1,1)` mesh 上执行。

### 5.3 验证

入口：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/test_one_step.py
```

真实 `Flux3Transformer.forward()` 输出形状：

```text
[1, 4096, 128]
```

状态：单芯片结构和真实前向已实现；多芯片并行未完成。

## 6. 第三步：实现 DiT 权重读取和映射

### 6.1 主要文件

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
```

关键方法：

```text
Flux3Transformer._block_state()
Flux3Transformer._flux3_state_dict()
Flux3Transformer.load_flux3_checkpoint()
```

### 6.2 如何实现

```text
FLUX.3 safetensors
  -> 读取 source state_dict
  -> 解析 time / mode / single / final 层命名
  -> QKV 拆分或合并
  -> SwiGLU 权重布局转换
  -> 映射到 TTNN module state_dict
  -> cache.load_model() 保存和复用
```

Transformer 结构解决“有哪些层以及如何连接”，权重映射解决“safetensors 中的张量如何放到这些层”。

### 6.3 验证结果

```text
source tensors：357
TTNN entries：271
```

状态：单芯片 BF16 权重映射已实现并验证；2/4 芯片分片未完成。

## 7. 第四步：接入 Qwen3-VL Text Encoder

### 7.1 文件

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/test_qwen_prompt_t2i.py
```

### 7.2 文本处理链路

```text
prompt
  -> 本地 tokenizer
  -> Qwen3-VL chat template
  -> token padding，pad_multiple=80
  -> Qwen3-VL M-RoPE
  -> 36 层 TTNN Text Encoder
  -> 取配置的 8 个层输出
  -> concat
  -> 20480 维 prompt embedding
```

`flux3_prompt_encoder.py` 负责 tokenizer、chat template、M-RoPE 和 encoder 调用；`flux3_components.py` 负责 Qwen3-VL encoder 结构和权重加载。

验证结果：36 层 Qwen3-VL、326 个缓存 tensor、20480 维 prompt embedding。

状态：已实现并验证。

## 8. 第五步：接入 VAE Decoder

### 8.1 文件和入口

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

关键调用：

```text
build_flux3_vae_decoder()
load_flux3_vae_weights()
Flux2VaeDecoder.preprocess_and_unpatchify()
Flux2VaeDecoder.forward()
```

执行链路：

```text
denoised latent -> VAE preprocess -> unpatchify -> VAE forward
  -> TTNN 到 host -> float_to_uint8 -> PIL.Image -> PNG/JPEG
```

VAE source tensors：251；实际输出分辨率：1024 x 1024。

状态：单芯片 VAE decode 和 PNG 输出已验证。

## 9. 第六步：实现采样基础模块

### 9.1 文件

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/sampling.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/test_sampling.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/test_pipeline_primitives.py
/local/lnn/tt-metal/models/tt_dit/tests/flux3/test_config.py
```

### 9.2 具体实现

`sampling.py` 实现：latent 初始化、patchify/unpatchify、初版 flow schedule、time embedding、text 1D RoPE、image 2D RoPE 和 host-side Euler update。

`t2i_contract.py` 定义 `Flux3T2IRequest`，统一 prompt、height、width、steps、seed 和 guidance 参数。

采样链路可以真实运行，但 scheduler 参数尚未与官方参考实现逐项数值对齐。因此“能生成图片”不等于“精度已经对齐”。

状态：采样基础功能已实现；参考实现精度对齐未完成。

## 10. 第七步：串联完整 T2I pipeline

### 10.1 文件和方法

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

当前新增或完善：

```text
Flux3T2IPipeline.run_external_prompt_embeds()
Flux3T2IPipeline.encode_prompt()
Flux3T2IPipeline.run()
```

### 10.2 外部 embedding 路径

```text
Flux3T2IRequest -> initialize_latents -> prepare_schedule -> patchify
  -> Transformer.forward -> unpatchify -> Euler update -> repeat steps
  -> VAE decode -> PNG
```

该路径绕过 tokenizer，用外部 `prompt_embeds` 验证 Transformer、多步采样和 VAE 组合。

### 10.3 完整文本路径

```text
prompt -> Flux3PromptEncoder.encode() -> Qwen3-VL embedding
  -> Transformer denoise loop -> VAE decode -> PNG
```

状态：外部 embedding 路径和 Qwen3-VL 文本路径都已在单芯片真实设备上出图。

## 11. 第八步：建立持久化权重和 kernel cache

运行时使用：

```text
TT_DIT_CACHE_DIR=/local/lnn/fluxcache/tt_dit_cache
TT_METAL_CACHE=/local/lnn/fluxcache/tt-metal-cache
TT_METAL_KERNEL_CACHE_PATH=/local/lnn/fluxcache/tt-metal-cache
HF_HOME=/local/lnn/fluxcache/hf
```

首次运行会出现权重转换和大量 `BuildKernels`；第二次运行可以命中 Transformer、Text Encoder、VAE 和 JIT cache。不同 mesh、dtype 和并行配置使用不同 cache key。

状态：已实现并验证。

## 12. 第九步：Dockerfile 使用本地 tt-metal 构建镜像

### 12.1 修改文件

```text
/local/lnn/TL-inference-server/tt-media-server/Dockerfile
```

### 12.2 修改内容

1. 删除从 GitHub clone `tt-metal` 的流程。
2. 增加本地 BuildKit context `tt-metal-local`。
3. 使用本地 `/local/lnn/tt-metal` 作为镜像内 TT-Metal 源码。
4. 保留 `TT_METAL_COMMIT_SHA_OR_TAG` 作为构建标识。
5. 不把 FLUX.3 权重复制进镜像，运行时只读挂载。
6. 增加 `/local/fluxcache` 相关环境变量。
7. 使用基础镜像已有 CMake，避免构建阶段重新下载 CMake。

状态：v5 镜像已经构建并实际使用。

## 13. 第十步：接入 tt-media-server

### 13.1 `config/constants.py`

文件：

```text
/local/lnn/TL-inference-server/tt-media-server/config/constants.py
```

增加 `FLUX_3_IMAGE_KLEIN`、`FLUX.3-Image-Klein` 和 `tt-flux.3-image-klein`，并将 runner 加入 image service、model-to-runner、Blackhole Galaxy 和 Galaxy 配置。

当前 FLUX.3 配置：

```text
device_mesh_shape=(1,1)
device_ids=(0),(1),...,(31)
max_batch_size=1
request_processing_timeout_seconds=3000
```

### 13.2 `open_ai_api/models.py`

将 `tt-flux.3-image-klein` 映射到 `BaseImageRequest`，支持 prompt、steps、seed、width、height 和 image_return_format。

### 13.3 `runner_fabric.py`

增加 `TTFlux3ImageKleinRunner` 的动态实例化注册，使每个 worker 都能创建 FLUX.3 runner。

### 13.4 `dit_runners.py`

增加 `TTFlux3ImageKleinRunner`：

1. 创建 `(1,1)` mesh 下的 CCL、DiT parallel config 和 encoder parallel config。
2. 从 `FLUX3_MODEL_ROOT` 读取绝对模型目录。
3. 将 `ImageGenerateRequest` 转换为 `Flux3T2IRequest`。
4. 调用 `Flux3T2IPipeline.run()`。
5. 返回 `[image]` 列表，以符合 worker 响应契约。
6. 设置 `FabricConfig.DISABLED`，因为单芯片 mesh 不需要 fabric。

## 14. 第十一步：32 个单芯片 worker 启动和修复

### 14.1 启动目标

```text
32 个 worker
每个 worker 1 个芯片
device_mesh_shape=(1,1)
DEVICE_IDS=(0),(1),...,(31)
```

启动前执行：

```bash
sudo tt-smi -r all
```

### 14.2 实际遇到的问题和解决

| 现象 | 原因 | 解决 |
|---|---|---|
| `FLUX.3-Image-Klein is not a valid ModelNames` | 容器从镜像内旧 server 目录加载配置 | 将本地 `tt-media-server` 挂载到 `/home/container_app_user/tt-metal/server` |
| `FileNotFoundError: flux-3-image-klein-base/config.json` | settings 中模型路径是相对路径 | `create_pipeline()` 使用 `FLUX3_MODEL_ROOT` 绝对路径 |
| `run() got an unexpected keyword argument 'device'` | 容器加载镜像内旧 `tt-metal/models` | 将本地 `/local/lnn/tt-metal/models` 挂载到 `/home/container_app_user/tt-metal/models` |
| `object of type 'Image' has no len()` | runner 返回单个 PIL Image，而 worker 要求 list | 返回 `[pipeline_result["image"]]` |
| HTTP 422，steps 小于 12 | 当前 validator 对该 runner 使用默认最小步数 12 | 测试请求使用 12 steps |

### 14.3 最终 warmup 结果

```text
split into 32 workers
All workers ready
model_ready=true
workers=32
device_mesh_shape=[1,1]
error_count=0
canary_state=healthy
```

## 15. 第十二步：单请求 HTTP T2I 验证

接口：

```text
POST http://127.0.0.1:8000/v1/images/generations
```

请求：

```json
{
  "prompt": "A small red fox sitting beneath a flowering tree, soft natural light, detailed digital painting.",
  "num_inference_steps": 12,
  "seed": 42,
  "image_return_format": "PNG"
}
```

结果：

```text
HTTP 200
generation_time=12.26 seconds
输出：/local/lnn/fluxcache/flux3_api_12step.png
大小：约 2.1 MB
```

该请求完整经过 Qwen3-VL、Transformer 多步 denoise、VAE decode、PNG 编码和 HTTP 返回。

## 16. 第十三步：64 个并发请求验证

同时发送 64 个 12-step T2I 请求，每个请求使用不同 seed 和 prompt 后缀。由于服务有 32 个 worker，64 个请求分两批执行。

```text
请求数：64
HTTP 200：64
成功生成图片：64
失败：0
实际处理请求的 worker：32/32，worker 0～31 全部出现
平均单请求耗时：28.94 秒
最大单请求耗时：39.89 秒
平均 generation_time：28.87 秒
测试后 model_ready：true
测试后 queue_size：0
测试后 error_count：0
测试后 canary_state：healthy
结果目录：/local/lnn/fluxcache/bench64_20261104
结果目录大小：约 173 MB
```

结论：当前 32 个 worker 不仅完成了 warmup，而且都能执行完整端到端推理。

## 17. 当前实际修改文件清单

### 17.1 TL-inference-server

| 文件 | 修改内容 |
|---|---|
| `tt-media-server/Dockerfile` | 使用本地 tt-metal BuildKit context；增加 FLUXCACHE 环境；默认设置 `FLUX3_COMPONENT_RESIDENCY=resident`；不复制模型权重；不再 clone 远端 tt-metal |
| `tt-media-server/config/constants.py` | 注册 FLUX.3 model、runner、model mapping、Galaxy 单芯片 32-worker 配置 |
| `tt-media-server/open_ai_api/models.py` | 将 FLUX.3 runner 接入 `BaseImageRequest` |
| `tt-media-server/tt_model_runners/dit_runners.py` | 增加 `TTFlux3ImageKleinRunner`、绝对模型路径读取、pipeline 调用和 list 响应修复；每个 worker 常驻 prompt encoder、Transformer、VAE，并提供 `per_request` 回退模式 |
| `tt-media-server/tt_model_runners/runner_fabric.py` | 注册 FLUX.3 runner 工厂 |

### 17.2 tt-metal

| 文件 | 修改内容 |
|---|---|
| `models/tt_dit/blocks/attention_opt.py` | FLUX.3 单芯片 attention/layout 适配 |
| `models/tt_dit/models/transformers/transformer_flux3.py` | Transformer、mode block、single block、forward、权重映射 |
| `models/tt_dit/pipelines/flux3/__init__.py` | FLUX.3 pipeline 包导出 |
| `models/tt_dit/pipelines/flux3/flux3_config.py` | checkpoint 配置和模型资产校验 |
| `models/tt_dit/pipelines/flux3/flux3_components.py` | Qwen3-VL、VAE 组件和权重加载 |
| `models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py` | tokenizer、chat template、M-RoPE、Qwen3-VL prompt encoder 调用 |
| `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 外部 embedding、Qwen3-VL 文本路径、多步 denoise、VAE decode、PNG 输出；schedule 缓存、设备端 CFG/Euler 和 denoise latent 常驻 |
| `models/tt_dit/pipelines/flux3/sampling.py` | latent、patchify、RoPE、schedule、time embedding、Euler step |
| `models/tt_dit/pipelines/flux3/t2i_contract.py` | T2I 请求数据结构 |
| `models/tt_dit/pipelines/flux3/test_one_step.py` | 真实 Transformer 单步验证 |
| `models/tt_dit/pipelines/flux3/test_external_prompt_t2i.py` | 外部 prompt embedding T2I 验证 |
| `models/tt_dit/pipelines/flux3/test_qwen_prompt_t2i.py` | Qwen3-VL 文本到图片验证 |
| `models/tt_dit/pipelines/flux3/test_pipeline_primitives.py` | pipeline 基础组件验证 |
| `models/tt_dit/pipelines/flux3/test_sampling.py` | sampling 基础验证 |
| `models/tt_dit/tests/flux3/test_config.py` | 配置校验测试 |

### 17.3 当前提交状态

```text
TL-inference-server 功能基线：e46186a9160d48344ec1f706f6ff4efd6f6c53ba
tt-metal 功能基线：3ff5dc122dae38d173870cf5525e9ed96d877c33
分支：flux-3-image-klein
工作树：clean
```

两个 commit 均已在 10.12.11.2 本地仓库创建，尚未 push 到远端 origin。

## 18. 尚未完成和下一步

### 18.1 多芯片

- 设计 2-chip 和 4-chip mesh shape；
- 完成 DiT、Text Encoder、VAE 的权重分片；
- 验证 CCL/fabric 通信和 layout；
- 分别建立 2/4 芯片 cache key；
- 重新测试单请求延迟和并发吞吐。

### 18.2 精度

- 固定 prompt、seed、height、width、steps 和 scheduler；
- 导出 vLLM/PyTorch/TTNN 的 prompt embedding、初始 latent、指定 denoise latent、关键 block 输入输出和 VAE 输入输出；
- 逐层比较 max absolute error、mean absolute error、cosine similarity 和图像差异；
- 对齐 scheduler、RoPE、latent scaling、VAE scaling 和 dtype。

### 18.3 性能

- [已完成] 持久化 `/local/lnn/fluxcache`，复用 TT-Metal JIT、TT-DiT、HF 和编译缓存；
- [已完成] 32 个单芯片 worker warmup，并完成 64 个并发请求回归；
- [历史数据] 早期 64 并发测试记录单请求约 12.26 秒、平均约 28.94 秒、最大约 39.89 秒；该数据来自修复 SwiGLU 和 Distilled flow-map 前的实现，只保留作阶段记录，不作为当前性能基线；
- [已完成] 2026-11-05 Distilled 1024×1024、16 步、32 worker、200 请求测试：200/200 成功，总耗时 132.499 秒，吞吐 1.5094 images/s，平均响应时间 73.215 秒，P95 121.740 秒；
- [已完成] 每个 worker 常驻 prompt encoder、Transformer、VAE；请求阶段重复权重加载事件为 0；
- [已完成] 相同步数的 flow schedule 缓存；初始 latent 只上传一次；CFG、Euler update 和 denoise latent 保持在 TT 设备端；
- [已完成] 优化后单请求 4 次测试，后 3 次平均 11.541 秒；32 并发墙钟 12.123 秒、吞吐 2.6397 images/s；
- [已完成] 优化后 200 个同时提交请求：200/200 成功，总耗时 77.820 秒，吞吐 2.5700 images/s，平均响应 40.971 秒，P95 67.038 秒；
- [未完成] matmul blocking/layout、trace 深度优化、功耗和内存峰值专项优化。

### 18.4 服务工程

- [已完成] 图片异步提交：`POST /v1/images/generations/async`；
- [已完成] 图片轮询：`GET /v1/images/generations/{job_id}`；
- [已完成] 图片取消：`POST /v1/images/generations/{job_id}/cancel`；
- [已完成] 图片任务复用 `JobManager` 的 queued/in_progress/completed/failed/cancelling/cancelled 状态；
- [已完成] 结果以每个 job 的 JSON 文件保存，轮询时返回 `images` base64 列表；
- [已完成] 真实服务验证 50-step cancel 和 12-step async completion；
- [未完成] image job 的独立 download 文件接口、服务重启恢复和长请求超时专项测试。

## 19. 主要验证产物

```text
/local/lnn/fluxcache/flux3_external_prompt_2steps.png
/local/lnn/fluxcache/flux3_qwen_prompt_1step.png
/local/lnn/fluxcache/flux3_api_12step.png
/local/lnn/fluxcache/bench64_20261104/
/local/fluxcache/benchmarks/flux3-resident-device-euler.png
/local/fluxcache/benchmarks/flux3-resident-device-euler-32x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x200.json
```

本文件的当前状态以第 1 节、第 15～18 节为准；早期单步验证只是完整流程中的一个阶段，不再作为当前总体状态。

## 20. 2026-11-04：异步图片任务与取消实现

### 20.1 实现位置和作用

| 文件 | 实现内容 |
|---|---|
| `tt-media-server/config/constants.py` | 增加 `JobTypes.IMAGE`，使图片任务可以进入统一 JobManager |
| `tt-media-server/model_services/image_service.py` | 增加 `create_image_job`、状态查询、结果读取、cancel 和 delete；生成完成后把 base64 列表写入每个 job 的 JSON 文件 |
| `tt-media-server/open_ai_api/image.py` | 增加 async submit、job metadata、cancel 和 delete 路由；原同步 `/generations` 保持兼容 |
| `tt-media-server/utils/job_manager.py` | 复用已有任务状态、队列准入、取消和清理逻辑，不重复实现视频任务管理器 |

### 20.2 请求流程

1. 客户端向 `/v1/images/generations/async` 发送 JSON 请求。
2. API 创建 `IMAGE` 类型 job，立即返回 `202` 和 job id。
3. JobManager 在后台调用现有 `ImageService.process_request()`，因此仍然经过 FLUX.3 runner、TTNN denoise、VAE 和图片后处理。
4. 完成后把 `{"images": [...]}` 写到 `IMAGE_JOB_OUTPUT_DIR/{job_id}.json`。
5. 客户端轮询 `/v1/images/generations/{job_id}`；完成时响应中包含 `images`。
6. cancel 调用 JobManager 的 cooperative cancellation，并触发现有 scheduler task cleanup。

### 20.3 真实验证结果

```text
服务：flux3-media-server-test
runner：tt-flux.3-image-klein
mesh：(1,1)

异步提交：HTTP 202，job_type=image，status=queued
50-step cancel：queued -> cancelling -> cancelled
12-step completion：queued -> completed，image_count=1
服务复启后：model_ready=true，canary_state=healthy
```

### 20.4 本轮仍明确暂缓

- [暂缓] 2 芯片和 4 芯片并行、权重分片和通信优化；
- [暂缓] 与 vLLM/PyTorch 的 Golden Data 精度对齐；
- [未开始] 专项 matmul/trace/kernel 性能优化；
- [未完成] 服务重启恢复、独立图片下载接口和长请求压力回归。

## 21. 按实施步骤对应的文件变更索引

本节是本文件的文件级索引。`修改`表示原有文件中增加或调整了 FLUX.3 逻辑；`新建`表示本次适配新增的文件。路径均以仓库根目录为基准。

### 步骤 1：模型资产和配置校验

| 类型 | 仓库 | 文件 | 作用 |
|---|---|---|---|
| 修改 | tt-metal | `models/tt_dit/pipelines/flux3/flux3_config.py` | 读取 checkpoint 配置、检查模型目录和权重资产 |
| 修改 | tt-metal | `models/tt_dit/pipelines/flux3/flux3_components.py` | 根据同一模型根目录构造 DiT、Qwen3-VL 和 VAE 组件 |
| 运行时输入 | 宿主机 | `/local/models/flux-3-image-klein-base` | 模型权重，不复制进镜像，运行时只读挂载 |

### 步骤 2：Transformer 结构和 TTNN 前向

| 类型 | 文件 | 作用 |
|---|---|---|
| 修改 | `models/tt_dit/models/transformers/transformer_flux3.py` | 定义 `Flux3TimeIn`、`Flux3ModeBlock`、`Flux3SingleBlock`、`Flux3Transformer` 及 forward |
| 修改 | `models/tt_dit/blocks/attention_opt.py` | 调整 Blackhole 单芯片 attention、Q/K/V、layout 和 RoPE 执行 |
| 修改 | `models/tt_dit/pipelines/flux3/__init__.py` | 导出 FLUX.3 pipeline 和组件 |
| 验证 | `models/tt_dit/pipelines/flux3/test_one_step.py` | 在真实 `(1,1)` mesh 上执行一个 denoise/Transformer step |

### 步骤 3：DiT 权重映射

| 类型 | 文件/目录 | 作用 |
|---|---|---|
| 修改 | `transformer_flux3.py` 的 `_block_state()` | 将单个 mode/single block 的源权重组织成目标 block 状态 |
| 修改 | `transformer_flux3.py` 的 `_flux3_state_dict()` | 处理 FLUX.3 safetensors key 命名和 tensor 形状 |
| 修改 | `transformer_flux3.py` 的 `load_flux3_checkpoint()` | 将 357 个源 tensor 映射到 TTNN 参数 |
| 运行时缓存 | `/local/lnn/fluxcache/tt_dit_cache` | 保存转换后的 TT-DiT 权重，避免重复转换 |

验证结果：15 个 mode block（text/image/image-condition 各 5 个）、28 个 single block、357 个源 tensor、271 个 TTNN 参数条目。

### 步骤 4：Qwen3-VL prompt encoder

| 类型 | 文件 | 作用 |
|---|---|---|
| 新建 | `models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py` | tokenizer、chat template、M-RoPE、Qwen3-VL 文本编码入口 |
| 修改 | `models/tt_dit/pipelines/flux3/flux3_components.py` | 加载 Qwen3-VL 36 层 encoder 及其权重 |
| 新建 | `models/tt_dit/pipelines/flux3/test_qwen_prompt_t2i.py` | 验证文本到 prompt embedding 的路径 |
| 缓存 | `/local/lnn/fluxcache/hf`、`/local/lnn/fluxcache/torch` | tokenizer、HF 和 PyTorch 相关缓存 |

### 步骤 5：VAE decode

| 类型 | 文件 | 作用 |
|---|---|---|
| 修改 | `models/tt_dit/pipelines/flux3/flux3_components.py` | 读取 VAE decoder 配置和 safetensors 权重 |
| 修改 | `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 将最终 latent 送入 VAE，并转换为 PIL/PNG |
| 验证 | `models/tt_dit/pipelines/flux3/test_pipeline_primitives.py` | 验证 VAE 和 pipeline 基础组件 |

验证结果：VAE 251 个源 tensor 加载通过。

### 步骤 6：latent、采样和多步 denoise

| 类型 | 文件 | 作用 |
|---|---|---|
| 修改 | `models/tt_dit/pipelines/flux3/sampling.py` | latent 初始化、patchify/unpatchify、RoPE、schedule、time embedding、Euler step |
| 新建 | `models/tt_dit/pipelines/flux3/t2i_contract.py` | 定义 T2I 请求参数和 pipeline 输入输出契约 |
| 修改 | `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 串联 prompt embedding、latent、denoise loop、unpatchify 和 VAE |
| 验证 | `models/tt_dit/pipelines/flux3/test_sampling.py` | 验证采样原语和多步更新 |
| 验证 | `models/tt_dit/pipelines/flux3/test_external_prompt_t2i.py` | 使用外部 prompt embedding 运行 T2I |

### 步骤 7：完整 pipeline

核心入口是：

```text
models/tt_dit/pipelines/flux3/pipeline_flux3.py:Flux3T2IPipeline.run()
```

该入口完成：

1. 接收 prompt、height、width、steps、seed、guidance；
2. 调用 Qwen3-VL 或读取外部 prompt embedding；
3. 根据 seed 初始化 latent；
4. 执行指定次数 denoise；
5. 执行 VAE decode；
6. 输出 PIL image，供 server runner 编码返回。

### 步骤 8：缓存和 Docker 镜像

| 类型 | 文件/目录 | 作用 |
|---|---|---|
| 修改 | `TL-inference-server/tt-media-server/Dockerfile` | 使用本地 tt-metal BuildKit context；不 clone 远端 tt-metal；不复制 FLUX.3 权重 |
| 修改 | Dockerfile cache ENV | 设置 `FLUXCACHE_DIR`、`TT_DIT_CACHE_DIR`、`TT_METAL_CACHE`、HF/Torch/XDG/TMP cache |
| 修改 | Dockerfile image job ENV | 设置 `IMAGE_JOB_OUTPUT_DIR=/local/fluxcache/image-jobs` |
| 宿主机 | `/local/lnn/fluxcache` | 持久化权重转换、JIT/kernel、文本 encoder 和异步图片 job 结果 |

### 步骤 9：tt-media-server 模型注册和 runner

| 类型 | 文件 | 作用 |
|---|---|---|
| 修改 | `tt-media-server/config/constants.py` | 注册 model、runner、服务类型、`(1,1)` mesh 和 32 个单芯片 worker；增加 `JobTypes.IMAGE` |
| 修改 | `tt-media-server/open_ai_api/models.py` | 将 FLUX.3 请求接入图片请求模型 |
| 修改 | `tt-media-server/tt_model_runners/runner_fabric.py` | 将 runner 名称映射到 `TTFlux3ImageKleinRunner` |
| 修改 | `tt-media-server/tt_model_runners/dit_runners.py` | 创建 pipeline、读取 `FLUX3_MODEL_ROOT`、传入请求参数并返回 image list |
| 修改 | `tt-media-server/model_services/image_service.py` | 图片后处理，以及异步任务结果 JSON 持久化 |
| 修改 | `tt-media-server/open_ai_api/image.py` | 同步、异步 submit、poll、cancel、delete API |

### 步骤 10：worker、HTTP 和并发验证

| 验证 | 结果 |
|---|---|
| 32 个 worker warmup | 32/32 ready，每个 worker 使用一个芯片，`(1,1)` mesh |
| 单请求 T2I | HTTP 200，12-step 约 12.26 秒 |
| 64 并发 T2I | 64/64 HTTP 200，worker 0～31 全部实际处理请求 |
| 异步图片提交 | HTTP 202，返回 `job_type=image` 和 job id |
| 异步 cancel | 50-step 任务 `queued -> cancelling -> cancelled` |
| 异步完成 | 12-step 任务 `completed`，轮询返回 1 张图片 |

### 步骤 11：commit 和当前边界

## 22. 2026-11-04：实现 `response_format=file`

### 22.1 实现文件

本次只修改 `TL-inference-server`，`tt-metal` 不需要修改：

| 文件 | 修改内容 |
|---|---|
| `tt-media-server/domain/image_generate_request.py` | 在 `BaseImageRequest` 增加 `response_format`，支持 `b64_json`（默认）和 `file`。 |
| `tt-media-server/open_ai_api/image.py` | 增加 Base64 解码、媒体类型选择、单图直接返回、多图 ZIP 打包逻辑。 |
| `tt-media-server/tests/test_kontext_api.py` | 增加文件响应、默认 Base64 响应和非法格式校验。 |

### 22.2 请求处理流程

```text
客户端 POST /v1/images/generations
        ↓
ImageGenerateRequest.response_format
        ↓
service.process_request() 返回 Base64 图片列表
        ↓
response_format=b64_json → 原有 JSON：{"images": [...], "generation_time": ...}
response_format=file    → Base64 解码为二进制图片
                         → 1 张：直接返回 image/png 或 image/jpeg
                         → 多张：打包返回 images.zip
```

### 22.3 使用方式

单张 PNG：

```bash
curl -fS http://127.0.0.1:8000/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"a red car","response_format":"file","image_return_format":"PNG","num_inference_steps":20}' \\
  -o output.png
```

默认 Base64 行为不变：

```json
{"prompt":"a red car","response_format":"b64_json"}
```

### 22.4 当前验证状态

- 本地和 11.2 上的三个 Python 文件已通过 Python AST/字节码语法检查。
- `response_format` 只接受 `b64_json` 或 `file`。
- 单图返回 `Content-Type: image/png` 或 `image/jpeg`。
- 多图返回 `Content-Type: application/zip`。
- 11.2 已通过真实 HTTP 端到端验证：使用现有 `flux3-media-server-test` 容器的源码挂载，重启 Uvicorn worker 后发送 12-step 请求。
- 验证请求省略自定义分辨率，使用 FLUX.3 当前 checkpoint 默认分辨率。
- 返回结果：HTTP `200`、`Content-Type: image/png`、`Content-Disposition: attachment; filename="image.png"`。
- 输出文件：`/tmp/tt-file-response.png`，PNG，`1024 x 1024`，`2,114,612` bytes。
- 额外验证：自定义 `256 x 256` 会被当前 FLUX.3 第一阶段 runner 拒绝，这是模型分辨率限制，不是文件响应实现问题。

```text
TL-inference-server commit 8ac20508
tt-metal commit a3550fc781
branch flux-3-image-klein
working tree clean
```

当前不在本轮范围内：

- 2 芯片/4 芯片 mesh、权重分片和 CCL 通信；
- PyTorch/vLLM/官方实现的逐层 Golden Data 精度对齐；
- matmul blocking、trace、layout、功耗和内存峰值专项优化；
- 服务重启后的历史 image job 恢复和独立下载接口。

## 23. 2026-11-04：实现客户端自定义分辨率

### 23.1 修改位置和作用

修改文件：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
```

修改 `Flux3T2IRequest.validate()`：删除必须等于 checkpoint 默认 1024×1024 的判断，增加正数校验，并要求宽高都是 `config.vae_scale_factor * config.patch_size` 的整数倍。当前配置为 `8 * 2 = 16`，所以客户端宽高必须是 **16 的倍数**。

这样既允许动态 shape，又能确保 VAE 下采样后得到完整 latent patch，避免后续 Transformer 或 VAE 处理不完整 tile。

### 23.2 客户端请求示例

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"a small red cube on a white table","response_format":"file","image_return_format":"PNG","width":512,"height":768,"num_inference_steps":50,"seed":42}' \\
  -o output.png
```

可用示例包括 `256×256`、`512×512`、`1024×1024` 和 `768×1024`；例如 `250×256` 会在请求校验阶段被拒绝，因为 250 不是 16 的倍数。

### 23.3 11.2 真实验证

测试服务为 10.12.11.2 上的 `flux3-media-server-test`，使用单芯片 `(1,1)` mesh、同步 `/v1/images/generations`、相同 prompt、`seed=42`、`num_inference_steps=50`、`response_format=file` 和 PNG 输出。

| 分辨率 | HTTP | 服务端处理时间 | 客户端总耗时 | 输出 |
|---|---:|---:|---:|---|
| 256×256 | 200 | 37.8345 s | 42.4204 s | PNG，256×256，138,294 bytes |
| 512×512 | 200 | 15.4046 s | 15.4158 s | PNG，512×512，529,403 bytes |
| 1024×1024 | 200 | 35.5721 s | 35.6047 s | PNG，1024×1024，2,117,127 bytes |

256×256 的客户端时间包含首次动态 shape/kernel 编译；日志中出现对应动态矩阵 shape 和 kernel 编译信息。编译缓存位于 `/local/lnn/fluxcache`，后续相同 shape 请求应单独进行热态测试。

### 23.4 提交记录

```text
tt-metal branch: flux-3-image-klein
tt-metal commit: 4cccb4584f Support client-defined FLUX3 T2I resolutions
```

## 24. 2026-11-04：每种分辨率连续 4 次热态测试

为排除动态 kernel 首次编译影响，对每种分辨率连续请求 4 次，使用相同 prompt、`seed=42`、`num_inference_steps=50` 和 PNG 文件响应。第 1 次作为预热，不计入平均值；平均值取第 2～4 次客户端 `curl` 总耗时。

| 分辨率 | 第1次 | 第2次 | 第3次 | 第4次 | 后3次平均 |
|---|---:|---:|---:|---:|---:|
| 256×256 | 11.2110 s | 11.1677 s | 11.1281 s | 11.2084 s | **11.1680 s** |
| 512×512 | 15.4146 s | 15.3210 s | 15.4079 s | 15.3293 s | **15.3528 s** |
| 1024×1024 | 38.4966 s | 38.4960 s | 38.5879 s | 38.5125 s | **38.5322 s** |

三种分辨率的 12 次请求均返回 HTTP 200，并生成了对应尺寸的 PNG。热态结果符合预期：256×256 < 512×512 < 1024×1024。之前 256×256 的 42.4204 秒是首次动态 shape/kernel 编译与推理的合计时间，不应与热态 512×512 直接比较。

## 25. 2026-11-04：修复 Prompt 语义错误并兼容 Base/Distilled

完整的问题现象、排查顺序、代码根因和验证记录见 `docs/FLUX3_IMAGE_KLEIN_TT_QUALITY_DEBUG.md`。

### 25.1 问题和根因

在端到端 pipeline、HTTP API 和图片文件输出均成功的情况下，提示词 `a black bird` 生成的是蓝色块或无语义纹理。主要根因位于：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
Flux3Transformer._block_state()
```

FLUX.3 checkpoint 的 `mlp_in.weight` 保存为 `[gate | up]`，共用 TTNN fused SwiGLU loader 期望 `[up | gate]`，使 15 个 mode block 和 28 个 single block 的 FFN 都计算错误。

修复代码：

```python
mlp_gate, mlp_up = reader.get_tensor(f"{prefix}.mlp_in.weight").chunk(2, dim=0)
"ff.ff1.weight": torch.cat((mlp_up, mlp_gate), dim=0)
```

交换只发生在 FLUX.3 checkpoint 映射边界，不修改 checkpoint 文件和通用 SwiGLU 层。

### 25.2 双模型实现

两套模型都位于 `/local/models`：

| 变体 | 路径 | 默认行为 |
|---|---|---|
| Base | `/local/models/flux-3-image-klein-base` | 50 steps、CFG 3.5、`flow_map=false` |
| Distilled | `/local/models/flux-3-image-klein` | 16 steps、无 CFG、`flow_map=true` |

`flux3_config.py` 根据 `config.json` 自动识别变体，并校验 Distilled 的 `ref_time_in` 权重。转换 cache 分别命名为 `flux-3-image-klein-base` 和 `flux-3-image-klein-distilled`。

### 25.3 同时修正的实现偏差

- 初始噪声按 `[N,H*W,128]` token 顺序生成并执行 BF16 舍入；
- Qwen hidden-state 层号从 HF `[4,8,...,32]` 映射到 decoder `[3,7,...,31]`；
- tokenizer 使用真实 `pad_token_id` 和 generation prompt；
- AdaLN、timestep、flow-shift scheduler 和四轴 RoPE 与 FLUX.3 定义对齐；
- Base 执行 negative prompt/CFG，Distilled 加载 `ref_time_in` 并执行无 CFG flow-map 路径；
- server 使用模型配置的默认步数，不再固定默认 20 步。

涉及文件：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_config.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/sampling.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
/local/lnn/TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py
```

### 25.4 真实质量验证

| 模型 | 请求 | 耗时 | 结果 |
|---|---|---:|---|
| Base | `a black bird`、1024×1024、50 steps、CFG 3.5、seed 42 | 68.37 s | 生成黑鸟 |
| Distilled | `a black bird`、1024×1024、16 steps、无 CFG、seed 42 | 17.45 s | 生成黑鸟 |

输出 SHA256：

```text
Base:      c268566ea53a9eec50143e87ec673d9916f491f06c8f74cd22f41a07df9ebf88
Distilled: 148d521f8b569880379c7bef7019106f1d3a6e28d2b0a5e1ae6fb4fb7cc5da58
```

### 25.5 当前限制

32 个 worker 首次启动时会同时转换和写入同一模型 cache，造成严重 I/O 竞争。当前质量回归使用单 worker 完成。恢复 32 worker 前，应先由单进程预生成 Base 和 Distilled cache，或实现跨进程锁、临时目录写入和原子 rename。

该节质量回归当时使用单 worker；截至 2026-11-05，`flux3-media-server-test` 已切换为 Distilled 模型的 32 个单芯片 worker。切换到 Base 仍只需要更换模型路径并使用对应 cache。

## 26. 2026-11-05：32 worker、200 请求并发性能验证

### 26.1 测试环境

```text
机器：10.12.11.2 / my01-galaxy02
硬件：Blackhole Galaxy，32 个芯片
模型：/local/models/flux-3-image-klein（Distilled）
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v5
运行代码：bind mount 当前 TL-inference-server 和 tt-metal/models
worker：32，每个 worker 使用一个芯片，mesh=(1,1)
DEVICE_IDS=(0),(1),...,(31)
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
TT_DIT_CACHE_DIR=/local/fluxcache/tt_dit_cache
```

开始前停止单 worker 容器，执行 `tt-smi -r all`，再用已有 Distilled 转换 cache 启动 32 worker。32 个 worker 从容器启动到全部 ready 约 89 秒；第一个 worker 于 `2026-11-05 08:48:30`（Asia/Shanghai）ready，最后一个于 `08:48:33` ready。

### 26.2 32 路预热

正式测试前同时提交 32 个请求，确认每个 worker 都进入端到端推理：

| 指标 | 结果 |
|---|---:|
| 请求 | 32 |
| 成功/失败 | 32 / 0 |
| worker 使用情况 | 0～31 每个恰好处理 1 个 |
| 总耗时 | 22.587 s |
| 吞吐 | 1.4168 images/s |
| 平均延迟 | 21.573 s |
| P95 | 22.477 s |
| 最大排队深度 | 5 |

### 26.3 200 请求参数

200 个客户端线程通过同一个释放屏障，在约 `45.7 ms` 内把请求全部发出：

```json
{
  "prompt": "a black bird",
  "num_inference_steps": 16,
  "width": 1024,
  "height": 1024,
  "response_format": "file",
  "image_return_format": "JPEG",
  "seed": "10000..10199"
}
```

测试时间：`2026-11-05 08:51:43` 至 `08:53:56`（Asia/Shanghai）。

### 26.4 总体性能结果

| 指标 | 结果 |
|---|---:|
| 请求总数 | 200 |
| HTTP 200 | 200 |
| 失败 | 0 |
| JPEG 响应 | 200 |
| 总墙钟时间 | 132.499 s |
| 整机吞吐 | **1.5094 images/s** |
| 平均响应时间 | 73.215 s |
| P50 | 78.752 s |
| P90 | 120.824 s |
| P95 | 121.740 s |
| P99 | 132.122 s |
| 最小/最大响应时间 | 21.103 / 132.440 s |
| 最大观测队列深度 | 168 |
| 测试期间 ready worker | 始终 32/32 |
| 测试后 worker error_count | 0 |
| 返回图片数据量 | 30,817,689 bytes |

### 26.5 32 个 worker 的实际请求分配

服务日志共记录 200 条 `Worker N processing tasks: 1`，worker 0～31 全部出现：

- worker `5、7、13、21、27、28、29、30`：每个处理 7 个请求；
- 其余 24 个 worker：每个处理 6 个请求；
- 合计：`8×7 + 24×6 = 200`。

因此这不是单 worker 或少数 worker 处理 200 请求，32 个芯片都实际执行了完整 T2I 请求，调度差异最多只有 1 个请求。

### 26.6 排队分轮特征

按客户端响应时间从小到大，每 32 个请求形成一轮：

| 轮次 | 数量 | 最小 | 平均 | 最大 |
|---:|---:|---:|---:|---:|
| 1 | 32 | 21.10 s | 22.18 s | 24.06 s |
| 2 | 32 | 41.07 s | 42.98 s | 45.02 s |
| 3 | 32 | 58.00 s | 62.80 s | 65.31 s |
| 4 | 32 | 72.52 s | 81.44 s | 84.69 s |
| 5 | 32 | 87.34 s | 99.39 s | 103.65 s |
| 6 | 32 | 103.95 s | 116.98 s | 121.95 s |
| 7 | 8 | 122.02 s | 127.27 s | 132.44 s |

平均响应时间 73.215 秒包含排队时间，不能理解为单张图片计算需要 73 秒。按整机吞吐折算，32-worker 稳态等效为每 21.20 秒完成一组 32 张图片。相对先前单 worker Distilled 17.45 秒数据，32 路吞吐扩展效率约为 82.3%；损失主要来自共享主机 I/O、CPU/postprocess、请求调度和 32 设备同时执行时的系统竞争。

### 26.7 原始产物

远端：

```text
/local/lnn/fluxcache/benchmarks/flux3-200-requests-20261105/
```

本地：

```text
outputs/flux3-200-requests-20261105/summary.json
outputs/flux3-200-requests-20261105/requests.jsonl
outputs/flux3-200-requests-20261105/liveness.jsonl
outputs/flux3-200-requests-20261105/server-window.log
outputs/flux3-200-requests-20261105/container-inspect.json
```

当前容器保持运行，32/32 worker ready，队列为 0，canary 为 healthy。

## 27. 2026-11-05：按 vLLM-Omni 优先级完成常驻组件和设备端 denoise 优化

### 27.1 优化前的问题

优化前的完整 T2I 已能正确生成图片，但一次请求中存在以下重复工作：

1. `TTFlux3ImageKleinRunner.run()` 每次调用 pipeline 时没有传入已创建的模型组件。
2. `Flux3T2IPipeline.run()` 因此会在每次请求中重新创建和加载 Qwen3-VL prompt encoder、FLUX.3 Transformer 和 VAE decoder。
3. denoise loop 每一步都把 host 侧 `image_tokens` 上传到 TT 设备。
4. Transformer 每一步输出都传回 host，在 PyTorch tensor 上完成 CFG 和 Euler update，下一步再重新上传。
5. 相同步数的 flow schedule 每次请求重新创建。

这会增加 cache 读取、对象构造、host/device 传输和 host 侧逐步采样开销，也使 32 worker 并发时共享 CPU、内存和 I/O 的竞争放大。

### 27.2 第一步：让每个 worker 常驻三个模型组件

修改文件：

```text
/local/lnn/TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py
```

在 `TTFlux3ImageKleinRunner.__init__()` 中增加：

```python
self._prompt_encoder = None
self._transformer = None
self._vae_decoder = None
self._component_residency = os.environ.get(
    "FLUX3_COMPONENT_RESIDENCY", "resident"
).strip().lower()
```

`create_pipeline()` 在 worker 初始化和设备打开后执行一次。`resident` 模式在这里创建并加载：

- `pipeline.create_prompt_encoder(...)`；
- `pipeline.create_transformer(...)` 和 `pipeline.load_transformer_weights(...)`；
- `pipeline.create_vae_decoder(...)` 和 `pipeline.load_vae_weights(...)`。

三个对象保存到 runner 实例中。后续 `run()` 调用把它们传给 `pipeline.run()`，所以一个 worker 生命周期内只加载一次，不在每个 HTTP 请求中重复加载。

32 个 worker 启动日志均出现：

```text
Device N: all FLUX.3 components are resident
```

从第一张真实请求开始统计后续服务日志：

```text
重复组件/cache 加载事件：0
worker error_count 合计：0
```

### 27.3 第二步：保留显存不足时的旧模式

同一文件支持两个值：

```text
FLUX3_COMPONENT_RESIDENCY=resident      默认；每 worker 常驻三个组件
FLUX3_COMPONENT_RESIDENCY=per_request   回退；沿用原逐请求创建和加载路径
```

如果未来单芯片设备内存不足，可在启动容器时设置：

```bash
-e FLUX3_COMPONENT_RESIDENCY=per_request
```

`per_request` 模式下 runner 传入的三个组件保持为 `None`，pipeline 会执行原有的按请求创建逻辑。该回退没有复制第二套 pipeline，实现路径仍由同一组 `create_*()` 和 `load_*_weights()` 方法维护。

修改文件：

```text
/local/lnn/TL-inference-server/tt-media-server/Dockerfile
```

Dockerfile 的持久化缓存环境块增加默认值：

```dockerfile
FLUX3_COMPONENT_RESIDENCY=resident
```

运行时 `docker run -e FLUX3_COMPONENT_RESIDENCY=per_request ...` 可以覆盖镜像默认值。

### 27.4 第三步：缓存 flow schedule

修改文件：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

`Flux3T2IPipeline.__init__()` 增加：

```python
self._schedule_cache: dict[int, Flux3FlowSchedule] = {}
```

`prepare_schedule()` 用 `request.steps` 作为 key。相同步数例如 16 步或 50 步只创建一次 schedule，后续请求直接复用不可变的 timesteps 和 sigmas。

### 27.5 第四步：把 latent、CFG 和 Euler update 保留在 TT 设备端

修改位置仍是：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
Flux3T2IPipeline.run_with_prompt_embeds()
```

优化后的执行顺序：

```text
host 初始 noise/latent
  -> 一次 H2D，得到 image_tt
  -> denoise step 0..N-1
       Transformer forward on TT
       CFG on TT（Base 模型）
       model_output *= sigma_next - sigma_current
       image_tt += model_output
  -> 设备端 image_tt 直接进入 VAE preprocess/unpatchify
  -> TTNN VAE decode
  -> 最终图片一次 D2H
  -> PNG/JPEG 编码和 HTTP 返回
```

Base 模型的 CFG 改为 TTNN 运算：

```python
model_output = ttnn.subtract(conditional_output, unconditional_output)
ttnn.multiply_(model_output, float(request.guidance))
ttnn.add_(model_output, unconditional_output)
```

Euler update 改为 TTNN 原地运算：

```python
sigma_delta = float((sigma_next - sigma_current).item())
ttnn.multiply_(model_output, sigma_delta)
ttnn.add_(image_tt, model_output)
```

因此 denoise loop 不再每一步执行 `image_tokens -> TT -> host -> image_tokens` 往返。VAE 的 `preprocess_and_unpatchify()` 也直接接收最终设备端 `image_tt`。

当前 accumulator 使用 `ttnn.bfloat16`。旧实现把每一步 Transformer BF16 输出传回 host 后以 PyTorch FP32 latent 累积。真实黑鸟图片语义和视觉质量已通过，但尚未执行 Golden Data 逐步数值对齐；如后续发现累积误差，应优先评估 TT 设备端 float32 accumulator，而不是恢复每步 D2H/H2D。

### 27.6 实施和验证中遇到的问题

| 步骤 | 现象 | 原因 | 处理 |
|---|---|---|---|
| 第一张 HTTP 验证 | 返回 422，服务提示 JSON decode error | Windows PowerShell、SSH、远端 shell 三层引号使手写 `curl -d` JSON 被破坏 | 改用 Python `urllib.request` 对字典执行 `json.dumps()`，消除 shell JSON 转义；后续请求 HTTP 200 |
| 设备端 Euler 风险检查 | 修改前担心 `ttnn.multiply_`/`ttnn.add_` 出现 dtype 或 layout 不兼容 | latent 和 Transformer 输出均为设备 BF16，原地算子要求布局兼容 | 真实 16-step 请求和 437 个累计请求均成功，无 layout/dtype 异常，无 worker error |
| 数值精度检查 | 设备端 BF16 累积与旧 host FP32 累积不同 | 为减少每步 D2H/H2D，accumulator 改为设备 BF16 | 使用固定 prompt/seed 生成 1024×1024 PNG，图片为清晰黑鸟且符合 prompt；保留后续 Golden Data 对齐项 |

### 27.7 验证步骤和结果

验证配置：

```json
{
  "prompt": "a black bird",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 16,
  "seed": 42,
  "response_format": "file",
  "image_return_format": "PNG"
}
```

1. 重新启动 32-worker 服务，确认 32/32 ready、`canary_state=healthy`、所有 `error_count=0`。
2. 检查 32 个 worker 均输出组件常驻完成日志。
3. 发送单张真实请求，得到有效 1024×1024 RGB PNG，内容与 `a black bird` 一致。
4. 从第一张请求之后统计日志，组件和 cache 重新加载事件为 0。
5. 连续执行 4 次单请求，按既定规则丢弃第一次，计算后 3 次平均值。
6. 执行 32 请求、32 并发，验证所有 worker 同时工作。
7. 执行 200 请求、32 个最大在途请求，验证持续饱和吞吐。
8. 执行 200 请求、200 个客户端同时提交，与旧基线同口径比较排队和总吞吐。

单请求结果：

| 次数 | 客户端总耗时 |
|---:|---:|
| 1 | 11.568 s |
| 2 | 11.522 s |
| 3 | 11.546 s |
| 4 | 11.555 s |
| 后 3 次平均 | **11.541 s** |

并发结果：

| 场景 | 成功 | 总墙钟 | 吞吐 | 平均延迟 | P95 | 最大延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 32 请求 / 32 并发 | 32/32 | 12.123 s | 2.6397 images/s | 11.923 s | 12.093 s | 12.112 s |
| 200 请求 / 32 并发 | 200/200 | 81.209 s | 2.4628 images/s | 11.626 s | 11.995 s | 12.154 s |
| 200 请求 / 200 同时提交 | 200/200 | 77.820 s | 2.5700 images/s | 40.971 s | 67.038 s | 77.782 s |

与旧的 200 同时提交基线比较：

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 总墙钟 | 132.499 s | 77.820 s | 降低 41.3% |
| 整机吞吐 | 1.5094 images/s | 2.5700 images/s | 提升 70.3% |
| 平均响应时间 | 73.215 s | 40.971 s | 降低 44.0% |
| P95 | 121.740 s | 67.038 s | 降低 44.9% |

### 27.8 代码提交和验证产物

```text
TL-inference-server：e46186a9160d48344ec1f706f6ff4efd6f6c53ba
提交说明：Keep FLUX.3 pipeline components resident per worker

tt-metal：3ff5dc122dae38d173870cf5525e9ed96d877c33
提交说明：Optimize FLUX.3 device-resident denoise loop
```

远端产物：

```text
/local/fluxcache/benchmarks/flux3-resident-device-euler.png
/local/fluxcache/benchmarks/resident-device-euler-4runs/
/local/fluxcache/benchmarks/flux3-resident-device-euler-32x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x32.json
/local/fluxcache/benchmarks/flux3-resident-device-euler-200x200.json
```

本地质量样图：

```text
outputs/flux3-resident-device-euler.png
```
