# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 上的完整适配记录

更新时间：2026-11-13

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
- 常驻组件和设备端 denoise 优化后，重新实测 Distilled 16-step 及 Base 50-step CFG，两者均返回语义正确的 1024×1024 PNG。
- 2026-11-05 完成 Base 模型 32-worker 全覆盖和 200 个同时提交请求压测：32 个 worker 全部实际执行，200/200 成功、0 错误。
- 有序 `reference_images` 数组和 1～10 张参考图接入，兼容旧单图字段；Base/Distilled 各 9 个同步、异步及非法请求场景全部通过。
- 单芯片 TTNN VAE reference encoder、111 个源张量映射、设备端 packing/BN 和条件 token 拼接；两模型含 1024 方图、横图/竖图及切回小图的 22 次 encoder 数值/尺寸切换测试及缓存重载通过。
- 2026-11-06 用已提交的本地仓库源码构建正式镜像 `tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4`（约 32.6 GB）；镜像本体包含 TTNN reference encoder 默认值，不需要挂载 `TL-inference-server` 或 `tt-metal` 源码目录。
- 该正式镜像已分别完成 Distilled 与 Base 的 32/32 worker 就绪、T2I、1/2/10 张参考图、异步轮询和 cancel 回归；受保护图片 API 使用运行时注入的唯一 `OPENAI_API_KEY`，`/tt-liveness` 不需要鉴权。
- 2026-11-13：按官方参考仓库重新核对后，图片 API 统一为唯一运行时 bearer token `OPENAI_API_KEY`，不再内置公开默认 token；FLUX.3 输出尺寸在入队前按官方 VAE 约束校验为每边 16--4096、16 像素对齐，不额外施加 8:1 或总像素限制。独立 SP4×8 镜像回归中 8/8 worker ready；无/错误密钥均为 HTTP 401、非 16 对齐为 HTTP 422，1024×1024/16-step 的有效 JPEG HTTP 用时 3.220 s；10 项单元测试通过。

### 1.1 当前交付状态

| 交付项 | 当前状态 | 证据或边界 |
|---|---|---|
| Base 与 Distilled T2I | 已实现并验证 | 单芯片、32-worker、文件响应和 200 请求压测均已完成 |
| 多参考图 | 已实现并验证 | 1～10 张有序参考图；两模型大图矩阵各 14/14 通过 |
| reference encoder | 已实现并验证 | TTNN 默认；`cpu` 是显式回退，22 次尺寸/缓存切换及服务负载通过 |
| 异步任务与 cancel | 已实现并验证 | 创建、轮询、取消语义见第 20 节 |
| 大图与 32-worker 参考图压测 | 已完成 | 32 冷/暖态和 200/200 均通过；结果见 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md` |
| 正式镜像 | 已完成并验证 | `tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4`；Base/Distilled 均在不挂源码目录时完成 32 worker、T2I、多参考图、async/poll/cancel |
| 源码提交 | 已完成 | tt-metal `6926f00a4a`；TL-inference-server `eca825c5`、`52eee2fc` |
| API key 与输出尺寸契约 | 已实现并验证 | 唯一 `OPENAI_API_KEY`；输出每边 16--4096、16 对齐，官方 4K 上限；无/错 token 401、非对齐尺寸 422 |
| 2/4 芯片单请求并行 | 暂缓 | 当前任务范围不实施 |
| Golden Data 逐层精度对齐 | 暂缓 | 当前任务范围不实施 |

### 1.1.1 2026-11-06：提交后正式镜像回归

构建命令同时使用 `/local/lnn/TL-inference-server` 和 `/local/lnn/tt-metal` 的本地提交版本；运行时没有挂载这两个源码目录。镜像为 `tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4`，对应 TL commit `eca825c5`、TT-Metal commit `6926f00a4a`。

```bash
cd /local/lnn/TL-inference-server
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=6926f00a4a \
  --file tt-media-server/Dockerfile \
  --tag tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4 \
  .
```

这里的 `--build-context tt-metal-local=/local/lnn/tt-metal` 是关键：Dockerfile 会把本地 TT-Metal 作为构建上下文使用；`--build-arg TT_METAL_COMMIT_SHA_OR_TAG=6926f00a4a` 则把镜像内的版本标识固定为本次已验证 commit。

| 模型 | 32 worker | 同步功能 | 异步完成 | cancel | 结果目录 |
|---|---|---|---|---|---|
| Distilled | 32/32 ready | T2I、旧单图、1/2/10 图、坏图/11 图校验 | 12-step、256×256 成功并返回 1 张 PNG | 50-step、1024×1024 → `cancelled` | `/local/lnn/fluxcache/official_image_eca825c5_distilled_20261106/` |
| Base | 32/32 ready | T2I、旧单图、1/2/10 图、坏图/11 图校验 | 12-step、256×256 成功并返回 1 张 PNG | 50-step、1024×1024 → `cancelled` | `/local/lnn/fluxcache/official_image_eca825c5_base_20261106/` |

同步请求使用 `response_format=file`，成功项返回有效 PNG；非法 Base64 图片和第 11 张图片均在入队前返回 HTTP 422。当前镜像不内置 API key，`/v1/images/*` 请求必须携带与运行时 `OPENAI_API_KEY` 相同的 `Authorization: Bearer ...`；`/tt-liveness` 为公开健康检查接口。测试脚本首次遗漏轮询请求的 Authorization header，造成 401；补齐 header 后 Base 与 Distilled 均成功完成异步轮询，因此该 401 不属于模型或服务功能失败。

### 1.2 当前未完成工作

- 32 worker 首次同时创建同一模型转换缓存的跨进程锁和原子发布；
- TTNN reference encoder 的自动 OOM 降级；目前 TTNN 默认，仍可显式选择 `cpu` 回退；
- 服务重启后的历史 image job 恢复和独立图片下载接口；
- 多小时稳定性、每芯片 DRAM/L1、温度和功耗专项测试；
- 以 profiling 为依据的 kernel blocking、layout、trace、峰值内存专项优化；
- 提交后的正式镜像已完成 Base、Distilled、多参考图、异步/轮询/cancel 回归；后续仅需持续性、资源与专项性能优化验证。

### 1.3 如何阅读本文件

第 2～17 节记录从模型资产、TTNN Transformer、权重映射、prompt/VAE、pipeline、Docker、服务接入到单请求与并发验证的完整实施过程。第 20 节记录异步任务和取消；第 22～28 节记录文件响应、分辨率、双模型质量修复和性能优化；第 29～31 节记录单图、多图、TTNN reference encoder、大尺寸和 32-worker 参考图验证。

本轮多参考图和 TTNN VAE 的逐文件/方法说明见 `FLUX3_MULTI_REFERENCE_AND_TTNN_VAE_IMPLEMENTATION.md`；大尺寸修复及负载结果见 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md`。第 29、30 节的“未完成”描述仅表示当时的历史阶段，不代表当前状态。原 8000 服务在大图压测后已恢复并通过 32/32 ready、queue=0、canary healthy 核验；自本次默认值修订起，reference backend 默认 TTNN，只有显式设置 `FLUX3_REFERENCE_ENCODER_BACKEND=cpu` 才使用 CPU 回退。

已完成的图片异步任务 API、任务状态管理和 cancel 语义见第 20 节。本文件只描述真实代码、真实修改、真实验证和当前状态，不把早期失败结论当作当前状态。

## 2. 实现环境和代码边界

### 2.1 机器、模型和缓存

```text
机器：10.12.11.2
硬件：Blackhole Galaxy，32 个芯片
模型（Base）：/local/models/flux-3-image-klein-base
模型（Distilled，服务默认）：/local/models/flux-3-image-klein
默认选择逻辑：若未设置 `FLUX3_MODEL_ROOT` 且 `MODEL_WEIGHTS_DIR` 未提供具体模型目录，runner 会把默认模型名解析为已挂载的 `/local/models/flux-3-image-klein`；显式配置始终优先。
持久化缓存：/local/lnn/fluxcache
服务容器：flux3-media-server-eca825c5-6926f00a4-base-validation（Base；Distilled 验证容器已停止保留）
镜像：tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4
分支：flux-3-image-klein
```

历史 `flux3-media-server-test` 使用 v5 镜像并依赖源码 bind mount，已在 2026-11-06 停止保留。当前正式镜像验证容器不挂载 `TL-inference-server` 或 `tt-metal` 源码目录，只挂载模型、缓存和 hugepage；已使用提交 SHA 构建的最终镜像完成 Base、Distilled、多参考图和异步/cancel 回归。

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
  -> 单参考图 CPU image_cond 条件推理
  -> 有序多参考图 API 与独立预处理
  -> TTNN VAE encoder、动态 GroupNorm 与尺寸相关卷积权重准备
  -> Base/Distilled 数值、同步/异步验证并恢复原服务
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

`/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py`

该文件负责 **VAE decoder 组件本身的创建和权重装载**，不负责执行一次完整的图片生成请求。与本节相关的入口是：

```text
build_flux3_vae_decoder()
  -> 按 FLUX.3 VAE 配置创建 Flux2VaeDecoder
  -> 保存 device 和 parallel_config，供后续权重缓存加载使用

load_flux3_vae_weights()
  -> 读取 vae/diffusion_pytorch_model.safetensors
  -> 调用 cache.load_model() 转换、缓存并加载 251 个 VAE decoder 源 tensor
```

同一文件还包含 Qwen3-VL text encoder 的装配，以及参考图预处理、CPU/TTNN reference VAE encoder；这些职责与“最终 latent 解码成图片”不同。

`/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py`

该文件负责 **请求级 pipeline 编排**：它持有已加载的 decoder，在全部 denoise step 完成后，调用 decoder 将最终 image token/latent 转为可见图片。与本节相关的职责是：

```text
接收 denoise loop 的最终 image_tt
  -> 转为 host token / latent 表示
  -> 调用 Flux2VaeDecoder.preprocess_and_unpatchify()
  -> 调用 Flux2VaeDecoder.forward()
  -> 得到 RGB tensor
  -> 转为 PIL Image / PNG，交给 server 返回
```

因此，`flux3_components.py` 解决“**decoder 是什么、权重如何上设备**”，而 `pipeline_flux3.py` 解决“**何时调用 decoder、如何把 decoder 输出返回给请求方**”。

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
Flux3T2IRequest -> initialize_image_tokens (official BF16-rounded packed noise)
  -> prepare_schedule -> Transformer.forward -> Euler update on TT -> repeat steps
  -> VAE decoder preprocess_and_unpatchify -> VAE decode -> PNG
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
| `tt-media-server/domain/image_generate_request.py` | 参考图单/数组字段、1～10 张约束、入队前图片及字段冲突校验；图片文件响应参数 |
| `tt-media-server/tests/test_flux3_reference_api.py` | 有图/无图、数量、坏图片、字段互斥、编码归一化及非 FLUX.3 runner 测试，17/17 通过 |

### 17.2 tt-metal

| 文件 | 修改内容 |
|---|---|
| `models/tt_dit/blocks/attention_opt.py` | FLUX.3 单芯片 attention/layout 适配 |
| `models/tt_dit/models/transformers/transformer_flux3.py` | Transformer、mode block、single block、forward、权重映射 |
| `models/tt_dit/pipelines/flux3/__init__.py` | FLUX.3 pipeline 包导出 |
| `models/tt_dit/pipelines/flux3/flux3_config.py` | checkpoint 配置和模型资产校验 |
| `models/tt_dit/pipelines/flux3/flux3_components.py` | Qwen3-VL、VAE 组件和权重加载；单/多图预处理、CPU/TTNN reference encoder、SHA256 区分缓存 |
| `models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py` | tokenizer、chat template、M-RoPE、Qwen3-VL prompt encoder 调用 |
| `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 外部 embedding、Qwen3-VL 文本路径、多步 denoise、VAE decode、PNG 输出；schedule 缓存、设备端 CFG/Euler 和 denoise latent 常驻；reference backend 选择、条件 token 设备拼接 |
| `models/tt_dit/models/vae/flux3_vae_encoder.py` | 新增 TTNN reference encoder；111 源张量映射、动态核心网格、几何变化重准备卷积权重、packing/BN |
| `models/tt_dit/tests/flux3/test_flux3_reference_preprocessing.py` | 新增多图裁剪、缩放和小图边界测试；与 reference RoPE 测试合计 4/4 通过 |
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
TL-inference-server 当前 HEAD：9931070523c19dff7e32a15ebf65b91e105443c2
tt-metal 当前 HEAD：06f77cf6bc51ee3a63027ffc8e2c19125b51004c
分支：flux-3-image-klein
工作树：本轮多参考图、TTNN encoder 和文档修改尚未提交
```

上述为当前 HEAD，不代表本轮新功能已包含在提交中。本轮没有执行 commit 或 push，运行时源码挂载包含工作树的修改。

## 18. 完整实现内容与后续状态

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

### 18.5 参考图片条件推理

- [已完成] `reference_images` 接收 1～10 张有序参考图，兼容旧单图参数，非法数组/坏图片入队前拒绝。
- [已完成] 单图填充、多图裁剪策略，以及每图独立参考位置轴。
- [已完成] 单芯片 TTNN VAE encoder、packing/BN 和设备条件 token 拼接；可显式选择 CPU 兼容 backend。
- [已完成] 两模型最终各 11 次、共 22 次 encoder 数值/尺寸切换检查，含 1024 方图/横竖图，缓存重载成功。
- [已完成] 两模型各 9 个 HTTP 场景；参考图数量实测 1、2、10，两图异步完成。
- [已完成] Base 与 Distilled 各 14/14 大图矩阵、32 首次/32 暖态、200/200 参考图请求通过；当前结果见第 31 节，不等同于已有 T2I 压测。
- [未实现] 自动 OOM 切换 reference backend、prompt upsampler 和首次转换缓存跨进程锁。

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

本文件的当前状态以第 1、17～18、30 节为准；早期单步验证只是完整流程中的一个阶段，不再作为当前总体状态。

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

当前 accumulator 使用 `ttnn.bfloat16`。旧实现把每一步 Transformer BF16 输出传回 host 后以 PyTorch FP32 latent 累积。Distilled 16-step 和 Base 50-step CFG 的真实黑鸟图片语义和视觉质量均已通过，但尚未执行 Golden Data 逐步数值对齐；如后续发现累积误差，应优先评估 TT 设备端 float32 accumulator，而不是恢复每步 D2H/H2D。

### 27.6 实施和验证中遇到的问题

| 步骤 | 现象 | 原因 | 处理 |
|---|---|---|---|
| 第一张 HTTP 验证 | 返回 422，服务提示 JSON decode error | Windows PowerShell、SSH、远端 shell 三层引号使手写 `curl -d` JSON 被破坏 | 改用 Python `urllib.request` 对字典执行 `json.dumps()`，消除 shell JSON 转义；后续请求 HTTP 200 |
| 设备端 Euler 风险检查 | 修改前担心 `ttnn.multiply_`/`ttnn.add_` 出现 dtype 或 layout 不兼容 | latent 和 Transformer 输出均为设备 BF16，原地算子要求布局兼容 | Distilled 累计 437 个请求和 Base 50-step CFG 请求均成功，无 layout/dtype 异常，无 worker error |
| 数值精度检查 | 设备端 BF16 累积与旧 host FP32 累积不同 | 为减少每步 D2H/H2D，accumulator 改为设备 BF16 | 固定 prompt/seed 的 Distilled 与 Base 均生成语义正确的 1024×1024 黑鸟 PNG；保留后续 Golden Data 对齐项 |

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
9. 暂停 Distilled 服务、重置设备，使用 Base 权重启动单 worker，真实执行 50-step、CFG 3.5；保存图片和日志后恢复原 32-worker Distilled 服务。

单请求结果：

| 次数 | 客户端总耗时 |
|---:|---:|
| 1 | 11.568 s |
| 2 | 11.522 s |
| 3 | 11.546 s |
| 4 | 11.555 s |
| 后 3 次平均 | **11.541 s** |

Base CFG 路径结果：

| 模型 | steps | guidance | HTTP/文件 | 耗时 | 视觉结果 |
|---|---:|---:|---|---:|---|
| Base | 50 | 3.5 | 200 / 1024×1024 PNG | 62.395 s | 清晰黑鸟，与 prompt 一致 |

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
/local/fluxcache/benchmarks/flux3-base-resident-device-cfg-euler.png
/local/fluxcache/benchmarks/flux3-base-resident-device-cfg-euler.log
```

本地质量样图：

```text
outputs/flux3-resident-device-euler.png
outputs/flux3-base-resident-device-cfg-euler.png
```

## 28. 2026-11-05：Base 模型 32-worker 和 200 请求验证

### 28.1 测试目标和环境

本轮专门验证 `/local/models/flux-3-image-klein-base`，不是 Distilled 模型。目标有两个：

1. 同时提交 32 个请求，确认 worker 0～31 每个都能完成一次 Base 端到端推理。
2. 同时提交 200 个请求，验证 Base 50-step CFG 在 32 个单芯片 worker 上的稳定性、调度均衡性和吞吐。

```text
机器：10.12.11.2 / Blackhole Galaxy
芯片：32
worker：32，每个 worker 独占一个芯片，mesh=(1,1)
模型：/local/models/flux-3-image-klein-base
容器：flux3-base-32worker-test（测试完成后已删除）
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v5
组件模式：FLUX3_COMPONENT_RESIDENCY=resident
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

请求参数：

```json
{
  "prompt": "a black bird",
  "negative_prompt": "",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 3.5,
  "seed": "10000 + request_index",
  "response_format": "file",
  "image_return_format": "PNG"
}
```

启动前停止原 Distilled 容器并执行全设备 reset。Base 测试完成后删除临时容器，再次 reset，并恢复原 `flux3-media-server-test` Distilled 32-worker 服务。

### 28.2 32 个 worker 全覆盖测试

同时提交 32 个 Base 请求：

| 指标 | 结果 |
|---|---:|
| 请求数 | 32 |
| 并发数 | 32 |
| HTTP 200 / 失败 | 32 / 0 |
| 有效 PNG | 32 |
| 总墙钟时间 | 62.790 s |
| 整机吞吐 | 0.5096 images/s |
| 平均延迟 | 62.520 s |
| P50 | 62.540 s |
| P95 | 62.729 s |
| 最小 / 最大延迟 | 62.224 / 62.776 s |
| 返回数据量 | 49,588,056 bytes |
| 服务日志错误 | 0 |

服务日志共匹配到 32 条 `Worker N processing tasks`：worker 0～31 每个恰好出现 1 次。这证明 32 个芯片都真实执行了完整的 Base prompt encoder → 50-step CFG denoise → VAE → PNG 链路，而不只是完成 worker 初始化或 warmup。

### 28.3 200 个同时提交请求结果

测试时间：2026-11-05 10:46:57 至 10:54:11（Asia/Shanghai）。

| 指标 | 结果 |
|---|---:|
| 请求总数 | 200 |
| 客户端并发 | 200 |
| HTTP 200 / 失败 | 200 / 0 |
| 有效 PNG | 200 |
| 总墙钟时间 | 433.557 s |
| 整机吞吐 | **0.4613 images/s** |
| 平均响应时间 | 225.999 s |
| P50 | 247.863 s |
| P90 | 372.112 s |
| P95 | 373.659 s |
| P99 | 433.337 s |
| 最小 / 最大响应时间 | 62.359 / 433.487 s |
| 返回图片数据量 | 304,311,302 bytes |
| 测试期间 worker error_count | 0 |
| 服务日志错误 | 0 |

worker 0～31 全部参与。8 个 worker 各处理 7 个请求：

```text
1、3、5、6、15、17、26、29
```

其余 24 个 worker 各处理 6 个请求，总计：

```text
8 × 7 + 24 × 6 = 200
```

调度差异最多 1 个请求，分配均衡。

### 28.4 分轮延迟

32 个 worker 每轮最多同时处理 32 张图片，因此 200 个请求形成 7 轮：

| 轮次 | 数量 | 最小延迟 | 平均延迟 | 最大延迟 |
|---:|---:|---:|---:|---:|
| 1 | 32 | 62.359 s | 62.674 s | 62.908 s |
| 2 | 32 | 124.070 s | 124.458 s | 124.838 s |
| 3 | 32 | 185.620 s | 186.312 s | 187.011 s |
| 4 | 32 | 247.751 s | 248.244 s | 249.190 s |
| 5 | 32 | 309.586 s | 310.267 s | 311.437 s |
| 6 | 32 | 371.350 s | 372.209 s | 373.784 s |
| 7 | 8 | 433.212 s | 433.325 s | 433.487 s |

平均响应时间包含排队时间，不表示单张 Base 图片计算需要 226 秒。第一轮 32 张的平均计算/响应时间约 62.67 秒；后续各轮延迟主要是在此基础上累加队列等待时间。

### 28.5 Base 与 Distilled 的性能关系

| 模型 | steps | CFG | 200 请求总耗时 | 吞吐 |
|---|---:|---|---:|---:|
| Distilled | 16 | flow-map，无双分支 CFG | 77.820 s | 2.5700 images/s |
| Base | 50 | guidance 3.5，conditional/unconditional 双 forward | 433.557 s | 0.4613 images/s |

Base 明显更慢是预期行为：步数从 16 增加到 50，并且每个 step 需要 conditional 和 unconditional 两次 Transformer forward。两者不能只按模型名称直接比较单步性能。

### 28.6 原始产物和恢复状态

远端：

```text
/local/fluxcache/benchmarks/flux3-base-32worker-20261105/
/local/fluxcache/benchmarks/flux3-base-200requests-20261105/
```

本地：

```text
outputs/flux3-base-32worker-20261105/
outputs/flux3-base-200requests-20261105/
```

测试结束后已恢复原 Distilled 服务：

```text
容器：flux3-media-server-test
worker：32/32 ready
queue_size：0
canary_state：healthy
所有 worker error_count：0
```

## 29. 单参考图 + prompt 条件推理实现（2026-11-05）

> 历史阶段：本节描述扩展多图/TTNN encoder 之前的单图 CPU 实现；“尚未实现”只表示该阶段状态。后续完成内容和当前边界见第 30 节。

### 29.1 实现依据和边界确认

复核 10.0.28.22 上 vLLM-Omni FLUX.3 实现所对应的参考代码后，确认核心生成链路是：

```text
reference PIL image
  -> FLUX.3 VAE encoder
  -> posterior mean
  -> 2x2 latent packing + frozen BatchNorm
  -> 128-channel image_cond tokens
  -> 5 image_cond mode blocks
  -> 28 text/image_cond/image joint single blocks
  -> denoise output image
```

核心 text encoder 继续只编码文字。参考实现中的 prompt upsampler 可以同时看文字和参考图并生成完整 caption，但它是模型外的可选预处理，不替代 `image_cond` 链路。

### 29.2 TT-Metal 修改

| 文件 | 修改内容 | 作用 |
|---|---|---|
| `models/tt_dit/pipelines/flux3/t2i_contract.py` | `Flux3T2IRequest` 增加 `reference_images`，校验 PIL 类型及最多 10 张 | 把参考图片从 server 传入 pipeline |
| `models/tt_dit/pipelines/flux3/flux3_components.py` | 增加官方单参考图 resize/pad 策略和 `Flux3ReferenceEncoder` | 使用本地 `AutoencoderKLFlux2` 权重产生真实 `image_cond` token |
| `models/tt_dit/pipelines/flux3/sampling.py` | 增加 `make_reference_image_rope()` | 生成空间坐标，并用第四轴负数区分参考图 |
| `models/tt_dit/models/transformers/transformer_flux3.py` | 增加三流 single-block 执行；启用 image_cond embedding、modulation 和 5 个 mode block | 让参考图实际参与全部 DiT 推理，而不是只增加 API 字段 |
| `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 在 denoise loop 前编码参考图，并把 token/RoPE 传给每一步 forward | 串联参考图编码和完整采样过程 |
| `models/tt_dit/tests/flux3/test_reference_conditioning.py` | 校验不同参考图索引只改变四轴 RoPE 的 reference 轴 | 防止多参考图身份坐标退化 |

当前 TTNN 只有匹配的 VAE decoder，尚无 FLUX2 VAE encoder。因此 `Flux3ReferenceEncoder` 是真实但暂时的 CPU fallback：加载同一模型目录下 `vae/` 的 Diffusers `AutoencoderKLFlux2`，不是伪造 latent，也不是把“参考图片”文字拼进 prompt。它在每个 worker 第一次收到参考图时懒加载，不影响纯 T2I worker 的启动和常驻内存。

### 29.3 TL-inference-server 修改

| 文件 | 修改内容 | 作用 |
|---|---|---|
| `tt-media-server/domain/image_generate_request.py` | 增加可选字符串字段 `reference_image` | `/v1/images/generations` 可随 prompt 提交 Base64/data URL 参考图 |
| `tt-media-server/tt_model_runners/dit_runners.py` | 解码 `reference_image` 或兼容 `/image-to-image` 的 `image`，懒加载 reference encoder，并构造 `reference_images` | API 输入进入真实 TT pipeline |
| `tt-media-server/tests/test_flux3_reference_api.py` | 覆盖有图和无图请求模型 | 保证新增字段可选且不破坏 T2I |

### 29.4 请求示例

以下命令从客户端本地图片生成 JSON，并发送 `prompt + reference_image`。图片以 Base64 随 HTTP 请求上传，不是服务器文件路径：

```bash
REFERENCE_IMAGE=/path/to/reference.png \
PROMPT='Preserve the same subject and place it in a snowy pine forest at sunrise.' \
python3 -c '
import base64, json, os, sys
with open(os.environ["REFERENCE_IMAGE"], "rb") as f:
    image = base64.b64encode(f.read()).decode("ascii")
json.dump({
    "prompt": os.environ["PROMPT"],
    "reference_image": "data:image/png;base64," + image,
    "width": 256,
    "height": 256,
    "num_inference_steps": 12,
    "seed": 42,
    "response_format": "file",
    "image_return_format": "PNG"
}, sys.stdout)
' | curl --fail-with-body -sS \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  http://127.0.0.1:8000/v1/images/generations \
  -o flux3_reference.png
```

也可以使用现有 `/v1/images/image-to-image`，把同一 Base64/data URL 放在 `image` 字段。不得在同一请求中同时设置 `reference_image` 和 `image`。

### 29.5 验证过程和结果

1. 主机侧加载模型自带 `assets/1_figurine.png`。
2. VAE encoder 输出 `(1, 3920, 128)` BF16 token；对应网格为 `70 × 56`。
3. reference RoPE 的 cos/sin 均为 `(3920, 128)`，token 数值全部有限。
4. 重启源码 bind-mount 的 32-worker 服务，`/tt-liveness` 显示 32/32 ready。
5. 发送参考图 + prompt 请求，输出保留原机器人/人偶主体，并应用“雪地松林、日出”的场景。
6. 再发送无参考图 T2I 请求，确认旧路径未回归。

| 测试 | HTTP/格式 | 时间 | 输出 |
|---|---|---:|---|
| reference + prompt，256×256，12 steps | 200 / PNG，111,797 bytes | 71.423 s（首次加载 encoder 和首次 image_cond 编译） | `/local/lnn/fluxcache/flux3_reference_edit_256_12steps.png` |
| T2I 回归，256×256，12 steps | 200 / PNG，109,706 bytes | 1.809 s（热态） | `/local/lnn/fluxcache/flux3_t2i_regression_256_12steps.png` |

单元测试结果：TT-Metal reference RoPE 测试 1/1 通过；server 请求模型测试 2/2 通过。服务日志没有 traceback 或 worker error；参考请求由 worker 22 执行并返回 HTTP 200。

### 29.6 尚未实现

- HTTP 多参考图字段和 2～10 张参考图端到端验证。
- FLUX2/FLUX.3 VAE encoder 的 TTNN 实现；当前是 PyTorch CPU fallback。
- prompt upsampler 服务化；当前 prompt 原样送入核心 text encoder。
- Base 50-step 模型的单参考图完整 smoke test；本次真实请求使用 Distilled 模型。

## 30. 多参考图与 TTNN VAE encoder 完整接入（2026-11-05）

### 30.1 核实能力和实现顺序

核对两套 `/local/models` README，均明确支持最多 10 张参考图。随后依次开放有序数组、实现独立多图预处理、建立 TTNN encoder、映射权重、设备端拼接条件、处理实机问题，最后验证两模型同步/异步并恢复原服务。

### 30.2 本轮具体文件和实现位置

下表路径相对于各自 `/local/lnn` 仓库；函数名和行号对应本轮源码。

| 仓库/文件 | 实现位置 | 修改内容及作用 |
|---|---|---|
| TL：`tt-media-server/domain/image_generate_request.py` | `reference_images` 第 32 行；`validate_reference_inputs()` 第 80 行 | 1～10 张、字段互斥、Base64/PIL 校验、编码归一化；非法数组入队前 422 |
| TL：`tt-media-server/tt_model_runners/dit_runners.py` | `TTFlux3ImageKleinRunner.run()` 第 575 行起 | 有序解码，传入现有 MeshDevice，按 worker 懒加载/复用 encoder，记录实际 backend |
| TL：`tt-media-server/tests/test_flux3_reference_api.py` | fixture 与各参数校验测试 | 17 项有图/无图、边界、坏输入和其他 runner 拒绝数组测试 |
| Metal：`models/tt_dit/pipelines/flux3/flux3_components.py` | 多图预处理第 64 行；`Flux3TTNNReferenceEncoder` 第 148 行 | 每图面积上限和 16 网格裁剪；检查配置、111 源张量加载、VAE 哈希 cache key，输出设备 token |
| Metal：`models/tt_dit/models/vae/flux3_vae_encoder.py`（新增） | `__init__` 第 141 行；权重映射第 168 行；`forward` 第 180 行 | 单芯片 encoder、quant_conv、posterior mean、TT packing/BN；仅适配本模块的 GN/Conv；行号已随大图修复更新 |
| Metal：`models/tt_dit/pipelines/flux3/pipeline_flux3.py` | `create_reference_encoder()` 第 87 行；条件处理第 308 行起 | `cpu|ttnn` backend；TT token 不经主机往返，多图设备 concat，逐图独立 RoPE |
| Metal：`models/tt_dit/tests/flux3/test_flux3_reference_preprocessing.py`（新增） | 3 个预处理测试 | 与原 reference RoPE 测试合计 4 项 |

图片解码、PIL/NumPy 预处理和 RoPE 构造仍在 CPU；TTNN backend 的 encoder、packing、BN、条件 token 拼接在 TT。不是“整个请求完全没有 CPU”，也不是将图片强制送入 Qwen 视觉编码器。

### 30.3 实机问题及修复

1. GroupNorm 在小空间尺寸上默认 8×8 核心网格不匹配：新增 `_ReferenceGroupNorm`，按空间 tile 数选可整除行数，全部测试尺寸通过。
2. 同一 encoder 从 16×16 切到 64×64，PCC 只有 0.03829394：卷积 prepared 权重依赖输入几何；新增 `_ReferenceConv2d`，几何变化时释放并重新准备，修复后 PCC 0.99988747，大图后切回小图也通过。
3. root 数值容器产生不可写的 kernel 目录：只修复本轮缓存 ownership，后续统一 UID 1000；未删除缓存、未 reset。
4. Base 临时容器使用未注册模型名退出：两变体均用 `MODEL=FLUX.3-Image-Klein`，通过权重目录区分；后续保留日志验证成功。
5. 主机型测试受镜像缺少根测试树及默认 JUnit 写权限影响：仅本组纯预处理/RoPE 测试限制 conftest 并覆盖 addopts，最终 exit 0。

### 30.4 最终验证与产物

- 请求模型测试 17/17；预处理/RoPE 测试 4/4。
- Base/Distilled 各执行 `16×16 → 64×64 → 128×96 → 256×256 → 512×384 → 64×64` encoder 比较，12/12 通过；PCC 为 0.999809～0.999932，relative RMSE 为 0.01175～0.01997，缓存重载通过。
- 两模型各 9 个 HTTP 场景，18/18 通过：T2I、旧单图、新单图、两图、交换顺序、10 图均 200 PNG；11 图/坏图片 422；两图异步 202 后 completed。
- 参考图片均缩至 64×64，输出 256×256；10 张是两张资产交替。Base 50 步 CFG 3.5，Distilled 12 步，seed 42。
- 同模型旧/新单图 PNG 哈希相同；交换两图顺序 PNG 哈希不同。这里是兼容性与功能验证，不是全模型语义/精度验收。

结果保存于 `/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-054733/results.json`、`multiref_ttnn_base_verified_20261105/` 和 `multiref_ttnn_distilled_verified_20261105/`。最终日志目录为 `/local/lnn/fluxcache/logs/multiref_validation_20261105/20261105-054705/`。

### 30.5 小图阶段的部署与未完成边界（历史）

该历史阶段恢复后的 `flux3-media-server-test` 使用 CPU 默认 backend；该记录不代表当前配置。自本次默认值修订起，reference backend 默认 TTNN，只有显式设置 `FLUX3_REFERENCE_ENCODER_BACKEND=cpu` 才使用 CPU 回退。

该小图阶段当时尚未逐项测试 3～9 张、10 张大图容量和 32-worker 参考图性能；这些项目的当前结论见第 31 节。自动 OOM fallback 仍未实现。已有 T2I 200 请求压测不能替代参考图测试。本轮修改为未提交工作树，没有新建镜像或执行 commit/push。

完整步骤、准确数值表、单次 HTTP 时间、启动参数、客户端 curl 和复现脚本见 `FLUX3_MULTI_REFERENCE_AND_TTNN_VAE_IMPLEMENTATION.md`。

## 31. 大尺寸多参考图与 32-worker 参考图压测（2026-11-05）

本节更新第 30 节小图阶段的验证范围。完整参数、逐项结果、失败日志、修复过程与性能统计见 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md`，不使用历史 T2I 压测替代参考图请求。

### 31.1 实机暴露的问题与代码修改

| 文件与实现入口 | 怎么修改 | 作用 |
|---|---|---|
| `/local/lnn/tt-metal/models/tt_dit/models/vae/flux3_vae_encoder.py:21`，`_reference_width_slice_limit()` | 按 kernel/stride/padding 计算输出宽，再计算 tiled 切片上限 | 下采样不能按输入宽沿用 decoder 切片数 |
| 同文件 `:47`，`_ReferenceConv2d.__init__()` | 按实例复制默认/mesh 切片表；stride-2 表中的片数按输出宽限制 | 修复 512 方图出现 4 片超过 2 片上限，且不改公共 decoder 配置 |
| 同文件 `:66`，`_ReferenceConv2d.forward()` | 大面积 RGB 输入使用显式 DRAM_HEIGHT 切片，首次从原始 OIHW 权重执行并缓存 prepared weight/bias；尺寸变化仍重新准备 | 规避 1024 RGB 输入卷积数值失真，保持后续请求复用 |
| `/local/lnn/TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py:635`，`get_pipeline_device_params()` | 加入 `FLUX3_L1_SMALL_SIZE`，默认 131072 B；限制 64～128 KiB、32 B 对齐 | 为大尺寸 halo/元数据预留空间，避免 256 KiB 试验侵占 DiT 静态 CB |
| `/local/lnn/tt-metal/models/tt_dit/tests/flux3/test_flux3_reference_conv_slicing.py`（新建） | 12 项输出宽、padding/stride 边界和公共表不变测试 | 回归下采样片数修复 |
| `/local/lnn/TL-inference-server/tt-media-server/tests/test_flux3_device_config.py`（新建） | 9 项默认预留、可配预留及非法值测试 | 检查设备参数和错误提示 |
| `/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py:137`，`_refresh_reference_program_cache()`；`:191`，`Flux3TTNNReferenceEncoder.encode()` | 参考几何变化时同步设备并清理旧运行时程序；相同几何直接复用 | 修复连续横竖尺寸切换造成 L1_SMALL 元数据累计；保留磁盘 kernel/权重缓存和常驻模型权重 |
| `/local/lnn/tt-metal/models/tt_dit/tests/flux3/test_flux3_reference_program_cache.py`（新建） | 3 项同步/清理顺序及同几何不清理测试 | 保证不在固定尺寸热态请求中逐次清理程序缓存 |
| `/local/lnn/TL-inference-server/tt-media-server/Dockerfile` | 保留 `FLUX3_COMPONENT_RESIDENCY=resident`，补充 TTNN backend、128 KiB 设备预留和 runtime 挂载说明 | 说明最终运行参数；仅更新构建源文件，本轮没有重新构建镜像 |

通用大面积 width strips 和 256 KiB 预留都曾试验失败，已撤回，不能当作交付配置。未改原始模型权重、公共 Conv2d/decoder 或 C++ runtime；本轮仍使用源码 bind mount，无需重建镜像。

### 31.2 验证如何执行

1. 重跑真实配置/API 测试 26/26、切片/预处理/RoPE 测试 16/16。
2. 新增 `/local/lnn/fluxcache/flux3_reference_layer_diagnostic.py`：逐个卷积读取实际 TT 输入，用同层权重做 CPU 卷积对照，定位到首层 RGB；调整后 25 个卷积全部通过。
3. 扩展 `/local/lnn/fluxcache/test_flux3_ttnn_vae_encoder.py`：最终两模型各执行 11 次、从 16 到 1024 方图及横竖图再切回 64，检查最终 token shape、有限性、PCC、relative RMSE，以及缓存重载。
4. 重跑两模型同步/异步 18 个场景，确认旧单图兼容、交换顺序影响结果、坏输入/11 图 422。
5. 新建 `/local/lnn/fluxcache/flux3_reference_load_test.py`：每模型执行 14 项数量/尺寸矩阵，随后单芯片预编译固定负载、32 冷覆盖、32 暖覆盖、200 请求同时提交。所有成功响应实际解码、校验尺寸/格式、保存图片及 SHA256。
6. 新建 `/local/lnn/fluxcache/flux3_large_reference_control.py`：原服务空闲后暂停，临时服务用 8001；每阶段保存客户端及服务日志，统计实际参与 worker，finally 恢复 8000 原服务。
7. 新建 `/local/lnn/fluxcache/flux3_reference_benchmark_report.py`：汇总请求成功率、吞吐、延迟、队列/主机内存采样、worker 启动重叠及图片非空检查；最终验收不能只看服务 ready。

### 31.3 已验证结果与当前状态

第一次扩大测试时，两模型 16 次 encoder 数值/尺寸切换及缓存重载通过；1024×1024 PCC 0.999818265、MAE 0.01389346、relative RMSE 0.01916481。该历史结果在 `/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-072719/results.json`，两模型 HTTP 兼容性 18/18 通过；最新数值验收采用下面的 22 次结果。

随后 `20261105-073240` 端到端轮次通过 Distilled 13/14 项，含十张 1024×1024 图，324.327 秒；混合长宽场景出现 L1_SMALL 元数据分配不足，未执行该轮 32/200 负载。追加 `flux3_components.py:_refresh_reference_program_cache()`：参考图几何变化时同步设备并调用 `MeshDevice.clear_program_cache()`，释放旧形状运行时程序对象，保留常驻模型权重和磁盘 JIT；相同参考几何复用原运行时缓存。新增 `test_flux3_reference_program_cache.py` 的 3 项切换/不切换测试。

扩展到每模型 11 次、共 22 次数值验证后全部通过，含 1024×768、768×1024、512×1024 及切回 64；PCC≥0.999798894、relative RMSE≤0.02044829，缓存重载通过。最新结果是 `/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-075049/results.json`。当时完整端到端矩阵和 32-worker 仍需重跑；其后的最终验收已在下文完成，不能以单层或 encoder 数值通过替代服务负载结果。

修复后的大图和参考图压测执行根为 `/local/lnn/fluxcache/large_reference_load_20261105/20261105-075518/`。Distilled 已完成 14/14 大图矩阵、32/32 首次覆盖、32/32 暖态覆盖、200/200 参考图请求；200 请求总墙钟 119.631 秒、吞吐 1.6718 张/秒、平均完整响应 62.015 秒、P95 102.270 秒。全部 32 worker 实际参与，各处理 6～7 个请求，错误/重启均 0。十张 1024×1024 参考图单芯片通过，315.165 秒；混合长宽场景通过，60.938 秒。

Base 也完成 14/14 大图矩阵、单芯片预编译、32/32 首次覆盖、32/32 暖态覆盖和 200/200 参考图请求。固定负载的 200 请求总墙钟 681.294 秒、吞吐 0.2936 张/秒、平均完整响应 354.129 秒、P95 585.335 秒；32 个 worker 全部实际参与，各处理 6 或 7 个请求，错误/重启均 0。十张 1024×1024 参考图单芯片通过，1922.377 秒；混合长宽场景通过，211.933 秒。

最终 `flux3_reference_benchmark_report.py --require-complete` 验收为 `distilled=true`、`base=true`、`original_restored=true`、`all_passed=true`。原 `flux3-media-server-test` 已恢复为 Distilled、8000、32/32 ready、queue=0、canary healthy，默认 reference backend 仍为 CPU。完整响应时间包含排队及客户端图片校验，不是纯设备计算时间；详细矩阵、负载和问题记录见 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md`。


## 32. 2026-11-07：joint RoPE 请求级复用

### 32.1 问题和边界

纯 T2I 路径此前在每个 denoise step 的每一个 single block 内重复调用 `_joint_rope(text_rope, image_rope)`。一次 forward 有 28 个 single block；一次请求则按 scheduler step 数重复。带参考图路径虽已在 single block 循环外构造 joint RoPE，但仍会在每一个 denoise step 重新构造。

text/image/reference-image 的 mode block 在数学上相互独立，但当前共享同一完整 TT mesh，且各 block 内含分布式 layer norm、attention、tensor-parallel 通信和 FFN。没有经过 command queue / collective 并发安全验证时，不能以 Python 线程强行并发；本轮明确不改变三路 mode block 的稳定串行调度。

### 32.2 实现

TT-Metal 的 `transformer_flux3.py:Flux3Transformer.forward()` 增加可选 `joint_rope` 参数。未提供时保持旧接口行为：按当前 text、reference-image、image token 现场构造；提供时校验 joint sequence length 后复用。这样既兼容其他调用方，也阻止将错误长度的 RoPE 用于三流或双流 attention。

`pipeline_flux3.py` 在进入 denoise loop 前，根据静态条件构造：

- `prompt_joint_rope`：prompt text + image，或 prompt text + reference-image + image；
- `negative_joint_rope`：仅在 Base CFG 负提示词存在时构造，对应 negative text 的独立序列长度。

它们都是 TT device tensor，在整次请求的全部 denoise step、全部 28 个 single block 中复用；没有改变 token、RoPE 数值、scheduler、权重或 attention 计算。

### 32.3 修改文件和验证

| 仓库 | 文件 | 修改 |
|---|---|---|
| tt-metal | `models/tt_dit/models/transformers/transformer_flux3.py` | 接收和校验预构造 joint RoPE；T2I 与参考图路径都只消费同一份 RoPE |
| tt-metal | `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 在 denoise loop 外建立 prompt/negative joint RoPE，并传给每一次 Transformer forward |

已完成 Python `py_compile` 和 `git diff --check`。TT-Metal 提交：`50f8a61b4e Optimize FLUX.3 joint RoPE reuse`。正式镜像和 API 回归的结果在本节后续版本记录补充。

## 33. 2026-11-11：Galaxy03 两芯片 SP2 实机验证与 Ring-SDPA 调优结论

### 33.1 目标、服务与最终可用配置

本轮在 Galaxy03 验证 FLUX.3 的 `TP1 × SP2`：Transformer 权重在两颗芯片复制，图像 token 沿第一轴切为两份；text token 保持复制并作为 Ring joint stream。实际服务容器为 `flux3-sp2-65169-layoutcache`，监听 `18093`，使用：

- `FLUX3_TP_SIZE=2`
- `FLUX3_PARALLELISM=sp`
- `FLUX3_DEVICE_IDS=(0,1)`
- `TT_MESH_GRAPH_DESC_PATH=/tmp/bh_2_1_line_line.textproto`
- `TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`

TP4 容器 `flux3-glx-1x4-ring` 因占用整机 32 颗芯片，已停止；本轮不恢复它。SP2 最终保持默认 Ring-SDPA 分块 `(q=256,k=256)`，服务健康、`model_ready=true`。

### 33.2 已实现：按 layout 缓存静态 RoPE

`/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py` 新增三个常驻于 pipeline 实例的设备端缓存：

| 实现位置 | 缓存内容 | cache key 包含的边界 |
|---|---|---|
| `__init__()` 的 `_text_rope_cache` | 正/负 prompt 的 text RoPE | device mesh、TP/SP factor、SP axis、text token 数 |
| `_cached_image_rope()` 的 `_image_rope_cache` | 指定宽高/参考图数量的 image RoPE | layout scope、latent 高宽、参考图布局 |
| `_cached_t2i_joint_rope()` 的 `_joint_rope_cache` | 非 SP T2I 的 text+image joint RoPE | layout scope、text/image token 长度 |

`run_external_prompt_embeds()` 改为首先查询这些缓存。SP2 路径直接将 `prompt_joint_rope` / `negative_joint_rope` 设为 `None`，因为 `Flux3Transformer.forward_sequence_parallel()` 直接将 text/image RoPE 传给 `ring_joint_scaled_dot_product_attention`，不会消费拼接后的 joint RoPE。这个改动不改变 RoPE 数值、模型权重或 scheduler。

### 33.3 基线实测

请求固定为 `a black bird`、`1024×1024`、`16 steps`、`seed=42`、`response_format=file`、PNG。先执行一次不计入统计的 warmup，再连续执行四次，取后 3 次。四个基线 PNG 的 SHA256 完全相同，说明该路径在同 seed 下确定性输出。

| 指标 | 后 3 次平均 |
|---|---:|
| curl 端到端 | 9.449 s |
| pipeline total | 8.981 s |
| denoise | 7.685 s |
| VAE decode | 1.244 s |

结果保存于 `/local/lnn/fluxcache/sp2-results/flux3_sp2_65169_layoutcache_1024_16steps_seed42_run{1..4}.png`；详细客户端耗时在 `/local/lnn/fluxcache/logs/flux3_sp2_65169_layoutcache_1024_16steps_metrics.txt`。

### 33.4 未保留：按 #65169 思路尝试 Ring-SDPA 分块

当前 `attention_opt.py` 对 `(Blackhole, SP2, TP1)` 没有专用条目，1024 图的全局图像长度 4096 使用默认 `(256,256)`。本轮对照上游 #65169“按每设备 rows 选择分块”的思想，做了三项真实设备尝试；所有候选在测试后已从源码撤回。

| 候选 | 真机结果 | 结论 |
|---|---|---|
| `(512,512)` | 静态 circular buffer `2,250,816 B`，超过 L1 `1,572,864 B` | 无法 warmup，撤回 |
| `(320,512)` | static CB `1,648,704 B`，仍超过 L1 | 无法 warmup，撤回 |
| `(288,512)` | static CB 与运行时 L1 buffer 重叠（static end `1,548,352`，buffer start `1,438,720`） | 无法 warmup，撤回 |
| `(256,384)` | 可运行；pipeline -0.21%、denoise -0.24%、curl -0.18% | 同 seed PNG PSNR 仅 30.33 dB，89.37% 像素不同；收益过小，不接受，撤回 |

`(256,384)` 的图像哈希不同不是服务编码问题，而是 Ring attention 的 BF16 分块/归约顺序变化造成的数值差异。当前没有官方 Golden Data 门限支持以约 0.2% 速度收益交换这一差异，因此最终恢复默认分块。恢复后同 seed 输出 SHA256 已再次与基线完全一致。

### 33.5 当前结论与下一步

- [已实现并验证] SP2 服务、2×1 Line mesh、resident prompt encoder/Transformer/VAE、请求级静态 RoPE 缓存。
- [已验证] 1024×1024、16-step T2I 的健康检查、文件响应、确定性输出和热态性能。
- [已撤回] 本轮三个 L1 不可用 Ring-SDPA 分块以及一个精度收益不划算的可运行候选；`attention_opt.py` 没有保留本轮试验改动。
- [未开始] 以 profiler/Golden Data 为门槛，研究 Ring-SDPA streaming FP32 destination、per-device row chunk table、matmul blocking 和 VAE 2D/H-sharded decode。后续不得仅凭 PNG 可解码或单次时间提交变更。

本轮结束时 TT-Metal 工作树仍有 `transformer_flux3.py` 与 `pipeline_flux3.py` 两个未提交文件：前者包含 SP2 路径，后者包含 SP2 调度和本节的 RoPE cache；在与完整回归/提交边界确认前没有新建 commit。

## 34. 2026-11-14：Galaxy03 四个 TP1×SP8 Ring worker

### 34.1 目标和拓扑

新增 `FLUX3_EXECUTION_PROFILE=sp8`。完整 Galaxy 的 32 颗芯片会自动拆成四个互不重叠的 `8×1` Ring worker；每个 worker 的 DiT 使用 `TP1×SP8`（权重复制、image token 沿 Ring 分片），因此服务总计为 4 workers，而不是把连续 8 张同 tray 芯片误当作 Ring。

Galaxy03 经 `tt-smi -glx_list_tray_to_device` 发现的 tray 顺序为 `0..7`、`8..15`、`24..31`、`16..23`。TT-Metal upstream 的 `create_submeshes(MeshShape(8,1))` 和 intragalaxy torus binding 对应的安全分组为：

```text
(0,1,2,3,11,10,9,8)
(4,5,6,7,15,14,13,12)
(28,29,30,31,23,22,21,20)
(24,25,26,27,19,18,17,16)
```

### 34.2 已实现文件

| 仓库 | 文件 | 改动 |
| --- | --- | --- |
| TL-inference-server | `tt-media-server/config/settings.py` | 增加 `sp8` / `sp-8` profile，逻辑 mesh `(8,1)`，自动创建 4 个 worker。 |
| TL-inference-server | `tt-media-server/model_services/scheduler.py` | SP8 后续 worker 必须等前一 Ring 的模型 warmup 真正完成；再执行 `NEW_DEVICE_DELAY_SECONDS`（默认 120 秒）的稳定等待。`All workers ready` 仅在 4 个预期 worker 都已注册且 ready 时输出。 |
| TL-inference-server | `tt-media-server/domain/image_generate_request.py` | 将 SP8 的容量边界前置为 API/Pydantic 校验：只接收正方形 `1024` 或 `2048`，4096 在入队前返回 422，而不是在设备 worker 中变为 500。 |
| TL-inference-server | `tt-media-server/utils/device_manager.py` | 新增 `get_flux3_sp8_groups()`；按 Galaxy 物理 tray 布局构造四个 column Ring，而非沿用 `get_device_groups_of_eight()` 的连续 tray 分组。 |
| TL-inference-server | `tt-media-server/utils/runner_utils.py` | SP8 自动指向专用 8×1 Ring descriptor；若继承旧 SP4 descriptor 则安全替换。 |
| TL-inference-server | `tt-media-server/descriptors/bh_galaxy_flux3_sp8_ring.textproto` | 新增无设备 ID 的通用 8×1 Ring mesh descriptor；每个 worker 由自身 `TT_VISIBLE_DEVICES` 绑定不同 column Ring。 |
| TL-inference-server | `tt-media-server/tt_model_runners/dit_runners.py` | 将 SP8 纳入 pipeline 创建、warmup、run、Ring CCL 和 `TP=(1,1), SP=(8,0)` 并行因子。 |
| TL-inference-server | `tt-media-server/tests/test_device_manager.py` | 为实际 Galaxy03 tray 编号加入两项 SP8 Ring 映射测试。 |
| TL-inference-server | `tt-media-server/tests/test_flux3_request_and_auth.py` | 新增 SP8 1024/2048 接受、4096/非正方形拒绝的 API 请求模型测试。 |
| tt-metal | `models/tt_dit/pipelines/flux3_klein/pipeline_flux3_klein.py` | TP1×SPN 下让 Qwen3-VL text encoder 在 SP 轴做 weight-only TP，避免 cache 中 replica 标签和实际 shard 内容不一致。 |
| tt-metal | `models/tt_dit/models/transformers/flux3_klein/transformer_block_flux3_klein.py` | TP1×SPN 禁用错误使用 factor-one TP 轴的 fused MMRS projection；Ring-SDPA 仍启用。 |
| tt-metal | `models/tt_dit/pipelines/flux3_klein/vae_flux3_klein.py` | 对 H 分片 decoder conv 保留显式 DRAM width slice；修复 4096 下 auto unsliced path 尝试 1 GiB 中间分配造成的 DRAM OOM。 |

### 34.3 问题记录与处理

1. 初版误用每 tray 连续 8 卡。它不是 Galaxy 的 8 卡 Ring；在某些 collective 上会出现 same-source/same-destination Fabric route。已替换为 upstream column Ring 映射。
2. 初版 SP8 的 4096 warmup 在 VAE H-sharded conv 报 `Out of Memory`：每芯片已分配约 3.96 GiB，仅剩约 307 MiB 连续 DRAM，而未切片路径试图跨 8 bank 分配 1 GiB。已改为显式 DRAM width slicing。
3. 首次 explicit-slice 尝试仍在同一 `514×4096×256→256` halo conv 报 L1 OOM（每 bank 需要 839,680 B、仅余 659,712 B）。一次错误的 16-row activation-block 尝试又被 TTNN 明确拒绝，因为 Conv2d block 必须为 tile height 32 的倍数。最终 SP8 配置使用合法的 32-row block、关闭 double buffer，并将 H-partitioned conv width slices 加倍；保留 SP1/SP2/SP4 原有的 64-row double-buffer schedule。
4. 全量 `test_device_manager.py` 在真实 Galaxy host 有一个既有环境相关失败：`test_get_device_pairs_failure` 仅 mock system-health、但代码会 fallback 到真实 `tt-smi`，所以机器实际返回 pairs。该失败与 SP8 改动无关；新增的两个 SP8 node tests 已单独通过。
5. 经过多个尺寸和参考图 layout 的 warmup 后，`Conv2d._prepared` 会保留每个输入 shape 的 `prepare_conv_weights` 结果。即使已启用 DRAM slicing，SP8 worker 仍会在下一轮 `514×4096×256→256` conv 前只剩约 323 MiB DRAM，无法分配 1 GiB output activation。先尝试“仅在 shape 切换时释放”，但第二轮 text bucket 中 shape 未切换仍复现 OOM，说明 prepared 状态/运行时分配仍会累积。最终在每次 VAE encode/decode 完成后，仅对 TP1×SP8 递归释放各层 `Conv2d._prepared`；基础权重与已编译 program 不释放。此前实测该策略下后续 4096 decode 约 3.4 秒，证明不会反复 JIT 编译；SP1/SP2/SP4 保持原有常驻策略。
6. 一次在 worker 关闭后未完成 Galaxy reset 就立即重新打开全部 32 芯片，UMD 记录了 `Setting power state failed ... Connection timed out`，继而三个 worker 报 `MMIO per-op timeout`。该启动作废；标准验证流程固定为：停止容器 → `tt-smi -glx_reset_auto` 完成且 `tt-smi -glx_list_tray_to_device` 恢复 32 芯片 → 再启动容器。不能以容器仍是 `Up` 判定设备已可用。
7. 即使每次回收 `Conv2d._prepared`，在第二个长文本 bucket 的 4096 仍会稳定失败：`514×4096×256` 输出本身约为 1 GiB/芯片，而完整 resident DiT + Qwen3-VL 后每颗 Blackhole 只余约 350 MiB 可分配 DRAM。这是 8 芯片总并行度下的容量边界，不是拓扑或缓存遗漏。因而 `sp8` profile 明确服务 `1024` 和 `2048`；`4096` 由已有的 `sp8tp4`（32 芯片，VAE 二维空间并行）profile 服务。API 会在入队前返回明确的 unsupported-size 错误，不再让 worker 在设备端 OOM。
8. 四个 SP8 worker 若在同一秒并发打开并 warmup，前三组可就绪而最后一组可能长时间卡在 device poll、既不报错也不发 ready 信号。`/tt-liveness` 的 `model_ready` 只表示至少一个 worker 可用，不能代表四组均已就绪。初版只按“进程 start 时刻”固定等待 120 秒，第三组 ready 后约 13 秒仍会启动第四组，不能消除重叠。最终改为 scheduler readiness barrier：前一 worker 发送真实 warmup ready 信号后，才开始 `NEW_DEVICE_DELAY_SECONDS=120`（用户显式指定时尊重）稳定等待，再启动下一 Ring；并且只有 `len(worker_info)==worker_count` 且全部 ready 时才写 `All workers ready`。正常推理时四 worker 仍并发处理请求。

### 34.4 交付镜像和验证状态

正式无源码 bind mount 镜像已构建：`galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8`，image ID `sha256:67b9e36224e328b1afe9f94f73b529fb4f711f1369d7c1fff97312699abef67c`。Docker context 显式复制本节的 server、Transformer、pipeline、VAE 和两种 Ring descriptor。

### 34.5 正式镜像真机验证（完成）

正式镜像：`galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.4-sp8`，image digest `sha256:a469908351b564896a56d834e9eb4d9506636ebd8798ff80d40f999a9089b3d3`。运行容器 `flux3-v104-sp8-formal` 只挂载 hugepages、`/local/models`、`/local/lnn/fluxcache` 和 TT-Metal cache；`docker inspect` 确认没有任何 TL-inference-server 或 tt-metal 源码 bind mount。

首次正式镜像验证曾在不完整 reset 后卡在首个 worker warmup 后段。根因不是镜像漏文件：容器内 scheduler、request model、pipeline 和 VAE 的 SHA256 均与仓库文件相同。根因是前台执行 `tt-smi -glx_reset_auto` 的外部会话在 22 秒左右被中断，未完成芯片 re-initialization。最终流程改为后台等待 reset 完整结束，日志明确出现 `All 32 chips found in "/dev/tenstorrent"` 和 `Re-initialized 32 boards after reset`，再启动容器。

正式验证时间线：第 1/2/3/4 Ring 分别在 `18:27:05`、`18:30:51`、`18:34:39`、`18:38:25`（Galaxy03 本机 2026-11-13）报告 ready；第 4 组 ready 后才出现一次 `All workers ready`。四组均 `is_alive=true`、`is_ready=true`、`error_count=0`。

随后向端口 `18130` 并发发送 4 个 `a black bird`、`1024×1024`、16-step、`response_format=file`、PNG 请求，四个 Ring 各处理一个任务。HTTP 总时间为 `3.028`、`3.078`、`2.992`、`2.958` 秒；4/4 返回有效 1024×1024 RGB PNG。设备端 `denoise` 为 `1.845–1.861 s`，VAE decode 为 `0.634–0.668 s`。图片保存在 `/local/lnn/fluxcache/sp8-formal-v104-20261113/black_bird_sp8_formal_seed{52..55}.png`。

对 `4096×4096` 请求，API 在调度前返回 HTTP 422：`FLUX.3 SP8 serves only square 1024x1024 or 2048x2048 ... Use the sp8tp4 profile for 4096x4096.`；日志没有 worker 执行任务或 error count 增长。这证实容量边界已在 API 层生效。
