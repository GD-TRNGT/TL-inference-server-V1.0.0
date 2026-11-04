# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 上的完整适配记录

更新时间：2026-11-04

## 1. 当前结论

当前已经完成并在 10.12.11.2 的 Blackhole Galaxy 上验证：

- FLUX.3 Image Klein 模型目录和配置校验；
- TTNN Transformer 结构，包括 5 个 text mode block、5 个 image mode block、5 个 image-condition mode block 和 28 个 single block；
- DiT、Qwen3-VL Text Encoder、VAE Decoder 的权重读取和 TTNN 缓存；
- 单芯片 `(1,1)` mesh 上的真实 Transformer 执行；
- 多步 denoise、Qwen3-VL prompt 编码、VAE decode 和 PNG 输出；
- `tt-media-server` 的 FLUX.3 runner 和 `/v1/images/generations` API；
- 32 个单芯片 worker 启动、warmup 和真实请求；
- 64 个并发 T2I 请求，64/64 成功，并确认 worker 0～31 全部实际处理请求。

当前明确暂缓或未完成：

- 2 芯片和 4 芯片 mesh 的权重分片、通信和性能验证；
- 与 PyTorch/vLLM/官方参考实现的逐层精度对齐；
- kernel、layout、trace、内存和吞吐专项优化；
- 服务重启后的历史 image job 恢复和独立图片下载接口。

已完成的图片异步任务 API、任务状态管理和 cancel 语义见第 19 节。本文件只描述真实代码、真实修改、真实验证和当前状态，不把早期失败结论当作当前状态。

## 2. 实现环境和代码边界

### 2.1 机器、模型和缓存

```text
机器：10.12.11.2
硬件：Blackhole Galaxy，32 个芯片
模型：/local/models/flux-3-image-klein-base
持久化缓存：/local/lnn/fluxcache
服务容器：flux3-media-server-test
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v5
分支：flux-3-image-klein
```

模型权重没有复制进 Docker image，而是运行时只读挂载：

```text
/local/models/flux-3-image-klein-base
  -> /local/models/flux-3-image-klein-base:ro
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
| `tt-media-server/Dockerfile` | 使用本地 tt-metal BuildKit context；增加 FLUXCACHE 环境；不复制模型权重；不再 clone 远端 tt-metal |
| `tt-media-server/config/constants.py` | 注册 FLUX.3 model、runner、model mapping、Galaxy 单芯片 32-worker 配置 |
| `tt-media-server/open_ai_api/models.py` | 将 FLUX.3 runner 接入 `BaseImageRequest` |
| `tt-media-server/tt_model_runners/dit_runners.py` | 增加 `TTFlux3ImageKleinRunner`、绝对模型路径读取、pipeline 调用和 list 响应修复 |
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
| `models/tt_dit/pipelines/flux3/pipeline_flux3.py` | 外部 embedding、Qwen3-VL 文本路径、多步 denoise、VAE decode、PNG 输出 |
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
TL-inference-server：020af3de Add FLUX3 image async jobs and cancel API
tt-metal：a3550fc781 Complete FLUX3 T2I pipeline integration
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
- [已完成] 记录单请求约 12.26 秒、64 并发平均约 28.94 秒、最大约 39.89 秒；
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
```

本文件的当前状态以第 1 节、第 15～18 节为准；早期单步验证只是完整流程中的一个阶段，不再作为当前总体状态。

## 19. 2026-11-04：异步图片任务与取消实现

### 19.1 实现位置和作用

| 文件 | 实现内容 |
|---|---|
| `tt-media-server/config/constants.py` | 增加 `JobTypes.IMAGE`，使图片任务可以进入统一 JobManager |
| `tt-media-server/model_services/image_service.py` | 增加 `create_image_job`、状态查询、结果读取、cancel 和 delete；生成完成后把 base64 列表写入每个 job 的 JSON 文件 |
| `tt-media-server/open_ai_api/image.py` | 增加 async submit、job metadata、cancel 和 delete 路由；原同步 `/generations` 保持兼容 |
| `tt-media-server/utils/job_manager.py` | 复用已有任务状态、队列准入、取消和清理逻辑，不重复实现视频任务管理器 |

### 19.2 请求流程

1. 客户端向 `/v1/images/generations/async` 发送 JSON 请求。
2. API 创建 `IMAGE` 类型 job，立即返回 `202` 和 job id。
3. JobManager 在后台调用现有 `ImageService.process_request()`，因此仍然经过 FLUX.3 runner、TTNN denoise、VAE 和图片后处理。
4. 完成后把 `{"images": [...]}` 写到 `IMAGE_JOB_OUTPUT_DIR/{job_id}.json`。
5. 客户端轮询 `/v1/images/generations/{job_id}`；完成时响应中包含 `images`。
6. cancel 调用 JobManager 的 cooperative cancellation，并触发现有 scheduler task cleanup。

### 19.3 真实验证结果

```text
服务：flux3-media-server-test
runner：tt-flux.3-image-klein
mesh：(1,1)

异步提交：HTTP 202，job_type=image，status=queued
50-step cancel：queued -> cancelling -> cancelled
12-step completion：queued -> completed，image_count=1
服务复启后：model_ready=true，canary_state=healthy
```

### 19.4 本轮仍明确暂缓

- [暂缓] 2 芯片和 4 芯片并行、权重分片和通信优化；
- [暂缓] 与 vLLM/PyTorch 的 Golden Data 精度对齐；
- [未开始] 专项 matmul/trace/kernel 性能优化；
- [未完成] 服务重启恢复、独立图片下载接口和长请求压力回归。

## 20. 按实施步骤对应的文件变更索引

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

## 21. 2026-11-04：实现 `response_format=file`

### 21.1 实现文件

本次只修改 `TL-inference-server`，`tt-metal` 不需要修改：

| 文件 | 修改内容 |
|---|---|
| `tt-media-server/domain/image_generate_request.py` | 在 `BaseImageRequest` 增加 `response_format`，支持 `b64_json`（默认）和 `file`。 |
| `tt-media-server/open_ai_api/image.py` | 增加 Base64 解码、媒体类型选择、单图直接返回、多图 ZIP 打包逻辑。 |
| `tt-media-server/tests/test_kontext_api.py` | 增加文件响应、默认 Base64 响应和非法格式校验。 |

### 21.2 请求处理流程

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

### 21.3 使用方式

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

### 21.4 当前验证状态

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

## 22. 2026-11-04：50 步分辨率请求测试

测试条件：11.2 的 `flux3-media-server-test`，同步 `/v1/images/generations`，相同 prompt、`seed=42`、`num_inference_steps=50`、`response_format=file`、PNG 输出。

| 分辨率 | HTTP 结果 | 服务端处理时间 | 客户端总耗时 | 结果 |
|---|---:|---:|---:|---|
| 256x256 | 500 | 0.0031 s | 0.0066 s | 未执行推理；当前第一阶段 runner 只支持 checkpoint 默认分辨率 |
| 512x512 | 500 | 0.0022 s | 0.0044 s | 未执行推理；同上 |
| 1024x1024 | 200 | 35.4095 s | 35.4413 s | 成功生成 PNG，1024x1024，2,117,127 bytes |

256x256 和 512x512 的失败发生在 worker 分辨率检查阶段，错误为：

```text
the first milestone supports only the checkpoint default resolution
```

因此这两个耗时不是推理耗时。当前可测的 1024x1024、50 步推理时间约为 **35.41 秒**（客户端观测约 **35.44 秒**）。
