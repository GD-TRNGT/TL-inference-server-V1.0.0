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

尚未完成：

- 2 芯片和 4 芯片 mesh 的权重分片、通信和性能验证；
- 与 PyTorch/vLLM/官方参考实现的逐层精度对齐；
- kernel、layout、trace、内存和吞吐优化；
- 独立的异步任务 API、任务状态管理和 cancel 语义。

本文件只描述真实代码、真实修改、真实验证和当前状态，不把早期失败结论当作当前状态。

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

### 17.3 当前工作树状态

- TL-inference-server 的 FLUX.3 服务注册和 runner 修复仍有未提交工作树改动；
- tt-metal 的 `pipeline_flux3.py` 仍有未提交 runner 兼容改动；
- `flux3_prompt_encoder.py`、两个 T2I 测试脚本是当前工作树新增文件；
- 已提交的历史 commit 包含 Transformer、权重映射、缓存、sampling 和早期验证实现；
- 本文档记录的是当前工作树实际运行状态，不等于所有代码已经 commit。

另外，`config/constants.py` 中存在一处与 FLUX.3 无关的 `SP_RUNNER` 设备 ID 改动，后续提交前应单独检查，避免把无关修改带入 FLUX.3 commit。

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
