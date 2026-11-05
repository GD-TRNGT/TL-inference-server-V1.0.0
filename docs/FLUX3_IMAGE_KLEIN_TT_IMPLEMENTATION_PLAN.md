# 在 10.12.11.2 的 Blackhole Galaxy 上实现 FLUX.3 Image Klein 推理

完整阶段性适配记录见：[FLUX3_IMAGE_KLEIN_TT_ADAPTATION_COMPLETE.md](C:/Users/Administrator/Documents/Codex/2026-10-27/https-github-com-tenstorrent-tt-inference/docs/FLUX3_IMAGE_KLEIN_TT_ADAPTATION_COMPLETE.md)

更新时间：2026-11-05
开发仓库：/local/lnn/TL-inference-server
开发分支：flux-3-image-klein
当前功能基线：TL-inference-server `e46186a9160d48344ec1f706f6ff4efd6f6c53ba`；tt-metal `3ff5dc122dae38d173870cf5525e9ed96d877c33`

本文基于当前仓库代码、10.12.11.2 的硬件信息以及已经检查过的 vLLM-Omni FLUX.3 实现整理。目标是在 TT-NN/TT-Metalium 上真正完成模型加载、推理、结果输出、精度对齐和性能优化，而不是只增加一个 API 名称。

## 0. 当前状态和阅读说明

本文前半部分是设计方案，后半部分保留按日期记录的实施历史。历史章节中的“未完成”描述只表示当时状态；当前状态以本节和 `FLUX3_IMAGE_KLEIN_TT_ADAPTATION_COMPLETE.md` 最后章节为准。

| 工作项 | 当前状态 |
|---|---|
| Base `/local/models/flux-3-image-klein-base` | 单芯片 T2I 已实现并完成语义验证 |
| Distilled `/local/models/flux-3-image-klein` | 单芯片 T2I 已实现并完成语义验证 |
| prompt/steps/seed/width/height | API 已支持 |
| Base CFG/negative prompt | 已支持 |
| Distilled flow-map/`ref_time_in` | 已支持 |
| `response_format=file` | 已支持 PNG/JPEG 文件响应 |
| 图片异步、轮询、cancel | 已实现 |
| 32 个单芯片 worker | 已启动并完成 200 请求测试 |
| prompt encoder / Transformer / VAE 常驻 | 已实现；默认每 worker 加载一次 |
| scheduler 缓存 | 已实现；按 `num_inference_steps` 复用 |
| latent / CFG / Euler 设备端执行 | 已实现；denoise loop 不再逐步 D2H/H2D |
| 内存不足回退 | 已实现；`FLUX3_COMPONENT_RESIDENCY=per_request` |
| 2/4 芯片单请求并行 | 按当前范围暂缓 |
| Golden Data 逐层数值对齐 | 按当前范围暂缓 |
| 32 worker 首次共同创建 cache | 未解决；应预生成 cache 或增加跨进程锁 |

2026-11-05 最新性能基线：Distilled、1024×1024、16 步、32 worker、组件常驻和设备端 Euler，200 个同时提交请求 200/200 成功，总耗时 77.820 秒，吞吐 2.5700 images/s，平均响应 40.971 秒，P50 44.782 秒，P95 67.038 秒。优化前同口径为 132.499 秒和 1.5094 images/s。

## 1. 目标和现状

### 1.1 目标

在 10.12.11.2 上实现 FLUX.3 Image Klein 的：

1. 文生图（T2I）。
2. 图生图/参考图生成（I2I，先支持 1 张，再扩展到多张）。
3. 通过 tt-media-server 的图像服务提供 HTTP API。
4. 先只使用 Blackhole Galaxy 中的 1 个芯片完成 bring-up、精度和性能测试。

整机虽然有 32 个芯片，但当前阶段不做 32 芯片并行。32 个芯片只是可用资源，首版目标是单芯片可运行、可调试、可复现。

### 1.2 机器信息

10.12.11.2 已确认是 Blackhole Galaxy：

- 主机名：my01-galaxy02。
- PCI ID：1e52:b140。
- b140 在 tt-kmd 中对应 Blackhole。
- 设备节点为 /dev/tenstorrent/0 到 /dev/tenstorrent/31，共 32 个芯片。

tt-smi 当前如果出现设备映射错误，应单独排查 KMD、UMD 和设备状态；不能据此否定 PCI ID 对硬件型号的判断。

### 1.3 初始 TL-inference-server 基线（2026-11-02）

当时的 flux-3-image-klein 分支已经具备：

- tt-media-server 的服务、调度器和 worker 生命周期。
- /v1/images/generations 图像生成 API。
- /v1/images/image-to-image 图生图 API。
- /v1/images/edits 编辑 API。
- 通用 TTDiTRunner。
- TTFlux1Runner、Qwen-Image、SD3.5、MiniMax-H3 等 runner。
- 通过 models.tt_dit.pipelines.* 调用 TT-Metal/TT-NN 侧 pipeline 的代码边界。

以下内容是制定计划时的历史基线；当前 FLUX.3 T2I 已实现，最终状态见第 0 节：

- config/constants.py 中没有 FLUX.3 的模型、runner 和服务映射。
- runner_fabric.py 中没有 FLUX.3 runner 注册。
- tt_model_runners 中没有 TTFlux3Runner。
- 当时仓库没有 models.tt_dit.pipelines.flux3。
- 当时请求模型只支持单个 image 字段，不能表达多张参考图片。
- 当时 Blackhole Galaxy 配置中没有经过 FLUX.3 验证的 mesh、L1、trace 和超时参数。

### 1.4 初始 tt-metal 代码基线（2026-11-02）

当时检查了 11.2 上的 `/local/lnn/tt-metal`；分支为 `flux-3-image-klein`，阶段提交为 `23ef0fd19b`，刚加入 FLUX.3 Transformer、Qwen3-VL 和 VAE 的权重映射代码。

当前可以确认的相关实现是：

~~~text
models/tt_dit/pipelines/flux2/pipeline_flux2.py
models/tt_dit/pipelines/flux2/prompt_encoder.py
models/tt_dit/models/transformers/transformer_flux2.py
models/tt_dit/models/vae/vae_flux2.py
models/tt_dit/tests/models/flux2/
~~~

FLUX.2 已经包含 transformer、prompt encoder、VAE、pipeline、精度测试和性能测试；测试中出现了 2×2、1×8 和 4×8 等 mesh 配置。

仓库中没有发现以下 FLUX.3 文件或符号：

~~~text
models/tt_dit/pipelines/flux3/
Flux3Pipeline
Flux3Transformer
models/tt_dit/models/transformers/transformer_flux3.py
~~~

另外，`models/experimental/moss_image` 是独立的 MOSS-Image 实验实现：它支持单芯片和双芯片的部分流程，并复用了 FLUX.2 VAE，但它不是 FLUX.3，也不提供当前 tt-media-server 的服务 API。因此它只能参考工程化和验证方法，不能作为 FLUX.3 已完成的实现。

### 1.5 当前 FLUX.3 本地模型资产

11.2 上已经存在两套本地模型目录：

~~~text
/local/models/flux-3-image-klein-base
/local/models/flux-3-image-klein
~~~

已检查到该目录约 41 GB，包含：

~~~text
ae.safetensors
model.safetensors
config.json
manifest.json
transformer/
text_encoder/
vae/
variants/fp8r/
variants/nvfp4/
~~~

当前 `config.json` 明确给出了 FLUX.3 Image Klein 的模型契约：

- 模型名：`flux-3-image-klein`。
- 默认 dtype：BF16。
- Transformer hidden size：3072。
- attention heads：24，head dim：128。
- 双流层数：5，single stream 层数：28。
- image/conditioning channels：128。
- text conditioning dimension：20480。
- 默认分辨率：1024×1024。
- 默认 steps：50。
- 默认 guidance：3.5。
- 文本/视觉编码器：Qwen3-VL，36 层文本配置，最大 token 数 8192。
- VAE 权重位于 `ae.safetensors`，配置声明为 float32。

`manifest.json` 表明 `model.safetensors` 是 `f3_dit_export`，包含 357 个 tensor、约 5.538B 参数，默认导出 dtype 为 BF16，并带有 SHA-256 校验值。目录还提供 FP8R 和 NVFP4 变体，但首版应先使用 BF16 完成精度对齐，再单独验证量化版本。

因此，计划已经从“等待模型权重”进入“本地权重接入和验证”阶段。下一步必须把这些文件与 vLLM-Omni 的 Qwen3-VL、FLUX.3 Transformer、conditioning 和 VAE 输入输出逐项对应，不能只按 FLUX.2 的权重名称直接加载。

### 1.6 单芯片运行约束

FLUX.3 首版使用单芯片配置：

~~~text
device_mesh_shape = (1, 1)
device_ids        = (0)
max_batch_size    = 1
~~~

仓库的 `DeviceIds.DEVICE_IDS_1` 已定义为 `(0)`，因此不应使用 `DEVICE_IDS_32` 或 4×8 mesh。为了切换到其他物理芯片，可以通过设备可见性或等价的设备选择配置，把目标芯片映射为逻辑设备 0；模型代码不应写死物理芯片编号。

单芯片模式下需要确认：

- FLUX.3 Klein 的完整权重、encoder、transformer 和 VAE 是否能放入一个 Blackhole 芯片的可用内存。
- 是否需要 BF16、FP8 或其他量化才能完成加载。
- 是否有算子隐式创建多芯片 mesh。
- pipeline 是否把 `mesh_device` 当成单设备使用。
- 当前 `DeviceTypes.BLACKHOLE_GALAXY` 的配置语义是否适合单芯片；如果不适合，应增加明确的 Blackhole 单芯片设备配置，而不是复用 32 芯片配置。

需要特别注意：当前 tt-metal 的 FLUX.2 测试中，`single_device_params_flux2` 主要用于 VAE 单设备测试，完整 FLUX.2 pipeline 测试并没有证明 `(1,1)` 单芯片完整生成已经可用。因此，FLUX.3 单芯片完整推理仍然是待验证目标，而不是现成能力。如果单芯片内存不足，第一步仍可使用单芯片完成算子、encoder、VAE 和小规模精度调试；完整模型再扩展到 2 芯片或 4 芯片。

### 1.7 面向 1/2/4 芯片的可扩展设计

单芯片配置只是首个可运行目标，最终实现需要兼容以下资源规模：

| 运行规模 | 目标 | 关键要求 |
|---|---|---|
| 1 芯片 | 算子、权重加载、精度 bring-up | `(1,1)` mesh，单芯片本地执行 |
| 2 芯片 | 扩大可用内存或降低单芯片压力 | 2 芯片 mesh、权重分片和跨芯片通信 |
| 4 芯片 | 性能和容量扩展 | 4 芯片 mesh、稳定的并行布局和 collective |

不要在 pipeline 或 runner 中写死 `(1,1)`、`(0)` 或 32 芯片设备列表。推荐由配置层提供：

- `device_ids`：实际参与本次请求的物理设备。
- `device_mesh_shape`：mesh 的逻辑形状。
- `tensor_parallel_degree`：Transformer/权重的分片程度。
- `sequence_parallel_degree`：序列维度是否分片。
- `dtype` 和量化策略。
- `max_batch_size` 和请求超时。

2 芯片和 4 芯片的具体 mesh 形状不能仅凭芯片数量猜定。例如 2 芯片可能采用 `(1,2)`，4 芯片可能采用 `(2,2)`，但最终必须由 FLUX.3 的权重分片、通信拓扑和 TT-Metal pipeline 实测确认。配置应做到只切换参数，不复制三套模型代码。

建议 pipeline 的创建接口保持统一：

~~~python
Flux3Pipeline.create_pipeline(
    checkpoint_name=...,
    mesh_device=...,              # 1/2/4 芯片均由配置创建
    tensor_parallel_degree=...,
    sequence_parallel_degree=...,
)
~~~

验证顺序应为 1 芯片 → 2 芯片 → 4 芯片。每扩展一个规模，都要重新记录权重分片、峰值内存、跨芯片通信、精度误差和端到端性能。

因此必须分成两层实现：

~~~text
TT-Metal/TT-NN 模型实现
    models/tt_dit/pipelines/flux3/...
              ↑
tt-media-server 适配层
    TTFlux3Runner + constants/settings/API/request/tests
~~~

## 2. 参考实现和代码边界

### 2.0 初始两个仓库的兼容性审计（历史记录）

当时结论是：**现有 FLUX.1/H3 服务适配接口基本兼容，构建链路尚未绑定到本地 tt-metal，FLUX.3 尚未兼容。** 当前这些缺口已经通过本地 BuildKit context、FLUX.3 pipeline 和 server runner 补齐。

静态代码核对结果：

| 检查项 | 结果 | 说明 |
|---|---|---|
| `TTFlux1Runner` 导入 FLUX.1 pipeline | 基本兼容 | TL runner 调用 `Flux1Pipeline.create_pipeline(mesh_device=..., checkpoint_name=...)`，当前 tt-metal 的签名包含这两个参数 |
| MiniMax-H3 runner 导入 | 基本兼容 | 当前 tt-metal 有 `models/tt_dit/pipelines/minimax_h3`，TL runner 也按该包路径导入 |
| `ttnn.MeshDevice` | 接口方向兼容 | TL 的 `BaseMetalDeviceRunner` 负责创建 mesh，pipeline 接收 `mesh_device` |
| FLUX.2 | 已有底层实现，未接入当前 TL runner | tt-metal 有 FLUX.2 pipeline 和测试，但 TL 没有 FLUX.2 runner 注册 |
| FLUX.3 | 不兼容/尚未实现 | 两个仓库当前都没有 FLUX.3 pipeline、runner 或模型注册 |
| Docker 使用本地 `/local/lnn/tt-metal` | 当前不兼容 | Dockerfile 会重新 clone 官方 tt-metal，不会自动使用本地 fork |

当前 TL 仓库的 Dockerfile 在构建阶段会重新 clone：

~~~text
https://github.com/tenstorrent-metal/tt-metal.git
~~~

因此，即使 `/local/lnn/tt-metal` 已经切换到 `flux-3-image-klein`，直接构建 `TL-inference-server/tt-media-server/Dockerfile` 仍可能使用另一个官方仓库版本。对于 FLUX.3，必须把 TL 和 tt-metal 固定到明确的仓库、分支和 commit。

当前已经把 Dockerfile 改为使用 BuildKit 的本地命名构建上下文。构建时使用：

~~~text
--build-context tt-metal-local=/local/lnn/tt-metal
~~~

推荐从 TL-inference-server 仓库根目录构建：

~~~bash
cd /local/lnn/TL-inference-server

docker buildx build \
  --load \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=3c8681218b0951637991d20ce6b7e341df3b6879 \
  -f tt-media-server/Dockerfile \
  -t tl-inference-server:flux-3-image-klein-local \
  .
~~~

Dockerfile 会通过 `COPY --from=tt-metal-local` 复制本地 tt-metal 源码，并在镜像内编译；不再执行远程 `git clone`。`TT_METAL_COMMIT_SHA_OR_TAG` 是镜像中的来源标记，构建前应替换为 `/local/lnn/tt-metal` 当前实际 commit。

构建日志还应打印镜像内记录的 tt-metal commit，确认服务和底层模型实现使用同一份代码。

### 2.1 vLLM-Omni FLUX.3 参考实现

已经检查过的 vLLM-Omni flux3-image-release 分支包含：

~~~text
vllm_omni/diffusion/models/flux3/
├── pipeline_flux3.py
├── flux3_transformer.py
├── flux3_text_encoder.py
├── flux3_conditioning.py
├── flux3_denoise.py
├── flux3_image_vae.py
├── flux3_kernel_adapter.py
├── flux3_prims_ts.py
├── flux3_positional.py
└── lora.py
~~~

该实现可作为以下内容的参考：

- FLUX.3 的模块组成和 forward 顺序。
- Qwen3-VL 文本/视觉编码器的输入输出约定。
- prompt、参考图片和 conditioning 的数据结构。
- Transformer 的 hidden state、image tokens、text tokens、timestep 和 positional embedding 形状。
- denoise 调度、guidance、seed 和 timestep 处理。
- image VAE 的输入输出布局、缩放和 dtype。
- 权重名称、权重格式和参数映射。
- T2I、I2I 以及多参考图请求的行为定义。

不能直接把 vLLM-Omni 的 CUDA/PyTorch 模块复制到 TT 上：

- vLLM-Omni 面向 CUDA/PyTorch 及其 kernel adapter。
- TT 侧需要使用 ttnn.Tensor、TT-NN 运算和 TT-Metal kernel。
- 设备 mesh、张量布局、内存分配、跨芯片通信、trace 和 program cache 都不同。
- vLLM 的 API 和 TT-media-server 的 worker 模型也不同。

推荐原则：

> vLLM-Omni 作为数学行为、张量契约和参考结果的 oracle；TT-Metal 负责真正的设备实现；TL-inference-server 只负责服务适配和请求调度。

### 2.2 当前仓库中的 FLUX.1 可复用部分

当前代码中的 TTFlux1Runner 位于：

~~~text
tt-media-server/tt_model_runners/dit_runners.py
~~~

其核心调用是：

~~~python
Flux1Pipeline.create_pipeline(
    checkpoint_name=self.settings.model_weights_path,
    mesh_device=self.ttnn_device,
)
~~~

FLUX.3 可以复用 TTDiTRunner 提供的：

- worker 初始化。
- pipeline 创建和权重分发。
- warmup。
- 设备生命周期。
- 图像请求执行。
- 图像结果后处理。
- telemetry 和错误处理。

但不能假定 FLUX.3 的 pipeline 可以直接复用 FLUX.1 的模型结构。FLUX.3 至少需要独立的 transformer、文本/视觉编码、conditioning、VAE 和权重映射。

## 3. 建议的代码结构

### 3.1 TT-Metal/TT-NN 层

建议新增：

~~~text
models/tt_dit/pipelines/flux3/
├── __init__.py
├── pipeline_flux3.py
├── flux3_transformer.py
├── flux3_text_encoder.py
├── flux3_conditioning.py
├── flux3_denoise.py
├── flux3_image_vae.py
├── flux3_positional.py
├── flux3_weights.py
├── flux3_config.py
└── tests/
~~~

这些文件属于 TT-Metal/TT-NN 模型实现层。TL-inference-server 通过 models.tt_dit 导入 pipeline，因此只修改 server 仓库不能完成 FLUX.3。

### 3.2 TL-inference-server 层

建议新增或修改：

~~~text
tt-media-server/
├── config/constants.py
├── config/settings.py
├── domain/flux3_image_request.py
├── open_ai_api/image.py
├── model_services/image_service.py
├── tt_model_runners/flux3_runner.py
├── tt_model_runners/runner_fabric.py
└── tests/
~~~

建议 FLUX.3 使用独立的 flux3_runner.py，而不是继续把所有模型堆进 dit_runners.py。这样更容易维护多参考图预处理、Qwen3-VL 输入拼接、conditioning、分辨率 bucket、latent shape 和后续 LoRA/量化开关。

## 4. 具体实现步骤

### 当前执行顺序调整

由于 11.2 当前没有可直接使用的 PyTorch/vLLM 参考环境，阶段 1 的 GPU/vLLM Golden Data 暂时后置，不作为当前开发阻塞项。当前先执行阶段 2，在本地 `/local/models/flux-3-image-klein-base` 权重和 `tt-metal` 环境上完成最小 TT T2I pipeline。

当前第一版范围固定为：

- 单个 Blackhole 芯片优先。
- BF16。
- 客户端可指定宽高；当前要求宽高均为 16 的倍数，1024×1024 是默认值。
- batch=1。
- 固定 seed。
- 固定 inference steps。
- 先实现文本生成图像（T2I）。
- 暂不实现 API、I2I、多参考图、LoRA、FP8R、NVFP4 和动态 batch。

截至 2026-11-02，阶段 2 已开始，已在 `/local/lnn/tt-metal` 的
`flux-3-image-klein` 分支加入第一批可验证代码：

~~~text
models/tt_dit/pipelines/flux3/__init__.py
models/tt_dit/pipelines/flux3/flux3_config.py
models/tt_dit/pipelines/flux3/t2i_contract.py
models/tt_dit/pipelines/flux3/pipeline_flux3.py
models/tt_dit/tests/flux3/test_config.py
~~

这批代码已经完成并在 10.12.11.2 上验证：

1. 读取 `/local/models/flux-3-image-klein-base/config.json`。
2. 校验 `manifest.json` 的模型类型为 `flux-3-image-klein`、导出类型为 `f3_dit_export`。
3. 读取 BF16 `model.safetensors` 的 header，不依赖 PyTorch 或 safetensors Python 包。
4. 校验本地导出包含 357 个 tensor，以及 FLUX.3 T2I 所需的关键权重名。
5. 固定 T2I 请求契约：单 prompt、batch=1、固定 seed/steps/guidance；宽高由客户端提供并按 latent patch 约束校验。
6. 明确了下一步的两个执行入口：Qwen3-VL prompt encoder 和 `Flux3Transformer`。

随后已加入第一版 TT-NN DiT 结构文件：

~~~text
models/tt_dit/models/transformers/transformer_flux3.py
~~~

当前实现包含：

- 5 个 FLUX.3 per-mode image/text block。
- 28 个 joint single block。
- image/text 输入投影。
- early/single shift-scale-gate modulation。
- 20480 → 3072 的 text projection。
- 3072 → 128 的 image output projection。
- 单芯片优先的 TT-NN attention、SwiGLU、动态 LayerNorm 和 mesh 参数接口。

该文件已经在 Docker/Blackhole 单芯片环境中加载真实权重并完成全量映射：357 个源 tensor、15 个实际 mode block（txt/image/image_cond 各 5 个）和 28 个 single block。

这不是“已经完成端到端出图”：当前 `Flux3T2IPipeline.run()` 会明确报告
TT transformer 和 Qwen3-VL encoder 尚未接入。这样可以先把权重格式、模型结构和
请求边界锁定，再接入设备计算，避免把未验证的 FLUX.2 代码误当成 FLUX.3。

阶段 1 后置后，仍然必须在后续精度阶段补回；它用于最终确认 TT 结果与 PyTorch/vLLM 的中间张量和最终图像差异。

### 阶段 0：建立基线和依赖检查

~~~bash
cd /local/lnn/TL-inference-server
git switch flux-3-image-klein
git status
git log -1 --oneline
~~~

需要确认：

1. 10.12.11.2 上实际使用的 tt-metal 源码路径和 commit。
2. models.tt_dit 对应的 Python 包来自哪个 checkout 或容器层。
3. 当前 tt-metal 是否包含 FLUX.3 所需的基础算子。
4. Blackhole UMD、KMD 和 ttnn 是否能正常枚举 32 个设备。
5. 模型权重目录 `/local/models/flux-3-image-klein-base` 是否可读，并校验 manifest 和权重 hash。
6. Qwen3-VL encoder 是在 TT 上运行，还是暂时由 CPU/PyTorch 运行。

交付物应包含：

- tt-metal commit。
- tt-inference-server commit。
- Python、PyTorch、ttnn 版本。
- 设备型号和 mesh。
- FLUX.3 权重版本、本地路径 `/local/models/flux-3-image-klein-base`、manifest 和 SHA-256。
- 一次参考推理的 prompt、seed、分辨率、步数和输出 hash。

### 阶段 1：建立 PyTorch/vLLM 参考基线（后置）

该阶段暂时后置。环境具备后，使用 vLLM-Omni 或官方参考实现固定：

~~~text
prompt
seed
width / height
num_inference_steps
guidance_scale
timesteps / sigmas
参考图片数量和内容
dtype
~~~

至少导出：

1. prompt token ids。
2. Qwen3-VL 文本/视觉 encoder 输出。
3. image conditioning。
4. 初始 latent/noise。
5. 每个 denoise step 前后的 latent。
6. Transformer 关键 block 的输入输出。
7. VAE 输入和最终图像。

这些结果组成 Golden Data，用来区分误差来自 tokenizer/processor、encoder、latent 初始化、position embedding、transformer、scheduler、VAE 还是后处理。

### 阶段 2：实现最小 T2I pipeline（已完成）

第一版只支持：

- 单 prompt。
- 单张输出。
- 客户端可指定分辨率；宽高必须是 `vae_scale_factor * patch_size` 的整数倍，当前为 16 的倍数。
- 固定 dtype。
- 固定 steps。
- 固定 seed。
- 不支持 LoRA、多参考图和动态 batch。

建议优先参考并逐步复用 FLUX.2 已有的设计，而不是从空白重新设计所有基础设施：

1. 参考 `models/tt_dit/pipelines/flux2/pipeline_flux2.py` 的 pipeline 生命周期、scheduler、trace state 和 CCL 管理。
2. 参考 `models/tt_dit/pipelines/flux2/prompt_encoder.py` 的 prompt encoder 接入和并行方式。
3. 参考 `models/tt_dit/models/transformers/transformer_flux2.py` 的 DiT block、TP/SP、attention 和通信实现。
4. 参考 `models/tt_dit/models/vae/vae_flux2.py` 的 VAE decode 和 layout 处理。
5. 复用 `models/tt_dit/tests/models/flux2/` 的测试组织方式，但替换为 FLUX.3 的模型结构、权重和 Golden Data。

然后按以下顺序实现 FLUX.3：

1. latent 初始化和 layout 转换。
2. positional embedding。
3. Qwen3-VL encoder 或 encoder 输出的设备输入接口。
4. FLUX.3 transformer 单 block。
5. 完整 transformer denoise loop。
6. scheduler/timestep。
7. VAE decode。
8. pipeline 统一封装。

阶段 2 的当前剩余工作不是 Transformer 权重映射，而是完整 pipeline 串联：

- 已完成 FLUX.3 的 per-mode block（每个 block 使用 3 路 shift/scale/gate）到 TT-NN 的映射。
- 已完成 28 个 joint single block 映射，并保持 text/image token 的调制顺序。
- 已完成 `model.safetensors` 的导出命名到 TT-Metal module state 的映射。
- 已完成 Qwen3-VL 36 层权重映射和 FLUX3 VAE 251 个源 tensor 映射。
- 先允许外部传入 20480 维 `prompt_embeds` 完成 DiT/VAE bring-up，再补 Qwen3-VL 的 TT encoder。
- 在当前无 PyTorch/safetensors 主机环境下，使用 Docker 构建环境完成真实权重加载和设备执行。

pipeline 至少提供类似入口：

~~~python
Flux3Pipeline.create_pipeline(
    checkpoint_name=..., mesh_device=...
)

pipeline.run_single_prompt(
    prompt=...,
    num_inference_steps=...,
    seed=...,
    width=...,
    height=...,
)
~~~

### 2026-11-03 映射完成边界

已完成并实测：

| 组件 | 结果 |
|---|---|
| FLUX.3 DiT | 357/357 源 tensor；15 个 mode block；28 个 single block |
| Qwen3-VL 文本编码器 | 36 层，8 个配置激活层 |
| FLUX3 VAE decoder | 251 个 safetensors 源 tensor |
| Docker 来源 | BuildKit `--build-context tt-metal-local=/local/lnn/tt-metal`，模型权重不进入镜像 |

以上表示完整 T2I pipeline 所需模型组件的权重映射已完成，不表示 prompt → denoise → VAE → PNG 的端到端出图已经完成。剩余顺序为：文本编码执行、latent/patch/RoPE、scheduler、denoise loop、VAE decode、PNG 输出和 tt-media-server API。

每个张量都要明确记录：

- shape。
- dtype。
- layout。
- device mesh。
- replicated、sharded 还是 interleaved。
- 是否发生 host/device 拷贝。

### 阶段 3：加入 TT-media-server runner

新增 TTFlux3Runner，职责只包括服务适配，不在 runner 中重新实现 transformer。

~~~python
class TTFlux3Runner(TTDiTRunner):
    def create_pipeline(self):
        return Flux3Pipeline.create_pipeline(
            checkpoint_name=self.settings.model_weights_path,
            mesh_device=self.ttnn_device,
        )

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": ...,
            "trace_region_size": self.settings.trace_region_size,
        }

    def run(self, requests):
        request = requests[0]
        return self.pipeline.run_single_prompt(...)
~~~

run() 中明确传递：

- prompt。
- seed。
- inference steps。
- guidance scale。
- width/height。
- 参考图片或 image conditioning。
- 输出格式。

不要把 vLLM 的 torch.Generator、CUDA stream 或 CUDA tensor 带入 runner。

### 阶段 4：注册 model、runner 和设备配置

需要修改 config/constants.py：

1. SupportedModels 增加 FLUX.3 权重标识。
2. ModelNames 增加稳定服务名，例如 FLUX.3-image-klein。
3. ModelRunners 增加唯一值，例如 tt-flux3-image-klein。
4. 将 runner 加入 MODEL_SERVICE_RUNNER_MAP 的 image 服务集合。
5. 增加 runner 到模型名的映射。
6. 增加 Blackhole Galaxy 的设备配置。

runner_fabric.py 增加类似：

~~~python
ModelRunners.TT_FLUX_3_IMAGE_KLEIN: lambda wid: __import__(
    "tt_model_runners.flux3_runner",
    fromlist=["TTFlux3Runner"],
).TTFlux3Runner(wid),
~~~

Blackhole 配置至少要包含单芯片、双芯片和四芯片三套可切换配置：

- `DeviceIds.DEVICE_IDS_1`，即 `(0)`。
- `device_mesh_shape=(1, 1)`。
- `is_galaxy`、设备类型和 worker 初始化语义的实测结果。
- max_batch_size。
- 请求超时。
- trace_region_size。
- 必要的 Fabric 配置。

当前 FLUX.2 的测试已经验证了部分 2×2、1×8 和 4×8 场景；这些结果可以作为通信、Fabric、TP/SP 和 VAE 分片的参考，但不能直接证明 FLUX.3 使用相同 mesh。FLUX.3 仍必须重新验证单芯片、2 芯片和 4 芯片的内存、权重布局、通信和 kernel 行为。

### 阶段 5：设计 FLUX.3 请求模型

当前 ImageGenerateRequest 已有：

- prompt。
- negative_prompt。
- num_inference_steps。
- seed。
- number_of_images。
- width/height。
- guidance_scale。
- timesteps/sigmas。

当前 ImageToImageRequest 只有一个 image 字段，而 FLUX.3 参考实现支持多张参考图片。建议新增 Flux3ImageRequest：

~~~python
class Flux3ImageRequest(ImageGenerateRequest):
    reference_images: list[str] = []
    mode: Literal["t2i", "i2i"] = "t2i"
    strength: float = 0.6
~~~

建议限制：

- reference_images 最多 4 张。
- API 层校验大小、格式、颜色空间和 base64。
- 统一转换为 RGB 或模型要求的格式。
- 统一 resize/crop/pad 行为。
- 明确参考图片顺序，因为顺序会影响 conditioning。

第一版可以使用 JSON + base64，避免立即修改 multipart C++ gateway；后续图片较大时再增加 multipart。两种入口必须分别写清楚字段和限制。

### 阶段 6：API 接入

优先复用：

~~~text
POST /v1/images/generations
~~~

它适合 T2I。带参考图的 FLUX.3 有两种选择：

方案 A：扩展 /v1/images/image-to-image，增加 reference_images。优点是复用现有语义；缺点是通用 request model 会变复杂。

方案 B：新增 /v1/images/flux3/generations。优点是字段清晰、不影响既有模型；缺点是要同步 Python/C++ gateway/OpenAPI。

建议先用方案 A 完成内部 bring-up，稳定后再决定是否提供专用 endpoint。

返回格式继续复用当前图像服务：

~~~json
{
  "images": ["<base64>"],
  "generation_time": 12.34
}
~~~

中间 tensor 不应默认放进 HTTP 返回，应使用 debug_dump_dir、request id 子目录、显式 debug 开关和 manifest。

### 阶段 7：权重和缓存

区分：

1. Hugging Face 原始权重。
2. TT 转换权重。
3. 预分片/预布局权重。
4. TT-Metal kernel cache。
5. tokenizer/processor cache。
6. VAE 和 encoder cache。

建议目录：

~~~text
/local/models/flux-3-image-klein-base
/local/cache/flux3-hf
/local/cache/flux3-tt-metal
/local/output/flux3
~~~

不要把 vLLM 转换权重和 TT-Metal 分片权重混在同一目录。每次转换记录源权重 hash、转换脚本版本、dtype、mesh、shard 规则和输出文件 hash。

## 5. 精度对齐方案

### 5.1 对齐顺序

~~~text
tokenizer / processor
    ↓
text/vision encoder
    ↓
conditioning
    ↓
initial latent
    ↓
positional embedding
    ↓
单个 transformer block
    ↓
完整 denoise step
    ↓
完整 denoise loop
    ↓
VAE decode
    ↓
最终图像
~~~

### 5.2 指标

每一级至少记录：

- max_abs_error。
- mean_abs_error。
- RMSE。
- cosine similarity。
- 相对误差。
- NaN/Inf 数量。
- 输出 shape、dtype、layout。

最终图像同时记录：

- PNG 无损输出。
- RGB 像素 MAE/RMSE。
- SSIM 或 LPIPS。
- 固定 seed 的可重复性。

BF16、FP8、NVFP4 或其他量化路径必须分别建立 Golden Data，不能直接共用 BF16 阈值。

### 5.3 精度分级

1. 结构正确：shape、dtype、输入输出链路正确，无 NaN/Inf。
2. 数值对齐：关键中间结果达到预设误差阈值。
3. 视觉对齐：相同 prompt、seed、steps 下语义和构图稳定。

## 6. Blackhole Galaxy 性能优化

### 6.1 必须拆分的耗时

- server 启动。
- 设备初始化。
- 权重读取和分发。
- 首次 kernel 编译。
- pipeline warmup。
- 文本/视觉 encoder。
- transformer denoise loop。
- VAE decode。
- host/device 拷贝。
- 图像编码和 HTTP 返回。

### 6.2 优先优化方向

1. 内存和 layout：已完成初始 latent 单次 H2D、设备端 CFG/Euler 和 denoise latent 常驻；后续继续减少 token/layout 转换并评估设备端 FP32 accumulator。
2. 跨芯片通信：单芯片阶段没有跨芯片通信；扩展到 2/4 芯片时，验证 tensor parallel、sequence parallel 和 collective。
3. kernel 编译：固定 shape、dtype 和 steps，减少动态编译。
4. trace：shape 稳定后启用，bring-up 阶段先关闭。
5. encoder：已按 worker 常驻 TT；后续单独统计 encoder 时间并评估 trace 和 layout。
6. VAE：已按 worker 常驻 TT；后续单独统计 decode 时间和峰值内存。
7. 请求并发：第一版保持 max_batch_size=1，单请求稳定后再考虑批处理。
8. 缓存：cache key 必须包含 dtype、shape、mesh 和模型版本。

### 6.3 当前性能记录

| 日期 | 变体 | worker/并发 | 分辨率 | steps | 结果 |
|---|---|---|---:|---:|---|
| 2026-11-04 | Base | 1/1 | 1024×1024 | 50 | 68.37 s，语义正确 |
| 2026-11-04 | Distilled | 1/1 | 1024×1024 | 16 | 17.45 s，语义正确 |
| 2026-11-05 | Distilled 预热 | 32/32 | 1024×1024 | 16 | 32/32 成功，22.587 s，1.4168 images/s |
| 2026-11-05 | Distilled 压测 | 32/200 | 1024×1024 | 16 | 200/200 成功，132.499 s，1.5094 images/s，P95 121.740 s |
| 2026-11-05 | Distilled 常驻+设备端 Euler | 1/1 | 1024×1024 | 16 | 4 次后 3 次平均 11.541 s，语义正确 |
| 2026-11-05 | Base 常驻+设备端 CFG/Euler | 1/1 | 1024×1024 | 50 | 62.395 s，CFG 3.5，语义正确 |
| 2026-11-05 | Distilled 常驻+设备端 Euler | 32/32 | 1024×1024 | 16 | 32/32 成功，12.123 s，2.6397 images/s |
| 2026-11-05 | Distilled 常驻+设备端 Euler | 32/200 同时提交 | 1024×1024 | 16 | 200/200 成功，77.820 s，2.5700 images/s，P95 67.038 s |

组件常驻、schedule 缓存、设备端 CFG/Euler 和 denoise latent 常驻已经完成。尚未拆分并记录 encoder、Transformer 每步、VAE、最终 D2H 和 PNG/JPEG 编码的独立耗时；这些仍是下一轮 profiling 工作。

## 7. 测试计划

### 7.1 单元测试

- runner 枚举和服务映射。
- Flux3ImageRequest 字段校验。
- 参考图片数量上限。
- base64、尺寸、颜色空间和空输入校验。
- seed 和 steps 传递。
- runner factory 创建 TTFlux3Runner。
- Blackhole 配置解析。
- pipeline mock 下 runner 参数正确。

### 7.2 pipeline 测试

- tokenizer/processor 输出 shape。
- latent 初始化。
- 单 block forward。
- 单 denoise step。
- 固定 steps 推理。
- VAE encode/decode。
- 1 张和 4 张参考图。
- 不同分辨率 bucket。
- BF16 和量化路径分别测试。

### 7.3 服务端测试

~~~bash
curl http://127.0.0.1:8000/tt-liveness

curl -X POST http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -d '{
    "prompt": "a red bird on a branch, cinematic lighting",
    "seed": 42,
    "num_inference_steps": 20,
    "width": 1024,
    "height": 1024
  }'
~~~

参考图请求的字段必须以最终 TT request schema 为准，不能直接假定 vLLM 的字段会被当前 TT API 接受。API 变更要同步：

- Python request model。
- FastAPI route。
- C++ gateway/OpenAPI（如果请求经过 C++ gateway）。
- curl 示例。
- 测试用例。

## 8. 建议的提交拆分

1. Add FLUX.3 model and runner constants。
2. Add FLUX.3 request schema and validation。
3. Add TTFlux3Runner adapter。
4. Add Blackhole Galaxy FLUX.3 device configuration。
5. Add FLUX.3 TT pipeline。
6. Add FLUX.3 T2I API tests。
7. Add FLUX.3 reference image support。
8. Add FLUX.3 accuracy and performance benchmarks。
9. Document FLUX.3 deployment and API usage。

每个 commit 都应说明修改边界，避免把 kernel、API 和性能调优混成一个不可回滚的大提交。

## 9. 第一阶段实际优先级

### P0：确认模型实现边界

- 确认 models.tt_dit 的实际源码位置。
- 确认 tt-metal 当前 commit 是否有 FLUX.3 基础实现。
- 确认权重格式和 Qwen3-VL 依赖。
- 固定 vLLM/GPU Golden Data。

### P1：最小 T2I bring-up

- 固定 1024×1024。
- 固定 batch=1。
- 固定 BF16。
- 固定 seed 和 steps。
- 先完成最终图片生成。

### P2：逐级精度对齐

- encoder。
- latent。
- transformer block。
- denoise loop。
- VAE。

### P3：接入 server API

- runner。
- constants/settings。
- factory。
- request schema。
- /v1/images/generations。

### P4：扩展 I2I 和多参考图

- 单参考图。
- 多参考图。
- resize/crop 策略。
- reference 顺序和 conditioning 对齐。

### P5：性能优化和交付

- trace。
- layout 和通信。
- encoder/VAE 优化。
- 热态吞吐和并发。
- 部署文档和 benchmark 结果。

## 10. 完成标准

- [ ] 10.12.11.2 能稳定识别 Blackhole Galaxy，并能选择其中 1 个芯片运行。
- [ ] 单芯片 `(1,1)` mesh、`(0)` device_ids 和 batch=1 路径通过验证。
- [ ] 2 芯片配置可以通过参数切换，并完成精度和性能验证。
- [ ] 4 芯片配置可以通过参数切换，并完成精度和性能验证。
- [ ] 1/2/4 芯片共用同一套 runner 和 pipeline 代码，没有复制三套模型实现。
- [ ] 权重加载、warmup 和热态请求均成功。
- [ ] `/local/models/flux-3-image-klein-base` 的 manifest、文件结构和 SHA-256 已校验。
- [ ] T2I API 能生成有效图片。
- [ ] 固定 seed 能重复生成可接受结果。
- [ ] vLLM/参考实现和 TT 结果完成中间张量及最终图像对齐。
- [ ] 单参考图 I2I 可用。
- [ ] 多参考图行为、上限和顺序已明确。
- [ ] 关键阶段耗时、峰值内存和 kernel 编译时间已记录。
- [ ] Blackhole Galaxy 的 mesh、trace、dtype 和超时参数已实测确认。
- [ ] 单元测试、pipeline 测试和服务端 API 测试通过。
- [ ] 部署命令、权重目录、缓存目录和 curl 示例已写入文档。

## 11. 最终建议

真实开发应以 TL-inference-server 的 main 代码为服务基线，在 flux-3-image-klein 分支上实现；不能把 FLUX.2 分支或 vLLM-Omni 分支直接当成 TT 的可运行实现。

推荐工程组合：

~~~text
TL-inference-server/main
    + Blackhole 单芯片的设备/mesh 配置
    + TT-Metal/TT-NN 的 FLUX.3 pipeline
    + vLLM-Omni FLUX.3 的模型结构和输入契约
    + Golden Data 精度对齐
    + tt-media-server API/runner 适配
~~~

第一步不是立即改 API，而是确认 models.tt_dit 的实际源码、tt-metal commit 和 Docker 构建来源，然后在单个 Blackhole 芯片、固定输入下让 T2I pipeline 跑通。只有单芯片 pipeline 能独立产生正确结果，并且镜像内确认使用了该 tt-metal commit 后，再接入 tt-media-server；多芯片扩展另立阶段，问题定位最清晰。



## 附录 A：2026-11-03 实施状态修订

| 项目 | 状态 | 证据 |
|---|---|---|
| 357 个 FLUX.3 DiT 源 tensor 的 key/shape 校验 | 已完成 | manifest 与 safetensors header |
| 三组 5 个 mode block（txt/image/image_cond） | 已完成 | FULL_MAPPING_OK, mode_blocks=15 |
| 28 个 single block | 已完成 | FULL_MAPPING_OK, single_blocks=28 |
| Q/K/V、attn_out、SwiGLU FFN 权重映射 | 已完成 | 单芯片 TTNN 转换日志 |
| 完整 T2I 端到端图片生成 | 未完成 | 文本编码器、scheduler、VAE 尚未串联 |

下一步必须按“文本编码 → latent/patch/rope → denoise loop → VAE decode
→ PNG → API runner”的顺序推进；不能把 transformer 权重映射通过当成端到端
T2I 已完成。

## 附录 B：2026-11-03 缓存持久化和 v4 镜像任务

### 已完成

- 在两个本地仓库中接入 FLUX.3 DiT/VAE 的 TTNN 权重缓存；
- 将 Dockerfile 的缓存环境变量统一到 `/local/fluxcache`；
- 宿主机缓存根目录确定为 `/local/lnn/fluxcache`，通过 volume 挂载到容器；
- 镜像构建使用本地 `/local/lnn/tt-metal`，不从远端 clone tt-metal；
- 模型权重仍通过只读 volume 挂载，不复制到镜像。

### v4 构建与运行命令

```bash
cd /local/lnn/TL-inference-server
sudo docker buildx build --load \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=23ef0fd19b \
  -f tt-media-server/Dockerfile \
  -t tl-inference-server:flux-3-image-klein-local-20261103-v4 .
```

运行时至少需要挂载：

```bash
-v /local/lnn/fluxcache:/local/fluxcache \
-v /local/models/flux-3-image-klein-base:/local/models/flux-3-image-klein-base:ro \
-v /dev/tenstorrent:/dev/tenstorrent \
-v /dev/hugepages:/dev/hugepages \
-v /dev/hugepages-1G:/dev/hugepages-1G
```

### 缓存验收

1. 启动 v4，确认容器内 `TT_DIT_CACHE_DIR=/local/fluxcache/tt_dit_cache`、
   `TT_METAL_CACHE=/local/fluxcache/tt-metal-cache`。
2. 第一次执行 DiT、text encoder、VAE 映射，确认出现 `Cache does not exist` 和
   `Writing cache to`，并检查宿主机 `/local/lnn/fluxcache` 大小增加。
3. 停止并删除容器后，用相同命令再次启动。
4. 第二次执行相同配置，确认出现 `loading cache at`，不再重复权重转换。
5. 记录两次 `weight_load`、首次 kernel 编译和 warm run 时间。

### 仍未完成

缓存和组件映射完成后，完整 T2I 还剩：

1. prompt tokenizer/processor 和 Qwen3-VL 输出契约；
2. FLUX.3 latent 初始化、patchify/unpatchify、RoPE 和时间步编码；
3. scheduler 与完整 denoise loop；
4. DiT 三类 block 的真实前向执行和中间结果对齐；
5. TTNN VAE decode 串联、图片后处理和 PNG/JPEG 输出；
6. tt-media-server runner、同步/异步 API 及取消/错误处理；
7. vLLM-Omni 或官方参考实现的 Golden Data 精度对齐；
8. 2 芯片、4 芯片 mesh 和权重分片；
9. 性能优化、kernel cache/trace、并发和内存峰值测试。

## 附录 C：2026-11-03 v4 实际验证结论

v4 镜像已经用本地两个仓库构建并加载成功：

```text
tl-inference-server:flux-3-image-klein-local-20261103-v4
size: 8,298,064,005 bytes
tt-metal: 23ef0fd19b
```

验证时必须同时使用：

```bash
-e CACHE_ROOT=/local/fluxcache \
-v /local/lnn/fluxcache:/local/fluxcache
```

其中 `CACHE_ROOT` 让 entrypoint 处理挂载卷权限；仅设置 `TT_DIT_CACHE_DIR` 会导致
TT-Metal JIT 目录无法写入。宿主机 `/local/lnn/fluxcache/tt-metal-cache` 已修正为
`tt:tt`（UID/GID 1000:1000）。

缓存验证结果：

| 组件 | cold | warm | 结论 |
|---|---:|---:|---|
| FLUX.3 DiT | 17.5 s | 3.4 s | 权重 `1 HIT/0 MISS`，JIT `6/6 hits` |
| Qwen3-VL text encoder | 11.9 s | 2.4 s | 权重 `1 HIT/0 MISS`，JIT `6/6 hits` |
| FLUX.3 VAE | 7.0 s | 0.0 s | 权重 `1 HIT/0 MISS`，JIT `6/6 hits` |

三项组件结果分别为：DiT `357` 个源 tensor、`271` 个 TTNN 参数、`15` 个 mode
block、`28` 个 single block；text encoder `36` 层和 8 个激活层；VAE `251` 个源
tensor。由此可以确认 `/local/lnn/fluxcache` 已经真正承载转换缓存和 JIT 缓存。

当前状态已更新：完整 prompt/tokenizer → latent/patch/RoPE → scheduler/denoise
→ VAE decode → PNG → tt-media-server 同步 API 已完成，并已通过单请求和 64 并发
验证。图片异步提交、轮询和 cancel 也已接入共享 JobManager，并完成真实服务验证。
2/4 芯片扩展和精度对齐按当前任务要求暂缓；matmul/trace 专项性能优化、服务重启
恢复和独立图片下载接口仍未完成。

## 附录 D：2026-11-04 执行状态更新

### 图片文件响应 API：`response_format=file`

在现有图片生成 API 保持 JSON/Base64 兼容的前提下，新增文件响应路径：

- 请求模型：`tt-media-server/domain/image_generate_request.py`
  增加 `response_format: Literal["b64_json", "file"]`，默认仍为 `b64_json`。
- API 处理：`tt-media-server/open_ai_api/image.py`
  将 runner 返回的 Base64 图片解码为二进制；单张图片直接返回 PNG/JPEG，多张图片返回 ZIP。
- 覆盖接口：`/v1/images/generations`、`/v1/images/image-to-image`、`/v1/images/edits`。
- 测试：`tt-media-server/tests/test_kontext_api.py` 增加单图 PNG、默认 Base64、非法响应格式测试。

示例：

```bash
curl -fS http://127.0.0.1:8000/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"a red car","response_format":"file",'\\
'"image_return_format":"PNG","num_inference_steps":20}' \\
  -o output.png
```

该项同时涉及 API 参数传递和 FLUX.3 TTNN 请求契约：`tt-media-server` 已经把客户端的 `width`/`height` 传给 `Flux3T2IRequest`；本轮在 `tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py` 删除固定 1024×1024 限制，改为 16 倍数校验。

### 2026-11-04：50 步分辨率测试

在同一服务上使用 `seed=42`、`num_inference_steps=50` 和 `response_format=file` 顺序测试 256x256、512x512、1024x1024：

| 分辨率 | 结果 | 服务端时间 | 客户端总耗时 |
|---|---|---:|---:|
| 256x256 | 200；PNG，256x256，138,294 bytes | 37.8345 s | 42.4204 s |
| 512x512 | 200；PNG，512x512，529,403 bytes | 15.4046 s | 15.4158 s |
| 1024x1024 | 200；PNG，1024x1024，2,117,127 bytes | 35.5721 s | 35.6047 s |

当前客户端可以通过 `width` 和 `height` 指定输出长宽，但两者必须是 16 的倍数。256x256 首次请求包含动态 shape/kernel 编译；编译结果保存在 `/local/lnn/fluxcache`，后续应单独进行热态测试。

本次实现对应 `tt-metal` 分支 `flux-3-image-klein` 的 commit `4cccb4584f`。

### 2026-11-04：每种分辨率连续 4 次测试

为排除首次动态 kernel 编译影响，每种分辨率连续请求 4 次，丢弃第 1 次，取第 2～4 次客户端 `curl` 总耗时平均值：

| 分辨率 | 第1次 | 第2次 | 第3次 | 第4次 | 后3次平均 |
|---|---:|---:|---:|---:|---:|
| 256×256 | 11.2110 s | 11.1677 s | 11.1281 s | 11.2084 s | **11.1680 s** |
| 512×512 | 15.4146 s | 15.3210 s | 15.4079 s | 15.3293 s | **15.3528 s** |
| 1024×1024 | 38.4966 s | 38.4960 s | 38.5879 s | 38.5125 s | **38.5322 s** |

12 次请求全部 HTTP 200。热态结果符合分辨率增长趋势；此前 256×256 的 42.4204 秒包含首次动态 shape/kernel 编译，不能作为热态性能数据。

| 工作项 | 状态 | 证据/实现位置 |
|---|---|---|
| 单芯片完整 T2I | 已完成 | `pipeline_flux3.py`、`TTFlux3ImageKleinRunner`、HTTP 200 图片输出 |
| 32 worker 单芯片并行 | 已完成 | `(1,1)` mesh；2026-11-05 完成 200 请求，200/200 成功，worker 0～31 各处理 6～7 个 |
| 持久化 cache | 已完成 | Dockerfile 的 `FLUXCACHE_DIR`、宿主 `/local/lnn/fluxcache` |
| 图片异步提交 | 已完成 | `open_ai_api/image.py` 的 `/generations/async` |
| 图片轮询 | 已完成 | `/v1/images/generations/{job_id}` |
| 图片 cancel | 已完成 | `ImageService.cancel_image_job()` 复用 `JobManager.cancel_job()` |
| 50-step cancel 真实验证 | 已完成 | `queued -> cancelling -> cancelled` |
| 12-step 异步完成真实验证 | 已完成 | `status=completed`、`image_count=1` |
| 2/4 芯片并行 | 暂缓 | 当前只保留 `(1,1)` 单芯片实现 |
| 精度对齐 | 暂缓 | 不导出或比较 Golden Data |
| matmul/trace 专项优化 | 未完成 | 需单独 profiling 后实施 |
| 服务重启恢复 | 未完成 | 当前重启后重新 warmup，未恢复历史 image job |

### 本轮代码变更

1. `config/constants.py` 增加 `JobTypes.IMAGE`，并移除无关的 `SP_RUNNER` 设备 ID 改动。
2. `model_services/image_service.py` 增加图片任务创建、JSON 结果持久化、状态查询、取消和删除。
3. `open_ai_api/image.py` 增加图片 async submit、poll、cancel、delete API；同步 API 不变。
4. Dockerfile 保持本地 tt-metal BuildKit context 和 `/local/fluxcache` 运行时缓存，模型权重仍只读挂载，不复制进镜像。


## 附录 E：2026-11-03 T2I pipeline 组件映射验证结果

| 组件 | 状态 | 实际证据 |
|---|---|---|
| Flux3 DiT | 已完成 | 357/357 源张量，15 mode block，28 single block |
| Qwen3-VL 文本编码器 | 已完成 | 36 层及 8 个配置激活层在单芯片加载通过 |
| Flux3 VAE decoder | 已完成 | 251 个 safetensors 源张量加载通过 |
| 端到端 T2I 运行 | 未完成 | 尚未串联 prompt、scheduler、denoise 和输出 API |

### 2026-11-03 v3 镜像重建结果

已用本地 `/local/lnn/TL-inference-server` 和 `/local/lnn/tt-metal` 重建：

```text
tl-inference-server:flux-3-image-klein-local-20261103-v3
tt-metal commit: 23ef0fd19b
镜像大小：约 8.30 GB
模型权重：未进入镜像，运行时只读挂载 /local/models/flux-3-image-klein-base
```

镜像内已检查 Transformer、Qwen3-VL 和 VAE 映射代码存在，且模型权重路径不存在；因此该镜像与当前两个本地仓库代码一致，模型文件仍保持独立部署。

### v3 真实设备复核

使用 v3 镜像在 Galaxy02 物理芯片 `0`、mesh `(1,1)` 上重新执行：

```text
FULL_MAPPING_V3_OK {'source_tensors': 357, 'ttnn_parameter_entries': 271,
                   'mode_blocks': 15, 'single_blocks': 28}
TEXT_ENCODER_V3_OK Qwen3VlTextEncoder [4, 8, 12, 16, 20, 24, 28, 32]
VAE_MAPPING_V3_OK {'source_tensors': 251}
```

三次容器运行均为 `STATUS=0`。这证明当前 v3 镜像内的本地源码可以在真实 Blackhole 设备上完成三类模型组件的权重转换；仍未证明完整 prompt → denoise → VAE → PNG 的端到端 T2I 运行。

## 附录 F：2026-11-04 双模型质量修复记录

早期章节保留了制定计划和逐步 bring-up 时的历史状态。当前真实状态以本节为准。

### 两套模型和切换方式

| 变体 | 模型目录 | 默认推理参数 | 状态 |
|---|---|---|---|
| Base | `/local/models/flux-3-image-klein-base` | 50 steps、CFG 3.5 | 单芯片 T2I 已实现并验证 |
| Distilled | `/local/models/flux-3-image-klein` | 16 steps、无 CFG、flow-map | 单芯片 T2I 已实现并验证 |

模型统一从宿主机 `/local/models` 只读挂载。`Flux3Config.from_checkpoint()` 根据 `config.json` 自动识别 `flow_map`，不维护两份 pipeline。Base 和 Distilled 使用不同 TTNN cache 名，避免交叉复用转换权重。

### 质量问题修复

`a black bird` 生成蓝色块的主要根因是 SwiGLU 半区顺序不一致：checkpoint 是 `[gate | up]`，TTNN fused loader 期望 `[up | gate]`。已经在 `Flux3Transformer._block_state()` 中交换两半。

完整排查记录：

```text
docs/FLUX3_IMAGE_KLEIN_TT_QUALITY_DEBUG.md
```

### 验证结果

| 变体 | 请求 | 耗时 | 结果 |
|---|---|---:|---|
| Base | `a black bird`、1024×1024、50 steps、CFG 3.5、seed 42 | 68.37 s | 生成语义正确的黑鸟 |
| Distilled | `a black bird`、1024×1024、16 steps、seed 42 | 17.45 s | 生成语义正确的黑鸟 |

### 当前完成度

| 工作项 | 状态 | 说明 |
|---|---|---|
| Base 单芯片 T2I | 已完成 | prompt → Qwen3-VL → 50-step CFG denoise → VAE → 图片 |
| Distilled 单芯片 T2I | 已完成 | prompt → Qwen3-VL → 16-step flow-map denoise → VAE → 图片 |
| API 参数 | 已完成 | 可指定 prompt、steps、seed、width、height；Base 可指定 guidance |
| `response_format=file` | 已完成 | 单图直接返回 PNG/JPEG |
| 两模型独立 cache | 已完成 | Base/Distilled cache 名分离 |
| 2/4 芯片并行 | 暂缓 | 按当前范围不实施 |
| Golden Data 逐层对齐 | 暂缓 | 按当前范围不实施 |
| 32 worker 首次 cache 创建 | 未完成 | 需预生成 cache 或增加跨进程锁和原子发布 |
| kernel/trace/layout 专项优化 | 部分完成 | 已完成组件常驻、schedule 缓存、设备端 CFG/Euler/latent；更深层 blocking、trace、峰值内存仍待 profiling |

该次质量修复验证使用单 worker。截至 2026-11-05，`flux3-media-server-test` 已运行 Distilled 模型的 32 个单芯片 worker，并完成 200 请求并发验证。切换 Base 时只需更换模型路径并使用对应 cache，无需复制权重或编写第二套实现。

## 附录 G：2026-11-05 vLLM-Omni 同优先级性能优化完成状态

本轮按以下顺序实施，详细代码和验证过程见适配记录第 27 节：

| 优先级 | 工作项 | 状态 | 实现位置 |
|---:|---|---|---|
| 1 | 每 worker 常驻 prompt encoder / Transformer / VAE | 已完成 | `tt-media-server/tt_model_runners/dit_runners.py` |
| 2 | 相同步数 scheduler 缓存 | 已完成 | `models/tt_dit/pipelines/flux3/pipeline_flux3.py` |
| 3 | 初始 latent 只执行一次 H2D | 已完成 | `Flux3T2IPipeline.run_with_prompt_embeds()` |
| 4 | Base CFG 保留在 TT 设备端 | 已完成 | `ttnn.subtract`、`ttnn.multiply_`、`ttnn.add_` |
| 5 | Euler update 和 denoise latent 保留在 TT 设备端 | 已完成 | `ttnn.multiply_`、`ttnn.add_` |
| 6 | 最终设备 latent 直接进入 VAE | 已完成 | `vae_decoder.preprocess_and_unpatchify(image_tt, ...)` |
| 7 | 显存不足回退逐请求加载 | 已完成 | `FLUX3_COMPONENT_RESIDENCY=per_request` |
| 8 | Docker 默认启用常驻模式 | 已完成 | `tt-media-server/Dockerfile` |

真实验证：

```text
模型：/local/models/flux-3-image-klein（Distilled）
请求：a black bird，1024x1024，16 steps，seed 42，PNG
服务：32 个单芯片 worker
组件重复加载事件：0
worker error_count：0

Base 补充验证：/local/models/flux-3-image-klein-base，50 steps，CFG 3.5，
单 worker，HTTP 200，62.395 s，1024x1024 PNG，语义正确，error_count=0
```

| 场景 | 结果 |
|---|---:|
| 单请求后 3 次平均 | 11.541 s |
| 32 请求 / 32 并发 | 12.123 s；2.6397 images/s |
| 200 请求 / 32 并发 | 81.209 s；2.4628 images/s |
| 200 请求 / 200 同时提交 | 77.820 s；2.5700 images/s；P95 67.038 s |

与旧的 200 同时提交基线相比，总墙钟降低 41.3%，吞吐提升 70.3%。Distilled flow-map 和 Base CFG 两条路径的生成样图均语义正确。当前 TT 设备端 Euler accumulator 为 BF16；后续精度工作应评估 FP32 accumulator，但不应直接恢复每个 denoise step 的 host/device 往返。

功能提交：

```text
TL-inference-server e46186a9160d48344ec1f706f6ff4efd6f6c53ba
tt-metal           3ff5dc122dae38d173870cf5525e9ed96d877c33
```
