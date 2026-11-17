# SD3.5 在 `bh-galaxy-sd35-wan-sdxl:0.22.0-da1e44c2c4b5` 镜像中的代码结构与推理流程

## 1. 分析范围

本文分析镜像：

```text
ghcr.io/tenstorrent/tt-inference-server/bh-galaxy-sd35-wan-sdxl:0.22.0-da1e44c2c4b5
```

镜像信息：

```text
Image ID: sha256:4106ccf117d472f4cbccce009e6e2dc06299070e0b490ce6d249f1e2bb2ce48a
TT_METAL_COMMIT_SHA_OR_TAG: da1e44c2c4b5e6fef1fa3fed896812f65026f86e
Entrypoint: ./docker-entrypoint.sh
Default server mode: uvicorn
```

镜像中的主要代码根目录是：

```text
/home/container_app_user/tt-metal/server
/home/container_app_user/tt-metal/models/tt_dit
```

这个 SD3.5 实现不是“请求到来后调用一次 diffusers 的 PyTorch pipeline”。它将服务、worker、runner、TTNN 文本编码器、TTNN DiT Transformer、TTNN VAE 和设备通信拆成多个层次：

```text
HTTP API
  -> Service / request queue
  -> device worker
  -> runner_fabric
  -> TTSD35Runner
  -> StableDiffusion3Pipeline
  -> TextEncoder + SD35Transformer2DModel + VAE decoder
  -> JPEG/PNG 或 Base64 响应
```

## 2. 镜像启动结构

镜像的默认启动链是：

```text
docker-entrypoint.sh
  -> run_uvicorn.sh
  -> FastAPI / Uvicorn
  -> media-server 主服务
```

容器启动时通常由环境变量选择服务和模型：

```text
MODEL_SERVICE=image
MODEL_RUNNER=tt-sd3.5
DEVICE=...
```

`MODEL_RUNNER=tt-sd3.5` 是关键选择。它告诉配置系统当前是图像服务，并且设备 runner 使用 SD3.5 专用实现。

## 3. 文件职责总览

| 文件 | 作用 |
|---|---|
| `server/open_ai_api/image.py` | 提供图像生成 HTTP 接口，检查 API key，并把请求交给 service |
| `server/open_ai_api/models.py` | `/v1/models` 和请求模型映射 |
| `server/config/constants.py` | 定义 `tt-sd3.5`、模型名、服务类型、设备 mesh 和设备 ID 策略 |
| `server/config/settings.py` | 读取环境变量，推导 model service、设备类型、mesh shape 和 runner 参数 |
| `server/device_workers/worker_utils.py` | 创建 event loop、创建 runner、打开设备并执行 warmup |
| `server/device_workers/device_worker.py` | worker 主循环，从队列取请求并调用 runner |
| `server/tt_model_runners/runner_fabric.py` | 根据 `MODEL_RUNNER` 创建具体 runner |
| `server/tt_model_runners/base_device_runner.py` | 所有设备 runner 的抽象生命周期 |
| `server/tt_model_runners/base_metal_device_runner.py` | TTNN mesh 打开、fabric 配置、父 mesh/子 mesh 生命周期 |
| `server/tt_model_runners/dit_runners.py` | `TTDiTRunner` 通用 DiT runner 和 `TTSD35Runner` SD3.5 runner |
| `models/tt_dit/pipelines/stable_diffusion_35_large/pipeline_stable_diffusion_35_large.py` | SD3.5 顶层推理编排 |
| `models/tt_dit/pipelines/stable_diffusion_35_large/text_encoder.py` | CLIP、CLIP、可选 T5 文本编码 |
| `models/tt_dit/models/transformers/transformer_sd35.py` | SD3.5 TTNN DiT Transformer、权重加载和 patch/unpatch |
| `models/tt_dit/models/transformers/attention_sd35.py` | SD3.5 joint attention 和 TP/通信逻辑 |
| `models/tt_dit/models/vae/vae_sd35.py` | 旧的 channel-TP VAE decoder |
| `models/tt_dit/models/vae/vae_sd35_spatial.py` | 默认的空间并行 VAE decoder |
| `models/tt_dit/parallel/config.py` | CFG、TP、SP、VAE 并行配置和 VAE CCL 辅助函数 |
| `models/tt_dit/parallel/manager.py` | CCL manager、all-gather、reduce-scatter 等集合通信 |
| `models/tt_dit/utils/cache.py` | TTNN 权重缓存和模型加载 |
| `models/tt_dit/utils/tracing.py` | 重复推理步骤的 device trace |
| `models/tt_dit/tests/models/sd35/run_sd35_submesh.py` | SD3.5 4 卡子网格、Ring/Linear 和性能测试入口 |

## 4. HTTP 请求到设备的路径

### 4.1 API 层

入口文件：

```text
server/open_ai_api/image.py
```

图像生成接口的主要逻辑是：

```python
@generate_image_router.post("/generations")
async def generate_image(
    image_generate_request: ImageGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    result = await service.process_request(image_generate_request)
    return JSONResponse(content={"images": result, "generation_time": elapsed})
```

这一层负责：

1. FastAPI 解析 JSON 请求；
2. `get_api_key` 校验 API key；
3. Pydantic request model 校验 prompt、步数、seed 等字段；
4. 调用 service，把任务放入共享请求队列；
5. 等待 worker 返回结果；
6. 把生成结果编码后返回 JSON。

SD3.5 在这个镜像中使用单请求推理，`TTDiTRunner.run()` 取 `requests[0]`，因此服务层虽然使用统一队列框架，但 SD3.5 runner 本身不是批量生成实现。

### 4.2 Runner 工厂

文件：

```text
server/tt_model_runners/runner_fabric.py
```

核心映射类似：

```python
ModelRunners.TT_SD3_5: lambda wid: __import__(
    "tt_model_runners.dit_runners",
    fromlist=["TTSD35Runner"],
).TTSD35Runner(wid)
```

所以 `MODEL_RUNNER=tt-sd3.5` 的实际结果是：

```text
get_device_runner(worker_id)
  -> ModelRunners.TT_SD3_5
  -> TTSD35Runner(worker_id)
```

这种设计是工厂模式：HTTP 层不需要知道 SD3.5 的类名，也不需要直接导入所有模型。增加新模型时，主要增加枚举、映射和 runner 类即可。

### 4.3 Worker 初始化和 warmup

文件：

```text
server/device_workers/worker_utils.py
server/device_workers/device_worker.py
```

每个 worker 的初始化大致为：

```text
创建独立 asyncio event loop
  -> get_device_runner(worker_id)
  -> runner.set_device()
  -> 打开 TTNN mesh
  -> runner.warmup()
  -> worker 把 ready 信号放入 warmup_signals_queue
  -> 进入请求队列循环
```

`TTDiTRunner.warmup()` 做两件关键事情：

1. 在后台线程调用 `create_pipeline()`，完成模型对象创建、权重加载和设备分布；
2. 构造一个 2-step 的临时图像请求，触发首次编译、trace 和程序缓存。

正式请求不会重复创建模型，直接复用常驻 pipeline。

## 5. `TTSD35Runner` 的设备逻辑

文件：

```text
server/tt_model_runners/dit_runners.py
```

`TTSD35Runner` 继承关系为：

```text
BaseDeviceRunner
  -> BaseMetalDeviceRunner
  -> TTDiTRunner
  -> TTSD35Runner
```

### 5.1 创建 pipeline

`TTSD35Runner.create_pipeline()` 调用：

```python
StableDiffusion3Pipeline.create_pipeline(
    mesh_device=self.ttnn_device,
    checkpoint_name=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value,
)
```

也就是说，runner 负责设备和生命周期，真正的模型结构由 `StableDiffusion3Pipeline` 负责。

### 5.2 设备参数

默认基础参数包括：

```text
l1_small_size = 32768
trace_region_size = 25000000
```

当 mesh 是 `(1,4)` 或 `(4,1)` 时，runner 改为：

```text
fabric_config = FABRIC_1D_RING
trace_region_size = 50000000
```

原因是 4 芯片 SD3.5 配置使用 Ring fabric，而且 trace 所需空间更大。

### 5.3 Blackhole Galaxy 的父 mesh/子 mesh

Galaxy 是物理环形连接的 32 芯片系统。这个版本不会直接打开一个孤立的 4 芯片 mesh，而是：

```text
打开完整父 mesh： (4,8)
  -> 从父 mesh 切出： (4,1)
  -> SD3.5 只使用这个 4 芯片子 mesh
```

代码中的计划是：

```python
SD35_BH_GALAXY_PARENT_MESH_PLANS = {
    (4, 1): ((4, 8), (4, 1), None),
}
```

这样做的原因不是模型必须使用 32 卡，而是 Galaxy 的 Fabric router 需要完整父 mesh 才能正确初始化。直接打开部分环形设备可能出现 `Fabric Router Sync` 超时。

这个设计的几个要点：

- 父 mesh 保持存活，子 mesh 只是父 mesh 的 view；
- 关闭设备时必须关闭父 mesh；
- `TT_VISIBLE_DEVICES` 不能随意只暴露 `(0,1,2,3)`，否则物理拓扑映射可能失败；
- 该 SD3.5 配置在 Galaxy 上是 4 卡计算，其他芯片不属于这个 worker 的 DiT mesh；
- 这是“4 卡一个 worker”的模型布局，不是 32 卡 TP。

## 6. `StableDiffusion3Pipeline` 的顶层结构

文件：

```text
models/tt_dit/pipelines/stable_diffusion_35_large/pipeline_stable_diffusion_35_large.py
```

pipeline 内部持有：

```text
TextEncoder
SD35Transformer2DModel（一个或多个 submesh 实例）
EulerSolver（每个 submesh 一个）
SD35SpatialVaeAdapter 或 VAEDecoderAdapter
CCLManager
Tracer
```

### 6.1 并行 preset

镜像中的 preset 主要如下：

| Mesh | CFG | SP | TP | links | 说明 |
|---|---:|---:|---:|---:|---|
| `(2,2)` | 1 | 2 | 2 | 2 | 2×2，DiT 使用 SP2×TP2 |
| `(2,4)` | 2 | 2 | 2 | 1 | 两个 CFG submesh |
| `(4,8)` | 2 | 4 | 4 | 4 | 32 卡完整 mesh 的 preset |
| `(4,1)` | 1 | 1 | 4 | 2 | Galaxy 四卡列布局，TP4 |
| `(1,4)` | 1 | 1 | 4 | 2 | QuietBox 2 的四卡行布局，TP4 |

这里的三个并行维度含义是：

- CFG parallel：把 unconditional/conditional 分到不同 submesh；
- SP：沿 token/sequence 维度切分 latent token；
- TP：沿 hidden/channel/head/linear 维度切分权重和激活。

### 6.2 初始化顺序

pipeline 初始化大致为：

```text
读取 mesh shape 和 preset
  -> create_submeshes()
  -> 为每个 submesh 创建 CCLManager
  -> 判断 encoder/VAE 是否需要 reshape
  -> 创建 SD35Checkpoint
  -> 为每个 submesh 构建 SD35Transformer2DModel
  -> 从 Hugging Face scheduler 目录创建 FlowMatchEulerDiscreteScheduler
  -> 创建 EulerSolver
  -> 创建 CLIP/可选 T5 TextEncoder
  -> 创建空间并行 VAE
  -> synchronize_device()
  -> 执行一次 pipeline allocation run
```

`SD35Checkpoint` 先用 diffusers 读取 checkpoint 配置和 PyTorch state dict，但实际推理模型是 TTNN `SD35Transformer2DModel`。因此 diffusers 在这里主要承担 checkpoint 格式和 scheduler 的兼容角色，不是最终计算后端。

## 7. 文本编码流程

文件：

```text
models/tt_dit/pipelines/stable_diffusion_35_large/text_encoder.py
```

SD3.5 文本条件由两个 CLIP 和一个可选 T5 组成：

```text
prompt
  -> CLIP tokenizer 1 -> TTNN CLIP encoder 1
  -> CLIP tokenizer 2 -> TTNN CLIP encoder 2
  -> T5 tokenizer      -> TTNN T5 encoder（可选）
```

### 7.1 CLIP

两个 CLIP encoder 分别输出：

- sequence hidden states；
- pooled embedding。

两个 CLIP 的 sequence embedding 在最后一个特征维拼接，再 pad 到 joint attention 所需宽度。

### 7.2 T5

T5 默认是否启用由 mesh shape 决定。pipeline 中的逻辑是：只有 CFG 处理后仍然能得到适合 T5 的 mesh 形状时才启用；在无法适配的布局上会自动关闭 T5。

T5 的 hidden states 与 CLIP sequence embedding 在 token 维拼接，形成 DiT 的 `prompt_embed`。

### 7.3 CFG

开启 CFG 时，正向 prompt 和 negative prompt 分别编码，然后拼成：

```text
[negative_prompt_embeddings, positive_prompt_embeddings]
```

若 CFG 没有单独的并行 submesh，`_traced_step()` 会在 batch 维复制 latent，按照 unconditional/conditional 两份一起跑 DiT。DiT 运行后，`CFGCombiner` 根据 guidance scale 合并两路 velocity：

```text
guided = unconditional + guidance_scale * (conditional - unconditional)
```

## 8. latent 和 DiT 去噪流程

### 8.1 latent 初始化

pipeline 根据 VAE scale factor 和 patch size 计算 latent 尺寸：

```text
VAE scale factor = 8
patch size       = 2
latent channels  = checkpoint.in_channels
```

例如 1024×1024 图像对应的 latent 空间是 128×128，之后再按 patch size 2 变成 token 序列。

随机 latent 在 host 侧按 seed 生成，然后调用 transformer 的 `patchify()`，最后按 SP 配置把 token 维分发到各个 device/submesh。

### 8.2 patchify

`patchify()` 的逻辑是把：

```text
(B, H/8, W/8, C)
```

转换成：

```text
(1, B, N, patch_size * patch_size * C)
```

其中：

```text
N = (H / 8 / patch_size) * (W / 8 / patch_size)
```

DiT 主要处理这个 token 序列，而不是直接处理二维图像。

### 8.3 Euler 去噪循环

每个 inference step 的主要顺序是：

```text
生成 timestep tensor
  -> 对每个 CFG/SP/TP submesh 执行 traced transformer
  -> 得到 velocity prediction
  -> CFG combine
  -> 各 device synchronize
  -> EulerSolver.step()
  -> 得到下一步 latents
```

伪代码为：

```python
for i, t in enumerate(timesteps):
    velocity_preds = []
    for idx, tracer in enumerate(self._tracers):
        velocity_preds.append(
            tracer(
                cfg_enabled=self._cfg_enabled,
                latents=latents[idx],
                prompt_embed=context[idx],
                pooled_projections=pooled[idx],
                timestep=timestep,
                N=latents_sequence_length,
                traced=traced,
            )
        )

    velocity_preds = self._cfg_combiner.combine(velocity_preds, guidance_scale)
    latents = [solver.step(...)]
```

### 8.4 Transformer block

文件：

```text
models/tt_dit/models/transformers/transformer_sd35.py
models/tt_dit/models/transformers/attention_sd35.py
```

每个 `SD35TransformerBlock` 主要包含：

```text
adaLN / timestep modulation
  -> spatial/text joint attention
  -> residual connection
  -> feed-forward network
  -> residual connection
```

关键实现方式：

- `ColParallelLinear` 和其他 parallel linear 对权重/激活执行 TP；
- attention 的 Q/K/V、to_out 使用 TTNN matmul；
- FFN 使用并行 feed-forward；
- distributed layer norm 通过 CCL 获取跨设备统计量；
- Ring TP 可以启用 fused all-gather matmul、fused matmul + reduce-scatter；
- timestep 产生的 adaLN gate 会扩展到每个 spatial token，保证 CFG 两个 batch 元素使用各自的 modulation；
- quant profile 由环境变量选择，必要时注册特定 mesh/grid 的 matmul blocking。

模型权重加载不是每次请求重新做。`SD35Checkpoint.build()` 创建 TTNN 模型后调用：

```python
cache.load_model(
    tt_model=model,
    get_torch_state_dict=lambda: self._state_dict,
    model_name=os.path.basename(self._name),
    subfolder="transformer",
    parallel_config=parallel_config,
    mesh_shape=tuple(device.shape),
    mesh_device=device,
)
```

这会把 PyTorch checkpoint 转换为对应 mesh/TP/SP 配置的 TTNN 权重并缓存。

## 9. VAE 解码流程

### 9.1 pipeline 入口

去噪结束后，pipeline 调用：

```python
output = self._decode_latents(latents[self.vae_submesh_idx], traced=vae_traced)
```

首先将 DiT 的 SP token 维 all-gather 回 VAE 需要的完整 latent 序列：

```python
tt_latents = ccl_manager.all_gather_persistent_buffer(
    tt_latents,
    dim=2,
    mesh_axis=self.dit_parallel_config.sequence_parallel.mesh_axis,
)
```

### 9.2 默认空间并行 VAE

默认 `vae_spatial=True`，使用：

```text
SD35SpatialVaeAdapter
  -> SD35VaeDecoder
```

流程为：

```text
patchified latent
  -> device 上逆 scaling/shift
  -> device 上 unpatchify
  -> 按 H/W 在 device 间切分
  -> halo/neighbor exchange
  -> VAE conv/resnet/mid-block/up-block
  -> device 上 uint8 转换
  -> H/W all-gather
  -> host 图像
```

空间并行 VAE 的设计重点是：

- 用空间切分代替每个卷积前的完整 channel all-gather；
- 邻居之间只交换 halo 区域；
- `conv3d` 路径使用针对 SD3.5 的 blocking 表；
- `GroupNorm + SiLU` 尽量融合；
- unpatchify 和 uint8 转换留在 device；
- 使用 `traced_function`，重复请求可以复用 VAE trace。

`sd35_vae_parallel_config()` 的策略是：

- mesh 只有一个非 1 轴：沿该轴做 width split；
- mesh 有两个非 1 轴：第一个非 1 轴分 width，第二个分 height；
- 单卡不做 H/W 并行。

### 9.3 旧的 channel-TP VAE

`vae_sd35.py` 中保留了 `VAEDecoderAdapter` 和旧 decoder 路径。它的特点是：

```text
host torch latent
  -> TTNN VAE
  -> channel TP all-gather
  -> host torch output
```

它可以 tracing，但 pipeline 默认优先使用 `SD35SpatialVaeAdapter`。关闭 `vae_spatial` 后，才会创建旧的 `VAEDecoderAdapter`。

因此镜像中同时存在两套 VAE，不是重复代码：

| VAE | 默认状态 | 并行方式 | 适用特点 |
|---|---|---|---|
| `SD35SpatialVaeAdapter` | 开启 | H/W 空间并行、halo exchange | 默认性能路径，减少 host 往返 |
| `VAEDecoderAdapter` | 关闭 | channel TP + all-gather | 兼容路径、调试路径 |

## 10. Trace、缓存和常驻模型

这个镜像的性能结构不是每次请求都重新编译 kernel：

```text
worker warmup
  -> pipeline allocation run
  -> 2-step warmup
  -> 生成 trace / 编译程序
  -> 后续请求复用权重、program 和 trace
```

主要机制包括：

- `cache.load_model()`：缓存转换后的 TTNN 权重；
- `Tracer`：缓存重复的 transformer step；
- VAE 的 `traced_function`：缓存重复的 VAE decode；
- `CCLManager`：复用集合通信资源和 semaphore；
- worker 常驻：一个 worker 初始化一次 device/pipeline，连续服务多个请求。

因此第一次 warmup 或新分辨率首次请求通常明显较慢，稳定性能应在 warmup 和首次编译完成后统计。

## 11. 这套代码采用的编程结构

### 11.1 Factory + Registry

`ModelRunners` 枚举和 `AVAILABLE_RUNNERS` 字典组成 runner registry：

```text
字符串 MODEL_RUNNER
  -> ModelRunners enum
  -> runner factory
  -> 具体 runner
```

优点是服务层与具体模型解耦。

### 11.2 Template Method

`TTDiTRunner` 提供通用流程：

```text
set device
warmup
load pipeline
run request
record telemetry
```

具体模型只实现：

```python
create_pipeline()
get_pipeline_device_params()
```

`TTSD35Runner` 只负责 SD3.5 的 pipeline 和设备布局，公共生命周期由父类处理。

### 11.3 Pipeline Composition

SD3.5 pipeline 不把所有代码放进一个巨大 `forward()`，而是组合多个组件：

```text
TextEncoder
SD35Transformer2DModel
EulerSolver
SD35SpatialVaeAdapter
CCLManager
Tracer
```

这样可以独立替换文本编码、DiT 并行策略和 VAE 路径。

### 11.4 Adapter

`VAEDecoderAdapter` 和 `SD35SpatialVaeAdapter` 都对外提供 decode 能力，但隐藏不同的设备布局和数据格式转换。pipeline 不需要知道 VAE 内部是旧 channel-TP 还是空间并行。

### 11.5 Preset + 自动推导

并行参数通过 mesh shape preset 选择，而不是在每一层手写大量 `if/else`：

```text
mesh shape
  -> cfg/sp/tp preset
  -> encoder/VAE submesh
  -> CCL topology
  -> transformer/VAE build
```

这使同一套 pipeline 可以覆盖 2×2、1×4、4×1、2×4 和 4×8 等布局。

## 12. 一次完整推理的时序图

```text
POST /v1/images/generations
        |
        v
API key + Pydantic request validation
        |
        v
BaseService.process_request()
        |
        v
shared task queue
        |
        v
device_worker
        |
        v
TTSD35Runner._run_async()
        |
        v
StableDiffusion3Pipeline.__call__()
        |
        +--> CLIP 1 / CLIP 2 / optional T5
        |       |
        |       +--> prompt embeddings + pooled projections
        |
        +--> seed -> random latent -> patchify -> SP distribute
        |
        +--> for each timestep:
        |       DiT joint attention + FFN
        |       CFG combine
        |       EulerSolver.step
        |
        +--> SP latent all-gather
                |
                +--> spatial VAE unpatchify
                +--> H/W shard + halo exchange
                +--> VAE decode
                +--> uint8 + H/W gather
        |
        v
host image -> Base64/JSON response
```

## 13. 对 FLUX.3 实现的可复用经验

这套 SD3.5 代码对 FLUX.3 有几项直接参考价值：

1. 让 `runner_fabric.py` 只负责“字符串到 runner”的映射，避免 API 层直接耦合模型类；
2. 把设备初始化、warmup、请求执行、关闭设备集中在 `BaseMetalDeviceRunner`/`TTDiTRunner`；
3. Galaxy 上先打开完整父 mesh，再切出 worker 子 mesh，避免 Fabric router 初始化失败；
4. 将 TP、SP、CFG 和 VAE 并行配置放进独立 dataclass/NamedTuple；
5. 将模型权重转换缓存、程序编译缓存、trace 缓存和 pipeline 常驻作为正式生命周期的一部分；
6. VAE 尽量使用空间并行，减少 token all-gather 后的 host 往返；
7. 对每个阶段发出 `encoder`、`denoising`、`vae` 和 `total` 事件，便于准确区分设备时间和 HTTP E2E 时间；
8. 把硬件拓扑限制写进 runner 的 mesh plan，而不是由用户手动填写物理 device ID。

## 14. 与当前 FLUX.3 结构的主要差异

| 项目 | 镜像中的 SD3.5 | 当前 FLUX.3 方向 |
|---|---|---|
| prompt encoder | CLIP1 + CLIP2 + 可选 T5 | Qwen3-VL prompt encoder / 外部 hidden states |
| DiT 条件 | text sequence + pooled projection | text/image/reference 条件和 joint RoPE |
| CFG | pipeline 内部 negative/positive 双路 | 由 FLUX.3 pipeline 的条件结构决定 |
| VAE | SD3.5 AutoencoderKL decoder | FLUX.3 专用 VAE encoder/decoder |
| 图像条件 | 该 SD3.5 runner 是纯文生图路径 | FLUX.3 支持参考图和 image-condition |
| 4 卡 Galaxy | `(4,8)` 父 mesh 切 `(4,1)`，TP4 | 需要根据 FLUX.3 的 SP/TP profile 选择 mesh |
| 请求 batch | runner 实际取第一个请求，通常 batch=1 | 需要单 worker batch 或多 worker 并发分别设计 |

SD3.5 的代码结构可以复用，但不能直接把 SD3.5 的 TP/SP preset、VAE 和 text encoder 拷贝给 FLUX.3；两者的 token 形状、条件输入、RoPE、DiT block 和 VAE 数据布局不同。

## 15. 结论

这个镜像中的 SD3.5 方案可以概括为：

```text
统一 OpenAI 风格 API
  + 多进程 device worker
  + runner registry
  + TTNN 常驻 pipeline
  + preset 驱动的 CFG/TP/SP
  + Galaxy 父 mesh -> 4 卡子 mesh
  + CLIP/T5 文本编码
  + TTNN DiT 去噪
  + 空间并行 VAE
  + warmup/trace/weight cache
```

它的核心编程思想不是把模型写成一个大函数，而是把“服务调度、硬件拓扑、并行策略、模型组件、缓存和性能采集”分别封装，再通过 pipeline 组合起来。

