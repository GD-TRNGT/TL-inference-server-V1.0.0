# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 中的代码架构与适配实现逻辑

更新时间：2026-11-18
代码基线：`1ae19312 flux3: complete parent child mesh profile regression`

本文按照 SD3.5 架构分析文档的方式，说明 FLUX.3 Image Klein 从 HTTP 请求到 TTNN 设备执行的完整代码结构、模块职责、并行布局、缓存策略和验证边界。

## 1. 分析范围与当前结论

当前 FLUX3 实现由两个仓库共同组成：

```text
/local/lnn/TL-inference-server
/local/lnn/tt-metal
```

它不是“收到请求后调用一次 PyTorch/diffusers pipeline”，而是：

```text
HTTP API
  -> service / request queue
  -> device worker
  -> FLUX3 runner
  -> Flux3T2IPipeline
  -> Qwen3-VL text encoder
  -> FLUX.3 Transformer denoise
  -> FLUX.3 VAE decoder
  -> JPEG/PNG file 或 JSON 响应
```

当前已经实现并在 Galaxy03 真机验证：

- FLUX.3 Image Klein Distilled 的文本生成路径；
- Qwen3-VL prompt encoder、FLUX.3 Transformer 和 VAE decoder；
- resident pipeline、权重缓存和 kernel/program cache；
- 单卡、双卡、四卡、八卡 profile；
- direct Mesh 和 Galaxy parent Mesh/child Mesh 路径；
- 1～10 张参考图的数据结构和 image-condition 路径；
- API key、输出格式、阶段耗时和 JPEG/PNG 响应；
- 16 steps、seed 42、同步请求的 2/4/8 卡 parent/child 回归。

本轮没有继续调试 21:9 和 9:21；也没有因为本文更新重新构建镜像。

## 2. 镜像和服务启动结构

当前服务的逻辑启动链为：

```text
docker-entrypoint.sh
  -> run_uvicorn.sh
  -> FastAPI / Uvicorn
  -> media-server 主服务
  -> worker scheduler
  -> FLUX3 device runner
```

典型模型选择由环境变量完成：

```text
MODEL_SERVICE=image
MODEL_RUNNER=tt-flux.3-image-klein
MODEL_WEIGHTS_DIR=/local/models/flux-3-image-klein
```

缓存使用容器内统一目录：

```text
/local/ttcache
└── flux-3-image-klein
    ├── tt_dit_cache
    ├── tt-metal-cache
    ├── hf
    └── torch
```

模型权重通常由宿主机只读挂载：

```text
/local/models/flux-3-image-klein -> /local/models/flux-3-image-klein:ro
```

权重不需要复制进镜像；镜像提供代码和运行时，模型与 cache 在运行时挂载。

## 3. 文件职责总览

### 3.1 TL-inference-server

| 文件/目录 | 作用 |
|---|---|
| `tt-media-server/open_ai_api/image.py` | OpenAI-compatible 图片生成 API、鉴权和响应格式 |
| `tt-media-server/open_ai_api/models.py` | 模型名称和 `/v1/models` 映射 |
| `tt-media-server/config/constants.py` | runner、service、设备类型等枚举 |
| `tt-media-server/config/settings.py` | 环境变量、mesh discovery、parent/child 规则 |
| `tt-media-server/config/flux3_profiles.py` | FLUX3 并行 profile 的唯一事实来源 |
| `tt-media-server/device_workers/worker_utils.py` | worker 创建、event loop、runner 初始化和 warmup |
| `tt-media-server/device_workers/device_worker.py` | 请求队列、worker ready、请求执行和重启 |
| `tt-media-server/tt_model_runners/runner_fabric.py` | runner factory，根据模型名创建具体 runner |
| `tt-media-server/tt_model_runners/base_device_runner.py` | runner 生命周期抽象 |
| `tt-media-server/tt_model_runners/base_metal_device_runner.py` | TTNN Mesh、Fabric、设备关闭和基础设备参数 |
| `tt-media-server/tt_model_runners/dit_runners.py` | `TTDiTRunner` 与 FLUX3 runner 的 pipeline/请求逻辑 |
| `tt-media-server/utils/runner_utils.py` | worker 设备布局、Fabric 和 parent Mesh 参数转换 |

### 3.2 tt-metal

| 文件/目录 | 作用 |
|---|---|
| `models/tt_dit/pipelines/flux3/pipeline_flux3_klein.py` | FLUX3 Klein 的顶层 pipeline 编排 |
| `models/tt_dit/pipelines/flux3/flux3_components.py` | Text Encoder、VAE、参考图组件和权重装载 |
| `models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py` | tokenizer、chat template、Qwen3-VL prompt 编码 |
| `models/tt_dit/models/transformers/transformer_flux3.py` | FLUX3 Transformer、block、forward 和权重映射 |
| `models/tt_dit/blocks/attention_opt.py` | attention、RoPE、Ring/序列并行相关实现 |
| `models/tt_dit/models/vae/flux3_vae_encoder.py` | 参考图 VAE encoder 和 image-condition token |
| `models/tt_dit/models/vae/flux2_vae.py` | FLUX3 使用的 decoder 基础实现和卷积路径 |
| `models/tt_dit/utils/cache.py` | 权重转换缓存和模型加载 |
| `tt_metal/fabric/mesh_graph_descriptors/` | Galaxy 物理拓扑和 Fabric descriptor |

## 4. HTTP 请求到设备的路径

### 4.1 API 层

入口为：

```text
POST /v1/images/generations
```

主要职责：

1. FastAPI 解析 JSON 或 multipart 请求；
2. 校验唯一 API key；
3. 校验 prompt、width、height、steps、seed、response format；
4. 校验参考图数量和图片数据；
5. 将请求放入 service queue；
6. 等待 worker 完成 pipeline；
7. 返回 JPEG/PNG 文件或 JSON/base64 数据。

请求的核心数据结构可以概括为：

```text
Flux3 request
├── model
├── prompt
├── width / height
├── num_inference_steps
├── seed
├── response_format
├── image_return_format
└── reference_images[0..9]
```

公共 API 尺寸约束与内部 latent/VAE 分片约束要分开处理：API 层检查官方允许范围和 16 像素对齐，pipeline 负责内部 token/latent geometry 的设备对齐。

### 4.2 Runner factory

服务层不直接实例化 FLUX3 pipeline，而是经过 runner factory：

```text
MODEL_RUNNER=tt-flux.3-image-klein
  -> ModelRunners.TT_FLUX_3_IMAGE_KLEIN
  -> TTFlux3ImageKleinRunner
  -> Flux3T2IPipeline
```

这种结构让 API 层与模型实现解耦。增加新模型时，主要增加模型枚举、runner registry 和具体 runner，不需要修改通用队列逻辑。

## 5. Worker 初始化与 warmup

一个 worker 的生命周期为：

```text
创建 event loop
  -> 发现设备/mesh
  -> 创建 runner
  -> 打开 MeshDevice
  -> 创建 parent/child 或 direct Mesh
  -> 加载 Text Encoder、Transformer、VAE
  -> 加载/生成 cache
  -> 执行 warmup
  -> reported ready
  -> 进入请求队列
```

warmup 的作用不是生成用户图片，而是提前触发：

- 权重转换和 cache load；
- 典型分辨率的 tensor allocation；
- Metal kernel 编译；
- CCL/Fabric 初始化；
- Transformer 和 VAE program cache；
- resident pipeline 的设备内存准备。

首次 warmup 或首次出现新 geometry 时会明显慢于热态请求，性能统计必须把两者分开。

## 6. FLUX3 profile 和设备布局

文件：

```text
tt-media-server/config/flux3_profiles.py
```

profile 同时描述 worker mesh、parent mesh、TP/SP 轴和 Fabric：

| Profile | Worker/Child Mesh | Parent Mesh | 语义 |
|---|---:|---:|---|
| `legacy-single` | 1x1 | — | 单卡 |
| `sp2` | 2x1 | 4x8 | TP1 × SP2 |
| `tp2` | 2x1 | 4x8 | TP2 |
| `sp4` | 4x1 | 4x8 | TP1 × SP4 Ring |
| `tp4` | 4x1 | 4x8 | TP4 |
| `tp4axis1` | 1x4 | 4x8 | 转置 TP4 |
| `tp2sp2` | 2x2 | 8x4 | TP2 × SP2 |
| `sp8` | 8x1 | 8x4 | TP1 × SP8 Ring |
| `tp8ring` | 1x8 | 4x8 | TP8 Ring |
| `tp2sp4` | 2x4 | 8x4 | TP2 × SP4 |
| `tp4sp2` | 4x2 | 8x4 | TP4 × SP2 |
| `sp8tp4` | 4x8 | — | 32 卡 direct 全 Galaxy |

父 Mesh/子 Mesh 的独立实现见：

```text
docs/FLUX3_PARENT_CHILD_MESH_IMPLEMENTATION_SUMMARY.md
```

不要把 `TT_VISIBLE_DEVICES` 手工写成任意连续卡号来代替 profile；Galaxy placement 应由真实 descriptor 和 mesh discovery 决定。

## 7. Flux3T2IPipeline 的顶层结构

pipeline 内部可以抽象为：

```text
Flux3T2IPipeline
├── Qwen3-VL Text Encoder
├── optional reference VAE encoder
├── Flux3Transformer
├── flow schedule / Euler update
├── CFG/condition handling
├── Flux3 VAE decoder
├── CCL / mesh context
└── image output conversion
```

pipeline 的职责是编排组件，不把所有模型逻辑塞进 HTTP handler 或 worker。runner 负责设备生命周期，pipeline 负责一次请求中各模型组件的执行顺序。

## 8. Qwen3-VL 文本编码流程

```text
prompt
  -> tokenizer
  -> Qwen3-VL chat template
  -> padding / attention mask
  -> M-RoPE
  -> Qwen3-VL Text Encoder
  -> 选取配置层输出
  -> concat / prompt embedding
```

需要明确的 shape 边界包括：

- token 数量和 padding multiple；
- hidden state dtype/layout；
- text embedding 在 SP worker 中的复制或分布方式；
- 正向 prompt 与 Base CFG negative prompt 的区别；
- 参考图 image-condition token 与 text token 的拼接位置。

Qwen3-VL encoder 在 resident 模式下常驻 worker，请求之间不重复加载权重。

## 9. FLUX.3 Transformer 去噪

### 9.1 latent 与 token

```text
seed
  -> 初始 noise/latent
  -> latent geometry 计算
  -> patchify
  -> image IDs / RoPE IDs
  -> image token
```

DiT 主要处理 token 序列，而不是直接处理二维 RGB 图片。分辨率变化会影响 image token 数、RoPE 长度、attention memory 和 VAE geometry。

### 9.2 单步执行

```text
timestep embedding
  -> text/image/reference mode blocks
  -> single blocks
  -> attention + RoPE
  -> modulation / residual
  -> output projection
  -> velocity prediction
```

FLUX3 Transformer 由三类 mode block 和 single block 组成。每个 block 包含调制、attention、归一化、残差和 feed-forward 路径；具体层数量和权重命名由 FLUX3 checkpoint config 决定。

### 9.3 多步 flow update

```text
for each timestep:
    velocity = Transformer(latent, prompt, condition, timestep)
    velocity = CFG/condition combine when needed
    latent = Euler/flow update(latent, velocity, schedule)
```

flow schedule、CFG 和 Euler update 应在设备侧或 resident pipeline 中复用，不能把每一步都退回 host 重新创建完整状态。

## 10. 权重映射和 cache

权重路径为：

```text
FLUX3 checkpoint / safetensors
  -> 读取 source state_dict
  -> 解析 text/mode/single/final 层命名
  -> QKV/SwiGLU/linear layout 转换
  -> 映射到 TTNN module
  -> 按 profile/mesh/dtype 写入 cache
  -> 后续 cache.load_model()
```

cache key 必须包含会影响 tensor layout 的边界：

- 模型变体；
- TP/SP factor；
- mesh shape；
- dtype；
- resolution/geometry；
- VAE parallel 方式；
- 必要的 matmul/core-grid 配置。

因此不能把单卡 cache 直接当成 TP4 或 SP8 cache 使用。

## 11. VAE decoder 流程

去噪完成后：

```text
final denoised latent
  -> preprocess / inverse scaling
  -> unpatchify
  -> VAE decoder conv/resnet/up-block
  -> untilize / uint8 conversion
  -> device-to-host
  -> JPEG/PNG encode
```

VAE 需要单独考虑：

- latent 与输出图片的 scale factor；
- H/W spatial split；
- GroupNorm 和 halo；
- Conv program 的 geometry；
- `act_block_h`、double buffer 和 core grid；
- decode 后的 gather 和 host readback。

`act_block_h=64` 并不是所有卷积 geometry 都合法。TT-Metal 会根据每核 padded output height 选择合法值；日志中出现“64 不合法，选择最接近的 32”表示运行时回退到了合法 block，并不是模型失败。

## 12. 参考图和 image-condition

FLUX3 的参考图路径不是把图片直接送入文本 encoder，而是：

```text
1～10 张 reference_images
  -> 图片解码/尺寸预处理
  -> reference VAE encoder
  -> latent/token packing
  -> image-condition token
  -> 与 text/image token 组合
  -> Transformer denoise
```

旧单图字段保持兼容，新路径使用有序数组。每张参考图的顺序会影响条件 token 顺序，因此请求解析、图片预处理和 token 拼接必须保持稳定顺序。

## 13. resident pipeline、trace 和 warmup

当前性能路径是 resident，而不是每请求加载：

```text
worker warmup
  -> 创建 pipeline
  -> load/cache Text Encoder
  -> load/cache Transformer
  -> load/cache VAE
  -> 编译代表 geometry 的 kernel
  -> 建立 program/trace cache
```

请求阶段主要复用：

- 转换后的权重；
- Metal kernel；
- program cache；
- flow schedule；
- layout 相关 RoPE；
- resident model objects。

可以使用 per-request 模式作为回退，但它不能作为稳定性能基线。

## 14. 阶段日志和性能口径

一次请求应区分：

```text
text encode
reference encode
denoise
VAE decode
to_host / image postprocess
HTTP E2E
```

其中 HTTP E2E 还可能包括队列等待、worker 调度、JSON/file 响应和 JPEG/PNG 编码。因此：

```text
HTTP E2E ≠ denoise + VAE decode
```

性能测试固定使用：

```bash
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
FLUX3_CORE_GRID_OVERRIDE=12x10
```

并采用同步请求、固定分辨率、固定步数、固定 seed、固定输出格式。首次 kernel 编译和热态性能必须分开记录。

## 15. 代码采用的实现风格

### 15.1 Factory / Registry

```text
MODEL_RUNNER
  -> ModelRunners enum
  -> runner factory
  -> TTFlux3ImageKleinRunner
```

服务层不直接耦合具体模型类。

### 15.2 Template Method

基础 runner 统一处理：

```text
set device
warmup
load pipeline
run request
telemetry
shutdown
```

FLUX3 runner 只补充模型创建、设备参数和请求编排。

### 15.3 Pipeline Composition

pipeline 组合 Text Encoder、Transformer、VAE、schedule、CCL 和 output conversion，各组件可以独立替换和验证。

### 15.4 Adapter

参考图 encoder、VAE decode 和不同 parallel layout 通过 adapter 隐藏设备布局差异；上层 pipeline 使用统一的 token/latent 接口。

### 15.5 Preset + 自动推导

```text
profile / mesh shape
  -> TP/SP factor
  -> Fabric / CCL
  -> Transformer layout
  -> VAE layout
  -> cache key
```

不要在每个模型层里重复判断设备编号；并行语义应集中在 profile 和 parallel config。

## 16. 一次完整推理时序图

```text
POST /v1/images/generations
        |
        v
API key + request validation
        |
        v
service queue
        |
        v
ready device worker
        |
        v
TTFlux3ImageKleinRunner
        |
        +--> Qwen3-VL text encode
        |
        +--> optional reference VAE encode
        |
        +--> seed -> latent -> patchify
        |
        +--> for each timestep:
        |       FLUX3 Transformer
        |       CFG/condition combine
        |       flow Euler update
        |
        +--> VAE preprocess/unpatchify
                |
                +--> VAE decode
                +--> host image
                +--> JPEG/PNG
        |
        v
HTTP response
```

## 17. 与 SD3.5 架构的对应关系

| 设计点 | SD3.5 | FLUX3 |
|---|---|---|
| API/队列 | OpenAI-compatible service + worker | 相同的服务/worker 分层 |
| runner | `TTSD35Runner` | `TTFlux3ImageKleinRunner` |
| 文本编码 | CLIP1/CLIP2/T5 | Qwen3-VL |
| DiT | SD35Transformer2DModel | Flux3Transformer |
| 条件 | text + pooled projection + CFG | text/image/reference token + joint RoPE |
| VAE | SD3.5 spatial/channel VAE | FLUX3 VAE decoder/reference encoder |
| 设备并行 | preset 驱动 CFG/TP/SP | profile 驱动 TP/SP/Ring/2D/parent-child |
| cache | weight/program/trace | weight/program/geometry/layout cache |
| Galaxy | parent mesh -> child mesh | direct 与 parent/child 两条路径并存 |

可以复用 SD3.5 的工程结构，但不能直接复用 SD3.5 的 Transformer、VAE、text encoder 或并行 preset，因为两者的 token shape、条件格式、RoPE 和 VAE layout 不同。

## 18. 当前验证结果

固定条件：1024×1024、16 steps、seed 42、同步串行、JPEG、12×10 core grid、160W TDP。

| Profile | Parent -> Child | HTTP | Denoise | VAE | 结果 |
|---|---|---:|---:|---:|---|
| `sp2` | 4x8 -> 2x1 | 6.329s | 5.544s | 0.658s | HTTP 200 |
| `tp2` | 4x8 -> 2x1 | 4.289s | 3.498s | 0.658s | HTTP 200 |
| `sp4` | 4x8 -> 4x1 | 3.327s | 2.857s | 0.343s | HTTP 200 |
| `tp4` | 4x8 -> 4x1 | 4.030s | — | — | HTTP 200 |
| `tp4axis1` | 4x8 -> 1x4 | 5.045s | — | — | HTTP 200 |
| `tp2sp2` | 8x4 -> 2x2 | 4.298s | 3.518s | 0.658s | HTTP 200 |
| `sp8` | 8x4 -> 8x1 | 2.580s | 1.832s | 0.624s | HTTP 200 |
| `tp8ring` | 4x8 -> 1x8 | 3.464s | — | — | HTTP 200 |
| `tp2sp4` | 8x4 -> 2x4 | 2.773s | 2.306s | 0.344s | HTTP 200 |
| `tp4sp2` | 8x4 -> 4x2 | 3.349s | 2.565s | 0.657s | HTTP 200 |

`sp8tp4` 继续保持 32 卡 direct 全 Galaxy；本轮没有把它拆为 parent/child。21:9 和 9:21 没有纳入本轮调试。

## 19. 结论

FLUX3 的适配不是单独实现一个 Transformer，而是完成一条端到端链路：

```text
模型契约
  + API 校验
  + profile / mesh / Fabric
  + worker 生命周期
  + Qwen3-VL
  + FLUX3 Transformer
  + flow scheduler
  + VAE
  + resident/cache/trace
  + JPEG/PNG 输出
  + 同步性能与真机回归
```

代码提交：

```text
1ae19312 flux3: complete parent child mesh profile regression
```

父 Mesh/子 Mesh 的独立说明见：

```text
docs/FLUX3_PARENT_CHILD_MESH_IMPLEMENTATION_SUMMARY.md
```
