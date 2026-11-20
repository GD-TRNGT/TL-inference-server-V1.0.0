# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 中的适配实现逻辑

更新时间：2026-11-20  
代码基线：`flux-3-image-klein-tl-v1.0.35`
镜像：`galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35`

## 1. 适配目标与代码边界

FLUX.3 Image Klein 在推理服务中采用独立的模型适配链路，覆盖请求契约、worker 调度、runner、pipeline、Transformer、VAE 和图像返回：

```text
HTTP API
  -> 请求契约与鉴权
  -> service queue
  -> worker scheduler
  -> TTFlux3ImageKleinRunner
  -> Flux3KleinPipeline
  -> Qwen3-VL prompt encoder
  -> 可选 reference VAE encoder
  -> FLUX.3 Transformer denoise
  -> FLUX.3 VAE decoder
  -> JPEG/PNG 编码
  -> HTTP response
```

两个仓库的职责保持分离：

```text
/local/lnn/TL-inference-server   服务、API、worker、profile、设备发现
/local/lnn/tt-metal               TTNN/Metal 算子、FLUX3 pipeline、Transformer、VAE、cache
```

当前版本的源码提交：

```text
tt-metal:             4cca8b1f8c5cb26a1a51893c68fa783049118e36
TL-inference-server:  b0d9de2f8c21ed07c524cf7d52b1da8339d01fa7
tag:                  flux-3-image-klein-tl-v1.0.35
```

## 2. 服务端代码结构

### 2.1 TL-inference-server

以下是当前 FLUX3 Image Klein 适配实际涉及的服务端文件。通用 media-server 文件没有全部列出；下表覆盖 API、鉴权、请求校验、调度、worker、runner、设备拓扑、健康检查、指标和回归测试。

#### 请求、鉴权和 API

**API 契约的主要文件是以下 5 个：**

1. `tt-media-server/domain/image_generate_request.py`：T2I 主请求契约，定义 `prompt`、`width`、`height`、`num_inference_steps`、`seed`、`number_of_images`、参考图和输出格式等字段及校验。
2. `tt-media-server/domain/image_edit_request.py`：图像编辑请求契约，负责编辑类请求字段的结构化定义。
3. `tt-media-server/domain/image_to_image_request.py`：I2I 请求契约，负责参考图字段归一化，并转换为 FLUX3 统一请求输入。
4. `tt-media-server/domain/base_request.py`：所有图片请求共享的基础字段、公共校验和继承基类。
5. `tt-media-server/open_ai_api/models.py`：对外 OpenAI 兼容 schema、模型描述和 API 契约暴露。

其中，`image_generate_request.py` 是 FLUX3 图片生成 API 的**主契约入口**；路由文件负责接收请求和调度，不负责重新定义完整的请求字段。

| 文件 | 责任 |
|---|---|
| `tt-media-server/main.py` | FastAPI/uvicorn 服务入口和路由装载 |
| `tt-media-server/open_ai_api/image.py` | `/v1/images/generations` 同步/异步生成、任务查询、取消和响应封装 |
| `tt-media-server/open_ai_api/models.py` | `/v1/models` 模型描述和请求 schema 暴露 |
| `tt-media-server/domain/base_request.py` | 通用请求基类、公共字段和校验基础 |
| `tt-media-server/domain/image_generate_request.py` | prompt、宽高、steps、seed、`number_of_images`、`reference_image(s)`、输出格式校验 |
| `tt-media-server/domain/image_edit_request.py` | 图像编辑兼容请求结构 |
| `tt-media-server/domain/image_to_image_request.py` | I2I 兼容字段归一化；参考图最终进入 FLUX3 统一请求 |
| `tt-media-server/security/api_key_checker.py` | `Authorization: Bearer ...` 校验；部署变量为 `OPENAI_API_KEY` |

#### FLUX3 配置、设备发现和拓扑

| 文件 | 责任 |
|---|---|
| `tt-media-server/config/flux3_profiles.py` | profile 唯一事实来源：mesh、TP/SP、Ring/Linear、core grid、VAE 并行、worker group 规则 |
| `tt-media-server/config/settings.py` | 环境变量解析、profile 选择、设备 mesh、parent/child mesh、缓存目录和启动策略 |
| `tt-media-server/config/constants.py` | 服务公共默认值和配置常量 |
| `tt-media-server/utils/runner_utils.py` | Fabric、core grid、cache、设备 ID 和 mesh 参数落地 |
| `tt-media-server/tt_model_runners/runner_fabric.py` | runner 级 Fabric/mesh 初始化辅助逻辑 |
| `tt-media-server/generated/fabric/*.yaml` | 生成的 Fabric 拓扑映射；只作为部署生成物，不作为 profile 逻辑的替代品 |

#### 服务、队列、worker 和生命周期

| 文件 | 责任 |
|---|---|
| `tt-media-server/model_services/image_service.py` | 图像服务实例、请求入队、worker 选择和结果回收 |
| `tt-media-server/model_services/scheduler.py` | worker group 拆分、固定 10 秒启动间隔、ready 状态、健康检查和失败重启 |
| `tt-media-server/model_services/queues/tt_queue.py` | 基础请求队列接口 |
| `tt-media-server/model_services/queues/tt_batch_fifo_queue.py` | FIFO/批处理队列实现；当前 FLUX3 输出仍固定单图 `batch=1` |
| `tt-media-server/model_services/queues/tt_faster_fifo_queue.py` | 高吞吐队列实现 |
| `tt-media-server/device_workers/device_worker.py` | 单 worker 进程、请求执行、runner 调用和 ready 信号 |
| `tt-media-server/device_workers/worker_utils.py` | worker 参数、设备组和进程辅助函数 |
| `tt-media-server/device_workers/encode_stage.py` | 图像输入编码阶段的 worker 辅助逻辑 |
| `tt-media-server/health_monitoring/canary_monitor.py` | canary、worker 存活和服务健康状态 |
| `tt-media-server/telemetry/image_metrics.py` | 图像阶段和请求指标 |
| `tt-media-server/telemetry/image_request_metrics.py` | HTTP 请求级指标 |
| `tt-media-server/telemetry/telemetry_client.py` | 指标上报和生命周期记录 |

#### Runner 和设备基础层

| 文件 | 责任 |
|---|---|
| `tt-media-server/tt_model_runners/dit_runners.py` | `TTFlux3ImageKleinRunner`：创建 pipeline、加载权重、warmup、T2I/I2I 调用和阶段计时 |
| `tt-media-server/tt_model_runners/base_device_runner.py` | 通用 runner 接口和生命周期约定 |
| `tt-media-server/tt_model_runners/base_metal_device_runner.py` | MeshDevice/Fabric 打开、设备关闭、异常清理和基础 Metal 生命周期 |
| `tt-media-server/tt_model_runners/sp_runner.py` | 序列/空间并行 runner 的公共能力 |
| `tt-media-server/utils/device_manager.py` | 设备状态和设备资源辅助管理 |

#### 服务端回归和性能测试

| 文件 | 用途 |
|---|---|
| `tt-media-server/tests/test_flux3_device_config.py` | profile、mesh、worker group 和设备配置回归 |
| `tt-media-server/tests/test_flux3_reference_api.py` | 单/多参考图、非法输入和 422/500 防回归 |
| `tt-media-server/tests/test_flux3_multipart_api.py` | 同步/异步 JSON 与 multipart 请求、文件顺序、别名和上传错误码回归 |
| `tt-media-server/tests/test_dit_runners_validation.py` | DiT runner 的请求到 runner 验证 |
| `tt-media-server/tests/test_runner_fabric.py` | Fabric/mesh 初始化回归 |
| `tt-media-server/tests/test_scheduler.py` | worker 启动、ready、重启和调度回归 |
| `tt-media-server/performance_tests/test_image_generation.py` | 图像 API 性能测试入口 |
| `TL-inference-server/Dockerfile` | FLUX3 镜像构建入口；通过 `--build-context ttmetal` 引入 tt-metal |

### 2.2 tt-metal

以下是当前 FLUX3 Image Klein 在 tt-metal 中涉及的文件和资源。FLUX3 的底层 matmul、conv、通信和 TTNN runtime 复用 tt-metal 公共实现，因此不把所有通用算子文件复制成“FLUX3 专属文件”；这里同时列出 FLUX3 专属层和实际影响性能/拓扑的公共层。

#### FLUX3 Klein pipeline 和模型组件

| 文件 | 责任 |
|---|---|
| `models/tt_dit/pipelines/flux3_klein/pipeline_flux3_klein.py` | 顶层 T2I/I2I 编排、geometry、prompt/reference 输入、warmup、阶段计时和输出回读 |
| `models/tt_dit/pipelines/flux3_klein/prompt_encoder.py` | Qwen3-VL tokenizer、chat template、文本 embedding 和 token 对齐 |
| `models/tt_dit/pipelines/flux3_klein/vae_flux3_klein.py` | VAE decoder、Conv schedule、`act_block_h`/double buffer、空间并行和图像回读 |
| `models/tt_dit/pipelines/flux3_klein/weights_flux3_klein.py` | checkpoint 名称、权重目录、转换后的权重 key 和 cache key |
| `models/tt_dit/pipelines/flux3_klein/__init__.py` | Klein pipeline 模块导出和注册 |

#### Transformer、attention 和采样

| 文件 | 责任 |
|---|---|
| `models/tt_dit/models/transformers/transformer_flux3.py` | FLUX3 Transformer block、joint/dual stream、RoPE、QKV、投影和 TP/SP 执行 |
| `models/tt_dit/blocks/attention_opt.py` | attention、Ring/序列并行、分片 token 和通信相关优化路径 |
| `models/tt_dit/blocks/*.py` 中被 `transformer_flux3.py` 导入的 block 文件 | RMSNorm、MLP、modulation、attention block 和 residual 组合 |
| `models/tt_dit/utils/matmul.py` | core grid 下的 matmul blocking 选择、profile sweep 和 fallback 日志 |
| `models/tt_dit/utils/progress.py` | 权重加载、warmup 和阶段进度记录 |
| `models/tt_dit/utils/cache.py` | 权重 tensor cache、resident cache 和 cache key 复用 |

#### TTNN/Metal 拓扑和 Fabric 资源

| 文件/目录 | 责任 |
|---|---|
| `tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_flux3_sp4_ring.textproto` | SP4 物理 Ring mesh descriptor |
| `tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_flux3_sp8_ring.textproto` | SP8 物理 Ring mesh descriptor |
| `tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto` | Galaxy 物理分组和可放置性描述 |
| `tt_metal/fabric/*` | Fabric 设备发现、Ring/Linear/2D mesh 和跨 mesh 通信实现 |
| `ttnn/operations/*` | matmul、conv、reshape、untilize、attention 等公共 TTNN 算子；FLUX3 通过 pipeline/Transformer/VAE 调用 |
| `tt_metal/impl/*` | Metal runtime、kernel 编译、设备队列、同步和缓存基础设施 |

#### 适配验证和运行时产物

| 文件/目录 | 用途 |
|---|---|
| `tests/tt_metal/tt_fabric/*` | Fabric descriptor、mesh placement 和通信回归 |
| `/local/ttcache/tt_dit_cache` | text encoder、Transformer、VAE 转换权重和 tensor cache |
| `/local/ttcache/tt-metal-cache` | Metal kernel/program cache |
| `/local/ttcache/hf` | Hugging Face/tokenizer cache |

本节列出的是当前 v1.0.35 适配实际使用的文件和资源；通用 TTNN/Metal 算子由 pipeline、Transformer、VAE 和 Fabric 层调用，不单独复制为 FLUX3 专属文件。

## 3. API 契约到 runner 的执行路径

### 3.1 请求进入 API

推荐入口：

```text
POST /v1/images/generations
```

API 层只负责公共契约，不负责决定 TTNN 的 tensor layout：

```text
model
prompt
width / height
num_inference_steps
seed
number_of_images
response_format
image_return_format
reference_image 或 reference_images
```

处理顺序为：

```text
Bearer 校验
  -> JSON 或 multipart/form-data 请求解析
  -> multipart 文件转 data:image/...;base64,...
  -> Pydantic 字段校验
  -> FLUX3 geometry 校验
  -> Base64/data URL 参考图校验
  -> 放入 service queue
  -> 调度到 ready worker
```

服务层不直接在 HTTP handler 中加载模型。模型由 runner 在 worker warmup 时创建并常驻。

### 3.1 同步/异步 API 的输入适配

同步 `POST /v1/images/generations` 和异步
`POST /v1/images/generations/async` 共用
`open_ai_api/image.py::_parse_image_generate_request()`，因此两条接口的请求契约保持一致：

| 输入方式 | 支持内容 | 进入 pipeline 前的形式 |
|---|---|---|
| `application/json` | prompt、尺寸、steps、seed、Base64/data URL 参考图 | `ImageGenerateRequest` |
| `multipart/form-data` | 普通字段 + `reference_images` 文件字段 | 文件读取后转换为 data URL，再生成 `ImageGenerateRequest` |

multipart 规则：

- 重复的 `reference_images` 文件字段按出现顺序保留，最多 10 张；
- 单文件字段 `reference_image` 和 `image` 作为兼容别名；
- 文件必须声明为 `image/*`，不能为空，并默认限制为 32 MiB/文件；
- 普通表单字段会经过 Pydantic 类型转换和 FLUX3 尺寸/步数校验；
- 非法图片或字段返回 `422`，超过上传大小返回 `413`，缺少 multipart 解析依赖返回 `415`；
- 同步接口可以用 `response_format=file` 直接返回 JPEG/PNG；异步接口提交后返回任务 ID，结果通过任务查询接口返回 JSON/Base64；
- JSON/Base64 旧接口保持兼容，multipart 只改变传输方式，不改变 runner 的参考图 token、去噪和 VAE 流程。

请求链路因此统一为：

```text
客户端 JSON 或 multipart 文件
  -> API key 校验
  -> 请求解析
  -> 文件转 data URL（仅 multipart）
  -> ImageGenerateRequest
  -> 同步 process_request 或异步 create_image_job
  -> worker queue
  -> FLUX3 runner
```

### 3.2 Worker 和 runner 分工

```text
Scheduler
  -> 按 profile 发现 worker group
  -> 启动 worker 进程
  -> 监控 is_alive / is_ready

Device worker
  -> 创建 TTFlux3ImageKleinRunner
  -> 打开 MeshDevice
  -> load/cache pipeline
  -> warmup
  -> 从 queue 取请求并调用 runner.run()

Runner
  -> 将 API request 转为 Flux3KleinRequest
  -> 调用 pipeline
  -> 记录 text/reference/denoise/VAE 阶段
  -> 返回图像和 timings
```

这三个层次不能合并：scheduler 管进程，worker 管生命周期，runner 管模型执行。

## 4. 请求几何和官方尺寸规则

### 4.1 输出图

当前 FLUX3 API 规则：

- `width`、`height` 必须同时提供，或同时省略；
- 每边 `16～4096`；
- 每边必须是 16 的倍数；
- 不额外限制总像素数或宽高比；
- 当前 FLUX3 一次请求只生成 1 张图片，等价于 `batch=1`；多图 batch 暂不支持。
- `num_inference_steps` 为 4～50。

因此 1024×1024、1344×768、768×1344、2048×1024 和 4096×4096 在 API 几何层面都符合规则。能否在特定 profile 上稳定运行，还取决于该 profile 的 VAE L1 和 warmup 能力；API 合法不等于每个硬件布局都已经完成性能回归。

### 4.2 单参考图

单张参考图按输入图片的原始宽高比进行动态预处理，不统一缩放为固定尺寸：

```text
输入图片
  -> EXIF 方向修正
  -> RGB
  -> 保持宽高比
  -> 受 4096 单边和约 4 Mi-pixel 限制
  -> 16 像素网格对齐/必要时 padding
  -> reference VAE encoder
  -> image-condition token
```

当前代码常量：

```text
REFERENCE_MIN_SIDE          = 64
REFERENCE_MAX_SIDE          = 4096
REFERENCE_MAX_PIXELS        = 4 * 1024 * 1024
REFERENCE_DOWNSAMPLING_RATIO = 16
```

例如，4096×4096 参考图受约 4MP 上限影响，会预处理为约 2048×2048 画布；1344×768 会保留为 1344×768。

### 4.3 多参考图

- 字段：`reference_images`；
- API 契约最多允许 10 张；当前 FLUX3 runner 会将 `reference_images` 解码成有序图片集合并传入 pipeline。单次请求仍只生成 1 张输出图，即 `batch=1`。
- SP4/SP8 真机回归：SP4 的 2 张、5 张已返回 HTTP 200；SP8 的 2 张、5 张和 10 张（10 张 512×512）已返回 HTTP 200 和有效 JPEG。混合 4096×4096 参考图时，主要受请求体上传和同步客户端超时影响，不能把客户端超时等同于 runner 拒绝。
- 每张参考图按约 1MP 预算处理；
- 保留数组顺序；
- 只接受 Base64 或 `data:image/...;base64,...`，不接受 HTTP URL；
- 多参考图不能同时与 `reference_image` 或 `image` 字段使用。

多参考图请求校验在进入 worker 前完成。校验器必须导入 `math`，因为它会根据输入尺寸计算多参考图的最小缩放比例；缺少该导入会直接造成 HTTP 500，而不是返回正常的 4xx 参数错误。该问题已包含在当前 v1.0.35 镜像中，并由参考图 API 回归覆盖。

参考图送给 Transformer 的不是原始图片，而是：

```text
RGB image
  -> VAE encoder
  -> latent
  -> patch/token packing
  -> image-condition token + image IDs
  -> Transformer
```

## 5. Profile、mesh 和通信

`config/flux3_profiles.py` 是 profile 的唯一事实来源。mesh 形状描述逻辑排列；物理芯片选择由 `discovery_kind` 和 Fabric/descriptor 决定，不能简单用连续设备号代替。`tp2linear` 不再是当前 canonical profile；纯 TP2 统一使用 `tp2`。

| Profile | worker mesh | parent mesh | TP | SP | 通信/拓扑 | VAE 并行 | direct worker 数 | parent/child child 数 |
|---|---:|---:|---:|---:|---|---|---:|
| `legacy-single` | 1×1 | 无 | 1 | 1 | 无 | replicated | 32 | — |
| `sp2` | 2×1 | 4×8 | 1 | 2 | 1D Ring | TP axis | 16 | 16 |
| `tp2` | 2×1 | 4×8 | 2 | 1 | 1D Linear | replicated | 16 | 16 |
| `sp4` | 4×1 | 4×8 | 1 | 4 | 1D Ring | TP axis | 8 | 8 |
| `tp4` | 4×1 | 4×8 | 4 | 1 | 1D Ring | TP axis | 8 | 8 |
| `tp4axis1` | 1×4 | 4×8 | 4 | 1 | 1D Ring | TP axis | 8 | 8 |
| `tp2sp2` | 2×2 | 8×4 | 2 | 2 | 2D Linear | TP axis | 8 | 8 |
| `tp2sp4` | 2×4 | 8×4 | 2 | 4 | 2D Linear | TP axis | 4 | 4 |
| `tp4sp2` | 4×2 | 8×4 | 4 | 2 | 2D Linear | TP axis | 4 | 4 |
| `sp8` | 8×1 | 8×4 | 1 | 8 | 1D Ring | TP axis | 4 | 4 |
| `tp8ring` | 1×8 | 4×8 | 8 | 1 | 1D Ring | TP axis | 4 | 4 |
| `sp8tp4` | 4×8 | 无 | 4 | 8 | 全 Galaxy Ring | TP axis | 1 | — |

### 5.1 direct 与 parent/child

默认生产路径是 direct：每个 worker 独立打开一个物理上合法的设备 group，因此 32 卡 Galaxy 可以拆分为 32、16、8 或 4 个独立 worker。

#### parent/child 的多 worker 流程

```text
完整 parent mesh
        ↓
按 profile 的 child shape 切分
        ↓
生成多个互不重叠的 child mesh
        ↓
worker manager 分配 child index
        ↓
每个 worker 只使用自己的 child mesh
```

`FLUX3_PARENT_MESH_WORKERS=N` 表示启用的 worker 数量。调度器为 worker 分配唯一的 `FLUX3_SUBMESH_INDEX`，编号从 `0` 开始。可用 worker 数量不能超过 parent mesh 能切出的 child 数量。

| 模式 | 设备组织方式 | 典型用途 |
|---|---|---|
| `direct` | 每个 worker 直接获得一组合法物理设备 | 默认生产部署和多 worker 服务 |
| `parent/child` | 一个 parent mesh 切分为多个 child mesh，再分配给 worker | 复用完整 Galaxy 拓扑的多 worker 部署 |

两种模式都可以支持多 worker。区别在于：`direct` 直接分配物理 worker group；`parent/child` 先建立完整 parent，再按 profile 分配 child。两者都必须使用合法 Fabric descriptor，并保证每个 worker 的 child/group 不重复。

不要手工构造与物理拓扑不匹配的 GPU 列表，也不要让两个 worker 使用同一个 `FLUX3_SUBMESH_INDEX`。否则会导致 Fabric placement、设备所有权或通信初始化失败。

### 5.2 自动发现

生产启动只推荐设置：

```text
FLUX3_EXECUTION_PROFILE=<profile>
```

服务自动根据 profile 选择 worker mesh、物理分组和 worker 数。`FLUX3_DEVICE_IDS` 对 FLUX3 不作为生产覆盖项；不要手写连续卡号伪造 Ring 或 Galaxy 子网格。

兼容变量 `FLUX3_TP_SIZE`、`FLUX3_TP_MESH`、`FLUX3_PARALLELISM` 由 profile 派生或仅用于旧配置兼容，不应与 profile 同时手工定义不同语义。

## 6. Worker 启动、warmup 和 resident pipeline

worker 启动顺序：

```text
发现设备
  -> 打开 MeshDevice
  -> 应用 core grid / Fabric
  -> 加载 Text Encoder、Transformer、VAE
  -> 加载或生成权重 cache
  -> warmup 代表分辨率
  -> worker ready
  -> 接收请求
```

warmup 负责提前触发权重转换、TTNN program、Metal kernel、CCL/Fabric、VAE Conv 和典型 geometry 分配。warmup 尺寸是启动编译矩阵，不是 API 白名单；运行时仍按请求 geometry 执行。

启动 warmup 的默认尺寸按 profile 和实际 worker mesh 选择；`FLUX3_WARMUP_SIZES` 只覆盖启动编译集合，不会改变 API 的官方几何校验：

```text
legacy-single / tp2 / sp4 / tp4 / tp4axis1 / tp4sp2: 1024
sp2 / sp8:                             1024（sp8 可扩展到 2048）
tp2sp4:                                1024、2048
tp2sp2 / sp8tp4:                       1024、2048、4096（按 runner warmup 路径）
FLUX3_WARMUP_SIZES=...                 # 显式覆盖启动编译集合
FLUX3_CORE_GRID_OVERRIDE=12x10
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

某些 pipeline 创建路径会保留更大的 program-cache 尺寸集合，而 runner 的实际 warmup 会根据 mesh 再收窄集合，以避免 4K VAE 在小型子网格上提前占满 L1。这个“创建 pipeline 的尺寸集合”和“启动时真正执行的 warmup 集合”是两个不同概念。

多 worker 默认按固定 10 秒 start-to-start 间隔启动；`sp8` 可用 `FLUX3_SP8_STARTUP_MODE=ready` 切换为等待前一个 worker ready 后再启动下一个。默认 fixed 模式不会等待前一个 worker ready。

模型组件采用 resident 模式：权重和 pipeline 不在每个请求中重复装载。首次 warmup、首次新 geometry 和首次 kernel 编译时间不得混入热态性能。

## 7. Transformer 去噪实现

Transformer 去噪阶段的输入不是一张图片，而是已经准备好的多路条件和一个噪声 latent。`Flux3KleinPipeline` 负责组织输入和采样循环，`transformer_flux3.py` 负责每一步的 DiT 计算，设备 mesh、TP/SP 和 Fabric 只影响这些张量如何分片与通信，不改变模型的数学语义。

### 7.1 输入条件准备

```text
文本 prompt
  -> tokenizer / chat template
  -> Qwen3-VL text encoder
  -> text hidden states、text mask、text IDs

参考图（可选，T2I 没有）
  -> EXIF/RGB/尺寸校验
  -> VAE latent
  -> patchify / token packing
  -> reference image-condition token、image IDs

输出 geometry + seed
  -> 初始化 noise latent
  -> patchify
  -> image token、image IDs
```

参考图传给 Transformer 的不是原始 RGB 图片，也不是 VAE 的四维 latent 原样传入，而是经过 VAE encoder 和 token packing 后的图像条件 token。多张参考图按请求顺序分别编码，再按 pipeline 约定拼接或打包，并保留对应的 image IDs。

### 7.2 token 和 geometry

输出宽高决定 latent 的空间尺寸，latent 空间尺寸再决定 image token 数量。文本 token 数由 prompt 决定；参考图 token 数由参考图预处理后的 geometry 决定。因此一次请求的 Transformer 输入长度通常是：

```text
总条件长度 = text token 数 + reference token 数（可选）+ image token 数
```

pipeline 同时生成与 token 对齐的 ID/位置元数据，用于 RoPE 和 attention 的位置计算。改变宽高、参考图数量或参考图画布尺寸，都会改变 token 数、attention 工作量、program cache key 和后续 VAE geometry。

### 7.3 单步 Transformer 计算

每一个 denoise step 都执行一次完整的 FLUX3 Transformer。逻辑可以简化为：

```text
image tokens
text tokens
reference tokens（可选）
    ↓
拼接/分流 + token IDs
    ↓
time / conditioning modulation
    ↓
RoPE
    ↓
QKV projection
    ↓
attention
    ↓
output projection
    ↓
MLP / gated feed-forward
    ↓
residual + norm
    ↓
image token velocity
```

实际 block 会在 joint/dual-stream 结构之间切换：文本和图像条件在指定层进行联合 attention，图像流继续经过 modulation、attention、MLP 和 residual；最后只取图像流对应的输出，形成当前 latent 的 velocity/更新量。

### 7.4 TP、SP 与通信位置

并行 profile 不改变上述计算顺序，只改变每个张量的分片方式：

| 并行方式 | 主要分片对象 | 典型通信 |
|---|---|---|
| SP | image token/sequence 维度 | Ring 或 sequence collective |
| TP | hidden/channel 和线性层权重 | linear、all-gather、reduce-scatter 或 profile 指定 collective |
| TP×SP | 分别沿 hidden 与 token 维度分片 | 两个轴按 profile 组合通信 |
| parent/child mesh | 设备的物理组织 | parent 负责合法 placement，child 提供 runner 使用的逻辑 mesh |

例如 `sp4` 主要沿 image token 维度切分；`tp4` 主要沿线性层权重/hidden 维度切分；`tp2sp4` 同时存在 TP 轴和 SP 轴。通信轴由 `flux3_profiles.py` 的 profile 定义，不能只根据 mesh 的行列数猜测。

### 7.5 flow denoise 采样循环

```text
seed
  -> 初始 noise latent
  -> scheduler 生成 timestep/sigma schedule

for step in range(num_inference_steps):
    velocity = transformer(
        latent,
        text_condition,
        reference_condition,
        token_ids,
        timestep,
    )
    latent = flow_update(latent, velocity, schedule[step])
```

`num_inference_steps` 直接决定 Transformer 调用次数；16 步请求通常对应 16 次主要 denoise step。`seed` 决定初始 noise，并用于固定性能和结果回归条件。每一步都必须在参与计算的设备上保持相同的 timestep、latent layout 和条件输入，不能让某个 worker 单独推进 schedule。

pipeline 在设备侧 resident 保存 Transformer 权重、已编译 program、layout 和可复用的 schedule 数据。每一步只更新 latent 和当前 timestep，不重新加载权重、不重新创建整套 pipeline。首次请求可能包含 kernel 编译和 program cache 建立，因此不能直接作为热态 denoise 性能。

### 7.6 T2I、I2I 和阶段边界

```text
T2I：prompt -> text condition -> denoise -> VAE decoder

I2I：prompt + reference_images
       -> text condition + reference encode
       -> denoise -> VAE decoder
```

参考图编码属于 `encode_reference` 阶段，不属于 Transformer 的 denoise 时间。Transformer 只接收编码后的 reference token；VAE decoder 只处理最终 denoised latent，不会再次读取原始参考图。

一次请求的阶段边界如下：

```text
请求解析/校验
  -> text encode
  -> encode_reference（I2I/多参考图才有）
  -> denoise（每 step 调用 Transformer）
  -> VAE decode
  -> device-to-host / JPEG 或 PNG
  -> HTTP 返回
```

因此 HTTP E2E 不能简单等同于 denoise 加 VAE decode；队列、调度、文本编码、参考图编码、主机回读、图片编码和网络返回都可能包含在 HTTP 时间内。

## 8. 权重转换和 cache

权重转换链路：

```text
safetensors/checkpoint
  -> source state_dict
  -> FLUX3 命名解析
  -> QKV / projection / FFN layout 转换
  -> TTNN dtype/layout
  -> 按 model/profile/mesh 写入 cache
  -> 后续 load_model() 直接加载 cache
```

cache 的逻辑边界必须包含模型变体、dtype、TP/SP、mesh、VAE 并行方式和 geometry。不同 profile 的权重 cache 不能盲目复用，因为同一权重在不同 mesh 上的 tensor layout 和 shard 方式不同。

宿主机推荐目录：

```text
/local/models/flux-3-image-klein        模型权重，只读挂载
/local/ttcache/flux-3-image-klein       cache、HF、JIT、临时文件
```

容器内统一使用 `/local/ttcache`。

## 9. VAE decode 和 L1 约束

```text
denoised latent
  -> inverse scaling / unpatchify
  -> VAE Conv/ResNet/up-block
  -> untilize / device-to-host
  -> JPEG/PNG
```

VAE 的性能和可运行性受空间分片、Conv geometry、每核 L1、`act_block_h`、double buffer、core grid 和回读方式共同影响。

`FLUX3_VAE_ACT_BLOCK_H=64` 不是所有 geometry 都合法。若当前 Conv 的 `padded_output_height_ntiles_per_core` 不允许 64，TT-Metal 会选择最近合法值，常见日志为：

```text
act_block_h_override 64 is not a valid override ... instead 32 was selected
```

这表示算子回退到合法的 32，并不表示 API 失败。常规 1024/2048 请求保持已验证的 `64 / true` 调度；对于特定 SP8 geometry，64/true 仍由 TTNN 按层回退到合法的 32。4 卡及以下不预热、也不接受 `4096×4096`，避免在不具备足够 L1 的小 worker 上进入该 VAE geometry。

4096×4096 是否成功不是 API 校验单独能决定的：Transformer 可以接受动态 token geometry，但 VAE decoder 仍必须满足 profile 对应的 L1 和 Conv buffer 规划。

## 10. 输出、计时和性能口径

一次请求应区分：

```text
text encode
reference encode       # T2I 为 0
denoise
VAE decode
image postprocess      # host readback、JPEG/PNG 编码等
HTTP E2E               # 队列、调度、上述阶段和响应传输总和
```

因此：

```text
HTTP E2E 不等于 denoise + VAE decode
```

正式性能测试必须固定：同步串行请求、分辨率、16 steps、seed、JPEG/PNG 格式、warmup 状态、`TT_MM_THROTTLE_PERF=0` 和 `TT_METAL_TDP_LIMIT_WATTS=160`。首次编译和队列等待要单独标注。

## 11. FLUX3 的工程分层与模块对应

| 工程层 | 当前 FLUX3 实现 | 主要职责 |
|---|---|---|
| API/鉴权/队列 | 通用 media-server 接口层 | 接收请求、校验参数、鉴权、入队和返回结果 |
| FLUX3 runner | `TTFlux3ImageKleinRunner` | 管理模型加载、warmup、请求转换和阶段计时 |
| 文本条件 | Qwen3-VL prompt encoder | tokenizer、chat template、文本 token 和 embedding |
| 图像条件 | reference image encoder | 将参考图编码为 pipeline 使用的图像条件 token/向量 |
| DiT | FLUX3 Transformer | joint/dual stream、RoPE、attention、MLP、投影和去噪 |
| VAE | FLUX3 VAE encoder/decoder | latent 与图像之间的转换、Conv 调度、空间并行和回读 |
| 并行配置 | FLUX3 canonical profile 表 | 统一 mesh、TP/SP、Ring/Linear、VAE parallel 和 worker discovery |
| Galaxy 设备 | Fabric descriptor + MeshDevice | 设备发现、物理分组、Ring/2D 通信和 parent/child mesh |
| 输出处理 | ImagePostprocessing + image codec | tensor 回主机、JPEG/PNG 编码和 HTTP file/Base64 返回 |

FLUX3 的模型结构、token 组织、RoPE、Transformer layout、VAE schedule 和权重映射都由 FLUX3 专属模块定义；通用部分只复用服务框架、worker 生命周期、队列、cache、设备管理和指标体系。

## 12. 新模型适配时应复用的步骤

1. 定义 API 契约、尺寸、步数、参考输入和输出格式。
2. 为模型建立独立 runner，不把模型逻辑写进通用 API handler。
3. 将 tokenizer/text encoder、主模型、VAE 拆成可独立 warmup 和测试的组件。
4. 定义 source checkpoint 到 TTNN 权重的命名和 layout 映射。
5. 建立 canonical profile 表，统一 mesh、TP/SP、Fabric、VAE parallel 和 discovery。
6. 先完成单卡/单 worker，再增加多卡、Ring、2D 和 parent/child。
7. 为每个 geometry 分别检查 L1、Conv、attention、core grid 和 cache key。
8. 将 warmup、resident、kernel/program cache 与请求路径分开验证。
9. 为 T2I、I2I、多参考图、JPEG/PNG 和错误输入建立 API 回归。
10. 在固定 TDP、同步请求、固定 seed 和固定输出格式下记录阶段时间及 HTTP E2E。
