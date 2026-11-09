# FLUX.3 Image Klein：一个新请求到达 TT Media Server Runner 前后的完整流程

## 1. 范围与结论

本文基于 2026-11-09 的 `TL-inference-server` 分支 `flux-3-image-klein` 源码，说明一个 FLUX.3 Image Klein 图片请求如何从 HTTP API 进入 `tt-media-server`，经过 Service、Scheduler 和 device worker，最后到达 `TTFlux3ImageKleinRunner`。

本文的终点是 runner 调用 `pipeline.run(...)`；TT-Metal 内部的 Qwen3-VL prompt encoder、TTNN Transformer、Euler denoise、VAE decode 属于下游 pipeline 实现，不在本文展开。

两个容易混淆的事实：

1. **runner 不会为每个新请求重新创建。** 服务启动时，Scheduler 为每个 worker 创建一个 runner，完成设备打开与模型 warmup；之后新请求只会被排队并送到已经就绪的 worker。
2. **当前 FLUX.3 的默认路径是普通 worker，而不是 dynamic-batch worker。** `Settings.use_dynamic_batcher` 默认值为 `false`，FLUX.3 的配置也没有将其改为 `true`。因此默认走 `device_workers/device_worker.py`。只有显式设置 `USE_DYNAMIC_BATCHER=true`（或其他配置使其为真）时，才会走 `device_worker_dynamic_batch.py`。

## 2. 两条必须分开的链路

```text
服务启动链路（一次/每个 worker）
FastAPI service 初始化
  → ImageService / Scheduler
  → Scheduler 启动 worker 进程
  → initialize_device_worker(worker_id)
  → get_device_runner(worker_id)
  → TTFlux3ImageKleinRunner(worker_id)
  → set_device() + warmup()
  → worker ready

单个请求链路（每次请求）
POST /v1/images/generations
  → ImageGenerateRequest 校验
  → ImageService.process_request()
  → Scheduler.task_queue
  → 一个空闲 worker 领取请求
  → TTFlux3ImageKleinRunner.run([request])
  → Flux3T2IPipeline.run(...)
  → result_queue
  → ImageService 后处理
  → HTTP JSON/Base64 或 image file 响应
```

## 3. 服务启动：runner 在请求之前如何建立

### 3.1 路由与 Service 单例

| 顺序 | 文件 | 关键对象/函数 | 功能 |
|---|---|---|---|
| 1 | `tt-media-server/open_ai_api/__init__.py` | `register_service_routes()` | 根据 `settings.model_service` 注册路由；FLUX.3 的值为 `image`，所以注册 `/v1/images`。 |
| 2 | `tt-media-server/resolver/service_resolver.py` | `service_resolver()` | 用 `settings.model_service` 选择 Service 单例。`image` 对应 `ImageService()`。 |
| 3 | `tt-media-server/model_services/image_service.py` | `ImageService.__init__()` | 初始化图片后处理 CPU worker，并继承 `BaseService`。 |
| 4 | `tt-media-server/model_services/base_service.py` | `BaseService.__init__()` | 调用 `get_scheduler()` 获取 Scheduler 单例。 |
| 5 | `tt-media-server/resolver/scheduler_resolver.py` | `get_scheduler()` | 以线程锁保证进程内只有一个 `Scheduler`。 |

### 3.2 Scheduler 创建 32 个单芯片 worker

| 顺序 | 文件 | 关键对象/函数 | 功能 |
|---|---|---|---|
| 6 | `tt-media-server/model_services/scheduler.py` | `Scheduler.__init__()` | 创建任务队列、结果队列、错误队列、warmup 信号队列和取消队列。 |
| 7 | 同上 | `Scheduler.get_worker_count()` / `_calculate_worker_count()` | 按 `settings.device_ids` 拆分 worker ID；FLUX.3 Galaxy 配置的 mesh 是 `(1, 1)`，每个 worker 使用一个芯片。 |
| 8 | 同上 | `Scheduler.start_workers()` | 启动结果监听、warmup 监听、错误监听，然后依次启动 worker。 |
| 9 | 同上 | `Scheduler._start_worker()` | 用 `multiprocessing.Process` 创建 worker 子进程。默认 `use_dynamic_batcher=false` 时，target 是 `device_workers.device_worker.device_worker`。 |

FLUX.3 的 Galaxy 配置位于：

```text
tt-media-server/config/constants.py
MODEL_RUNNER = tt-flux.3-image-klein
device_mesh_shape = (1, 1)
device_ids = DEVICE_IDS_32
max_batch_size = 1
request_processing_timeout_seconds = 3000
```

这表示 32-worker 部署是 **32 个独立的单芯片 runner**，不是一个跨 32 芯片的 FLUX.3 mesh。

### 3.3 worker 创建 runner、打开芯片、加载模型

| 顺序 | 文件 | 关键对象/函数 | 功能 |
|---|---|---|---|
| 10 | `tt-media-server/device_workers/device_worker.py` | `device_worker()` | 默认 worker 子进程入口；首先调用 `initialize_device_worker()`。 |
| 11 | `tt-media-server/device_workers/worker_utils.py` | `initialize_device_worker(worker_id, logger)` | 创建 event loop，取得 runner，打开 TTNN device，并等待模型 warmup。 |
| 12 | `tt-media-server/tt_model_runners/runner_fabric.py` | `get_device_runner(worker_id)` | 读取 `settings.model_runner`，从 `AVAILABLE_RUNNERS` 找到 runner factory。 |
| 13 | 同上 | `AVAILABLE_RUNNERS[ModelRunners.TT_FLUX_3_IMAGE_KLEIN]` | 动态导入 `tt_model_runners.dit_runners`，创建 `TTFlux3ImageKleinRunner(worker_id)`。 |
| 14 | `tt-media-server/tt_model_runners/base_metal_device_runner.py` | `BaseMetalDeviceRunner.set_device()` | 使用 `ttnn.open_mesh_device(...)` 打开该 worker 对应的 `(1,1)` TTNN mesh。 |
| 15 | `tt-media-server/tt_model_runners/dit_runners.py` | `TTDiTRunner.warmup()` | 在线程中创建 pipeline，完成权重加载，然后提交一个 2-step warmup 请求以触发必要的首次准备。 |
| 16 | 同上 | `TTFlux3ImageKleinRunner.create_pipeline()` | 建立 `Flux3T2IPipeline`；默认 resident 模式加载 Qwen prompt encoder、Transformer 和 VAE decoder。 |

FLUX.3 runner 的模型目录优先级是：

```text
FLUX3_MODEL_ROOT（最高优先级）
  → settings.model_weights_path
      （若它是默认的 Distilled 名称且 /local/models/flux-3-image-klein 已挂载，则解析为此绝对目录）
```

当前默认 checkpoint 是 Distilled：

```text
SupportedModels.FLUX_3_IMAGE_KLEIN = "flux-3-image-klein"
/local/models/flux-3-image-klein
```

Base 必须显式选择，例如：

```bash
-e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein-base
```

### 3.4 worker ready

warmup 成功后，`device_worker()` 将 `worker_id` 写入 `warmup_signals_queue`。

`Scheduler.device_warmup_listener()` 收到信号后：

1. 将该 worker 标记为 `is_ready=true`；
2. 第一张芯片 ready 后将服务标记为可用；
3. 启动 worker health monitor 与 canary monitor；
4. 全部 worker ready 后记录全员就绪日志。

因此，请求到来之前，runner、TTNN device、主要模型权重和 warmup 状态通常已经存在于每个 worker 中。

## 4. 同步图片请求：从 HTTP 到 runner

以下针对标准同步接口：

```text
POST /v1/images/generations
```

### 步骤 1：FastAPI 路由与请求体校验

文件：`tt-media-server/open_ai_api/image.py`

入口函数：

```python
generate_image(image_generate_request, service=Depends(service_resolver), ...)
```

FastAPI 会先用 Pydantic 将 JSON 请求体构造为 `ImageGenerateRequest`。如果 JSON 类型错误、字段缺失或字段范围不合法，请求在进入 Service 前直接以 HTTP 422 失败。

主要请求定义文件：`tt-media-server/domain/image_generate_request.py`

FLUX.3 相关字段包括：

| 字段 | 类型/限制 | 用途 |
|---|---|---|
| `prompt` | 必填字符串 | 文本提示词。 |
| `negative_prompt` | 可选 | Base 模型可用于 CFG；Distilled 最终不会使用 CFG guidance。 |
| `num_inference_steps` | 4～50 | denoise steps；通用校验对 FLUX.3 当前适用的最低值是 12。runner 仍会读取该值。 |
| `seed` | 可选整数 | 初始噪声随机种子；runner 中未提供时使用 `0`。 |
| `width`、`height` | 256～1536，必须同时提供 | FLUX.3 逐请求输出分辨率。 |
| `guidance_scale` | 1.0～20.0 | Base 使用；Distilled runner 会设为 `None`。 |
| `reference_image` | 可选 Base64/data URL | 单参考图条件。 |
| `reference_images` | 1～10 张 Base64/data URL | 多参考图条件，仅 FLUX.3 runner 支持。 |
| `image_return_format` | `JPEG` 或 `PNG` | 最终编码格式。 |
| `response_format` | `b64_json` 或 `file` | JSON Base64 响应或直接图片文件响应。 |

`reference_images` 的校验也会在 Pydantic 层完成：只允许 Base64 或 `data:image/...;base64,...`，拒绝 HTTP URL，检查图像可解码，并验证多图缩放后每边至少保留 16 像素。

### 步骤 2：获取 ImageService 单例

文件：`tt-media-server/resolver/service_resolver.py`

`Depends(service_resolver)` 根据：

```text
settings.model_service == "image"
```

返回已创建的 `ImageService` 单例，而不会为每次 HTTP 请求新建 Scheduler 或 worker。

### 步骤 3：Service 预处理、分段与等待结果

文件：

```text
tt-media-server/model_services/image_service.py
tt-media-server/model_services/base_service.py
```

调用关系：

```text
ImageService.process_request()
  → ImageService.pre_process()
  → BaseService.process()
  → Scheduler.process_request()
```

具体行为：

1. `ImageService.pre_process()` 记录图片请求指标。
2. 如果 `number_of_images > 1`，创建多个 segment；每个 segment 被拆成一条单图请求，后续 seed 自动按序号递增。
3. `BaseService.process()` 为每个 request 的 `_task_id` 创建一个进程内 `asyncio.Queue`，保存到 `scheduler.result_queues`。
4. 调用 `Scheduler.process_request(request)` 将请求放进跨进程 `task_queue`。
5. 当前 API coroutine 在自己的 `asyncio.Queue` 上等待 worker 的结果，超时受 `request_processing_timeout_seconds` 控制。

## 5. Scheduler 入队与 worker 领取

文件：`tt-media-server/model_services/scheduler.py`

`Scheduler.process_request()` 会：

1. 调用 `check_is_model_ready()`；没有 ready worker 时返回 HTTP 405/503。
2. 检查 `task_queue` 是否已达 `max_queue_size`；队列满时返回 HTTP 429。
3. 将 request 放入全局跨进程 `task_queue`。

请求没有在 Scheduler 层预先固定绑定某颗芯片。任意空闲 worker 都可以从同一个任务队列领取它。

文件：`tt-media-server/device_workers/device_worker.py`

默认 worker 在循环中调用：

```python
dequeue_and_claim_requests(
    task_queue,
    worker_id,
    settings.max_batch_size,
    block=True,
    timeout=0.2,
)
```

对 FLUX.3，`max_batch_size=1`，所以一个 worker 一次领取一条图片请求。

`dequeue_and_claim_requests()` 位于 `device_workers/worker_utils.py`，它会进一步调用 `claim_job_for_worker()`：

1. 先检查 `_cancel_event`，已取消则不领取；
2. 写入 `_worker_assignment.identity = (worker_id, pid)`；
3. 再检查一次取消，以处理领取与 cancel 并发发生的竞态；
4. 设置 `_start_event`，表示 worker 已真正开始处理。

## 6. 进入 TTFlux3ImageKleinRunner

文件：`tt-media-server/tt_model_runners/dit_runners.py`

普通 worker 对 FLUX.3 调用：

```python
responses = device_runner.run(requests)
```

实际绑定到：

```python
TTFlux3ImageKleinRunner.run(requests)
```

该方法的工作顺序是：

1. 取 `requests[0]`，因为当前 FLUX.3 worker 的模型 batch 是 1。
2. 解析 `height`、`width`；未提供时均使用 1024。
3. 解析 `num_inference_steps`；未提供时使用 checkpoint 的 `default_steps`。
4. 判断 checkpoint：
   - Base（`flow_map=false`）：读取 `guidance_scale` 或默认 guidance；
   - Distilled（`flow_map=true`）：强制 `guidance=None`，不走 CFG。
5. 检查 `reference_image`、`reference_images` 和兼容字段 `image` 只能三选一。
6. 将 Base64 参考图通过 `ImageManager("flux3-reference")` 解码为 RGB PIL 图像。
7. 首次需要参考图时，惰性创建 `self._reference_encoder`；后续请求复用它。
8. 将 API 请求转换为 TT-Metal pipeline 的 `Flux3T2IRequest`：

```text
prompt / negative_prompt / height / width / steps / seed / guidance / reference_images
```

9. 调用：

```python
self.pipeline.run(
    flux_request,
    device=self.ttnn_device,
    ccl_manager=self._ccl_manager,
    parallel_config=self._dit_parallel,
    encoder_parallel_config=self._encoder_parallel,
    prompt_encoder=self._prompt_encoder,
    transformer=self._transformer,
    vae_decoder=self._vae_decoder,
    reference_encoder=self._reference_encoder,
)
```

这就是 TT Media Server runner 与 TT-Metal FLUX.3 pipeline 的边界。`pipeline.run(...)` 返回的 `image` 被 runner 放在单元素列表中交回 worker。

## 7. runner 返回后：结果怎样回到客户端

| 顺序 | 文件 | 功能 |
|---|---|---|
| 1 | `device_workers/device_worker.py` | 把 `(worker_id, task_id, response)` 放入 `result_queue`。 |
| 2 | `model_services/scheduler.py` | `result_listener()` 从各 worker result queue 读取结果，并按 `task_id` 放入 `scheduler.result_queues[task_id]`。 |
| 3 | `model_services/base_service.py` | 等待该 `task_id` 的 asyncio queue，收到结果后返回给 `process_request()`。 |
| 4 | `model_services/image_service.py` | `post_process()` 交给 `CpuWorkloadHandler`，使用 `ImageManager` 做 PNG/JPEG 编码、Base64 或文件形式的图片后处理。 |
| 5 | `open_ai_api/image.py` | `_build_image_response(...)` 根据 `response_format` 返回 `b64_json` JSON，或直接返回图片 file。 |

无论成功、超时、失败还是 API coroutine 被取消，`BaseService.process()` 的 `finally` 都会调用 `_teardown_task(task_id)`：删除该 request 的 result queue，并向 Scheduler 发出 best-effort cancel 信号。

## 8. 异步 API 与同步 API 的差异

异步入口：

```text
POST /v1/images/generations/async
```

文件：`tt-media-server/open_ai_api/image.py`、`model_services/image_service.py`。

异步接口首先由 `ImageService.create_image_job()` 交给 `JobManager` 建立 job，立即返回 202 和 job ID。后台 task 最终仍调用同一个：

```text
ImageService.process_request()
→ Scheduler
→ device worker
→ TTFlux3ImageKleinRunner
```

不同仅在最后：异步 job 将结果持久化为 JSON，客户端通过：

```text
GET  /v1/images/generations/{job_id}
POST /v1/images/generations/{job_id}/cancel
DELETE /v1/images/generations/{job_id}
```

查询、取消或删除任务。

## 9. `device_worker_dynamic_batch.py` 为什么不是当前默认 FLUX.3 路径

该文件只有在：

```text
settings.use_dynamic_batcher == true
```

时由 `Scheduler._start_worker()` 选择。它通过 `asyncio.create_task()` 同时维护多个 in-flight task，并有 `cancel_listener()` 从 `cancel_queue` 寻找并 `task.cancel()`。

它的设计目标主要是 vLLM 等有 `_run_async()` 的连续/流式任务。当前 `TTFlux3ImageKleinRunner` 实现的是同步的：

```python
run(requests)
```

而不是 `_run_async()`。因此在默认 FLUX.3 配置中：

```text
use_dynamic_batcher = false
→ device_worker.py
→ device_runner.run(requests)
→ TTFlux3ImageKleinRunner.run(requests)
```

不要仅因文件名中有 `dynamic_batch` 就认为当前 FLUX.3 请求经过它。

## 10. 关键文件总表

| 层级 | 文件 | 对 FLUX.3 请求的职责 |
|---|---|---|
| API 路由 | `open_ai_api/__init__.py` | 注册 `/v1/images` 图片路由。 |
| API endpoint | `open_ai_api/image.py` | 接受同步/异步图片请求，构造 HTTP 响应。 |
| 请求模型 | `domain/image_generate_request.py` | Pydantic 校验 prompt、steps、分辨率、参考图与输出格式。 |
| Service 选择 | `resolver/service_resolver.py` | 将 `image` 服务解析为 `ImageService` 单例。 |
| 图片 Service | `model_services/image_service.py` | 图片指标、multi-image 拆分、CPU 编码、异步 job 管理。 |
| 通用 Service | `model_services/base_service.py` | request task queue、结果等待、超时与 teardown。 |
| Scheduler 单例 | `resolver/scheduler_resolver.py` | 保证一个服务进程只有一个 Scheduler。 |
| Scheduler | `model_services/scheduler.py` | 队列、worker 进程、result/error/warmup listener、健康检查与重启。 |
| 默认 worker | `device_workers/device_worker.py` | FLUX.3 默认路径：取任务、同步调用 runner、回传结果。 |
| 可选 dynamic worker | `device_workers/device_worker_dynamic_batch.py` | 仅 `use_dynamic_batcher=true` 时的异步任务实现；不是当前 FLUX.3 默认路径。 |
| worker 公共工具 | `device_workers/worker_utils.py` | 领取/释放任务、创建 runner、打开 device、warmup。 |
| runner factory | `tt_model_runners/runner_fabric.py` | `MODEL_RUNNER` → runner 类实例。 |
| FLUX.3 runner | `tt_model_runners/dit_runners.py` | API request → `Flux3T2IRequest` → `pipeline.run(...)`。 |
| TT device 基类 | `tt_model_runners/base_metal_device_runner.py` | 打开、配置和关闭 TTNN mesh device。 |

## 11. 一句话定位问题的方法

```text
422：优先看 ImageGenerateRequest/Pydantic 校验
405/503：看 Scheduler readiness 与 worker warmup
429：看 Scheduler.task_queue 是否满
worker 初始化失败：看 worker_utils → runner_fabric → warmup/create_pipeline
请求执行失败：看 device_worker → TTFlux3ImageKleinRunner.run → pipeline.run
有结果但 HTTP 响应异常：看 Scheduler.result_listener → BaseService.process → ImageService.post_process
```
