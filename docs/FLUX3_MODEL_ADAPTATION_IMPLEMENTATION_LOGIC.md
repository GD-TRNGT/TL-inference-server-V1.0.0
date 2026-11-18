# FLUX.3 Image Klein 模型适配实现逻辑

更新时间：2026-11-18

本文用于讲解 FLUX.3 Image Klein 如何接入 TT-Metal 与 TL-inference-server，重点说明代码职责、数据流、设备并行和请求生命周期。

## 1. 总体架构

```text
OpenAI-compatible HTTP API
        ↓
请求校验：API key、prompt、宽高、steps、seed、参考图
        ↓
FLUX3 profile：worker mesh、TP/SP、Fabric、parent/child 模式
        ↓
worker / device runner
        ↓
Flux3T2IPipeline
   ├─ Qwen3-VL text encoder
   ├─ FLUX.3 Transformer denoise
   ├─ flow schedule / CFG / Euler update
   └─ VAE decoder
        ↓
device-to-host、JPEG/PNG 编码、HTTP 响应
```

两仓库职责不同：

| 仓库 | 主要职责 |
|---|---|
| `tt-metal` | TTNN Transformer、Qwen3-VL、VAE、权重映射、设备执行、kernel/cache |
| `TL-inference-server` | API、profile、worker、设备生命周期、队列、warmup、健康检查和 Docker |

## 2. 模型契约

适配首先固定 checkpoint 结构和公共 API 契约：

- Base 与 Distilled 使用各自权重目录；
- 文本条件由 Qwen3-VL encoder 产生；
- DiT 使用 image token、text token、timestep 和 RoPE；
- Distilled 路径执行 flow denoise；
- Base 路径支持需要 CFG 的条件组合；
- 最终 latent 由 FLUX.3 VAE decoder 还原成 RGB 图片；
- API 宽高按官方范围和 16 像素对齐校验；
- 参考图作为独立 image-condition 输入，数量上限为 10 张。

21:9 和 9:21 的专项调试不属于本次文档更新范围。

## 3. server 侧 profile 与 worker

`tt-media-server/config/flux3_profiles.py` 是并行 profile 的单一事实来源，声明 worker mesh、parent mesh、TP/SP factor、axis、Fabric 和 resident pipeline 行为。

`config/settings.py` 将环境变量解析成运行时设置：

```text
FLUX3_EXECUTION_PROFILE
  → worker device shape
  → parent mesh shape
  → submesh shape/index
  → discovery 和启动约束
```

`utils/runner_utils.py` 把 scheduler/worker 的设备布局转换成 Fabric 和 parent Mesh 配置；`tt_model_runners/dit_runners.py` 负责 pipeline、warmup、请求执行和阶段日志。

## 4. TT-Metal 模型执行链路

### 4.1 文本编码

```text
prompt
  → tokenizer / chat template
  → padding / mask / M-RoPE
  → Qwen3-VL Text Encoder
  → prompt embedding
```

Text encoder 在 worker 中常驻，缓存命中后不重复转换权重。

### 4.2 Transformer 去噪

```text
seed
  → 初始 latent / image token
  → patchify 与 image IDs
  → text/image/reference 条件拼接
  → timestep embedding
  → FLUX.3 mode blocks
  → FLUX.3 single blocks
  → CFG（Base 需要时）
  → flow Euler update
  → 重复 N steps
```

FLUX.3 Transformer 负责 mode block、single block、attention、RoPE、调制和输出投影。joint RoPE 可在 denoise loop 外构造，并在多个 step 中复用。

### 4.3 VAE 解码

```text
最终 denoised latent
  → preprocess / unpatchify
  → VAE decoder
  → untilize / host readback
  → RGB tensor
  → JPEG 或 PNG
```

VAE 的权重加载、空间分片、GroupNorm、Conv program 和 decode geometry 按 mesh 与分辨率建立独立 cache key。

## 5. 常驻组件与缓存

resident 模式下，每个 worker 常驻：

1. Qwen3-VL text encoder；
2. FLUX.3 Transformer；
3. VAE decoder；
4. 与 mesh/layout/geometry 相关的权重和 kernel cache。

典型缓存目录：

```text
/local/ttcache/flux-3-image-klein
├── tt_dit_cache
├── tt-metal-cache
├── hf
└── torch
```

请求之间复用权重、程序和 schedule，不重复加载完整模型。

## 6. 参考图路径

```text
1～10 张参考图
  → 解码与尺寸预处理
  → reference VAE encoder
  → image-condition token
  → 与 text/image token 组成条件输入
```

旧单图字段保持兼容；新接口使用有序 `reference_images` 数组。非法图片和超出数量的请求在入队前校验。

## 7. API 到设备的生命周期

```text
HTTP request
  → bearer token 校验
  → width/height/steps/参考图校验
  → scheduler 选择 ready worker
  → pipeline encode / denoise / decode
  → 阶段计时日志
  → 图片编码
  → file/base64 响应
```

性能统计区分 HTTP E2E、text/reference encode、denoise、VAE decode、host readback 和图片编码，避免把排队或 JPEG 编码误算为设备计算时间。

## 8. 当前验证口径

当前父子 Mesh 回归使用：

```text
1024x1024、16 steps、seed=42、同步串行请求、JPEG
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
FLUX3_CORE_GRID_OVERRIDE=12x10
```

所有已列出的 2/4/8 卡 parent/child profile 均已 HTTP 200 并返回有效 JPEG。详细数据见：

```text
docs/FLUX3_PARENT_CHILD_MESH_REGRESSION_20261118.md
```

## 9. 代码版本与边界

当前提交：

```text
1ae19312 flux3: complete parent child mesh profile regression
```

本文不把未完成的 21:9/9:21 专项调试、镜像重建或 Golden Data 精度验收描述成已完成项目。
