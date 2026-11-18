# TTNN 模型权重转换指南

更新时间：2026-11-18

本文说明将 PyTorch / Safetensors 模型权重转换为 TTNN 可加载权重时的规则、工具、并行分片方式，以及 FLUX3 Klein 中的实际落地路径。重点是“如何适配一个新模型”，而不是某个 checkpoint 的一次性转换命令。

## 1. 权重转换不是文件复制

原始权重通常是 PyTorch 格式：

```text
model.safetensors / pytorch_model.bin
        ↓
PyTorch state_dict
        ↓  名称映射
TT 模型参数名
        ↓  shape / layout 转换
适合 TTNN 算子的张量
        ↓  dtype / 量化处理
BF16、FP8、NVFP4 或对应 scale
        ↓  TP / SP 分片或复制
每个 device / mesh 的权重
        ↓
ttnn.from_torch
        ↓
TTNN cache
```

因此，权重转换同时处理五类问题：

1. 参数名称是否能被 TT 模型找到；
2. 参数 shape 是否符合 TT 算子约定；
3. row-major、tile layout、interleaved 或 sharded layout 是否匹配；
4. dtype、量化 scale 和 zero-point 是否匹配；
5. 权重是复制到每个 device，还是沿 TP/SP 轴切分。

只把 Safetensors 文件挂载到容器里，并不会自动完成这些步骤。

## 2. 转换前先固定模型契约

开始写转换代码前，应先记录：

- 模型变体：Base、Distilled、FP8、NVFP4 等；
- checkpoint 文件和配置文件；
- hidden size、intermediate size、层数、head 数和 latent channels；
- patch size、VAE scale factor 和位置编码规则；
- scheduler、timestep、flow update 和 CFG 规则；
- 支持的宽高、对齐约束、最大像素数和 batch 范围；
- TP、SP、VAE 并行度以及通信轴；
- 哪些权重必须复制，哪些权重必须切分。

这一步能避免 API、pipeline、Transformer、VAE 和权重 cache 对同一个 shape 产生不同解释。

## 3. 参数名称映射

### 3.1 为什么需要映射

官方模型和 TTNN 模型的命名通常不同。例如官方实现可能使用：

```text
transformer.blocks.0.attn.to_q.weight
```

而 TTNN 模型可能使用：

```text
transformer_blocks.0.attention.q_proj.weight
```

转换层必须建立确定性的映射，不能依赖模糊匹配。建议将映射拆成：

- 前缀映射：模型主体、VAE、text encoder；
- block 索引映射：第 `i` 个 Transformer block；
- 子模块映射：Q/K/V、output projection、MLP、norm；
- 参数后缀映射：`weight`、`bias`、`scale`、`zero_point`。

### 3.2 映射表示例

```python
def map_flux3_name(name: str) -> str:
    name = name.replace("transformer.blocks.", "transformer_blocks.")
    name = name.replace("attn.to_q.", "attention.q_proj.")
    name = name.replace("attn.to_k.", "attention.k_proj.")
    name = name.replace("attn.to_v.", "attention.v_proj.")
    name = name.replace("attn.to_out.0.", "attention.out_proj.")
    return name
```

实际项目应使用模型专用映射函数，并在转换结束后检查：

- 原始参数是否全部被消费；
- TT 模型需要的参数是否全部存在；
- 是否有重复映射；
- 是否有未识别的参数名。

## 4. Shape 和 layout 转换

### 4.1 Linear 权重

PyTorch `Linear` 通常保存为：

```text
[out_features, in_features]
```

TTNN matmul 可能要求同样的逻辑方向，但实际 device tensor 还需要 tile 对齐、分片和指定 memory layout。不能只看二维 shape 是否相等。

常见处理顺序是：

```text
PyTorch tensor
  → dtype 转换
  → 必要时转置或重排
  → pad 到 tile 对齐
  → ttnn.from_torch
  → device layout / sharding
```

### 4.2 Conv 权重

PyTorch Conv2d 权重通常为：

```text
[out_channels, in_channels, kernel_h, kernel_w]
```

需要结合 TTNN Conv2d 的输入 layout、padding、stride、block shape 和 device shard 规则确认是否需要重排。VAE 中的卷积、上采样卷积和残差块尤其不能用 Transformer Linear 的规则直接处理。

### 4.3 QKV 权重

Q、K、V 可能在官方模型中是三个独立 Linear，也可能已经合并为一个 QKV 权重。TTNN 实现必须明确：

- 是保持三个权重独立；
- 还是在转换阶段沿输出维拼接；
- 拼接顺序是 Q-K-V 还是其他顺序；
- head 维度和 RoPE 维度是否保持一致。

QKV 顺序错误通常不会导致加载失败，而会导致图像质量明显错误，因此必须做数值和端到端验证。

### 4.4 SwiGLU / gated MLP

SwiGLU 常见形式是：

```text
gate = x @ W_gate
up   = x @ W_up
y    = activation(gate) * up
out  = y @ W_down
```

官方实现可能把 `gate_proj` 和 `up_proj` 合并，TTNN 实现也可能使用 fused projection。转换时要确认：

- 合并前后输出通道顺序；
- 权重拼接轴；
- bias 是否存在；
- fused 算子期望的输入和输出 shape。

## 5. dtype、FP8 和 NVFP4

### 5.1 BF16 / FP16

BF16 通常是最容易验证的基线。先使用 BF16 完成：

1. 权重加载；
2. 单层数值对比；
3. 单步 denoise 对比；
4. 完整图片生成；

再引入 FP8 或 NVFP4。否则量化误差和布局错误会混在一起，难以定位。

### 5.2 FP8

FP8 通常不仅有低精度权重，还需要对应 scale、block scale 或 per-channel scale。转换时必须同时保存：

- FP8 weight；
- scale 的粒度和 shape；
- scale 的存储 dtype；
- 算子使用的 scale 方向和广播规则。

不能只把 BF16 tensor 强制 cast 成 FP8 就认为完成了 FP8 适配。

### 5.3 NVFP4

NVFP4 依赖更严格的量化布局、scale 元数据和硬件算子支持。应确认：

- checkpoint 是否已经是 NVFP4 格式；
- TTNN 是否支持对应 device、算子和 layout；
- scale / group size / zero-point 是否与算子一致；
- cache key 是否包含量化变体，避免 FP8、NVFP4 和 BF16 cache 混用。

量化变体必须使用不同的 cache 名称，例如：

```text
transformer_..._bf16
transformer_..._fp8
transformer_..._nvfp4
```

## 6. TTNN 转换工具

### 6.1 读取 Safetensors

```python
from safetensors.torch import load_file

state_dict = load_file("model.safetensors")
```

### 6.2 读取 PyTorch checkpoint

```python
import torch

state_dict = torch.load(
    "pytorch_model.bin",
    map_location="cpu",
    weights_only=True,
)
```

读取后建议立即输出参数数量、名称、shape 和 dtype，先确认 checkpoint 本身正确。

### 6.3 转成 TTNN tensor

```python
import ttnn

tt_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)
```

实际项目还需要根据 mesh、memory config 和 shard spec 选择 device layout。`ttnn.from_torch` 负责建立 TTNN tensor，不负责替模型决定参数名、分片策略或量化 scale。

### 6.4 使用模型 cache

FLUX3 使用模型专用的 cache 加载路径，将转换后的权重持久化，避免每次启动重新转换：

```python
cache.load_model(
    tt_model=model,
    get_torch_state_dict=lambda: state_dict,
    model_name=model_name,
    parallel_config=parallel_config,
    mesh_shape=tuple(device.shape),
    mesh_device=device,
)
```

实际函数参数以当前 `tt-metal/models/tt_dit/utils/cache.py` 为准。缓存 key 至少应区分：

- 模型变体；
- block / layer 配置；
- TP/SP mesh shape；
- dtype 和量化格式；
- 设备架构；
- 必要的 layout 或编译配置。

## 7. TP、SP 和复制规则

### 7.1 复制权重

适合复制的内容通常包括：

- 小型 embedding；
- layer norm 的 weight / bias；
- 每个 device 都需要完整访问的控制参数；
- 某些 text encoder 权重；
- VAE 中尚未做空间分片的参数。

复制不会减少单个 device 的权重容量，但能避免运行时通信。

### 7.2 TP 分片

Tensor Parallel 通常沿 Linear 的输出或输入通道切分：

- column-parallel：沿输出通道切分，多个 device 分别计算部分输出；
- row-parallel：沿输入通道切分，部分结果需要 all-reduce / reduce-scatter；
- QKV：要按 head 或投影通道保持一致切分；
- MLP：gate/up/down 的切分轴必须互相匹配。

TP 分片的关键不是“平均切成 N 份”，而是切分轴必须与下一个算子的输入、通信和 layout 一致。

### 7.3 SP 分片

Sequence / spatial parallel 通常沿 token、序列或空间维切分：

- Transformer token 在 device 间分片；
- attention 前后需要按照实现进行 all-gather、reduce-scatter 或 ring 通信；
- VAE 可能沿高度、宽度或 latent 空间做特殊分片；
- SP 和 TP 可以组成二维 mesh，例如 TP2×SP4。

### 7.4 二维 TP×SP

二维 mesh 中，两个轴承担不同职责：

```text
父 mesh
  └── 子 mesh：TP × SP
```

权重通常沿 TP 轴切分或复制，token / spatial 激活沿 SP 轴切分。通信也应明确区分：

- TP 轴：row-parallel 的归约或权重相关通信；
- SP 轴：token / spatial 的 gather、scatter 或 ring；
- 两个轴都参与时：使用与物理 Fabric 拓扑匹配的二维通信方式。

不能仅凭 `mesh_shape=(2, 4)` 推断所有权重都要切成 8 份；每个参数需要单独指定复制、TP 切分或 SP 相关处理。

## 8. FLUX3 Klein 的实际转换路径

当前实现的逻辑可以概括为：

```text
Hugging Face / Safetensors checkpoint
  → FLUX3 参数名映射
  → Transformer / text encoder / VAE 分组
  → BF16 或量化 dtype 处理
  → Linear、QKV、SwiGLU、Conv layout 处理
  → 根据 execution profile 复制或 TP/SP 分片
  → ttnn.from_torch
  → 写入 /local/ttcache 对应 cache
  → 启动时 cache.load_model
```

相关代码应重点检查：

```text
tt-metal/models/tt_dit/utils/cache.py
tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
```

Server 仓库负责请求、worker、mesh 配置、缓存目录和 API；真正的参数转换和算子执行主要位于 `tt-metal` 的模型实现中。构建 Flux3 镜像时，至少要确认镜像内的 TL-inference-server 和 tt-metal 版本来自预期的提交，而不是只更新了 server 层。

## 9. VAE 权重的特殊处理

VAE 不是 Transformer 权重的简单复制。它通常包含：

- Conv2d / ConvTranspose2d；
- 上采样和下采样；
- residual block；
- latent scaling / shifting；
- 不同于 Transformer 的空间 layout；
- 可能独立的 VAE 并行度。

因此，VAE 需要单独验证：

1. 输入 latent shape；
2. latent scaling 和 shift；
3. Conv 权重 shape；
4. 上采样后的空间尺寸；
5. device shard 和 L1 使用量；
6. 最终 RGB 输出范围和格式。

当分辨率升高时，VAE 的激活空间会快速增大。即使 denoise 权重转换正确，VAE 仍可能因为 L1、program 配置或空间分片不匹配而 OOM。

## 10. Cache 生命周期和目录规则

推荐将 cache 与模型目录分开：

```text
/local/models/flux-3-image-klein/       # 原始模型权重
/local/ttcache/flux-3-image-klein/      # TTNN 转换权重和编译缓存
```

cache key 不应只使用模型名。至少要包含 mesh、dtype 和并行配置，否则切换 `sp4`、`sp8`、`tp4` 或 `tp8` 时可能错误复用不兼容缓存。

首次启动出现大量 `BuildKernels` 是正常的编译缓存建立过程；如果随后出现 Bus error、设备 timeout 或 L1 OOM，应区分：

- 权重转换错误；
- mesh / Fabric placement 错误；
- kernel 编译或 program 配置错误；
- cache 污染或复用错误；
- host / MPI / 容器资源问题。

## 11. 正确性验证顺序

建议按以下顺序验证，而不是直接进行多 worker 压测：

### 11.1 权重完整性

- checkpoint 参数数量和名称；
- TT 模型缺失参数；
- 未消费的源参数；
- 每个参数的 shape、dtype 和分片信息。

### 11.2 单层数值验证

对 Linear、QKV、attention、SwiGLU、Conv 和 norm 做 CPU / PyTorch 与 TTNN 输出对比。先比较 BF16，再比较 FP8 / NVFP4。

### 11.3 单步 denoise 验证

固定 prompt、latent、timestep 和 seed，对比单步输出误差。这样可以把 scheduler、VAE 和图片编码因素排除在外。

### 11.4 完整推理验证

固定以下条件：

```text
resolution
steps
seed
model variant
response format
parallel profile
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

同时记录：

- HTTP E2E；
- denoise；
- VAE decode；
- postprocess / image encoding；
- 是否 JPEG / PNG；
- 是否排队；
- worker 数和每 worker 卡数。

性能数据必须保持同步请求、相同图片格式和相同 warmup 条件，否则不同测试不可直接比较。

## 12. 新模型的推荐适配顺序

```text
1. 固定官方模型契约
2. 读取 checkpoint 并检查名称 / shape
3. 实现名称映射
4. 实现 BF16 单卡转换
5. 做单层数值对比
6. 接入 Transformer / VAE pipeline
7. 做单步 denoise 对比
8. 接入 cache.load_model
9. 实现 TP 或 SP 分片
10. 实现二维 TP×SP / 父子 mesh
11. 验证多分辨率和 VAE
12. 再引入 FP8 / NVFP4
13. 最后做多 worker 并发和镜像回归
```

## 13. 常见错误

### 参数能加载但结果错误

通常检查 QKV 顺序、SwiGLU 拼接顺序、RoPE、latent scaling 和 scheduler。

### 只有某个 mesh 失败

通常检查 TP/SP 切分轴、子 mesh 的物理 placement、Fabric descriptor、通信轴和 cache key。

### 启动很慢或长期不 ready

先确认是否正在建立 kernel / weight cache，再检查是否错误地为每个 worker 重复加载全模型、是否误用了不兼容 cache，或是否出现 MPI / host 资源问题。

### 高分辨率 VAE OOM

检查 VAE 空间 shard、activation block、double buffer、layout、L1 使用量和分辨率约束。不要仅通过降低 denoise 的 batch 或 token 数解决 VAE 的空间激活问题。

### FP8 / NVFP4 与 BF16 混用

检查 cache key、scale 元数据和模型变体目录，确保不同量化格式不会复用同一套转换结果。

## 14. 与 Server 和 Docker 的关系

权重转换本身属于模型运行时和 `tt-metal`；server 主要负责：

- API 校验；
- worker 生命周期；
- `TT_VISIBLE_DEVICES` 和 mesh 配置；
- 模型 / cache 挂载；
- 并发调度；
- 输出格式和鉴权。

Dockerfile 则负责把指定版本的 server、`tt-metal`、Python 环境和依赖封装起来。为了保证可复现，构建时应记录：

- TL-inference-server commit；
- tt-metal commit；
- 基础镜像 digest；
- 模型变体；
- cache 是否在镜像内还是运行时挂载；
- 构建时和运行时的环境变量。

## 15. 总结

新模型适配的核心不是把权重文件转换成另一种文件，而是建立一条完整且可验证的映射链：

```text
官方参数
  → TT 模型参数名
  → 正确的 shape / layout
  → 正确的 dtype / scale
  → 正确的 TP / SP 分片
  → 正确的 mesh / CCL 通信
  → 可复用且隔离的 cache
  → 单层、单步、端到端验证
```

只要其中任意一层不一致，模型可能表现为加载失败、结果错误、性能下降、长期不 ready、设备 timeout 或高分辨率 OOM。因此应先完成 BF16 单卡正确性，再逐步增加并行度、量化格式和 worker 数量。
