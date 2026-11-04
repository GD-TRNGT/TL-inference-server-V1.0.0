# FLUX.3 Image Klein 生成内容与 Prompt 不一致问题排查及修复记录

更新时间：2026-11-04

## 1. 问题结论

在 10.12.11.2 的 Blackhole Galaxy 上，请求 `a black bird` 时，早期 TTNN 实现生成的是蓝色方块或无语义纹理。最终确认主要根因是 **FLUX.3 checkpoint 与 TTNN 共用 fused SwiGLU 层对 `mlp_in.weight` 两半的解释顺序相反**。

```text
FLUX.3 checkpoint: [gate | up]，计算 silu(gate) * up
TTNN loader 期望: [up | gate]
```

修复是在 checkpoint 映射边界交换两半。修复后 Base 和 Distilled 两套模型都能根据相同提示词生成黑鸟。

## 2. 两套模型

模型统一放在宿主机 `/local/models`，不复制进镜像：

| 变体 | 目录 | 默认行为 | DiT tensors |
|---|---|---|---:|
| Base | `/local/models/flux-3-image-klein-base` | 50 steps、CFG 3.5、`flow_map=false` | 357 |
| Distilled | `/local/models/flux-3-image-klein` | 16 steps、无 CFG、`flow_map=true` | 359 |

Distilled 额外包含 `ref_time_in.in_layer.weight` 和 `ref_time_in.out_layer.weight`。两套模型共用实现，由 `config.json` 自动识别变体。

## 3. 问题现象

修复前使用 `prompt=a black bird`、`seed=42`、`1024x1024`：

- HTTP 200，能够返回合法图片；
- Transformer、denoise loop、VAE decode 均执行完成；
- 图片却是蓝色块或无意义纹理，没有黑鸟；
- 改步数或 seed 不能稳定恢复语义。

因此“端到端能运行”只能证明链路连通，不能证明模型数学实现正确。

## 4. 排查过程

### 4.1 API 参数

检查：

```text
/local/lnn/TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py
```

确认 prompt、negative prompt、steps、seed、width、height 和 guidance 都进入 `Flux3T2IRequest`，排除 API 丢失提示词。

### 4.2 Prompt encoder

检查并修正：

```text
models/tt_dit/pipelines/flux3/flux3_prompt_encoder.py
models/tt_dit/pipelines/flux3/flux3_components.py
```

- chat template 改为 `add_generation_prompt=True`；
- padding 使用 tokenizer 的真实 `pad_token_id`；
- HF `hidden_states[4,8,...,32]` 映射到 TT decoder 索引 `[3,7,...,31]`。

这些修复必要，但单独修复后仍未恢复黑鸟语义。

### 4.3 Noise、scheduler、time embedding 和 RoPE

检查并修正：

```text
models/tt_dit/pipelines/flux3/sampling.py
models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

- 初始噪声直接按 `[N,H_tokens*W_tokens,128]` 生成；
- 噪声先舍入 BF16，再转回 FP32 用于 Euler 积分；
- timestep 在正弦编码前乘 1000；
- 使用 `alpha=6.93` 的 flow-shift schedule；
- 使用 FLUX.3 四轴 RoPE 和相邻通道成对布局；
- AdaLN 使用 `1 + scale`；
- Base 使用 CFG，Distilled flow-map 禁用 CFG。

完成这些修复后仍出现蓝色块，因此继续检查所有 block 共用的 FFN。

### 4.4 定位 SwiGLU

文件：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
```

15 个 mode block 和 28 个 single block 都经过同一个 FFN/SwiGLU 路径。权重两半顺序错误会让每一层都执行错误非线性，足以完全破坏语义。

## 5. 根因和代码修复

位置：`Flux3Transformer._block_state()`。

修复前：

```python
"ff.ff1.weight": reader.get_tensor(f"{prefix}.mlp_in.weight")
```

修复后：

```python
mlp_gate, mlp_up = reader.get_tensor(
    f"{prefix}.mlp_in.weight"
).chunk(2, dim=0)

"ff.ff1.weight": torch.cat((mlp_up, mlp_gate), dim=0)
```

实现效果：把 checkpoint 的 `[gate | up]` 转成 TTNN loader 需要的 `[up | gate]`。交换只发生在 FLUX.3 权重映射边界，不修改模型文件，也不影响其他模型的通用 SwiGLU 实现。

## 6. Base/Distilled 共用实现

`flux3_config.py` 从 `config.json` 读取 `flow_map`，并校验它与 `ref_time_in` 权重是否一致。`transformer_flux3.py` 分别使用：

```text
flux-3-image-klein-base
flux-3-image-klein-distilled
```

作为转换 cache 名，避免两种模型交叉复用权重。

| 项目 | Base | Distilled |
|---|---|---|
| 默认步数 | 50 | 16 |
| CFG | 默认 3.5 | 禁用 |
| negative prompt | 编码并运行 unconditional 分支 | 不编码 |
| `ref_time_in` | 无 | 有 |

## 7. 真实验证结果

### Base

```text
模型=/local/models/flux-3-image-klein-base
prompt=a black bird
steps=50, guidance=3.5, seed=42, size=1024x1024
generation_time=68.37 s
SHA256=c268566ea53a9eec50143e87ec673d9916f491f06c8f74cd22f41a07df9ebf88
本地图片=base_black_bird_swiglu_fixed.jpg
```

图片中能够识别出黑鸟。

### Distilled

```text
模型=/local/models/flux-3-image-klein
prompt=a black bird
steps=16, CFG=disabled, seed=42, size=1024x1024
generation_time=17.45 s
SHA256=148d521f8b569880379c7bef7019106f1d3a6e28d2b0a5e1ae6fb4fb7cc5da58
本地图片=distilled_black_bird_16.jpg
```

图片中同样能够识别出黑鸟。

## 8. 缓存问题

SwiGLU 映射改变后，旧转换缓存不能继续使用。旧缓存备份为：

```text
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein-base.pre-swiglu-fix-20261104
```

32 worker 首次同时转换并写入相同 cache 会造成严重 I/O 竞争。一次可疑并发缓存已备份：

```text
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein-base.concurrent-backup-20261104-1441
```

当前质量验证使用单 worker。恢复 32 worker 前，应先单进程预生成两套 cache，或增加跨进程锁、临时目录和原子 rename。

## 9. 当前状态

| 工作项 | 状态 |
|---|---|
| Base 单芯片 T2I | 已实现并通过语义验证 |
| Distilled 单芯片 T2I | 已实现并通过语义验证 |
| prompt、steps、seed、width、height | 已接入 API |
| Base guidance/negative prompt | 已接入 |
| Distilled flow-map/`ref_time_in` | 已接入 |
| 两模型独立 cache | 已实现 |
| 2/4 芯片并行 | 暂缓 |
| Golden Data 逐层对齐 | 暂缓 |
| 32 worker 首次 cache 并发创建 | 未解决 |
| kernel/trace/layout 专项优化 | 未开始 |

当前容器 `flux3-media-server-test` 运行 Distilled 模型 `/local/models/flux-3-image-klein`，单 worker、单芯片。
