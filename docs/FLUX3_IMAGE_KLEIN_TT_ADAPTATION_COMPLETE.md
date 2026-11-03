# FLUX.3 Image Klein 在 TT-Metal/TT-inference-server 上的适配记录

更新时间：2026-11-03  
状态：FLUX.3 T2I 三个模型组件的权重映射已完成并通过单芯片验证；端到端出图、服务化和多芯片扩展仍未完成。

## 1. 目标

在 `10.12.11.2` 的 Blackhole Galaxy 上，使用本地 `TL-inference-server` 和
`tt-metal` 适配 FLUX.3 Image Klein，最终实现：

1. 单芯片 T2I bring-up；
2. 2 芯片和 4 芯片扩展；
3. BF16 精度对齐；
4. VAE 解码和图片输出；
5. Qwen3-VL 文本编码器接入；
6. tt-media-server runner/API 接入；
7. 性能、缓存、trace 和服务化验证。

## 2. 开发环境和仓库

### 2.1 机器

```text
主机：10.12.11.2
主机名：my01-galaxy02
硬件：Blackhole Galaxy
芯片数：32
首版目标：单芯片
逻辑 mesh：(1, 1)
目标 device：(0)
batch：1
```

代码不能写死为 32 芯片。后续应通过配置支持 1、2、4 芯片，包括 device list、mesh、
TP/SP 和权重分片策略。

### 2.2 TL-inference-server

```text
路径：/local/lnn/TL-inference-server
分支：flux-3-image-klein
commit：561f80d1
origin：https://github.com/GD-TRNGT/TL-inference-server-V1.0.0.git
```

负责服务生命周期、worker、runner、API、请求调度、设备初始化和输出。

### 2.3 tt-metal

```text
路径：/local/lnn/tt-metal
分支：flux-3-image-klein
commit：23ef0fd19b
origin：https://github.com/GD-TRNGT/TL-metal-V1.0.0.git
```

负责 TT-NN 模型层、attention、FFN、VAE、权重加载、kernel、cache、trace、mesh 和 CCL。

FLUX.3 不能只修改 `TL-inference-server`；模型结构和权重加载必须在 `tt-metal` 中实现，
再由 server 仓库负责服务适配。

## 3. 本地模型资产

模型目录：

```text
/local/models/flux-3-image-klein-base
```

已确认包含：

```text
ae.safetensors
model.safetensors
config.json
manifest.json
transformer/
text_encoder/
vae/
variants/fp8r/
variants/nvfp4/
```

### 3.1 关键配置

| 参数 | 值 |
|---|---:|
| hidden | 3072 |
| attention heads | 24 |
| head dim | 128 |
| mode depth | 5 |
| single depth | 28 |
| image channels | 128 |
| text channels | 20480 |
| 默认分辨率 | 1024×1024 |
| 默认 steps | 50 |
| 默认 guidance | 3.5 |
| 文本编码器 | Qwen3-VL |
| 默认 dtype | BF16 |

### 3.2 manifest

`manifest.json` 已确认：

```text
kind：f3_dit_export
model：flux-3-image-klein
tensors：357
params：5538458368
dtype：bfloat16
```

`model.safetensors` 的 SHA-256：

```text
5c921759dc03663dbb67df11d8e3d317d2d5b893b31b969c95dc77c7ebf861b19c
```

首版应先使用 BF16；FP8R 和 NVFP4 暂不作为 bring-up 路径。

## 4. FLUX.3 与 FLUX.2 的区别

当前 `tt-metal` 已有 FLUX.2：

```text
models/tt_dit/pipelines/flux2/pipeline_flux2.py
models/tt_dit/pipelines/flux2/prompt_encoder.py
models/tt_dit/models/transformers/transformer_flux2.py
models/tt_dit/models/vae/vae_flux2.py
```

FLUX.2 只能作为基础设施参考，不能直接改名复用为 FLUX.3，主要差异是：

- FLUX.3 hidden 为 3072，层数为 5 + 28；
- FLUX.3 有独立的 image/text mode block；
- FLUX.3 有 28 个 joint single block；
- FLUX.3 使用 Qwen3-VL，text conditioning 为 20480 维；
- FLUX.3 使用 early/single stream modulation；
- FLUX.3 的权重导出命名、输入投影和输出调制不同。

## 5. Docker 和本地源码

已修改 `TL-inference-server/tt-media-server/Dockerfile`，使用 BuildKit 本地源码：

```bash
cd /local/lnn/TL-inference-server

docker buildx build \
  --load \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=3c8681218b0951637991d20ce6b7e341df3b6879 \
  -f tt-media-server/Dockerfile \
  -t tl-inference-server:flux-3-image-klein-local \
  .
```

必须确认镜像内使用的是本地 `/local/lnn/tt-metal`，而不是 Dockerfile 重新 clone 的其他版本。

## 6. 已完成的代码

### 6.1 配置和权重校验

新增：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_config.py
```

功能：

- 读取 `config.json` 和 `manifest.json`；
- 使用 Python 标准库读取 safetensors header；
- 校验模型名、导出类型、关键 tensor 名和 357 个 tensor；
- SHA-256 校验改为显式开启，避免启动时默认读取 11 GB 权重。

### 6.2 T2I 请求契约

新增：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
```

当前首版限制：

```text
单 prompt、单输出、1024×1024、batch=1、BF16、固定 seed/steps/guidance
暂不支持 I2I、多参考图、LoRA、FP8R、NVFP4、动态 batch
```

### 6.3 T2I pipeline 边界

新增：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

已完成：

- 创建 `Flux3T2IPipeline`；
- 校验本地模型；
- 校验 T2I 请求；
- 根据模型配置创建 `Flux3Transformer`。

已完成模型组件构建入口和权重加载入口；尚未完成：Qwen3-VL 的实际文本编码执行、scheduler、latent、denoise loop、VAE decode 串联和图片输出。

### 6.4 TT-NN FLUX.3 transformer 初版

新增：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
```

已加入：

- `Flux3ModeBlock`；
- `Flux3SingleBlock`；
- `Flux3Transformer`；
- image/text 输入投影；
- early/single shift-scale-gate modulation；
- 5 个 image mode block；
- 5 个 text mode block；
- 28 个 joint single block；
- TT-NN attention、SwiGLU、动态 LayerNorm；
- image output projection。

该文件已在 Docker/Blackhole 单芯片环境中加载真实权重并完成全量映射验证：357 个源 tensor、15 个实际 mode block（txt/image/image_cond 各 5 个）和 28 个 single block。

## 7. 已完成验证

在 10.12.11.2 上已验证：

```text
模型配置读取：通过
manifest 校验：通过
357 tensor 校验：通过
Transformer 全量权重映射：通过（FULL_MAPPING_OK）
Qwen3-VL 权重映射：通过（36 层，8 个配置激活层）
VAE 权重映射：通过（251 个 safetensors tensor）
T2I 请求契约校验：通过
Python syntax compile：通过
```

当前 `pipeline.run()` 仍会明确提示文本编码执行、scheduler、latent、denoise loop 和 VAE 输出尚未串联，
不会把“权重映射通过”伪装成“端到端图片生成完成”。

## 8. 当前阻塞点

10.12.11.2 宿主机 Python 环境检查结果：

```text
torch：不可用
safetensors：不可用
ttnn：可用
```

直接在宿主机导入 TT-Metal 模型模块会遇到：

```text
ModuleNotFoundError: No module named 'torch'
```

这表示宿主机没有完整模型运行环境，不代表模型文件不存在或硬件不支持。后续必须在 Docker
环境中验证真实权重加载和设备执行。

## 9. 尚未完成的适配工作

### 9.1 权重映射

以下模型组件的导出权重到 TT-NN module state 映射已经完成：

- `emb_in.image`、`txt_in`、`time_in`；
- early/single modulation；
- mode、text mode、single blocks；
- final layer；
- Q/K/V 合并和 per-head RMSNorm；
- SwiGLU fused 权重布局；
- 1/2/4 芯片权重分片仍待实现和验证。

已实测结果：DiT 357 个源 tensor、TTNN 271 个参数条目；Qwen3-VL 36 层；VAE 251 个源 tensor。

### 9.2 Qwen3-VL

建议先允许外部传入 20480 维 `prompt_embeds`，完成 DiT/VAE bring-up，再单独实现 Qwen3-VL
TT-NN encoder。这样可以避免文本编码器、DiT 和 VAE 同时出问题。

### 9.3 Scheduler、latent 和 VAE

需要实现并固定：seed、Gaussian latent、patchify/unpatchify、timestep、sigma、flow matching
更新、denoise step、latent scaling 和 VAE decode。

### 9.4 tt-media-server

底层 pipeline 可以独立出图后，再增加：

```text
TTFlux3Runner
runner_fabric 注册
model constants/settings
模型到 runner 映射
图像 API 参数适配
服务启动配置
单元测试和 curl 测试
```

## 10. 推荐恢复顺序

```text
1. 构建本地 TL-inference-server + tt-metal Docker 镜像
2. 确认容器内 torch/ttnn/safetensors 版本
3. 单 block 权重加载
4. 单 block forward
5. 5 个 mode block forward
6. 28 个 joint block forward
7. 外部 prompt embedding 的完整 DiT denoise
8. 复用/适配 VAE decode
9. 输出第一张图片
10. 接入 Qwen3-VL
11. 接入 TTFlux3Runner
12. 接入 HTTP API
13. 扩展 2 芯片和 4 芯片
14. 精度、性能和服务稳定性测试
```

## 11. 暂停时的准确状态

```text
模型资产和配置：已确认
T2I 请求边界：已确定
FLUX.3 TT-NN transformer 初版：已加入
代码 syntax compile：已通过
真实权重加载：已完成（DiT/Qwen3-VL/VAE 映射级验证）
真实设备推理：已完成设备初始化和映射路径执行；端到端 denoise 仍未完成
最终图片生成：未完成
tt-media-server API：未接入
2/4 芯片扩展：未开始
性能优化：未开始
```

恢复开发时，从第 10 节开始即可。暂停不会删除当前分支、未提交代码、模型文件或文档。

## 2026-11-03 v3 本地源码镜像

已使用 10.12.11.2 上的本地源码重新构建镜像：

```text
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v3
构建来源：/local/lnn/TL-inference-server + /local/lnn/tt-metal
tt-metal：23ef0fd19b
模型权重：未复制进镜像，运行时只读挂载
镜像大小：约 8.30 GB
```

构建命令使用 BuildKit 的 `--build-context tt-metal-local=/local/lnn/tt-metal`，并通过
`TT_METAL_COMMIT_SHA_OR_TAG=23ef0fd19b` 写入镜像环境变量。镜像内已检查：

- `flux3_components.py` 包含 Qwen3-VL 和 VAE 权重加载入口；
- `transformer_flux3.py` 包含完整 DiT 映射代码；
- `/local/models/flux-3-image-klein-base/model.safetensors` 不存在于镜像内。

## 2026-11-03 v3 单芯片真实映射复核

使用上述 v3 镜像、物理芯片 `0`、TTNN mesh `(1,1)`，并挂载 `/dev/tenstorrent`、
`/dev/hugepages` 和 `/dev/hugepages-1G` 后重新执行三类权重加载。原始日志保存在：

```text
/local/lnn/flux3-v3-dit-map.log
/local/lnn/flux3-v3-text-map.log
/local/lnn/flux3-v3-vae-map.log
```

实际结果：

```text
FULL_MAPPING_V3_OK {
  'source_tensors': 357,
  'ttnn_parameter_entries': 271,
  'mode_blocks': 15,
  'single_blocks': 28
}

TEXT_ENCODER_V3_OK Qwen3VlTextEncoder [4, 8, 12, 16, 20, 24, 28, 32]

VAE_MAPPING_V3_OK {'source_tensors': 251}
```

三次容器运行均记录 `STATUS=0`。其中 Qwen3-VL 实际转换 326 个 TTNN tensor，
VAE 实际转换 168 个 TTNN tensor；源文件 tensor 数量分别按模型文件统计。

## 2026-11-03 持久化缓存改造

### 1. 目标

将模型权重转换结果、TT-Metal/TTNN kernel 编译结果以及 Python/Hugging Face/
Torch 等运行缓存统一放在宿主机 `/local/lnn/fluxcache`。容器启动时把该目录挂载为
`/local/fluxcache`，这样删除或重建容器不会丢失已转换的权重和已编译 kernel。

### 2. 代码改动

1. `tt-metal/models/tt_dit/models/transformers/transformer_flux3.py`
   - 保存 `parallel_config`；
   - 将 357 个 FLUX.3 DiT 源 tensor 组织成 TTNN state dict；
   - 使用 `cache.load_model(model_name="flux-3-image-klein", subfolder="transformer")`；
   - 首次运行从 safetensors 转换并写 cache，后续运行直接 `Module.load()`。
2. `tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py`
   - VAE decoder 保存 device 和 parallel config；
   - 使用 `cache.load_model(model_name="flux-3-image-klein", subfolder="vae")`。
3. `tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py`
   - `create_vae_decoder()` 支持传入显式 parallel config，保证缓存 key 与设备配置一致。
4. `TL-inference-server/tt-media-server/Dockerfile`
   - 增加 `FLUXCACHE_DIR=/local/fluxcache`；
   - 设置 `TT_DIT_CACHE_DIR`、`TT_METAL_CACHE`、`TT_METAL_KERNEL_CACHE_PATH`、
     `HF_HOME`、`TORCH_HOME`、`XDG_CACHE_HOME`、`TMPDIR` 等环境变量指向该根目录。

### 3. 宿主机目录布局

```text
/local/lnn/fluxcache/
├── tt-metal-cache/   # TT-Metal kernel/JIT 编译缓存
├── tt_dit_cache/     # TTNN 转换后的 transformer、VAE、text_encoder 权重
├── hf/               # Hugging Face 下载缓存
├── torch/            # PyTorch 缓存
├── xdg/              # XDG 通用缓存
├── triton/           # Triton 缓存（如使用）
├── mpl/              # Matplotlib 缓存
├── tmp/              # 运行时临时文件
├── logs/             # 验证日志
└── outputs/          # 输出图片/结果
```

`TT_DIT_CACHE_DIR` 的实际目录还会按并行配置、mesh shape 和 dtype 自动展开，不能
手工把不同 mesh 的缓存混在一起。例如单芯片缓存会位于类似：

```text
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein/transformer/<parallel>mesh1x1_bf16/
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein/vae/<parallel>mesh1x1_bf16/
```

### 4. 首次运行与后续运行

- cold run：没有 `cache_dict.json` 时读取模型 safetensors，执行权重转换、设备写入、
  TTNN/TT-Metal kernel 编译并将结果写入 `/local/lnn/fluxcache`。
- warm run：相同模型、相同 parallel config、相同 mesh 和 dtype 下，日志应出现
  `loading cache at ...`，直接读取 `.tensorbin`；kernel 也从 `tt-metal-cache` 复用。
- 如果修改模型权重、模型结构、mesh 或并行配置，应使用新的 cache 子目录或清理对应
  子目录，避免错误复用。

### 5. 当前验证边界

截至本次改造，已完成缓存机制接入和镜像配置；待 v4 镜像完成 cold/warm 两次真实运行
后补充具体 cache 命中日志和耗时对比。缓存机制本身不等于端到端出图，prompt/tokenizer、
scheduler、latent、patch/rope、denoise loop、VAE decode、PNG 输出和图像 API 仍未完成。

## 2026-11-03 v4 镜像及缓存真实验证

### 镜像结果

```text
image: tl-inference-server:flux-3-image-klein-local-20261103-v4
image_id: sha256:cd2badff9ea0d87fb3722c52261a795dfa2150955fb28d3ef981de1c43ac92b3
size: 8,298,064,005 bytes（约 8.30 GB）
tt-metal: 23ef0fd19b
build source: /local/lnn/TL-inference-server + /local/lnn/tt-metal
```

模型权重仍未复制进镜像，运行时挂载：

```text
/local/models/flux-3-image-klein-base:/local/models/flux-3-image-klein-base:ro
```

### 启动时的关键点

必须把宿主缓存挂载为 `/local/fluxcache`，并设置：

```bash
-e CACHE_ROOT=/local/fluxcache \
-v /local/lnn/fluxcache:/local/fluxcache
```

`CACHE_ROOT` 用于让镜像 entrypoint 自动为挂载卷设置容器用户可写的 group 权限。
如果只设置 `TT_DIT_CACHE_DIR` 而不设置 `CACHE_ROOT`，entrypoint 不会处理权限，
可能在 `/local/fluxcache/tt-metal-cache` 下面出现 `Permission denied`。本次验证中还
修复了宿主机子目录 `tt-metal-cache` 的遗留 `root:root` 所有权，最终统一为 UID/GID
`1000:1000`（主机用户 `tt`）。

### Cold/warm 证据

| 组件 | cold 结果 | warm 结果 | 缓存目录 |
|---|---|---|---|
| FLUX.3 DiT | 17.5 s，357 源 tensor，271 TTNN 参数 | 3.4 s，`1 HIT / 0 MISS`，JIT `6/6 hits` | `tt_dit_cache/flux-3-image-klein/transformer/CP1_0_TP1_0_SP1_0_mesh1x1_bf16` |
| Qwen3-VL text encoder | 11.9 s，326 TTNN tensor | 2.4 s，`1 HIT / 0 MISS`，JIT `6/6 hits` | `tt_dit_cache/flux-3-image-klein/text_encoder/TP1_0_mesh1x1_bf16` |
| FLUX.3 VAE | 7.0 s，168 TTNN tensor | 0.0 s，`1 HIT / 0 MISS`，JIT `6/6 hits` | `tt_dit_cache/flux-3-image-klein/vae/mesh1x1_bf16` |

日志文件：

```text
/local/lnn/fluxcache/logs/flux3-image-klein-v4-build.log
/local/lnn/fluxcache/logs/flux3-v4-dit-cold.log
/local/lnn/fluxcache/logs/flux3-v4-dit-warm.log
/local/lnn/fluxcache/logs/flux3-v4-text.log
/local/lnn/fluxcache/logs/flux3-v4-text-warm.log
/local/lnn/fluxcache/logs/flux3-v4-vae.log
/local/lnn/fluxcache/logs/flux3-v4-vae-warm.log
```

本次验证证明的是“模型组件权重转换和缓存可复用”，不是端到端 T2I 出图。日志中的
`No Physical Grouping Descriptor`、`TT_CLUSTER_NAME not set` 和 shared-memory warning
未阻止单芯片 `(1,1)` 映射通过，但多芯片运行前仍应补齐相应部署配置。



## 2026-11-03 实际验证更新

### Transformer 完整权重映射

已根据 /local/models/flux-3-image-klein-base/model.safetensors 的实际
357 个 key/shape 实现逐块加载，不再只验证第一个 block。加载器使用
safetensors.safe_open 按模块读取，避免额外 materialize 一份 11 GB 权重。

验证使用 tl-inference-server:flux-3-image-klein-local-20261103，运行时只读
挂载宿主机的 transformer_flux3.py 和模型目录；设备为 Blackhole Galaxy 的
(1,1) mesh。最终日志：

    FULL_MAPPING_OK {'source_tensors': 357, 'ttnn_parameter_entries': 271,
                     'mode_blocks': 15, 'single_blocks': 28}

这里的 15 个 mode block 是三组各 5 个：txt、image、image_cond；其中 T2I
主路径使用 txt 和 image，image_cond 为参考图/条件图路径。每个 block 的
Q/K/V 三个源权重在 TTNN 中合并为一个 to_qkv 参数，因此 357 个源 tensor
对应 271 个 TTNN 参数条目。attn_out 已接入 attention 输出投影，single
block 不再使用 pre_only=True。

### 当前边界

已完成：配置校验、单芯片设备初始化、所有 transformer 权重映射、三组 5/5
mode block 和 28/28 single block 的 TTNN 权重转换。

尚未完成：Qwen3-VL 在 TT 上的文本编码、scheduler/latent/denoise loop 的
端到端串联、FLUX.3 VAE decode、图片输出和 /v1/images/generations 服务化。
因此当前不能称为完整端到端 T2I 出图。


## 2026-11-03 T2I pipeline 组件权重映射更新

在 Transformer 全量映射之外，已补齐并实测两个 pipeline 组件：

| 组件 | 实际实现/权重 | 实测结果 |
|---|---|---|
| Qwen3-VL text encoder | `text_encoder/model-*.safetensors`，36 层，使用配置中的 8 个激活层 | `TEXT_ENCODER_MAPPING_OK 36 (4, 8, 12, 16, 20, 24, 28, 32)` |
| FLUX3 VAE decoder | `vae/diffusion_pytorch_model.safetensors` | `VAE_MAPPING_OK {'source_tensors': 251}` |
| FLUX3 DiT transformer | `model.safetensors` | `FULL_MAPPING_OK {'source_tensors': 357, ...}` |

新增 `pipelines/flux3/flux3_components.py`，并在 `pipeline_flux3.py` 中提供：

- `create_text_encoder()`；
- `create_vae_decoder()`；
- `load_vae_weights()`；
- `load_transformer_weights()`。

因此完整 T2I pipeline 的模型组件权重已经可以分别映射到 TT-NN。当前仍未宣称
端到端出图完成：prompt processor/tokenizer、时间步/latent/patch/rope、scheduler
denoise loop、VAE decode 和 server API 的串联仍需下一步验证。
