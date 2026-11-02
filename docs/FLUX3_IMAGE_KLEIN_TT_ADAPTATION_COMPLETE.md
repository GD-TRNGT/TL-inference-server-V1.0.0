# FLUX.3 Image Klein 在 TT-Metal/TT-inference-server 上的适配记录

更新时间：2026-11-02  
状态：阶段性暂停。当前代码和文档保留，但尚未完成端到端出图。

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
commit：5ebf2d8a0d04ce8801adb8deec07198c9c686775
origin：https://github.com/GD-TRNGT/TL-inference-server-V1.0.0.git
```

负责服务生命周期、worker、runner、API、请求调度、设备初始化和输出。

### 2.3 tt-metal

```text
路径：/local/lnn/tt-metal
分支：flux-3-image-klein
commit：3c8681218b0951637991d20ce6b7e341df3b6879
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

尚未完成：Qwen3-VL、scheduler、latent、denoise loop、VAE decode 和图片输出。

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

该文件已通过 Python syntax compile 检查，但尚未在真实 Docker 运行时加载权重并执行设备计算。

## 7. 已完成验证

在 10.12.11.2 上已验证：

```text
模型配置读取：通过
manifest 校验：通过
357 tensor 校验：通过
关键权重名校验：通过
T2I 请求契约校验：通过
Python syntax compile：通过
```

当前 `pipeline.run()` 会明确提示 transformer、Qwen3-VL、scheduler、latent 和 VAE 尚未接入，
不会伪装成已经完成推理。

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

需要完成以下导出权重到 TT-NN module state 的映射：

- `emb_in.image`、`txt_in`、`time_in`；
- early/single modulation；
- mode、text mode、single blocks；
- final layer；
- Q/K/V 合并和 per-head RMSNorm；
- SwiGLU fused 权重布局；
- 1/2/4 芯片权重分片。

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
真实权重加载：未完成
真实设备推理：未完成
最终图片生成：未完成
tt-media-server API：未接入
2/4 芯片扩展：未开始
性能优化：未开始
```

恢复开发时，从第 10 节开始即可。暂停不会删除当前分支、未提交代码、模型文件或文档。

