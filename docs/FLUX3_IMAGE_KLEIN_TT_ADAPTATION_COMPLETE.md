# FLUX.3 Image Klein 在 TT-Metal / TL-inference-server 上的实现记录

更新时间：2026-11-03  
当前结论：已完成 FLUX.3 的模型配置校验、TTNN 模型结构、DiT/Text Encoder/VAE 权重映射、单芯片真实设备加载以及持久化缓存验证；尚未完成端到端 T2I 出图和 HTTP 服务化。

本文只记录实际完成的实现内容、验证方法和当前未完成项，不记录历史版本的重复过程。

## 1. 实现环境

### 1.1 硬件和模型

```text
主机：10.12.11.2（my01-galaxy02）
设备：Blackhole Galaxy，32 芯片
本阶段使用：物理芯片 0，TTNN mesh=(1,1)，batch=1
模型：/local/models/flux-3-image-klein-base
```

模型目录中的主要资产：

```text
model.safetensors                         # FLUX.3 Transformer
text_encoder/model-*.safetensors         # Qwen3-VL Text Encoder
vae/diffusion_pytorch_model.safetensors  # VAE
config.json
manifest.json
```

已确认的 DiT 结构：

```text
hidden size：3072
attention heads：24
head dim：128
mode depth：5
single depth：28
image channels：128
text channels：20480
dtype：BF16
```

### 1.2 本地代码仓库

```text
/local/lnn/TL-inference-server
/local/lnn/tt-metal
```

两个仓库均使用 `flux-3-image-klein` 分支。`TL-inference-server` 负责服务、worker、runner、设备生命周期和 API；`tt-metal` 负责 TTNN 模型结构、权重加载、设备执行、kernel 和缓存。FLUX.3 的模型适配不能只修改 server 仓库。

## 2. 第一步：确认模型资产和实现边界

### 做了什么

1. 读取 `config.json`，确认 FLUX.3 的层数、隐藏维度、attention 配置和文本条件维度。
2. 读取 `manifest.json`，确认模型类型为 `f3_dit_export`，DiT 权重包含 357 个 source tensor。
3. 检查 DiT、Qwen3-VL 和 VAE 的 safetensors 文件及 tensor key。
4. 先把目标限制为单芯片、BF16、batch=1 的 T2I bring-up。

### 代码和执行入口

这一步由以下代码负责，不是人工查看文件后直接写结论：

```text
实现文件：
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_config.py

pipeline 调用入口：
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

`flux3_config.py` 中的 `validate_checkpoint()` 读取 `config.json` 和 `manifest.json`，检查模型名、导出类型、dtype、关键结构参数和权重文件是否存在，并返回 `Flux3Config`。`Flux3T2IPipeline.__init__()` 调用这个函数，把结果保存到 `self.config`。

实际验证入口为：

```text
/local/lnn/flux3_full_mapping_test.py
/local/lnn/flux3_text_mapping_test.py
/local/lnn/flux3_vae_mapping_test.py
```

这三个脚本都会使用 `/local/models/flux-3-image-klein-base`，分别触发 DiT、Text Encoder 和 VAE 的构建，从而验证模型目录布局和配置是否满足实现边界。模型权重不是在 server 仓库中生成的，而是通过运行时只读挂载进入容器。

### 作用

这一步用来确定真实权重和真实硬件配置，避免直接复用 FLUX.2 的结构或把“能读取文件”误认为“模型已经适配”。先固定单芯片也便于定位结构、权重和算子问题。

### 状态

**已完成。**

## 3. 第二步：建立 FLUX.3 的 TTNN Transformer 结构

### 修改位置

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
```

### 执行入口

实际验证不是只 import 这个文件，而是由下面的脚本实例化 `Flux3Transformer` 并调用 `load_flux3_checkpoint()`：

```text
/local/lnn/flux3_full_mapping_test.py
```

调用关系：

```text
flux3_full_mapping_test.py
  -> Flux3Transformer(...)
  -> model.load_flux3_checkpoint(model.safetensors)
  -> transformer_flux3.py 中的权重映射和 TTNN 参数写入
```

### 做了什么

新增并实现以下模块：

```text
Flux3ModeBlock
Flux3SingleBlock
Flux3Transformer
```

具体实现包括：

1. image/text/image-condition 输入投影；
2. mode block 的 attention、FFN 和调制逻辑；
3. joint token 流的 single block；
4. early/single shift-scale-gate modulation；
5. TTNN attention、SwiGLU、动态 LayerNorm 和输出投影；
6. 5 个 text mode block、5 个 image mode block、5 个 image-condition mode block；
7. 28 个 joint single block。

### 结构数量说明

```text
15 个 mode block = text 5 个 + image 5 个 + image_condition 5 个
28 个 single block = joint token 流上的 28 层
```

T2I 主路径主要使用 text 和 image 两组 mode block；`image_condition` 是为条件图/参考图路径保留的结构。这里的 15 是实际构建的三组 block 总数，不是 15 层同一种 block。

### 作用

把 FLUX.3 参考实现中的 Transformer 计算图转换为 TTNN 模块，使后续可以把真实 safetensors 权重放入对应的 TTNN 参数，并在 Tenstorrent 设备上执行。

### 状态

**结构和模块构建已完成。**完整的输入预处理、时间步、RoPE、scheduler 和 denoise loop 尚未接入，因此这一步不等于完整 T2I 推理完成。

## 4. 第三步：实现 DiT 权重映射

### 实现和执行位置

实现文件：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
```

验证脚本：

```text
/local/lnn/flux3_full_mapping_test.py
```

脚本中的核心调用是：

```python
model = Flux3Transformer(...)
report = model.load_flux3_checkpoint(
    "/local/models/flux-3-image-klein-base/model.safetensors"
)
```

### 做了什么

1. 使用 safetensors 读取真实 `model.safetensors` 的 key 和 shape。
2. 按 `Flux3Transformer`、mode block、single block 的层级，将 source tensor 映射到 TTNN module state。
3. 将参考实现中的 Q、K、V 权重按 TTNN attention 所需格式合并到 `to_qkv`。
4. 按 TTNN 线性层布局转换 SwiGLU 权重。
5. 映射 attention 输出投影、输入投影、调制参数和 final layer。
6. 使用 `safe_open` 按 tensor 读取，避免把约 11 GB 权重一次性复制到额外内存。

### 实际结果

```text
source tensors：357
TTNN parameter entries：271
mode blocks：15
single blocks：28
结果：FULL_MAPPING_OK
```

357 个 source tensor 和 271 个 TTNN 参数条目数量不同，是因为部分 source tensor 在 TTNN 中按算子布局合并或重组，例如 Q/K/V 合并为 `to_qkv`。

### 作用

验证真实 FLUX.3 权重是否可以完整进入 TTNN 模型。只有这一步正确，后续的 block forward、数值对齐和性能测试才有意义。

### 状态

**单芯片 BF16 权重映射已完成并在真实设备上验证。**2 芯片和 4 芯片的权重分片、通信和并行映射尚未完成。

## 5. 第四步：接入 Qwen3-VL Text Encoder

### 修改位置

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

### 执行入口

组件构建和权重加载实现位于：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
```

实际验证脚本：

```text
/local/lnn/flux3_text_mapping_test.py
```

调用关系：

```text
flux3_text_mapping_test.py
  -> build_flux3_text_encoder(...)
  -> flux3_components.py 读取 text_encoder/config.json 和 manifest.json
  -> load_flux3_text_state_dict(...)
  -> cache.load_model(...)
  -> Qwen3VlTextEncoder
```

### 做了什么

1. 增加 Qwen3-VL Text Encoder 的构建入口。
2. 读取 text encoder safetensors 权重并映射到 TTNN 参数。
3. 按配置加载 36 层 encoder 和激活层 `[4, 8, 12, 16, 20, 24, 28, 32]`。
4. 在 pipeline 中提供统一的 `create_text_encoder()` 入口。

### 实际结果

```text
encoder layers：36
配置激活层：4, 8, 12, 16, 20, 24, 28, 32
实际转换 TTNN tensor：326
结果：TEXT_ENCODER_MAPPING_OK
```

### 作用

为后续 `prompt -> Qwen3-VL -> text embedding -> DiT` 串联准备模型组件和权重路径。

### 状态

**权重映射已完成。**实际 tokenizer/processor、prompt 编码调用和 Text Encoder 的完整 TTNN 前向输出还没有接入 T2I pipeline。

## 6. 第五步：接入 FLUX.3 VAE Decoder

### 做了什么

1. 在 `flux3_components.py` 中增加 VAE decoder 构建入口。
2. 读取 `vae/diffusion_pytorch_model.safetensors`。
3. 将 VAE 权重映射到 TTNN module。
4. 在 `pipeline_flux3.py` 中增加 `create_vae_decoder()` 和 `load_vae_weights()`。
5. 显式传入 device 和 parallel config，保证 VAE 缓存 key 与实际设备配置一致。

### 执行入口

实现文件：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
```

实际验证脚本：

```text
/local/lnn/flux3_vae_mapping_test.py
```

调用关系：

```text
flux3_vae_mapping_test.py
  -> build_flux3_vae_decoder(...)
  -> load_flux3_vae_weights(...)
  -> 读取 vae/diffusion_pytorch_model.safetensors
  -> decoder.load_torch_state_dict(..., strict=True)
```

### 实际结果

```text
VAE source tensors：251
实际转换 TTNN tensors：168
结果：VAE_MAPPING_OK
```

### 作用

为 `denoised latent -> VAE decoder -> image tensor` 准备模型组件。

### 状态

**权重映射已完成。**VAE 尚未与完整 denoise loop 串联，也尚未完成 PNG/JPEG 编码和 HTTP 输出。

## 7. 第六步：建立 T2I pipeline 外壳

### 修改位置

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
```

### 做了什么

1. 增加 `Flux3T2IPipeline`。
2. 增加 T2I 请求契约校验。
3. 在 pipeline 中统一创建 Transformer、Text Encoder 和 VAE 组件。
4. 固定首版请求边界：单 prompt、单输出、batch=1、BF16、固定 seed/steps/guidance、默认 1024×1024。
5. 对尚未实现的执行链路明确报出未完成，而不是伪装成已经生成图片。

### 执行入口

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/t2i_contract.py
```

调用关系：

```text
Flux3T2IPipeline.__init__()
  -> validate_checkpoint()
  -> Flux3Config

Flux3T2IPipeline.prepare_request()
  -> Flux3T2IRequest.validate()

Flux3T2IPipeline.create_transformer()
  -> Flux3Transformer
```

当前 `pipeline.run()` 只到组件创建和请求边界检查，尚未进入完整采样执行；这是文档标记“端到端执行未完成”的直接代码依据。

### 作用

把三个模型组件放到同一个 T2I 入口下，为后续逐步填入预处理、采样和解码逻辑提供稳定边界。

### 状态

**pipeline 外壳和组件创建入口已完成；端到端执行未完成。**

## 8. 第七步：实现持久化权重和 JIT 缓存

### 修改位置

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py
```

### 实现位置和验证入口

缓存实现分布在：

```text
/local/lnn/tt-metal/models/tt_dit/models/transformers/transformer_flux3.py
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
```

关键调用是 `cache.load_model(...)`。验证仍使用三个真实设备加载脚本：

```text
/local/lnn/flux3_full_mapping_test.py
/local/lnn/flux3_text_mapping_test.py
/local/lnn/flux3_vae_mapping_test.py
```

第一次运行触发 safetensors 到 TTNN 的转换和 kernel 编译；第二次在相同 mesh、dtype 和 parallel config 下再次运行同一个脚本，检查日志中的 `HIT` 和 `JIT cache hits`。

### 做了什么

1. DiT、Text Encoder 和 VAE 使用 `cache.load_model()` 保存/加载 TTNN 权重。
2. 将宿主机缓存统一放到：

   ```text
   /local/lnn/fluxcache
   ```

3. 容器内挂载为 `/local/fluxcache`。
4. 按模型、组件、mesh、dtype 和 parallel config 区分缓存 key。
5. TT-Metal kernel/JIT、TTNN 权重、Hugging Face、Torch、XDG 和临时文件分别落到缓存子目录。

### 缓存目录

```text
/local/lnn/fluxcache/
├── tt-metal-cache/
├── tt_dit_cache/
├── hf/
├── torch/
├── xdg/
├── mpl/
├── tmp/
├── logs/
└── outputs/
```

### 作用

容器删除或重建后不必重新转换几十 GB 权重，也不必重复编译已经生成的 kernel。不同 mesh 或并行配置使用不同缓存 key，避免错误复用。

### 实际 cold/warm 结果

| 组件 | cold | warm |
|---|---:|---:|
| DiT | 17.5 秒 | 3.4 秒 |
| Qwen3-VL Text Encoder | 11.9 秒 | 2.4 秒 |
| VAE | 7.0 秒 | 0.0 秒 |

warm run 中验证到：

```text
weight cache：1 HIT / 0 MISS
JIT cache：6/6 hits
缓存目录：约 19 GB
```

### 状态

**已完成并在真实设备上验证。**

## 9. 第八步：修改 Dockerfile 并使用本地仓库构建镜像

### 修改位置

```text
/local/lnn/TL-inference-server/tt-media-server/Dockerfile
```

### 执行命令

实际构建使用本地两个仓库作为源码输入：

```bash
cd /local/lnn/TL-inference-server

docker buildx build \
  --load \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=23ef0fd19b \
  -f tt-media-server/Dockerfile \
  -t tl-inference-server:flux-3-image-klein-local-20261103-v4 \
  .
```

这里的 `--build-context tt-metal-local` 决定镜像使用 `/local/lnn/tt-metal` 的本地代码；模型目录没有放入镜像，而是在容器启动时挂载。

### 做了什么

1. 使用 BuildKit 的 `--build-context` 把本地 `/local/lnn/tt-metal` 作为构建上下文传入。
2. 不把模型权重复制到镜像；模型通过运行时只读挂载。
3. 将 `TT_DIT_CACHE_DIR`、`TT_METAL_CACHE`、`TT_METAL_KERNEL_CACHE_PATH`、`HF_HOME`、`TORCH_HOME`、`XDG_CACHE_HOME` 和 `TMPDIR` 指向 `/local/fluxcache`。
4. 设置 `CACHE_ROOT=/local/fluxcache`，使 entrypoint 可以处理挂载缓存目录权限。

### 构建结果

```text
镜像：tl-inference-server:flux-3-image-klein-local-20261103-v4
镜像大小：8,298,064,005 bytes，约 8.30 GB
模型权重：未复制进镜像
```

镜像中的源码来自本地两个仓库；运行时需要挂载：

```text
/local/models/flux-3-image-klein-base:/local/models/flux-3-image-klein-base:ro
/local/lnn/fluxcache:/local/fluxcache
```

### 作用

确保镜像中使用本地修改后的 TT-Metal 和 TL-inference-server，同时把大模型权重和可复用缓存放在宿主机，降低重建镜像的成本。

### 状态

**本地源码镜像已构建。**镜像只验证模型组件加载和缓存路径，不代表 server 已经提供 FLUX.3 T2I API。

## 10. 第九步：真实 Blackhole 单芯片验证

### 执行脚本

容器内执行的三个验证入口分别是：

```text
/local/lnn/flux3_full_mapping_test.py
/local/lnn/flux3_text_mapping_test.py
/local/lnn/flux3_vae_mapping_test.py
```

它们不是模拟测试：每个脚本都调用 `ttnn.open_mesh_device(ttnn.MeshShape(1, 1))`，创建真实 TTNN 设备，完成后调用 `ttnn.close_device(device)`。

### 验证方法

使用 v4 镜像启动容器，挂载模型、缓存、`/dev/tenstorrent`、hugepages，并选择物理芯片 0 和 `(1,1)` mesh。随后分别运行：

1. DiT 配置读取和完整权重映射；
2. Qwen3-VL Text Encoder 权重映射；
3. VAE 权重映射；
4. 相同参数下再次运行 warm load，检查 cache hit 和 JIT hit。

### 验证结果

```text
配置读取：通过
manifest 校验：通过
DiT：357 source tensors，271 TTNN 参数，FULL_MAPPING_OK
mode blocks：15/15
single blocks：28/28
Qwen3-VL：36 层，TEXT_ENCODER_MAPPING_OK
VAE：251 source tensors，VAE_MAPPING_OK
设备初始化：通过
cold/warm cache：通过
```

### 作用

确认代码不是只在静态 import 或 CPU 环境中通过，而是能够在真实 Blackhole 设备上初始化 TTNN、转换权重、写入设备并复用缓存。

### 状态

**单芯片模型组件加载验证已完成。**

## 11. 当前未实现部分

以下内容不能写成“已完成”，是下一阶段工作：

| 部分 | 当前状态 | 需要完成的内容 |
|---|---|---|
| tokenizer/processor | 未实现 | 将 prompt 转为 Qwen3-VL 输入并生成文本条件 |
| Text Encoder 前向 | 未串联 | 在 TTNN 上执行 prompt 编码并输出 20480 维条件 |
| latent 初始化 | 未实现 | 按 seed 生成 Gaussian latent，并固定形状和 dtype |
| patchify/unpatchify | 未实现 | 在 latent 网格和 Transformer token 之间转换 |
| RoPE/time embedding | 未串联 | 生成位置编码和时间条件 |
| scheduler | 未实现 | 实现 flow-matching 的 timestep、sigma 和更新规则 |
| denoise loop | 未实现 | 执行全部 steps，并调用 5+28 个 block 的实际前向 |
| VAE decode 串联 | 未实现 | 将最终 latent 传入 VAE 并得到图像 tensor |
| 图片输出 | 未实现 | 将图像 tensor 转为 PNG/JPEG |
| tt-media-server runner | 未接入 | 增加 FLUX.3 runner、runner 注册和模型配置 |
| HTTP API | 未接入 | 增加 `/v1/images/generations` 及请求参数映射 |
| 同步/异步任务 | 未接入 | 接入 server 的同步、异步队列和任务状态 |
| cancel | 未接入 | 增加任务取消语义并处理中断状态 |
| 2 芯片/4 芯片 | 未实现 | mesh、权重分片、通信和多芯片缓存验证 |
| 精度对齐 | 未开始 | 与参考实现逐层比较 embedding、block、latent 和最终图片 |
| 性能优化 | 未开始 | kernel、memory、layout、trace、并发和吞吐优化 |

## 12. 当前准确结论

### 已实现

```text
FLUX.3 配置和 manifest 校验
FLUX.3 TTNN Transformer 结构
15 个 mode block 的结构和权重映射
28 个 single block 的结构和权重映射
357 个 DiT source tensor 的完整映射
Qwen3-VL Text Encoder 权重映射
FLUX.3 VAE 权重映射
单芯片 Blackhole 真实设备加载
权重 cache 和 JIT cache 持久化
本地 TL-inference-server + tt-metal 镜像构建
```

### 尚未实现

```text
完整 prompt 到图片的 T2I 执行链
真实 denoise loop
VAE decode 与 PNG 输出串联
tt-media-server runner/API
同步、异步和 cancel 接口
2 芯片和 4 芯片运行
精度对齐和性能优化
```

因此，当前成果应准确描述为：

> 已完成 FLUX.3 模型组件在 TTNN 上的单芯片权重映射、真实设备加载和持久化缓存验证；尚未完成可对外提供图片生成服务的端到端 FLUX.3 推理实现。

## 13. 相关验证日志和提交

缓存和验证日志位于：

```text
/local/lnn/fluxcache/logs/flux3-image-klein-v4-build.log
/local/lnn/fluxcache/logs/flux3-v4-dit-cold.log
/local/lnn/fluxcache/logs/flux3-v4-dit-warm.log
/local/lnn/fluxcache/logs/flux3-v4-text.log
/local/lnn/fluxcache/logs/flux3-v4-text-warm.log
/local/lnn/fluxcache/logs/flux3-v4-vae.log
/local/lnn/fluxcache/logs/flux3-v4-vae-warm.log
```

关键提交：

```text
tt-metal：91d937c036 Add persistent FLUX3 transformer and VAE TTNN caches
TL-inference-server：feda92e8 Complete FLUX3 adaptation status details
TL-inference-server：1f39d624 Document FLUX3 implementation steps and purposes
```
