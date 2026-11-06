# FLUX.3 多参考图与 TTNN VAE Encoder：完整实现过程

更新时间：2026-11-05。本文记录已落地的代码、验证过程、问题及解决方法，不把设计上限等同于全面实测。

## 1. 当前结论和交付边界

两个本地模型都声明支持 prompt 加最多 **10 张参考图**：

- `/local/models/flux-3-image-klein/README.md:28`：Distilled。
- `/local/models/flux-3-image-klein-base/README.md:23`：Base。

已开放有序 `reference_images` 数组，兼容原单图 API；已实现可选择的 TTNN VAE encoder。两个模型均在单芯片上实测：无图 T2I、旧单图、新数组单图、两图、交换两图顺序、10 张小图，以及两图异步任务。坏图片和 11 张参考图返回 422。

| 内容 | 状态 |
|---|---|
| API 接收 1～10 张有序参考图 | 已实现 |
| 单图与多图独立预处理规则 | 已实现 |
| TTNN encoder、权重映射、packing 和冻结 BN | 已实现 |
| 两套模型 encoder 数值与尺寸切换测试 | 含 1024 方图/横竖图，22/22 通过；缓存重新加载通过 |
| 两套模型同步/异步 HTTP 场景 | 18/18 通过；参考图数量实际覆盖 1、2、10 |
| 原 32-worker Distilled 服务 | 已恢复，8000 端口、32/32 ready、queue=0、canary healthy |
| 当前服务默认参考图 backend | `cpu`；TTNN 需设置 `FLUX3_REFERENCE_ENCODER_BACKEND=ttnn` |
| 10 张大图容量、性能及 32-worker 参考图压测 | 两模型各 14/14 矩阵、32 首次/32 暖态、200/200 已通过；最终 `all_passed=true`。详见 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md` |
| 自动 OOM 后切换 CPU | 未实现；目前只能显式选择 backend |
| 完整 DiT/最终图像逐层精度对齐 | 未完成；本轮只做 reference encoder token 数值比较 |

代码基线为 11.2 上两个仓库的当前 HEAD：

```text
TL-inference-server 9931070523c19dff7e32a15ebf65b91e105443c2
tt-metal           06f77cf6bc51ee3a63027ffc8e2c19125b51004c
分支               flux-3-image-klein
本轮功能           位于上述提交之上的未提交工作树
```

未修改模型权重，也未重建镜像。现有 v5 镜像提供运行环境，两仓库源码通过 bind mount 生效；脱离源码挂载单独运行旧镜像，不会获得本轮新功能。

## 2. 第一步：确认模型契约及原来的单图路径

核对上述 README，并检查参考实现 `/local/lnn/fluxcache/reference_pipeline.py` 的单图、多图策略。原服务已能将单张参考图通过 CPU PyTorch VAE 编码为真实 `image_cond`，不是伪造 latent。

参考图片进入 FLUX.3 的路径为：

```text
API 中的有序 Base64 图片
  -> PIL 解码、EXIF/RGB、缩放/填充或裁剪
  -> VAE encoder + quant_conv，取 posterior mean
  -> 2×2 latent packing + 冻结 BatchNorm
  -> [1, L, 128] image_cond token + 每图独立 RoPE
  -> TTNN DiT 多步 denoise
  -> VAE decoder -> PNG
```

核心 Qwen3-VL prompt encoder 仍只编码文字。可选的 prompt upsampler 看图并改写 caption 是另外一项功能，本轮未接入，不是参考图进入 DiT 的必要路径。

## 3. 第二步：增加 API 参数及入队前校验

修改文件：

```text
/local/lnn/TL-inference-server/tt-media-server/domain/image_generate_request.py
```

实现位置：

- `BaseImageRequest.reference_images`（第 32 行）：可选字符串列表，长度 1～10。
- `validate_reference_inputs()`（第 80 行）：只允许 FLUX.3 runner 使用数组，禁止与旧 `reference_image` 或 `image` 同时设置。
- 逐项检查非空、Base64/data URL 格式及 PIL 可识别的图片内容；拒绝 HTTP URL、非图片 data URL、解压炸弹及过小多图。
- 归一化空白、data URL 前缀大小写和缺失的 Base64 padding，使校验通过的数据也能被 worker 解码。
- 多图缩放后的任一边不足 16 像素时拒绝请求。

作用：在 FastAPI/Pydantic 入队前返回 422，避免把明显无效请求交给设备。旧单图字段保留原有兼容行为；新增的严格内容校验针对数组，不声称旧字段已全部统一改造。

多参考图使用 `POST /v1/images/generations`。现有 `/v1/images/image-to-image` 的 `image` 仍必填，不用于新数组请求。

修改 `/local/lnn/TL-inference-server/tt-media-server/tests/test_flux3_reference_api.py`，覆盖有图/无图、1/2/10/11 张、空数组、字段冲突、坏图片、尺寸边界、编码归一化及其他 runner 拒绝数组。最终 **17/17 通过**。

## 4. 第三步：保留顺序并接入 worker 生命周期

修改文件及位置：

```text
/local/lnn/TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py
TTFlux3ImageKleinRunner.run()，第 575 行起
```

实现过程：

1. 读取新数组，或把旧 `reference_image` / `image` 归一为一元素数组。
2. 按原顺序解码为 PIL 图片，放入 `Flux3T2IRequest.reference_images`。
3. worker 第一次收到参考图时，调用 `pipeline.create_reference_encoder(device=self.ttnn_device)`。
4. 保存到 `self._reference_encoder`，后续请求复用；纯 T2I 不触发 reference encoder 加载。
5. 日志记录参考图数量以及实际 `encoder=cpu|ttnn`。

作用：复用 worker 已占用的单芯片 MeshDevice，不为每张图片再打开一个设备；图片顺序不被重排。

## 5. 第四步：实现正确的单图与多图预处理

文件：

```text
/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/flux3_components.py
```

| 方法/位置 | 规则 | 作用 |
|---|---|---|
| `prepare_flux3_reference_image()`，第 32 行 | EXIF、RGB；单图画布最多 4×1024×1024 像素、最长边 4096；填充到 16 网格，最小画布边 64；反射/小图对称填充 | 保留既有单图兼容策略 |
| `prepare_flux3_multi_reference_image()`，第 64 行 | 每张最多 1024×1024 像素，等比缩小后中心裁剪到 16 网格；不足 16 的边拒绝 | 多图不能复用单图反射填充规则 |

这里限制的是预处理后的几何和像素数，不等于统一的上传字节上限，也不保证所有合法几何都能在单芯片显存内运行。

新增 `/local/lnn/tt-metal/models/tt_dit/tests/flux3/test_flux3_reference_preprocessing.py`；与既有 `test_reference_conditioning.py` 合计 **4/4 通过**，覆盖裁剪、面积缩放、过小输入和参考图位置轴。

## 6. 第五步：建立 TTNN encoder 结构及严格权重映射

新增文件：

```text
/local/lnn/tt-metal/models/tt_dit/models/vae/flux3_vae_encoder.py
```

### 6.1 网络结构

`Flux3VaeEncoder.__init__()`（第 141 行）限定单芯片 `(1,1)` mesh，构建：

```text
RGB NHWC -> conv_in
  -> 4 个 down block，通道 128/256/512/512，前 3 个下采样
  -> mid block -> GroupNorm+SiLU -> conv_out(64 通道)
  -> quant_conv(64 -> 64)
```

复用现有 VAE 卷积、ResNet/down/mid block；每个 down block 有两个 ResNet 层。只在本 encoder 内替换需要适配的 GroupNorm/Conv2d，不修改全局公共层的行为。

### 6.2 权重映射

`_prepare_torch_state()`（第 168 行）：

- 去除 `encoder.` 前缀，匹配本 encoder 模块路径。
- 将 `downsamplers.0` 映射到复用层的 `downsampler`。
- 将 `quant_conv.weight` 的 1×1 卷积维度 squeeze，映射为逐位置 Linear。
- 重命名 BN running mean/variance，移除推理不用的 `num_batches_tracked`。

`Flux3TTNNReferenceEncoder.__init__()` 位于 `flux3_components.py` 第 153 行起：检查 VAE 配置，从本地 safetensors **仅选择 encoder、quant_conv、BN 共 111 个源张量**，经 `cache.load_model()` 做转换和严格加载。

两模型 VAE 文件的完整 SHA256 相同：

```text
d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5
```

仍按模型目录名区分缓存，避免把两套模型的其他组件或配置混为一套。

### 6.3 encoder 输出、packing 和 BN

`Flux3VaeEncoder.forward()`（第 180 行）接收 `[1,H,W,3]`、范围 [-1,1] 的设备 tensor，要求长宽为正的 16 倍数。

1. encoder 下采样 8 倍。
2. quant_conv 输出 posterior 参数；取前 32 通道作为 mean，即 `latent_dist.mode()`，不重新抽样。
3. 将相邻 2×2 latent 打包为 128 通道。
4. 用 checkpoint 的 running mean/variance 执行冻结 BN：
   `(packed - running_mean) / sqrt(running_var + 1e-4)`。
5. 返回设备端 `[1,(H/16)×(W/16),128]` token。

例如 64×64 参考图产生 `[1,16,128]`；这里的 token 是图片条件特征，不是生成图的初始随机噪声。

## 7. 第六步：选择 backend 并保持条件 token 在设备端

实现位置：

- `flux3_components.py:Flux3TTNNReferenceEncoder.encode()`，第 191 行：PIL/NumPy 像素预处理仍在 CPU，上传 BF16 NHWC；encoder、packing、BN 在 TT，直接返回设备 token。参考几何变化时调用第 137 行 `_refresh_reference_program_cache()`，同步后释放旧形状运行时程序缓存，不删除磁盘 JIT/权重缓存；相同参考几何不清理。
- `/local/lnn/tt-metal/models/tt_dit/pipelines/flux3/pipeline_flux3.py:create_reference_encoder()`，第 87 行：选择 `cpu|ttnn`，默认 `cpu`；配置非法时报错。
- 同文件 `run_external_prompt_embeds()`，第 308 行起：逐图编码，构建独立 RoPE，并拼接条件。

CPU backend 返回 Torch token，只上传一次；TTNN backend 已返回设备 token，直接复用。多图用 `ttnn.concat(..., dim=1)` 拼接设备条件。

`make_reference_image_rope()` 为第 i 张图设置第四轴 `-(i+1)`，区分图片身份。RoPE 仍在 CPU 构造，随后上传。全部参考图的 token/RoPE 在 denoise loop 前准备并传入各步 DiT，不是每个去噪步骤重新编码参考图。

## 8. 第七步：处理实机执行和测试环境问题

| 出现阶段/部分 | 问题及现象 | 定位依据 | 修改及结果 |
|---|---|---|---|
| TTNN down block 的 GroupNorm | 64×64 下采样后 `Ht=2`，默认 8×8 核心网格不满足约束 | 复用层固定网格，短空间序列无法按原行数划分 | `_ReferenceGroupNorm.forward()`（当前第 32 行）按 `ceil(N×H×W/32)` 选择可整除核心行数，保持虚拟列数；全部测试尺寸通过 |
| 同一 encoder 更换输入尺寸 | 16×16 正常，随后 64×64 PCC 仅 0.03829394、relative RMSE 1.05319，执行不报错但数值失真 | 公共 `/local/lnn/tt-metal/models/tt_dit/layers/conv2d.py` 第 553 行附近的准备权重依赖输入几何，却只首次准备 | `_ReferenceConv2d.forward()`（当前第 66 行）记录 shape/layout/dtype/memory_config；变化时释放旧 prepared weight/bias，再由原始缓存权重准备；修复后 PCC 0.99988747，切回小图也正确 |
| 数值测试后的服务 kernel 编译 | 服务 UID 1000 无法写 root 创建的 kernel 子目录内 `chlkc_descriptors.h` | 数值容器绕过 entrypoint 以 root 运行，而服务用 `gosu` 切换 UID 1000 | 仅修复本轮相关缓存目录 ownership，后续数值容器和 HTTP 测试 exec 统一 UID 1000；未删缓存、未 reset |
| 首次 Base 临时服务启动 | 临时容器立即退出；初次 `--rm` 未保留完整 stderr | 源码只注册 `MODEL=FLUX.3-Image-Klein`，没有独立 Base 注册名 | 改用共同注册名，以权重目录选 Base；后续复制原配置并保留日志，两变体启动及请求通过；不把已丢失日志当作已证实证据 |
| 最后重跑主机型单元测试 | 缺少 `tests.scripts`；绕过根 conftest 后，默认 JUnit 报告又写入不可写目录 | 镜像挂载的是模型源码，未完整挂载 tt-metal 的根测试树；仓库 pytest.ini 默认报告路径受权限限制 | 仅对这组纯预处理/RoPE 测试使用 `--confcutdir` 和 `-o addopts=-q`，不改公共测试配置；最终 exit 0、4 passed |

CPU encoder 的说明注释也更新为“保留的兼容 backend”，不再写成“尚无 TTNN encoder”。

## 9. 第八步：持久化缓存与复用边界

统一存放在主机 `/local/lnn/fluxcache`。现有容器同时映射该目录到 `/local/fluxcache` 和 `/local/lnn/fluxcache`。

主要目录：

```text
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein-refvae-d64f3a68e1cc4f9f
/local/lnn/fluxcache/tt_dit_cache/flux-3-image-klein-base-refvae-d64f3a68e1cc4f9f
/local/lnn/fluxcache/tt-metal-cache
/local/lnn/fluxcache/logs/multiref_validation_20261105/20261105-054705
```

缓存 key 为模型目录名加 VAE SHA256 前 16 位，encoder 子目录为 `vae-encoder-v1`。明确区分：

- **原始权重转换缓存**：避免重复读 safetensors 和转换；可跨服务重启复用。
- **JIT/kernel 缓存**：已编译相同配置可复用；新几何仍可能产生新 kernel。
- **运行时 prepared 卷积权重**：依赖当前几何；尺寸改变必须重新准备，不能错误地沿用前一尺寸。

因此，“有缓存”不等于所有新尺寸都完全无需编译或运行时权重准备。数值测试另外完成一次 encoder 缓存重载，并记录 `REFERENCE_CACHE_RELOAD_OK`。

## 10. 第九步：两模型 encoder 数值验证

脚本：`/local/lnn/fluxcache/test_flux3_ttnn_vae_encoder.py`。

同一模型资产图片分别送 PyTorch CPU encoder 与 TTNN encoder。CPU 为 float32 计算后输出 BF16；TT 使用 BF16 输入及权重。比较的是 packing+BN 后的最终参考 token。

每个模型按如下顺序运行，专门覆盖尺寸变化和切回：

```text
16×16 -> 64×64 -> 128×96 -> 256×256 -> 512×384 -> 512×512 -> 1024×1024
-> 1024×768 -> 768×1024 -> 512×1024 -> 64×64
```

16×16 使用多图预处理策略，其他尺寸使用单图策略。本表“尺寸”为宽×高；不是最终生成图片分辨率。

| 输入尺寸 | token shape | PCC | MAE | relative RMSE |
|---|---|---:|---:|---:|
| 16×16 | [1,1,128] | 0.999931574 | 0.00881243 | 0.0117531 |
| 64×64 | [1,16,128] | 0.999887466 | 0.01147127 | 0.0151115 |
| 128×96 | [1,48,128] | 0.999867022 | 0.01240382 | 0.0164530 |
| 256×256 | [1,256,128] | 0.999857903 | 0.01277214 | 0.0170505 |
| 512×384 | [1,768,128] | 0.999804676 | 0.01461640 | 0.0199534 |
| 512×512 | [1,1024,128] | 0.999846339 | 0.01310325 | 0.0177671 |
| 1024×1024 | [1,4096,128] | 0.999818265 | 0.01389346 | 0.0191648 |
| 1024×768 | [1,3072,128] | 0.999800205 | 0.01478564 | 0.0204483 |
| 768×1024 | [1,3072,128] | 0.999798894 | 0.01463271 | 0.0199492 |
| 512×1024 | [1,2048,128] | 0.999842465 | 0.01334614 | 0.0180222 |

两套 VAE 权重相同，因此本次对应数值一致。22 次均 shape 正确、数值有限，并满足测试阈值 PCC≥0.997、relative RMSE<0.06。这是阶段验收阈值，不是完整模型的最终精度标准。当前使用 128 KiB L1_SMALL、encoder 大面积 RGB 卷积的显式高度切片及参考几何变化时的运行时程序缓存清理，详见压测报告中的修复记录；上轮 12 次结果是未扩展大方图/横竖图的历史结果。

最终成功结果：

```text
/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-075049/results.json
```

该目录根部的早期失败 JSON 只保留作问题追溯，不作为最终通过结果。

## 11. 第十步：同步、异步及兼容性端到端验证

单 worker、单芯片测试容器显式设置 TTNN backend。输入为 figurine/coffee 两张模型资产缩至 64×64；10 图使用两图交替，**不是 10 个不同主体**。输出 256×256、seed 42；Base 50 步 CFG 3.5，Distilled 12 步。

每模型 9 个场景，共 18 个：

| 场景 | Base 秒 | Distilled 秒 | 结果 |
|---|---:|---:|---|
| 无图 T2I | 9.309 | 1.792 | 200，PNG |
| 旧单图 `reference_image` | 16.257 | 2.152 | 200，PNG |
| 新数组单图 | 9.463 | 1.312 | 200；同模型 PNG 哈希与旧单图一致 |
| 两图 | 9.912 | 1.471 | 200，PNG |
| 两图交换顺序 | 9.828 | 1.384 | 200；与原顺序 PNG 哈希不同 |
| 10 图 | 10.966 | 1.714 | 200，PNG |
| 坏图片 | 0.052 | 0.049 | 422 |
| 11 图 | 0.004 | 0.004 | 422 |
| 两图异步，含轮询至完成 | 10.069 | 2.028 | 提交 202，completed，解码图片成功 |

这里是单次功能测试，混合冷暖状态、不同步数，不是公平的 CPU/TT 性能对比或热态均值。哈希相同证明旧新单图路径兼容；顺序改变影响结果，不单独证明参考语义和全模型数值正确。

结果目录：

```text
/local/lnn/fluxcache/multiref_ttnn_base_verified_20261105/
/local/lnn/fluxcache/multiref_ttnn_distilled_verified_20261105/
```

各目录保存 `results.json` 以及 `t2i.png`、`legacy_one.png`、`list_one.png`、`list_two.png`、`list_two_reversed.png`、`list_ten.png`、`async_two.png`。

以上是小图兼容性阶段，当时未逐项实测 3～9 张、大尺寸×10 图或 32-worker 参考图负载。随后 Base 与 Distilled 都完成完整 14 项矩阵及 32 首次/32 暖态/200 请求；当前大图结果以 `FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md` 为准，不将历史未验证条目当作当前结论。

## 12. 第十一步：启动参数和 curl 使用

服务已有镜像、源码挂载、权重及缓存配置保持不变。开启 TTNN reference encoder，需要在新建容器时加入：

```bash
-e MODEL=FLUX.3-Image-Klein \
-e MODEL_RUNNER=tt-flux.3-image-klein \
-e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
-e MODEL_WEIGHTS_PATH=/local/models/flux-3-image-klein \
-e FLUX3_REFERENCE_ENCODER_BACKEND=ttnn \
-e FLUX3_L1_SMALL_SIZE=131072 \
-e TT_MM_THROTTLE_PERF=0 \
-e TT_METAL_TDP_LIMIT_WATTS=160
```

Base 时将两个权重目录都改为 `/local/models/flux-3-image-klein-base`，`MODEL` 注册名不改。环境变量不能通过 `docker restart` 改写；本轮只在临时容器中开启 TTNN，已恢复的原容器仍保持默认 CPU backend。

以下是在 Linux 客户端执行的两图同步请求。文件从客户端读出并随请求上传，不是让服务器打开客户端路径：

```bash
set -o pipefail

python3 -c '
import base64, json, sys
paths = ["/path/to/reference1.png", "/path/to/reference2.jpg"]
refs = []
for path in paths:
    with open(path, "rb") as source:
        refs.append(base64.b64encode(source.read()).decode("ascii"))
json.dump({
    "prompt": "Combine the subject of the first picture with the setting of the second picture.",
    "reference_images": refs,
    "width": 256,
    "height": 256,
    "num_inference_steps": 12,
    "seed": 42,
    "response_format": "file",
    "image_return_format": "PNG"
}, sys.stdout)
' | curl --fail-with-body -sS --max-time 1200 \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  http://10.12.11.2:8000/v1/images/generations \
  -o flux3_two_references.png
```

Base 示例改成 50 步，并可增加 `guidance_scale:3.5`。减少到一张时仍使用数组；也可沿用旧 `reference_image`。不要在同一请求中混用这两个字段。

异步时将提交地址改为 `/v1/images/generations/async`、`response_format` 改为 `b64_json`，响应保存为 JSON。提交返回 202 和 `id`；随后：

```bash
curl --fail-with-body -sS \
  http://10.12.11.2:8000/v1/images/generations/JOB_ID
```

直至 `status=completed`，读取 `images[0]` 的 Base64 并解码保存；轮询响应不是直接 PNG。如果部署启用 API key，请为提交和轮询增加对应的 Authorization header。

## 13. 验证脚本、复现命令及恢复检查

下列三个脚本为本轮验证工具，保存在 cache，不是已提交的生产代码：

```text
/local/lnn/fluxcache/test_flux3_ttnn_vae_encoder.py
/local/lnn/fluxcache/flux3_multiref_smoke_v3.py
/local/lnn/fluxcache/flux3_multiref_validation_control.py
```

单元测试可在原服务运行时执行，不占用设备：

```bash
sudo docker exec --user 1000:1000 -w /app -e PYTHONPATH=/app \
  flux3-media-server-test \
  /home/container_app_user/tt-metal/python_env/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_flux3_reference_api.py

sudo docker exec --user 1000:1000 \
  -w /home/container_app_user/tt-metal \
  -e PYTHONPATH=/home/container_app_user/tt-metal \
  flux3-media-server-test \
  /home/container_app_user/tt-metal/python_env/bin/python \
  -m pytest -o addopts=-q -p no:cacheprovider \
  --confcutdir=models/tt_dit/tests/flux3 \
  models/tt_dit/tests/flux3/test_flux3_reference_preprocessing.py \
  models/tt_dit/tests/flux3/test_reference_conditioning.py
```

控制脚本会停止原服务、依次启动数值/Base/Distilled 单芯片验证容器，最后在 `finally` 恢复原服务；复跑前必须确认允许暂停服务，且没有正在执行的请求，不能仅凭队列为 0 就认定设备闲置。命令是 `sudo python3 /local/lnn/fluxcache/flux3_multiref_validation_control.py`。

最终控制运行 exit 0，日志记录两模型 `HTTP_VALIDATION_OK` 和 `ORIGINAL_SERVICE_RESTARTED`。原服务恢复后检查：

```text
flux3-media-server-test：Distilled，端口 8000
worker：32/32 alive、ready
queue_size：0
全部 worker error_count：0
canary_state：healthy
TT_MM_THROTTLE_PERF：0
TT_METAL_TDP_LIMIT_WATTS：160
reference encoder 默认 backend：cpu
```

保留的验证容器为 `flux3-reference-numeric-validation-uid1000-20261105`、`flux3-base-multiref-ttnn-20261105`、`flux3-distilled-multiref-ttnn-20261105`，目前停止；未删除原服务和持久化缓存。

## 14. 尚未实现或未验证的内容

- 10 张 1024×1024 参考图容量和 32-worker 多参考图并发已按本轮固定负载验证完成；峰值设备 DRAM/L1、任意输入组合容量和长时间耐久性尚未测量。
- Base/Distilled 的 200 请求性能已记录；相同输入下 CPU 与 TTNN reference encoder 的热态性能对比尚未完成。
- 相同输入、相同热态测量口径的 CPU 与 TTNN encoder 性能比较。
- 自动显存不足回退；现在可以显式设置 `cpu`，不支持自动捕获 OOM 后无缝切换。
- 全模型逐层精度与参考编辑语义验收；本轮 encoder 数值检查不代表这两项已完成。
- prompt upsampler、跨 worker 首次缓存创建的锁和原子发布。
- 2/4 芯片单请求并行按用户当前范围暂缓。
