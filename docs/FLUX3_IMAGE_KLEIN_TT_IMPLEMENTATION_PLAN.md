# FLUX.3 Image Klein 在 Blackhole Galaxy 上的后续实施计划

更新时间：2026-11-06
开发仓库：`/local/lnn/TL-inference-server`，分支：`flux-3-image-klein`
TT-Metal 仓库：`/local/lnn/tt-metal`，分支：`flux-3-image-klein`

完整的已实施过程、代码位置、问题现象及验证证据见
[FLUX3_IMAGE_KLEIN_TT_ADAPTATION_COMPLETE.md](FLUX3_IMAGE_KLEIN_TT_ADAPTATION_COMPLETE.md)。
多参考图的代码级说明见
[FLUX3_MULTI_REFERENCE_AND_TTNN_VAE_IMPLEMENTATION.md](FLUX3_MULTI_REFERENCE_AND_TTNN_VAE_IMPLEMENTATION.md)；
大图和 32-worker 压测的原始结果见
[FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md](FLUX3_LARGE_REFERENCE_AND_32_WORKER_BENCHMARK.md)。

## 1. 当前基线

当前实现不是 API 占位：Base 与 Distilled 两套权重均已在 10.12.11.2 的 Blackhole Galaxy 上完成真实单芯片推理、32 个单芯片 worker 启动和 HTTP 请求验证。

| 能力 | 当前状态 | 已验证范围 |
|---|---|---|
| T2I：prompt、宽高、steps、seed | 已实现 | Base 与 Distilled；`/v1/images/generations` |
| Base CFG、negative prompt | 已实现 | Base 50-step CFG |
| Distilled flow-map、`ref_time_in` | 已实现 | Distilled 16-step |
| PNG/JPEG 文件响应 | 已实现 | `response_format=file` |
| 异步、轮询、cancel | 已实现 | image job 生命周期和取消语义 |
| 组件常驻 | 已实现 | prompt encoder、Transformer、VAE 每 worker 常驻；可回退 `per_request` |
| 设备端 denoise | 已实现 | latent、CFG、Euler update 留在 TT 设备端 |
| 1～10 张参考图 | 已实现 | 有序 `reference_images`，兼容旧单图字段 |
| reference-image VAE encoder | 已实现 | TTNN 默认；CPU 作为显式回退 backend |
| 32-worker T2I 压测 | 已完成 | Base/Distilled 各 200/200 成功，worker 0～31 全参与 |
| 大图、多参考图压测 | 已完成 | 两模型各 14/14 矩阵、32 冷/暖态、200/200 |
| 2/4 芯片单请求模型并行 | 暂缓 | 当前范围不实施 |
| Golden Data 逐层精度对齐 | 暂缓 | 当前范围不实施 |

性能基线必须区分请求口径，不能混用：

| 模型 | 场景 | 结果 |
|---|---|---|
| Distilled | 纯 T2I，200 个同时提交 | 77.820 s，2.5700 images/s，200/200 成功 |
| Base | 纯 T2I，200 个同时提交 | 433.557 s，0.4613 images/s，200/200 成功 |
| Distilled | 两张 512×512 参考图 → 1024×1024，200 个同时提交 | 119.631 s，1.6718 images/s，200/200 成功 |
| Base | 两张 512×512 参考图 → 1024×1024，200 个同时提交 | 681.294 s，0.2936 images/s，200/200 成功 |

## 2. 本阶段范围和不在范围的工作

本阶段目标是把已验证的功能整理成**可提交、可重建、可复现、可部署**的正式交付，而不是继续扩展未经验证的新模型能力。

暂不实施：

- 2 芯片、4 芯片单请求模型并行、权重分片和通信优化；
- 与 PyTorch、vLLM-Omni 或官方实现的逐层 Golden Data 数值对齐；
- 参考视频、参考音频等 MiniMax-H3 Ref2VA 能力；它们不属于 FLUX.3 Image Klein 的图片模型范围。

## 3. P0：可复现交付

### 3.1 整理并提交两仓库改动

**目标**：将已验证的功能从未提交工作树变为可追溯源码。

| 工作 | 交付物 | 验收 |
|---|---|---|
| 检查 TL-inference-server 改动，按 runner/API/测试/文档拆分 | 清晰的 Git commits | `git diff --check` 无输出；每个 commit 可独立说明 |
| 检查 tt-metal 改动，按 pipeline/encoder/测试拆分 | 清晰的 Git commits | 单元测试与服务回归通过 |
| 保留用户已有的无关文件和 swap 文件 | 不误删、不覆盖用户内容 | `git status` 人工复核 |

当前状态：**已完成**。`tt-metal` 已提交 `6926f00a4a Add FLUX.3 TTNN multi-reference encoder`；`TL-inference-server` 已提交 `eca825c5 Add FLUX.3 multi-reference API and TTNN defaults` 及 `52eee2fc Document FLUX.3 validation and delivery status`。两仓库提交后工作树均为空。

### 3.2 使用当前源码重建正式镜像

**目标**：正式镜像包含最新 TL-inference-server 与本地 tt-metal 代码，而不是依赖 source bind mount。

| 工作 | 交付物 | 验收 |
|---|---|---|
| BuildKit 同时使用本地 TL-inference-server 和 tt-metal context | 带 Git SHA/tag 的正式镜像 | `docker image inspect` 可识别标签 |
| 运行时挂载 `/local/models` 和 `/local/lnn/fluxcache` | 可复用模型和转换/JIT cache | 容器可启动，不挂源码目录 |
| Base、Distilled 分别启动 | 两套启动命令 | 健康检查、单请求和文件响应通过 |

当前状态：**已完成**。使用本地两个已提交仓库构建正式镜像 `tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4`（镜像 ID `sha256:5a16aed14e2d…`，约 32.6 GB）。运行容器只挂载 `/local/models`、`/local/lnn/fluxcache` 和 hugepages；不挂载两个源码目录。

在 10.12.11.2 上从提交源码重建该镜像的完整命令如下。`tt-metal-local` 是 Docker BuildKit 的附加 build context，因此 Dockerfile 使用本地 `tt-metal`，而不是在构建时再 clone 上游仓库：

```bash
cd /local/lnn/TL-inference-server

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=6926f00a4a \
  --file tt-media-server/Dockerfile \
  --tag tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4 \
  .
```

构建完成后可确认镜像：

```bash
docker image inspect tl-inference-server:flux-3-image-klein-eca825c5-6926f00a4 \
  --format 'ID={{.Id}} CREATED={{.Created}} SIZE={{.Size}}'
```

### 3.3 正式镜像验收清单

正式镜像至少执行以下回归：

1. Base 与 Distilled 各完成一个 T2I PNG 文件响应。
2. Base 与 Distilled 各完成 1、2、10 张参考图请求。
3. 异步提交、轮询、cancel 各验证一次。
4. 32 worker 全部 ready；做一轮并发健康验证。
5. 对比当前性能基线，记录明显回归。

当前状态：**已完成**。Base 和 Distilled 分别以 32/32 worker ready 启动；两者均通过 T2I、旧 `reference_image`、`reference_images` 1/2/10 张、非法图片和 11 张图片 422、异步提交/带鉴权轮询完成、1024×1024/50-step cancel。Base 同步 256×256/12-step 热态 T2I 为 2.874 s，单图 3.784 s、双图 3.891 s、十图 4.182 s；Distilled 对应 T2I 1.841 s、单图 2.738 s、双图 2.820 s、十图 2.768 s。测试结果在 `/local/lnn/fluxcache/official_image_eca825c5_{distilled,base}_20261106/`。

## 4. P1：可靠性和服务工程

### 4.1 首次 cache 创建的并发安全

问题：32 个 worker 在首次同时转换同一模型缓存时，可能发生重复 I/O 或不完整缓存读取。

实施内容：

- 为转换 cache 创建跨进程锁；
- 写入临时目录，校验成功后原子 rename；
- 读取方只接受标记为完整的 cache；
- 保留预生成 cache 作为部署前措施。

验收：清空指定测试 cache 后同时启动 32 worker，所有 worker 成功、无损坏 cache、无重复失败。

状态：**未实现**。

### 4.2 reference encoder 自动 OOM 回退

当前行为：TTNN 是默认 backend；可显式使用 `FLUX3_REFERENCE_ENCODER_BACKEND=cpu` 回退。TTNN backend 已验证，但不会在资源不足时自动降级。

实施内容：

- 定义可回退的 TTNN 内存/分配异常范围；
- 释放该请求相关 TT tensor；
- 以 CPU encoder 重新执行该请求；
- 在任务状态和日志中记录降级原因；
- 不把模型逻辑错误、输入非法或结果校验失败误判为 OOM。

验收：注入或构造资源不足场景，任务仍成功且响应/日志标识实际 backend。

状态：**未实现**。

### 4.3 image job 持久化与下载

当前异步、查询与 cancel 已实现，但服务重启后历史 job 不恢复，图片文件也没有独立下载资源接口。

实施内容：

- 明确任务元数据和输出文件的持久化位置；
- 服务启动时恢复可恢复任务状态，不能恢复的任务必须标记明确失败原因；
- 增加受控的图片下载接口或稳定的文件 URL；
- 定义取消后文件保留与清理策略。

验收：提交异步任务、重启服务、查询任务和下载已完成图片。

状态：**未实现**。

### 4.4 稳定性与资源监控

实施内容：

- 多小时持续负载；
- 采集每芯片 DRAM/L1、主机内存、温度、功耗、worker restart/error；
- 明确失败阈值和恢复流程。

验收：指定时长内无 worker restart/error，资源曲线无持续泄漏。

状态：**未开始**。

## 5. P2：性能工程

### 5.1 建立分阶段 profiling

当前只有端到端 HTTP 延迟和吞吐，不能准确说明瓶颈在何处。

需要分开记录：

- prompt encoder；
- reference encoder（CPU 与 TTNN）；
- Transformer 单步和完整 denoise；
- VAE decode；
- TT 到 host 传输；
- PNG/JPEG 编码；
- 队列等待与 worker 调度。

交付物：统一 JSON/CSV 性能记录、P50/P95、每阶段耗时图表和优化前后对比。

状态：**未实现**。

### 5.2 算子与内存优化

在 profiling 确认瓶颈后依次研究：

- 大 token matmul blocking/layout；
- trace 覆盖和重放；
- kernel compile/cache 命中；
- TT tensor layout 与 host/device copy；
- 峰值内存、L1 和 DRAM 占用；
- CPU 与 TTNN reference encoder 的同输入热态对比。

状态：**未开始**。不得在没有 profile 证据时把端到端性能差异归因到某一个 kernel。

## 6. P3：质量与精度

### 6.1 Golden Data 对齐

当前完成的是“服务能生成可解码、非全平、语义合理的图”，不等于数值精度已正式对齐。

后续需要在可用 PyTorch/vLLM 环境中导出并比较：

- tokenizer/prompt embedding；
- 初始 latent；
- flow schedule；
- 指定 DiT block 输入输出；
- 指定 denoise step latent；
- VAE 输入与输出；
- 最终图片的统计和人工质量。

状态：**按当前范围暂缓**。

### 6.2 参考图语义验收

当前大图压测证明 1～10 张参考图、不同尺寸及 32-worker 服务路径可靠，但不是主观编辑质量评测。

后续需要定义固定提示词和参考图集，评价主体一致性、构图、颜色/风格、参考图顺序影响和多图冲突处理。

状态：**未开始**。

## 7. 代码边界和部署约束

| 层级 | 仓库 | 主要职责 |
|---|---|---|
| 服务 API、请求校验、队列、异步任务、runner | TL-inference-server / `tt-media-server` | 接收请求、调度单芯片 worker、返回 PNG/JPEG 或任务状态 |
| 模型 pipeline、TTNN Transformer、VAE/reference encoder | tt-metal | 权重读取、TT tensor、算子执行和图像生成 |
| 模型权重 | `/local/models/flux-3-image-klein*` | 运行时只读挂载，不复制进镜像 |
| 编译与转换缓存 | `/local/lnn/fluxcache` | JIT kernel、转换权重、HF/Torch 和临时缓存 |

正式运行要求：

- 模型通过 `/local/models:/local/models:ro` 挂载；
- 缓存通过 `/local/lnn/fluxcache:/local/fluxcache` 挂载；
- 服务默认使用 Distilled；在 `/local/models` 挂载存在时，runner 自动将默认名称解析为 `/local/models/flux-3-image-klein`；`FLUX3_MODEL_ROOT` 或 `MODEL_WEIGHTS_DIR` 可在启动时显式选择 Base 或 Distilled；
- 默认使用 TTNN reference encoder；资源受限或故障排查时可显式设定 `FLUX3_REFERENCE_ENCODER_BACKEND=cpu`；
- 性能测试如需关闭节流，显式设定 `TT_MM_THROTTLE_PERF=0` 与 `TT_METAL_TDP_LIMIT_WATTS=160`。

## 8. 完成标准

在不实施 2/4 芯片并行和 Golden Data 对齐的当前范围内，以下项目完成才视为 FLUX.3 Image Klein 的工程交付完成：

- 两仓库改动已审查、测试并 commit；
- 当前源码已重建为正式镜像，运行不依赖源码 bind mount；
- Base/Distilled、T2I、多参考图、文件响应、异步/cancel 回归通过；
- cache 首次并发创建安全；
- 自动 OOM 降级、job 恢复/下载接口有明确实现和测试；
- 有可复现的性能 profile 与长期稳定性报告；
- 文档、镜像、代码提交 SHA 和测试产物能互相追溯。


## 9. 2026-11-07：低风险 RoPE 调度优化（已实施）

- [已完成] 将 joint RoPE 从纯 T2I 的 28 个 single block 内移出；同一 Transformer forward 只使用一份 joint RoPE。
- [已完成] 再将 joint RoPE 移到 pipeline 的 denoise loop 外：prompt 条件、负提示词 CFG 条件、以及参考图条件各按其静态 token 长度创建一次，并在整次请求复用。
- [已完成] 对调用方提供的 joint RoPE 检查其 sequence length，避免双流/三流 RoPE 被错误复用。
- [暂缓] text/image/reference-image 三路 mode block 并发。原因是三路仍共享同一完整 Galaxy mesh 和分布式 collective；未验证独立 command queue、通信安全性和端到端收益前，不改变默认串行调度。
- [后续] 通过 profiler 记录 mode block、single block、RoPE 构造和同步等待的耗时比例；若 mode block 占比有意义，再评估 command queue 或 mesh 切分设计。