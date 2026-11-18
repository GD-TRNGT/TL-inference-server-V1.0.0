# 新模型接入 TT 推理引擎的通用适配流程

更新时间：2026-11-18

本文总结将一个新的图像生成模型接入 TT-Metal / TL-inference-server 时应遵循的完整步骤。SD3.5 和 FLUX3 的实现可以作为参考，但不能直接复制模型结构、权重映射或并行参数。

## 1. 先确定模型契约

在写 TTNN 代码之前，先固定官方模型契约：

- 模型变体：Base、Distilled、FP8、NVFP4 等；
- checkpoint 文件、配置文件和 tokenizer；
- 输入 prompt、negative prompt、参考图或视频条件；
- latent channels、VAE scale factor、patch size；
- hidden size、head 数、block 数、层归一化和激活函数；
- scheduler、timestep、CFG 和 Euler/flow update 规则；
- RoPE/position ID 的生成规则；
- 输出宽高范围、对齐要求和最大像素数；
- batch、参考图数量和异步请求语义。

先形成一份模型契约，避免 API、pipeline、Transformer 和 VAE 对 shape 规则理解不一致。

## 2. 分析参考实现

优先分析官方实现或已有 TT 模型的以下内容：

```text
模型入口
  → prompt/text encoder
  → latent 初始化
  → patchify
  → Transformer forward
  → scheduler/update
  → VAE decode
  → image output
```

同时记录：

- source tensor 名称和 shape；
- 每一层的输入输出 dtype/layout；
- 哪些张量在 host，哪些张量在 device；
- CFG 是 batch 并行、submesh 并行还是顺序执行；
- VAE 是 channel parallel 还是 spatial parallel；
- 是否有参考图 encoder 或额外条件 token。

## 3. 划分两个仓库的职责

### 3.1 `tt-metal`

负责真正的模型计算：

- TTNN Transformer；
- Text Encoder；
- VAE encoder/decoder；
- 权重转换和 layout；
- attention、matmul、conv、norm 等算子；
- CCL 和设备通信；
- kernel/program/trace cache。

### 3.2 `TL-inference-server`

负责服务和设备生命周期：

- OpenAI-compatible API；
- request validation；
- worker 和队列；
- runner factory；
- device discovery；
- direct/parent-child Mesh；
- warmup、health、重启和关闭；
- Dockerfile 和部署配置。

不要把模型计算逻辑全部塞进 HTTP handler，也不要让 API 层直接操作 TTNN Mesh。

## 4. 先实现单卡最小闭环

第一个目标不是性能，而是单卡得到正确图片：

```text
checkpoint
  → Text Encoder
  → latent/noise
  → Transformer 单步
  → scheduler update
  → VAE decode
  → host image
```

单卡阶段至少完成：

1. checkpoint 能加载；
2. 一步 Transformer forward 能运行；
3. 多步 denoise 能运行；
4. VAE 能输出正确尺寸；
5. JPEG/PNG 能被真实解码；
6. 固定 seed 时输出稳定；
7. 日志能区分 encode、denoise、decode 和 HTTP 时间。

## 5. 实现 Text Encoder 和条件输入

明确以下边界：

```text
prompt
  → tokenizer/chat template
  → padding/mask
  → text encoder
  → hidden states / pooled embedding
  → Transformer condition
```

需要验证：

- token 长度和 padding multiple；
- hidden size 和 dtype；
- positive/negative prompt；
- CFG 的拼接顺序；
- 多参考图或其他条件 token 的顺序；
- text embedding 是否复制到 SP 分片。

## 6. 实现 Transformer 和权重映射

推荐顺序：

```text
先实现模块结构
  → 再实现 source state_dict 映射
  → 再实现 TTNN layout
  → 最后实现多卡分片和 CCL
```

权重映射必须明确处理：

- QKV 拆分或合并；
- column/row parallel linear；
- SwiGLU 或其他 gated FFN；
- transpose/reshape/permute；
- bias 是否复制或分片；
- checkpoint dtype 与 TTNN dtype；
- TP/SP 下权重和激活分别在哪个轴切分。

每个模块都应有 source shape、TTNN shape 和 mesh layout 记录，不能只依赖“能否加载成功”判断映射正确。

## 7. 实现 scheduler、latent 和多步循环

统一执行链路：

```text
seed
  → 初始 latent
  → patchify / image IDs
  → timestep embedding
  → Transformer velocity/noise prediction
  → CFG/condition combine
  → scheduler update
  → 下一步 latent
```

验证重点：

- timestep 数量和顺序；
- scheduler 参数；
- Euler/flow update 公式；
- CFG scale；
- latent 是否每一步错误地回到 host；
- 固定 seed 的确定性；
- 16、20、50 steps 等不同步数。

## 8. 实现 VAE encoder/decoder

VAE 必须单独适配，不能因为 Transformer 能运行就认为模型完成。

### Decoder

```text
denoised latent
  → inverse scaling
  → unpatchify
  → VAE conv/resnet/up-block
  → uint8
  → host image
```

### Reference encoder

```text
reference image
  → decode/resize
  → VAE encoder
  → latent/token packing
  → condition token
```

需要分别验证：

- latent 与输出尺寸比例；
- H/W 或 channel 并行；
- GroupNorm、halo 和邻居交换；
- conv slicing、`act_block_h`、double buffer；
- 不同分辨率的 program/cache key；
- decode 后 gather 和 host readback。

## 9. 建立 profile 和设备拓扑

profile 应集中声明：

- worker/child Mesh；
- parent Mesh；
- TP/SP/CFG factor；
- mesh axis；
- Ring、Linear 或 2D Fabric；
- VAE 并行策略；
- CCL link 数和 descriptor。

设备路径分两类：

```text
Direct
  → worker 直接打开合法设备组

Parent/child
  → 打开完整父 Mesh
  → create_submeshes(child_shape)
  → 模型绑定到 child Mesh
```

2D torus 或 Galaxy Ring 不代表一定必须使用 parent/child。只要 direct 使用真实匹配的 Fabric descriptor 和合法物理设备组，就可以 direct；parent/child 的主要价值是拓扑合法性和子网格管理。

## 10. 接入 runner 和 worker

采用分层结构：

```text
BaseDeviceRunner
  → BaseMetalDeviceRunner
  → 通用 DiT runner
  → 新模型 runner
  → 新模型 pipeline
```

新 runner 负责：

1. 选择模型 pipeline；
2. 返回 core grid、L1、trace 等设备参数；
3. 初始化 direct 或 parent/child Mesh；
4. 创建 Text Encoder、Transformer、VAE；
5. 执行 warmup；
6. 执行同步/异步请求；
7. 输出阶段计时；
8. 关闭 parent owner 和 child view。

## 11. 设计缓存和 resident 生命周期

缓存至少分为：

```text
权重转换 cache
kernel cache
program cache
trace cache
geometry/layout cache
tokenizer/HF cache
```

resident 模式：

```text
worker 启动一次
  → 加载模型一次
  → warmup 一次
  → 多个请求复用模型和 cache
```

cache key 必须包含模型变体、mesh、TP/SP、dtype、分辨率和 VAE geometry。不能把单卡权重 cache 直接用于 TP4 或 SP8。

## 12. 实现 API 和输出协议

API 层需要固定：

- 唯一 API key 参数；
- model 名称；
- prompt 和参考图字段；
- width/height/steps/seed；
- response format；
- JPEG/PNG；
- batch 和异步 job 语义；
- 401、422、404、500 的边界。

输出必须真实校验：

```text
HTTP status
  → MIME type
  → 图片可解码
  → 实际 width/height
  → 非空像素
```

## 13. 日志和性能测试

阶段日志至少包括：

```text
request accepted
text encode
reference encode
step i / N
denoise total
VAE decode
to_host
image postprocess
HTTP total
```

性能测试固定：

```bash
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

并且固定：

- 同步或异步方式；
- 分辨率；
- steps；
- seed；
- JPEG 或 PNG；
- warmup 状态；
- 是否包含排队。

`HTTP E2E` 不等于 `denoise + VAE`，图片编码、host readback、队列和调度必须单独记录。

## 14. 推荐验证顺序

```text
1. Python 单元测试 / py_compile
2. 单卡、单请求、1024 方图
3. 单卡多步和固定 seed
4. Text Encoder、Transformer、VAE 分组件测试
5. Direct 双卡/四卡/八卡
6. Parent/child 双卡/四卡/八卡
7. 2048、4096 正方形
8. 官方允许的非正方形尺寸
9. 无参考图、1 张和最多 10 张参考图
10. JPEG 与 PNG
11. 多 worker ready 和并发请求
12. OOM、timeout、Fabric placement、设备重启回归
13. 性能汇总和日志归档
```

首次 kernel 编译、warmup、热态单请求和并发吞吐必须分开记录。

## 15. 常见错误与处理原则

| 问题 | 常见原因 | 处理原则 |
|---|---|---|
| Fabric placement 失败 | 手工卡号或 descriptor 不匹配 | 使用真实拓扑和合法 mesh plan |
| warmup 超时 | 首次 kernel/权重 cache 编译 | 先看进程和 Metal 日志，不立即判死 |
| VAE L1 OOM | spatial layout、block 或 buffer 过大 | 调整 VAE geometry，不影响已验证 DiT 路径 |
| `act_block_h` 回退 | 指定值不满足该卷积 geometry | 接受合法 closest value 或按 geometry 选择 |
| HTTP 比 denoise 长很多 | host readback、JPEG、排队或 postprocess | 分阶段计时，不只看 HTTP |
| 图片可解码但结果不对 | 权重顺序、RoPE、scheduler 或 latent layout 错误 | 做固定 seed、数值和语义回归 |
| 多卡比单卡提升小 | 通信、VAE、worker 调度或错误轴 | 分别测算子、CCL、denoise 和 VAE |

## 16. 镜像固化顺序

只有代码和真机验证完成后才构建镜像：

```text
代码审查
  → 单元测试
  → direct/parent-child 真机回归
  → 性能记录
  → commit
  → tag
  → 更新使用手册
  → 从 TL-inference-server + tt-metal 构建镜像
  → 1/2/4/8/32 卡完整回归
  → 推送和部署
```

镜像构建时固定源码版本，运行时只挂载模型、cache 和必要设备资源。临时容器内的临时 Python 文件不能作为正式实现来源。

## 17. 最终验收标准

一个新模型只有同时满足以下条件，才应称为“适配完成”：

- checkpoint 和权重映射正确；
- 单卡完整 T2I 成功；
- 多步 scheduler/CFG 正确；
- VAE 输出尺寸和格式正确；
- direct 多卡路径成功；
- parent/child 路径在需要的 profile 上成功；
- API 错误边界正确；
- 参考图/批量/异步功能符合模型契约；
- resident/cache/warmup 可重复；
- 性能数据使用同一测试口径；
- 日志能区分每个阶段；
- 代码已 commit，镜像由该 commit 构建；
- 使用说明、性能记录和限制条件完整。

## 18. 参考实现

当前可参考的两套实现文档：

```text
docs/SD35_IMAGE_CODE_ARCHITECTURE_0.22.0.md
docs/FLUX3_MODEL_ADAPTATION_IMPLEMENTATION_LOGIC.md
```

FLUX3 的父 Mesh/子 Mesh 单独说明：

```text
docs/FLUX3_PARENT_CHILD_MESH_IMPLEMENTATION_SUMMARY.md
```
