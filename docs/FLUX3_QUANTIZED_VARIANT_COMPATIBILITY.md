# FLUX.3 Image Klein 量化权重变体兼容性评估

评估日期：2026-11-13

## 结论

当前已交付的 TTNN FLUX.3 服务只可直接加载 BF16 `model.safetensors`。`fp8r` 和 `nvfp4` 两个目录都不是可直接传给现有 TTNN loader 的格式。

| 变体 | 文件大小 | 权重存储 | 当前直接加载 | 可做的兼容方案 | 原生 TTNN 运行 |
| --- | ---: | --- | --- | --- | --- |
| BF16 | 11.10 GB | BF16 | 已支持 | 不需要 | 已支持 |
| FP8R | 6.27 GB | `torch.float8_e4m3fn` + 每输出通道 FP32 scale | 不支持 | host 端反量化到 BF16 后复用现有 loader | 未实现 |
| NVFP4 | 5.28 GB | packed `uint8` E2M1 + 分组 E4M3 scale + FP32 scale | 不支持 | host 端解包、反量化到 BF16 后复用现有 loader | 未实现 |

“可做兼容方案”只能节省模型文件的磁盘空间；首次载入后会重新形成 BF16 TTNN cache，不能降低设备权重内存，也不能保证更快。

## 1. 已核对的模型文件

模型根目录：`/local/models/flux-3-image-klein`。

- BF16：`model.safetensors`，11,097,404,512 bytes。
- FP8R：`variants/fp8r/model.safetensors`，6,270,949,144 bytes。
- NVFP4：`variants/nvfp4/model.safetensors`，5,277,674,144 bytes。

两个变体目录只有量化 DiT 权重及其配置；Qwen3-VL text encoder 和 VAE 仍应从根模型目录加载。因此不能简单将 `FLUX3_MODEL_ROOT` 指向 `variants/fp8r` 或 `variants/nvfp4`。

## 2. FP8R 格式与可行路径

FP8R manifest 声明 234 个量化矩阵。矩阵如
`single_blocks.0.q_proj.weight` 的 dtype 是 `torch.float8_e4m3fn`，并配有同名
`weight_scale`，形状为 `[N]`。其恢复规则是：

```python
bf16_weight = (weight.float() * weight_scale[:, None]).to(torch.bfloat16)
```

因此可增加一个明确的 `FLUX3_DIT_VARIANT=fp8r` 路径：读取 FP8R 文件、在 checkpoint 边界反量化为 BF16、使用现有 `_block_state()` 与 `cache.load_model()`。缓存 key 必须包含 `fp8r-dequant-bf16`，避免和原 BF16 checkpoint 混用。

此方案适合作为“磁盘分发格式兼容”，但不是设备端 FP8 优化：反量化后的 host 张量、TTNN DRAM 权重和已转换缓存仍是 BF16。

## 3. NVFP4 格式与可行路径

NVFP4 有 198 个量化矩阵。量化矩阵是 `[N, K/2]` 的 `uint8`，每个 byte 保存两个 E2M1 值；另有：

- `[N, K/16]` 的 `float8_e4m3fn` `weight_scale`；
- 标量 `weight_scale_2`；
- 标量 `input_scale`。

兼容加载必须先按 low/high nibble 解包 E2M1，再按每 16 个 K 元素应用 `weight_scale` 和 `weight_scale_2`。这比 FP8R 多一个正确性风险：E2M1 解码、nibble 顺序和分组广播只要有一个错误，生成图像就会明显失真。

即使 host 端反量化完成，`input_scale` 也不能自动产生真正的 NVFP4 activation path；它只适用于原始量化推理实现。因此 NVFP4 的 BF16 回退兼容价值比 FP8R 更低，必须先完成权重重构与 BF16 基准的逐矩阵误差检查、固定 prompt/seed 出图对比后才可启用。

## 4. 为什么当前不能原生执行

当前 TTNN Python API 没有 `ttnn.float8_e4m3fn` dtype；它提供的是不同编码格式的 `ttnn.bfloat8_b`。同时当前 matmul gate 只接受 `ttnn.bfloat16`、`ttnn.float32` 和 `ttnn.bfloat8_b`。因此：

- 不能把 BFL 的 E4M3FN 字节直接标记为 `bfloat8_b`；两者数值编码不同。
- TTNN 当前没有将 packed NVFP4/E2M1 权重连同 per-group scale 作为 matmul 权重的加载与计算路径。
- 原生支持需要新增 dtype 转换/权重布局、线性层和 matmul kernel 路径、cache 格式以及完整数值与性能验证；这不是仅修改 FLUX.3 loader 能完成的工作。

## 5. 推荐实施顺序

1. 保持 BF16 为默认和生产唯一支持格式。
2. 若目标是减少模型分发/磁盘占用，先实现 FP8R 的 host 端反量化兼容加载，并验证同 seed 的图像质量和首次/热态加载时间。
3. 只有 FP8R 质量与运行稳定性通过后，再实现 NVFP4 的离线解包工具与逐矩阵误差验证。
4. 若目标是减少 TT 设备内存或提升推理速度，则应单独开展 TTNN 原生 FP8/NVFP4 项目；不要把 BF16 回退兼容误称为 FP8/NVFP4 加速。
