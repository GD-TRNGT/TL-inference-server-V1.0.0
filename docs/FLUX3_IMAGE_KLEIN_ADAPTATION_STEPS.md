# FLUX.3 Image Klein 在 TT 推理引擎中的适配步骤

本文记录把新的 FLUX.3 Image Klein 模型接入 TT 推理引擎时的完整适配顺序，便于后续更新权重、TT-Metal 或增加并行 profile 时保持结构一致。

## 1. 固定模型契约

先从 checkpoint 和官方 pipeline 固定以下内容：

- 模型名称、base/distilled 变体和权重目录；
- latent channels、VAE scale factor、DiT patch size；
- hidden size、head 数、depth、single depth、text channels；
- flow-matching scheduler、步数和 guidance 规则；
- RoPE 的 image/text ID 生成规则；
- 参考图数量、尺寸、预处理和 image-condition token 格式；
- 官方输出宽高范围、16 像素 API 网格以及内部并行所需的额外对齐条件。

模型契约必须先独立成文，再进入 TT 实现，避免 API、DiT 和 VAE 各自使用不同的 shape 规则。

## 2. 接入 API 和请求校验

在请求模型中完成：

1. `model`、`prompt`、`width`、`height`、`seed` 和 `num_inference_steps` 解析；
2. API key 统一使用一个入口，例如 `OPENAI_API_KEY`；
3. width/height 同时出现且为整数；
4. 按官方模型约束检查最小值、最大值和 16 像素倍数；
5. 支持 `response_format`、JPEG/PNG 和 batch 参数；
6. 参考图数量限制为 1～10 张，并区分单参考图和多参考图预处理。

API 校验只验证公共契约。VAE 的空间分片对齐属于内部执行约束，应由 pipeline 自动补齐并在最终输出阶段裁剪，不能把合法的官方比例错误地拒绝为 422。

## 3. 建立 profile 和设备拓扑

在 `config/flux3_profiles.py` 建立唯一 profile 表，至少同时声明：

- worker mesh；
- parent mesh；
- TP/SP factor 和 mesh axis；
- Ring/Linear/2D Fabric；
- CCL link 数；
- 是否使用 resident Klein pipeline。

随后由 settings 和 DeviceManager 处理：

```text
profile → 合法 worker 组 → direct 或 parent/child Mesh → pipeline device
```

设备分组必须来自 TT-Metal 的真实拓扑发现或 mesh descriptor，禁止从 API 参数直接传入任意卡号。

## 4. 接入模型 runner 和生命周期

runner 负责：

1. 打开 MeshDevice；
2. 在 parent 模式下打开完整父 Mesh 并选择子 Mesh；
3. 创建 CCLManager、文本编码器、DiT、VAE 和缓存；
4. 处理 warmup、health、错误重启和关闭；
5. 关闭时只关闭父 Mesh owner，不能错误关闭子 Mesh view。

worker 启动日志必须打印 profile、mesh shape、Fabric、core grid、TDP 限制、缓存位置和 ready 状态。

## 5. 实现文本条件路径

接入 tokenizer 和 Qwen3-VL text encoder，明确：

- token 长度和 text bucket；
- padding/mask；
- hidden state 的 dtype/layout；
- text embedding 是否复制到各 SP 分片；
- 多参考图 image-condition token 如何与 text token 拼接；
- prompt upsampler 是可选阶段，不应改变基础生成路径的 shape 契约。

## 6. 实现 DiT 去噪路径

按照参考 pipeline 的顺序接入：

```text
seed → latent noise
    → patchify / image_ids
    → text/image-condition embedding
    → timestep embedding
    → transformer blocks
    → CFG（如 base 模型需要）
    → flow Euler update
    → 重复 N 步
```

每一步应能够记录 step time；固定 `seed=42`、步数和输出格式后，才能比较不同 profile 的性能。

## 7. 处理空间尺寸和并行对齐

公共 API 允许官方 16 像素网格的宽高，但内部 VAE 的 H/W 分片还需要满足：

```text
latent spatial size / patch size 能被对应 mesh 轴的分片数整除
```

正确做法是：

1. 根据 VAE scale、patch size、H/W 分片因子计算内部合法尺寸；
2. 只对内部 latent/noise/image_ids/DiT token 几何做向上补齐；
3. DiT 和 VAE 使用补齐后的 geometry；
4. decode 回 host 后裁剪到请求的 width×height；
5. 返回图片的实际尺寸必须仍等于 API 请求尺寸。

这一步覆盖 SP4/SP8 和 21:9、9:21 等非正方形比例，同时不改变已经合法的 1024、2048、4096 正方形路径。

## 8. 实现 VAE decoder

VAE 适配需要固定：

- patch/unpatchify 的 token 顺序；
- latent BatchNorm 的 running mean/variance；
- TP/SP 下的空间分片轴；
- GroupNorm 的局部与跨分片路径；
- halo、conv slicing、`act_block_h` 和 double buffer；
- decode 后的 gather、untilize、host readback 和裁剪。

每个 VAE geometry 都要有独立的 conv program/cache key，不能把一种分辨率的配置错误复用于另一种分辨率。

## 9. 权重、缓存和 warmup

权重加载必须经过：

```text
checkpoint 校验
    → 权重名称/shape/dtype 映射
    → profile/mesh 相关分片
    → /local/ttcache/flux-3-image-klein 缓存
    → cache hit 后快速启动
```

warmup 使用代表性的 1024、2048、4096 正方形；非正方形请求不应通过把公共尺寸改成 bucket 来伪装支持，而应走运行时 geometry 对齐。

## 10. API、图像输出和阶段日志

一次请求至少应区分：

- prompt upsample；
- text encode；
- reference encode；
- denoise 总耗时；
- 每一步 denoise 耗时；
- VAE decode；
- device-to-host/readback；
- JPEG/PNG 编码和 HTTP 总耗时。

阶段时间应在服务日志中输出，并保存请求参数、profile、mesh、输出尺寸和结果格式，避免把 JPEG 编码或排队时间误算成 denoise。

## 11. 验证顺序

建议按以下顺序执行同步测试：

1. 单卡 `legacy-single`，1024×1024；
2. 双卡 `sp2`、`tp2`；
3. 四卡 `sp4`、`tp4`、`tp4axis1`、`tp2sp2`；
4. 八卡 `sp8`、`tp8ring`、`tp2sp4`、`tp4sp2`；
5. 全 Galaxy `sp8tp4`；
6. 每条路径分别验证 direct 和 parent/child（如果 profile 支持）；
7. 1024×1024、2048×2048、4096×4096 以及官方非正方形比例；
8. 无参考图、1 张参考图、最多 10 张参考图；
9. JPEG 与 PNG；
10. 输出文件解码并检查实际像素尺寸。

所有性能测试固定：

```bash
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

请求必须同步发送，热态和首次 kernel 编译结果分开记录。

## 12. 固化顺序

完成验证后再按以下顺序固化：

```text
代码审查与回归
    → commit
    → tag
    → 使用说明和性能 Markdown
    → Dockerfile 只引用 TL-inference-server + tt-metal
    → 从头构建镜像
    → 1/2/4/8/32 卡真机回归
    → 推送并部署
```

镜像构建前不要用临时容器内 Python 文件覆盖正式代码；所有临时补丁都必须回写到仓库并通过测试后再制作镜像。
