# FLUX.3 Klein：四卡布局性能对比

## 1. 对比范围

本文汇总四卡一个 worker 的四种布局：

- TP1×SP4（SP4，4×1 Ring）
- 纯 TP4（1×4）
- 纯 TP4（4×1）
- TP2×SP2（2×2 父 mesh/子 mesh）

统一请求为 Distilled、`a black bird`、1024×1024、16 steps、seed=42、无参考图、PNG file response。测试均为同步请求，前一个请求完成后才发送下一个请求；测试条件为 `TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`。

## 2. 最新实验数据

下表优先采用 Galaxy03 上最近一次热态日志中的设备计时。HTTP 是服务端同步请求总耗时；denoise 和 VAE decode 来自 FLUX.3 pipeline stage timing。

| 布局 | Mesh / 语义 | HTTP E2E | Denoise | VAE decode | 样本数 | 状态 |
|---|---|---:|---:|---:|---:|---|
| SP4 | `4×1`，TP1×SP4，Ring | **2.657 s** | **1.891 s** | **0.285 s** | 2 | 成功 |
| SP4（父子 mesh） | 父 mesh `4×8` → 子 mesh `4×1`，TP1×SP4，Ring | **3.664 s** | **2.855 s** | **0.341 s** | 3 | 成功 |
| 纯 TP4 | `1×4`，TP4×SP1，Linear/TP 路径 | **3.796 s** | **2.887 s** | **0.490 s** | 1 | 成功 |
| 纯 TP4 | `4×1`，TP4×SP1，显式 `tp4` profile | **3.803 s** | **2.895 s** | **0.491 s** | 3 | 成功 |
| TP2×SP2 | `2×2`，TP axis 0、SP axis 1，Linear，父 mesh 8×4 切子 mesh 2×2 | **4.642 s** | **3.517 s** | **0.657 s** | 3 | 成功 |

### 计算依据

- SP4：HTTP `2.6641 / 2.6503 s`；denoise `1.8903 / 1.8918 s`；VAE `0.2851 / 0.2850 s`。
- SP4（父子 mesh）：HTTP `3.6756 / 3.6659 / 3.6517 s`，平均 `3.6644 s`；denoise 平均 `2.8552 s`；VAE 平均 `0.3411 s`。
- 纯 TP4、1×4：HTTP `3.7964 s`；denoise `2.8866 s`；VAE `0.4904 s`。
- 纯 TP4、4×1：HTTP `3.8109 / 3.7905 / 3.8080 s`，平均 `3.8031 s`；denoise 平均 `2.8947 s`；VAE 平均 `0.4910 s`。
- TP2×SP2：HTTP `4.6498 / 4.6426 / 4.6338 s`，平均 `4.6421 s`；denoise 平均 `3.5169 s`；VAE 平均 `0.6572 s`。

## 3. 性能结论

以本轮实验 SP4 的 `2.657 s` 为基准：

| 布局 | 相对 SP4 的 HTTP 延迟 |
|---|---:|
| SP4 | 基准 |
| SP4（父子 mesh） | 约慢 **37.9%**（与本轮 direct SP4 实验值比较） |
| 纯 TP4、1×4 | 约慢 **42.9%** |
| 纯 TP4、4×1 | 约慢 **43.1%** |
| TP2×SP2 | 约慢 **74.7%** |

纯 TP4 的两个方向在当前实现下基本相同，4×1 比 1×4 只快/慢约 0.2%，没有形成明确优势。TP2×SP2 当前最慢，主要表现为 denoise 和 VAE 两段都明显高于 SP4。

## 4. 实现和通信方式

| 布局 | 主要通信/拓扑 | pipeline 特征 |
|---|---|---|
| SP4 | 4×1 Ring；image-token 做 SP | 使用针对 FLUX.3 Klein 优化的 resident pipeline |
| SP4（父子 mesh） | 父 `4×8` → 子 `4×1`；image-token 做 SP | 与 direct SP4 使用同一 resident pipeline，区别在设备打开方式 |
| TP4、1×4 | 1×4 TP mesh；hidden/head 维度做 TP | 纯 TP 路径，不能直接复用 SP4 的 Ring 语义 |
| TP4、4×1 | 4×1 TP mesh；显式 `FLUX3_EXECUTION_PROFILE=tp4` | 纯 TP4 的 canonical profile |
| TP2×SP2 | 父 mesh 8×4，再切 2×2 子 mesh；TP axis 0、SP axis 1 | Linear 2D 子 mesh，当前为实验 profile `tp2sp2` |

注意：`(4,1)` 这个 mesh 形状本身不能区分 SP4 和纯 TP4。必须通过显式 profile 指定语义；不能让 `auto` 根据 mesh 形状猜测。

## 5. 数据口径说明

本轮最新日志来自实验容器 `flux3-tp8ring-debug-no-throttle`（SP4、纯 TP4）以及 `galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16`（TP2×SP2）。因此表中的数字适合判断布局趋势，不应当作为“同一镜像、同一代码提交”的严格 A/B benchmark。

此前正式 v1.0.16 SP4 基线为：HTTP **3.345 s**、denoise **2.829 s**、VAE decode **0.344 s**。该数字采用 8 worker 并发基准口径；本文件主表采用同步串行请求，不能直接混合平均。

## 6. 当前推荐

四卡生产路径优先使用 **direct SP4 / 4×1 Ring**。父子 mesh SP4 也已验证可用，但在本轮实验中比 direct SP4 慢；纯 TP4 两个方向都已能出图，但当前没有性能优势；2×2 已验证父子 mesh 创建和推理正确，但性能暂不适合作为四卡默认路径。

