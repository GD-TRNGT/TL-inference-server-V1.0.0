# FLUX3 与 SD3.5 推理架构对齐说明

## 目标

FLUX3 继续使用现有的性能路径，但采用 SD3.5 已验证的工程分层：配置、设备拓扑、worker 生命周期和模型 runner 各自负责单一职责。

## 本次整理

- `config/flux3_profiles.py`
  - 集中维护所有 FLUX3 profile 的 worker mesh、parent mesh、TP/SP 轴、Fabric、Ring/Linear、CCL link 数和 pipeline 类型。
  - 统一 profile alias 和 `auto` 解析逻辑。
  - 对 `(4,1)` 的 SP4/TP4 歧义继续强制显式选择，不静默猜测。
- `config/settings.py`
  - 只从 profile 表读取 canonical mesh、TP size 和并行模式，删除重复 profile 字典。
- `tt_model_runners/dit_runners.py`
  - runner 初始化时解析一次不可变 profile，pipeline、并行配置和设备参数都复用同一份 profile。
  - 保留旧的 `_flux3_*` 辅助函数，兼容现有测试和外部调用；实际 runner 不再重复推导 profile。
- `tt_model_runners/base_metal_device_runner.py`
  - parent mesh 校验复用统一 profile 表，避免基础设备层维护第二份 FLUX3 拓扑表。
- `utils/runner_utils.py`
  - 使用统一 alias 解析，再根据 canonical profile 选择 Galaxy descriptor。
- `model_services/image_service.py`
  - 清理重复 import 和重复 job manager 初始化，不改变 API 行为。

## 与 SD3.5 的一致点

```text
API → Service → Scheduler → DeviceWorker → ModelRunner → Pipeline
```

FLUX3 采用 SD3.5 中值得复用的工程规则：

1. API 层不直接创建 TTNN 设备。
2. 每个 worker 独立持有 mesh、模型和编译状态。
3. warmup 完成后才报告 ready。
4. parent mesh/child mesh 由物理拓扑决定，不允许任意拼接 PCIe 编号。
5. profile 是配置对象，而不是散落在多个函数中的字符串判断。
6. 模型计算路径仍保留在 FLUX3 pipeline 内，不复制 SD3.5 的模型逻辑。

## 当前 profile 约束

| Profile | Worker mesh | TP/SP | Pipeline |
|---|---:|---:|---|
| `legacy-single` | `1x1` | TP1/SP1 | legacy FLUX3 |
| `sp2` | `2x1` | TP1/SP2 | Klein |
| `tp2` | `2x1` | TP2/SP1 | Klein |
| `sp4` | `4x1` | TP1/SP4 | Klein |
| `tp4` | `4x1` | TP4/SP1 | legacy FLUX3 |
| `tp2sp2` | `2x2` | TP2/SP2 | Klein |
| `tp2sp4` | `2x4` | TP2/SP4 | Klein |
| `tp4sp2` | `4x2` | TP4/SP2 | Klein |
| `sp8` | `8x1` | TP1/SP8 | Klein |
| `tp8ring` | `1x8` | TP8/SP1 | legacy FLUX3 |
| `sp8tp4` | `4x8` | TP4/SP8 | Klein |

`sp4` 与 `tp4` 共用 `4x1` worker mesh，但语义不同，因此 `auto` 模式不能替用户选择。

## 验证边界

本次整理不改变已验证的 Direct Ring、TP8 Ring、SP2、SP4、SP8、TP2SP4 和 TP4SP2 计算路径；后续镜像测试仍需逐 profile 验证：

- 配置解析；
- worker 启动和 warmup；
- 1024×1024、16 steps、seed 42、PNG 同步请求；
- denoise、VAE decode 和 HTTP E2E 时间；
- 返回 PNG 有效性。

所有性能测试固定：

```text
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```
