# FLUX.3 Image Klein：Worker、自动拓扑与父子 Mesh 使用说明

本文对应 Galaxy03 上的 FLUX.3 Image Klein 代码版本，说明单 worker、自动多 worker、GPU 自动分组以及父 mesh/子 mesh 的实际实现方式。

## 1. 统一环境变量

生产环境建议显式设置以下变量：

```bash
MODEL=flux-3-image-klein
MODEL_RUNNER=tt-flux.3-image-klein
MODEL_SERVICE=image
DEVICE=bh-galaxy
OPENAI_API_KEY=turing123
FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein
FLUX3_MODEL_VARIANT=distilled
FLUX3_CORE_GRID_OVERRIDE=12x10
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
```

`FLUX3_EXECUTION_PROFILE` 是唯一推荐的并行配置入口。不要再通过 `FLUX3_DEVICE_IDS` 手工填写物理卡号；服务会根据 Galaxy 拓扑自动发现合法分组。

## 2. 自动分组模式

自动分组模式不设置 `FLUX3_PARENT_MESH_MODE=parent`，默认使用 direct 路径。服务启动时会发现整台 Galaxy，并按照 profile 自动生成 worker 列表。

| 场景 | Profile | 子 mesh | 每个 worker 卡数 | 一台 Galaxy 的默认 worker 数 | 自动发现方式 |
|---|---|---:|---:|---:|---|
| 单卡 | `legacy-single` | `1x1` | 1 | 32 | `tt-smi` 平铺单卡 |
| 双卡 SP | `sp2` | `2x1` | 2 | 16 | `test_system_health` 发现物理双卡链路 |
| 双卡 TP | `tp2` | `2x1` | 2 | 16 | `test_system_health` 发现物理双卡链路 |
| 四卡 SP | `sp4` | `4x1` | 4 | 8 | `test_system_health` 发现物理四卡组；Ring |
| 四卡 TP | `tp4` | `4x1` | 4 | 8 | `test_system_health` 发现物理四卡组 |
| 四卡 TP 轴转置 | `tp4axis1` | `1x4` | 4 | 8 | `test_system_health` 发现物理四卡组 |
| 八卡 SP | `sp8` | `8x1` | 8 | 4 | `tt-smi` 发现跨 tray 的 SP8 Ring 分组 |
| 八卡纯 TP Ring | `tp8ring` | `1x8` | 8 | 4 | `tt-smi` 发现 TP8 Ring 分组 |
| 八卡 TP2×SP4 | `tp2sp4` | `2x4` | 8 | 4 | `tt-smi` 按 tray 生成 8 卡组 |
| 八卡 TP4×SP2 | `tp4sp2` | `4x2` | 8 | 4 | `tt-smi` 按 tray 生成 8 卡组 |
| 全 Galaxy | `sp8tp4` | `4x8` | 32 | 1 | `tt-smi` 验证完整 32 卡组 |

例如，四卡 8 worker：

```bash
FLUX3_EXECUTION_PROFILE=sp4
FLUX3_PARENT_MESH_MODE=direct
```

八卡 4 worker、SP8 Ring：

```bash
FLUX3_EXECUTION_PROFILE=sp8
FLUX3_PARENT_MESH_MODE=direct
```

八卡 4 worker、纯 TP8 Ring：

```bash
FLUX3_EXECUTION_PROFILE=tp8ring
FLUX3_PARENT_MESH_MODE=direct
```

## 3. 单 worker 模式

自动发现仍然执行，但用 `FLUX3_WORKER_GROUP_INDEX` 只保留一个发现出来的 worker 组。单卡 `legacy-single` 也会通过 `tt-smi` 发现单卡列表后选择一个设备。这个变量适合单 worker 性能测试或调试，不建议生产服务使用。

单卡一个 worker：

```bash
FLUX3_EXECUTION_PROFILE=legacy-single
FLUX3_WORKER_GROUP_INDEX=0
```

双卡一个 worker：

```bash
FLUX3_EXECUTION_PROFILE=sp2
FLUX3_WORKER_GROUP_INDEX=0
```

四卡一个 worker：

```bash
FLUX3_EXECUTION_PROFILE=sp4
FLUX3_WORKER_GROUP_INDEX=0
```

八卡一个 worker：

```bash
FLUX3_EXECUTION_PROFILE=sp8
FLUX3_WORKER_GROUP_INDEX=0
```

如果需要测试第 N 个物理 worker 组，将 `FLUX3_WORKER_GROUP_INDEX` 改为 `0` 到 `worker_count-1` 的值。该方式仍由 `DeviceManager` 生成物理合法分组，不允许任意卡号组合。

## 4. 父 mesh/子 mesh 模式

父子 mesh 模式设置：

```bash
FLUX3_PARENT_MESH_MODE=parent
FLUX3_SUBMESH_INDEX=0
```

实现流程为：

```text
打开完整 Galaxy 父 mesh
        ↓
create_submeshes(child_shape)
        ↓
选择 FLUX3_SUBMESH_INDEX 对应的子 mesh
        ↓
模型只在选中的子 mesh 上运行
```

父子 mesh 模式和 direct 自动多 worker 模式是两种不同的部署方式。父 mesh 由一个进程独占，因此当前实现强制 `FLUX3_PARENT_MESH_WORKERS=1`；不能把同一个父 mesh 同时交给多个独立 worker 进程。

### 已支持的四卡父子 mesh

| Profile | 父 mesh | 子 mesh | 子 mesh 数量 |
|---|---:|---:|---:|
| `sp4` | `4x8` | `4x1` | 8 |
| `tp4` | `4x8` | `4x1` | 8 |
| `tp4axis1` | `4x8` | `1x4` | 8 |

示例：

```bash
FLUX3_EXECUTION_PROFILE=sp4
FLUX3_PARENT_MESH_MODE=parent
FLUX3_SUBMESH_INDEX=0
```

### 已支持的八卡父子 mesh

| Profile | 父 mesh | 子 mesh | 子 mesh 数量 |
|---|---:|---:|---:|
| `sp8` | `8x4` | `8x1` | 4 |
| `tp8ring` | `4x8` | `1x8` | 4 |
| `tp2sp4` | `8x4` | `2x4` | 4 |

示例：

```bash
FLUX3_EXECUTION_PROFILE=sp8
FLUX3_PARENT_MESH_MODE=parent
FLUX3_SUBMESH_INDEX=0
```

## 5. 当前明确的边界

- `sp4` direct Ring 是四卡默认生产路径；不要把它自动切换成父子 mesh。
- `sp8` 和 `tp8ring` 的 direct Ring 路径保留，父子 mesh 只通过显式 `FLUX3_PARENT_MESH_MODE=parent` 启用。
- `tp2sp4` 已有 direct 自动分组和父子 mesh 两条路径。
- `tp4sp2` 已有 direct 自动分组；当前没有纳入父子 mesh profile，不能在使用说明中写成已验证的父子 mesh 场景。
- `FLUX3_WORKER_GROUP_INDEX` 只用于单 worker 选组，不改变物理拓扑发现算法。
- 所有性能测试应固定 `TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`，并使用同步串行请求。

## 6. 代码职责

| 文件 | 职责 |
|---|---|
| `tt-media-server/config/settings.py` | 规范化 profile、确定 mesh、选择 direct 或 parent 模式、生成 worker 数量 |
| `tt-media-server/utils/device_manager.py` | 使用 `tt-smi` 或 `test_system_health` 发现物理合法 GPU 分组 |
| `tt-media-server/utils/runner_utils.py` | 将 profile 映射为父 mesh、Fabric、Ring/Linear 和 core-grid 环境 |
| `tt-media-server/tt_model_runners/base_metal_device_runner.py` | 打开父 mesh、创建子 mesh、选择 `FLUX3_SUBMESH_INDEX` |
| `tt-media-server/tt_model_runners/dit_runners.py` | 解释 TP/SP 语义、建立模型 pipeline 和并行参数 |
| `tt-metal/tt_metal/fabric/mesh_graph_descriptors/` | 提供 Galaxy 的 mesh/Fabric descriptor |

## 7. 启动前检查

```bash
echo "$FLUX3_EXECUTION_PROFILE"
echo "$FLUX3_PARENT_MESH_MODE"
echo "$FLUX3_WORKER_GROUP_INDEX"
echo "$FLUX3_SUBMESH_INDEX"
echo "$TT_MM_THROTTLE_PERF $TT_METAL_TDP_LIMIT_WATTS"
```

日志中应重点确认：

```text
Applied FLUX.3 execution profile ...
device_mesh_shape=(...)
device_ids(after)=...
TT_VISIBLE_DEVICES=...
Created mesh device with ... devices
```

父子 mesh 还应出现：

```text
Configured FLUX.3 ... parent mesh ...
Created parent mesh (...) and selected child mesh (...)
```
