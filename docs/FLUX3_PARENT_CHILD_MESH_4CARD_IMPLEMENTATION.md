# FLUX.3 Klein：4 卡父 Mesh / 子 Mesh 实现步骤

## 结论

4 卡父 Mesh / 子 Mesh 已按 TT-Metal 的真实 `MeshDevice` 语义实现：一个进程打开完整 Galaxy 父 Mesh，再通过 `create_submeshes()` 创建合法的 4 卡子 Mesh，模型只绑定到选中的子 Mesh。不会手工拼接 PCIe 卡号，也不会让多个独立进程重复打开同一个 32 卡父 Mesh。

当前 4 卡 profile：

| Profile | 模型子 Mesh | 父 Mesh | 并行语义 |
| --- | ---: | ---: | --- |
| `sp4` | `4x1` | `4x8` | TP1 × SP4，Ring |
| `tp4` | `4x1` | `4x8` | TP4，Ring |
| `tp4axis1` | `1x4` | `4x8` | TP4，轴转置 |
| `tp2sp2` | `2x2` | `8x4` | TP2 × SP2，2D |

## 启动链路

```text
读取 FLUX3_EXECUTION_PROFILE
        ↓
解析 worker_mesh、parent_mesh、TP/SP 轴和 Fabric 配置
        ↓
DeviceManager 发现完整 32 卡 Galaxy 组
        ↓
一个进程打开父 Mesh：ttnn.open_mesh_device(parent_shape)
        ↓
父 Mesh create_submeshes(child_shape)
        ↓
根据 FLUX3_SUBMESH_INDEX 选择一个合法子 Mesh
        ↓
模型、CCL、权重缓存和 pipeline 全部绑定到子 Mesh
```

## 代码职责

1. `tt-media-server/config/flux3_profiles.py`

   保存 profile 的唯一事实来源：worker mesh、parent mesh、TP/SP 轴、Ring/Linear 和 Fabric 配置。

2. `tt-media-server/config/settings.py`

   将 `FLUX3_EXECUTION_PROFILE` 解析为 worker 配置。父子模式下把完整 Galaxy 组交给一个 worker，并强制 `FLUX3_PARENT_MESH_WORKERS=1`，避免多个进程争用同一个父 Mesh。

3. `tt-media-server/tt_model_runners/base_metal_device_runner.py`

   先打开父 Mesh，再调用：

   ```python
   parent_device = ttnn.open_mesh_device(mesh_shape=parent_shape, ...)
   child_meshes = parent_device.create_submeshes(ttnn.MeshShape(child_shape))
   mesh_device = child_meshes[FLUX3_SUBMESH_INDEX]
   ```

   runner 保存父 Mesh 引用，关闭时关闭父 Mesh，不直接关闭子 Mesh view。

4. `tt-media-server/utils/runner_utils.py`

   处理 scheduler 的 `::submesh=N` 标识，并把它转换为 `FLUX3_SUBMESH_INDEX`。子 Mesh 的物理位置由 TT-Metal descriptor 决定。

5. `tt-metal/tt_metal/fabric/mesh_graph_descriptors/`

   提供 Galaxy 的物理连接和 Fabric placement。父子 Mesh 只使用 descriptor 能表达的真实拓扑。

## 4 卡启动示例

### SP4

```bash
FLUX3_EXECUTION_PROFILE=sp4
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=4x8
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
```

### TP4

```bash
FLUX3_EXECUTION_PROFILE=tp4
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=4x8
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
```

### TP4 轴转置

```bash
FLUX3_EXECUTION_PROFILE=tp4axis1
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=4x8
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
```

### TP2 × SP2

```bash
FLUX3_EXECUTION_PROFILE=tp2sp2
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=8x4
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
```

## 验证结果

在 Galaxy03 的父子 Mesh 临时容器中，使用以下固定条件完成 4 卡 SP4 子 Mesh 的真实请求：

```text
父 Mesh：4x8
子 Mesh：4x1
Profile：sp4
Fabric：FABRIC_1D_RING
Core grid：12x10
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
分辨率：1024x1024
步数：16
seed：42
输出：JPEG
```

结果：HTTP 200，图片有效；日志确认出现 `Created parent mesh (4, 8) and selected child mesh (4, 1)`。本次请求包含首次 kernel 编译，不作为热态性能基线。

## 安全边界

- 不使用 `FLUX3_DEVICE_IDS` 手工指定任意 GPU 列表。
- 不把连续的 `(0,1,2,3)` 当作合法 SP4 Ring；物理 placement 必须由 Fabric descriptor 决定。
- 父子模式是单进程持有父 Mesh 的模式；需要 8 个独立 4 卡 worker 时使用 direct 自动分组模式。
- `sp4` 的 direct Ring 默认路径保留，父子模式通过 `FLUX3_PARENT_MESH_MODE=parent` 显式启用。
