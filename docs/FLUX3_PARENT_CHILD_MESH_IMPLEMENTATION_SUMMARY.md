# FLUX.3 Klein 父 Mesh / 子 Mesh 实现总结

更新时间：2026-11-18

## 1. 机制目的

Galaxy 不是简单地把前 N 个 GPU 传给进程。真实设备具有 Galaxy 拓扑、Fabric placement 和合法 mesh descriptor。父 Mesh/子 Mesh 模式的目标是：

```text
一个 worker 打开完整、拓扑合法的 Galaxy 父 Mesh
        ↓
从父 Mesh 创建一个合法的 2/4/8 卡子 Mesh
        ↓
模型、CCL、权重和 VAE 只绑定到子 Mesh
```

这样可以避免手工指定 `(0,1,2,3)` 导致 Fabric placement 与物理连接不匹配。

## 2. Direct 与 parent/child

| 模式 | 设备打开方式 | 适用场景 |
|---|---|---|
| Direct | worker 直接打开自己的合法设备组 | 历史高性能路径、多个 worker 并行 |
| Parent/child | 单进程打开完整父 Mesh，再选择一个子 Mesh | Galaxy 子网格和父拓扑验证 |

父子模式不是让多个独立进程同时争用同一个 32 卡父 Mesh。一个 parent owner 负责打开和关闭父设备，子 Mesh 是父设备创建的 view。

## 3. 代码职责

### 3.1 Profile 表

文件：

```text
tt-media-server/config/flux3_profiles.py
```

当前父子 profile 映射：

| Profile | Parent Mesh | Child Mesh | 并行语义 |
|---|---:|---:|---|
| `sp2` | 4x8 | 2x1 | TP1 × SP2 |
| `tp2` | 4x8 | 2x1 | TP2 |
| `sp4` | 4x8 | 4x1 | TP1 × SP4 |
| `tp4` | 4x8 | 4x1 | TP4 |
| `tp4axis1` | 4x8 | 1x4 | 转置 TP4 |
| `tp2sp2` | 8x4 | 2x2 | TP2 × SP2 |
| `sp8` | 8x4 | 8x1 | TP1 × SP8 |
| `tp8ring` | 4x8 | 1x8 | TP8 Ring |
| `tp2sp4` | 8x4 | 2x4 | TP2 × SP4 |
| `tp4sp2` | 8x4 | 4x2 | TP4 × SP2 |

### 3.2 Settings 与设备发现

文件：

```text
tt-media-server/config/settings.py
```

职责：

1. 读取 `FLUX3_EXECUTION_PROFILE`；
2. 为 profile 选择 parent shape 和 child shape；
3. 将 parent profile 加入 discovery；
4. 要求 parent/child 使用完整 Galaxy 的合法 descriptor；
5. 防止子网格被误判成普通独立设备组。

### 3.3 Runner 与 Fabric

文件：

```text
tt-media-server/utils/runner_utils.py
```

处理链路：

```text
profile
  → parent shape
  → parent Fabric
  → descriptor
  → child mesh selection
```

当前 parent Fabric 规则：

- SP/Ring profile：`1d_ring`；
- 纯 TP2：`1d`；
- `tp2sp2`、`tp2sp4`、`tp4sp2`：`2d`。

Parent Fabric 负责父 Mesh placement；child topology 负责模型子 Mesh 的 collective 语义，两者不是同一个字段。

## 4. 启动流程

```text
读取 FLUX3_EXECUTION_PROFILE
        ↓
解析 parent/child shape
        ↓
发现完整 32 芯片 Galaxy 组
        ↓
打开 parent Mesh
        ↓
parent.create_submeshes(child_shape)
        ↓
选择 FLUX3_SUBMESH_INDEX
        ↓
在 child Mesh 上创建 CCL、Transformer、VAE 和 pipeline
        ↓
warmup 完成，worker ready
```

关闭时由 parent owner 负责释放设备，不能把 child view 当作独立设备重复关闭或重新打开。

## 5. 环境变量示例

4 卡 SP4：

```bash
FLUX3_EXECUTION_PROFILE=sp4
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=4x8
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
FLUX3_CORE_GRID_OVERRIDE=12x10
```

8 卡 `tp2sp4`：

```bash
FLUX3_EXECUTION_PROFILE=tp2sp4
FLUX3_PARENT_MESH_MODE=parent
FLUX3_PARENT_MESH_SHAPE=8x4
FLUX3_SUBMESH_INDEX=0
FLUX3_PARENT_MESH_WORKERS=1
```

## 6. 真机回归结果

固定条件：1024×1024、16 steps、seed 42、同步串行、JPEG、`12x10` core grid、TDP 160W、`TT_MM_THROTTLE_PERF=0`。

| Profile | Parent -> Child | Parent Fabric | HTTP | Denoise | VAE | 结果 |
|---|---|---|---:|---:|---:|---|
| `sp2` | 4x8 -> 2x1 | 1D Ring | 6.329s | 5.544s | 0.658s | HTTP 200 |
| `tp2` | 4x8 -> 2x1 | 1D Linear | 4.289s | 3.498s | 0.658s | HTTP 200 |
| `sp4` | 4x8 -> 4x1 | 1D Ring | 3.327s | 2.857s | 0.343s | HTTP 200 |
| `tp4` | 4x8 -> 4x1 | 1D Ring | 4.030s | — | — | HTTP 200 |
| `tp4axis1` | 4x8 -> 1x4 | 1D Ring | 5.045s | — | — | HTTP 200 |
| `tp2sp2` | 8x4 -> 2x2 | 2D | 4.298s | 3.518s | 0.658s | HTTP 200 |
| `sp8` | 8x4 -> 8x1 | 1D Ring | 2.580s | 1.832s | 0.624s | HTTP 200 |
| `tp8ring` | 4x8 -> 1x8 | 1D Ring | 3.464s | — | — | HTTP 200 |
| `tp2sp4` | 8x4 -> 2x4 | 2D | 2.773s | 2.306s | 0.344s | HTTP 200 |
| `tp4sp2` | 8x4 -> 4x2 | 2D | 3.349s | 2.565s | 0.657s | HTTP 200 |

`sp8tp4` 继续保持 32 卡 direct 全 Galaxy；本轮没有拆成 parent/child。21:9 和 9:21 也没有纳入本轮调试。

## 7. 重要边界

- 不手工把连续卡号当成 Ring placement；
- 不让多个独立 worker 同时争用同一个 parent Mesh；
- parent Fabric 只负责父 Mesh placement，不能据此推断 child topology；
- 首次 parent/child warmup 会编译大量 kernel，不能把首次启动时间当成热态性能；
- 现有 direct 路径保留，可作为回退和性能对照；
- 镜像尚未因本轮文档更新而重建。

## 8. 代码版本与日志

```text
提交：1ae19312 flux3: complete parent child mesh profile regression
回归日志：/local/ttcache/flux3_parent_submesh_tests.log
结果目录：/local/ttcache/flux3-parent-submesh-v124/
```
