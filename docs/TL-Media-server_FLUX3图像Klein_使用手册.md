# TL-Media-server — FLUX.3 Image Klein v1.0.35 使用手册

> 当前正式使用手册。完整适配逻辑见 `FLUX3模型适配实现逻辑.md`。

## 1. 镜像与构建

正式镜像：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

镜像可直接由当前两个仓库构建；模型权重不会进入镜像。

```bash
cd /local/lnn/TL-inference-server
docker build --progress=plain \
  --build-context ttmetal=/local/lnn/tt-metal \
  -f Dockerfile \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=4cca8b1f8c \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35 .
```

构建前应核对 tag：

```bash
git -C /local/lnn/TL-inference-server describe --tags --always
git -C /local/lnn/tt-metal describe --tags --always
```

## 2. Galaxy worker 拓扑

生产部署只设置一个 `FLUX3_EXECUTION_PROFILE`，服务会按 profile 和 Fabric 拓扑自动发现设备组。不要手工设置 `FLUX3_DEVICE_IDS` 或连续卡号。

| profile | 每 worker 芯片数 | Galaxy worker 数 | 并行布局 |
| --- | ---: | ---: | --- |
| `legacy-single` | 1 | 32 | 独立单卡 |
| `sp2` | 2 | 16 | TP1×SP2 Ring |
| `tp2` | 2 | 16 | TP2 Linear |
| `sp4` | 4 | 8 | TP1×SP4 Ring |
| `tp4` | 4 | 8 | TP4，4×1 Ring |
| `tp4axis1` | 4 | 8 | TP4，1×4 Ring |
| `tp2sp2` | 4 | 8 | TP2×SP2 2D Linear |
| `tp2sp4` / `tp4sp2` | 8 | 4 | 2D Linear |
| `sp8` | 8 | 4 | TP1×SP8，8×1 Ring |
| `tp8ring` | 8 | 4 | TP8，1×8 Ring |
| `sp8tp4` | 32 | 1 | TP4×SP8 Ring |

所有 profile 使用 12×10 core grid。多 worker 默认按固定约 10 秒间隔启动，不会等待前一个 worker ready。

## 3. 在 Galaxy 上启动服务

以下示例启动 8 个四卡 SP4 worker，并监听 8000。当前镜像内统一使用 `/local/ttcache`；服务只识别 `OPENAI_API_KEY`，不使用旧的 `API_KEY` 配置。

11.3 已验证的双服务部署为：9000 端口单 worker、4 卡 `sp4`；8000 端口单 worker、8 卡 `sp8`。当前验证镜像为 `galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35`。同时启动时先启动 SP4 并等待 `/health` 为 200，再启动 SP8；这样可以避免设备发现阶段的 worker group 竞争。

```bash
export FLUX3_API_KEY='replace-with-a-long-random-secret'

docker rm -f flux3-image-klein 2>/dev/null || true
docker run -d --name flux3-image-klein \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image \
  -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy \
  -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e FLUX3_COMPONENT_RESIDENCY=resident \
  -e FLUX3_REFERENCE_ENCODER_BACKEND=ttnn \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY="${FLUX3_API_KEY}" \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

切换 profile 时仅替换 `FLUX3_EXECUTION_PROFILE` 值。多卡服务启动前先停止旧容器并执行 `tt-smi -glx_reset_auto`。多 worker 默认按固定 10 秒 start-to-start 间隔启动，不等待前一个 worker ready；只有日志出现 `All workers ready (model readiness gated by per-runner warmup)` 后，才表示全部 worker 已可用。

标准部署不要强行把所有 profile 的 warmup 都设成 `1024,2048,4096`。未设置 `FLUX3_WARMUP_SIZES` 时，runner 会按 profile/mesh 选择安全的启动编译集合。warmup 只影响启动阶段，不改变 API 对官方尺寸的校验。

默认生产部署使用 direct mesh：每个 worker 自动获得一个独立且拓扑合法的物理 group，因此可拆分为 32、16、8 或 4 个 worker。parent/child mesh 也支持多 worker：一个完整 parent mesh 会切分为多个互不重叠的 child mesh，再由 worker manager 为每个 worker 分配唯一的 `FLUX3_SUBMESH_INDEX`。direct 是默认推荐路径；parent/child 通过显式 `FLUX3_PARENT_MESH_MODE=parent` 启用。两种模式都不能重复占用同一个设备组或手工构造非法 GPU 列表。

## 4. API 概览

- `GET /health`、`GET /tt-liveness`、`GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/generations/async`
- `GET /v1/images/generations/{job_id}`
- `POST /v1/images/generations/{job_id}/cancel`

同步图片生成：

```bash
curl --fail-with-body --show-error \
  -X POST http://127.0.0.1:8000/v1/images/generations \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42,
    "response_format": "file",
    "image_return_format": "JPEG"
  }' --output black_bird.jpg
```

尺寸契约：宽高各为 16–4096、且为 16 的倍数，不额外限制总像素或宽高比。当前 FLUX3 一次请求只生成 1 张图片，等价于 `batch=1`；多图 batch 暂不支持。`reference_images` API/runner 入口支持 1–10 张参考图，当前代码会将它们按顺序传入 pipeline。单参考图保持宽高比，单边 64–4096、约 4MP 上限并按 16 像素网格对齐。详细字段参见 `FLUX3图像Klein_API手册.md`。

## 5. 启动方式总览

启动方式分为两类，必须先确定模式，再选择对应命令：

1. **direct 模式**：服务直接从 Fabric 自动发现合法的物理 worker group。适合生产部署，也适合只启动一个 worker 的调试部署。多 worker 时由服务按 profile 自动拆分设备，不手工填写连续 GPU 卡号。
2. **parent/child mesh 模式**：服务先打开一个完整的 parent mesh，再从 parent mesh 中创建互不重叠的 child mesh，并把 child mesh 分配给 worker。适合验证父子 mesh、2D 子 mesh 和统一 parent Fabric 的部署方式。

两种模式都要求 child/worker 的设备分组合法且不能重叠；不能把 `TT_VISIBLE_DEVICES`、连续卡号和自动拓扑发现混用。

## 6. direct 模式

### 6.1 公共启动参数

所有场景都建议使用下面的公共配置：

```bash
IMAGE=galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
MODEL_ROOT=/local/models/flux-3-image-klein
CACHE_ROOT=/local/ttcache/flux-3-image-klein
FLUX3_API_KEY=turing123

COMMON_ARGS="\
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v ${CACHE_ROOT}:/local/ttcache \
  -e MODEL_SERVICE=image \
  -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy \
  -e FLUX3_MODEL_ROOT=${MODEL_ROOT} \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e FLUX3_COMPONENT_RESIDENCY=resident \
  -e FLUX3_REFERENCE_ENCODER_BACKEND=ttnn \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=${FLUX3_API_KEY} \
  ${IMAGE}"
```

实际执行时不要把 `COMMON_ARGS` 原样作为一个字符串传给 `docker run`；下面的场景命令已经把关键参数展开。生产环境只设置 `OPENAI_API_KEY`。

### 6.2 direct：单 worker

单 worker 只占用一个 profile 对应的设备组。适用于 1 卡、2 卡、4 卡、8 卡或 32 卡单 worker 场景。通过 `FLUX3_WORKER_GROUP_INDEX` 选择已发现的合法 worker group；不要设置 `TT_VISIBLE_DEVICES`。

例如，4 卡 SP4 单 worker：

```bash
docker run -d --name flux3-sp4-1worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_WORKER_GROUP_INDEX=0 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

将 `sp4` 替换为 `legacy-single`、`sp2`、`tp2`、`sp8`、`tp8ring` 或 `sp8tp4`，即可选择其它已支持的单 worker profile；profile 对应的设备数量必须与实际资源一致。

### 6.3 direct：多 worker

多 worker 时不设置 `FLUX3_WORKER_GROUP_INDEX`，服务自动发现整机并按 profile 拆分 worker：

| 场景 | Profile 示例 | Worker 数 |
|---|---|---:|
| 1 卡 × 32 worker | `legacy-single` | 32 |
| 2 卡 × 16 worker | `sp2` 或 `tp2` | 16 |
| 4 卡 × 8 worker | `sp4` 或 `tp4` | 8 |
| 8 卡 × 4 worker | `sp8` 或 `tp8ring` | 4 |
| 32 卡 × 1 worker | `sp8tp4` | 1 |

下面的 2 卡、4 卡、8 卡和 32 卡章节都是 direct 多 worker 示例。多 worker 启动时服务按固定间隔拉起 worker，不等待前一个 worker ready；只有日志出现 `All workers ready` 才表示整组服务可用。

### 6.4 direct 多 worker：单卡：1 卡 × 32 worker

一个容器发现整机 32 个单卡 worker，每个 worker 占用 1 张卡：

```bash
docker rm -f flux3-legacy-single 2>/dev/null || true
docker run -d --name flux3-legacy-single \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=legacy-single \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

这里不要设置 `FLUX3_WORKER_GROUP_INDEX`，因为服务需要自动发现并打开 32 个单卡 worker。

### 6.5 direct 多 worker：双卡：2 卡 × 16 worker

推荐使用空间并行 `sp2`：

```bash
docker rm -f flux3-sp2-16worker 2>/dev/null || true
docker run -d --name flux3-sp2-16worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled -e FLUX3_EXECUTION_PROFILE=sp2 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

若要测试纯 TP2，把 `FLUX3_EXECUTION_PROFILE=sp2` 改为 `tp2`。多 worker 模式不要设置单 worker 专用的 `FLUX3_WORKER_GROUP_INDEX`。

### 6.6 direct 多 worker：四卡：4 卡 × 8 worker

默认生产 profile 是 `sp4`，使用 TP1×SP4 Ring：

```bash
docker rm -f flux3-sp4-8worker 2>/dev/null || true
docker run -d --name flux3-sp4-8worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

四卡可选 profile：

| profile | 布局 | 通信/用途 |
|---|---|---|
| `sp4` | TP1×SP4，4×1 | Ring；默认、性能最佳路径 |
| `tp4` | TP4，4×1 | 纯 TP4 Ring |
| `tp4axis1` | TP4，1×4 | 纯 TP4 轴向对照 |
| `tp2sp2` | TP2×SP2，2×2 | 2D Linear 对照 |

只需要替换 `FLUX3_EXECUTION_PROFILE`，不要手动填写连续卡号。

### 6.7 direct 多 worker：八卡：8 卡 × 4 worker

默认生产 profile 是 `sp8`，使用 TP1×SP8 Ring：

```bash
docker rm -f flux3-sp8-4worker 2>/dev/null || true
docker run -d --name flux3-sp8-4worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled -e FLUX3_EXECUTION_PROFILE=sp8 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

八卡可选 profile：

| profile | 布局 | 通信/用途 |
|---|---|---|
| `sp8` | TP1×SP8，8×1 | Ring；默认八卡路径 |
| `tp8ring` | TP8，1×8 | 纯 TP8 Ring |
| `tp2sp4` | TP2×SP4，2×4 | 2D Linear |
| `tp4sp2` | TP4×SP2，4×2 | 2D Linear |

### 6.8 direct 多 worker：三十二卡：32 卡 × 1 worker

整机单 worker 使用 `sp8tp4`：

```bash
docker rm -f flux3-sp8tp4-1worker 2>/dev/null || true
docker run -d --name flux3-sp8tp4-1worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled -e FLUX3_EXECUTION_PROFILE=sp8tp4 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

## 7. parent/child mesh 模式

### 7.1 parent/child mesh 公共参数

父子 mesh 模式通过以下参数启用：

```bash
-e FLUX3_PARENT_MESH_MODE=parent \
-e FLUX3_PARENT_MESH_SHAPE=<parent-shape> \
-e FLUX3_PARENT_MESH_WORKERS=<worker-count> \
-e FLUX3_CORE_GRID_OVERRIDE=12x10
```

`FLUX3_PARENT_MESH_SHAPE` 是整机 parent 的逻辑形状，`FLUX3_PARENT_MESH_WORKERS` 是从 parent 中分配的 child worker 数量。多 worker 时不要设置 `FLUX3_SUBMESH_INDEX`，由 worker manager 自动为每个 worker 分配唯一 child；单 worker 调试时才显式设置 `FLUX3_SUBMESH_INDEX=0`。

### 7.2 parent/child mesh 多 worker

父子 mesh 模式不需要手工填写 GPU 卡号。服务先打开完整 parent mesh，再按 profile 创建 child mesh；worker manager 为每个 worker 自动分配不同的 `FLUX3_SUBMESH_INDEX`。

| 场景 | Parent mesh | Child mesh | 最大 worker 数 |
|---|---:|---:|---:|
| 双卡 `sp2` / `tp2` | `4x8` | `2x1` | 16 |
| 四卡 `sp4` / `tp4` | `4x8` | `4x1` | 8 |
| 四卡 `tp4axis1` | `4x8` | `1x4` | 8 |
| 四卡 `tp2sp2` | `8x4` | `2x2` | 8 |
| 八卡 `sp8` | `8x4` | `8x1` | 4 |
| 八卡 `tp8ring` | `4x8` | `1x8` | 4 |
| 八卡 `tp2sp4` | `8x4` | `2x4` | 4 |
| 八卡 `tp4sp2` | `8x4` | `4x2` | 4 |

例如，启动 8 个四卡 `sp4` worker：

```bash
docker rm -f flux3-sp4-parent-8worker 2>/dev/null || true
docker run -d --name flux3-sp4-parent-8worker \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -e MODEL_SERVICE=image \
  -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy \
  -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_PARENT_MESH_MODE=parent \
  -e FLUX3_PARENT_MESH_SHAPE=4x8 \
  -e FLUX3_PARENT_MESH_WORKERS=8 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY=turing123 \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.35
```

多 worker 模式不要手工设置 `FLUX3_SUBMESH_INDEX`；该变量只用于单 worker 调试，例如 `FLUX3_PARENT_MESH_WORKERS=1` 配合 `FLUX3_SUBMESH_INDEX=0`。不要同时设置 `FLUX3_WORKER_GROUP_INDEX`，也不要设置 `TT_VISIBLE_DEVICES`。

## 8. 当前 11.3 运行的单 worker 双服务

Galaxy03 当前验证过的部署如下：

```text
9000: FLUX3_EXECUTION_PROFILE=sp4，FLUX3_WORKER_GROUP_INDEX=4，单 worker、4 卡
8000: FLUX3_EXECUTION_PROFILE=sp8，FLUX3_WORKER_GROUP_INDEX=0，单 worker、8 卡
```

单 worker 模式只选择一个已发现的 worker group；不要设置 `TT_VISIBLE_DEVICES` 或手写物理卡号。两个容器同时启动时必须先让 9000/SP4 ready，再启动 8000/SP8，否则设备发现可能暂时报告 0 个可用 worker group。

## 9. 启动完成检查、日志和停止

```bash
# 健康检查；模型未 ready 时可能返回非 200 和 {"detail":"Model is not ready"}
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/tt-liveness
curl -sS http://127.0.0.1:8000/v1/models | jq .

# 观察 worker ready、设备、warmup 和每次推理阶段日志
docker logs -f --since 0s flux3-sp8-4worker

# 停止和删除当前容器
docker stop flux3-sp8-4worker
docker rm flux3-sp8-4worker
```

只有看到下面这些日志，才算模型真正 ready：

```text
upstream FLUX.3 Klein warmup completed
Worker ... reported ready
All workers ready (model readiness gated by per-runner warmup)
```

## 10. API 详细说明

### 10.1 鉴权和通用请求头

```bash
export ENDPOINT=http://127.0.0.1:8000
export FLUX3_API_KEY=turing123

COMMON_HEADERS=(
  -H "Authorization: Bearer ${FLUX3_API_KEY}"
  -H "Content-Type: application/json"
)
```

未携带鉴权或密钥错误返回 `401 Unauthorized`。请求模型名固定使用 `tt-flux.3-image-klein`。

### 10.2 JSON 请求方式

JSON 方式适合普通文生图，也适合把参考图作为 Base64/data URL 放进
`reference_images`。请求头使用 `Content-Type: application/json`；同步接口可以返回图片文件或 JSON/Base64。

#### 10.2.1 T2I 同步请求，返回 JPEG 文件

```bash
curl --fail-with-body --show-error \
  -X POST "${ENDPOINT}/v1/images/generations" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird in a forest",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42,
    "number_of_images": 1,
    "response_format": "file",
    "image_return_format": "JPEG",
    "image_quality": 85
  }' \
  --output black_bird.jpg
```

`response_format=file` 时响应 body 是图片二进制，必须使用 `--output` 或程序以二进制方式读取。验证文件：

```bash
file black_bird.jpg
identify black_bird.jpg 2>/dev/null || true
```

#### 10.2.2 T2I JSON 请求，返回 Base64

```bash
curl --fail-with-body --show-error \
  -X POST "${ENDPOINT}/v1/images/generations" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird in a forest",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42,
    "response_format": "b64_json",
    "image_return_format": "PNG"
  }' > response.json

# 需要 jq 时把 Base64 解码成图片
jq -r '.data[0].b64_json' response.json | base64 -d > black_bird.png
```

#### 10.2.3 I2I：JSON 携带单张参考图

I2I 不是“只上传图片”。请求仍然必须带 `prompt`；参考图通过 `reference_images` 传入，模型同时使用文本条件和图像条件。

```bash
REF_B64=$(base64 -w 0 reference.jpg)
cat > i2i.json <<EOF
{
  "model": "tt-flux.3-image-klein",
  "prompt": "turn the bird into a watercolor illustration",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 16,
  "seed": 42,
  "reference_images": ["data:image/jpeg;base64,${REF_B64}"],
  "response_format": "file",
  "image_return_format": "JPEG"
}
EOF

curl --fail-with-body --show-error \
  -X POST "${ENDPOINT}/v1/images/generations" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @i2i.json --output i2i_result.jpg
```

#### 10.2.4 JSON 携带多参考图：2～10 张

`reference_images` 是有序数组，数组元素使用 Base64 或 data URL；不接受 HTTP URL。一次请求仍只返回 1 张输出图，`number_of_images` 必须保持为 1。

```bash
REF1=$(base64 -w 0 reference_1.jpg)
REF2=$(base64 -w 0 reference_2.jpg)
cat > multi_ref.json <<EOF
{
  "model": "tt-flux.3-image-klein",
  "prompt": "create one coherent product image using the reference images",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 16,
  "seed": 42,
  "number_of_images": 1,
  "reference_images": [
    "data:image/jpeg;base64,${REF1}",
    "data:image/jpeg;base64,${REF2}"
  ],
  "response_format": "file",
  "image_return_format": "JPEG"
}
EOF

curl --fail-with-body --show-error --max-time 300 \
  -X POST "${ENDPOINT}/v1/images/generations" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @multi_ref.json --output multi_ref_result.jpg
```

把数组继续扩展到最多 10 个元素即可。参考图越多，`encode_reference` 和 denoise 时间越长；大尺寸图片还会显著增大请求体，因此客户端应提高连接和读取超时。

#### 10.2.5 异步 JSON 请求

```bash
curl --fail-with-body --show-error \
  -X POST "${ENDPOINT}/v1/images/generations/async" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42,
    "response_format": "b64_json",
    "image_return_format": "JPEG"
  }'
```

返回 JSON 中取得 `job_id` 后轮询：

```bash
curl --fail-with-body --show-error \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  "${ENDPOINT}/v1/images/generations/<job_id>"
```

异步任务完成后返回 JSON/Base64，不返回同步接口的直接图片二进制。取消任务：

```bash
curl --fail-with-body --show-error -X POST \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  "${ENDPOINT}/v1/images/generations/<job_id>/cancel"
```

### 10.3 multipart/form-data 文件上传方式

multipart 方式适合客户端直接上传本地图片文件，不需要先在客户端做 Base64 编码。
普通参数作为文本 form field 发送，参考图作为文件 field 发送；同步和异步接口都支持这种方式。

#### 10.3.1 同步 multipart 请求

同步和异步图片接口都支持直接上传参考图文件。普通参数使用 `-F` 传递，参考图使用
重复的 `reference_images` 文件字段；服务按字段出现顺序接收 1～10 张图片。

```bash
curl --fail-with-body --show-error --max-time 1200 \
  -X POST "${ENDPOINT}/v1/images/generations" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -F "model=tt-flux.3-image-klein" \
  -F "prompt=combine the object from image one with the scene from image two" \
  -F "width=1024" \
  -F "height=1024" \
  -F "num_inference_steps=16" \
  -F "seed=42" \
  -F "response_format=file" \
  -F "image_return_format=JPEG" \
  -F "reference_images=@reference_1.jpg;type=image/jpeg" \
  -F "reference_images=@reference_2.png;type=image/png" \
  --output multipart_result.jpg
```

#### 10.3.2 异步 multipart 请求

```bash
curl --fail-with-body --show-error \
  -X POST "${ENDPOINT}/v1/images/generations/async" \
  -H "Authorization: Bearer ${FLUX3_API_KEY}" \
  -F "model=tt-flux.3-image-klein" \
  -F "prompt=create one coherent product image from these references" \
  -F "width=1024" \
  -F "height=1024" \
  -F "num_inference_steps=16" \
  -F "seed=42" \
  -F "reference_images=@reference_1.jpg;type=image/jpeg" \
  -F "reference_images=@reference_2.jpg;type=image/jpeg"
```

两种接口都兼容单文件字段 `reference_image` 和 `image`。上传文件必须声明为
`image/*` 且不能为空；默认单文件大小上限为 32 MiB，可通过
`FLUX3_REFERENCE_UPLOAD_MAX_BYTES` 调整。非法图片返回 `422`，超过上限返回 `413`。
JSON/Base64 请求仍然可用，multipart 不改变 1～10 张参考图和一次只生成 1 张图片的
现有契约。

### 10.4 API 字段和错误

| 字段 | 当前规则 |
|---|---|
| `prompt` | 必填；T2I/I2I 都需要 |
| `width` / `height` | 16～4096，且必须是 16 的倍数 |
| `num_inference_steps` | 12～50；Distilled 常用 16 |
| `seed` | 整数或 `null` |
| `number_of_images` | 当前只能为 1；多图 batch 尚未开放 |
| `reference_image` | 单图兼容字段 |
| `reference_images` | 1～10 个 Base64/data URL；与 `reference_image` 互斥 |
| `response_format` | `file` 或 `b64_json` |
| `image_return_format` | `JPEG` 或 `PNG` |
| `image_quality` | JPEG 时 50～100 |

常见状态码：`200` 成功；`401` 鉴权失败；`422` 请求字段、尺寸、参考图数量或格式非法；`405` 服务尚未 ready 时的健康接口响应；`500` 表示服务内部错误，应优先查看 runner 和 pipeline 日志。

## 11. 性能测试规范

性能比较必须使用同步请求，固定 `1024×1024`、16 steps、seed 42、JPEG，并保证：

```text
TT_MM_THROTTLE_PERF=0
TT_METAL_TDP_LIMIT_WATTS=160
warmup 已完成
无并发排队
```

pipeline 日志会输出：

```text
encode
encode_reference
denoise
decode
to_host
```

HTTP E2E 还包含请求解析、排队、图像后处理、JPEG 编码和网络返回，所以不能直接把 `denoise + decode` 当作 HTTP 总耗时。
