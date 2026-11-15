# 已废弃：请使用 GALAXY_INFERENCE_MEDIA_SERVER_FLUX3_V1.0.16_USAGE.md。

# Galaxy Inference Media Server — FLUX.3 Image Klein TL V1.0.3

## 1. 正式镜像

Docker repository 名必须使用小写，因此正式名称为：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.4-sp8
```

构建来源：

| 项目 | 版本 |
| --- | --- |
| TL-inference-server | `flux3-sp4-12x10-rebuild`（自动 worker 拓扑、12×10、鉴权、官方尺寸契约及 SP8 Ring 分组） |
| tt-metal | FLUX.3 Klein TP1×SPN safe projection path（SP4/SP8 避开 factor-one TP 轴的非法 fused MMRS） |
| 默认模型 | `/local/models/flux-3-image-klein` 的 Distilled checkpoint |

模型权重不会被复制到镜像。运行时必须只读挂载 `/local/models`。

## 2. 在 Galaxy03 构建

```bash
docker build --progress=plain \
  -f /local/lnn/flux3-sp4-12x10-image-context/Dockerfile \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8 \
  /local/lnn/flux3-sp4-12x10-image-context
```

本次构建日志：

```text
/local/lnn/fluxcache/logs/galaxy-inference-media-server-flux3-v1.0.3-build.log
```

构建完成后核对：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8 \
  --format '{{.Id}} {{.Size}}'
```

## 3. 同步到 Galaxy02

在 Galaxy03 上直接流式导出、传输和载入，不在中间落一个几十 GB 的未压缩 tar：

```bash
docker save galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8 \
  | zstd -T0 -3 \
  | ssh tt@10.12.11.2 'zstd -d | docker load'
```

Galaxy02 上确认：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8 \
  --format '{{.Id}} {{.Size}}'
```

## 4. Galaxy：自动 worker 拓扑

不要设置 `FLUX3_DEVICE_IDS`。从当前版本开始，FLUX.3 服务会拒绝该变量：Galaxy 的子网格必须由物理 Fabric 拓扑决定，不能用连续的 PCIe 编号手工拼接。服务只根据 `FLUX3_EXECUTION_PROFILE` 自动发现并拆分 Galaxy 的 32 张芯片；每个 worker 只打开其拓扑合法的芯片组。

SP2 与 SP4 都只接受 `test_system_health` 返回的物理 Fabric 分组；该工具不可用或失败时服务会拒绝启动相应的多卡 worker，而不会按 PCIe 编号猜测拓扑。
| `FLUX3_EXECUTION_PROFILE` | 每 worker 芯片数 | 自动创建的 worker 数 | 并行方式 |
| --- | ---: | ---: | --- |
| `legacy-single` | 1 | 32 | 独立单卡推理 |
| `sp2` | 2 | 16 | SP2 |
| `sp4` | 4 | 8 | SP4 Ring |
| `sp8` | 8 | 4 | TP1 × SP8 Ring |
| `sp8tp4` | 32 | 1 | SP8 × TP4 |

`legacy-single` 因而并不是“只启动一个单卡 worker”：默认会创建 **32 个单卡 worker**。如需只验证一张卡，应在隔离的单卡机器或测试环境中运行；不要通过手工设备列表改变 Galaxy 服务的 worker 拓扑。

所有 profile 均保持以下运行约束：`TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`、`FLUX3_CORE_GRID_OVERRIDE=12x10`。profile 自行确定 mesh 和 TP/SP 轴，因此无需再设置 `FLUX3_PARALLELISM` 或 `FLUX3_TP_SIZE`。

`sp8` 不是将每个 tray 的连续 8 张卡直接作为一个 worker。Galaxy 的有效 8 卡 Ring 是跨相邻两个 tray 的 column 子网格；服务依据 TT-Metal upstream 的 `create_submeshes(MeshShape(8,1))` 顺序自动发现四组 Ring：`(0,1,2,3,11,10,9,8)`、`(4,5,6,7,15,14,13,12)`、`(28,29,30,31,23,22,21,20)`、`(24,25,26,27,19,18,17,16)`。服务不接受手工 `FLUX3_DEVICE_IDS`，因此不能绕过该安全映射。

## 5. Galaxy02：SP4×8（8 个四卡 worker）启动

该配置使用 32 个芯片，8 个 worker，每 worker 4 卡。容器使用 host network，因此服务直接监听 Galaxy02 的 `8000` 端口。

```bash
docker rm -f flux3-image-klein-sp4x8-v1 2>/dev/null || true

docker run -d \
  --name flux3-image-klein-sp4x8-v1 \
  --network host \
  --ipc host \
  --privileged \
  --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/fluxcache \
  -v /local/ttcache/flux-3-image-klein/tt-metal-cache:/home/container_app_user/.cache/tt-metal-cache \
  -e MODEL_SERVICE=image \
  -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy \
  -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e FLUX3_TT_METAL_CACHE_ROOT=/local/fluxcache/flux3-tt-metal-workers \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY='replace-with-a-long-random-secret' \
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.3-sp8
```

如需切换拓扑，只需要替换容器名和下面这一行；不要添加 `FLUX3_DEVICE_IDS`：

```bash
# 32 个单卡 worker
-e FLUX3_EXECUTION_PROFILE=legacy-single

# 16 个双卡 SP2 worker
-e FLUX3_EXECUTION_PROFILE=sp2

# 8 个四卡 SP4 Ring worker（上例的默认配置）
-e FLUX3_EXECUTION_PROFILE=sp4

# 4 个八卡 TP1×SP8 Ring worker
-e FLUX3_EXECUTION_PROFILE=sp8

# 1 个 32 卡 SP8×TP4 worker
-e FLUX3_EXECUTION_PROFILE=sp8tp4
```

查看 warmup。`/health` 在首个 worker ready 后即可返回 200；只有日志出现 `All workers ready`，才表示表中全部 worker 已加载完成，可以开始性能或业务请求：

```bash
docker logs -f flux3-image-klein-sp4x8-v1
curl -fsS http://127.0.0.1:8000/health
```

## 6. 生成图片请求

```bash
export FLUX3_API_KEY='replace-with-a-long-random-secret'

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
  }' \
  --output black_bird.jpg
```

输出尺寸规则：宽高必须同时提供；每边 `16–4096`、均为 16 的倍数。该上限与官方模型卡的“up to 4K”一致；官方参考实现没有额外的总像素或宽高比校验。

`sp8` 是容量受限的部署 profile：每个 8 卡 worker 保持完整的 resident Qwen3-VL、DiT 和 VAE，因此只接受正方形 `1024×1024` 或 `2048×2048`。`4096×4096` 会在 API 输入校验阶段返回 HTTP 422；请改用 32 卡 `sp8tp4` profile。其它 profile 仍按上述官方 16–4096、16 对齐规则校验。

## 6.1 SP8 cold start

SP8 会自动形成 4 个 8×1 column Ring worker。启动前必须完成完整 Galaxy reset：停止旧容器后在后台运行 `tt-smi -glx_reset_auto`，确认日志包含 `All 32 chips found` 和 `Re-initialized 32 boards after reset`，再启动容器。默认每一个 Ring 在前一个 worker 报告真实 warmup ready 后再等待 120 秒才启动下一组；全量 cold start 约 8 分钟。`/health=200` 仅代表首个 worker ready，必须检查 `/tt-liveness` 中 4 个 worker 都是 `is_ready=true`，或日志出现 `All workers ready`。

## 7. 鉴权规则

服务只读取一个 bearer token：`OPENAI_API_KEY`。镜像不内置默认密钥；未设置时，受保护 API 返回 `503`。已配置服务缺少或携带错误 token 时返回 `401`。

不要在 shell 历史、文档、镜像层或 Git 中保存真实 token。更新 token 只需以新的 `-e OPENAI_API_KEY=...` 重建容器，不需要重建镜像。
