# Galaxy Inference Media Server — FLUX.3 Image Klein TL V1.0.2

## 1. 正式镜像

Docker repository 名必须使用小写，因此正式名称为：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild
```

构建来源：

| 项目 | 版本 |
| --- | --- |
| TL-inference-server | `flux3-sp4-12x10-rebuild`（自动 worker 拓扑、12×10、鉴权与官方尺寸契约） |
| tt-metal | 已验证的 SP4 safe-path snapshot（派生基础镜像中） |
| 默认模型 | `/local/models/flux-3-image-klein` 的 Distilled checkpoint |

模型权重不会被复制到镜像。运行时必须只读挂载 `/local/models`。

## 2. 在 Galaxy03 构建

```bash
docker build --progress=plain \
  -f /local/lnn/flux3-sp4-12x10-image-context/Dockerfile \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild \
  /local/lnn/flux3-sp4-12x10-image-context
```

本次构建日志：

```text
/local/lnn/fluxcache/logs/galaxy-inference-media-server-flux3-v1.0.2-build.log
```

构建完成后核对：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild \
  --format '{{.Id}} {{.Size}}'
```

## 3. 同步到 Galaxy02

在 Galaxy03 上直接流式导出、传输和载入，不在中间落一个几十 GB 的未压缩 tar：

```bash
docker save galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild \
  | zstd -T0 -3 \
  | ssh tt@10.12.11.2 'zstd -d | docker load'
```

Galaxy02 上确认：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild \
  --format '{{.Id}} {{.Size}}'
```

## 4. Galaxy：自动 worker 拓扑

正常部署**不要**设置 `FLUX3_DEVICE_IDS`。服务会基于 `FLUX3_EXECUTION_PROFILE` 自动发现并拆分 Galaxy 的 32 张芯片；每个 worker 只打开自己的芯片组。

| `FLUX3_EXECUTION_PROFILE` | 每 worker 芯片数 | 自动创建的 worker 数 | 并行方式 |
| --- | ---: | ---: | --- |
| `legacy-single` | 1 | 32 | 独立单卡推理 |
| `sp2` | 2 | 16 | SP2 |
| `sp4` | 4 | 8 | SP4 Ring |
| `sp8tp4` | 32 | 1 | SP8 × TP4 |

`legacy-single` 因而并不是“只启动一个单卡 worker”：默认会创建 **32 个单卡 worker**。如需只为调试启动 chip 0 的一个 worker，可额外设置 `FLUX3_DEVICE_IDS='(0)'`；这会关闭自动拆分，不可用于完整 Galaxy 吞吐部署。

所有 profile 均保持以下运行约束：`TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`、`FLUX3_CORE_GRID_OVERRIDE=12x10`。profile 自行确定 mesh 和 TP/SP 轴，因此无需再设置 `FLUX3_PARALLELISM` 或 `FLUX3_TP_SIZE`。

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
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.2-rebuild
```

如需切换拓扑，只需要替换容器名和下面这一行；不要添加 `FLUX3_DEVICE_IDS`：

```bash
# 32 个单卡 worker
-e FLUX3_EXECUTION_PROFILE=legacy-single

# 16 个双卡 SP2 worker
-e FLUX3_EXECUTION_PROFILE=sp2

# 8 个四卡 SP4 Ring worker（上例的默认配置）
-e FLUX3_EXECUTION_PROFILE=sp4

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

## 7. 鉴权规则

服务只读取一个 bearer token：`OPENAI_API_KEY`。镜像不内置默认密钥；未设置时，受保护 API 返回 `503`。已配置服务缺少或携带错误 token 时返回 `401`。

不要在 shell 历史、文档、镜像层或 Git 中保存真实 token。更新 token 只需以新的 `-e OPENAI_API_KEY=...` 重建容器，不需要重建镜像。
