# Galaxy Inference Media Server — FLUX.3 Image Klein TL V1.0.1

## 1. 正式镜像

Docker repository 名必须使用小写，因此正式名称为：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1
```

构建来源：

| 项目 | 版本 |
| --- | --- |
| TL-inference-server | `d85e5f43`（SP4 12×10、鉴权与官方尺寸契约） |
| tt-metal | 已验证的 SP4 safe-path snapshot（派生基础镜像中） |
| 默认模型 | `/local/models/flux-3-image-klein` 的 Distilled checkpoint |

模型权重不会被复制到镜像。运行时必须只读挂载 `/local/models`。

## 2. 在 Galaxy03 构建

```bash
docker build --progress=plain \
  -f /local/lnn/flux3-sp4-12x10-image-context/Dockerfile \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1 \
  /local/lnn/flux3-sp4-12x10-image-context
```

本次构建日志：

```text
/local/lnn/fluxcache/logs/galaxy-inference-media-server-flux3-v1.0.1-build.log
```

构建完成后核对：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1 \
  --format '{{.Id}} {{.Size}}'
```

## 3. 同步到 Galaxy02

在 Galaxy03 上直接流式导出、传输和载入，不在中间落一个几十 GB 的未压缩 tar：

```bash
docker save galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1 \
  | zstd -T0 -3 \
  | ssh tt@10.12.11.2 'zstd -d | docker load'
```

Galaxy02 上确认：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1 \
  --format '{{.Id}} {{.Size}}'
```

## 4. Galaxy02：SP4×8（8 个四卡 worker）启动

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
  -e FLUX3_PARALLELISM=sp4 \
  -e FLUX3_TP_SIZE=1 \
  -e FLUX3_CORE_GRID_OVERRIDE=12x10 \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY='replace-with-a-long-random-secret' \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.1
```

查看 warmup。只有日志出现 `All workers ready` 后，才开始性能或业务请求：

```bash
docker logs -f flux3-image-klein-sp4x8-v1
curl -fsS http://127.0.0.1:8000/health
```

## 5. 生成图片请求

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

## 6. 鉴权规则

服务只读取一个 bearer token：`OPENAI_API_KEY`。镜像不内置默认密钥；未设置时，受保护 API 返回 `503`。已配置服务缺少或携带错误 token 时返回 `401`。

不要在 shell 历史、文档、镜像层或 Git 中保存真实 token。更新 token 只需以新的 `-e OPENAI_API_KEY=...` 重建容器，不需要重建镜像。
