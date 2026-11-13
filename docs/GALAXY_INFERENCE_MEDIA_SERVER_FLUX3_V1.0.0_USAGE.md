# Galaxy Inference Media Server — FLUX.3 Image Klein TL V1.0.0

## 1. 正式镜像

Docker repository 名必须使用小写，因此正式名称为：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0
```

构建来源：

| 项目 | 版本 |
| --- | --- |
| TL-inference-server | `f98b3f6c`（API/并行代码）；后续文档提交 `b0daf6c1` |
| tt-metal | `6ba31472d0`，标签 `flux3-tt-metal-v2026.11.13` |
| 默认模型 | `/local/models/flux-3-image-klein` 的 Distilled checkpoint |

模型权重不会被复制到镜像。运行时必须只读挂载 `/local/models`。

## 2. 在 Galaxy03 构建

```bash
docker buildx build --load --progress=plain \
  --build-context tt-metal-local=/local/lnn/tt-metal \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=6ba31472d0 \
  -f /local/lnn/TL-inference-server.Dockerfile \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0 \
  /local/lnn/TL-inference-server
```

本次构建日志：

```text
/local/lnn/fluxcache/logs/galaxy-inference-media-server-flux3-v1.0.0-build.log
```

构建完成后核对：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0 \
  --format '{{.Id}} {{.Size}}'
```

## 3. 同步到 Galaxy02

在 Galaxy03 上直接流式导出、传输和载入，不在中间落一个几十 GB 的未压缩 tar：

```bash
docker save galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0 \
  | zstd -T0 -3 \
  | ssh tt@10.12.11.2 'zstd -d | docker load'
```

Galaxy02 上确认：

```bash
docker image inspect galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0 \
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
  -e FLUX3_TP_SIZE=4 \
  -e FLUX3_SP4_WORKERS=8 \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e FLUX3_L1_SMALL_SIZE=131072 \
  -e OPENAI_API_KEY='replace-with-a-long-random-secret' \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.0
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

输出尺寸规则：宽高必须同时提供；每边 `256–4096`、均为 16 的倍数、最大 `16,777,216` 像素、最大宽高比 `8:1`。合法矩形示例包括 `1024×2048` 与 `256×2048`。

## 6. 鉴权规则

服务按下列优先级读取 bearer token：

1. `OPENAI_API_KEY`；
2. `API_KEY`（旧部署兼容）；
3. 两者均未设置时才使用不安全的默认值 `your-secret-key`。

不要在 shell 历史、文档、镜像层或 Git 中保存真实 token。更新 token 只需以新的 `-e OPENAI_API_KEY=...` 重建容器，不需要重建镜像。
