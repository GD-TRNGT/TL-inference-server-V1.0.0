# Galaxy Inference Media Server — FLUX.3 Image Klein TL v1.0.16 使用手册

## 镜像与构建

正式镜像：

```text
galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16
```

Galaxy03 于 2026-11-15 从当前两份本地源码完整重建并验证的镜像：

```text
sha256:d08eed45d56c09dbb1a4a61afa0a68bbcfe6c210612061f1a6020052c1acd8b9
```

镜像可直接由当前两个仓库构建；模型权重不会进入镜像。

```bash
cd /local/lnn/TL-inference-server
docker build --pull --no-cache --progress=plain \
  --build-context ttmetal=/local/lnn/tt-metal \
  -f Dockerfile.flux3-sp8-4096 \
  -t galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16 .
```

构建前应核对 tag：

```bash
git -C /local/lnn/TL-inference-server describe --tags --always
git -C /local/lnn/tt-metal describe --tags --always
```

## Galaxy worker 拓扑

不要设置 `FLUX3_DEVICE_IDS`、`FLUX3_PARALLELISM` 或 `FLUX3_TP_SIZE`。只设置一个 `FLUX3_EXECUTION_PROFILE`，服务会按 Fabric 拓扑自动发现设备组。

| profile | 每 worker 芯片数 | Galaxy worker 数 | 并行布局 |
| --- | ---: | ---: | --- |
| `legacy-single` | 1 | 32 | 独立单卡 |
| `sp2` | 2 | 16 | SP2 |
| `sp4` | 4 | 8 | SP4 Ring |
| `sp8` | 8 | 4 | TP1×SP8 Ring |
| `sp8tp4` | 32 | 1 | TP4×SP8 Ring |

所有 profile 使用 12×10 core grid。SP8 的四个 worker 以固定约 10 秒间隔启动，不会等待前一个 worker ready。

## 在 Galaxy 上启动服务

以下示例启动 8 个四卡 SP4 worker，并监听 8000。将 token 替换为自定义强密钥；服务只识别 `OPENAI_API_KEY`，不识别旧的 `API_KEY`。

```bash
export FLUX3_API_KEY='replace-with-a-long-random-secret'

docker rm -f flux3-image-klein 2>/dev/null || true
docker run -d --name flux3-image-klein \
  --network host --ipc host --privileged --security-opt label=disable \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /local/models:/local/models:ro \
  -v /local/ttcache/flux-3-image-klein:/local/ttcache \
  -v /local/ttcache/flux-3-image-klein/tt-metal-cache:/home/container_app_user/.cache/tt-metal-cache \
  -e MODEL_SERVICE=image \
  -e MODEL_RUNNER=tt-flux.3-image-klein \
  -e DEVICE=bh-galaxy \
  -e SERVICE_PORT=8000 \
  -e FLUX3_MODEL_ROOT=/local/models/flux-3-image-klein \
  -e FLUX3_MODEL_VARIANT=distilled \
  -e FLUX3_EXECUTION_PROFILE=sp4 \
  -e FLUX3_TT_METAL_CACHE_ROOT=/local/ttcache/flux3-worker-cache \
  -e FLUX3_WARMUP_SIZES=1024 \
  -e TT_MM_THROTTLE_PERF=0 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e OPENAI_API_KEY="${FLUX3_API_KEY}" \
  galaxy-inference-media-server:flux-3-image-klein-tl-v1.0.16
```

切换 profile 时仅替换 `FLUX3_EXECUTION_PROFILE` 值。多卡服务启动前先停止旧容器并执行 `tt-smi -glx_reset_auto`。只有日志出现 `All workers ready (model readiness gated by per-runner warmup)` 后，才表示全部 worker 已可用。

## 当前重建镜像的 SP4 验证

2026-11-15 在 Galaxy03 对上述 `sha256:d08eed45...` 镜像进行了真实服务验证：Distilled、1024×1024、16 steps、无参考图、JPEG、8 个 SP4 worker。服务完成 warmup 后先发送一笔不计时 prime，再同时发送 8 笔请求（每 worker 一笔）。结果为 8/8 JPEG 有效，平均 HTTP `3.342s`（`3.308–3.383s`）、平均去噪 `2.833s`、平均 VAE decode `0.342s`。原始请求、HTTP 时间和设备日志位于：

```text
/local/lnn/fluxcache/v1016-rebuild-sp4/benchmark/
```

## API

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

尺寸契约：宽高各为 16–4096、且为 16 的倍数。支持 `number_of_images: 1–4`，以及 `reference_images` 中 1–10 张 Base64/data-URL 参考图。详细字段参见 `FLUX3_IMAGE_KLEIN_API_MANUAL.md`。
