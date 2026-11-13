# FLUX.3 Image Klein API 手册（Galaxy02 SP4×8 服务）

## 1. 服务信息

| 项目 | 当前部署值 |
| --- | --- |
| 主机 | `10.12.11.2`（Galaxy02） |
| 服务根地址 | `http://10.12.11.2:8000` |
| 容器 | `flux3-image-klein-sp4x8` |
| 镜像 | `tt-media-inference-server-h3:flux3-unified-verify-20261113` |
| 执行方式 | 8 个 SP4 worker；每个 worker 使用 4 个芯片，共 32 芯片 |
| 模型目录 | `/local/models/flux-3-image-klein` |
| 缓存根目录 | `/local/ttcache/flux-3-image-klein` |
| 默认模型变体 | Distilled |

容器采用 host network，故 Docker 不显示 `-p` 端口映射；客户端直接访问主机的 `8000` 端口。

## 2. 鉴权

图像生成接口要求 Bearer token：

```http
Authorization: Bearer <API_KEY>
```

生产环境必须在容器启动时设置唯一的自定义变量 `OPENAI_API_KEY`，并在客户端使用相同值。镜像不再内置 `your-secret-key` 这类公开默认密钥；未设置 `OPENAI_API_KEY` 时，受保护接口会明确返回 `503`，不会在不安全的已知密钥下继续服务。修改变量后重启容器即可；不要把真实密钥写入脚本或文档。

```bash
docker run ... -e OPENAI_API_KEY='replace-with-a-long-random-secret' ...
```

`/health`、`/tt-liveness`、`/v1/models` 等查询接口可不带 token；图像生成接口必须带 token。

## 3. 常用接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 服务与至少一个可用 worker 的健康状态 |
| `GET` | `/tt-liveness` | TT worker 存活状态 |
| `GET` | `/v1/models` | 查询当前模型及请求 schema |
| `POST` | `/v1/images/generations` | 同步生成图像 |
| `POST` | `/v1/images/generations/async` | 创建异步生成任务 |
| `GET` | `/v1/images/generations/{job_id}` | 查询异步任务 |
| `POST` | `/v1/images/generations/{job_id}/cancel` | 取消异步任务 |
| `DELETE` | `/v1/images/generations/{job_id}` | 删除已完成/已取消的任务记录 |

兼容别名为 `/image/generations` 及对应的 `/async`、`/{job_id}`、`/{job_id}/cancel` 路径。建议客户端统一使用 OpenAI 风格的 `/v1/images/generations`。

## 4. 同步生成：直接保存 JPEG/PNG

`response_format=file` 时，HTTP 响应 body 就是图片二进制，而不是 JSON。必须使用 `--output` 保存响应。

```bash
curl --fail-with-body --show-error \
  -X POST "http://10.12.11.2:8000/v1/images/generations" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
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

file black_bird.jpg
```

`--fail-with-body` 很重要：当服务返回 `401` 或 `400` 时，curl 会报错，而不是把错误 JSON 静默写成一个伪 `.jpg` 文件。

验证成功时，`file black_bird.jpg` 应显示 `JPEG image data` 或 `PNG image data`，而不是 `JSON text data`。

## 5. 同步生成：JSON/Base64 返回

将 `response_format` 改成 `b64_json`，服务返回 OpenAI 风格 JSON；图片在 `data[0].b64_json` 中。

```bash
curl --fail-with-body --show-error \
  -X POST "http://10.12.11.2:8000/v1/images/generations" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42,
    "response_format": "b64_json",
    "image_return_format": "PNG"
  }' > response.json
```

提取 PNG：

```bash
jq -r '.data[0].b64_json' response.json | base64 -d > black_bird.png
```

## 6. 请求字段

| 字段 | 类型/允许值 | 说明 |
| --- | --- | --- |
| `model` | string | 使用 `tt-flux.3-image-klein` |
| `prompt` | string，必填 | 文本提示词 |
| `negative_prompt` | string 或 `null` | 可选负面提示词 |
| `width` | 整数，`16–4096`，且为 16 的倍数 | 输出宽度；必须与 `height` 一起提供 |
| `height` | 整数，`16–4096`，且为 16 的倍数 | 输出高度；必须与 `width` 一起提供 |
| `num_inference_steps` | 整数 `4–50` | 去噪步数。Distilled 常用 16 步 |
| `seed` | 整数或 `null` | 固定时可复现；省略或 `null` 时随机 |
| `response_format` | `file` / `b64_json` | `file` 返回图片二进制；`b64_json` 返回 JSON |
| `image_return_format` | `JPEG` / `PNG` | 返回图片格式，默认 JPEG |
| `image_quality` | 整数 `50–100` | JPEG 质量，默认 85 |
| `number_of_images` | 整数 `1–4` | 一次请求生成 1–4 张图片；每张图会独立排队执行 |
| `reference_image` | Base64/data URL 或 `null` | 单张参考图快捷字段 |
| `reference_images` | Base64/data URL 数组，`1–10` 张 | 多参考图字段；顺序会保留；不支持 HTTP URL |

FLUX.3 支持方形和矩形输出。官方参考实现的输出约束为：每边 `16–4096`，且宽高均为 VAE 下采样比例 `16` 的倍数；模型卡声明最大分辨率为单边 4K。实现不额外施加总像素数或长宽比上限。因此 `16×4096`、`1024×2048`、`4096×4096` 都通过 API 几何校验；实际部署仍应优先使用经 warmup/性能验证的常用尺寸。

## 7. 带参考图的请求

参考图片必须以 data URL 或 Base64 字符串传入。下面示例将本地 `reference.jpg` 编码为 data URL 后发起请求：

```bash
ref="data:image/jpeg;base64,$(base64 -w 0 reference.jpg)"

jq -n --arg ref "$ref" '{
  model: "tt-flux.3-image-klein",
  prompt: "turn the reference bird into a black bird flying above a lake",
  width: 1024,
  height: 1024,
  num_inference_steps: 16,
  seed: 42,
  reference_images: [$ref],
  response_format: "file",
  image_return_format: "JPEG"
}' > request.json

curl --fail-with-body --show-error \
  -X POST "http://10.12.11.2:8000/v1/images/generations" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  --data-binary @request.json \
  --output reference_result.jpg
```

传入多张参考图时，将 `reference_images: [$ref]` 改为按期望顺序排列的 1–10 个 data URL 数组。

## 8. 异步任务

创建任务：

```bash
curl --fail-with-body --show-error \
  -X POST "http://10.12.11.2:8000/v1/images/generations/async" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tt-flux.3-image-klein",
    "prompt": "a black bird",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 16,
    "seed": 42
  }'
```

从响应中取得 `job_id` 后查询：

```bash
curl --fail-with-body --show-error \
  -H "Authorization: Bearer your-secret-key" \
  "http://10.12.11.2:8000/v1/images/generations/<job_id>"
```

异步任务完成后统一返回 JSON/Base64 结果，不会返回直接图片二进制。取消任务：

```bash
curl --fail-with-body --show-error \
  -X POST \
  -H "Authorization: Bearer your-secret-key" \
  "http://10.12.11.2:8000/v1/images/generations/<job_id>/cancel"
```

## 9. Python 示例

```python
import requests

response = requests.post(
    "http://10.12.11.2:8000/v1/images/generations",
    headers={
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    },
    json={
        "model": "tt-flux.3-image-klein",
        "prompt": "a black bird",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 16,
        "seed": 42,
        "response_format": "file",
        "image_return_format": "JPEG",
    },
    timeout=180,
)
response.raise_for_status()
open("black_bird.jpg", "wb").write(response.content)
```

## 10. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `401 Unauthorized` | 缺少或错误的 `Authorization: Bearer ...`。确认客户端 token 与启动时的 `OPENAI_API_KEY` 一致。 |
| `503 Service Unavailable`（API key） | 服务启动时未设置 `OPENAI_API_KEY`；设置一个自定义随机密钥后重启容器。 |
| 图片文件只有几十字节 | 常见原因是把 401/400 错误 JSON 写进了 `.jpg`。使用 `--fail-with-body` 并检查 HTTP 状态。 |
| `400` 尺寸错误 | `width`/`height` 必须同时提供，范围为 16–4096，均为 16 的倍数。 |
| `/v1/videos/jobs` 返回 404 | 这是视频任务探针访问了图像服务未实现的路由；不影响 FLUX.3 图像 API。 |
| `/health` 返回非 200 | worker 仍在 warmup 或已有 worker 失败；查看容器日志。 |

查看部署日志：

```bash
ssh tt@10.12.11.2 'docker logs -f flux3-image-klein-sp4x8'
```
