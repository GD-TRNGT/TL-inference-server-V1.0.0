# FLUX.3 大尺寸多参考图与 32-worker 压测

日期：2026-11-05。目标机器：10.12.11.2 / Blackhole Galaxy。当前状态：两套模型的大尺寸多图矩阵、32-worker 冷暖覆盖和 200 请求负载均已通过；最终验收 `all_passed=true`，原 8000 服务已恢复。

## 1. 范围与固定口径

- Distilled：`/local/models/flux-3-image-klein`，16 步。
- Base：`/local/models/flux-3-image-klein-base`，50 步、CFG 3.5。
- 两套测试均显式使用 `FLUX3_REFERENCE_ENCODER_BACKEND=ttnn`，每个 worker 占一个芯片。
- 功耗参数为 `TT_MM_THROTTLE_PERF=0`、`TT_METAL_TDP_LIMIT_WATTS=160`。
- 测试容器显式设置 `FLUX3_L1_SMALL_SIZE=131072`；`get_settings()` 已核实临时服务的实际请求超时为 2400 秒，原服务的请求超时不改。保留下采样输出宽度切片上限修复，encoder 大面积 RGB 卷积采用显式高度切片；通用大面积宽切片试验已因数值偏差撤回。
- 模型和两仓库源码继续使用原有 bind mount，不重建镜像、不删除缓存。
- 所有响应除检查 HTTP 200 外，还实际解码，校验图片尺寸/格式并保存图片及 SHA256。
- 原服务空闲后临时停止，测试容器使用 8001；控制脚本在 finally 中恢复原 8000 的 32-worker 服务。

参考输入由模型自带 figurine、coffee、balloons 三张资产制作成不同裁剪/镜像版本；每个多图请求的图片字节哈希互不相同，但不声称是十个不同的原始主体。

## 2. 单芯片大尺寸矩阵

每套模型按由小到大的顺序执行以下 14 项，先确定容量与尺寸切换能否正常运行，再执行全机负载：

| 项目 | 参考数量 | 每张参考图宽×高 | 输出宽×高 | 条件 token 数 |
|---|---:|---|---|---:|
| 数量补测 | 3～9，逐项执行 | 256×256 | 256×256 | 768～2304 |
| 512 两图 | 2 | 512×512 | 512×512 | 2048 |
| 512 四图 | 4 | 512×512 | 512×512 | 4096 |
| 512 十图 | 10 | 512×512 | 512×512 | 10240 |
| 1024 两图 | 2 | 1024×1024 | 1024×1024 | 8192 |
| 1024 四图 | 4 | 1024×1024 | 1024×1024 | 16384 |
| 1024 十图 | 10 | 1024×1024 | 1024×1024 | 40960 |
| 混合长宽 | 3 | 1024×768、768×1024、512×1024 | 1024×768 | 8192 |

这里列的是参考条件 token，不包含输出图像和文本 token。多图每张最多 1024×1024 像素，不保证任意长宽组合都能运行。单次串行时间混合新形状编译、encoder 首次加载和热态执行，不作为纯热态均值。

## 3. 32-worker 与 200 请求

固定负载：两张不同的 512×512 参考图，输出 1024×1024，JPEG 文件响应，每请求 seed=10000+序号。

每套模型分四步执行：

1. 单芯片跑一次该负载，提前编译完全相同几何的 kernel，保留时间。
2. 启动 32 个单芯片 worker，同时提交 32 个请求，记录 reference encoder 首次懒加载影响。
3. 再提交 32 个请求，确认预热后全 worker 覆盖。
4. 同时提交 200 个请求，记录墙钟时间、吞吐、平均/P50/P95/P99、队列峰值和成功率。

监控每 0.5 秒采样 liveness 与主机 MemAvailable；记录 ready worker、error/restart 计数。主机内存采样不等于芯片 DRAM 峰值。本轮不声称完成每算子显存峰值或 TT 硬件执行时间剖析。

通过测试时间窗口内的 `Worker N processing tasks` 日志核对实际 worker 参与和各 worker 请求数；只有启动 ready 不视为真实参与。服务日志、逐请求客户端起止时间一并保存，用于核对任务重叠。

## 4. 脚本及输出

测试脚本：

```text
/local/lnn/fluxcache/flux3_reference_load_test.py
/local/lnn/fluxcache/flux3_large_reference_control.py
```

启动命令（会暂时停止原服务）：

```bash
sudo python3 /local/lnn/fluxcache/flux3_large_reference_control.py
```

输出根目录在启动时打印为 `RESULT_ROOT`，位于 `/local/lnn/fluxcache/large_reference_load_20261105/时间戳/`。保存配置、动态 state.json、图片、matrix/results.json、逐请求 JSONL、liveness、summary.json、worker-participation.json 和服务日志。

当前执行根目录为 `/local/lnn/fluxcache/large_reference_load_20261105/20261105-075518/`；之前失败轮次（含 `20261105-073240`）另存原时间戳，不覆盖失败日志。

控制脚本遇到非预期执行错误会停止后续重负载并恢复原服务。容量不足、超时或硬件错误必须按实际现象记录，不将失败场景标记为已通过，也不会自动 reset 整机。

## 5. 当前结果

Distilled 和 Base 都完成 14/14 大图矩阵、32/32 首次参考图覆盖、32/32 暖态覆盖和 200/200 同时提交请求。最终汇总的 `distilled=true`、`base=true`、`original_restored=true`、`all_passed=true`。此前的 64×64 小参考图结果不能替代本轮测试。

### 5.1 Distilled 已完成的负载

固定负载是两张 512×512 参考图 → 1024×1024 JPEG，16 步。下表保留客户端完整测量口径：延迟含排队、网络和图片校验/落盘，不能解释成纯设备推理时间。

| 阶段 | 成功 | 总墙钟秒 | 吞吐张/秒 | 平均响应秒 |
|---|---:|---:|---:|---:|
| 单芯片预编译 | 1/1 | 22.111 | 0.0452 | 22.110 |
| 32-worker 首次参考图覆盖 | 32/32 | 19.259 | 1.6616 | 18.866 |
| 32-worker 暖态覆盖 | 32/32 | 17.682 | 1.8097 | 17.438 |
| 200 请求同时提交 | 200/200 | 119.631 | 1.6718 | 62.015 |

200 请求 P50/P95/P99 分别为 67.711/102.270/118.147 秒，队列峰值 168，始终 32 ready；worker 错误/重启均 0。日志中 0～31 号 worker 每个处理 6～7 个请求，共 200 个启动事件。监控到的最小主机 MemAvailable 为 508.135 GiB，不代表芯片 DRAM 余量。首次覆盖是 reference encoder 懒加载/运行时冷态，磁盘 kernel 已由单芯片预编译，不能称为完全冷编译。

单芯片十张 1024×1024 参考图 → 1024×1024 输出通过，条件 token 为 40960，单次完整响应 315.165 秒。混合横竖三图 → 1024×768 输出通过，60.938 秒；此前 13 项成功场景与失败前旧轮次的输出 SHA256 全部相同。这里只验证所列容量和非空图片，不代替完整编辑语义验收。

### 5.2 Base 已完成的负载

固定负载与 Distilled 相同，但 Base 使用 50 步 CFG 3.5。

| 阶段 | 成功 | 总墙钟秒 | 吞吐张/秒 | 平均响应秒 |
|---|---:|---:|---:|---:|
| 单芯片预编译 | 1/1 | 97.593 | 0.0102 | 97.593 |
| 32-worker 首次参考图覆盖 | 32/32 | 99.495 | 0.3216 | 99.077 |
| 32-worker 暖态覆盖 | 32/32 | 97.999 | 0.3265 | 97.684 |
| 200 请求同时提交 | 200/200 | 681.294 | 0.2936 | 354.129 |

200 请求 P50/P95/P99 分别为 388.464/585.335/679.722 秒，队列峰值 168，始终 32 ready；worker 错误/重启均 0。日志显示 worker 0～31 各处理 6 或 7 个请求，共 200 个任务启动事件。最小主机 MemAvailable 为 507.940 GiB。

Base 单芯片十张 1024×1024 参考图 → 1024×1024 输出通过，条件 token 为 40960，完整响应 1922.377 秒；混合横竖三图 → 1024×768 通过，211.933 秒。十图场景已接近本轮配置的 2400 秒请求上限，但未超时。

### 5.3 逐项单芯片完整响应时间

以下每项均为单次请求，不是后三遍热态均值。Base 用 50 步及 CFG，Distilled 用 16 步 flow-map，不能把两列当作相同模型计算量的后端比较。

| 场景 | Distilled 秒 | Base 秒 | 状态 |
|---|---:|---:|---|
| 3 张 256 → 256 | 4.340 | 19.196 | 两模型通过 |
| 4 张 256 → 256 | 4.066 | 22.815 | 两模型通过 |
| 5 张 256 → 256 | 4.310 | 24.373 | 两模型通过 |
| 6 张 256 → 256 | 4.526 | 25.696 | 两模型通过 |
| 7 张 256 → 256 | 5.557 | 31.943 | 两模型通过 |
| 8 张 256 → 256 | 5.805 | 33.290 | 两模型通过 |
| 9 张 256 → 256 | 6.574 | 37.825 | 两模型通过 |
| 2 张 512 → 512 | 8.416 | 45.715 | 两模型通过 |
| 4 张 512 → 512 | 13.593 | 78.390 | 两模型通过 |
| 10 张 512 → 512 | 36.069 | 213.788 | 两模型通过 |
| 2 张 1024 → 1024 | 41.677 | 240.362 | 两模型通过 |
| 4 张 1024 → 1024 | 86.125 | 512.474 | 两模型通过 |
| 10 张 1024 → 1024 | 315.165 | 1922.377 | 两模型通过 |
| 横竖混合三图 → 1024×768 | 60.938 | 211.933 | 两模型通过 |

### 5.4 最终验收方法和实际结果

控制脚本完成并恢复原服务后，使用下列命令汇总。2026-11-05 本轮已正常退出，`acceptance.all_passed=true`。

```bash
sudo docker exec -w /app flux3-media-server-test \
  /home/container_app_user/tt-metal/python_env/bin/python \
  /local/lnn/fluxcache/flux3_reference_benchmark_report.py \
  /local/lnn/fluxcache/large_reference_load_20261105/20261105-075518 \
  --require-complete
```

汇总要求两模型各 14 项矩阵、1 项预编译、32 首次/32 暖态/200 请求全部成功；三组全机负载每组 32 worker 实际参与、任务启动事件与请求数匹配、在首个客户端完成前全部启动，采样错误为 0、worker error/restart 为 0。冷态第一组请求可能在 background canary 完成初次转换前采到 `starting`；因此以每组完成后健康探针及 200 请求完成后的 healthy canary 作为验收。每模型保存 279 张可解码非全平图片，两模型合计 558 张。原服务已恢复为 32/32 alive/ready、queue=0、canary healthy。

统计脚本使用 root 写本轮 root-owned 报告目录，不在 root 下重新编译 kernel。产物为 `report.json` 和两模型 `*-image-statistics.json`。RGB 标准差检查只能排除全平图片，不能证明参考编辑语义或 Golden Data 精度通过。最终产物根目录为 `/local/lnn/fluxcache/large_reference_load_20261105/20261105-075518/`。

2026-11-05 07:16 UTC 检查：128 KiB 配置下原服务恢复 32/32 ready、队列 0、全部 worker error/restart 为 0；真实配置/API 测试 26/26、切片测试 12/12 通过。这只证明启动和单元测试，不代表大尺寸数值或压测通过。

2026-11-05 07:28 UTC：25 个 1024×1024 卷积的同输入逐层比较通过；两模型各 8 次整网 encoder 数值与尺寸切换检查全部通过，缓存重载通过。最终数值文件为 `/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-072719/results.json`。1024×1024 PCC 0.999818265、MAE 0.01389346、relative RMSE 0.01916481。该结果仍不代表大图多参考 DiT/并发请求已经通过。

## 6. 问题记录

首轮 `/local/lnn/fluxcache/large_reference_load_20261105/20261105-064755/` 在客户端写 `config.json` 时出现 PermissionError，尚未向模型发送矩阵请求。控制脚本用 root 建目录，而客户端按服务 UID 1000 运行，目录 owner 不匹配。修复位置为 `flux3_large_reference_control.py:run()`：新建当前测试输出目录后将该目录 chown 为 1000:1000，不递归修改模型或公共缓存。首轮日志保留，脚本 finally 恢复原服务，修正后重新执行。

第二轮 `20261105-065124` 的 Distilled 3～9 张 256×256 输入均通过；随后两张 512×512 参考图出现 HTTP 500：`Number of slices (4) exceeds the maximum allowed (2)`。定位到 reference encoder 的 stride-2 downsampler 复用了面向 decoder stride-1 的 Conv2d 切片表：输入宽 128 时输出宽为 64，而 tiled DRAM_WIDTH 的最大片数为 ceil(64/32)=2。修复在 `/local/lnn/tt-metal/models/tt_dit/models/vae/flux3_vae_encoder.py` 的 `_ReferenceConv2d.__init__()`：为下采样卷积创建独立切片表，依据真实输出宽度限制片数，不修改公共 decoder 表；新增 `_reference_width_slice_limit()` 与 8 项算术回归测试。大图及既有小图的 TTNN/CPU 数值复测中，尚未提前标记修复验收通过。

扩大 encoder 数值测试后，512×512 的 PCC 为 0.99983567、relative RMSE 为 0.01794943，切片修复通过该尺寸。随后 1024×1024 在 `(1,512,512,256)` stride-1 卷积分配 L1_SMALL 时失败：每 bank 已用 65088 B，64 KiB 区域只剩 448 B，下一操作需要 1648 B。该错误是小型 L1 配置/halo 缓冲区容量不足，不是模型权重 DRAM 不够。修复在 `TL-inference-server/tt-media-server/tt_model_runners/dit_runners.py:TTFlux3ImageKleinRunner.get_pipeline_device_params()`：加入 `FLUX3_L1_SMALL_SIZE`，默认从 65536 提升至 262144 B；允许 64～512 KiB、32 B 对齐。独立数值脚本使用相同默认预留。新增 9 项默认值、覆盖值及非法值测试；该修改需要重新打开设备/启动 worker，不能在已运行请求中改变预留。后续需重新验证大图矩阵和全机负载，不以修改参数本身当作已通过。

256 KiB 预留复测又遇到非方形 `(1,384,512,128)` 下采样卷积 halo 的静态 CB/L1 缓冲区重叠，边界为 145664/145856，仅差 192 B。仅增加预留不足以解决计算缓冲区的布局问题。进一步修改 `_ReferenceConv2d.forward()`：encoder 输入面积达到 256×256 且无专用配置时，使用不超过 4 个、受输出宽度上限约束的 DRAM width strips，避免自动规划走贴边的全图 halo；配置表始终按实例复制，不改公共 decoder。新增 4 项动态切片算术测试；后续数值回归发现该通用策略失真，已撤回。

设备配置测试首次被 server 根 conftest 的 runner 模块 mock 拦截，并非服务缺少 runner。改用 `pytest --noconftest` 运行这组真实模块的配置测试和独立请求模型测试，最终 26/26 通过；不修改公共 conftest。

256 KiB 方案随后也使原服务 DiT warmup 失败：L1 缓冲区起点 1307648，静态 CB 结束于 1419200，二者冲突。该试验参数已撤回，不是最终交付配置。`FLUX3_L1_SMALL_SIZE` 默认/测试值改为 131072 B（128 KiB），有效范围收紧到 64～128 KiB，兼顾小型元数据与 DiT 计算缓冲区；同时保留 encoder 大卷积的受限 width strips。需要重新验证服务启动及大尺寸数值，确认前不标记容量问题已解决。此故障无需据此认定硬件损坏，不自动 reset 整机。

128 KiB 与通用 width strips 的组合在数值复测 `20261105-071636` 出现不报执行错误但数值失真的情况：256×256 最终 token PCC 0.76437408、relative RMSE 0.65015280。16×16、64×64、128×96 仍正确。通用大面积切片策略未经数值验证，不能因为避开内存报错就保留；已从 `_ReferenceConv2d.forward()` 撤回该动态策略及相应 helper，保留实例独立配置表、stride-2 输出宽上限和几何变化重新准备权重。随后使用 128 KiB 与原有卷积规划重新逐级数值验证。以上旧方案均为失败历史，不是当前已验收配置。

撤回后 `20261105-071845` 的 256×256、512×384、512×512 恢复正确，但 1024×1024 最终 token PCC 仍只有 0.73616374、relative RMSE 0.67989182。新增 `/local/lnn/fluxcache/flux3_reference_layer_diagnostic.py`：对每个卷积读取实际 TT 输入，使用该层原始 BF16 权重转换成 Torch float32 运行 `F.conv2d`，与该层 TT 输出比较。首个失败即第一层 `[1,1024,1024,3] -> 128` RGB 卷积，PCC 0.64511538、relative RMSE 0.77913582，说明偏差已在输入卷积出现，不应归因于 prompt、随机 seed 或最终 VAE decode。

进一步在 `_ReferenceConv2d.forward()` 仅为大面积 RGB 输入改用显式 DRAM_HEIGHT 切片：从原始 OIHW host 权重首次执行 `ttnn.conv2d()`，让权重准备和执行使用同一切片配置；返回的 prepared weight/bias 在同尺寸后续请求复用，尺寸改变仍释放并重新准备。片数最多 4，并受输出高度 tiled 上限约束。其他卷积保留原执行路径，公共 decoder 不变。该路径首层逐层比较恢复 PCC 1.0、relative RMSE 0.00294678；整网数值、尺寸切换及服务负载仍需随后验证。该对照确定了有效规避路径，但没有证明 TTNN 自动切片内部的唯一根因。

端到端轮次 `20261105-073240` 的 Distilled 13/14 通过，含十张 1024 方图：324.326957 秒。最后混合长宽场景 `[1024×768,768×1024,512×1024] -> 1024×768` 失败，第二张 encoder 的 `(1,512,384,256)` 卷积需要每 bank 2464 B L1_SMALL，而已用 128832 B、剩余 2240 B。这一轮未进入 32-worker 或 200 请求阶段，控制脚本恢复原服务成功。不是主机内存耗尽，不能据此增加到已知会冲突的 256 KiB。

追加 `flux3_components.py:_refresh_reference_program_cache()`：参考几何变化时先 `ttnn.synchronize_device()`，再 `MeshDevice.clear_program_cache()`；同几何不清理。源码 `tt_metal/distributed/mesh_device.cpp:1326` 和运行时 Python API 均核实存在，清理保持缓存启用状态；不删除磁盘 kernel/权重转换缓存，不释放常驻模型权重。作用是避免旧形状运行时程序保有的 halo/元数据跨尺寸积累；代价是尺寸改变后重新创建运行时程序对象，新形状仍可能首次编译，固定形状热态不逐请求清理。新增 3 项同步/清理顺序和相同尺寸不清理测试。

追加复测：两模型各执行 11 次共 22/22，含 1024×768、768×1024、512×1024 及切回 64；PCC≥0.999798894、relative RMSE≤0.02044829，缓存重载成功。最新数值文件 `/local/lnn/fluxcache/ttnn_reference_validation_20261105/20261105-075049/results.json`，控制日志 `logs/multiref_validation_20261105/20261105-075021/`。完整端到端矩阵及 32/200 负载仍需重新验收。

配置复核脚本从 `/local/lnn/fluxcache` 执行时曾出现 `ModuleNotFoundError: config`：Python 的脚本目录不在 `/app`，仅指定工作目录不足以保证应用模块导入。改用 `docker exec -w /app -e PYTHONPATH=/app` 执行，检查成功；无需修改生产代码。2026-11-05 08:26 UTC 实查 Base 单芯片容器使用 TTNN backend、128 KiB、0/160，`get_settings()` 的实际请求超时为 2400 秒，而非模型注册表里的 3000 秒。

大 token 形状还出现 `No known best blocking ... using default 8x8x8` 警告。该日志表示未命中手工最优 matmul blocking，使用默认规划，不是 HTTP 请求失败；本轮保留日志，不把容量验收宣称为该形状的 kernel 性能已经优化。

首次最终汇总还暴露了两个测试控制问题，而不是模型执行失败。控制脚本最初只检查恢复后的原服务存在 32 个 worker 条目，未要求每个 worker ready 及 canary healthy；已将 `flux3_large_reference_control.py` 的 finally 恢复条件改为 32/32 alive/ready、queue=0、error/restart=0 和 healthy canary。报告脚本最初要求冷态阶段的所有采样均为 healthy，但首个请求可先于 canary 的初次转换完成，造成 Distilled 冷态出现 `starting`。改为要求每组完成后服务 ready，且最终 200 请求完成后的 canary healthy。修正后 `--require-complete` 正常退出，三个 acceptance 标志均为 true。

## 7. 本轮验收边界

- 大图矩阵验证的是单芯片串行的 2/4/10 张方图，以及所列混合长宽场景，不承诺任意长宽、任意 token 数或无限容量。
- 32-worker 负载固定为每请求两张 512×512 图、输出 1024×1024；没有让 32 个 worker 同时处理十张 1024 大图。
- 200 请求同时提交是排队负载，不是 200 个模型同时计算；实际计算池为 32 个单芯片 worker。
- 单芯片预编译之后再启动全机负载，验证了现有磁盘缓存下的 encoder 懒加载/运行时冷态和暖态，没有验收 32 个 worker 同时首次创建空转换缓存的锁和原子发布。
- liveness 与主机 MemAvailable 每 0.5 秒采样；没有测量芯片 DRAM/L1 峰值、瞬时硬件占用率或连续多日耐久性。worker 日志证明任务参与及起止重叠，不等于硬件 profiler。
- 图片均做真实解码和尺寸/格式验证，汇总再检查非全平；另有代表性输出抽查。上述检查不替代全模型 Golden Data 或所有参考主体一致性的语义验收。
- CPU/TTNN encoder 同输入热态性能对照、自动 OOM fallback、2/4 芯片单请求并行不包含在本轮；2/4 芯片和完整精度对齐按用户要求继续暂缓。
- 测试完成后保持原 8000 Distilled 服务及其默认 CPU reference backend，不在用户未要求的情况下改为 TTNN 默认。临时 TTNN 测试服务停止，日志、图片、转换和编译缓存保留。
