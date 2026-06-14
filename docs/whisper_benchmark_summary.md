# Whisper-base 性能基准测试总结

## 1. 测试目的

对比 VideoBot 项目当前使用的 **faster-whisper (CTranslate2/int8)** 方案与 **OpenVINO (FP16)** 方案，在 CPU 环境下不同音频时长（1min ~ 2hour）的转录性能差异，为技术选型提供数据支撑。

## 2. 测试环境

| 项目 | 规格 |
|------|------|
| OS | Linux 5.4.0-216-generic |
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80GHz |
| Memory | 26.8 GB |
| 模型 | whisper-base |
| 测试音频 | ffmpeg 生成 440Hz 正弦波 MP3，16kHz mono |

## 3. 测试方案

### 3.1 方案 A：faster-whisper (CTranslate2)

- **推理后端**：CTranslate2（C++ 底层）
- **模型格式**：CTranslate2 int8 量化
- **模型路径**：`models/whisper-base/`（ModelScope 下载的 faster-whisper 格式）
- **模型内存占用**：106 MB
- **核心调用**：`WhisperModel.transcribe(audio)` 接收 numpy 数组，内部自动处理分段和 KV Cache

### 3.2 方案 B：OpenVINO (FP16)

- **推理后端**：OpenVINO Runtime + transformers generate()
- **模型格式**：OpenVINO IR (FP16)，来自 ModelScope `whisper-base-fp16-ov`
- **模型路径**：`models/OpenVINO/whisper-base-fp16-ov/`
- **模型内存占用**：59 MB
- **核心调用**：`OVModelForSpeechSeq2Seq.generate()` + `WhisperProcessor`，手动按 30 秒切分音频逐段转录

### 3.3 测试流程

```
1. 使用 ffmpeg 生成 7 种时长的测试音频（1/5/10/20/30/60/120 分钟）
2. 使用 librosa 将 MP3 加载为 16kHz mono float32 numpy 数组（内存缓冲区）
3. 对每个音频文件，分别用两种方案执行 X 次转录
4. 记录每次转录的 wall-clock 耗时
5. 计算平均耗时、实时倍率（处理时长/音频时长）
```

- faster-whisper：7 个时长 x 2 次迭代 = 14 次实测
- OpenVINO：1min x 2 次 + 5min x 1 次 = 3 次实测，其余时长基于 5min 实测结果线性外推

> OpenVINO 因解码器无 KV Cache 导致极慢（5min 音频需 ~78 分钟），长时长采用外推避免耗时过长。

## 4. 测试脚本

脚本位置：`benchmark_whisper.py`

```bash
# 常用命令
python benchmark_whisper.py --generate --iterations 2 --output benchmark_results.md
python benchmark_whisper.py --audio test.mp3 --iterations 3 --backend faster-whisper
python benchmark_whisper.py --audio test.mp3 --iterations 1 --backend openvino
```

参数说明：
- `--audio`：指定音频文件路径（不指定则自动生成 7 种时长测试音频）
- `--iterations`：每个测试的迭代次数（默认 2）
- `--backend`：`faster-whisper` / `openvino` / `both`（默认 both）
- `--generate`：自动生成不同时长的测试音频
- `--output`：报告输出路径（默认 benchmark_results.md）

## 5. 测试结果

### 5.1 性能对比

| 音频时长 | faster-whisper(s) | OpenVINO(s) | 实时倍率(FW) | 实时倍率(OV) | 数据来源 |
|---------|-----------------|------------|------------|------------|---------|
| 1min | 14.07 | 226.11 | 0.235x | 3.769x | 实测 |
| 5min | 32.88 | 4,705.24 | 0.110x | 15.684x | 实测 |
| 10min | 131.63 | 9,410.47 | 0.219x | 15.684x | FW实测/OV外推 |
| 20min | 313.72 | 18,820.95 | 0.261x | 15.684x | FW实测/OV外推 |
| 30min | 314.16 | 28,231.42 | 0.175x | 15.684x | FW实测/OV外推 |
| 1hour | 795.05 | 56,462.84 | 0.221x | 15.684x | FW实测/OV外推 |
| 2hour | 1,332.42 | 112,925.68 | 0.185x | 15.684x | FW实测/OV外推 |

> **实时倍率** = 处理耗时 / 音频时长，越小越快。0.2x 表示 1 分钟音频仅需 12 秒处理。

### 5.2 关键指标汇总

| 指标 | faster-whisper | OpenVINO |
|------|---------------|----------|
| 平均实时倍率 | **0.201x** | 9.726x |
| 模型加载内存 | 106 MB | 59 MB |
| 1min 音频处理时间 | 14s | 226s |
| 2hour 音频处理时间 | 22min | ~31h (外推) |
| 整体性能差距 | 基准 | **慢约 48x** |

## 6. 根因分析

### 6.1 性能差异的技术原因

| 维度 | faster-whisper | OpenVINO |
|------|---------------|----------|
| 推理引擎 | CTranslate2 (C++) | OpenVINO Runtime + transformers generate() |
| 量化精度 | int8（模型更小、计算更快） | FP16（模型更大、计算更慢） |
| **KV Cache** | **有 — 增量解码，O(n)** | **无 — 每步全量重算，O(n²)** |
| Python 开销 | 低（C++ 底层推理） | 高（transformers generate 循环） |
| 音频分段 | 内部自动处理 | 手动 30s 切分 |

### 6.2 核心瓶颈：KV Cache 缺失

Whisper 的解码器是自回归模型，逐 token 生成文本：

- **有 KV Cache（faster-whisper）**：每生成一个新 token，仅计算最新的 attention，复用之前的缓存 → **O(n)**
- **无 KV Cache（OpenVINO IR 模型）**：每生成一个新 token，重算全部历史 attention → **O(n²)**

实测数据验证：
- OpenVINO encoder 处理 30s chunk 仅需 **1.25s**（足够快）
- OpenVINO decoder 生成 30s chunk 的文本需 **215s**（瓶颈所在）

### 6.3 OpenVINO 模型结构验证

```
models/OpenVINO/whisper-base-fp16-ov/
├── openvino_encoder_model.xml/bin  (40MB)   ← encoder，有独立 IR
├── openvino_decoder_model.xml/bin  (100MB)  ← decoder，无 past_key_values 输入
├── openvino_tokenizer.xml/bin
└── openvino_detokenizer.xml/bin

decoder 输入: input_ids, encoder_hidden_states, beam_idx
decoder 输出: logits
→ 没有 past_key_values / past_kv_cache 输入端口，确认无 KV Cache 支持
```

## 7. 结论与建议

### 结论

在 CPU 环境下（Intel i7-1165G7），**faster-whisper (CTranslate2/int8) 显著优于 OpenVINO (FP16) 方案，快约 48 倍**。核心差距来自 OpenVINO 导出模型缺少 KV Cache 导致解码器 O(n²) 复杂度。

### 建议

1. **继续使用 faster-whisper** 作为 VideoBot 的音频转录引擎
2. OpenVINO 方案仅在以下条件同时满足时才值得考虑：
   - 硬件具备 Intel GPU / NPU（可利用 OpenVINO 的硬件加速能力）
   - 模型导出时支持 KV Cache（需 optimum-intel 或 OpenVINO 后续版本支持）
   - 可使用 int8 量化版本的 OpenVINO 模型以减少计算量

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `benchmark_whisper.py` | 性能基准测试脚本 |
| `benchmark_results.md` | 自动生成的测试报告 |
| `docs/whisper_benchmark_summary.md` | 本文档（测试总结） |

## 9. 附录：实测原始数据

### faster-whisper 详细数据（2 次迭代）

| 时长 | iter1(s) | iter2(s) | 平均(s) |
|------|---------|---------|--------|
| 1min | 15.41 | 12.73 | 14.07 |
| 5min | 40.00 | 25.75 | 32.88 |
| 10min | 57.76 | 205.50 | 131.63 |
| 20min | 505.79 | 121.66 | 313.72 |
| 30min | 423.37 | 204.96 | 314.16 |
| 1hour | 784.37 | 805.74 | 795.05 |
| 2hour | 1,549.37 | 1,115.48 | 1,332.42 |

> 注意：10min~30min 的 iter1 与 iter2 差异较大，推测与 CPU 热节流或系统后台负载波动有关。

### OpenVINO 详细数据（实测）

| 时长 | 迭代 | 耗时(s) |
|------|------|--------|
| 1min | 2次 | 302.85, 149.37（avg: 226.11） |
| 5min | 1次 | 4,705.24 |
