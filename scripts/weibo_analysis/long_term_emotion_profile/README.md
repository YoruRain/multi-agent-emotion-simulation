# 用户微博情绪推理脚本

本目录用于把已经微调好的 MacBERT 三分类情绪模型迁移到用户历史微博数据上，输出微博级情绪推理结果表。脚本只做推理与结果整理，不会重新训练模型。

## 模型文件

默认模型目录：

```text
D:\GraduationProject\models\macbert_finetuned_sentiment
```

该目录需要包含：

- `task_config.json`：任务配置，至少包含 `num_class`、`max_len`、`label2id`
- tokenizer 文件：由 `BertTokenizer.save_pretrained(...)` 保存
- BERT 底座配置与权重：由 `BertModel.save_pretrained(...)` 保存
- `model_state.pt`：完整分类模型权重

训练 Notebook 中的原模型分类头字段是 `fc`，结构为 `bert + dropout(0.3) + fc(hidden_size, 3)`。当前代码中的 `MacBertSentimentClassifier` 与该结构保持一致，并兼容常见的 `module.`、`classifier.`、`linear.` 等 state_dict 前缀。

## 输入与过滤

默认输入：

```text
D:\GraduationProject\data\high_quality\user_weibo.parquet
```

脚本直接使用输入表中的 `cleaned_content` 作为模型输入文本。默认只保留 `text_quality >= 3` 的记录，并且默认只分析原创微博，也就是 `is_repost == False` 的记录。可通过参数关闭质量过滤，或把转发微博也纳入分析。

结果文件中的 `content` 字段仍然保留原始微博文本，不会写入 `cleaned_content`。

运行前建议先激活项目环境：

```powershell
conda activate d:\GraduationProject\.gp
```

由于输入和默认输出都使用 Parquet，环境中需要安装 `pyarrow` 或 `fastparquet`。

## 输出字段

输出表包含：

- `weibo_id`
- `user_id`
- `content`
- `cleaned_text_length`
- `year`
- `is_repost`
- `sentiment_label_en`
- `sentiment_label`
- `p_neg`
- `p_neu`
- `p_pos`
- `model_confidence`
- `polarity_score`
- `model_intensity`
- `rule_intensity`
- `emotion_intensity_score`
- `text_weight`

其中：

```python
polarity_score = p_pos - p_neg
model_intensity = 1 - p_neu
emotion_intensity_score = 0.5 * model_intensity + 0.5 * rule_intensity
```

`rule_intensity` 会根据强标点、连续标点、重复字符、强语气词、积极/消极情绪词和常见表情标记累加，并裁剪到 `[0, 1]`。`text_weight` 用于长期情绪画像聚合，会综合考虑文本长度、是否转发、模型置信度和文本画像价值，最终裁剪到 `[0.1, 1.0]`。

## 规则特征优化说明

`rule_intensity` 保持“外显情绪表达强度”的定位，扩充了强语气词、积极情绪词、消极情绪词和微博表情标记词表。计算逻辑仍然是可解释的加权累加：标点、连续问号/感叹号、重复字符、强语气词、情绪词、表情标记都会贡献强度分，普通文本通常接近 0，强情绪表达会更容易超过 0.2。

`emotion_intensity_score` 已调整为模型强度和规则强度各占一半：

```python
emotion_intensity_score = 0.5 * model_intensity + 0.5 * rule_intensity
```

`text_weight` 从简单长度权重升级为最终聚合权重。它现在综合文本长度、原创/转发、模型置信度和低画像价值文本降权；极短文本如果包含明显情绪词、表情、强标点或重复字符，不会被过度降权。构建结果表时已同步把 `model_confidence` 传入 `compute_text_weight`，但该参数是可选的，旧代码继续调用 `compute_text_weight(text, is_repost)` 也能正常运行。

## 参数说明

- `--input_path`：输入 Parquet 文件路径
- `--output_path`：输出路径，后缀为 `.parquet` 或 `.jsonl`
- `--model_dir`：微调后模型目录
- `--min_text_quality`：质量过滤阈值，默认 `3`
- `--no_quality_filter`：关闭 `text_quality` 过滤
- `--include_reposts`：把转发微博也纳入分析；默认只分析原创微博
- `--batch_size`：推理 batch size，默认 `64`
- `--max_records` / `--limit`：最多分析 N 条微博；默认在质量过滤和断点续跑过滤后取前 N 条
- `--random_sample`：配合 `--max_records` 使用，从未分析微博中随机抽样
- `--no_resume`：关闭断点续跑，不读取已有输出中的 `weibo_id`
- `--overwrite_output`：覆盖输出文件；默认会将新结果与已有输出合并并按 `weibo_id` 去重
- `--max_len`：覆盖 `task_config.json` 中的 `max_len`
- `--seed`：随机种子，默认 `42`
- `--log_dir`：日志目录，默认 `D:\GraduationProject\.log`
- `--log_file`：指定日志文件路径；不指定时默认写入 `D:\GraduationProject\.log\user_weibo_emotion.log`
- `--verbose`：输出更详细日志

默认每次运行都会同时输出终端日志，并持续追加写入 `.log\user_weibo_emotion.log` 这一个日志文件。

默认开启断点续跑：如果 `--output_path` 指向的结果文件已经存在，程序会读取其中的 `weibo_id`，从输入数据中排除已分析微博，再继续推理剩余微博。输出时会把旧结果和本次新结果合并，并按 `weibo_id` 去重。JSONL 和 Parquet 输出都支持该逻辑。

## 示例运行命令

```powershell
python .\scripts\weibo_analysis\infer_user_weibo_emotion\infer.py `
  --input_path .\data\high_quality\user_weibo.parquet `
  --output_path .\data\analysis\weibos\emotion_profile\user_weibo_emotion_profile.parquet `
  --model_dir .\models\macbert_finetuned_sentiment `
  --batch_size 64 `
  --max_records 100 `
  --random_sample `
  --log_dir .\.log
```

如果希望把转发微博也一起分析，可以额外添加：

```powershell
--include_reposts
```

## 用户级画像聚合

微博级情感结果生成后，可以运行 `build_user_emotion_profile.py` 按 `user_id` 聚合生成用户级长期情绪画像。默认输入路径为：

```text
D:\GraduationProject\data\analysis\weibos\emotion_profile\user_weibo_emotion_analysis.parquet
```

默认输出路径为：

```text
D:\GraduationProject\data\analysis\weibos\emotion_profile\user_emotion_profile.parquet
```

默认会同时生成两个 `profile_version`：

- `2025`：只使用 `year == 2025` 的微博级结果
- `all`：使用全部微博级结果

用户级画像包含 `analyzable_weibo_count`、`weighted_weibo_count`、三类情绪比例、极性均值/中位数/标准差、平均强度、强情绪比例、主导情绪、画像类型、可靠性和中文摘要。`pos_ratio`、`neu_ratio`、`neg_ratio`、`avg_polarity_score`、`avg_intensity_score` 和强情绪比例均使用 `text_weight` 加权；`median_polarity_score` 和 `polarity_std` 使用普通统计口径。

`--strong_threshold` 用于判定强情绪微博，默认值为 `0.7`。`emotion_profile_type` 和 `emotion_profile_summary` 使用规则生成，不会重新训练模型。

示例运行命令：

```powershell
python .\scripts\weibo_analysis\infer_user_weibo_emotion\build_user_emotion_profile.py `
  --input_path .\data\analysis\weibos\emotion_profile\user_weibo_emotion_analysis.parquet `
  --output_path .\data\analysis\weibos\emotion_profile\user_emotion_profile.parquet `
  --strong_threshold 0.7
```
