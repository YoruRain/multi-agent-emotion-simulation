# Comment Analysis 批处理脚本使用说明

本目录用于对微博一级评论进行情绪与立场分析。主入口脚本为：

```powershell
python scripts/comment_analysis/batch_comment_analysis.py
```

推荐先小批量测试：

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --limit 20 --concurrency 2
```

## 基础配置参数

### `--api-key`

DeepSeek API Key。

默认从项目根目录 `.env` 中读取：

```env
DEEPSEEK_API_KEY=你的_api_key
```

也可以命令行传入：

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --api-key sk-xxx
```

### `--api-base`

DeepSeek API Base URL。

默认值：

```text
https://api.deepseek.com
```

也可在 `.env` 中配置：

```env
DEEPSEEK_API_BASE=https://api.deepseek.com
```

### `--model`

使用的 DeepSeek 模型名称。

默认值：

```text
deepseek-chat
```

也可在 `.env` 中配置：

```env
DEEPSEEK_MODEL=deepseek-chat
```

## 输入输出路径参数

### `--comment-path`

评论数据输入路径。

默认值：

```text
data/high_quality/topic_comment.parquet
```

脚本会读取其中同时满足以下条件的评论：

```text
parent_id == -1
text_quality == 3
```

### `--weibo-path`

话题微博数据输入路径。

默认值：

```text
data/high_quality/topic_weibo.parquet
```

脚本会通过 `weibo_id` 将评论关联到话题微博，并读取：

```text
topic
analysis_context
```

作为评论分析上下文。

### `--output-jsonl`

结构化 JSONL 输出路径。

默认值：

```text
data/analysis/comment_analysis_result.jsonl
```

每条评论分析成功后会立即追加写入 JSONL，因此它也是断点续跑的主要依据。

### `--output-parquet`

扁平化 Parquet 输出路径。

默认值：

```text
data/analysis/comment_analysis_result.parquet
```

Parquet 会在 checkpoint 和程序结束时从 JSONL 生成，便于后续统计分析。

### `--failed-jsonl`

失败样本记录路径。

默认值：

```text
data/analysis/comment_analysis_failed.jsonl
```

如果单条评论多次重试后仍失败，会写入该文件。

注意：失败样本不会被断点续跑跳过，下次运行仍会重新尝试。

### `--log-file`

日志文件路径。

默认值：

```text
.log/comment_analysis.log
```

脚本会同时向控制台和日志文件输出运行信息。

## 并发、超时与重试参数

### `--concurrency`

并发请求数，即同时最多发起多少个 DeepSeek API 请求。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_CONCURRENCY=5
```

如果未配置，默认值为：

```text
5
```

建议小批量测试时使用：

```powershell
--concurrency 2
```

全量运行时可根据 API 稳定性调整为 `3`、`5` 或更高。

### `--timeout`

单次 API 请求超时时间，单位为秒。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_TIMEOUT=60
```

如果未配置，默认值为：

```text
60
```

### `--max-retries`

单条评论失败后的最大重试次数。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_MAX_RETRIES=3
```

如果未配置，默认值为：

```text
3
```

注意：总尝试次数为 `1 + max_retries`。

### `--retry-base-delay`

指数退避的基础等待时间，单位为秒。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_RETRY_BASE_DELAY=1.5
```

如果未配置，默认值为：

```text
1.5
```

### `--retry-max-delay`

重试等待时间的上限，单位为秒。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_RETRY_MAX_DELAY=30
```

如果未配置，默认值为：

```text
30
```

## 采样与测试参数

### `--limit`

限制本次运行最多处理多少条待分析评论。

常用于小批量测试：

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --limit 20 --concurrency 2
```

如果不设置，则处理全部待分析评论。

### `--random-sample` / `--no-random-sample`

当设置 `--limit` 时，是否从待处理评论中随机抽样。

默认开启随机抽样：

```powershell
--random-sample
```

如果希望恢复旧行为，直接取前 N 条待处理评论：

```powershell
--no-random-sample
```

也可在 `.env` 中配置：

```env
COMMENT_ANALYSIS_RANDOM_SAMPLE=1
```

关闭则设置为：

```env
COMMENT_ANALYSIS_RANDOM_SAMPLE=0
```

### `--random-seed`

随机采样种子。

设置后，相同数据和相同 seed 会抽到相同评论，便于复现实验：

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --limit 20 --random-seed 42
```

也可在 `.env` 中配置：

```env
COMMENT_ANALYSIS_RANDOM_SEED=42
```

## 保存策略参数

### `--checkpoint-every`

每成功分析多少条评论后保存一次 Parquet checkpoint。

默认从环境变量读取：

```env
COMMENT_ANALYSIS_CHECKPOINT_EVERY=50
```

如果未配置，默认值为：

```text
50
```

例如每成功 20 条保存一次：

```powershell
--checkpoint-every 20
```

如果设置为 `0`，则关闭定期 Parquet checkpoint：

```powershell
--checkpoint-every 0
```

即使关闭 checkpoint，程序正常结束时仍会保存一次最终 Parquet。

## 响应格式参数

### `--no-response-format`

默认情况下，脚本会向 DeepSeek 请求中加入：

```json
{"response_format": {"type": "json_object"}}
```

这有助于模型返回合法 JSON。

如果接口环境不支持该参数，可以使用：

```powershell
--no-response-format
```

脚本中也包含兼容逻辑：如果 API 明确拒绝 `response_format`，会对当前请求自动移除该字段并重试一次。

## 常用运行示例

### 小批量随机测试

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --limit 20 --concurrency 2
```

### 可复现小批量测试

```powershell
python scripts/comment_analysis/batch_comment_analysis.py `
  --limit 20 `
  --concurrency 2 `
  --random-seed 42
```

### 全量运行

```powershell
python scripts/comment_analysis/batch_comment_analysis.py --concurrency 3
```

### 高频保存 checkpoint

```powershell
python scripts/comment_analysis/batch_comment_analysis.py `
  --concurrency 3 `
  --checkpoint-every 20
```

## 断点续跑说明

脚本使用 `comment_id` 判断某条评论是否已经成功分析过。

启动时会读取：

```text
data/analysis/comment_analysis_result.jsonl
data/analysis/comment_analysis_result.parquet
```

如果某个 `comment_id` 已存在于成功结果中，则本次运行会跳过该评论，不会重复调用 API。

失败文件 `comment_analysis_failed.jsonl` 不参与跳过判断，因此失败样本下次会重新尝试。
