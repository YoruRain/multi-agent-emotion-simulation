下面请你为我的项目编写一套可直接运行、结构清晰、便于后续维护的 Python 程序，用于基于评论文本的情绪与立场分析。程序主要通过调用 DeepSeek API 完成分析，并使用异步编程提升处理效率。

## 一、任务背景

我现在要对微博评论数据进行情绪与立场分析，分析结果将用于后续的 Agent 建模。

本阶段的分析对象是评论文本，分析时需要结合评论所属话题微博的上下文信息。大模型的提示词设计和输出 Schema 已经在项目中的 `D:\GraduationProject\scripts\deepseek_sentiment_demo.py` 文件里给出，你需要读取并复用其中已有的 Prompt 思路和 Pydantic Schema 定义。

## 二、需要使用的数据集

1. 评论数据集 `df_topic_comment`
路径：
`D:\GraduationProject\data\high_quality\topic_comment.parquet`

2. 话题微博数据集 `df_topic_weibo`
路径：
`D:\GraduationProject\data\high_quality\topic_weibo.parquet`

### 数据关系说明

- `df_topic_comment` 与 `df_topic_weibo` 共有字段：`weibo_id`
- 需要通过 `df_topic_comment.weibo_id` 关联到对应的话题微博
- `df_topic_weibo` 中包含字段 `analysis_context`，它是对微博正文主要信息的概括，分析评论时应将它作为评论的上文背景输入给大模型
- 如果 `df_topic_comment` 中还有 `content` 字段，则表示评论原文
- 如果 `df_topic_comment` 中还有 `comment_id`、`user_id` 等字段，也应尽量保留并写入结果表中

## 三、程序目标

请编写一个异步批处理分析程序，对评论逐条调用 DeepSeek API，完成情绪与立场分析，并将分析结果分别保存为：

1. JSONL 文件  
路径：
`D:\GraduationProject\data\analysis\comment_analysis_result.jsonl`

要求：
- 每行对应一条评论的完整模型输出结果
- 在模型原始结构化输出基础上，额外加入：
  - `comment_id`
  - `content`（评论原文）
  - 一个键名为 `"analysis_info"` 的字典，值包括：
    - `model_name`：进行分析的模型名
    - `analyzed_at`：进行分析的时间
- 保留原有层次化嵌套结构，不要强行扁平化 JSONL 内容

2. Parquet 文件  
路径：
`D:\GraduationProject\data\analysis\comment_analysis_result.parquet`

要求：
- 保存为 Pandas DataFrame
- 采用扁平化字段结构，便于后续统计与聚合
- 至少包含以下字段：

基础字段：
- `comment_id`
- `weibo_id`
- `content`（评论原文）
- `analysis_context`

模型分析字段（与模型输出格式基本相同）：
- `emotion_target_type`
- `emotion_target_text`
- `stance_target_type`
- `stance_target_text`
- `cause_or_stimulus`
- `target_explicit`
- `focus_type`
- `responsibility`
- `control`
- `norm_violation`
- `emotion_label`
- `emotion_intensity`
- `stance_label`
- `stance_intensity`
- `argument_type`
- `confidence`
- `emotion_evidence`
- `stance_evidence`
- `needs_more_context`
- `low_confidence_reason`

运行追踪字段：
- `model_name`
- `analyzed_at`

## 四、实现要求

### 1. 使用异步编程

由于评论数量较多，必须使用异步方式调用 API。

要求：
- 使用 `asyncio`
- 并发数可配置，不要写死
- 需要使用信号量等方式控制并发，避免请求过载
- 单次请求需要设置超时机制

### 2. DeepSeek API 调用要求

- 使用 DeepSeek API 完成评论分析
- API Key 不要写死在代码中，必须从环境变量或 `.env` 文件中读取
- 模型名称、API Base URL、并发数、重试次数、输入输出路径等都应集中配置，避免散落在代码中
- 需要兼容项目中已有的 Prompt 和 Pydantic Schema 设计，优先复用 `deepseek_sentiment_demo.py` 中的相关内容

### 3. 上下文构造要求

每条评论送入模型时，至少应包含：
- 评论 ID `comment_id`
- 微博话题 `topic`（可通过 `df_topic_weibo[topic]` 访问）
- 微博内容摘要 `source_weibo_summary`（可通过 `df_topic_weibo[analysis_context]` 访问）
- 评论原文 `comment_text`（可通过 `df_topic_comment[content]` 访问）

特别注意：在本阶段先进行一级评论的分析，一级评论在数据集的特征为 `df_topic_comment[parent_id]` 字段为 `-1`，进行数据传输时需注意。

### 4. Schema 校验要求

模型返回结果后，必须进行严格校验。

要求：
- 使用 Pydantic Schema 校验返回结果
- 若返回内容不是合法 JSON，应先尝试提取 JSON，再校验
- 若校验失败，应记录失败原因
- 校验失败的样本支持重试
- 若最终仍失败，不应影响整体程序继续运行

### 5. 重试与容错机制

程序必须具备较强的鲁棒性。

要求至少处理以下异常情况：
- 网络异常
- 请求超时
- API 返回错误
- 非法 JSON
- Pydantic 校验失败
- 单条样本处理失败

建议：
- 重试次数可配置
- 使用指数退避或适当等待机制
- 达到最大重试次数后，将失败样本记录下来并继续处理后续样本

### 6. 断点续跑与去重

为了节省 API 成本，程序需要支持断点续跑。

要求：
- 以 `comment_id` 作为唯一标识
- 如果输出 JSONL 或已有结果中已经存在该 `comment_id`，则默认跳过，不重复分析
- 程序中断后重新运行时，应能自动识别已完成样本并续跑
- 不允许在最终结果中出现重复的 `comment_id`

### 7. 保存策略

要求同时保存两种格式：

#### JSONL
- 适合保存原始层次化结构结果
- 建议边处理边追加写入
- 保证单条结果及时落盘，减少中断损失

#### Parquet
- 适合保存扁平化后的分析表
- 可在程序结束时统一保存
- 也可支持每处理一定批次后 checkpoint 保存，避免长任务中断导致结果丢失

此外，请额外保存一份失败样本记录文件，例如：
`D:\GraduationProject\data\analysis\comment_analysis_failed.jsonl`

失败记录至少应包含：
- `comment_id`
- `content`
- `error_type`
- `error_message`
- `retry_count`

### 8. 日志与进度输出

程序应具备清晰的日志输出。

至少记录：
- 总样本数
- 待处理数
- 已成功数
- 已失败数
- 已跳过数
- 当前进度
- 重试信息
- 最终统计摘要

尽量使用 `logging`，不要只使用简单的 `print`。

### 9. 代码结构要求

不要把所有逻辑都堆在一个脚本里。请尽量写得清晰、模块化、便于维护。

至少应体现出这些部分：
- 配置管理
- 数据读取与预处理
- 上下文拼接
- API 异步调用
- 返回结果解析与 Schema 校验
- JSONL / Parquet 保存
- 失败记录保存
- 主流程调度

即使最终输出为单文件脚本，也请通过函数和类清晰组织代码，并写出必要注释。

如果需要创建多个代码文件，请在 `D:\GraduationProject\scripts` 目录下新建一个 `comment_analysis` 目录，并在其中创建代码文件。

### 10. 可读性与可维护性

请生成的代码满足以下要求：
- Python 代码风格清晰规范
- 关键函数有类型注解
- 关键步骤有注释
- 路径、字段名、配置项尽量集中定义
- 对缺失字段和异常情况有合理判断
- 不要省略关键实现逻辑，不要只给框架

## 五、补充说明

1. 请优先兼容 Windows 本地运行环境
2. 读取 Parquet 文件时使用 Pandas
3. 对扁平化字段的提取，要严格对应 Pydantic Schema 的层次结构
4. 请提供完整可运行代码，而不是伪代码
5. 如果你认为某些实现细节需要做合理假设，可以直接采用合理默认方案，并在代码注释中说明
6. 若 `deepseek_sentiment_demo.py` 中已有现成的 Prompt、Schema、调用方式或辅助函数，请尽量复用，而不是重复发明
7. 如果有必要，请顺手补充少量辅助函数，例如：
   - 读取已完成 `comment_id`
   - 扁平化嵌套结果为表格记录
   - 从原始响应中提取 JSON
   - 保存失败样本
   - 定期 checkpoint 保存 parquet

## 六、最终输出要求

请直接输出完整的 Python 代码，实现上述功能。

如果代码较长，请优先保证：
- 主流程完整
- 关键函数完整
- 异步调用完整
- 保存逻辑完整
- 可直接运行和后续修改

不要只输出设计思路或伪代码。