你现在需要在本项目中完成“Agent 系统事件输入构建”的代码开发。

本阶段目标是：根据已有的话题微博数据 df_topic_weibo 和评论情绪观点分析结果 df_comment_analysis，生成可供后续 AgentScope 单轮反应模拟使用的 events.jsonl。


请编写脚本：

scripts/agent_data_preparation/event_builder.py


输出文件为：

data/scope/events.jsonl

## 一、输入数据说明

1. df_topic_weibo

字段包括：

[
    "weibo_id", "user_id", "screen_name", "gender", "topic", "content",
    "text_length", "create_time", "like_count", "comment_count",
    "repost_count", "engagement", "comment_crawled_count",
    "comment_hq_count", "comment_hq_ratio", "comment_hq_user_count",
    "topic_value", "topic_value_label", "trending_date",
    "trending_type", "trending_click", "summary_text", "analysis_context" 
]

优先使用 analysis_context 作为 event_context。

2. df_comment_analysis

评论分析结果是扁平字段，由前期 Pydantic Schema 展开而来。

请优先兼容扁平字段形式，例如：

['comment_id', 'weibo_id', 'parent_id', 'content', 'parent_comment_text',
 'analysis_context', 'topic', 'emotion_target_type', 'emotion_target_text', 
 'stance_target_type', 'stance_target_text', 'cause_or_stimulus', 
 'target_explicit', 'focus_type', 'responsibility', 'control', 'norm_violation', 
 'emotion_label', 'emotion_intensity', 'stance_label', 'stance_intensity', 
 'argument_type', 'confidence', 'emotion_evidence', 'stance_evidence', 
 'needs_more_context', 'low_confidence_reason', 'semantic_validation_warnings', 
 'model_name', 'analyzed_at', 'run_id']


## 二、输出文件结构

请生成 events.jsonl，每一行对应一个话题微博事件，结构如下：

{
  "event_id": "event_{weibo_id}",
  "weibo_id": "...",
  "topic": "...",
  "event_context": "...",
  "event_type": "public_issue",
  "event_emotion_tendency": "negative",
  "event_emotion_summary": "评论区整体以负向情绪为主，主导情绪接近 anger，整体情绪强度中等偏高。",
  "event_stance_focus": "评论区主要围绕相关机构的处理方式和责任归属展开评价，其中不少评论带有规范违背或道德谴责判断。",
  "dominant_emotion_label": "anger",
  "dominant_stance_label": "against",
  "dominant_stance_target_type": "institution",
  "dominant_stance_target_text": "相关机构",
  "dominant_responsibility": "institution",
  "dominant_norm_violation": "high",
  "comment_count_used": 128,
  "emotion_distribution": {
    "negative": 0.67,
    "positive": 0.04,
    "neutral": 0.18,
    "mixed": 0.06,
    "unclear": 0.05
  },
  "stance_distribution": {
    "favor": 0.10,
    "against": 0.62,
    "neutral": 0.18,
    "mixed": 0.05,
    "unclear": 0.05
  },
  "metadata": {
    "comment_crawled_count": 0,
    "comment_hq_count": 0,
    "comment_hq_ratio": 0.0,
    "topic_value": 0,
    "topic_value_label": "",
    "trending_type": "",
    "trending_click": 0
  }
}

请注意：

1. stance_label 请优先使用项目已有评论分析结果中的标签：
   favor / against / neutral / mixed / unclear


2. event_emotion_tendency 建议使用：
   negative / positive / neutral / mixed / unclear

3. event_type 可直接定义为 public_issue。

## 三、情绪聚合规则

请按 weibo_id 对 df_comment_analysis 分组，对每条话题微博下的评论进行聚合。

评论情绪标签包括：

[
    "anger", "sadness", "fear", "joy", "disgust",
    "disappointment", "surprise", "sympathy",
    "confusion", "admiration", "none", "mixed", "unclear"
]

请将它们映射为事件层面的情绪组：

NEGATIVE_EMOTIONS = {
    "anger", "sadness", "fear", "disgust",
    "disappointment", "confusion"
}

POSITIVE_EMOTIONS = {
    "joy", "sympathy", "admiration"
}

NEUTRAL_EMOTIONS = {
    "none", "surprise"
}

MIXED_EMOTIONS = {
    "mixed"
}

UNCLEAR_EMOTIONS = {
    "unclear"
}

请实现：

infer_emotion_group(label: str) -> str

返回：
negative / positive / neutral / mixed / unclear

请实现轻量加权：

weight = confidence * (1 + 0.3 * emotion_intensity)

其中：
- confidence 缺失时默认为 0.0
- emotion_intensity 缺失时默认为 0
- needs_more_context 为 True 时，weight *= 0.5
- 如果所有 weight 之和为 0，则退化为每条评论 weight = 1.0

请聚合得到：

- negative_emotion_ratio
- positive_emotion_ratio
- neutral_emotion_ratio
- mixed_emotion_ratio
- unclear_emotion_ratio
- dominant_emotion_label
- avg_emotion_intensity
- strong_emotion_ratio，其中 emotion_intensity >= 2 视为较强情绪

event_emotion_tendency 规则建议如下：

1. 如果 negative_ratio >= 0.45 且 negative_ratio - positive_ratio >= 0.15，则为 negative
2. 如果 positive_ratio >= 0.45 且 positive_ratio - negative_ratio >= 0.15，则为 positive
3. 如果 negative_ratio >= 0.25 且 positive_ratio >= 0.25，则为 mixed
4. 如果 mixed_ratio >= 0.25，则为 mixed
5. 如果 neutral_ratio >= 0.50，则为 neutral
6. 否则为 unclear

请实现：

aggregate_emotion_features(group: pd.DataFrame) -> dict
infer_event_emotion_tendency(features: dict) -> str
build_event_emotion_summary(features: dict) -> str

event_emotion_summary 需要生成简短中文描述，例如：

- “评论区整体以负向情绪为主，主导情绪接近 anger，整体情绪强度较高。”
- “评论区情绪较为分化，正负向反应同时存在。”
- “评论区整体情绪较为中性，强烈情绪表达相对较少。”
- “评论区整体情绪倾向不够明确。”

## 四、立场焦点聚合规则

请根据以下字段聚合事件层面的立场焦点：

- stance_target_type
- stance_target_text
- stance_label
- stance_intensity
- responsibility
- norm_violation
- cause_or_stimulus
- confidence
- needs_more_context

stance_label 可能包括：

[
    "favor", "against", "neutral", "mixed", "unclear"
]

请实现立场加权：

stance_weight = confidence * (1 + 0.3 * stance_intensity)

其中：
- confidence 缺失时默认为 0.0
- stance_intensity 缺失时默认为 0
- needs_more_context 为 True 时，stance_weight *= 0.5
- 如果所有 stance_weight 之和为 0，则退化为每条评论 stance_weight = 1.0

请实现：

aggregate_stance_features(group: pd.DataFrame) -> dict

具体要求：

1. 优先使用 stance_target_type != "unclear" 且 stance_label != "unclear" 的评论。
2. stance_target_text 可以为空。若文本太分散，可以退回到 stance_target_type 级别。
3. 按 stance_target_type + stance_target_text 进行加权聚合，选出 dominant_stance_target_type 和 dominant_stance_target_text。
4. 在 dominant target 对应评论中，聚合 stance_label，得到 dominant_stance_label。
5. 在 dominant target 对应评论中，聚合 responsibility，得到 dominant_responsibility。
6. 在 dominant target 对应评论中，聚合 norm_violation，得到 dominant_norm_violation。
7. 生成 event_stance_focus 的中文描述。

请实现：

build_stance_focus_description(
    target_type: str,
    target_text: str,
    responsibility: str,
    norm_violation: str
) -> str

可使用以下模板：

- target_type == "person":
  评论区主要围绕人物“{target_text}”的言行是否合理展开评价。

- target_type == "institution":
  评论区主要围绕机构“{target_text}”的处理方式和责任归属展开评价。

- target_type == "policy":
  评论区主要围绕相关政策“{target_text}”是否合理、是否应被支持展开讨论。

- target_type == "group":
  评论区主要围绕群体“{target_text}”的行为、处境或责任展开评价。

- target_type == "media":
  评论区主要围绕媒体“{target_text}”的报道方式或舆论引导展开评价。

- target_type == "behavior":
  评论区主要围绕“{target_text}”这一行为是否合理、是否违反公共规范展开评价。

- target_type == "event":
  评论区主要围绕“{target_text}”这一事件本身的性质、影响和处理方式展开评价。

- 其他情况：
  评论区主要围绕该事件本身的性质、责任归属和处理方式展开讨论。

如果 norm_violation 为 high 或 medium，请追加：
“其中不少评论带有规范违背或道德谴责判断。”

如果 responsibility 为：
- institution：追加 “评论中较多将责任归于相关机构或管理方。”
- other_individual：追加 “评论中较多将责任归于具体个人。”
- media：追加 “评论中较多关注媒体报道或舆论传播责任。”
- society：追加 “评论中较多将问题归因于社会层面的结构性因素。”

如果无法判断具体立场焦点，请使用默认描述：
“评论区主要围绕该事件本身展开讨论，但具体争议焦点不够明确。”


## 五、事件上下文 event_context

event_context 用于后续输入给 Agent。

请实现：

build_event_context(event_row: pd.Series) -> str

优先使用 analysis_context。

## 七、主流程

请实现：

build_event_inputs(
    df_topic_weibo: pd.DataFrame,
    df_comment_analysis: pd.DataFrame
) -> list[dict]

逻辑：

1. 遍历 df_topic_weibo 的每一行。
2. 根据 weibo_id 找到对应评论分析结果。
3. 调用 aggregate_emotion_features。
4. 调用 aggregate_stance_features。
5. 调用 infer_event_type。
6. 调用 build_event_context。
7. 组装事件记录。
8. 返回 records。

如果某个 weibo_id 没有评论分析结果，也要生成事件记录，但：
- event_emotion_tendency = "unclear"
- event_emotion_summary = "评论区暂无足够的情绪分析结果。"
- event_stance_focus = "评论区暂无足够的立场分析结果。"
- comment_count_used = 0

## 八、文件读取与写入

请实现通用函数：

read_table(path: Path) -> pd.DataFrame
write_jsonl(records: list[dict], path: Path) -> None

读取格式支持：
- parquet

写出 JSONL 时：
- ensure_ascii=False
- 每行一个 JSON 对象
- 输出目录不存在时自动创建

相关文件路径如下：

TOPIC_WEIBO_PATH = Path("data/high_quality/topic_weibo.parquet")
COMMENT_ANALYSIS_PATH = Path("data/profile/comments/comment_analysis_result.parquet")
OUTPUT_PATH = Path("data/scope/events.jsonl")


## 九、日志要求

请使用 logging 模块输出：

1. 输入文件路径
2. df_topic_weibo 行数
3. df_comment_analysis 行数
4. 成功生成的事件数
5. 没有评论分析结果的事件数
6. event_emotion_tendency 分布
7. dominant_emotion_label 分布
8. dominant_stance_label 分布
9. event_type 分布
10. 输出文件路径
11. 前 3 条事件输入样例，注意不要输出过长 event_context

## 十、代码质量要求

请遵循以下要求：

1. Python 代码应具有清晰函数划分。
2. 使用类型注解。
3. 使用 pathlib。
4. 使用 UTF-8。
5. 对缺失字段做好兼容，不要轻易中断。
6. 对数值字段做好 safe_float / safe_int 转换。
7. 对布尔字段 needs_more_context 做安全解析，兼容 True、False、"true"、"false"、1、0。
8. 如果出现未知 emotion_label 或 stance_label，请归入 unclear。
9. 不要修改原始数据文件。
10. 运行完成后请给出简短总结，包括新增/修改文件、输出路径和运行方式。

## 十一、脚本入口

请提供 main() 函数，并支持命令行运行：

python agent_simulation/src/event_builder.py