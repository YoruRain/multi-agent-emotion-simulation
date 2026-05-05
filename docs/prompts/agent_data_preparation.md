你现在需要在本项目中完成“微博用户 Agent 建模准备层”的第一阶段代码开发。

本阶段先根据已有数据集构建标准化的 Agent Profile、系统 Prompt 和记忆样本文件，为后续使用 AgentScope 的 ReActAgent 或其他 Agent 类做准备。

请务必遵循以下要求：

1. 使用 Python 编写代码。
2. 使用 UTF-8 编码读取和写入所有文本文件。
3. 优先使用 pathlib 处理路径。
4. 使用 pandas 读取 parquet/csv/json/jsonl 等数据。
5. 代码应具有清晰的函数划分、类型注解、日志输出和基本异常处理。
6. 字段缺失时不要直接报错中断，应使用默认值，并在日志中记录缺失情况。
7. 不要把所有原始数值字段直接暴露给大模型 prompt。数值字段应主要放入 behavior_parameters 或 metadata。
8. 输出文件统一使用 JSONL 格式，便于后续逐行加载。


## 一、profile_builder.py


请编写 profile_builder.py，用于根据项目中已有的以下数据构建标准化的 agent_profiles.jsonl：

1. 用户基础信息数据集

路径：data\high_quality\user_info.parquet
   典型字段包括：
   user_id, screen_name, gender, verified, verified_type, verified_type_name, description, user_value, user_value_label

2. 用户长期情绪画像

路径：data\profile\weibos\emotion_profile\user_emotion_profile.parquet
   典型字段包括：
   user_id, pos_ratio, neu_ratio, neg_ratio, avg_intensity_score, strong_emotion_ratio, emotion_profile_type, profile_reliability, emotion_profile_summary

3. 用户主题画像

路径：data\profile\weibos\subject_profile\user_topic_profile_final.parquet
   典型字段包括：
   user_id, public_issue_topic_ratio, final_public_issue_topic_ratio,
   entertainment_topic_ratio, final_entertainment_topic_ratio,
   daily_life_topic_ratio, final_daily_life_topic_ratio,
   repost_topic_dependency, final_topic_profile_reliability,
   topic_summary

4. 用户传播画像

路径：data\profile\weibos\propagation_profile\user_propagation_profile.parquet
   典型字段包括：
   user_id, propagation_activity_level,original_ratio, repost_ratio, repost_with_comment_ratio, media_dependency_score, kol_sensitivity_score, avg_engagement, influence_score, influence_level, propagation_role, propagation_summary

5. 用户记忆样本摘要集
路径：data\profile\weibos\memory_sample\user_memory_summary.parquet
   典型字段包括：
   user_id, memory_user_level, selected_memory_count, memory_type_counts,
   selected_weibo_ids, memory_summary_for_agent

请将这些数据按 user_id 合并，生成每行一个 Agent 的标准 JSONL 文件：agent_profiles.jsonl。

每一行的结构如下：

{
  "agent_id": "weibo_user_{user_id}",
  "user_id": "...",
  "base_identity": {
    "screen_name": "...",
    "gender": "...",
    "verified_type_name": "...",
    "memory_user_level": "...",
    "user_value_label": "...",
    "influence_level": "..."
  },
  "prompt_profile": {
    "identity_summary": "...",
    "emotion_summary": "...",
    "topic_summary": "...",
    "propagation_summary": "...",
  },
  "behavior_parameters": {
    "pos_ratio": 0.0,
    "neg_ratio": 0.0,
    "neu_ratio": 0.0,
    "avg_intensity_score": 0.0,
    "strong_emotion_ratio": 0.0,
    "final_public_issue_topic_ratio": 0.0,
    "final_entertainment_topic_ratio": 0.0,
    "final_daily_life_topic_ratio": 0.0,
    "repost_ratio": 0.0,
    "repost_with_comment_ratio": 0.0,
    "media_dependency_score": 0.0,
    "kol_sensitivity_score": 0.0,
    "influence_score": 0.0
  },
  "metadata": {
    "selected_memory_count": 0,
    "memory_type_counts": {},
    "selected_weibo_ids": [],
  }
}

请实现以下功能：

1. 自动读取各画像数据文件。
   - 如果项目已有固定路径，请使用项目已有路径。

2. 合并策略：
   - 以 user_id 为主键。
   - 建议以用户基础信息为主表。

3. 默认值策略：
   - 字符串缺失：使用 "未知" 或 "暂无相关画像信息"。
   - 数值缺失：使用 0.0 或 0。
   - 比例字段缺失：使用 0.0。
   - memory_user_level 缺失：使用 "background"。
   - selected_memory_count 缺失：使用 0。


4. prompt_profile 的生成规则：

   identity_summary：
   根据 verified_type_name、follower_level、propagation_activity_level、propagation_role 等字段生成一句简短中文摘要。
   示例：
   “该用户是普通微博用户，粉丝规模较低，活跃程度适中，主要作为普通参与者出现。”

   emotion_summary：
   优先使用 emotion_profile_summary。

   topic_summary：
   优先使用 topic_summary。

   propagation_summary：
   优先使用 propagation_summary。

6. 输出日志：
   请使用 logging 模块输出：
   - 各输入文件读取成功与否
   - 各数据表行数
   - 合并后的用户数
   - 各画像缺失数量
   - 输出文件路径

7. 脚本入口：
   请提供 main() 函数，并支持命令行运行：
   python profile_builder.py

## 二、prompt_renderer.py

请编写 prompt_renderer.py，用于读取 agent_profiles.jsonl，并根据其中的 prompt_profile 字段生成每个 Agent 的系统 Prompt，输出到 agent_sys_prompts.jsonl。

输入：
agent_profiles.jsonl

输出：
agent_sys_prompts.jsonl

每一行结构如下：

{
  "agent_id": "weibo_user_{user_id}",
  "user_id": "...",
  "sys_prompt": "..."
}

系统 Prompt 使用固定模板，不要让模板过长。要求适合后续传入 AgentScope 的 ReActAgent。

模板内容如下，可根据代码风格适当调整，但含义不要改变：

你正在扮演一个微博用户 Agent。你需要根据该用户的长期画像、主题兴趣、传播习惯和代表性记忆，对给定热点事件或微博语境做出符合该用户特征的反应。

【基础身份】
{identity_summary}

【长期情绪倾向】
{emotion_summary}

【主题兴趣】
{topic_summary}

【传播行为】
{propagation_summary}


【行为要求】
1. 你应根据该用户画像判断是否参与讨论，而不是默认每次都发言。
2. 如果参与讨论，表达应符合该用户的情绪倾向、主题偏好和传播习惯。
3. 如果该用户对当前事件兴趣较低，或者画像可靠性不足，可以选择低强度反应或不参与。
4. 不要机械复述画像字段，不要提到“画像”“数据集”“模型”等元信息。
5. 输出必须是严格 JSON，不要输出额外解释。

【输出格式】
{
  "participate": true,
  "action_type": "comment",
  "emotion_label": "neutral",
  "emotion_intensity": 1,
  "stance_label": "neutral",
  "stance_intensity": 1,
  "reaction_text": "这里写该用户可能发表的微博式反应",
  "reason": "用一句话说明为什么该用户会产生这种反应"
}

字段约束：
- participate: true 或 false
- action_type: "ignore", "comment", "repost", "repost_with_comment"
- emotion_label: "positive", "neutral", "anger", "sadness", "disgust", "worry", "surprise"
- emotion_intensity: 0, 1, 2
- stance_label: "support", "against", "neutral", "unclear"
- stance_intensity: 0, 1, 2
- reaction_text: 如果 participate 为 false，可以为空字符串
- reason: 简短说明即可

请注意：
1. 不要把 behavior_parameters 中的大量数值直接拼接进 sys_prompt。
2. prompt 中只使用 prompt_profile 中的自然语言摘要。
3. 如果某个摘要缺失，请使用默认文本。
4. 输出 JSONL 时确保中文不被转义，即 ensure_ascii=False。
5. 输出日志包括读取数量、成功渲染数量、输出路径、前 2 条样例。

请提供 main() 函数，并支持命令行运行：
python prompt_renderer.py

## 三、memory_builder.py

请编写 memory_builder.py，用于根据微博记忆样本集构建每个 Agent 的记忆文件 agent_memories.jsonl。

输入数据：
微博记忆样本集，典型字段包括：
user_id, weibo_id, memory_user_level, memory_type, content_for_agent,
topic_tags_for_agent, mentions_for_agent, source_context_for_agent,
is_repost, has_repost_comment, sentiment_label, final_topic_categories,
final_topic_labels, topic_signal_source, source_author_type, engagement_score,
selection_reason

输出文件：
agent_memories.jsonl

每一行对应一个用户 Agent，结构如下：

{
  "agent_id": "weibo_user_{user_id}",
  "user_id": "...",
  "memory_user_level": "core",
  "memories": [
    {
      "memory_id": "...",
      "weibo_id": "...",
      "mark": "style_memory",
      "content": "历史微博样本：[原创｜日常生活｜中性] ……",
      "topics": "topic_tags_for_agent",
      "mentions": "mentions_for_agent",
      "metadata": {
        "memory_type": "...",
        "is_repost": false,
        "has_repost_comment": false,
        "sentiment_label": "...",
        "final_topic_categories": [],
        "final_topic_labels": [],
        "topic_signal_source": "...",
        "source_author_type": "...",
        "selection_reason": "..."
      }
    }
  ]
}

请实现以下规则：

1. 按 user_id 分组构建记忆。
2. 每个用户的记忆样本数量原则上遵循 memory_user_level：
   - core：最多 6 条
   - normal：最多 3 条
   - background：最多 1 条
   数据集中已经完成筛选，可以保留现有样本；但仍需做上限保护。
5. content 字段应为面向 Agent 的自然语言记忆，不要只放原文。
   推荐格式：
   历史微博样本：[原创｜{主题类别}｜{情绪标签}] {content_for_agent}
   或：
   历史微博样本：[转发附评｜{主题类别}｜{情绪标签}] {content_for_agent}
   或：
   历史微博样本：[纯转发｜{主题类别}｜{情绪标签}] {content_for_agent}

   如果有 source_context_for_agent，追加：
   源微博背景：{source_context_for_agent}
6. mark 字段根据 memory_type 和内容类型生成：
   - 如果 memory_type 包含 style 或 表达风格，使用 "style_memory"
   - 如果 final_topic_categories 包含“社会公共事件”或“政策民生”或“时事政治”，使用 "public_issue_memory"
   - 如果 is_repost 为 true，使用 "propagation_memory"
   - 其他情况使用 "general_memory"
8. 对 topic_tags_for_agent、mentions_for_agent、final_topic_categories、final_topic_labels 等字段，它们都是以字符串格式存储的对象。如果字段中包含多值，则会使用英文逗号 `,` 分隔。
9. 输出日志包括：
   - 读取记忆样本数
   - 用户数
   - core/normal/background 用户数
   - 每类 mark 的数量
   - 输出路径

请提供 main() 函数，并支持命令行运行：
python memory_builder.py

## 四、代码质量要求

上述3个脚本文件存储在 scripts\agent_data_preparation 路径下，输出的数据文件存储在 data\scope 路径下

请尽量将公共逻辑抽成函数，例如：

- read_table(path: Path) -> pd.DataFrame
- write_jsonl(records: list[dict], path: Path) -> None
- safe_get(row, field, default)
- safe_float(value, default=0.0)
- safe_int(value, default=0)
- parse_list_like(value) -> list
- build_identity_summary(row) -> str
- build_emotion_summary(row) -> str
- build_topic_summary(row) -> str
- build_propagation_summary(row) -> str
- render_sys_prompt(prompt_profile: dict) -> str
- build_memory_content(row) -> str


请注意：
1. 不要删除或覆盖项目已有重要文件。
2. 输出目录不存在时自动创建。
3. 写文件前可以覆盖同名输出文件，但需在日志中说明。
4. 尽量保证脚本可以单独运行。
5. 运行后请给出简短总结，包括新增/修改文件、输出文件路径、运行方式和注意事项。


请先完成上述三个脚本，并确保它们可以独立运行。