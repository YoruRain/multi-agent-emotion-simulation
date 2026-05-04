你是我的项目代码开发助手。当前项目为“基于多智能体的社会群体情绪模拟系统的设计与实现”。现在需要在已有用户画像结果的基础上，为后续 Agent 大模型构建“用户记忆样本”。

## 一、任务背景

目前已经完成了三类用户级画像：

1. 长期情绪画像
2. 主题画像
3. 传播画像

现在的目标是：基于历史数据构建长期用户画像，因此记忆样本的取样范围可以扩展到用户在2025年和2026年中发布的微博。

记忆样本的作用是从历史微博中选取能够体现用户长期情绪倾向、主题偏好、传播行为和语言表达特征的代表性文本，为后续大语言模型驱动的 Agent 提供个体化上下文。

此外，记忆样本还需要承担一部分表达风格参考的作用。

## 二、已有数据与字段

请先阅读项目中已有的数据文件、字段说明和相关脚本。

当前已有或预计会用到的数据包括：

### 1. 建模用户数据集

路径：`data\high_quality\user_info.parquet`

字段包括：
```python
[
    'user_id', 'screen_name', 'gender', 'ip_location', 'registration_time',
    'account_age_days', 'verified', 'verified_type', 'verified_type_name',
    'total_weibo_count', 'follower_count', 'following_count',
    'follower_following_ratio', 'user_rank', 'weibo_crawled_count',
    'weibo_2025_count', 'weibo_hq_count', 'weibo_hq_ratio', 'original_ratio',
    'comment_crawled_count', 'comment_hq_count', 'comment_hq_ratio',
    'active_days', 'description', 'user_value', 'user_value_label'
]
```

### 2. 用户历史微博数据集

路径：
   - 前期曾做过一次用户及微博筛选，因此这里提供两个历史微博数据集
   - `data\high_quality\user_weibo.parquet`：仅包含高质量的用户发布的微博，不包含他们转发的源微博
   - `data\cleaned\user_weibo.parquet`：包含所有用户微博和他们转发的源微博
   - 两张表的字段完全一致

字段包括：

```python
[
    'weibo_id', 'user_id', 'screen_name', 'content', 'cleaned_content',
    'text_length', 'cleaned_text_length', 'text_quality', 'text_quality_label',
    'create_time', 'year', 'month', 'day', 'hour', 'weekday', 'like_count',
    'comment_count', 'repost_count', 'engagement', 'is_repost', 'reposted_weibo_id',
    'topics', 'at_users'
]
```

### 3. 微博级长期情绪分析结果

路径：`data\profile\weibos\emotion_profile\user_weibo_emotion_analysis.parquet`

字段包括：

```python
[
    'weibo_id', 'user_id', 'content', 'cleaned_text_length', 'year',
    'is_repost', 'sentiment_label_en', 'sentiment_label',
    'p_neg', 'p_neu', 'p_pos', 'model_confidence',
    'polarity_score', 'model_intensity', 'rule_intensity',
    'emotion_intensity_score', 'text_weight'
]
```

### 4. 用户级长期情绪画像

路径：`data\profile\weibos\emotion_profile\user_emotion_profile.parquet`

字段包括：

```python
[
    'user_id', 'profile_version', 'analyzable_weibo_count',
    'weighted_weibo_count', 'pos_ratio', 'neu_ratio', 'neg_ratio',
    'avg_polarity_score', 'median_polarity_score', 'polarity_std',
    'avg_intensity_score', 'strong_emotion_ratio',
    'strong_positive_ratio', 'strong_negative_ratio',
    'dominant_emotion', 'emotion_profile_type',
    'profile_reliability', 'emotion_profile_summary'
]
```

### 5. 微博级综合主题分析结果

路径：`data\profile\weibos\subject_profile\user_weibo_topic_fusion.parquet`

字段包括：

```python
[
    'weibo_id', 'user_id', 'content', 'text_quality', 'is_repost',
    'reposted_weibo_id', 'source_content', 'has_repost_comment',
    'user_topics', 'source_topics', 'explicit_keywords',
    'explicit_topic_categories', 'signal_confidence',
    'implicit_cluster_id', 'implicit_analysis_text',
    'implicit_analysis_text_length', 'distance_to_center',
    'implicit_topic_label', 'implicit_topic_category',
    'implicit_topic_confidence_level', 'implicit_topic_base_score',
    'cluster_distance_quantile_group', 'distance_factor',
    'implicit_topic_confidence_score', 'implicit_topic_valid',
    'final_topic_categories', 'final_topic_labels',
    'topic_signal_source', 'final_topic_confidence'
]
```

### 6. 用户级主题画像

路径：`data\profile\weibos\subject_profile\user_topic_profile_final.parquet`

字段包括：

```python
[
    'user_id', 'total_weibo_count', 'repost_weibo_count',
    'original_weibo_count', 'top_user_topics', 'top_source_topics',
    'top_all_topics', 'top_categories', 'top_implicit_topic_labels',
    'final_category_distribution', 'public_issue_topic_ratio',
    'final_public_issue_topic_ratio', 'entertainment_topic_ratio',
    'final_entertainment_topic_ratio', 'daily_life_topic_ratio',
    'final_daily_life_topic_ratio', 'repost_topic_dependency',
    'explicit_topic_coverage', 'implicit_valid_topic_coverage',
    'final_topic_coverage', 'topic_source_balance_label',
    'explicit_topic_profile_reliability',
    'final_topic_profile_reliability', 'avg_signal_confidence',
    'avg_final_topic_confidence', 'topic_summary',
    'topic_summary_quality'
]
```

### 7. 用户级传播画像

路径：`data\profile\weibos\propagation_profile\user_propagation_profile.parquet`

字段包括：

```python
[
    'user_id', 'weibo_hq_count', 'active_days',
    'propagation_activity_level', 'original_ratio', 'repost_ratio',
    'repost_with_comment_ratio', 'source_media_ratio',
    'source_government_ratio', 'source_institution_ratio',
    'source_personal_verified_ratio', 'source_high_follower_ratio',
    'high_personal_verified_ratio', 'media_dependency_score',
    'kol_sensitivity_score', 'avg_engagement',
    'high_engagement_weibo_ratio', 'influence_score',
    'influence_level', 'propagation_role', 'propagation_summary'
]
```

## 三、脚本参考

上述生成各分析结果与用户画像的相关脚本均位于路径 `scripts\weibo_analysis` 下。

如有必要，可以在该目录下查看脚本具体实现细节。

## 四、开发目标

请新增一个“用户记忆样本筛选”模块，完成以下工作：

1. 读取上述提到的数据集与分析结果。
2. 以 `user_id` 和 `weibo_id` 为主键整合微博级情绪与主题结果。
3. 基于已有画像结果，为每个用户从历史微博中筛选代表性记忆样本。
4. 输出微博级记忆样本表。
5. 可选地输出用户级记忆摘要表，以便后续 Agent Prompt 构建。
6. 代码应尽量模块化、可配置、可复用，并与项目现有代码风格保持一致。

## 五、记忆样本筛选原则

记忆样本应优先体现以下信息：

1. 用户长期关注的主要主题
2. 用户典型情绪倾向与强情绪表达
3. 用户日常语言风格与表达方式
4. 用户是否偏原创表达、转发扩散、媒体依赖或 KOL 跟随
5. 用户在公共议题中的参与倾向
6. 用户是否存在较高互动或潜在影响力

记忆样本不追求覆盖所有历史微博，也不追求选择“最极端”的微博，而是要在有限条数内尽量体现该用户的长期画像。

## 六、候选池过滤规则

请为每个用户构建候选微博池。建议优先保留：

1. `text_quality >= 3` 的微博。
2. 原创微博，即 `is_repost == False`。
3. 带有转发评语的转发微博，即 `is_repost == True` 且 `has_repost_comment == True`。
4. 情绪分析可信度较高的微博，例如 `model_confidence` 较高。
5. 主题识别可信度较高的微博，例如 `final_topic_confidence` 较高。
6. 属于用户高频主题类别或高频主题标签的微博。
   - 高频主题类别可参考 `final_category_distribution` 字段；高频主题标签可参考 `top_implicit_topic_labels` 字段
   - 两字段均以 `[(category/label_name, count, ratio), (...)]` 的 object 形式存储在 parquet 文件中，并且列表元素按 `count` 的降序排列
7. 能体现用户主要情绪倾向或强情绪倾向的微博。
   - 主要情绪倾向可参考 `pos_ratio`, `neg_ratio` 和 `neu_ratio` 字段
   - 强情绪倾向可参考 `strong_emotion_ratio` 字段
8. 能体现传播行为的微博，例如带评语转发、媒体源转发、高粉个人源转发等。
   - 传播行为可参考 `repost_ratio`, `repost_with_comment_ratio`, `media_dependency_score`, `kol_sensitivity_score` 字段
9. 文本长度适中、语义完整、适合放入 Agent Prompt 的微博。

建议排除：

1. 文本过短、语义不完整的微博。
2. 仅包含链接、话题标签、表情、媒体占位、转发占位的微博。
   - 这一点可以通过 `text_quality < 3` 来判断
3. 过长且不适合直接放入 Prompt 的微博。
4. 重复文本或高度相似文本。
5. 主题与情绪分析均不可靠的微博。

## 七、记忆样本类型

请为每条入选记忆样本分配 `memory_type`。建议支持以下类型：

1. `typical_style`：典型表达样本
   用于体现用户日常语言风格、表达完整度、语气和表达习惯。

2. `emotion_representative`：情绪代表样本
   用于体现用户主要情绪倾向或强情绪表达。

3. `topic_representative`：主题代表样本
   用于体现用户长期关注的主要主题领域。

4. `public_issue`：公共议题样本
   用于体现用户是否参与社会公共事件、政策民生、时事政治等议题。

5. `repost_behavior`：传播行为样本
   用于体现用户转发扩散、媒体依赖、政务/机构信息源依赖、KOL 敏感等传播行为。

6. `high_engagement`：高互动样本
   用于体现用户影响力或被关注程度。该类型不是必须，只有在互动指标明显较高时使用。

如果同一条微博同时符合多个类型，请选择最主要的类型。

## 八、核心用户与普通用户的差异化策略

请根据已有画像字段，为用户划分记忆样本筛选级别。可以先采用简单规则，后续再优化。

建议划分为：

### 1. 核心用户 `core`

满足以下条件之一即可考虑为核心用户：

* `profile_reliability` 较高，且 `final_topic_profile_reliability` 较高；
* `propagation_activity_level` 较高；
* `influence_level` 较高；
* `weibo_hq_count` 较高；
* `final_public_issue_topic_ratio` 较高；
* `repost_with_comment_ratio` 较高；
* `strong_emotion_ratio` 较高；
* `propagation_role` 属于较有建模价值的角色，例如原创表达者、转发评论者、KOL 敏感型用户、媒体信息跟随者、潜在影响者等。

核心用户建议选取 6 条记忆样本。

建议配额：

* 1 条 `typical_style`
* 1 条 `emotion_representative`
* 2 条 `topic_representative`
* 1 条 `public_issue`，如果没有合适样本，则用高可信主题样本替代
* 1 条 `repost_behavior`，如果用户转发倾向不明显，则用 `high_engagement` 或 `typical_style` 替代

### 2. 普通用户 `normal`

普通用户建议选取 3 条记忆样本。

建议配额：

* 1 条 `typical_style`
* 1 条 `topic_representative`
* 1 条 `emotion_representative` 或 `repost_behavior`

### 3. 背景用户 `background`

对于历史微博较少、画像可靠性较低、可分析文本不足的用户，可以不强制选取真实记忆样本，或只选 1 条最可靠样本。此类用户后续可更多依赖用户级画像摘要或用户原型参与仿真。

## 九、记忆样本评分

请为候选微博计算 `memory_score`，并保留若干子分数，方便调试与人工审查。

建议子分数包括：

1. `quality_score`

   * 根据 `text_quality`、文本长度、是否为空、是否过长等计算。
   * 文本长度建议优先选择 15 到 100 字之间的内容。
   * 太短或太长都应降权。

2. `emotion_score`

   * 根据 `emotion_intensity_score`、`model_confidence`、`sentiment_label_en` 与用户级情绪画像计算。
   * 如果微博情绪方向与用户 `dominant_emotion` 一致，可适当加分。
   * 如果用户 `polarity_std` 较高，可允许选择不同情绪方向的代表样本。

3. `topic_score`

   * 根据 `final_topic_confidence`、`final_topic_categories`、`final_topic_labels` 与用户 `top_categories`、`top_implicit_topic_labels`、`final_category_distribution` 的匹配程度计算。
   * 属于用户高频主题的微博应加分。

4. `public_issue_score`

   * 如果微博主题属于社会公共事件、政策民生、时事政治等公共议题类别，则加分。
   * 如果用户 `final_public_issue_topic_ratio` 较高，则公共议题样本应进一步加分。

5. `propagation_score`

   * 对带转发评语的转发微博加分。
   * 如果用户 `repost_ratio`、`repost_with_comment_ratio`、`media_dependency_score`、`kol_sensitivity_score` 较高，则相应传播行为样本加分。
   * 没有转发评语的转发微博不建议作为语言风格样本。

6. `engagement_score_norm`

   * 根据互动量或已有互动指标归一化。
   * 高互动样本可加分，但不要让互动量完全主导记忆样本选择。

7. `style_score`

   * 根据文本长度、标点、是否包含完整表达、是否包含观点性表达等启发式规则估计。
   * 不需要复杂 NLP，只需尽量避免选择过短、碎片化、无观点的文本。

8. `diversity_penalty`

   * 如果已经选中的样本中存在相同主题、相同情绪、相同类型或高度相似文本，则降低后续相似样本的得分。

总分可以先采用加权和，权重写入配置，便于后续调整。例如：

```python
memory_score = (
    0.20 * quality_score
    + 0.20 * topic_score
    + 0.20 * emotion_score
    + 0.15 * style_score
    + 0.10 * public_issue_score
    + 0.10 * propagation_score
    + 0.05 * engagement_score_norm
    - diversity_penalty
)
```

以上权重只是初始建议，请实现为可配置常量，不要硬编码在多处。

## 十、样本选择流程

请实现如下流程：

1. 读取并合并微博级情绪结果与微博级主题结果。
2. 合并用户级情绪画像、主题画像和传播画像。
3. 为每个用户判断 `memory_user_level`，取值为 `core`、`normal`、`background`。
4. 为每个用户构建候选池。
5. 为候选微博计算各项子分数和 `memory_score`。
6. 按用户级别确定目标样本数与记忆类型配额。
7. 按配额优先选择样本：

   * 先选 `typical_style`
   * 再选 `topic_representative`
   * 再选 `emotion_representative`
   * 再选 `public_issue`
   * 再选 `repost_behavior`
   * 最后用综合得分最高的样本补齐
8. 每次选择样本后，进行去重和多样性控制：

   * 避免同一用户选中重复文本。
   * 避免同一用户样本全部来自同一主题。
   * 避免同一用户样本全部为同一情绪方向。
   * 避免样本全部为原创或全部为转发，除非该用户画像本身高度偏向某一类。
9. 如果某类配额没有合适候选样本，可以用其他高分样本补齐，并在 `selection_reason` 中说明。
10. 输出最终记忆样本表和日志统计。

## 十一、输出设计

请输出微博级记忆样本表，建议字段如下：

```python
[
    'user_id',
    'weibo_id',
    'memory_user_level',
    'memory_type',
    'content_for_agent',
    'source_context_for_agent',
    'is_repost',
    'has_repost_comment',
    'sentiment_label',
    'polarity_score',
    'emotion_intensity_score',
    'model_confidence',
    'final_topic_categories',
    'final_topic_labels',
    'topic_signal_source',
    'final_topic_confidence',
    'source_author_type',
    'engagement_score',
    'quality_score',
    'emotion_score',
    'topic_score',
    'public_issue_score',
    'propagation_score',
    'engagement_score_norm',
    'style_score',
    'diversity_penalty',
    'memory_score',
    'selection_reason'
]
```

其中：

* `content_for_agent`：最终给 Agent 大模型看的文本。

  * 原创微博：使用用户微博内容。
  * 带评语转发：优先使用用户自己的转发评语或用户可分析文本。
  * 不建议直接放入过长的源微博全文。

* `source_context_for_agent`：源微博或源作者的简要上下文。

  * 如果是转发微博，可以记录源作者类型、源微博主题、源微博话题标签等。
  * 不要存放过长的源微博原文。
  * 如果没有源微博上下文，则为空。

* `selection_reason`：用规则生成一句简短中文说明，解释为什么选择该样本。例如：

  * “该微博为用户高频主题下的原创表达，文本完整，适合作为典型表达样本。”
  * “该微博情绪强度较高，且情绪方向与用户长期情绪画像一致，适合作为情绪代表样本。”
  * “该微博为带评语转发，能够体现用户的转发扩散行为和信息源偏好。”

输出路径可位于 `scripts\weibo_analysis\memory_sample\` 路径下，视情况可生成一个或多个 Python 脚本文件实现功能。

## 十二、实现要求

1. 使用 Python 实现。
2. 优先使用 pandas 完成数据处理。
3. 保持与项目现有代码风格一致。
4. 所有路径、阈值、样本数量、评分权重应集中配置。
5. 对关键步骤添加日志输出。
6. 对缺失字段、空值、异常类型进行稳健处理。
7. 不要因为个别用户数据异常导致整个流程中断。
8. 输出结果建议保存为 parquet。
9. 添加必要的统计日志，例如：

   * 总用户数
   * core / normal / background 用户数
   * 成功选出记忆样本的用户数
   * 平均每用户样本数
   * 各 memory_type 的样本数量
   * 原创 / 转发样本比例
   * 主题覆盖情况
   * 情绪标签分布
10. 如项目中已有通用工具函数、路径配置、日志配置，请复用，不要重复造轮子。
11. 在修改完成后，请简单检查脚本是否能够运行，并说明输出文件位置。

## 十三、注意事项

1. 不需要按事件时间过滤微博年份。
2. 记忆样本应优先服务于长期用户画像，而不是严格事件回放。
3. 不要让高互动量完全主导样本选择。
4. 不要只选强情绪微博，否则会扭曲用户长期风格。
5. 不要只选主题最集中的微博，也要保留一定表达风格和情绪代表性。
6. 对低可靠用户可以减少样本数，不必强行凑满。
7. 代码应便于后续将记忆样本接入 Agent Prompt。

## 十四、期望最终结果

完成后，我希望得到：

1. 一个可运行的记忆样本筛选脚本。
2. 一份微博级记忆样本结果表。
3. 可选的一份用户级记忆摘要表。
4. 简要的运行统计输出。
5. 代码中清晰的配置项，方便后续调整样本数量、权重和筛选阈值。