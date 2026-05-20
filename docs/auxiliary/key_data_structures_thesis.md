# 3.5 关键数据结构设计

本系统围绕微博热点事件的群体情绪与立场演化模拟，设计了事件数据、用户画像、智能体状态和模拟结果四类关键数据结构。四类结构分别对应模拟链路中的外部事件输入、Agent 长期属性建模、运行时状态更新和结果分析输出，共同支撑从原始微博数据到多轮群体仿真的转换过程。

## 3.5.1 事件数据结构

事件数据结构用于描述一次微博热点事件及其评论区总体语境，是模拟系统的外部刺激源。该结构在模拟过程中保持只读，主要为 Agent Prompt 构造、初始状态生成和情绪动态更新提供事件背景、总体情绪倾向和主要立场焦点。

事件数据不追求完整保存原始微博与评论数据，而是将事件主题、背景摘要、评论区情绪分布和立场分布压缩为可供 Agent 理解和模拟器计算的结构化输入。

| 字段类别 | 代表字段 | 设计作用 |
| --- | --- | --- |
| 标识字段 | `event_id`、`weibo_id` | 用于事件唯一标识和原始微博数据回溯。 |
| 语义字段 | `topic`、`event_context`、`event_type` | 描述事件主题、事件背景和事件类型，为 Prompt 提供事实边界。 |
| 情绪字段 | `event_emotion_tendency`、`dominant_emotion_label`、`emotion_distribution` | 表示评论区总体情绪倾向和主导情绪，用于反应生成和事件刺激计算。 |
| 立场字段 | `event_stance_focus`、`dominant_stance_label`、`stance_distribution` | 表示评论区主要评价焦点和立场分布，用于初始化立场和动态更新。 |

简化后的事件记录示例如下：

```json
{
  "event_id": "event_5177192956301027",
  "weibo_id": "5177192956301027",
  "topic": "官方通报罗帅宇事件",
  "event_context": "官方通报罗帅宇事件，维持不予立案决定……",
  "event_type": "public_issue",
  "event_emotion_tendency": "negative",
  "dominant_emotion_label": "disgust",
  "event_stance_focus": "评论区主要围绕相关行为是否合理展开评价。",
  "dominant_stance_label": "against",
  "emotion_distribution": {
    "negative": 0.9169,
    "positive": 0.0,
    "neutral": 0.0692
  },
  "stance_distribution": {
    "favor": 0.0304,
    "against": 0.9179,
    "neutral": 0.0386
  }
}
```

在系统链路中，事件数据首先被加载为本次模拟的事件输入；随后，事件背景和评论区统计信息被组织进 Agent 的用户消息中，帮助 Agent 生成符合事件语境的反应。同时，主导情绪和主导立场会被转化为事件层面的情绪刺激与立场刺激，参与多轮状态更新。

## 3.5.2 用户画像数据结构

用户画像数据结构用于将微博用户的长期行为特征、情绪倾向、主题偏好和传播习惯压缩为 Agent 可用的建模输入。相比原始用户信息表，用户画像更强调“可模拟性”：它需要能够支持 Agent 初始化、Prompt 渲染、参与倾向估计、互动权重计算和分组分析。

用户画像由结构化画像、历史记忆片段和系统提示词共同构成。其中，结构化画像提供长期属性和数值参数；历史记忆片段提供用户过往表达样本；系统提示词则将长期画像压缩为 Agent 可理解的角色设定。

| 字段类别 | 代表字段 | 设计作用 |
| --- | --- | --- |
| 身份属性字段 | `agent_id`、`user_id`、`verified_type_name`、`user_level` | 标识 Agent 与原始用户，并提供用户层级、认证类型等身份信息。 |
| 情绪画像字段 | `emotion_summary`、`pos_ratio`、`neg_ratio` | 表示用户长期情绪倾向，用于初始化情绪状态。 |
| 主题画像字段 | `topic_summary`、`public_issue_topic_ratio` | 描述用户长期关注主题，影响公共事件中的参与倾向和初始立场强度。 |
| 传播画像字段 | `propagation_role`、`propagation_summary`、`influence_score`、`repost_ratio`、`media_dependency_score`、`kol_sensitivity_score` | 描述用户传播角色、影响力、转发倾向以及对媒体和 KOL 的敏感度。 |
| 记忆与提示词字段 | `memories[].content`、`sys_prompt` | 为 Agent 提供历史表达样本和角色扮演约束。 |

简化后的用户画像记录示例如下：

```json
{
  "agent_id": "weibo_user_1008520675",
  "user_id": "1008520675",
  "user_level": "background",
  "verified_type_name": "普通用户",
  "emotion_summary": "暂无明确长期情绪倾向。",
  "pos_ratio": 0.0,
  "neg_ratio": 0.0,
  "topic_summary": "该用户主要关注政策民生、娱乐文化和社会公共事件话题。",
  "public_issue_topic_ratio": 0.3043,
  "propagation_role": "转发扩散者",
  "influence_score": 0.0235,
  "repost_ratio": 1.0,
  "media_dependency_score": 0.25,
  "kol_sensitivity_score": 0.4083,
  "memories": [
    {
      "content": "历史微博样本：……"
    }
  ],
  "sys_prompt": "你正在扮演一个微博用户 Agent……"
}
```

在模拟过程中，用户画像首先被加载并合并为 Agent 输入记录。随后，画像中的情绪比例会转化为初始情绪分数，传播角色和用户层级会影响活跃度估计，影响力和敏感度参数会参与互动选择与互动权重计算。记忆片段和系统提示词则进入反应生成过程，使 Agent 的输出更接近该用户的长期表达风格。

## 3.5.3 智能体状态数据结构

智能体状态数据结构是多轮模拟中的核心动态结构，用于表示某个 Agent 在某一轮中的情绪、立场、行为和受影响状态。与用户画像不同，用户画像描述的是相对稳定的长期属性，而 AgentState 描述的是会随轮次不断变化的运行时状态。

第 0 轮状态由用户画像和事件数据共同初始化；第 1 到第 N 轮状态则由上一轮状态、本轮参与行为、邻居影响、事件刺激和自身表达共同更新。最终，所有轮次的状态数据会进入群体指标计算和可视化分析。

| 字段类别 | 代表字段 | 设计作用 |
| --- | --- | --- |
| 标识字段 | `run_id`、`event_id`、`agent_id`、`round_id` | 标识一次模拟运行中某个 Agent 在某一轮的状态快照。 |
| 静态引用字段 | `user_level`、`propagation_role`、`influence_score`、`activity_score`、`susceptibility_score` | 承接用户画像中的长期属性，用于参与选择、互动选择和影响权重计算。 |
| 情绪状态字段 | `emotion_score`、`emotion_label`、`emotion_delta` | 表示 Agent 当前情绪状态及其相对上一轮的变化。 |
| 立场状态字段 | `stance_score`、`stance_label`、`stance_delta` | 表示 Agent 当前立场状态及其相对上一轮的变化。 |
| 行为状态字段 | `is_active`、`last_action_type`、`last_reaction_text` | 记录 Agent 本轮是否参与、采取何种行为以及上一轮表达内容。 |
| 社会影响字段 | `neighbor_emotion_score`、`neighbor_stance_score`、`neighbor_count`、`neighbor_influence_weight_sum` | 聚合本轮可见邻居评论对目标 Agent 的潜在影响。 |
| 事件与表达影响字段 | `event_emotion_score`、`event_stance_score`、`own_reaction_emotion_score`、`own_reaction_stance_score` | 表示事件刺激和自身表达对状态更新的贡献。 |

简化后的状态记录示例如下：

```json
{
  "run_id": "multiround_20260517_001",
  "event_id": "event_5177192956301027",
  "agent_id": "weibo_user_1008520675",
  "round_id": 2,
  "user_level": "background",
  "propagation_role": "转发扩散者",
  "influence_score": 0.0235,
  "activity_score": 0.55,
  "susceptibility_score": 0.52,
  "emotion_score": -0.31,
  "emotion_label": "negative",
  "emotion_delta": -0.06,
  "stance_score": -0.28,
  "stance_label": "against",
  "stance_delta": -0.04,
  "is_active": true,
  "last_action_type": "comment",
  "last_reaction_text": "这事还是得继续追问清楚。",
  "neighbor_count": 2,
  "neighbor_influence_weight_sum": 0.37
}
```

AgentState 在系统中起到连接长期画像和短期演化过程的作用。一方面，它保留影响力、活跃度和易感性等由用户画像映射而来的稳定属性；另一方面，它持续记录每轮情绪、立场和行为变化。模拟器在每一轮结束后都会输出新的状态快照，并以这些状态为下一轮参与决策、互动选择和动态更新的基础。

## 3.5.4 模拟结果数据结构

模拟结果数据结构用于保存一次多轮模拟的配置、输入快照、Agent 状态、可见反应、互动关系和群体统计结果。与前三类结构相比，模拟结果更偏向分析和复现：它既要支持对单个 Agent 行为轨迹的追踪，也要支持对群体情绪、立场和互动网络的整体分析。

系统主要输出文件及其作用如下表所示。

| 输出文件 | 记录内容 | 设计作用 |
| --- | --- | --- |
| `config.json` | 运行配置 | 保存事件 ID、模拟轮数、Agent 数量、互动开关和动态参数，用于复现实验设置。 |
| `selected_event.json` | 事件快照 | 保存本次模拟实际使用的事件输入，避免上游事件数据变化影响结果解释。 |
| `agent_initial_states.csv` | 初始状态 | 记录第 0 轮 Agent 状态，用于观察画像到初始状态的映射结果。 |
| `agent_states_by_round.csv` | 各轮 Agent 状态 | 记录所有 Agent 在每一轮的情绪、立场和行为状态。 |
| `active_reactions.jsonl` | 可见反应 | 记录实际参与讨论的 Agent 在每一轮生成的行为和文本。 |
| `round_metrics.csv` | 群体统计指标 | 记录每轮参与率、平均情绪、平均立场、分布比例和波动指标。 |
| `interactions.csv` | 互动边 | 记录可见评论形成的 `source_agent -> target_agent` 候选影响关系。 |
| `dynamics_summary.json` | 动态汇总结果 | 汇总本次模拟初末状态变化、最终分布和整体动态指标。 |

其中，`active_reactions.jsonl` 主要回答“谁在第几轮说了什么”。它围绕 `run_id`、`event_id`、`round_id`、`agent_id`、`action_type`、`reaction_text`、`emotion_label` 和 `stance_label` 等字段组织，用于分析不同类型用户在事件讨论中的可见表达。

`agent_states_by_round.csv` 主要回答“每个 Agent 的状态如何变化”。它按轮记录每个 Agent 的情绪分数、立场分数、参与状态和社会影响信息，使研究者能够观察单个 Agent 从初始状态到最终状态的演化轨迹。

`round_metrics.csv` 主要回答“群体情绪、立场和参与情况如何演化”。该文件将个体状态聚合为群体层面指标，例如参与率、平均情绪分数、平均立场分数、情绪分布、立场分布、波动度和极化程度等。

`interactions.csv` 主要回答“哪些 Agent 之间形成了候选影响关系”。当系统启用互动机制时，高影响力用户或较早发声用户的评论会进入其他 Agent 的可见上下文，从而形成候选影响边。该文件记录影响来源、影响目标、互动权重和上下文排序等信息。

`dynamics_summary.json` 主要回答“本次模拟初末状态变化和整体动态结果如何”。它汇总初始与最终平均情绪、平均立场、最终状态分布、总互动数和动态参数，使研究者可以快速比较不同实验设置下的整体模拟结果。

综上，模拟结果数据结构将个体层面的反应与状态变化转化为可分析的群体指标和互动网络，为论文后续的实验分析、结果解释和可视化展示提供数据基础。
