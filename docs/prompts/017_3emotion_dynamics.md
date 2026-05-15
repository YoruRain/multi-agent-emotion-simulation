你现在需要在现有“单事件多轮互动仿真系统”的基础上，实现第三阶段扩展：情绪传染与立场演化。

请注意：
本阶段的目标是在第二阶段已经生成的 Agent 互动结构基础上，新增一个独立、可解释、可配置的状态更新模块。

当前系统已经完成：

1. 第一阶段：
   - AgentState 多轮状态记录；
   - 第 0 轮初始化；
   - 第 1 到 N 轮保存每个 Agent 的状态；
   - 输出 agent_states_by_round.csv、round_metrics.csv、active_reactions.jsonl。

2. 第二阶段：
   - KOL / 高影响力 Agent 优先发声；
   - 普通 Agent 观察上下文评论后响应；
   - 生成 source_agent -> target_agent 的互动边；
   - 输出 interactions.csv；
   - 输出 network.graphml；
   - round_metrics.csv 已包含 interaction_count 等互动统计字段。

第三阶段要做的是：

基于 interactions.csv / 当前轮 InteractionRecord：
source_agent 的 emotion_score / stance_score
→ 通过 interaction weight 影响 target_agent
→ 结合 target_agent 自身易感性、事件刺激、自身反应
→ 更新 target_agent 的 emotion_score / stance_score
→ 输出每轮情绪传染与立场演化指标。

## 一、本阶段重要限制

请严格控制任务范围：

1. 不要重写 MultiRoundSimulator。
2. 不要重写 InteractionEngine。
3. 不要改变 interactions.csv 的基本含义。
4. 不要引入复杂机器学习模型。
5. 不要引入语义相似度模型。
6. 不要强制调用 LLM。
7. 必须支持 use_llm=false 的稳定 demo。
8. 本阶段的情绪传染和立场演化使用规则模型即可。
9. 所有随机过程必须受 seed 控制。
10. 所有分数必须裁剪到 [-1, 1]。
11. 如果某一轮没有 interactions，也不能报错。
12. 输出结果必须便于后续 Streamlit 可视化读取。

本阶段的目标是：
“让互动边真正影响 Agent 状态，并输出可解释的情绪/立场演化结果。”

## 二、请先阅读现有代码

请先阅读当前多轮仿真相关文件：

- scope/src/simulation/agent_state.py
- scope/src/simulation/multiround_config.py
- scope/src/simulation/multiround_simulator.py
- scope/src/simulation/interaction_engine.py
- scope/src/simulation/interaction_schema.py
- scope/src/simulation/network_builder.py
- scope/src/simulation/multiround_analyzer.py
- scope/run_multiround_simulation.py
- scope/docs/multiround_simulation.md

请优先复用已有代码结构，不要把所有逻辑塞进 multiround_simulator.py。

此外，系统的输入文件可以从 scope\data\inputs 路径下读取。

## 三、建议新增或修改的文件
建议新增：

- scope/src/simulation/emotion_dynamics.py

建议修改：

- scope/src/simulation/agent_state.py
- scope/src/simulation/multiround_config.py
- scope/src/simulation/multiround_simulator.py
- scope/src/simulation/multiround_analyzer.py
- scope/run_multiround_simulation.py
- scope/docs/multiround_simulation.md

如果已有类似模块，请优先复用。

## 四、统一情绪标签与状态分数映射

当前项目中的 Reaction 层 EmotionLabel 为：

EmotionLabel = Literal[
    "anger",
    "sadness",
    "fear",
    "joy",
    "disgust",
    "disappointment",
    "surprise",
    "sympathy",
    "confusion",
    "admiration",
    "mixed"
]

请注意区分两层概念：

1. Reaction 层：
   使用细粒度 emotion_label，例如 anger、joy、mixed、disappointment。

2. AgentState 层：
   使用连续 emotion_score 表示情绪状态，范围为 [-1, 1]。
   可以使用粗粒度 emotion_state_label：
   - positive
   - neutral
   - negative

请不要把 Reaction 层的 mixed、joy 等直接当作状态层标签。

请在 emotion_dynamics.py 中实现：

normalize_emotion_label(label: str) -> str

要求：
- 转为小写；
- 去除空白；
- 如果遇到 unknown / none / 空值，返回 mixed；
- 如果遇到旧标签 positive，可以映射为 joy 兼容；
- 如果遇到旧标签 neutral，可以映射为 mixed 或 neutral 兼容；
- 如果遇到旧标签 negative，可以映射为 disappointment 兼容。

## 五、实现情绪分数映射函数

请在 emotion_dynamics.py 中实现：

emotion_label_to_score(emotion_label: str, intensity: int | float = 1) -> float

建议基础映射如下：

- anger: -0.85
- disgust: -0.80
- fear: -0.70
- sadness: -0.70
- disappointment: -0.65
- confusion: -0.25

- mixed: 0.00
- surprise: 0.00

- sympathy: 0.35
- admiration: 0.70
- joy: 0.75

intensity 处理：
- intensity <= 0: 返回 0
- intensity == 1: 使用基础值
- intensity >= 2: 基础值乘以 1.15，但最终裁剪到 [-1, 1]

注意：
mixed 和 surprise 暂时设为 0，因为它们不一定代表明确正负方向。
后续如果要结合事件语境调整，可以再扩展。

## 六、实现立场分数映射函数

请在 emotion_dynamics.py 中实现：

stance_label_to_score(stance_label: str, intensity: int | float = 1) -> float

建议映射：

- favor:
  - intensity 0: 0
  - intensity 1: +0.50
  - intensity 2: +0.85

- against:
  - intensity 0: 0
  - intensity 1: -0.50
  - intensity 2: -0.85

- neutral / unclear / mixed:
  - 0


## 七、实现状态标签函数

请实现：

score_to_emotion_state_label(score: float) -> str

规则：
- score >= 0.25: positive
- -0.25 < score < 0.25: neutral
- score <= -0.25: negative

请实现：

score_to_stance_state_label(score: float) -> str

规则：
- score >= 0.25: support
- -0.25 < score < 0.25: neutral
- score <= -0.25: against

## 八、实现事件刺激分数

请实现：

build_event_influence_scores(event: dict) -> dict

返回：

{
  "event_emotion_score": float,
  "event_stance_score": float,
  "event_emotion_reason": str,
  "event_stance_reason": str
}

event_emotion_score 计算规则：

优先读取 event["dominant_emotion_label"]。

使用 emotion_label_to_score() 转换。

event_stance_score 计算规则：

优先读取 event["dominant_stance_label"]。

- anger: -0.85
- disgust: -0.80
- fear: -0.70
- sadness: -0.70
- disappointment: -0.65
- confusion: -0.25

- mixed: 0.00
- surprise: 0.00

- sympathy: 0.35
- admiration: 0.70
- joy: 0.75

如果 dominant_stance_label 缺失，可以从 event_stance_focus 简单判断，但不要做复杂 NLP。

注意：
事件刺激分数不要太大，否则会压过 Agent 个体差异。

## 九、实现邻居影响聚合

请在 emotion_dynamics.py 中实现：

aggregate_neighbor_influence(
    target_agent_id: str,
    interactions_for_round: list[InteractionRecord | dict],
    state_by_agent: dict[str, AgentState],
) -> dict

返回：

{
  "neighbor_emotion_score": float,
  "neighbor_stance_score": float,
  "neighbor_influence_weight_sum": float,
  "neighbor_count": int,
  "high_influence_neighbor_count": int,
  "media_neighbor_count": int,
  "kol_neighbor_count": int,
}

计算逻辑：

1. 找出当前轮所有 target_agent_id == 当前 Agent 的互动边。
2. 对每条边，取 source_agent_id 的当前轮或上一轮状态。
3. 使用 interaction.weight 作为基础权重。
4. 对 source_agent 的 emotion_score 做加权平均：
   weighted_neighbor_emotion =
       sum(source_emotion_score * weight) / sum(weight)
5. 对 source_agent 的 stance_score 做加权平均：
   weighted_neighbor_stance =
       sum(source_stance_score * weight) / sum(weight)
6. 如果没有邻居影响：
   - neighbor_emotion_score = 0
   - neighbor_stance_score = 0
   - neighbor_influence_weight_sum = 0
   - neighbor_count = 0

注意：
本阶段不要重新生成 interactions，只消费第二阶段已经生成的互动边或当前轮 InteractionRecord。

## 十、实现情绪传染与立场演化公式

请实现核心函数：

update_agent_state_with_dynamics(
    old_state: AgentState,
    own_reaction: dict | None,
    neighbor_influence: dict,
    event_influence: dict,
    config: MultiRoundSimulationConfig,
) -> AgentState

情绪更新公式：

new_emotion =
    self_retention * old_emotion
    + social_influence_strength * susceptibility_score * neighbor_emotion_score
    + event_influence_strength * event_emotion_score
    + reaction_influence_strength * own_reaction_emotion_score

立场更新公式：

new_stance =
    stance_retention * old_stance
    + social_stance_strength * susceptibility_score * neighbor_stance_score
    + event_stance_strength * event_stance_score
    + reaction_stance_strength * own_reaction_stance_score

默认参数：

情绪：
- self_retention = 0.65
- social_influence_strength = 0.25
- event_influence_strength = 0.10
- reaction_influence_strength = 0.15

立场：
- stance_retention = 0.75
- social_stance_strength = 0.20
- event_stance_strength = 0.10
- reaction_stance_strength = 0.15

其中：

old_emotion = old_state.emotion_score
old_stance = old_state.stance_score

susceptibility_score = old_state.susceptibility_score

neighbor_emotion_score 来自 aggregate_neighbor_influence()
neighbor_stance_score 来自 aggregate_neighbor_influence()

event_emotion_score 来自 build_event_influence_scores()
event_stance_score 来自 build_event_influence_scores()

own_reaction_emotion_score:
- 如果 own_reaction 存在，使用 emotion_label_to_score(own_reaction["emotion_label"], own_reaction["emotion_intensity"])
- 如果 own_reaction 不存在，设为 0

own_reaction_stance_score:
- 如果 own_reaction 存在，使用 stance_label_to_score(own_reaction["stance_label"], own_reaction["stance_intensity"])
- 如果 own_reaction 不存在，设为 0

最后：
- new_emotion 裁剪到 [-1, 1]
- new_stance 裁剪到 [-1, 1]

## 十一、加入边界阻尼，避免分数过快撞到 -1 或 1

请加入可配置的饱和阻尼机制，避免 Agent 情绪或立场过快到达边界。

建议实现：

apply_saturation_damping(old_score, raw_new_score, damping_strength=0.5) -> float

逻辑：

delta = raw_new_score - old_score
damping_factor = 1 - damping_strength * abs(old_score)
adjusted_delta = delta * damping_factor
new_score = old_score + adjusted_delta

然后裁剪到 [-1, 1]。

默认：
- enable_saturation_damping = True
- saturation_damping_strength = 0.5

说明：
当 old_score 接近 -1 或 1 时，状态变化会变慢；当 old_score 接近 0 时，变化比较自由。

## 十二、状态更新解释 reason

请为每个 Agent 每轮生成简短中文 update_reason。

请实现：

build_dynamics_update_reason(
    old_state,
    new_state,
    neighbor_influence,
    event_influence,
    own_reaction,
) -> str

规则示例：

1. 如果 neighbor_count > 0 且 emotion_delta < -0.05：
   “受评论区负向观点影响，情绪分数下降”

2. 如果 neighbor_count > 0 且 emotion_delta > 0.05：
   “受较积极的上下文评论影响，情绪分数上升”

3. 如果 stance_delta < -0.05：
   “受邻近评论和自身表达影响，立场更偏反对”

4. 如果 stance_delta > 0.05：
   “受邻近评论和自身表达影响，立场更偏支持”

5. 如果 abs(emotion_delta) < 0.03 且 abs(stance_delta) < 0.03：
   “本轮状态变化较小，整体保持稳定”

6. 如果 neighbor_count == 0 且 own_reaction 存在：
   “本轮主要根据自身表达更新状态”

7. 如果 neighbor_count == 0 且 own_reaction 不存在：
   “本轮未参与互动，状态主要保持稳定”

注意：
reason 不要太长，控制在 30 字以内。

## 十三、修改 AgentState 输出字段

请确保 agent_states_by_round.csv 中包含以下字段：

原有字段继续保留：

- run_id
- event_id
- agent_id
- user_id
- round_id
- memory_user_level
- verified_type_name
- propagation_role
- influence_score
- susceptibility_score
- activity_score
- kol_sensitivity_score
- media_dependency_score
- repost_tendency_score
- emotion_score
- stance_score
- emotion_label
- stance_label
- is_active
- last_action_type
- last_reaction_text
- state_update_reason

第三阶段新增或确保存在：

- old_emotion_score
- new_emotion_score
- emotion_delta
- old_stance_score
- new_stance_score
- stance_delta
- neighbor_emotion_score
- neighbor_stance_score
- neighbor_influence_weight_sum
- neighbor_count
- high_influence_neighbor_count
- event_emotion_score
- event_stance_score
- own_reaction_emotion_score
- own_reaction_stance_score
- dynamics_enabled

要求：
1. 第 0 轮这些动态字段可以为空或 0。
2. 第 1 到 N 轮必须有值。
3. emotion_score 应等于 new_emotion_score。
4. stance_score 应等于 new_stance_score。
5. emotion_delta = new_emotion_score - old_emotion_score。
6. stance_delta = new_stance_score - old_stance_score。

## 十四、修改 MultiRoundSimulationConfig

请在 multiround_config.py 中新增参数：

- enable_emotion_dynamics: bool = False

情绪动态参数：

- self_retention: float = 0.65
- social_influence_strength: float = 0.25
- event_influence_strength: float = 0.10
- reaction_influence_strength: float = 0.15

立场动态参数：

- stance_retention: float = 0.75
- social_stance_strength: float = 0.20
- event_stance_strength: float = 0.10
- reaction_stance_strength: float = 0.15

阻尼参数：

- enable_saturation_damping: bool = True
- saturation_damping_strength: float = 0.5

其他参数：

- min_delta_threshold_for_reason: float = 0.03

要求：
1. enable_emotion_dynamics=False 时，保留第二阶段原有状态更新逻辑。
2. enable_emotion_dynamics=True 时，启用第三阶段动态更新。
3. 所有参数写入 config.json。
4. 不要破坏已有 --enable-interactions 的行为。

## 十五、修改命令行参数

请修改 scope/run_multiround_simulation.py，新增参数：

- --enable-emotion-dynamics
- --self-retention
- --social-influence-strength
- --event-influence-strength
- --reaction-influence-strength
- --stance-retention
- --social-stance-strength
- --event-stance-strength
- --reaction-stance-strength
- --disable-saturation-damping
- --saturation-damping-strength

示例命令：

python scope/run_multiround_simulation.py \
  --event-id event_5177192956301027 \
  --max-agents 30 \
  --rounds 5 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3 \
  --enable-emotion-dynamics

要求：
1. 如果用户开启 --enable-emotion-dynamics 但没有开启 --enable-interactions，也要能运行。
   此时只使用事件刺激和自身反应，不使用邻居影响。
2. 推荐在日志中提示：
   “emotion dynamics enabled, neighbor influence disabled because interactions are disabled”
3. 如果同时开启 --enable-interactions 和 --enable-emotion-dynamics，则使用 interactions 进行邻居影响聚合。
4. 运行结束时打印：
   - run_id
   - output_dir
   - total_agents
   - rounds
   - interaction_count
   - final_avg_emotion_score
   - final_avg_stance_score
   - avg_abs_emotion_delta
   - avg_abs_stance_delta

## 十六、修改每轮仿真流程

请在 MultiRoundSimulator 中集成 emotion_dynamics。

在 interaction_mode = kol_first 且 enable_emotion_dynamics=True 时，每轮建议流程：

1. 从上一轮 AgentState 复制当前轮初始状态。
2. 选择 KOL speakers。
3. KOL speakers 生成 reaction。
4. 选择 regular agents。
5. regular agents 根据 context_comments 生成 reaction。
6. 生成 interactions。
7. 对本轮每个 Agent 聚合 neighbor_influence。
8. 对本轮每个 Agent 调用 update_agent_state_with_dynamics()。
9. 保存本轮 AgentState。
10. 计算 round_metrics。

注意：
对于没有参与的 Agent：
- own_reaction = None；
- 如果它没有入边，则主要保持稳定；
- 如果它被 observe 影响，可以根据 interactions 受到轻微邻居影响；
- 但当前第二阶段通常只给 regular_agent 生成 target 入边，因此未参与 Agent 多数没有邻居影响，这是可以接受的。

对于 KOL speaker：
- 通常没有入边；
- 主要根据自身 reaction 和事件刺激更新。

对于 regular_agent：
- 有 context_comments 和 interactions；
- 受到邻居影响、事件刺激和自身 reaction 共同作用。

## 十七、round_metrics.csv 增强

请在 multiround_analyzer.py 中增强 round_metrics。

在原有字段基础上新增：

- emotion_volatility
- stance_volatility
- avg_abs_emotion_delta
- avg_abs_stance_delta
- max_abs_emotion_delta
- max_abs_stance_delta
- dominant_emotion_state
- dominant_stance_state
- polarization_score
- avg_neighbor_count
- agents_affected_by_neighbors
- avg_neighbor_influence_weight
- dynamics_enabled

定义：

emotion_volatility：
本轮 emotion_score 的标准差。

stance_volatility：
本轮 stance_score 的标准差。

avg_abs_emotion_delta：
本轮 abs(emotion_delta) 平均值。

avg_abs_stance_delta：
本轮 abs(stance_delta) 平均值。

max_abs_emotion_delta：
本轮 abs(emotion_delta) 最大值。

max_abs_stance_delta：
本轮 abs(stance_delta) 最大值。

dominant_emotion_state：
positive / neutral / negative 中数量最多的标签。

dominant_stance_state：
support / neutral / against 中数量最多的标签。

polarization_score：
stance_score 的标准差。
如果后续想扩展，可以再引入更复杂极化指标；本阶段用标准差即可。

avg_neighbor_count：
本轮 Agent 平均 neighbor_count。

agents_affected_by_neighbors：
本轮 neighbor_count > 0 的 Agent 数量。

avg_neighbor_influence_weight：
本轮 neighbor_influence_weight_sum 的平均值。

dynamics_enabled：
当前是否启用情绪动态。

要求：
1. enable_emotion_dynamics=False 时，这些字段也可以存在，值为 0 或基于已有状态计算。
2. round_metrics.csv 不应因为字段缺失报错。

## 十八、输出 dynamics_summary.json

请在每次运行输出目录下新增：

dynamics_summary.json

内容建议包括：

- run_id
- event_id
- topic
- dynamics_enabled
- interaction_enabled
- total_agents
- rounds
- initial_avg_emotion_score
- final_avg_emotion_score
- emotion_score_change
- initial_avg_stance_score
- final_avg_stance_score
- stance_score_change
- final_emotion_distribution
- final_stance_distribution
- max_emotion_volatility
- max_stance_volatility
- final_polarization_score
- total_interaction_count
- agents_ever_affected_by_neighbors
- parameter_config

这个文件用于后续可视化和验收说明。

## 十九、文档更新

请更新：

scope/docs/multiround_simulation.md

新增第三阶段说明：

1. 本阶段实现了什么：
   - 根据互动边聚合邻居情绪和立场；
   - 根据 Agent 易感性计算社会影响；
   - 融合自身保持、邻居影响、事件刺激和自身表达；
   - 更新 Agent 的 emotion_score 和 stance_score；
   - 输出情绪/立场演化指标。

2. 情绪更新公式：

new_emotion =
    self_retention * old_emotion
    + social_influence_strength * susceptibility_score * neighbor_emotion_score
    + event_influence_strength * event_emotion_score
    + reaction_influence_strength * own_reaction_emotion_score

3. 立场更新公式：

new_stance =
    stance_retention * old_stance
    + social_stance_strength * susceptibility_score * neighbor_stance_score
    + event_stance_strength * event_stance_score
    + reaction_stance_strength * own_reaction_stance_score

4. 标签映射说明：
   - Reaction 层使用细粒度 emotion_label；
   - State 层使用连续 emotion_score；
   - 展示层可归纳为 positive / neutral / negative。

5. 参数说明：
   - self_retention
   - social_influence_strength
   - event_influence_strength
   - reaction_influence_strength
   - stance_retention
   - social_stance_strength
   - event_stance_strength
   - reaction_stance_strength
   - saturation_damping_strength

6. 当前限制：
   - 情绪传染为简化规则模型；
   - 未使用真实语义相似度；
   - 未进行真实世界预测；
   - 主要用于毕业设计原型展示和机制验证。

7. 运行示例：

python scope/run_multiround_simulation.py \
  --event-id event_5177192956301027 \
  --max-agents 30 \
  --rounds 5 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3 \
  --enable-emotion-dynamics

## 二十、测试要求

请完成以下测试或手动验证。

### 测试 1：不开启 dynamics，兼容第二阶段

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5177192956301027 \
  --max-agents 20 \
  --rounds 3 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first

确认：
- 程序仍能运行；
- interactions.csv 存在；
- network.graphml 存在；
- round_metrics.csv 存在；
- 不启用 dynamics 时不破坏第二阶段结果。

### 测试 2：开启 dynamics

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5177192956301027 \
  --max-agents 30 \
  --rounds 5 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3 \
  --enable-emotion-dynamics

确认输出目录中存在：

- config.json
- selected_event.json
- agent_initial_states.csv
- agent_states_by_round.csv
- active_reactions.jsonl
- interactions.csv
- network.graphml
- round_metrics.csv
- dynamics_summary.json

确认：

1. agent_states_by_round.csv 中包含：
   - neighbor_emotion_score
   - neighbor_stance_score
   - neighbor_count
   - event_emotion_score
   - event_stance_score
   - own_reaction_emotion_score
   - own_reaction_stance_score
   - dynamics_enabled

2. 第 1 到 N 轮中：
   - emotion_score 等于 new_emotion_score
   - stance_score 等于 new_stance_score
   - emotion_delta = new_emotion_score - old_emotion_score
   - stance_delta = new_stance_score - old_stance_score

3. 所有 emotion_score 和 stance_score 都在 [-1, 1]。

4. round_metrics.csv 中包含：
   - avg_abs_emotion_delta
   - avg_abs_stance_delta
   - polarization_score
   - agents_affected_by_neighbors

5. dynamics_summary.json 可以被 json.load 正常读取。

### 测试 3：无 interaction 但开启 dynamics

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5177192956301027 \
  --max-agents 10 \
  --rounds 3 \
  --use-llm false \
  --seed 42 \
  --enable-emotion-dynamics

确认：
- 程序不报错；
- neighbor_count 基本为 0；
- 状态仍可根据事件刺激和自身反应更新；
- 文档和日志说明未启用 interactions，因此没有邻居影响。


## 二十二、最终交付说明

完成后请给出：

1. 新增了哪些文件。
2. 修改了哪些文件。
3. 情绪分数映射规则。
4. 立场分数映射规则。
5. 情绪传染公式。
6. 立场演化公式。
7. 如何运行不开启 dynamics 的兼容测试。
8. 如何运行开启 dynamics 的完整 demo。
9. 输出文件说明。
10. 当前实现了什么。
11. 当前还没有实现什么。
12. 建议的 git commit message。

建议 commit message：

feat: add emotion dynamics for multiround simulation

本阶段最终目标：
在现有多轮互动仿真基础上，根据 interactions.csv 中的 source_agent -> target_agent 候选影响边，实现简化的情绪传染与立场演化规则，使 Agent 状态能够在多轮评论区互动中随邻居影响、事件刺激和自身表达发生变化。