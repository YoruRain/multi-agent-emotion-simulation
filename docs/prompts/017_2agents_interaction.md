你现在需要在现有“单事件多轮仿真骨架与 AgentState 状态系统”的基础上，实现第二阶段扩展：Agent 互动机制、KOL 先发声机制与互动边记录。

请注意：
本阶段只需要让多轮仿真从“每个 Agent 独立决定是否参与”扩展为“高影响力 Agent 先发声，普通 Agent 观察已有评论后再响应，并记录 Agent 之间的互动边”。

后续第三阶段会基于本阶段产生的 interactions.csv 实现情绪传染与立场演化。因此本阶段重点是：
1. 生成可追踪的评论上下文；
2. 记录 source_agent -> target_agent 的互动关系；
3. 输出 interactions.csv；
4. 输出基础 network.graphml；
5. 保持 demo 稳定运行。

## 一、当前项目状态

目前已经完成第一阶段多轮仿真骨架，已经具备以下能力：

1. 可以运行：

python scope/run_multiround_simulation.py \
  --event-id event_xxx \
  --max-agents 30 \
  --memory-user-level core \
  --rounds 5 \
  --use-llm false \
  --seed 42

2. 每次运行会输出到：

scope/data/outputs/simulation/multiround/{run_id}/

3. 已有输出文件包括：

- config.json
- selected_event.json
- agent_initial_states.csv
- agent_states_by_round.csv
- round_metrics.csv
- active_reactions.jsonl

4. 已经定义了 AgentState，包含：

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

请先阅读现有多轮仿真相关代码，尤其是：

- scope/src/simulation/agent_state.py
- scope/src/simulation/multiround_config.py
- scope/src/simulation/multiround_simulator.py
- scope/src/simulation/multiround_analyzer.py
- scope/run_multiround_simulation.py
- scope/docs/multiround_simulation.md

请优先复用现有代码，不要重写多轮仿真器。

## 二、本阶段重要限制

请严格控制本阶段范围：

1. 不要推翻第一阶段的 MultiRoundSimulator。
2. 不要破坏已有 use_llm=false fallback demo。
3. 不要在本阶段实现复杂情绪传染公式。
4. 不要在本阶段引入复杂语义相似度模型。
5. 不要让每个 Agent 每轮都强制调用 LLM。
6. 不要让网络分析过度复杂，本阶段只需要构建基础互动图。
7. 所有互动逻辑必须支持无 API key 环境运行。
8. 即使某一轮没有互动边，也不能让程序崩溃。
9. 输出文件必须便于第三阶段读取和复用。
10. 所有随机过程必须受 seed 控制。

本阶段可以对 Agent 状态做非常轻微的自身表达更新，但不要根据邻居影响真正改变 emotion_score 和 stance_score。邻居影响更新留到第三阶段。

## 三、建议新增或修改文件

建议新增：

- scope/src/simulation/interaction_engine.py
- scope/src/simulation/interaction_schema.py
- scope/src/simulation/network_builder.py

建议修改：

- scope/src/simulation/multiround_config.py
- scope/src/simulation/multiround_simulator.py
- scope/run_multiround_simulation.py
- scope/docs/multiround_simulation.md

如果已有类似文件，请优先复用，不要重复创建功能相同的模块。

## 四、实现 InteractionRecord 数据结构

请新增 scope/src/simulation/interaction_schema.py。

定义 InteractionRecord，可以使用 dataclass 或 Pydantic BaseModel。

字段至少包括：

基础字段：

- run_id: str
- event_id: str
- topic: str | None
- round_id: int

互动双方：

- source_agent_id: str
- target_agent_id: str
- source_user_id: str | None
- target_user_id: str | None

互动类型：

- interaction_type: str
- weight: float

上下文字段：

- source_action_type: str
- source_reaction_text: str
- target_action_type: str | None
- target_reaction_text: str | None
- context_rank: int | None

状态字段：

- source_emotion_score: float
- target_emotion_score_before: float
- target_emotion_score_after: float | None
- source_stance_score: float
- target_stance_score_before: float
- target_stance_score_after: float | None

画像字段：

- source_influence_score: float
- target_susceptibility_score: float
- target_kol_sensitivity_score: float
- target_media_dependency_score: float
- source_verified_type_name: str | None
- source_propagation_role: str | None
- target_propagation_role: str | None

其他字段：

- reason: str
- source: str
- created_at: str

interaction_type 取值建议：

- observe
- same_round_context
- reply
- repost
- influence_candidate

含义：

1. observe：
   target_agent 看到了 source_agent 的评论。

2. same_round_context：
   source_agent 的评论被放入 target_agent 本轮响应的上下文中。

3. reply：
   target_agent 对 source_agent 的内容形成回复或反驳。第一版可以用规则启发式判断。

4. repost：
   target_agent 的 action_type 是 repost 或 repost_with_comment，并且上下文来源是 source_agent。

5. influence_candidate：
   当前边是后续情绪传染和立场演化的候选影响边。本阶段只记录，不做正式状态传播。

注意：
本阶段可以主要使用 same_round_context 和 influence_candidate，reply / repost 可用简单规则补充。

## 五、实现 InteractionEngine

请新增 scope/src/simulation/interaction_engine.py。

实现 InteractionEngine，负责每一轮中的 Agent 发声顺序、上下文评论选择和互动边生成。

建议核心方法：

### 1. select_kol_speakers(agent_states, config) -> list[AgentState]

功能：
选择本轮优先发声的高影响力 Agent。

选择规则：
- influence_score 较高；
- activity_score 较高；
- propagation_role 包含 潜在影响者、KOL、高影响力、媒体信息跟随者、转发评论者 时可适当加权；
- verified_type_name 包含 媒体、政府、机构、企业、个人认证 时可适当加权；
- 仍然需要经过参与概率判断，不是所有高影响力 Agent 都一定发声。

建议计算：

speaker_score =
    0.50 * influence_score
  + 0.30 * activity_score
  + 0.10 * media_dependency_score
  + 0.10 * repost_tendency_score
  + role_bonus

role_bonus 建议：
- 潜在影响者 / KOL / 高影响力：+0.15
- 媒体信息跟随者：+0.08
- 转发评论者：+0.05
- 低活跃观察者：-0.10

最终根据 speaker_score 排序，选取前 kol_speaker_limit 个。
默认 kol_speaker_limit = 5。

注意：
如果符合条件的高影响力 Agent 数量不足，也不要报错，可以少于 kol_speaker_limit。

### 2. select_regular_candidates(agent_states, kol_speakers, config) -> list[AgentState]

功能：
选择普通候选参与者。

规则：
- 排除已经作为 KOL speaker 发声的 Agent；
- 根据 activity_score 和一定随机性决定是否参与；
- active_agent_limit 仍然生效；
- 如果 active_agent_limit 存在，需要在 KOL speakers 和 regular candidates 总数上共同限制。

### 3. select_context_comments(target_agent, candidate_comments, config) -> list[dict]

功能：
为普通 Agent 选择它在本轮看到的代表性评论。

candidate_comments 来自：
- 当前轮已经发声的 KOL speakers；
- 本轮中更早发声的部分普通 Agent；
- 可选：上一轮活跃 Agent 的评论。

第一版优先使用当前轮 KOL speakers 的评论。

选择规则：
每个普通 Agent 最多看到 top_k_context_comments 条评论，默认 3。

排序分数建议：

context_score =
    0.55 * source_influence_score
  + 0.25 * abs(source_emotion_score)
  + 0.20 * abs(source_stance_score)

如果 target_agent.kol_sensitivity_score 较高，则进一步提高高 influence_score 评论的排序。
如果 source_verified_type_name 包含 媒体、政府、机构，且 target_agent.media_dependency_score 较高，则提高该评论排序。

### 4. build_interaction_records(source_comments, target_agent, target_reaction, config) -> list[InteractionRecord]

功能：
根据 target_agent 看到的上下文评论，生成互动边记录。

每条上下文评论至少生成一条：

source_agent_id -> target_agent_id

interaction_type 默认为 same_round_context 或 influence_candidate。

如果 target_reaction.action_type 是 repost 或 repost_with_comment，则 interaction_type 可以为 repost。
如果 target_reaction.reaction_text 中出现明显回应词，如“确实”“同意”“不是吧”“别急”“这说法”“我觉得”，可以标记为 reply，但不要过度复杂。

### 5. compute_interaction_weight(source_agent, target_agent, interaction_type) -> float

权重计算建议：

base_weight =
    source_agent.influence_score
  * target_agent.susceptibility_score

然后根据来源类型调整：

- 如果 source_agent influence_score >= 0.75：
  base_weight *= 1.15

- 如果 source_agent propagation_role 包含 KOL / 潜在影响者 / 高影响力：
  base_weight *= 1.15

- 如果 source_agent verified_type_name 包含 媒体 / 政府 / 机构 / 企业：
  base_weight *= (1 + 0.30 * target_agent.media_dependency_score)

- 如果 source_agent 是普通高影响力用户：
  base_weight *= (1 + 0.30 * target_agent.kol_sensitivity_score)

- 如果 interaction_type == repost：
  base_weight *= 1.20

- 如果 interaction_type == reply：
  base_weight *= 1.10

最终裁剪到 [0.01, 1.00]。

注意：
这个 weight 是后续第三阶段情绪传染用的候选影响权重，本阶段只记录，不用它更新状态。

## 六、修改每轮仿真流程

请在 MultiRoundSimulator 中增加 interaction_mode。

原第一阶段每轮可能是：

遍历 Agent
→ 根据 activity_score 判断是否活跃
→ 生成 fallback reaction
→ 更新状态
→ 保存快照

第二阶段请改造为支持两种模式：

1. interaction_mode = "none"

保持第一阶段原有逻辑，用于兼容。

2. interaction_mode = "kol_first"

新增互动逻辑，默认推荐使用。

kol_first 模式下，每轮流程如下：

### 第 0 步：准备当前轮状态

复制上一轮 AgentState，作为当前轮待更新状态。

### 第 1 步：选择 KOL speakers

使用 InteractionEngine.select_kol_speakers() 选择本轮优先发声的高影响力 Agent。

这些 Agent 先生成 reaction。

如果 use_llm=False，继续使用 fallback reaction。
如果 use_llm=True，后续可以复用已有单事件 Agent 调用逻辑；如果当前接入复杂，可以暂时 fallback，并在文档说明。

KOL speakers 的 reaction 不需要上下文评论，主要基于事件和自身状态生成。

### 第 2 步：选择普通参与 Agent

从剩余 Agent 中选择 regular candidates。

普通 Agent 的参与概率可以在原 activity_score 基础上受到上下文刺激影响。

建议：

context_activation_bonus =
    min(0.20, 0.05 * number_of_available_context_comments)

prob =
    activity_score
  + 0.10 * influence_score
  + context_activation_bonus

如果 target_agent.kol_sensitivity_score 高，且上下文中有高 influence_score 评论，可以再增加一点：

prob += 0.10 * target_agent.kol_sensitivity_score

最终裁剪到 [0.05, 0.95]。

### 第 3 步：为普通 Agent 选择上下文评论

对每个普通参与 Agent：

- 从本轮已经出现的 KOL 评论中选择 top_k_context_comments；
- 如果 allow_previous_round_context=True，则可以加入上一轮活跃评论；
- 记录这些上下文来源。

注意：
不要把上下文文本拼得太长。fallback 模式下只需要用于规则生成和互动边记录。

### 第 4 步：生成普通 Agent reaction

修改或扩展 generate_fallback_reaction()，使其支持 context_comments 参数：

generate_fallback_reaction(agent_state, event, round_id, context_comments=None) -> dict

如果 context_comments 不为空，reaction_text 可以轻微体现“受到评论区已有观点影响”的效果，但不要出现元话语。

示例：

负向 + 看到高影响力质疑评论：
“看了前面的说法，感觉这事还是得继续追问清楚。”

中性 + 看到争议评论：
“评论区分歧挺大，还是等更完整的信息吧。”

支持 + 看到官方/媒体来源：
“如果通报内容属实，那这个处理方向还算明确。”

反对 + 看到高影响力批评：
“前面说得有道理，这个解释确实还不够让人信服。”

要求：
- 不要输出“我看到了 KOL 评论”“根据上下文评论”等元话语。
- reaction_text 仍然保持微博评论风格。
- emotion_label 必须兼容已有 ReactionSchema：
  positive, neutral, anger, sadness, disgust, worry, surprise
- 不要输出 negative 作为 reaction emotion_label。
- AgentState 内部可以继续使用 positive / neutral / negative 作为状态标签。

### 第 5 步：生成 InteractionRecord

对每个普通 Agent：

如果它的 context_comments 中包含 source_agent 的评论，则生成：

source_agent -> target_agent

InteractionRecord。

一条普通 Agent 可以对应多条互动边。

如果普通 Agent 没有 context_comments，则不生成边。

KOL speakers 本轮不需要生成入边，但它们的评论可以作为其他 Agent 的 source。

### 第 6 步：状态更新

本阶段不要做正式邻居影响更新。

可以保留第一阶段逻辑：
- 不活跃 Agent 状态保持；
- 活跃 Agent 根据自身 reaction 轻微更新 emotion_score 和 stance_score。

但 state_update_reason 可以更具体：

- KOL speaker:
  “本轮作为高影响力用户优先发声，状态根据自身表达轻微更新”

- 普通 Agent 且有上下文评论:
  “本轮参考评论区已有观点后参与表达，状态根据自身表达轻微更新”

- 普通 Agent 且无上下文评论:
  “本轮主动参与评论，状态根据自身表达轻微更新”

注意：
虽然普通 Agent 看到了上下文评论，但本阶段不根据 source_agent 的情绪和立场正式更新 target_agent 状态。第三阶段再做。

## 七、修改配置参数

请在 MultiRoundSimulationConfig 中新增或确认以下字段：

- enable_interactions: bool = False
- interaction_mode: str = "none"
- kol_speaker_limit: int = 5
- top_k_context_comments: int = 3
- allow_previous_round_context: bool = False
- max_context_comment_length: int = 80

如果用户传入 --enable-interactions，则默认：

interaction_mode = "kol_first"

如果用户没有传入 --enable-interactions，则保持第一阶段 none 模式。

## 八、修改命令行参数

请修改 scope/run_multiround_simulation.py，新增参数：

- --enable-interactions
- --interaction-mode
- --kol-speaker-limit
- --top-k-context-comments
- --allow-previous-round-context

示例命令：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 30 \
  --memory-user-level core \
  --rounds 5 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3

要求：
1. 不传 --enable-interactions 时，原第一阶段 demo 仍能运行。
2. 传入 --enable-interactions 后，应输出 interactions.csv 和 network.graphml。
3. 命令行结束时打印：
   - run_id
   - 输出目录
   - Agent 数量
   - 仿真轮数
   - interaction_mode
   - interactions.csv 路径
   - network.graphml 路径
   - interaction_count

## 九、输出 interactions.csv

每次运行输出：

scope/data/outputs/simulation/multiround/{run_id}/interactions.csv

字段至少包括：

- run_id
- event_id
- topic
- round_id
- source_agent_id
- target_agent_id
- source_user_id
- target_user_id
- interaction_type
- weight
- source_action_type
- source_reaction_text
- target_action_type
- target_reaction_text
- context_rank
- source_emotion_score
- target_emotion_score_before
- target_emotion_score_after
- source_stance_score
- target_stance_score_before
- target_stance_score_after
- source_influence_score
- target_susceptibility_score
- target_kol_sensitivity_score
- target_media_dependency_score
- source_verified_type_name
- source_propagation_role
- target_propagation_role
- reason
- source
- created_at

要求：
1. 每一行表示一条 source_agent -> target_agent 的候选影响边。
2. source_reaction_text 和 target_reaction_text 可以截断到合理长度，例如 120 字以内。
3. interaction_type 必须有值。
4. weight 必须在 [0.01, 1.00]。
5. 如果某一轮没有互动边，也要输出空 CSV 或至少不报错。
6. interactions.csv 应该可以被 pandas 正常读取。

## 十、输出 network.graphml

请新增 scope/src/simulation/network_builder.py。

实现：

- build_interaction_graph(agent_states, interactions) -> nx.DiGraph
- save_graphml(graph, output_path)

图类型：
NetworkX DiGraph。

节点：
每个 Agent 一个节点，节点 ID 使用 agent_id。

节点属性至少包括：

- agent_id
- user_id
- memory_user_level
- verified_type_name
- propagation_role
- influence_level
- influence_score
- susceptibility_score
- activity_score
- kol_sensitivity_score
- media_dependency_score
- final_emotion_score
- final_stance_score
- final_emotion_label
- final_stance_label

边：
每条 interaction 一条有向边 source_agent_id -> target_agent_id。

边属性至少包括：

- weight
- round_id
- interaction_type
- source_action_type
- target_action_type

如果同一对 source_agent -> target_agent 在多轮中出现多次：
可以选择以下方案：

在 DiGraph 中合并边，推荐使用：
- weight_sum
- interaction_count
- first_round
- last_round
- interaction_types


要求：
1. 如果没有 interactions，也仍然输出只有节点、没有边的 graphml。
2. GraphML 不要写入复杂 Python 对象，只写字符串、数字、布尔等简单属性。
3. 如果 networkx 未安装，请在错误信息或文档中说明需要安装 networkx，但不要让主仿真结果丢失。

## 十一、active_reactions.jsonl 增强

请在 active_reactions.jsonl 中增加以下字段，便于后续查看互动过程：

- speaker_type: "kol_speaker" | "regular_agent"
- context_agent_ids: list[str] 或用逗号分隔字符串
- context_comment_count: int
- influenced_by_high_influence: bool

如果 JSONL 中保存 list 不方便，context_agent_ids 可以保存为用英文逗号连接的字符串。

要求：
1. KOL speakers 的 speaker_type 为 kol_speaker。
2. 普通参与 Agent 的 speaker_type 为 regular_agent。
3. 如果普通 Agent 看到了上下文评论，context_comment_count > 0。
4. 如果上下文中至少有一个 source_agent influence_score >= 0.75，则 influenced_by_high_influence = True。

## 十二、round_metrics.csv 增强

请在 round_metrics.csv 中新增以下字段：

- kol_speaker_count
- regular_active_count
- interaction_count
- avg_interaction_weight
- high_influence_interaction_count
- agents_with_context_count
- avg_context_comment_count

说明：
- kol_speaker_count：本轮优先发声的高影响力 Agent 数量。
- regular_active_count：本轮普通参与 Agent 数量。
- interaction_count：本轮互动边数量。
- avg_interaction_weight：本轮互动边平均权重。
- high_influence_interaction_count：source_influence_score >= 0.75 的互动边数量。
- agents_with_context_count：看到上下文评论的普通 Agent 数量。
- avg_context_comment_count：普通参与 Agent 平均看到的上下文评论数。

如果 enable_interactions=False，这些字段可以为 0 或空值，但 round_metrics.csv 的生成不能失败。

## 十三、dry_run 行为

请增强 dry_run。

当 dry_run=True 且 enable_interactions=True 时，不执行完整仿真，但需要展示：

1. 加载到的事件摘要。
2. 筛选后的 Agent 数量。
3. 前 5 个候选 KOL speakers：
   - agent_id
   - influence_score
   - activity_score
   - propagation_role
   - verified_type_name
4. 示例普通 Agent 会看到的 context_comments 结构。
5. 预计输出文件列表。

dry_run 不应该写入正式输出，或者只写入 dry_run 临时日志。

## 十四、文档更新

请更新：

scope/docs/multiround_simulation.md

新增第二阶段说明：

1. 本阶段实现了什么：
   - KOL / 高影响力 Agent 优先发声；
   - 普通 Agent 观察上下文评论后响应；
   - 生成 source_agent -> target_agent 的互动边；
   - 输出 interactions.csv；
   - 构建基础 NetworkX 互动图 network.graphml。


2. interactions.csv 字段解释。

3. network.graphml 字段解释。

4. 运行示例：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 30 \
  --memory-user-level core \
  --rounds 5 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3


## 十五、测试要求

请完成以下测试或手动验证：

### 测试 1：兼容第一阶段 none 模式

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 10 \
  --memory-user-level core \
  --rounds 3 \
  --use-llm false \
  --seed 42

确认：
- 程序仍能运行；
- agent_states_by_round.csv 存在；
- round_metrics.csv 存在；
- 不要求 interactions.csv 必须有内容；
- 第一阶段功能没有被破坏。

### 测试 2：启用互动模式

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 20 \
  --memory-user-level core \
  --rounds 3 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --kol-speaker-limit 5 \
  --top-k-context-comments 3

确认输出目录中存在：

- config.json
- selected_event.json
- agent_initial_states.csv
- agent_states_by_round.csv
- round_metrics.csv
- active_reactions.jsonl
- interactions.csv
- network.graphml

确认：
- interactions.csv 可以被 pandas 读取；
- interactions.csv 中 source_agent_id 和 target_agent_id 不为空；
- source_agent_id != target_agent_id；
- weight 在 [0.01, 1.00]；
- round_id 范围为 1 到 rounds；
- active_reactions.jsonl 中包含 speaker_type；
- round_metrics.csv 中包含 interaction_count；
- network.graphml 能被 networkx.read_graphml 读取。

### 测试 3：dry_run

运行：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 10 \
  --memory-user-level core \
  --rounds 2 \
  --use-llm false \
  --seed 42 \
  --enable-interactions \
  --interaction-mode kol_first \
  --dry-run

确认：
- 能打印候选 KOL speakers；
- 能展示 context_comments 示例；
- 不执行完整仿真；
- 不报错。

本阶段最终目标：
在现有单事件多轮仿真骨架基础上，实现“高影响力 Agent 先发声、普通 Agent 观察上下文评论后响应、记录 Agent 间候选影响边”的互动机制，为下一阶段情绪传染与立场演化提供 interactions.csv 和 network.graphml 数据基础。