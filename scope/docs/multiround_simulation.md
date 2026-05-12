# 多轮微博用户状态仿真骨架

## 目标

本模块在现有单事件微博用户 Agent 仿真器基础上，新增一个可复用的“单事件多轮群体状态记录框架”。当前阶段重点是稳定记录每个 Agent 在第 0 轮到第 N 轮的情绪、立场、活跃状态和规则反应，并输出每轮群体统计指标。

当前实现服务于毕业设计 MVP 展示：优先保证稳定、可解释、可复现，不追求复杂传播模型。

第二阶段已增加可选的 Agent 互动机制：启用 `--enable-interactions` 后，多轮仿真会采用 `kol_first`
模式，让高影响力 Agent 优先发声，普通 Agent 在观察评论上下文后响应，并记录 `source_agent -> target_agent`
候选影响边。

第三阶段已增加可选的情绪传染与立场演化机制：启用 `--enable-emotion-dynamics` 后，系统会基于互动边聚合邻居情绪和立场，结合 Agent 易感性、事件刺激和自身表达，更新每轮 `emotion_score` 与 `stance_score`，并输出可视化友好的演化指标。

## 与单事件仿真器的关系

单事件仿真器负责“静态画像驱动的一次性反应生成”，入口为 `scope/run_single_event_simulation.py`。

多轮仿真器不重写单事件流程，而是复用以下能力：

- `event_loader.py`：按 `event_id` 加载事件。
- `agent_loader.py`：合并 Agent 画像、记忆和系统提示词。
- 既有输入文件：`agent_profiles.jsonl`、`agent_memories.jsonl`、`agent_sys_prompts.jsonl`、`events.jsonl`。

新增入口为 `scope/run_multiround_simulation.py`，核心模块位于 `scope/src/simulation/`。

## AgentState 字段

`AgentState` 定义在 `agent_state.py`，每条记录表示某个 Agent 在某一轮的状态快照。

主要字段包括：

- 基础标识：`run_id`、`event_id`、`weibo_id`、`topic`、`agent_id`、`user_id`、`round_id`。
- 用户画像：`memory_user_level`、`verified_type_name`、`propagation_role`、`influence_level`。
- 行为参数：`influence_score`、`susceptibility_score`、`activity_score`、`kol_sensitivity_score`、`media_dependency_score`、`repost_tendency_score`。
- 状态字段：`emotion_score`、`stance_score`、`emotion_label`、`stance_label`、`is_active`、`last_action_type`、`last_reaction_text`、`last_reason`。
- 状态变化：`old_emotion_score`、`new_emotion_score`、`emotion_delta`、`old_stance_score`、`new_stance_score`、`stance_delta`、`state_update_reason`。
- 第三阶段动态字段：`neighbor_emotion_score`、`neighbor_stance_score`、`neighbor_count`、`neighbor_influence_weight_sum`、`event_emotion_score`、`event_stance_score`、`own_reaction_emotion_score`、`own_reaction_stance_score`、`dynamics_enabled`。
- 其他字段：`source`、`created_at`。

分数范围会自动裁剪：情绪和立场分数为 `[-1, 1]`，行为参数分数为 `[0, 1]`。

## 画像到初始状态的映射

初始状态通过 `build_initial_agent_state(agent_record, event, run_id)` 构建：

- `memory_user_level`、`verified_type_name`、`propagation_role`、`influence_level` 优先来自 `base_identity`。
- `influence_score`、`kol_sensitivity_score`、`media_dependency_score` 优先来自 `behavior_parameters`。
- `susceptibility_score` 优先读取显式字段；缺失时由 KOL 敏感度和媒体依赖度弱估计。
- `activity_score` 根据 `memory_user_level` 和 `propagation_role` 启发式估计。
- `repost_tendency_score` 使用 `repost_ratio`。
- `emotion_score = pos_ratio - neg_ratio`。
- `stance_score` 根据事件 `dominant_stance_label` 做弱初始化，公共议题兴趣越高，初始立场越明确。

初始状态的 `state_update_reason` 为“根据用户长期画像和事件基本倾向初始化状态”。

## 多轮逻辑

第 0 轮为初始化：

- `round_id = 0`
- `is_active = False`
- `last_action_type = "ignore"`
- `last_reaction_text = ""`
- `state_update_reason = "初始状态"`

第 1 到 N 轮执行轻量状态记录：

- 根据 `activity_score` 决定是否参与。
- `influence_score` 较高会略微提高参与概率。
- 若事件为 `public_issue`，且用户公共议题兴趣较高，也会略微提高参与概率。
- 如果设置 `active_agent_limit`，每轮按活跃度和影响力优先保留活跃 Agent。
- 当前阶段只使用 fallback 规则生成反应，不依赖 API key。
- 活跃 Agent 根据自身表达做很小的情绪和立场更新。
- 未活跃 Agent 状态保持不变。

启用互动后，第 1 到 N 轮会改为：

- 先按 `influence_score`、`activity_score`、传播角色和认证类型选择本轮 KOL / 高影响力发声者。
- 再选择普通参与 Agent。
- 普通 Agent 会看到本轮已发声评论中的前 `top_k_context_comments` 条代表性评论。
- 系统为每条可见上下文评论记录一条候选影响边。
- 未启用情绪动态时，只记录互动和轻微自身表达更新，不根据邻居正式传播情绪或立场。

启用情绪动态后，第 1 到 N 轮会在反应和互动边生成后执行状态演化：

- 根据 `source_agent -> target_agent` 边聚合邻居情绪和立场。
- 使用 `susceptibility_score` 调节社会影响强度。
- 融合自身保持、邻居影响、事件刺激和自身表达。
- 将更新后的 `emotion_score` 和 `stance_score` 裁剪到 `[-1, 1]`。
- 记录每个 Agent 的变化量、邻居影响、事件影响、自身反应影响和中文解释。

情绪更新公式：

```text
new_emotion =
  self_retention * old_emotion
  + social_influence_strength * susceptibility_score * neighbor_emotion_score
  + event_influence_strength * event_emotion_score
  + reaction_influence_strength * own_reaction_emotion_score
```

立场更新公式：

```text
new_stance =
  stance_retention * old_stance
  + social_stance_strength * susceptibility_score * neighbor_stance_score
  + event_stance_strength * event_stance_score
  + reaction_stance_strength * own_reaction_stance_score
```

默认启用饱和阻尼，避免分数过快撞到边界：

```text
delta = raw_new_score - old_score
damping_factor = 1 - saturation_damping_strength * abs(old_score)
new_score = old_score + delta * damping_factor
```

Reaction 层仍使用细粒度 `emotion_label`，例如 `anger`、`joy`、`mixed`；State 层使用连续 `emotion_score`，展示层可归纳为 `positive`、`neutral`、`negative`。

## 输入文件

默认输入目录为 `scope/data/inputs/`：

- `events.jsonl`
- `agent_profiles.jsonl`
- `agent_memories.jsonl`
- `agent_sys_prompts.jsonl`

所有文件按 UTF-8 读取。

## 输出文件

正式运行输出到：

```text
scope/data/outputs/simulation/multiround/{run_id}/
```

文件包括：

- `config.json`：本次仿真配置。
- `selected_event.json`：本次事件记录。
- `agent_initial_states.csv`：第 0 轮初始状态。
- `agent_states_by_round.csv`：所有轮次所有 Agent 状态。
- `round_metrics.csv`：每轮群体统计。
- `active_reactions.jsonl`：仅保存活跃 Agent 的规则反应。
- `interactions.csv`：启用互动时输出，每行表示一条 `source_agent -> target_agent` 候选影响边。
- `network.graphml`：启用互动时输出，基于 `interactions.csv` 构建的基础有向互动图。
- `dynamics_summary.json`：每次运行输出，汇总初始/最终平均情绪和立场、分布、波动性、极化分数、邻居影响覆盖和动态参数。

如果 `rounds=5` 且 Agent 数量为 30，`agent_states_by_round.csv` 应有 `30 * 6` 条数据行，不含表头。

### `interactions.csv` 字段

核心字段包括：

- 基础信息：`run_id`、`event_id`、`topic`、`round_id`。
- 互动双方：`source_agent_id`、`target_agent_id`、`source_user_id`、`target_user_id`。
- 互动类型和权重：`interaction_type`、`weight`。
- 上下文信息：`source_action_type`、`source_reaction_text`、`target_action_type`、`target_reaction_text`、`context_rank`。
- 状态快照：`source_emotion_score`、`target_emotion_score_before`、`target_emotion_score_after`、`source_stance_score`、`target_stance_score_before`、`target_stance_score_after`。
- 画像字段：`source_influence_score`、`target_susceptibility_score`、`target_kol_sensitivity_score`、`target_media_dependency_score`、`source_verified_type_name`、`source_propagation_role`、`target_propagation_role`。
- 其他字段：`reason`、`source`、`created_at`。

`interaction_type` 当前主要包括 `same_round_context`、`reply`、`repost`。`weight` 会裁剪到 `[0.01, 1.00]`，供后续情绪传染阶段复用。

### `network.graphml` 字段

节点 ID 使用 `agent_id`。节点属性包括：

- `agent_id`、`user_id`、`memory_user_level`、`verified_type_name`、`propagation_role`、`influence_level`。
- `influence_score`、`susceptibility_score`、`activity_score`、`kol_sensitivity_score`、`media_dependency_score`。
- `final_emotion_score`、`final_stance_score`、`final_emotion_label`、`final_stance_label`。

边会合并同一对 `source_agent_id -> target_agent_id` 的多轮互动，并保存：

- `weight`、`weight_sum`、`interaction_count`。
- `first_round`、`last_round`、`round_id`。
- `interaction_type`、`interaction_types`、`source_action_type`、`target_action_type`。

如果环境未安装 `networkx`，主仿真结果仍会保留，日志会提示需要安装 `networkx` 才能写出 `network.graphml`。

## 正式运行示例

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_multiround_simulation.py `
  --event-id event_5223110724290198 `
  --max-agents 30 `
  --memory-user-level core `
  --rounds 5 `
  --use-llm false `
  --seed 42
```

## 互动模式运行示例

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_multiround_simulation.py `
  --event-id event_5223110724290198 `
  --max-agents 5 `
  --rounds 5 `
  --use-llm true `
  --seed 42 `
  --enable-interactions `
  --interaction-mode kol_first `
  --kol-speaker-limit 5 `
  --top-k-context-comments 3
```

## 情绪动态运行示例

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_multiround_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 5 `
  --rounds 5 `
  --use-llm true `
  --seed 42 `
  --enable-interactions `
  --interaction-mode kol_first `
  --kol-speaker-limit 5 `
  --top-k-context-comments 3 `
  --enable-emotion-dynamics
```

如果只开启 `--enable-emotion-dynamics` 而不启用 `--enable-interactions`，程序仍可运行，但邻居影响为 0，仅使用事件刺激和自身表达进行状态更新。

## 情绪动态参数

- `self_retention`：情绪自身保持系数，默认 `0.65`。
- `social_influence_strength`：邻居情绪影响系数，默认 `0.25`。
- `event_influence_strength`：事件情绪刺激系数，默认 `0.10`。
- `reaction_influence_strength`：自身表达情绪影响系数，默认 `0.15`。
- `stance_retention`：立场自身保持系数，默认 `0.75`。
- `social_stance_strength`：邻居立场影响系数，默认 `0.20`。
- `event_stance_strength`：事件立场刺激系数，默认 `0.10`。
- `reaction_stance_strength`：自身表达立场影响系数，默认 `0.15`。
- `saturation_damping_strength`：边界阻尼强度，默认 `0.5`。

## 情绪和立场映射

情绪基础分数：

- 负向：`anger=-0.85`、`disgust=-0.80`、`fear=-0.70`、`sadness=-0.70`、`disappointment=-0.65`、`confusion=-0.25`。
- 中性：`mixed=0.00`、`surprise=0.00`。
- 正向：`sympathy=0.35`、`admiration=0.70`、`joy=0.75`。

立场基础分数：

- `favor`：强度 1 为 `0.50`，强度 2 为 `0.85`。
- `against`：强度 1 为 `-0.50`，强度 2 为 `-0.85`。
- `neutral`、`unclear`、`mixed`：`0.00`。

## 新增动态指标

`round_metrics.csv` 在原有字段基础上新增：

- `emotion_volatility`、`stance_volatility`：本轮情绪和立场标准差。
- `avg_abs_emotion_delta`、`avg_abs_stance_delta`：本轮平均绝对变化量。
- `max_abs_emotion_delta`、`max_abs_stance_delta`：本轮最大绝对变化量。
- `dominant_emotion_state`、`dominant_stance_state`：本轮占比最高状态标签。
- `polarization_score`：立场分数标准差。
- `avg_neighbor_count`、`agents_affected_by_neighbors`、`avg_neighbor_influence_weight`：邻居影响统计。
- `dynamics_enabled`：本轮是否启用情绪动态。

## dry_run 示例

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_multiround_simulation.py `
  --event-id event_5223110724290198 `
  --max-agents 3 `
  --memory-user-level core `
  --rounds 2 `
  --dry-run
```

`dry_run` 只加载事件和 Agent，构建初始 `AgentState` 并打印前 3 个状态摘要，不执行多轮循环，也不写输出目录。
启用互动时，`dry_run` 还会展示前 5 个候选 KOL speakers、一个 `context_comments` 示例和预计输出文件列表。

## 当前限制

- 情绪传染与立场演化为简化规则模型。
- 未使用真实语义相似度模型。
- 未进行真实世界预测。
- 互动边仍是候选影响边，用于毕业设计原型展示和机制验证。
- NetworkX 图为基础互动图，不包含复杂社区发现或传播路径分析。
- `use_llm=True` 仅预留参数；本阶段主测 `use_llm=False` fallback 规则。

## 后续扩展方向

- 引入情绪传染与立场演化公式。
- 扩展 NetworkX 传播网络分析指标。
- 接入 Streamlit 可视化面板。
