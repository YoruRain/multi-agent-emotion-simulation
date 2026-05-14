# 多轮 Agent 互动与情绪动态模拟流程说明

本文说明运行 `scope/run_multiround_simulation.py`，并同时开启 Agent 之间互动与情绪动态时，程序内部如何完成一次多轮微博用户状态仿真。

推荐运行形态示例：

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_multiround_simulation.py `
  --event-id <EVENT_ID> `
  --rounds 5 `
  --enable-interactions `
  --enable-emotion-dynamics
```

如果需要调用大模型生成反应，可额外加入 `--use-llm true`、`--max-llm-agents-per-round`、`--llm-concurrency` 等参数；否则系统会使用规则 fallback 生成 Agent 反应。

## 入口与配置组装

入口脚本 `run_multiround_simulation.py` 负责解析命令行参数，并构造 `MultiRoundSimulationConfig`。

当使用 `--enable-interactions` 时，如果没有显式指定 `--interaction-mode`，程序会把互动模式设置为 `kol_first`。这个模式的含义是：每一轮先让高影响力 Agent 发声，再让普通活跃 Agent 在可见评论上下文中参与讨论。

当使用 `--enable-emotion-dynamics` 时，程序会启用规则化的情绪与立场动态更新。相关权重包括：

- `self_retention`：上一轮自身情绪保留系数，默认 `0.65`。
- `social_influence_strength`：邻居评论对情绪的影响系数，默认 `0.25`。
- `event_influence_strength`：事件刺激对情绪的影响系数，默认 `0.10`。
- `reaction_influence_strength`：本轮自身表达对情绪的影响系数，默认 `0.15`。
- `stance_retention`：上一轮自身立场保留系数，默认 `0.75`。
- `social_stance_strength`：邻居评论对立场的影响系数，默认 `0.20`。
- `event_stance_strength`：事件刺激对立场的影响系数，默认 `0.10`。
- `reaction_stance_strength`：本轮自身表达对立场的影响系数，默认 `0.15`。
- `saturation_damping_strength`：情绪或立场接近边界时的阻尼强度，默认入口参数为 `0.5`。

配置对象还包含 `rounds`、`active_agent_limit`、`kol_speaker_limit`、`top_k_context_comments`、`allow_previous_round_context`、`max_context_comment_length`、`seed`、`output_dir` 等参数。

## 初始化阶段

`MultiRoundSimulator.run()` 启动后，首先加载事件与 Agent：

1. 用 `get_event_by_id()` 从 `events.jsonl` 中按 `event_id` 找到目标事件。
2. 用 `load_agent_records()` 合并 Agent 画像、记忆和系统提示词。
3. 根据 `memory_user_level`、`max_agents`、`seed` 等配置过滤或抽样 Agent。

然后程序为每个 Agent 构建第 0 轮状态 `AgentState`。初始状态来自用户画像和事件倾向：

- `influence_score` 来自画像中的影响力参数。
- `activity_score` 根据 `memory_user_level` 和 `propagation_role` 估计。
- `susceptibility_score` 由 KOL 敏感度和媒体依赖度估计。
- `emotion_score = pos_ratio - neg_ratio`，用于表达用户长期情绪倾向。
- `stance_score` 根据事件主导立场或事件焦点做弱初始化。

第 0 轮并不表示真实发声，只是初始快照：

- `round_id = 0`
- `is_active = False`
- `last_action_type = "ignore"`
- `source = "initial_profile"`
- `state_update_reason = "初始状态"`

随后程序创建本次运行目录，写入：

- `config.json`
- `selected_event.json`
- `agent_initial_states.csv`

## 每轮总体结构

从第 1 轮到第 N 轮，开启互动后会走 `_run_interaction_round()`。每一轮可以理解为六步：

1. 选择本轮高影响力发声者。
2. 选择本轮普通活跃候选 Agent。
3. 高影响力 Agent 先生成反应，并形成评论池。
4. 普通 Agent 读取评论上下文后生成反应。
5. 根据可见评论构造互动边。
6. 对所有 Agent 应用情绪与立场动态更新，并输出本轮状态。

这六步完成后，当前轮 `next_states` 会成为下一轮的 `current_states`。也就是说，后续轮次会继承上一轮更新后的情绪、立场、最后表达和参与状态。

## 第一步：选择高影响力发声者

互动模式下，程序先调用 `InteractionEngine.select_kol_speakers()`。

每个 Agent 会得到一个 `speaker_score`：

```text
speaker_score =
  0.50 * influence_score
+ 0.30 * activity_score
+ 0.10 * media_dependency_score
+ 0.10 * repost_tendency_score
+ role_bonus
```

`role_bonus` 会根据传播角色加减分。例如，潜在影响者、KOL、高影响力用户会加分，低活跃观察者会扣分。

随后程序用下面的概率判断该 Agent 是否能成为本轮高影响力发声候选：

```text
participation_prob = clamp(0.15 + speaker_score, 0.05, 0.95)
```

被选中的候选会按 `speaker_score`、`influence_score`、`agent_id` 排序，再受两个限制裁剪：

- `kol_speaker_limit`：每轮最多保留多少个高影响力发声者，默认 `5`。
- 总人口上限：最多不超过总 Agent 数的 `30%`。
- 如果设置了 `active_agent_limit`，还要受总活跃 Agent 上限限制。

这些 Agent 本轮的 `speaker_type` 会记为 `kol_speaker`。

## 第二步：选择普通活跃候选 Agent

高影响力发声者确定后，程序调用 `select_regular_candidates()` 选择普通参与者。

普通 Agent 的参与概率为：

```text
probability = clamp(activity_score + 0.10 * influence_score, 0.05, 0.95)
```

已经作为 KOL 发声的 Agent 不会重复进入普通候选。普通候选会按 `activity_score + influence_score` 排序。如果设置了 `active_agent_limit`，普通候选只能占用 KOL 之后剩余的活跃名额。

这些 Agent 本轮的 `speaker_type` 会记为 `regular_agent`。

## 第三步：KOL 先发声

程序先为所有 KOL 发声者生成反应。

如果 `--use-llm true` 且该 Agent 在本轮 LLM 预算内，程序会调用 `LLMReactionGenerator`。LLM 输入中会包含：

- 事件信息。
- Agent 被选中的相关记忆。
- 当前轮次。
- 本轮角色：高影响力用户优先发声。
- 上一轮自身情绪与立场分数。
- 上一轮可见行为与表达。
- 本轮可见评论上下文。KOL 阶段通常为空。

如果未启用 LLM、没有 API key、解析失败或调用异常，系统会使用 `generate_fallback_reaction()` 生成规则反应。规则反应会根据 Agent 当前情绪、当前立场、事件主导情绪、事件类型和上下文模板生成：

- `participate`
- `action_type`
- `emotion_label`
- `emotion_intensity`
- `stance_label`
- `stance_intensity`
- `reaction_text`
- `reason`

KOL 生成反应后，程序先用 `_update_state_from_reaction()` 做一次基础状态更新，并把该反应加入：

- `active_reactions`
- `round_comments`
- `next_state_by_id`
- `reaction_by_id`

`round_comments` 是本轮已经产生的可见评论池，后续普通 Agent 会从这里选择上下文。

## 第四步：普通 Agent 读取上下文后发声

普通候选 Agent 会按顺序处理。每个普通 Agent 面前的候选评论由两部分组成：

- 当前轮已经产生的 `round_comments`，即 KOL 评论和更早处理的普通 Agent 评论。
- 如果开启 `--allow-previous-round-context`，还会加入上一轮的评论。

然后程序调用 `select_context_comments()`，为当前 Agent 选择最多 `top_k_context_comments` 条可见评论，默认 `3` 条。

每条候选评论的上下文得分大致由评论源 Agent 的影响力、情绪强度、立场强度，以及目标 Agent 对 KOL 和媒体的敏感度决定：

```text
score =
  0.55 * source.influence_score
+ 0.25 * abs(source.emotion_score)
+ 0.20 * abs(source.stance_score)
+ 0.20 * target.kol_sensitivity_score * source.influence_score
```

如果评论源带有媒体、政府、机构等认证，还会额外加上与目标 Agent `media_dependency_score` 相关的分数。

选出的上下文评论会被附加 `context_rank`，并进入本轮反应生成。若启用 LLM，LLM prompt 中会显示这些评论，包括源 Agent、传播角色、认证类型和评论文本。若使用规则 fallback，模板也会根据是否有高影响力来源、是否有认证来源、当前情绪和立场生成不同反应。

普通 Agent 反应生成后，同样会更新基础状态、写入 `active_reactions`，并把自己的新评论追加到 `round_comments`，供后续 Agent 看到。

## 第五步：构造互动边

普通 Agent 只要读取了上下文评论，程序就会调用 `build_interaction_records()`，把每条可见上下文记录为一条候选影响边：

```text
source_agent -> target_agent
```

这里的 `source_agent` 是被目标 Agent 看见的评论作者，`target_agent` 是当前正在发声的普通 Agent。

互动类型由目标 Agent 的行为和文本推断：

- `repost`：目标行为是 `repost` 或 `repost_with_comment`。
- `reply`：目标文本中出现“确实”“同意”“前面”“有道理”等回应提示词。
- `same_round_context`：其他普通可见上下文影响。

互动边权重由评论源影响力和目标易感性开始计算：

```text
weight = source.influence_score * target.susceptibility_score
```

然后根据以下因素放大：

- 源 Agent 影响力不低于 `0.75`。
- 源 Agent 传播角色包含 KOL、潜在影响者或高影响力。
- 源 Agent 是媒体、政府、机构、企业等认证类型。
- 源 Agent 影响力较高且目标 Agent 的 KOL 敏感度较高。
- 互动类型是转发或回复。

最终权重会被裁剪到 `[0.01, 1.0]`。这些互动边本质上不是“真实回复链”，而是“目标 Agent 本轮表达时参考过哪些评论”的候选影响记录。

## 第六步：情绪与立场动态更新

当 `--enable-emotion-dynamics` 开启时，基础状态更新之后，程序会对每个 Agent 统一调用 `_apply_emotion_dynamics()`。

对每个 Agent，程序先聚合邻居影响：

1. 在本轮互动边中找到 `target_agent_id` 等于当前 Agent 的边。
2. 取每条边的 `source_agent` 状态。
3. 用边权重对源 Agent 的 `emotion_score` 和 `stance_score` 做加权平均。
4. 同时统计邻居数量、高影响力邻居数量、媒体邻居数量、KOL 邻居数量。

如果当前 Agent 本轮没有被任何评论影响，邻居情绪和邻居立场都为 `0`，邻居数量也为 `0`。

事件影响由 `build_event_influence_scores()` 生成：

- 事件主导情绪会映射为 `event_emotion_score`。
- 事件主导立场或事件立场焦点会映射为 `event_stance_score`。

自身表达影响来自本轮 reaction：

- `emotion_label` 和 `emotion_intensity` 会映射为 `own_reaction_emotion_score`。
- `stance_label` 和 `stance_intensity` 会映射为 `own_reaction_stance_score`。
- 未参与 Agent 的自身表达影响为 `0`。

最终情绪更新公式为：

```text
raw_emotion =
  self_retention * old_emotion
+ social_influence_strength * susceptibility_score * neighbor_emotion_score
+ event_influence_strength * event_emotion_score
+ reaction_influence_strength * own_reaction_emotion_score
```

最终立场更新公式为：

```text
raw_stance =
  stance_retention * old_stance
+ social_stance_strength * susceptibility_score * neighbor_stance_score
+ event_stance_strength * event_stance_score
+ reaction_stance_strength * own_reaction_stance_score
```

如果启用饱和阻尼，程序不会直接使用 `raw_emotion` 和 `raw_stance`，而是根据旧分数接近边界的程度压缩变化幅度：

```text
damping_factor = 1 - saturation_damping_strength * abs(old_score)
new_score = old_score + (raw_new_score - old_score) * damping_factor
```

这可以避免情绪或立场已经接近 `-1` 或 `1` 时继续过快冲到边界。

动态更新后，程序会更新：

- `emotion_score`
- `stance_score`
- `emotion_label`
- `stance_label`
- `old_emotion_score`
- `new_emotion_score`
- `emotion_delta`
- `old_stance_score`
- `new_stance_score`
- `stance_delta`
- `neighbor_*`
- `event_*`
- `own_reaction_*`
- `dynamics_enabled = True`
- `state_update_reason`

`state_update_reason` 会根据变化幅度和影响来源生成。例如，受负向评论影响导致情绪下降、受积极上下文影响导致情绪上升、立场更偏支持或反对、主要根据自身表达更新、状态变化较小等。

需要注意：在互动模式下，未活跃 Agent 也会进入情绪动态更新。因为事件刺激和上一轮自身状态仍然会参与公式，所以它们可能出现轻微变化；如果没有邻居影响和自身表达，则主要由自保持与事件刺激决定。

## 本轮收尾

当所有 Agent 的动态状态都生成后，程序会回填互动边中的最终状态字段：

- `source_emotion_score`
- `source_stance_score`
- `target_emotion_score_after`
- `target_stance_score_after`

随后构造本轮统计摘要：

- `kol_speaker_count`
- `regular_active_count`
- `interaction_count`
- `avg_interaction_weight`
- `high_influence_interaction_count`
- `agents_with_context_count`
- `avg_context_comment_count`

这些摘要会与本轮所有 Agent 状态一起进入 `compute_round_metrics()`，生成群体级指标。

## 跨轮传递

每一轮返回：

- `next_states`：本轮结束后的所有 Agent 状态。
- `round_reactions`：本轮活跃 Agent 的反应记录。
- `round_interactions`：本轮上下文影响边。
- `round_summary`：本轮互动统计摘要。
- `round_comments`：本轮产生的评论池。

主循环会把：

- `next_states` 作为下一轮 `current_states`。
- `round_comments` 作为下一轮 `previous_round_comments`。
- 所有状态、反应、互动边和指标持续累积。

如果未开启 `--allow-previous-round-context`，上一轮评论不会被普通 Agent 看到，但仍然会在输出中保留。若开启该参数，下一轮普通 Agent 选择上下文时会同时考虑上一轮评论。

## LLM 预算与 fallback 机制

如果启用 LLM，程序会先根据候选 Agent 的优先级选出本轮可以调用 LLM 的 Agent。没有设置 `max_llm_agents_per_round` 时，所有候选活跃 Agent 都可调用 LLM；设置后会按优先级截断。

优先级排序依据包括：

- 是否是 KOL 发声者。
- `speaker_score`
- `influence_score`
- `activity_score`
- `agent_id`

LLM 调用使用异步并发，最大并发数由 `llm_concurrency` 控制。

任何以下情况都会回退到规则反应：

- 没有启用 `--use-llm true`。
- Agent 不在本轮 LLM 预算内。
- 缺少 API key。
- 缺少 Agent 记录。
- LLM 输出无法解析为合法 reaction。
- LLM 调用发生异常。

回退不是中止条件。多轮仿真会继续运行，并在 `active_reactions.jsonl` 中记录 `source`、`llm_attempted`、`parse_status`、`error_message`、`raw_output` 等字段，方便之后排查。

## 输出文件

一次成功运行会在默认目录下创建：

```text
scope/data/outputs/simulation/multiround/<run_id>/
```

关键输出包括：

- `config.json`：本次运行的完整配置。
- `selected_event.json`：本次模拟使用的事件。
- `agent_initial_states.csv`：第 0 轮初始状态。
- `agent_states_by_round.csv`：所有 Agent 每一轮的状态快照，是分析情绪和立场轨迹的主表。
- `round_metrics.csv`：每轮群体统计指标。
- `active_reactions.jsonl`：所有活跃 Agent 的表达、来源、上下文数量和 LLM 状态。
- `interactions.csv`：所有候选影响边。只有开启互动时写入。
- `network.graphml`：基于互动边聚合出的有向网络图。需要 `networkx` 可用。
- `dynamics_summary.json`：本次模拟的最终摘要，包括初末均值、分布变化、最大波动、极化程度、互动数量和动态参数。

## 结果阅读建议

如果想看单个 Agent 的演化轨迹，优先读 `agent_states_by_round.csv`，按 `agent_id` 和 `round_id` 排序，观察：

- `emotion_score`
- `stance_score`
- `emotion_delta`
- `stance_delta`
- `neighbor_count`
- `neighbor_influence_weight_sum`
- `last_reaction_text`
- `state_update_reason`

如果想看互动传播结构，读 `interactions.csv` 或 `network.graphml`：

- `source_agent_id -> target_agent_id` 表示目标 Agent 本轮参考了源 Agent 的评论。
- `weight` 表示该上下文评论对目标 Agent 的候选影响强度。
- `context_rank` 表示这条评论在目标 Agent 可见上下文中的排序。
- `target_emotion_score_before/after` 和 `target_stance_score_before/after` 可以用于观察被影响前后的变化。

如果想看群体趋势，读 `round_metrics.csv`：

- `participation_rate` 看每轮参与率。
- `avg_emotion_score` 和 `avg_stance_score` 看整体均值变化。
- `emotion_volatility` 和 `stance_volatility` 看群体分化程度。
- `polarization_score` 当前等同于立场分数标准差。
- `agents_affected_by_neighbors` 和 `avg_neighbor_influence_weight` 看互动影响覆盖面。

## 简化版执行链

```text
run_multiround_simulation.py
  -> parse_args()
  -> MultiRoundSimulationConfig
  -> MultiRoundSimulator.run()
     -> load event
     -> load agents
     -> build round 0 AgentState
     -> for round_id in 1..N:
        -> select KOL speakers
        -> select regular candidates
        -> generate KOL reactions
        -> convert KOL reactions to visible comments
        -> for each regular candidate:
           -> select visible context comments
           -> generate reaction with context
           -> build source -> target interaction records
           -> append its comment to current round comment pool
        -> for every Agent:
           -> apply base reaction state update
           -> aggregate neighbor influence
           -> combine self retention, neighbor influence, event stimulus, own reaction
           -> apply saturation damping
           -> produce next AgentState
        -> compute round metrics
     -> write states, reactions, interactions, network and summary files
```

## 核心理解

这个多轮模拟不是简单地重复单事件反应生成，而是在每一轮维护一个可继承的 Agent 状态。互动机制决定“谁先说、谁能看见谁、谁可能影响谁”；情绪动态机制决定“上一轮状态、邻居评论、事件刺激和自身表达如何合成为下一轮情绪与立场”。

因此，开启 `--enable-interactions --enable-emotion-dynamics` 后，系统输出的不只是评论文本，而是一组可以追踪的群体状态轨迹、上下文影响边和轮次级统计指标。
