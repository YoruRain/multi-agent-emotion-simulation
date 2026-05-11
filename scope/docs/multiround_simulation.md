# 多轮微博用户状态仿真骨架

## 目标

本模块在现有单事件微博用户 Agent 仿真器基础上，新增一个可复用的“单事件多轮群体状态记录框架”。当前阶段重点是稳定记录每个 Agent 在第 0 轮到第 N 轮的情绪、立场、活跃状态和规则反应，并输出每轮群体统计指标。

当前实现服务于毕业设计 MVP 展示：优先保证稳定、可解释、可复现，不追求复杂传播模型。

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

如果 `rounds=5` 且 Agent 数量为 30，`agent_states_by_round.csv` 应有 `30 * 6` 条数据行，不含表头。

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

## 当前限制

- 尚未实现 Agent 之间的真实互动。
- 尚未实现 KOL 先发声。
- 尚未实现情绪传染。
- 尚未实现 NetworkX 网络分析。
- `use_llm=True` 仅预留参数；本阶段主测 `use_llm=False` fallback 规则。

## 后续扩展方向

- 加入 KOL 先发声和普通 Agent 后响应。
- 构建 `interactions.csv`。
- 根据 `influence_score` 和 `susceptibility_score` 计算互动权重。
- 引入情绪传染与立场演化公式。
- 构建 NetworkX 传播网络。
- 接入 Streamlit 可视化面板。
