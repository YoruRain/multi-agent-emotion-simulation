你现在需要在现有“单事件微博用户 Agent 仿真器”的基础上，实现第一阶段扩展：多轮仿真骨架与 AgentState 状态系统。

请注意：本阶段不是实现完整的 Agent 互动、情绪传染和复杂意见演化，而是先建立一个稳定、可复用、可验收展示的“多轮状态记录框架”。后续步骤会继续在此基础上加入 KOL 先发声、普通 Agent 响应、互动边、情绪传染和网络分析。

## 一、项目背景

本项目主题是“基于多智能体的社会群体情绪模拟系统的设计与实现”。

目前项目已经实现了一个基于 AgentScope 的单事件微博用户 Agent 仿真器，基本流程是：

给定 event_id
→ 加载事件信息
→ 加载一批微博用户 Agent 的 profile、memories、sys_prompt
→ 构建 AgentScope Agent
→ 输入事件语境和少量记忆样本
→ 让 Agent 输出结构化 JSON 反应
→ 保存 agent_reactions.jsonl
→ 生成 summary_report.csv

已有单事件模拟器的第一版定位是：
“静态画像驱动的单轮群体反应模拟”。

当前扩展目标是：
在不破坏现有单事件模拟器的基础上，新增“单事件多轮群体状态仿真”的骨架，使系统能够记录每个 Agent 在多个轮次中的情绪、立场、参与状态和反应文本，为后续实现 Agent 互动、情绪传染和网络分析打基础。

## 二、重要限制

本阶段请严格控制任务范围：

1. 不要重写已有单事件模拟器。
2. 不要删除或破坏 run_single_event_simulation.py。
3. 不要在本阶段实现复杂 Agent 之间互相对话。
4. 支持 use_llm=False 的规则模板 fallback，使 demo 在没有 API 配置时也能稳定运行。
5. 新功能应尽量放在新模块中，减少对已有模块的侵入。
6. 所有路径使用 pathlib。
7. 所有输出文件使用 UTF-8 编码。
8. 随机过程必须支持 seed，保证 demo 可复现。
9. 对缺失字段要 warning，而不是直接崩溃。

本阶段的目标是：
“先让多轮状态系统跑起来，并稳定输出每轮统计结果”。

## 三、请先阅读现有代码

请先查看当前项目中与单事件仿真器相关的文件。如果文件名或路径与下面不完全一致，请根据项目结构自行定位：

- scope/src/simulation/single_event_simulator.py
- scope/src/simulation/agent_loader.py
- scope/src/simulation/event_loader.py
- scope/src/simulation/reaction_schema.py
- scope/run_single_event_simulation.py

请理解现有单事件模拟器的数据加载、Agent 合并、ReactionSchema、JSON 解析、结果保存、resume、dry_run 和 logging 逻辑。

本次新增功能应优先复用已有能力，而不是复制粘贴出一套完全独立的数据读取系统。

## 四、建议新增或修改的文件

建议新增：

- scope/src/simulation/agent_state.py
- scope/src/simulation/multiround_config.py
- scope/src/simulation/multiround_simulator.py
- scope/src/simulation/multiround_analyzer.py
- scope/run_multiround_simulation.py
- scope/docs/multiround_simulation.md

必要时可以小幅修改：

- scope/src/simulation/agent_loader.py
- scope/src/simulation/event_loader.py
- scope/src/simulation/result_analyzer.py

但请不要破坏已有单事件运行逻辑。

## 五、实现 AgentState 数据结构

请在 scope/src/simulation/agent_state.py 中定义 AgentState。

可以使用 dataclass 或 Pydantic BaseModel。字段至少包括：

基础标识字段：

- run_id: str
- event_id: str
- weibo_id: str | None
- topic: str | None
- agent_id: str
- user_id: str | None
- round_id: int

用户画像字段：

- memory_user_level: str | None
- verified_type_name: str | None
- propagation_role: str | None
- influence_level: str | None

行为参数字段：

- influence_score: float
- susceptibility_score: float
- activity_score: float
- kol_sensitivity_score: float
- media_dependency_score: float
- repost_tendency_score: float

状态字段：

- emotion_score: float
- stance_score: float
- emotion_label: str
- stance_label: str
- is_active: bool
- last_action_type: str
- last_reaction_text: str
- last_reason: str

状态变化字段：

- old_emotion_score: float | None
- new_emotion_score: float | None
- emotion_delta: float | None
- old_stance_score: float | None
- new_stance_score: float | None
- stance_delta: float | None
- state_update_reason: str

其他字段：

- source: str
- created_at: str

字段要求：

1. emotion_score 范围必须裁剪到 [-1, 1]。
2. stance_score 范围必须裁剪到 [-1, 1]。
3. influence_score、susceptibility_score、activity_score、kol_sensitivity_score、media_dependency_score、repost_tendency_score 范围建议裁剪到 [0, 1]。
4. emotion_label 由 emotion_score 推导：
   - emotion_score >= 0.25: positive
   - -0.25 < emotion_score < 0.25: neutral
   - emotion_score <= -0.25: negative
5. stance_label 由 stance_score 推导：
   - stance_score >= 0.25: support
   - -0.25 < stance_score < 0.25: neutral
   - stance_score <= -0.25: against
6. last_action_type 默认 ignore。
7. last_reaction_text 默认空字符串。
8. state_update_reason 用一句简短中文说明本轮状态来源。

请同时实现若干工具函数：

- clamp(value, min_value, max_value) -> float
- score_to_emotion_label(score: float) -> str
- score_to_stance_label(score: float) -> str
- agent_state_to_dict(state: AgentState) -> dict

## 六、实现画像到 AgentState 的映射

请在 agent_state.py 或 multiround_simulator.py 中实现：

build_initial_agent_state(agent_record, event, run_id) -> AgentState

agent_record 应尽量复用已有 agent_loader 合并后的结构，包括：

- agent_id
- user_id
- profile
- memories
- sys_prompt
- memory_user_level
- prompt_profile
- behavior_parameters
- metadata

event 来自 events.jsonl 中的事件记录。

映射规则如下：

### 1. 基础字段映射

agent_id:
优先使用 agent_record["agent_id"]。

user_id:
优先使用 agent_record["user_id"]。

memory_user_level:
优先从 base_identity 中读取。

verified_type_name:
优先从 base_identity 中读取。

propagation_role:
优先从 base_identity 中读取。

influence_level:
优先从 base_identity 中读取。


### 2. influence_score 映射

优先读取 behavior_parameters 中已有的影响力分数：

- influence_score

### 3. susceptibility_score 映射

susceptibility_score 表示该 Agent 在群体环境中受他人影响的程度。

优先读取：

- kol_sensitivity_score
- media_dependency_score

susceptibility_score = 0.35 + 0.35 * kol_sensitivity_score + 0.25 * media_dependency_score

### 4. activity_score 映射

activity_score 表示该 Agent 在每轮仿真中主动参与的基础概率。

请根据 memory_user_level 和 propagation_role 启发式估计：

- memory_user_level == "core": 0.75
- memory_user_level == "normal": 0.55
- memory_user_level == "background": 0.35
- propagation_role 包含 原创表达者、高活跃、转发评论者：额外 +0.10
- propagation_role 包含 低活跃观察者：额外 -0.15

最终裁剪到 [0.05, 0.95]。

### 5. kol_sensitivity_score 映射

优先读取 behavior_parameters 中的：

- kol_sensitivity_score

### 6. media_dependency_score 映射

优先读取 behavior_parameters 中的：

- media_dependency_score

### 7. repost_tendency_score 映射

优先读取：

- repost_ratio

### 8. emotion_score 初始化

优先从 behavior_parameters 中读取长期情绪比例，例如：

- pos_ratio
- neu_ratio
- neg_ratio

emotion_score = pos_ratio - neg_ratio

最终裁剪到 [-1, 1]。

### 9. stance_score 初始化

第一阶段不要求复杂立场预测。

可以根据事件 dominant_stance_label 和用户主题兴趣做弱初始化：

- 如果用户 public_issue_topic_ratio 较高，且事件 event_type 为 public_issue，则 Agent 更可能形成明确立场。
- 如果事件 dominant_stance_label == support，则基础 event_stance_bias = +0.2。
- 如果 dominant_stance_label == against，则基础 event_stance_bias = -0.2。
- neutral / unclear / 缺失则为 0。

如果无法判断，stance_score 默认 0。

注意：
这里不是最终立场演化，只是初始状态估计，不要做复杂逻辑。

### 10. state_update_reason

初始状态的 state_update_reason 写为：

“根据用户长期画像和事件基本倾向初始化状态”

## 七、实现 MultiRoundSimulationConfig

请在 scope/src/simulation/multiround_config.py 中定义配置对象 MultiRoundSimulationConfig。

字段建议包括：

- event_id: str
- max_agents: int | None
- memory_user_level: str | None
- rounds: int = 5
- active_agent_limit: int | None = None
- use_llm: bool = False
- seed: int = 42
- output_dir: Path
- overwrite: bool = False
- resume: bool = True
- dry_run: bool = False

以及后续预留字段：

- enable_interactions: bool = False
- enable_emotion_dynamics: bool = False
- interaction_mode: str = "none"

注意：
本阶段 enable_interactions 和 enable_emotion_dynamics 默认 False，只作为后续扩展参数预留，不实现复杂逻辑。

## 八、实现 MultiRoundSimulator 骨架

请在 scope/src/simulation/multiround_simulator.py 中实现 MultiRoundSimulator。

基本流程：

1. 初始化
   - 接收 MultiRoundSimulationConfig
   - 加载事件数据
   - 加载 Agent profile、memory、sys_prompt
   - 设置随机种子
   - 创建 run_id
   - 创建输出目录

2. run()
   - 根据 event_id 找到事件
   - 根据 memory_user_level 和 max_agents 筛选 Agent
   - 构建初始 AgentState
   - 保存第 0 轮状态
   - 执行第 1 到 rounds 轮循环
   - 每轮决定哪些 Agent 活跃
   - 为活跃 Agent 生成 fallback reaction
   - 更新 last_action_type、last_reaction_text、last_reason、is_active
   - 本阶段暂时不做邻居影响更新，emotion_score 和 stance_score 可以只做轻微扰动或保持不变
   - 保存每轮 AgentState 快照
   - 计算每轮 round_metrics
   - 保存输出文件

### 第 0 轮

第 0 轮表示初始状态，不生成评论，不参与互动。

要求：
- round_id = 0
- is_active = False
- last_action_type = "ignore"
- last_reaction_text = ""
- state_update_reason = “初始状态”

### 第 1 到 N 轮

本阶段每轮只做“是否参与”和“状态快照记录”。

参与概率：

base_prob = activity_score

可以加入轻微调整：

- influence_score 高的 Agent，参与概率略高：
  prob += 0.1 * influence_score

- 如果事件 event_type 为 public_issue，且用户主题兴趣中 public_issue_topic_ratio 较高：
  prob += 0.1 * public_issue_topic_ratio

最终 prob 裁剪到 [0.05, 0.95]。

然后：

is_active = random.random() < prob

如果设置 active_agent_limit，则每轮最多保留 active_agent_limit 个活跃 Agent。
当活跃 Agent 超过限制时，优先保留 activity_score 和 influence_score 较高的 Agent。

### fallback reaction 生成

本阶段必须支持 use_llm=False。

当 use_llm=False 时，不调用 AgentScope 或大模型，而是根据当前 AgentState 和 event 生成规则模板 reaction。

请实现：

generate_fallback_reaction(agent_state, event, round_id) -> dict

输出字段应尽量兼容已有 ReactionSchema：

{
  "participate": true,
  "action_type": "comment",
  "emotion_label": "...",
  "emotion_intensity": 1,
  "stance_label": "...",
  "stance_intensity": 1,
  "reaction_text": "...",
  "reason": "..."
}

action_type 规则：
- 如果 is_active=False，则 action_type="ignore"，reaction_text=""。
- 如果 repost_tendency_score >= 0.65，则 action_type 可以是 "repost_with_comment"。
- 否则 action_type 默认为 "comment"。

reaction_text 模板要求：
1. 不要出现“作为 Agent”“根据画像”“系统判断”等元话语。
2. 尽量像微博评论区自然表达。
3. 根据 emotion_label 和 stance_label 生成不同模板。
4. 文本长度控制在 10 到 60 个中文字符左右。
5. 只作为 demo fallback，不追求高度真实。

示例模板：

- negative + against:
  “这事看着真的有点离谱，希望后续能给个清楚说法。”
- negative + neutral:
  “先观望吧，现在信息还不太完整。”
- positive + support:
  “如果情况属实，这个处理还是比较及时的。”
- neutral + neutral:
  “继续看看后续通报，先不急着下结论。”
- positive + neutral:
  “目前看还有不少细节，等更多信息出来再说。”

reason 示例：
“根据该用户活跃度和当前事件倾向生成的规则反应”

### 状态更新

本阶段不实现真正情绪传染。

为了让多轮结果不完全静止，可以采用非常轻微、可解释的规则：

1. 如果 Agent 本轮不活跃：
   - emotion_score 和 stance_score 保持不变。
   - state_update_reason = “本轮未参与，状态保持稳定”。

2. 如果 Agent 本轮活跃：
   - 根据自身 reaction 的 emotion_label / stance_label，对 emotion_score / stance_score 做轻微更新。
   - 更新幅度不要太大，例如 0.05 到 0.10。
   - 更新后裁剪到 [-1, 1]。
   - state_update_reason = “本轮主动参与评论，状态根据自身表达轻微更新”。

注意：
复杂的邻居影响、KOL 影响、情绪传染公式留到下一阶段实现，不要在本阶段加入。

## 九、输出文件

每次运行输出到：

scope/data/outputs/simulation/multiround/{run_id}/

至少保存以下文件：

1. config.json

保存本次仿真参数。

2. selected_event.json

保存本次仿真的事件信息。

3. agent_initial_states.csv

保存第 0 轮 Agent 初始状态。

4. agent_states_by_round.csv

保存所有轮次的 Agent 状态。
要求：
- 每个 Agent 每一轮都应该有一条记录。
- 如果 rounds=5，且 Agent 数量为 30，则应该有 30 * 6 条记录，因为包括第 0 轮。
- 必须包含 round_id。

5. round_metrics.csv

每轮群体统计。

字段至少包括：

- run_id
- event_id
- topic
- round_id
- total_agents
- active_agent_count
- participation_rate
- avg_emotion_score
- avg_stance_score
- positive_count
- neutral_emotion_count
- negative_count
- positive_ratio
- neutral_ratio
- negative_ratio
- support_count
- neutral_stance_count
- oppose_count
- support_ratio
- neutral_stance_ratio
- oppose_ratio
- avg_influence_score_active
- avg_activity_score_active

6. active_reactions.jsonl

只保存每轮活跃 Agent 的规则反应结果。

每条记录至少包含：

- run_id
- event_id
- round_id
- agent_id
- user_id
- memory_user_level
- propagation_role
- influence_score
- activity_score
- participate
- action_type
- emotion_label
- emotion_intensity
- stance_label
- stance_intensity
- reaction_text
- reason
- source

source 本阶段可以写为 "fallback_rule"。

## 十、实现 multiround_analyzer.py

请新增 scope/src/simulation/multiround_analyzer.py。

实现函数：

- compute_round_metrics(states: list[AgentState], round_id: int) -> dict
- save_round_metrics(metrics_list, output_path)
- summarize_run(output_dir) -> dict

功能：
1. 根据每轮 AgentState 计算 round_metrics。
2. 支持从 agent_states_by_round.csv 重新计算 round_metrics。
3. 不依赖 LLM。
4. 如果某一轮没有活跃 Agent，也不能报错。

## 十一、实现命令行脚本

请新增：

scope/run_multiround_simulation.py

支持命令：

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 30 \
  --memory-user-level core \
  --rounds 5 \
  --use-llm false \
  --seed 42

参数至少包括：

- --event-id
- --max-agents
- --memory-user-level
- --rounds
- --active-agent-limit
- --output-dir
- --seed
- --use-llm
- --overwrite
- --resume
- --dry-run

要求：
1. dry_run=True 时，只加载事件和 Agent，构建初始 AgentState，打印前 3 个 Agent 的状态摘要，不执行多轮仿真。
2. 正式运行结束后，在终端输出：
   - run_id
   - 输出目录
   - Agent 数量
   - 仿真轮数
   - 最终轮平均情绪分数
   - 最终轮平均立场分数
   - round_metrics.csv 路径
3. 如果 event_id 不存在，需要给出清晰错误信息。
4. 如果筛选后 Agent 数量为 0，需要给出清晰错误信息。

## 十二、文档要求

请新增或更新：

scope/docs/multiround_simulation.md

内容包括：

1. 多轮仿真骨架的目标
2. 与单事件仿真器的关系
3. AgentState 字段说明
4. 画像字段如何映射为 Agent 初始状态
5. 当前阶段的多轮逻辑：
   - 第 0 轮初始化
   - 第 1 到 N 轮根据 activity_score 判断是否参与
   - 使用 fallback 规则生成评论
   - 仅做轻微自身状态更新
6. 输入文件说明
7. 输出文件说明
8. 命令行运行示例
9. dry_run 示例
10. 当前限制：
   - 尚未实现 Agent 之间的真实互动
   - 尚未实现 KOL 先发声
   - 尚未实现情绪传染
   - 尚未实现 NetworkX 网络分析
   - use_llm=True 可以预留，但本阶段主测 use_llm=False
11. 后续扩展方向：
   - 加入 KOL 先发声和普通 Agent 后响应
   - 构建 interactions.csv
   - 根据 influence_score 和 susceptibility_score 计算互动权重
   - 引入情绪传染与立场演化公式
   - 构建 NetworkX 传播网络
   - 接入 Streamlit 可视化面板

## 十三、测试要求

请至少完成以下测试或手动验证：

1. dry_run 测试

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 3 \
  --memory-user-level core \
  --rounds 2 \
  --dry-run

确认：
- 能加载事件
- 能加载 Agent
- 能构建 AgentState
- 能打印状态摘要

2. 小规模正式运行

python scope/run_multiround_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 10 \
  --memory-user-level core \
  --rounds 3 \
  --use-llm false \
  --seed 42

确认输出目录中存在：

- config.json
- selected_event.json
- agent_initial_states.csv
- agent_states_by_round.csv
- round_metrics.csv
- active_reactions.jsonl

确认：
- agent_states_by_round.csv 行数 = Agent 数量 * (rounds + 1)
- round_metrics.csv 有 rounds + 1 行
- active_reactions.jsonl 中只包含活跃 Agent 反应
- 不配置 API key 也能正常运行

## 十四、代码质量要求

1. 保持模块化，函数职责清晰。
2. 使用类型注解。
3. 使用 logging，不要大量 print。
4. 核心逻辑添加简洁中文注释。
5. 不要在代码中硬编码绝对路径。
6. 不要在代码中硬编码 API key。
7. 不要破坏已有单事件模拟器。
8. 不要修改已有数据生成逻辑。
9. 对字段缺失进行兼容处理。
10. 对 JSONL / CSV 写入要确保 UTF-8 编码。
11. 每次运行生成独立 run_id，避免覆盖旧结果；除非 overwrite=True。
12. 代码应便于后续加入 interaction_engine.py 和 emotion_dynamics.py。

## 十五、最终交付

完成后请给出：

1. 修改了哪些文件。
2. 新增了哪些文件。
3. 如何运行 dry_run。
4. 如何运行小规模 demo。
5. 输出文件在哪里。
6. 当前实现了什么。
7. 当前还没有实现什么。
8. 建议的 git commit message。

建议 commit message：

feat: add multiround simulation state skeleton

本阶段最终目标是：
在现有单事件 Agent 反应模拟器基础上，新增一个可运行的多轮状态仿真骨架，使系统能够输出每个 Agent 每一轮的状态、活跃反应和群体统计指标，为后续 Agent 互动、情绪传染和网络分析提供基础。