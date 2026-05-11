你现在需要实现一个“基于 AgentScope 的单事件微博用户 Agent 仿真器”。

## 项目背景
本项目主题是“基于多智能体的社会群体情绪模拟系统的设计与实现”。目前已经完成数据准备与用户画像构建，已经生成以下 JSONL 数据文件：

1. agent_profiles.jsonl

文件路径：data\scope\agent_profiles.jsonl

每行对应一个微博用户 Agent 的结构化画像，包含：
- agent_id
- user_id
- base_identity
- prompt_profile
- behavior_parameters
- metadata

其中 prompt_profile 已经包括：
- identity_summary
- emotion_summary
- topic_summary
- propagation_summary

behavior_parameters 中包含长期情绪比例、主题兴趣比例、转发比例、KOL 敏感度、媒体依赖度、影响力分数等数值参数。

2. agent_memories.jsonl

文件路径：data\scope\agent_memories.jsonl

每行对应一个微博用户 Agent 的历史记忆样本，包含：
- agent_id
- user_id
- memory_user_level
- memories

其中 memories 是列表，每条 memory 包含：
- memory_id
- weibo_id
- mark，例如 general_memory、style_memory、public_issue_memory、propagation_memory
- content
- topics
- mentions
- metadata

metadata 中包含 memory_type、is_repost、has_repost_comment、sentiment_label、final_topic_categories、final_topic_labels、topic_signal_source、source_author_type、selection_reason 等字段。

3. agent_sys_prompts.jsonl

文件路径：data\scope\agent_sys_prompts.jsonl

每行对应一个微博用户 Agent 的系统提示词，包含：
- agent_id
- user_id
- sys_prompt

sys_prompt 已经要求 Agent 根据用户长期画像、主题兴趣、传播习惯和代表性记忆，对给定热点事件或微博语境做出符合该用户特征的反应，并要求输出严格 JSON。

4. events.jsonl

文件路径：data\scope\events.jsonl

每行对应一个待模拟的热点事件，包含：
- event_id
- weibo_id
- topic
- event_context
- event_type
- event_emotion_tendency
- event_emotion_summary
- event_stance_focus
- dominant_emotion_label
- dominant_stance_label
- dominant_stance_target_type
- dominant_stance_target_text
- dominant_responsibility
- dominant_norm_violation
- comment_count_used
- emotion_distribution
- stance_distribution
- metadata

你的任务：
请基于 AgentScope 框架实现一个“单事件仿真器”。第一版不需要实现复杂的多轮传播，也不需要让 Agent 之间互相对话。目标是完成以下最小闭环：

给定一个 event_id
    ↓
加载该事件信息
    ↓
加载一批微博用户 Agent 的 sys_prompt、profile 和 memories
    ↓
为每个 Agent 构建 AgentScope Agent
    ↓
输入事件语境和该 Agent 的少量记忆样本
    ↓
让 Agent 输出结构化 JSON 反应
    ↓
保存所有 Agent 的仿真结果
    ↓
生成基础统计报告

请先查阅当前 AgentScope 官方文档和 GitHub 仓库，确认当前版本中 ReActAgent、model、formatter、memory、structured output 或 JSON 输出的推荐写法。不要凭空猜 API。如果项目中尚未安装 AgentScope，请在代码或 README 中说明推荐安装方式，但不要擅自改动全局环境配置。

AgentScope 官方文档：https://docs.agentscope.io/

GitHub 仓库：https://github.com/agentscope-ai/agentscope（或：仓库已经克隆至本地，可通过路径 D:\科研\Agent\AgentScope 查看）

实现目标：
请优先实现一个稳定、可复现、易调试的 MVP，而不是复杂系统。

建议新增或修改以下文件：

- scope/src/simulation/single_event_simulator.py
- scope/src/simulation/agent_loader.py
- scope/src/simulation/event_loader.py
- scope/src/simulation/reaction_schema.py
- scope/src/simulation/result_analyzer.py
- scope/run_single_event_simulation.py

## 核心功能要求

### 一、数据加载

实现 JSONL 加载工具，读取：
- agent_profiles.jsonl
- agent_memories.jsonl
- agent_sys_prompts.jsonl
- events.jsonl

要求：
1. 按 agent_id 对 profile、memories、sys_prompt 进行合并。
2. 支持按 memory_user_level 过滤 Agent，例如只运行 core、normal 或 background。
3. 支持 max_agents 参数，方便只抽样运行前 N 个 Agent。
4. 支持 event_id 参数，指定单个事件。
5. 对缺失字段做容错处理。如果某个 Agent 缺少 memories，可以继续运行，但需要记录 warning。
6. 对缺失 sys_prompt 的 Agent 应跳过或使用 prompt_profile 渲染一个兜底 prompt，并在日志中记录。

### 二、Agent 构建

请基于 AgentScope 当前版本推荐方式构建 Agent。原则如下：

1. sys_prompt 使用 agent_sys_prompts.jsonl 中已经生成好的 sys_prompt。
2. agent_id 作为 Agent 的 name 或可追踪标识。
3. memories 不要全部塞进 sys_prompt，而是优先尝试使用 AgentScope memory 机制存储；如果当前 AgentScope API 使用成本较高，可以在 MVP 中把 memories 压缩成“当前上下文记忆片段”传入用户消息，但代码中要保留后续替换为 memory 模块的接口。
4. mark 字段应尽量保留，例如 style_memory、public_issue_memory、propagation_memory，以便后续按记忆类型检索。
5. 不要让 Agent 看到大量原始数值参数。behavior_parameters 第一版只用于日志、后续扩展和可选的启发式控制，不要直接完整拼进大模型 prompt。

### 三、事件输入格式

对每个 Agent 输入的事件消息建议包含以下内容：

- topic
- event_context
- event_type
- event_emotion_tendency
- event_emotion_summary
- event_stance_focus
- dominant_stance_target_text
- emotion_distribution 的简要摘要
- stance_distribution 的简要摘要

要求：
1. 输入应是中文。
2. 不要把过多原始字段机械堆叠给 Agent。
3. 可以构造一个函数 build_event_message(event, memories)，输出当前 Agent 的用户消息。
4. memories 建议最多传入 3 到 6 条，优先选择：
   - public_issue_memory
   - style_memory
   - propagation_memory
   - general_memory
5. 如果事件类型是 public_issue，则优先传入 public_issue_memory；如果用户没有该类记忆，再退回 general_memory 或 style_memory。

### 四、Agent 输出格式

每个 Agent 必须输出严格 JSON，字段如下：

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
- 如果 participate 为 false，则 action_type 必须为 "ignore"
- 如果 participate 为 false，则 reaction_text 必须为空字符串
- 如果 participate 为 false，则 emotion_intensity 和 stance_intensity 必须为 0
- emotion_label: "positive", "neutral", "anger", "sadness", "disgust", "worry", "surprise"
- emotion_intensity: 0, 1, 2
- stance_label: "support", "against", "neutral", "unclear"
- stance_intensity: 0, 1, 2
- reaction_text 应尽量像微博用户的自然表达，不要出现“根据画像”“数据表明”“作为 Agent”等元话语
- reason 用于实验分析，可以简短说明原因，但不要过长

请使用 Pydantic 或等价方式定义 ReactionSchema，对模型输出进行校验。

如果 AgentScope 当前版本支持 structured output，请优先使用官方推荐方式。
如果不方便使用 structured output，请实现稳健的 JSON 解析与修复流程：
1. 先尝试直接 json.loads。
2. 如果失败，尝试从文本中提取第一个 JSON 对象。
3. 如果仍失败，进行一次 retry，提示模型必须输出严格 JSON。
4. 如果最终失败，记录为 failed，并保存原始输出。

### 五、仿真运行逻辑

实现 SingleEventSimulator，基本流程：

1. 初始化：
   - 读取路径配置
   - 加载事件
   - 加载 Agent profile、memory、sys_prompt
   - 初始化 AgentScope model / formatter / memory 相关组件

2. run_event(event_id, max_agents=None, memory_user_level=None)
   - 找到指定事件
   - 遍历 Agent
   - 为每个 Agent 构建输入消息
   - 调用 Agent
   - 校验输出
   - 保存结果

3. 结果应逐条写入，避免中途失败导致全部丢失。
   推荐输出 JSONL，同时可以额外导出 parquet。

4. 支持 resume：
   如果输出文件中已经存在某个 event_id + agent_id 的结果，默认跳过，除非传入 overwrite=True。

5. 支持 dry_run：
   dry_run=True 时不调用大模型，只打印将要输入给 Agent 的 sys_prompt 摘要、memory 摘要和 event message，方便检查。

### 六、输出结果

每条仿真结果至少包含：

- run_id
- event_id
- weibo_id
- topic
- agent_id
- user_id
- memory_user_level
- verified_type_name
- influence_level
- propagation_role，如果能从 profile 中拿到
- participate
- action_type
- emotion_label
- emotion_intensity
- stance_label
- stance_intensity
- reaction_text
- reason
- raw_output
- parse_status
- error_message
- model_name
- created_at

输出路径建议：
- scope/outputs/simulation/single_event/{run_id}/agent_reactions.jsonl
- scope/outputs/simulation/single_event/{run_id}/summary_report.csv

### 七、基础统计分析

实现 result_analyzer.py，读取 agent_reactions.jsonl 或 csv，输出：

1. 总体参与率
2. action_type 分布
3. emotion_label 分布
4. stance_label 分布
5. emotion_intensity 平均值
6. stance_intensity 平均值
7. 按 memory_user_level 分组统计
8. 按 influence_level 分组统计
9. 按 action_type 分组统计
10. failed / parse_failed 数量

summary_report.csv 至少应包含 event_id、topic、total_agents、success_count、failed_count、participation_rate、comment_count、repost_count、repost_with_comment_count、ignore_count、dominant_emotion、dominant_stance 等字段。

### 八、命令行脚本

请实现 scope/run_single_event_simulation.py，支持命令行参数，例如：

python scope/run_single_event_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 50 \
  --memory-user-level core \
  --output-dir outputs/simulation/single_event \
  --dry-run

以及正式运行：

python scope/run_single_event_simulation.py \
  --event-id event_5223110724290198 \
  --max-agents 50 \
  --memory-user-level core

建议支持参数：
- --event-id
- --max-agents
- --memory-user-level
- --output-dir
- --overwrite
- --resume
- --dry-run
- --seed
- --concurrency

第一版如果异步并发接入 AgentScope 有困难，可以先串行运行，但代码结构要便于后续加入并发。

### 九、模型配置

请不要在代码中硬编码 API key。
请优先使用环境变量或项目已有配置文件。

建议支持：
- DEEPSEEK_API_KEY
- MODEL_NAME
- BASE_URL

如果项目中已有统一 LLM 配置，请复用项目已有配置。

如果使用 DeepSeek 或 OpenAI-compatible API，请根据 AgentScope 当前文档确认 model 初始化方式。

### 十、日志与错误处理

要求：
1. 使用 logging，不要使用大量 print。
2. 每个 Agent 调用前后记录简要日志。
3. 对以下情况做 warning 或 error：
   - event_id 不存在
   - agent 缺少 sys_prompt
   - agent 缺少 memories
   - 模型调用失败
   - JSON 解析失败
   - Pydantic 校验失败
4. 失败结果也要保存，不能让单个 Agent 失败导致整个事件运行中断。

### 十一、代码质量要求

1. 代码应模块化，函数职责清晰。
2. 使用类型注解。
3. 使用 dataclass 或 Pydantic model 定义核心数据结构。
4. 路径不要写死，尽量通过参数或配置传入。
5. 不要破坏现有数据文件。
6. 不要修改已有画像生成逻辑。
7. 只新增仿真相关模块，必要时最小范围修改公共工具。
8. 保持中文注释简洁清晰。
9. 运行脚本前请先提供 dry_run 能力，方便检查输入内容。

### 十二、文档要求

请新增 scope/docs/single_event_simulation.md，说明：

1. 单事件仿真器的目标
2. 输入文件说明
3. 输出文件说明
4. 命令行运行示例
5. dry_run 示例
6. Agent 输出 JSON 字段解释
7. 当前 MVP 的限制：
   - 只做单事件
   - Agent 之间暂不交互
   - 不做多轮传播
   - behavior_parameters 暂未深度参与状态更新
8. 后续扩展方向：
   - 多轮传播
   - KOL 先发声、普通用户后响应
   - 根据 influence_score 和 kol_sensitivity_score 构造影响权重
   - 引入群体情绪状态更新
   - 使用 AgentScope memory 模块进行更精细的记忆检索

### 十三、建议实现顺序

请按以下顺序实现，不要一开始就做复杂多 Agent 传播：

1. 阅读 AgentScope 官方文档和当前项目依赖，确认 ReActAgent、model、formatter、memory 的实际 API。
2. 实现 JSONL 数据加载与按 agent_id 合并。
3. 实现 ReactionSchema 和 JSON 输出解析。
4. 实现 dry_run，确认单个 event + 单个 agent 的输入消息合理。
5. 实现单个 Agent 的真实调用。
6. 扩展到 max_agents 批量调用。
7. 实现结果逐条保存和 resume。
8. 实现 result_analyzer.py。
9. 补充文档。
10. 运行一个小规模测试，例如 event_5223110724290198 + 5 个 core agent，确认输出结构正确。

### 十四、特别注意

本阶段的目标是“静态画像驱动的单轮群体反应模拟”，不是完整的动态传播系统。

最终交付应能做到：

给定一个事件 ID 和一批 Agent，运行脚本后得到 agent_reactions.jsonl 和 summary_report.csv，可以用于后续可视化与论文实验分析。