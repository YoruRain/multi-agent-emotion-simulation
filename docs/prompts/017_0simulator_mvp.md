你接下来要在现有单事件微博用户 Agent 仿真器的基础上，增量实现“单事件多轮群体情绪演化模拟”功能。

重要背景：
当前项目已经实现了静态的单事件 Agent 响应模拟器：给定 event_id，加载事件、Agent 画像、记忆和系统提示词，让每个 Agent 对事件产生一次结构化 JSON 反应，并输出 agent_reactions.jsonl 和 summary_report.csv。

接下来的目标是在现有代码基础上增加一个新的多轮仿真层，使系统形成：

事件输入
→ Agent 初始状态
→ KOL / 高影响力 Agent 先发声
→ 普通 Agent 根据事件和已有评论产生反应
→ 构建互动边
→ 更新 Agent 情绪与立场状态
→ 输出每轮群体统计
→ 进行网络分析与可视化

请注意：
1. 不要重写已有单事件模拟器。
2. 优先复用现有的数据加载、ReactionSchema、Agent 调用、JSON 解析、结果保存、resume、dry_run、logging 等能力。
3. 新功能应尽量放在新的模块中，避免破坏已有功能。
4. 所有新功能必须有 fallback 逻辑，即使没有配置 LLM API，也可以用规则模板跑通 demo。
5. 当前目标是毕业设计验收用的 MVP，优先保证稳定、可解释、可展示，而不是追求复杂真实传播模型。
6. 不要让每个 Agent 在每一轮都无限制调用 LLM。需要支持 max_llm_agents_per_round 或 use_llm=False。
7. 输出文件要足够完整，便于后续 Streamlit 可视化读取。