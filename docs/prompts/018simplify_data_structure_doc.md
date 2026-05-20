请阅读并修改当前的“项目关键数据结构说明”文档，将其压缩改写为适合毕业论文正文中“3.5 关键数据结构设计”部分使用的版本。

修改目标：
当前文档偏向工程字段说明，字段数量较多，适合作为开发文档或附录。请将其改写为论文正文风格，重点体现四类关键数据结构的设计目的、字段分组、数据流关系和系统作用，避免逐一罗列所有工程字段。

总体修改要求：

1. 不要完整保留所有字段表。
   每个数据结构只保留最能体现设计意图的核心字段，原则上每个结构正文中展示 8—12 个代表性字段即可。

2. 优先采用“字段类别表”，而不是“全字段定义表”。
   表格建议包含以下列：
   - 字段类别
   - 代表字段
   - 设计作用

   例如：
   | 字段类别 | 代表字段 | 设计作用 |
   | --- | --- | --- |
   | 标识字段 | event_id、weibo_id | 用于事件唯一标识和原始数据回溯 |
   | 语义字段 | topic、event_context、event_type | 描述事件主题、背景和类型 |
   | 情绪字段 | event_emotion_tendency、dominant_emotion_label、emotion_distribution | 表示事件讨论中的总体情绪倾向 |
   | 立场字段 | event_stance_focus、dominant_stance_label、stance_distribution | 表示主要评价对象和立场分布 |

3. 每个小节采用统一写法：
   - 第一段：说明该数据结构的设计目的和在系统中的作用；
   - 第二部分：使用字段类别表说明核心字段；
   - 第三段：说明该结构与其他数据结构之间的关系；
   - 必要时加入一个简化 JSON 示例，但不要放完整工程结构。

4. 删除或合并过细的工程实现字段。
   例如：
   - 审计字段、调试字段、解析状态字段、更新时间字段、异常排查字段，不必进入正文；
   - 仅用于代码内部判断的辅助字段，不必逐项说明；
   - 如果某些字段只用于回溯、复现或可视化，可以在文字中概括说明，不要全部列入表格。


各小节具体压缩要求：

一、事件数据结构

请将事件数据结构压缩为以下几类字段：
- 标识字段：event_id、weibo_id
- 语义字段：topic、event_context、event_type
- 情绪字段：event_emotion_tendency、dominant_emotion_label、emotion_distribution
- 立场字段：event_stance_focus、dominant_stance_label、stance_distribution

重点说明：
事件数据是模拟系统的外部刺激源，用于提供事件背景、评论区总体情绪和主要立场焦点。它在模拟过程中不被改写，主要用于 Prompt 构造、初始状态生成和动态更新参数计算。

不要完整保留原来的字段表。可以保留一个简化 JSON 示例，例如只展示 event_id、topic、event_context、event_emotion_tendency、event_stance_focus、emotion_distribution、stance_distribution。

二、用户画像数据结构

请将用户画像数据结构压缩为以下几类字段：
- 身份属性字段：agent_id、user_id、verified_type_name、user_level
- 情绪画像字段：emotion_summary、pos_ratio、neg_ratio
- 主题画像字段：topic_summary、public_issue_topic_ratio
- 传播画像字段：propagation_role、propagation_summary、influence_score、repost_ratio、media_dependency_score、kol_sensitivity_score
- 记忆与提示词字段：memories[].content、sys_prompt

重点说明：
用户画像用于把微博用户的长期行为特征、情绪倾向、主题偏好和传播习惯压缩为 Agent 可用的建模输入。它主要服务于 Agent 初始化、Prompt 渲染、参与倾向估计、互动权重计算和分组分析。

不要列出所有原始画像字段。重点突出“长期属性如何转化为 Agent 建模输入”。

三、智能体状态数据结构

请将智能体状态数据结构压缩为以下几类字段：
- 标识字段：run_id、event_id、agent_id、round_id
- 静态引用字段：user_level、propagation_role、influence_score、activity_score、susceptibility_score
- 情绪状态字段：emotion_score、emotion_label、emotion_delta
- 立场状态字段：stance_score、stance_label、stance_delta
- 行为状态字段：is_active、last_action_type、last_reaction_text
- 社会影响字段：neighbor_emotion_score、neighbor_stance_score、neighbor_count、neighbor_influence_weight_sum
- 事件与表达影响字段：event_emotion_score、event_stance_score、own_reaction_emotion_score、own_reaction_stance_score

重点说明：
AgentState 是模拟过程中的核心动态结构，用于表示某个 Agent 在某一轮中的情绪、立场、行为和受影响状态。它既继承用户画像中的长期属性，又会在多轮模拟过程中根据事件刺激、邻居影响和自身表达不断更新。

这一节不要写成超长字段表。可以重点解释：
- 第 0 轮状态由用户画像和事件数据初始化；
- 第 1 到 N 轮状态由上一轮状态、本轮参与行为、邻居影响和事件刺激共同更新；
- 状态数据最终进入群体指标计算和可视化分析。

四、模拟结果数据结构

这一节不要展开所有结果文件的完整字段。请先保留一张“输出文件—记录内容—设计作用”的概览表，字段包括：
- config.json：记录运行配置；
- selected_event.json：记录事件快照；
- agent_initial_states.csv：记录初始状态；
- agent_states_by_round.csv：记录各轮 Agent 状态；
- active_reactions.jsonl：记录可见反应；
- round_metrics.csv：记录群体统计指标；
- interactions.csv：记录互动边；
- dynamics_summary.json：记录动态汇总结果。

然后分别用简短段落概括：
- active_reactions.jsonl 主要回答“谁在第几轮说了什么”；
- agent_states_by_round.csv 主要回答“每个 Agent 的状态如何变化”；
- round_metrics.csv 主要回答“群体情绪、立场和参与情况如何演化”；
- interactions.csv 主要回答“哪些 Agent 之间形成了候选影响关系”；
- dynamics_summary.json 主要回答“本次模拟初末状态变化和整体动态结果如何”。

不要把 active_reactions、round_metrics、interactions、dynamics_summary 的所有字段全部写入正文。只保留代表性字段或字段组说明。

文风要求：

1. 不要过多出现“当前代码中”“fallback”“parse_status”等偏工程调试的说法，除非确有必要。
2. 保留必要的技术准确性，例如 JSONL、CSV、AgentState、Prompt、Agent 等术语可以保留。
3. 每个小节篇幅适中，避免字段表过长。
4. 文中不要出现对相关文件的引用超链接。
5. 新生成的文档中先不生成数据流图。

输出要求：

请直接在原文档基础上，在同目录下生成一个压缩后的 Markdown 版本。