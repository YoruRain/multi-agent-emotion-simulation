# 项目基本介绍

主要内容为毕业设计开题阶段撰写的任务书与开题报告，可作为项目的基本介绍。

## 1. 任务书部分

### 1.1 任务及背景

社交媒体已成为公众情绪表达和汇聚的重要场所，对社会群体情绪进行分析有助于理解社会动态、预警社会风险。

本项目旨在设计并实现一个基于多智能体的社会群体情绪分析系统。

该系统将利用多智能体框架，通过模拟不同情绪倾向的用户（Agent），对社交网络中的文本数据进行情绪识别和分类。

系统将分析群体情绪的分布、演化趋势以及关键意见领袖的影响力，并利用简化的情绪传播模型对情绪在社交网络中的扩散过程进行可视化展示。

本研究旨在探索多智能体技术在社会计算领域的应用，为舆情分析和群体心理研究提供新的工具和视角。

### 1.2 成果形式

- 开发一个社会群体情绪分析系统原型，包含数据采集、情绪分析、网络构建和可视化等模块，并提供完整的程序代码和技术文档。

- 完成一篇高质量的毕业设计论文，详细阐述系统的设计思路、技术方案、实现过程和分析结果。

### 1.3 工具、环境

- 环境：Windows 10/11 或 Linux (Ubuntu 22.04)

- 开发语言：Python 3.9+

- 核心框架：Scrapy (数据采集), NetworkX (网络分析), Matplotlib/D3.js (可视化)

- 大模型API：调用 OpenAI GPT 系列或开源情感分析模型

- 数据集：Twitter, 微博等社交媒体公开数据集

### 1.4 参考文献

[1] Stieglitz, S., & Dang-Xuan, L. (2013). Emotions and information diffusion in social media—sentiment of microblogs and sharing behavior. Journal of Management Information Systems, 29(4), 217-248.

[2] Ferrara, E., & Yang, Z. (2015). Measuring emotional contagion in social media. PloS one, 10(11), e0142399.

[3] Park, A., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. arXiv preprint arXiv:2304.03442.

[4] Stieglitz, S., et al. (2018). Social media analytics. Business & Information Systems Engineering, 60(2), 199-213.

[5] Gao, C., et al. (2023). $S^3$: Social-network Simulation System with Large Language Model-Empowered Agents. arXiv preprint arXiv:2307.14984.

[6] Koch, F., et al. (2025). Generative Intelligence Systems in the Flow of Group Emotions. arXiv preprint arXiv:2507.11831.

[7] van Haeringen, E. S., et al. (2023). Emotion contagion in agent-based simulations of crowds: a systematic review. Autonomous Agents and Multi-Agent Systems, 37(1), 6.

[8] Yin, F., et al. (2020). Influence of User Emotion on Information Propagation with Public Sentiment in the Chinese Sina-microblog. arXiv preprint arXiv:2011.07694.

[9] Cheng, Z., et al. (2025). Interactive simulation and visual analysis of social media event dynamics with LLM-based multi-agent modeling. Visual Informatics, 9, 100260.

[10] Gao, C., et al. (2023). Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives. arXiv preprint arXiv:2312.11970.

### 1.5 着重培养的能力

本课题旨在培养学生处理和分析大规模社会数据的能力。具体包括：网络数据采集与处理能力；社会网络分析与可视化能力；基于AI的情绪分析模型应用能力；系统设计与实现能力；跨学科研究与论文撰写能力。


## 2. 开题报告部分

### 2.1 课题的作用及意义

在社交媒体深度融入公众生活的当下，社交网络已成为群体情绪表达与汇聚的核心场域，其产生的情绪波动不仅反映社会心态，更直接影响公共决策与社会稳定。

本课题通过构建具备感知与推理能力的智能体（Agent），能够从微观个体的交互中模拟出宏观情绪涌现的动力学过程。

课题将多智能体系统（MAS）与情绪传播模型相结合，探索了大语言模型（LLM）赋能下的智能体在处理复杂情感文本与社交互动时的行为仿真能力，为社会计算与舆情演化研究提供了新的工具视角；

系统通过对关键意见领袖影响力的识别及情绪扩散的可视化展示，能够为相关部门实时监测舆情走向、精准预警社会风险以及制定科学的引导策略提供量化依据。

这种“自下而上”的建模方法不仅有助于揭示情绪传染的内在机理，也为理解大规模在线社交网络中的复杂人类行为模式提供了关键的技术支撑，对维护清朗的网络空间和辅助社会治理具有重要的现实意义。

### 2.2 国内外研究现状和发展趋势

当前社交媒体群体情绪研究正从宏观统计向微观行为仿真跨越。

国外研究起步较早，Stieglitz 等通过 Twitter 数据证实了情感倾向对信息扩散速度的显著影响，Ferrara 等则通过大规模实验揭示了网络中的“情感传染”机制。

近年来，随着生成式智能体概念的兴起，斯坦福与谷歌的研究者利用大语言模型赋能智能体，使其在感知、推理与社会交互上展现出高度的人类行为仿真度。

国内清华大学等团队在此领域进展迅速，$S^3$ 等系统的构建标志着社会网络模拟已能实现情感与态度的涌现。

总体趋势上，研究正由单纯的文本情感分类转向融合复杂动力学模型与大语言模型技术的深度模拟。

未来，如何利用多智能体框架精准还原意见领袖的心理博弈，并结合交互式可视化技术揭示群体情绪的演化边界，已成为国内外舆情计算与社会计算领域的共同前沿方向。

### 2.3 尚待研究的问题

首先是智能体行为的“深层对齐”问题，现有模型虽能模仿语言风格，但在复杂社会关系下，如何模拟智能体情感决策的长期连贯性与个体差异性仍需深入探索。

其次，情绪传染机理的刻画尚显单一，多源异构信息对不同性格倾向智能体的差异化冲击，以及情绪极化过程中“回声壁效应”的量化建模仍不完善。

最后，仿真环境与现实数据的动态闭环尚未完全建立，如何有效验证大规模智能体涌现现象的社会真实性，并将其转化为精准的实时舆情预测工具，是未来研究的关键突破点。

### 2.4 任务初步思路与方案

1. **数据采集与预处理**：使用爬虫系统，以2025年高热度公共热点事件为切入点，抓取核心微博及其多级评论数据，同步追溯互动用户的历史微博，随后实施数据整理与清洗流程，构建后续实验所需的高质量基础语料集。

2. **情感计算与用户画像构建**：调用大语言模型（如 Kimi, DeepSeek 等）的 API，集成心理学的相关方法（如大五人格模型、PAD 情感模型），对社交文本进行情绪提取与立场判定。通过融合用户历史行为信息与情感倾向，构建数字化用户画像，实现从“文本数据”到“智能体属性”的语义映射。

3. **基于大语言驱动的多智能体模拟**：基于 AgentScope 多智能体开发框架，构建具备认知能力的虚拟智能体集群。将用户画像映射为智能体的初始化参数，并将历史情感轨迹转化为长短期记忆网络，通过模拟社交媒体评论区的交互网络，设计情绪传染与意见演化算法，观测在特定舆论下，个体决策逻辑如何驱动群体情绪的涌现、扩散及演化过程。

4. **网络分析与可视化**：利用 NetworkX 框架构建社会网络图谱，通过各种网络指标定量评估关键意见领袖（KOL）在情绪传播中的枢纽作用。最终结合可视化框架开发交互式可视化面板，呈现群体情绪的演化态势及社交网络的结构变迁，为舆情规律研究提供直观的实证支撑。
