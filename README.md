# 基于多智能体的社会群体情绪模拟系统

本仓库是本科毕业设计项目“基于多智能体的社会群体情绪模拟系统的设计与实现”的代码与实验材料。

项目围绕微博热点事件，完成从原始社交媒体数据清洗、评论情绪与立场分析、用户长期画像构建，到微博用户 Agent 单事件反应模拟、多轮交互式情绪演化仿真和可视化展示的一套原型系统。

项目目标是把真实微博数据映射为可解释的用户 Agent，观察不同影响力、不同情绪倾向和不同传播角色的用户在公共事件讨论中的参与、互动、情绪传播和立场演化过程。

## 核心功能

- 微博热点事件、评论、用户信息和用户历史微博的数据整理与高质量样本筛选，主要见 [notebooks/](notebooks/) 与 [data_loader.py](apps/dashboard_midterm/data_loader.py)。
- 基于 DeepSeek API 的评论级情绪、立场、责任归因和规范违背分析，支持断点续跑、并发请求和失败样本记录，主要见 [batch_comment_analysis.py](scripts/comment_analysis/batch_comment_analysis.py)、[schema.py](scripts/comment_analysis/schema.py) 和 [deepseek_client.py](scripts/comment_analysis/deepseek_client.py)。
- 基于微调 MacBERT 的用户历史微博情绪推理，并聚合为用户长期情绪画像，主要见 [infer.py](scripts/weibo_analysis/long_term_emotion_profile/infer.py) 和 [build_user_emotion_profile.py](scripts/weibo_analysis/long_term_emotion_profile/build_user_emotion_profile.py)。
- 从用户基础信息、情绪画像、主题画像、传播画像和记忆样本构建微博用户 Agent 输入文件，主要见 [profile_builder.py](scripts/agent_data_preparation/profile_builder.py)、[memory_builder.py](scripts/agent_data_preparation/memory_builder.py)、[event_builder.py](scripts/agent_data_preparation/event_builder.py) 和 [prompt_renderer.py](scripts/agent_data_preparation/prompt_renderer.py)。
- 单事件模拟：给定热点事件，生成不同用户 Agent 的结构化微博式反应，主要见 [run_single_event_simulation.py](scope/run_single_event_simulation.py)、[single_event_simulator.py](scope/src/simulation/single_event_simulator.py) 和 [reaction_schema.py](scope/src/simulation/reaction_schema.py)。
- 多轮模拟：记录每个 Agent 的情绪分数、立场分数、活跃状态和群体统计指标，主要见 [run_multiround_simulation.py](scope/run_multiround_simulation.py)、[multiround_simulator.py](scope/src/simulation/multiround_simulator.py)、[agent_state.py](scope/src/simulation/agent_state.py) 和 [multiround_analyzer.py](scope/src/simulation/multiround_analyzer.py)。
- KOL 优先互动：高影响力 Agent 先发声，普通 Agent 基于可见评论上下文响应，并生成候选影响边，主要见 [interaction_engine.py](scope/src/simulation/interaction_engine.py)、[interaction_schema.py](scope/src/simulation/interaction_schema.py) 和 [network_builder.py](scope/src/simulation/network_builder.py)。
- 规则化情绪动态：基于邻居影响、事件刺激、自身表达和个体易感性更新情绪与立场状态，主要见 [emotion_dynamics.py](scope/src/simulation/emotion_dynamics.py) 和 [multiround_interaction_emotion_flow.md](scope/docs/multiround_interaction_emotion_flow.md)。
- Streamlit / Plotly / PyVis 可视化：展示数据概览、评论网络、多轮仿真指标、Agent 状态和交互网络，主要见 [app.py](apps/dashboard_midterm/app.py)、[simulation_dashboard.py](scope/visualization/simulation_dashboard.py) 和 [simulation_network.py](scope/visualization/simulation_network.py)。

## 项目结构

```text
.
├── apps\dashboard_midterm\          # 数据探索与中期展示用 Streamlit 面板
├── data\                            # 原始、清洗、高质量样本与画像结果数据
├── docs\                            # 数据字典、提示词与阶段性实现提示
├── models\macbert_finetuned_sentiment\
│                                      # 用户微博情绪推理模型
├── notebooks\                        # 数据读取、清洗、特征工程和样本筛选 Notebook
├── papers\                           # 相关文献资料
├── scripts\
│   ├── comment_analysis\             # 评论情绪与立场批处理分析
│   ├── weibo_analysis\               # 用户情绪、主题、传播与记忆样本画像
│   └── agent_data_preparation\       # 生成 Agent 仿真输入 JSONL
├── scope\
│   ├── data\inputs\                  # 仿真输入：events / profiles / memories / prompts
│   ├── data\outputs\simulation\      # 单事件与多轮仿真输出
│   ├── docs\                         # 仿真模块说明文档
│   ├── src\simulation\               # Agent 仿真核心实现
│   ├── tests\                        # 仿真相关测试
│   ├── visualization\                # 多轮仿真结果可视化面板
│   ├── run_single_event_simulation.py
│   └── run_multiround_simulation.py
├── explanation.md                    # 任务书与开题报告背景
└── README.md
```

## 数据与处理流水线

项目当前使用的基础数据包括话题微博、话题评论、评论用户信息和用户历史微博。`docs/data_dictionary_cleaned.md` 记录了清洗后数据规模与字段概要：话题微博约 202 条、话题评论约 19093 条、评论用户约 5069 人、用户历史微博约 243464 条。

可理解为以下流水线：

```text
原始/清洗微博数据
  -> 高质量样本筛选
  -> 评论情绪与立场分析
  -> 用户长期情绪画像、主题画像、传播画像、记忆样本
  -> Agent profiles / memories / system prompts / events
  -> 单事件反应模拟或多轮群体情绪演化模拟
  -> 指标、网络与可视化面板
```

仿真模块默认读取：

```text
scope\data\inputs\events.jsonl
scope\data\inputs\agent_profiles.jsonl
scope\data\inputs\agent_memories.jsonl
scope\data\inputs\agent_sys_prompts.jsonl
```

这些文件由 `scripts\agent_data_preparation\` 下的脚本从 `data\profile\` 画像结果生成。

## 环境说明

本项目主要在 Windows 与 Conda 环境下开发。运行仓库内 Python 脚本时，优先使用项目环境：

```powershell
conda run -p .\.gp python <script.py>
```

常用依赖包括 `pandas`、`numpy`、`pyarrow`、`streamlit`、`plotly`、`networkx`、`pyvis`、`python-dotenv`、`torch`、`transformers`、`scikit-learn` 等。多轮仿真在 `--use-llm false` 时可以使用规则 fallback 离线运行；如果开启 LLM 生成，需要安装 `agentscope` 并配置 API key。

`.env` 可配置：

```env
DEEPSEEK_API_KEY=你的_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
MODEL_NAME=deepseek-chat
BASE_URL=https://api.deepseek.com/v1
```

## 常用运行命令

### 1. 评论情绪与立场分析

```powershell
conda run -p .\.gp python scripts\comment_analysis\batch_comment_analysis.py `
  --limit 20 `
  --concurrency 2 `
  --comment-level first
```

默认输入为 `data\high_quality\topic_comment.parquet` 与 `data\high_quality\topic_weibo.parquet`，默认输出到：

```text
data\profile\comments\comment_analysis_result.jsonl
data\profile\comments\comment_analysis_result.parquet
data\profile\comments\comment_analysis_failed.jsonl
```

### 2. 用户长期情绪画像

微博级情绪推理：

```powershell
conda run -p .\.gp python scripts\weibo_analysis\long_term_emotion_profile\infer.py `
  --max_records 100 `
  --random_sample
```

用户级画像聚合：

```powershell
conda run -p .\.gp python scripts\weibo_analysis\long_term_emotion_profile\build_user_emotion_profile.py
```

### 3. 构建仿真输入

```powershell
conda run -p .\.gp python scripts\agent_data_preparation\event_builder.py
conda run -p .\.gp python scripts\agent_data_preparation\memory_builder.py
conda run -p .\.gp python scripts\agent_data_preparation\profile_builder.py
conda run -p .\.gp python scripts\agent_data_preparation\prompt_renderer.py
```

生成结果位于 `scope\data\inputs\`，供单事件与多轮仿真直接读取。

### 4. 单事件 Agent 反应模拟

先 dry run 检查事件和 Agent 输入：

```powershell
conda run -p .\.gp python scope\run_single_event_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 3 `
  --memory-user-level core `
  --dry-run
```

正式运行：

```powershell
conda run -p .\.gp python scope\run_single_event_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 5 `
  --memory-user-level core `
  --seed 42
```

输出目录：

```text
scope\data\outputs\simulation\single_event\{run_id}\
```

主要产物包括 `agent_reactions.jsonl`、`summary_report.csv`、`memory_user_level_summary.csv`、`influence_level_summary.csv` 和 `action_type_summary.csv`。

### 5. 多轮互动与情绪演化模拟

规则 fallback 版本，适合答辩演示和离线复现：

```powershell
conda run -p .\.gp python scope\run_multiround_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 30 `
  --rounds 5 `
  --active-agent-limit 12 `
  --enable-interactions `
  --enable-emotion-dynamics `
  --seed 42
```

如需让部分活跃 Agent 调用 LLM 生成反应，可开启预算控制：

```powershell
conda run -p .\.gp python scope\run_multiround_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 30 `
  --rounds 5 `
  --enable-interactions `
  --enable-emotion-dynamics `
  --use-llm true `
  --max-llm-agents-per-round 5 `
  --llm-concurrency 2
```

多轮输出目录：

```text
scope\data\outputs\simulation\multiround\{run_id}\
```

主要产物包括：

- `config.json`：本次运行配置。
- `selected_event.json`：目标事件记录。
- `agent_initial_states.csv`：第 0 轮 Agent 初始状态。
- `agent_states_by_round.csv`：所有轮次的 Agent 状态快照。
- `round_metrics.csv`：每轮群体情绪、立场和活跃度指标。
- `active_reactions.jsonl`：活跃 Agent 的反应文本、来源和解析信息。
- `interactions.csv`：Agent 间候选影响边。
- `network.graphml`：用于网络分析和可视化的互动图。
- `dynamics_summary.json`：最终情绪/立场变化、波动性、极化和邻居影响覆盖摘要。

## 可视化

### 数据探索面板

```powershell
cd .\apps\dashboard_midterm
conda run -p .\.gp python -m streamlit run app.py
```

该面板默认读取 `data\high_quality\`，用于查看话题微博、评论链路、用户信息和历史微博数据。

### 多轮仿真面板

```powershell
conda run -p .\.gp python -m streamlit run scope\visualization\simulation_dashboard.py
```

该面板默认读取 `scope\data\outputs\simulation\multiround\`，可选择已有 run，查看事件摘要、轮次指标、Agent 状态、活跃反应、互动边、网络中心性和 PyVis 交互图。

## 实现要点

多轮仿真没有重写单事件模拟器，而是在 `scope\src\simulation\` 中按模块逐步扩展：

- `agent_loader.py` 与 `event_loader.py` 负责读取事件和 Agent 输入。
- `agent_state.py` 定义 Agent 在每一轮的情绪、立场、活跃和行为状态。
- `single_event_simulator.py` 实现静态画像驱动的单事件反应模拟。
- `multiround_simulator.py` 负责多轮调度、状态记录和输出。
- `interaction_engine.py` 实现 KOL 优先发声和上下文影响边构造。
- `emotion_dynamics.py` 实现规则化情绪与立场更新。
- `llm_reaction_generator.py` 封装可选 AgentScope / OpenAI-compatible LLM 调用。
- `multiround_analyzer.py` 与 `network_builder.py` 生成群体指标和 GraphML 网络。

情绪与立场动态使用可解释的加权更新：上一轮自身状态、邻居评论影响、事件刺激和本轮自身表达共同决定新状态，并通过饱和阻尼避免分数过快撞到边界。默认 `use_llm=false` 时仍能稳定输出完整模拟结果，便于无网络或无 API key 的演示场景。

## 参考文档

- `explanation.md`：毕业设计任务书与开题报告背景。
- `docs\data_dictionary_cleaned.md`：清洗后数据集字段与规模说明。
- `scripts\comment_analysis\README.md`：评论分析脚本参数说明。
- `scripts\weibo_analysis\long_term_emotion_profile\README.md`：用户长期情绪画像流程说明。
- `scope\docs\single_event_simulation.md`：单事件模拟说明。
- `scope\docs\multiround_simulation.md`：多轮仿真输入、输出和 CLI 参数说明。
- `scope\docs\multiround_interaction_emotion_flow.md`：KOL 互动与情绪动态内部流程说明。

## 当前定位

本项目当前是一个面向毕业设计展示和实验分析的原型系统。它已经具备完整的数据到 Agent 仿真的闭环，以及可解释、可复现的离线模拟路径；后续可以继续扩展更真实的传播网络、更多事件类型、更严格的模型评估，以及与论文实验章节对应的对比实验。
