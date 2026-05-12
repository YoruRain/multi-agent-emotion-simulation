你现在需要为项目新建一个 Streamlit 可视化面板，用于展示“多智能体群体情绪演化仿真系统”的运行结果，并支持通过面板直接调用已有 CLI 脚本运行新的仿真实验。

技术栈：
- Streamlit
- Plotly
- Pandas
- NetworkX
- PyVis
- subprocess

## 一、任务目标

请新建一个 Streamlit 应用，实现两个核心功能：

功能 1：读取已有仿真结果并可视化

读取 scope/outputs/simulation/multiround/{run_id}/ 下的仿真输出文件，展示：
- 仿真运行概览
- Agent 群体画像
- 情绪演化趋势
- 立场演化趋势
- 邻居影响与互动规模
- Agent 评论流
- 互动网络图
- KOL / 关键节点指标

功能 2：通过面板运行新的仿真

在页面中提供参数配置区域，用户可以设置：
- event_id
- max_agents
- rounds
- seed
- use_llm
- enable_interactions
- interaction_mode
- kol_speaker_limit
- top_k_context_comments
- enable_emotion_dynamics

点击按钮后，面板通过 subprocess 调用已有 CLI 脚本：

scope/run_multiround_simulation.py

运行完成后，自动扫描最新 run 目录，并刷新页面展示新的仿真结果。

## 二、重要限制

请严格遵守以下限制：

1. 不要重写仿真核心逻辑。
2. 不要在 Streamlit 中重新实现 MultiRoundSimulator。
3. 不要直接 import 并调用仿真器内部复杂对象，优先通过 CLI 脚本运行。
4. 不要在页面启动时自动运行仿真。
5. 只有用户点击“运行仿真”按钮时才调用 CLI。
6. CLI 运行失败时，不要让页面崩溃，应显示错误信息、stdout 和 stderr。
7. 支持读取已有 run 结果，即使不运行新仿真也能展示。
8. 如果某些文件或字段缺失，应给出友好提示，而不是报错中断。
9. 不要伪造数据。
10. 页面语言使用中文。
11. 默认展示应适合毕业设计验收现场演示。
12. 所有路径使用 pathlib。
13. 代码必须模块化、清晰、有必要注释。
14. 不要修改已有仿真输出文件，只读取和展示。

## 三、建议新增文件

请新建：

- scope/visualization/simulation_dashboard.py

如有必要，可以额外新建：

- scope/visualization/simulation_dashboard_utils.py
- scope/visualization/simulation_network.py


运行命令：

streamlit run scope/visualization/simulation_dashboard.py

## 四、输入输出目录

默认仿真输出根目录：

scope/outputs/simulation/multiround/

每次运行会生成一个 run_id 子目录，例如：

scope/outputs/simulation/multiround/20260512_143802_58c202/

每个 run 目录下可能包含以下文件：

标准文件名：
- config.json
- selected_event.json
- dynamics_summary.json
- agent_initial_states.csv
- agent_states_by_round.csv
- active_reactions.jsonl
- interactions.csv
- network.graphml
- round_metrics.csv

## 五、数据加载函数要求

请实现以下函数或等价逻辑：

1. list_simulation_runs(base_dir: Path) -> list[Path]

功能：
扫描 multiround 输出目录下的 run 子目录，按修改时间倒序排列。


2. load_json(path: Path | None) -> dict

3. load_jsonl(path: Path | None) -> pd.DataFrame

4. load_csv(path: Path | None) -> pd.DataFrame

5. load_graph(path: Path | None) -> nx.DiGraph | None

要求：
- CSV 使用 UTF-8 读取。
- JSONL 一行一个 JSON。
- GraphML 使用 networkx.read_graphml。
- 读取失败时捕获异常并返回空对象。
- 页面上展示加载状态。

## 六、CLI 运行功能

请在面板中实现“运行新仿真”的功能。

请使用 subprocess.run 调用已有脚本：

scope/run_multiround_simulation.py

不要使用 shell=True。

请使用 sys.executable 调用当前 Python 解释器。

示例：

import subprocess
import sys

cmd = [
    sys.executable,
    "scope/run_multiround_simulation.py",
    "--event-id", event_id,
    "--max-agents", str(max_agents),
    "--rounds", str(rounds),
    "--seed", str(seed),
    "--use-llm", "false",
]

根据用户选择动态追加：

- --enable-interactions
- --interaction-mode
- --kol-speaker-limit
- --top-k-context-comments
- --enable-emotion-dynamics

如果 use_llm=True，则传入：

--use-llm true

如果 use_llm=False，则传入：

--use-llm false

运行时要求：

1. 点击按钮后才运行。
2. 使用 st.spinner 或 st.status 显示运行状态。
3. 使用 capture_output=True。
4. 使用 text=True。
5. timeout 默认设置为 300 秒。
6. 运行完成后展示 stdout / stderr。
7. stdout / stderr 放入 st.expander 中，避免页面过长。
8. returncode != 0 时，st.error 显示失败。
9. returncode == 0 时，st.success 显示成功。
10. 运行成功后重新扫描输出目录，自动选择最新 run 目录。
11. 将最新 run 目录保存到 st.session_state["selected_run_dir"]。
12. 自动刷新数据展示。

推荐实现逻辑：

运行前：
- 记录当前 multiround 目录下已有 run 目录集合。

运行后：
- 重新扫描 run 目录。
- 如果发现新增目录，选择新增目录中修改时间最新的。
- 如果没有发现新增目录，则选择修改时间最新的 run 目录。
- 如果仍无法识别，提示用户手动从下拉框选择。

## 七、运行参数 UI 设计

请在页面侧边栏或单独 Tab 中提供“运行新仿真”区域。

基础参数：

- event_id: text_input，默认 event_5177192956301027
- max_agents: number_input，默认 30，范围 5~200
- rounds: number_input，默认 5，范围 1~20
- seed: number_input，默认 42
- use_llm: checkbox，默认 False

互动参数：

- enable_interactions: checkbox，默认 True
- interaction_mode: selectbox，默认 kol_first，可选 none / kol_first
- kol_speaker_limit: number_input，默认 5，范围 1~20
- top_k_context_comments: number_input，默认 3，范围 1~10

动态参数：

- enable_emotion_dynamics: checkbox，默认 True

高级参数放入：

st.expander("高级运行参数")

可选高级参数：
- timeout 秒数，默认 300
- output_base_dir
- 是否运行后自动切换到最新 run


## 八、页面总体结构

请使用：

st.set_page_config(
    page_title="多智能体群体情绪演化仿真面板",
    layout="wide"
)

页面标题：

“多智能体群体情绪演化仿真面板”

建议使用以下 Tab：

Tab 1：运行与结果选择  
Tab 2：仿真运行概览  
Tab 3：Agent 群体画像  
Tab 4：情绪与立场演化  
Tab 5：互动网络与关键节点  
Tab 6：评论流与状态明细  

## 九、Tab 1：运行与结果选择

功能目标：
同时支持运行新仿真和选择已有结果。

内容：

1. 当前输出根目录
- text_input
- 默认 scope/outputs/simulation/multiround

2. 已有 run 选择
- selectbox 展示 run_id
- 按修改时间倒序排列
- 默认选择最新 run
- 如果 st.session_state["selected_run_dir"] 存在，优先选中它

3. 文件读取状态
展示实际读取文件名：
- dynamics_summary
- agent_initial_states
- agent_states_by_round
- active_reactions
- interactions
- network
- round_metrics

4. 运行新仿真区域
展示参数配置和运行按钮。

5. 运行输出
- 成功/失败状态
- stdout
- stderr
- 新 run 目录路径

## 十、Tab 2：仿真运行概览

数据来源：
优先使用 dynamics_summary.json。

展示内容：

1. 事件信息：
- run_id
- event_id
- topic
- dynamics_enabled
- interaction_enabled

2. 指标卡：
- Agent 数量
- 仿真轮数
- 总互动边数
- 受邻居影响 Agent 数
- 初始平均情绪
- 最终平均情绪
- 初始平均立场
- 最终平均立场
- 情绪变化量
- 立场变化量

3. 简短说明：
“本次仿真以热点事件为输入，初始化一批微博用户 Agent，在多轮评论区互动中模拟高影响力用户先发声、普通用户观察并响应，以及由互动边驱动的情绪传染与立场演化。”

4. dynamics_summary 原始 JSON：
放入 expander 中展示。

## 十一、Tab 3：Agent 群体画像

数据来源：
agent_initial_states.csv

展示目标：
说明 Agent 由用户画像映射而来，不是随机节点。

展示内容：

1. 指标卡：
- 总 Agent 数
- core / normal / background 数量
- 平均 influence_score
- 平均 susceptibility_score
- 平均 activity_score

2. Plotly 图表：
- memory_user_level 分布
- verified_type_name 分布
- propagation_role 分布
  注意 propagation_role 可能是逗号分隔的多个角色，需要展开统计。
- influence_score 直方图
- susceptibility_score 直方图
- activity_score 直方图
- 初始 emotion_score 分布
- 初始 stance_score 分布

3. Agent 表格：
字段包括：
- agent_id
- user_id
- memory_user_level
- verified_type_name
- propagation_role
- influence_score
- susceptibility_score
- activity_score
- emotion_score
- stance_score
- emotion_label
- stance_label

数值保留 3 位小数。

## 十二、Tab 4：情绪与立场演化

数据来源：
round_metrics.csv
agent_states_by_round.csv

展示目标：
说明群体情绪和立场随轮次发生变化。

图表：

1. 群体平均情绪随轮次变化
x = round_id
y = avg_emotion_score

2. 群体平均立场随轮次变化
x = round_id
y = avg_stance_score

3. 情绪分布变化
使用：
- positive_ratio
- neutral_ratio
- negative_ratio

4. 立场分布变化
使用：
- support_ratio
- neutral_stance_ratio
- oppose_ratio

5. 情绪波动与立场极化
如果字段存在：
- emotion_volatility
- stance_volatility
- polarization_score

6. 状态变化强度
如果字段存在：
- avg_abs_emotion_delta
- avg_abs_stance_delta
- max_abs_emotion_delta
- max_abs_stance_delta

7. 邻居影响指标
如果字段存在：
- avg_neighbor_count
- agents_affected_by_neighbors
- avg_neighbor_influence_weight

8. 每轮互动规模
使用：
- interaction_count
- avg_interaction_weight
- high_influence_interaction_count

说明文字：
- 情绪分数范围为 [-1, 1]，越低表示越偏负向。
- 立场分数范围为 [-1, 1]，越低表示越偏反对，越高表示越偏支持。
- polarization_score 表示立场分布离散程度。
- agents_affected_by_neighbors 表示每轮受到互动边影响的 Agent 数。

要求：
- 使用 Plotly。
- 字段不存在时跳过对应图表并 st.info 提示。
- 侧边栏提供“是否显示第 0 轮”选项。

## 十三、Tab 5：互动网络与关键节点

数据来源：
network.graphml
interactions.csv
agent_initial_states.csv
agent_states_by_round.csv

功能 1：网络概览

展示：
- 节点数
- 边数
- 网络密度
- 平均入度
- 平均出度
- 最大入度
- 最大出度

功能 2：中心性指标

使用 NetworkX 计算：

- in_degree
- out_degree
- degree_centrality
- in_degree_centrality
- out_degree_centrality
- betweenness_centrality
- pagerank

将结果与 Agent 画像合并。

展示 Top K 表格：

- agent_id
- pagerank
- degree_centrality
- in_degree
- out_degree
- betweenness_centrality
- influence_score
- susceptibility_score
- propagation_role
- final_emotion_score
- final_stance_score

Top K 默认 10，由侧边栏控制。

功能 3：PyVis 网络图

使用 PyVis 渲染 Agent 互动网络。

节点：
- ID: agent_id
- label: agent_id 后 4~6 位
- size: 根据 influence_score 或 pagerank
- hover 展示：
  - agent_id
  - memory_user_level
  - propagation_role
  - influence_score
  - susceptibility_score
  - final_emotion_score
  - final_stance_score

边：
- 宽度根据 weight_sum 或 interaction_count
- hover 展示：
  - interaction_count
  - weight_sum
  - first_round
  - last_round
  - interaction_types

网络图参数：
- 最大边数由侧边栏控制，默认 100
- 选择边时优先保留 weight_sum 或 interaction_count 较高的边
- 如果 PyVis 渲染失败，展示错误提示和中心性表格

功能 4：互动边统计

Plotly 图：
- 每轮 interaction_count
- 每轮 avg_interaction_weight
- interaction_type 分布
- source_influence_score 分布
- high_influence_interaction_count 随轮次变化

如果 network.graphml 缺失：
- 从 interactions.csv 构建 DiGraph
- 节点来自 agent_initial_states.csv
- 边来自 interactions.csv
- 同一 source-target 多次出现时合并边

## 十四、Tab 6：评论流与状态明细

数据来源：
active_reactions.jsonl
agent_states_by_round.csv
interactions.csv

功能 1：按轮次展示评论流

选择 round_id。

展示该轮 active reactions。

字段：
- speaker_type
- agent_id
- memory_user_level
- propagation_role
- action_type
- emotion_label
- emotion_intensity
- stance_label
- stance_intensity
- reaction_text
- context_comment_count
- influenced_by_high_influence

要求：
- KOL speaker 和 regular agent 分开展示。
- 每轮默认展示前 N 条，N 由侧边栏控制。
- 完整表格放入 expander。

功能 2：Agent 状态轨迹

提供 agent_id 下拉选择。

展示该 Agent 各轮：

- emotion_score
- stance_score
- emotion_delta
- stance_delta
- neighbor_count
- state_update_reason

用 Plotly 画：
- emotion_score 折线
- stance_score 折线

功能 3：邻居影响明细

选择 target_agent_id 和 round_id。

展示 interactions.csv 中对应 source_agent：

- source_agent_id
- interaction_type
- weight
- source_reaction_text
- target_reaction_text
- source_emotion_score
- source_stance_score
- source_influence_score

如果没有对应边，显示提示。

## 十五、PyVis 嵌入要求

请实现稳定的 PyVis 嵌入方式：

from pyvis.network import Network
import streamlit.components.v1 as components

基本流程：

1. 创建 Network(height="650px", width="100%", directed=True, notebook=False)
2. 添加节点和边
3. 写入临时 HTML 文件
4. 读取 HTML 字符串
5. components.html(html, height=700, scrolling=True)

要求：
- 使用 tempfile 或固定临时目录。
- 捕获异常。
- 不要让 PyVis 失败导致整个页面崩溃。

## 十六、通用图表函数

建议实现：

- truncate_text(text, max_len=80)
- safe_round(value, ndigits=3)
- plot_line(df, x, y, title)
- plot_multi_line(df, x, y_cols, title)
- plot_bar(df, x, y, title)
- plot_stacked_ratios(df, x, ratio_cols, title)
- explode_roles(series)
- build_graph_from_interactions(interactions, agents)

图表使用 Plotly Express 或 graph_objects 均可。

## 十七、健壮性要求

1. 文件缺失时显示 st.warning。
2. 字段缺失时显示 st.info，并跳过对应图表。
3. DataFrame 为空时不绘图。
4. JSON 解析失败时显示错误。
5. GraphML 读取失败时尝试从 interactions.csv 重建。
6. subprocess 运行失败时展示 stderr。
7. subprocess timeout 时提示运行超时。
8. 页面启动默认加载最新 run。
9. 不要因为 PyVis 缺失导致页面完全不可用。
10. 所有 dataframe 展示前注意 copy。
11. 不要硬编码绝对路径。
12. 中文文本显示要正常。
13. 大图默认限制边数，避免浏览器卡顿。

## 十八、运行说明

请在页面底部或注释中说明：

运行命令：

streamlit run scope/visualization/simulation_dashboard.py

依赖：

streamlit
pandas
plotly
networkx
pyvis

面板使用方式：

1. 启动页面后，默认读取最新已有仿真结果。
2. 可以在“运行与结果选择”页选择已有 run。
3. 可以设置参数并点击“运行仿真”。
4. 运行成功后，面板自动切换到新生成的 run 结果。
5. 运行失败时，在页面中查看 stdout / stderr。

## 十九、最终交付说明

完成后请输出：

1. 新增了哪些文件。
2. 如何运行面板。
3. 如何读取已有结果。
4. 如何通过面板运行新仿真。
5. 运行失败时如何查看错误。
6. 运行成功后如何自动切换结果。
7. 面板包含哪些 Tab。
8. 每个 Tab 使用哪些数据文件。
9. 如果缺少 network.graphml 或 PyVis，如何降级展示。
10. 建议 git commit message。


本任务最终目标：
新增一个独立的 Streamlit 面板，既可以读取已有多轮仿真结果进行可视化，也可以通过页面参数调用现有 CLI 脚本运行新的仿真实验，并自动加载新结果，为毕业设计验收提供可交互、可运行、可解释的演示界面。