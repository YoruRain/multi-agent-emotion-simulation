from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from simulation_dashboard_utils import (
    build_graph_from_interactions,
    coerce_numeric,
    explode_roles,
    has_columns,
    latest_states,
    list_simulation_runs,
    load_csv,
    load_graph,
    load_json,
    load_jsonl,
    metric_value,
    missing_columns,
    plot_bar,
    plot_histogram,
    plot_line,
    plot_multi_line,
    plot_stacked_ratios,
    resolve_run_files,
    round_numeric_columns,
    safe_round,
    truncate_text,
    value_count_frame,
)
from simulation_network import (
    build_network_view_graph,
    compute_centrality_table,
    graph_overview,
    prepare_comment_table,
    render_pyvis_network,
    role_color_legend_html,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_CURRENT_OUTPUT_DIR = PROJECT_ROOT / "scope" / "data" / "outputs" / "simulation" / "multiround"
CLI_SCRIPT = PROJECT_ROOT / "scope" / "run_multiround_simulation.py"

FIELD_LABELS = {
    "agent_id": "智能体编号",
    "user_id": "用户编号",
    "memory_user_level": "用户分层",
    "verified_type_name": "认证类型",
    "propagation_role": "传播角色",
    "influence_score": "影响力分数",
    "susceptibility_score": "易感性分数",
    "activity_score": "活跃度分数",
    "emotion_score": "情绪分数",
    "stance_score": "立场分数",
    "emotion_label": "情绪标签",
    "stance_label": "立场标签",
    "pagerank": "页面排名中心性",
    "degree_centrality": "度中心性",
    "in_degree_centrality": "入度中心性",
    "out_degree_centrality": "出度中心性",
    "in_degree": "入度",
    "out_degree": "出度",
    "betweenness_centrality": "中介中心性",
    "final_emotion_score": "最终情绪分数",
    "final_stance_score": "最终立场分数",
    "round_id": "轮次",
    "avg_emotion_score": "平均情绪分数",
    "avg_stance_score": "平均立场分数",
    "positive_ratio": "正向情绪比例",
    "neutral_ratio": "中性情绪比例",
    "negative_ratio": "负向情绪比例",
    "support_ratio": "支持比例",
    "neutral_stance_ratio": "中立立场比例",
    "oppose_ratio": "反对比例",
    "emotion_volatility": "情绪波动",
    "stance_volatility": "立场波动",
    "polarization_score": "极化程度",
    "avg_abs_emotion_delta": "平均情绪变化幅度",
    "avg_abs_stance_delta": "平均立场变化幅度",
    "max_abs_emotion_delta": "最大情绪变化幅度",
    "max_abs_stance_delta": "最大立场变化幅度",
    "avg_neighbor_count": "平均邻居数量",
    "agents_affected_by_neighbors": "受邻居影响的智能体数量",
    "avg_neighbor_influence_weight": "平均邻居影响权重",
    "interaction_count": "互动次数",
    "avg_interaction_weight": "平均互动权重",
    "high_influence_interaction_count": "高影响力互动次数",
    "interaction_type": "互动类型",
    "source_influence_score": "源智能体影响力分数",
    "speaker_type": "发言者类型",
    "action_type": "行动类型",
    "emotion_intensity": "情绪强度",
    "stance_intensity": "立场强度",
    "reaction_text": "反应文本",
    "context_comment_count": "上下文评论数",
    "influenced_by_high_influence": "是否受高影响力节点影响",
    "emotion_delta": "情绪变化量",
    "stance_delta": "立场变化量",
    "neighbor_count": "邻居数量",
    "state_update_reason": "状态更新原因",
    "target_agent_id": "目标智能体编号",
    "source_agent_id": "源智能体编号",
    "weight": "权重",
    "source_reaction_text": "源反应文本",
    "target_reaction_text": "目标反应文本",
    "source_emotion_score": "源情绪分数",
    "source_stance_score": "源立场分数",
}

FILE_LABELS = {
    "dynamics_summary": "动态摘要",
    "agent_initial_states": "智能体初始状态",
    "agent_states_by_round": "智能体轮次状态",
    "active_reactions": "活跃反应记录",
    "interactions": "互动明细",
    "network": "互动网络",
    "round_metrics": "轮次指标",
}

VALUE_LABELS = {
    "core": "核心用户",
    "normal": "普通用户",
    "background": "背景用户",
    "none": "无互动",
    "kol_first": "关键意见领袖优先",
    "weight_sum": "累计权重",
    "interaction_count": "互动次数",
    "weight": "平均权重",
    "pagerank": "页面排名中心性",
    "influence_score": "影响力分数",
    "in_degree": "入度",
    "out_degree": "出度",
    "same_round_context": "同轮上下文",
    "reply": "回复",
    "repost": "转发",
    "influence_candidate": "候选影响边",
    "kol_speaker": "关键意见领袖发言者",
}


def display_label(name: Any) -> str:
    return FIELD_LABELS.get(str(name), VALUE_LABELS.get(str(name), str(name)))


def display_value(value: Any) -> Any:
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str):
        return value
    text = str(value)
    if "," in text or "，" in text:
        parts = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
        mapped = [VALUE_LABELS.get(part, part.replace("KOL", "关键意见领袖")) for part in parts]
        return "，".join(mapped)
    return VALUE_LABELS.get(text, text.replace("KOL", "关键意见领袖"))


def localize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.rename(columns={column: display_label(column) for column in df.columns}).copy()
    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = result[column].map(display_value)
    return result


def graph_source_label(source: str) -> str:
    if source == "network.graphml":
        return "图文件（network.graphml）"
    if source == "interactions.csv 重建":
        return "由互动明细重建（interactions.csv）"
    return source


def localize_plotly_figure(fig):
    if fig is None:
        return None
    fig.for_each_trace(lambda trace: trace.update(name=display_label(trace.name)) if trace.name else None)
    for trace in fig.data:
        if hasattr(trace, "x") and trace.x is not None:
            trace.x = [display_value(value) for value in trace.x]
        if hasattr(trace, "y") and trace.y is not None and any(isinstance(value, str) for value in trace.y):
            trace.y = [display_value(value) for value in trace.y]
    x_title = getattr(fig.layout.xaxis.title, "text", None)
    y_title = getattr(fig.layout.yaxis.title, "text", None)
    if x_title:
        fig.update_xaxes(title_text=display_label(x_title))
    if y_title:
        fig.update_yaxes(title_text=display_label(y_title))
    return fig


@dataclass
class RunData:
    files: dict[str, Path | None]
    config: dict[str, Any]
    selected_event: dict[str, Any]
    summary: dict[str, Any]
    agents: pd.DataFrame
    states: pd.DataFrame
    reactions: pd.DataFrame
    interactions: pd.DataFrame
    graph: Any
    metrics: pd.DataFrame
    graph_source: str


def project_relative(path: Path | None) -> str:
    if path is None:
        return "未找到"
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def initial_output_dir() -> Path:
    if "output_base_dir" in st.session_state:
        return Path(st.session_state["output_base_dir"])
    return CLI_CURRENT_OUTPUT_DIR


def load_run(run_dir: Path | None) -> RunData:
    files = resolve_run_files(run_dir)
    agents = load_csv(files["agent_initial_states"])
    states = load_csv(files["agent_states_by_round"])
    interactions = load_csv(files["interactions"])
    graph = load_graph(files["network"])
    graph_source = "network.graphml"
    if graph is None:
        graph = build_graph_from_interactions(interactions, latest_states(states) if not states.empty else agents)
        graph_source = "interactions.csv 重建" if graph.number_of_edges() or graph.number_of_nodes() else "无网络数据"
    return RunData(
        files=files,
        config=load_json(files["config"]),
        selected_event=load_json(files["selected_event"]),
        summary=load_json(files["dynamics_summary"]),
        agents=agents,
        states=states,
        reactions=load_jsonl(files["active_reactions"]),
        interactions=interactions,
        graph=graph,
        metrics=load_csv(files["round_metrics"]),
        graph_source=graph_source,
    )


def metric_card(label: str, value: Any) -> None:
    if isinstance(value, float):
        value = safe_round(value, 3)
    st.metric(label, "暂无" if value is None or value == "" else value)


def apply_dashboard_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stTextInput"] input:disabled,
div[data-testid="stTextArea"] textarea:disabled {
    color: #111827;
    -webkit-text-fill-color: #111827;
    background-color: #ffffff;
    border-color: #d0d7de;
    opacity: 1;
    cursor: default;
}

div[data-testid="stTextInput"] input:disabled:focus,
div[data-testid="stTextArea"] textarea:disabled:focus {
    border-color: #9aa4b2;
    box-shadow: 0 0 0 1px #9aa4b2;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def show_file_status(data: RunData) -> None:
    status_rows = []
    for key in [
        "dynamics_summary",
        "agent_initial_states",
        "agent_states_by_round",
        "active_reactions",
        "interactions",
        "network",
        "round_metrics",
    ]:
        path = data.files.get(key)
        status_rows.append(
            {
                "文件": key,
                "文件内容": FILE_LABELS.get(key, key),
                "读取状态": "已读取" if path and path.exists() else "缺失",
                "实际文件名": path.name if path else "",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)


def build_run_command(params: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "--event-id",
        str(params["event_id"]),
        "--max-agents",
        str(params["max_agents"]),
        "--rounds",
        str(params["rounds"]),
        "--seed",
        str(params["seed"]),
        "--use-llm",
        "true" if params["use_llm"] else "false",
        "--output-dir",
        str(params["output_base_dir"]),
        "--interaction-mode",
        str(params["interaction_mode"]),
        "--kol-speaker-limit",
        str(params["kol_speaker_limit"]),
        "--top-k-context-comments",
        str(params["top_k_context_comments"]),
    ]
    if params["enable_interactions"]:
        cmd.append("--enable-interactions")
    if params["enable_emotion_dynamics"]:
        cmd.append("--enable-emotion-dynamics")
    return cmd


def run_simulation(params: dict[str, Any]) -> None:
    base_dir = Path(params["output_base_dir"])
    before = set(list_simulation_runs(base_dir))
    cmd = build_run_command(params)
    st.session_state["last_run_command"] = " ".join(cmd)
    try:
        with st.spinner("正在运行仿真，请稍候..."):
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=int(params["timeout"]),
                shell=False,
            )
    except subprocess.TimeoutExpired as exc:
        st.session_state["last_run_result"] = {
            "ok": False,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "message": f"运行超时，已超过 {params['timeout']} 秒。",
            "new_run_dir": "",
        }
        return
    except OSError as exc:
        st.session_state["last_run_result"] = {
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
            "message": "启动命令行仿真失败。",
            "new_run_dir": "",
        }
        return

    runs_after = list_simulation_runs(base_dir)
    after = set(runs_after)
    new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    selected = new_runs[0] if new_runs else (runs_after[0] if runs_after else None)
    if completed.returncode == 0 and selected is not None and params["auto_select_latest"]:
        st.session_state["selected_run_dir"] = str(selected)

    st.session_state["last_run_result"] = {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "message": "仿真运行成功。" if completed.returncode == 0 else f"仿真运行失败，returncode={completed.returncode}。",
        "new_run_dir": str(selected) if selected else "",
    }


def show_run_output() -> None:
    result = st.session_state.get("last_run_result")
    if not result:
        return
    if result["ok"]:
        st.success(result["message"])
    else:
        st.error(result["message"])
    if result.get("new_run_dir"):
        st.caption(f"识别到的运行结果目录：{result['new_run_dir']}")
    with st.expander("命令行标准输出", expanded=False):
        st.code(result.get("stdout", "") or "无标准输出")
    with st.expander("命令行错误输出", expanded=not result["ok"]):
        st.code(result.get("stderr", "") or "无错误输出")


def render_chart(fig, missing_message: str) -> None:
    if fig is None:
        st.info(missing_message)
    else:
        st.plotly_chart(localize_plotly_figure(fig), width="stretch")


def tab_run_selection(base_dir: Path, runs: list[Path], selected_run: Path | None, data: RunData) -> None:
    st.subheader("结果目录与已有运行结果")
    output_text = st.text_input("当前输出根目录", value=str(base_dir))
    new_base = Path(output_text)
    if new_base != base_dir:
        st.session_state["output_base_dir"] = str(new_base)
        st.session_state.pop("selected_run_dir", None)
        st.rerun()

    if runs:
        run_labels = [path.name for path in runs]
        selected_index = 0
        if selected_run is not None:
            for index, path in enumerate(runs):
                if path.resolve() == selected_run.resolve():
                    selected_index = index
                    break
        chosen = st.selectbox("已有运行结果选择", run_labels, index=selected_index)
        chosen_path = base_dir / chosen
        if str(chosen_path) != st.session_state.get("selected_run_dir"):
            st.session_state["selected_run_dir"] = str(chosen_path)
            st.rerun()
    else:
        st.warning("当前输出根目录下还没有可展示的运行结果。可以先运行新仿真，或切换到已有结果目录。")

    st.subheader("文件读取状态")
    show_file_status(data)

    st.subheader("运行新仿真")
    col1, col2, col3 = st.columns(3)
    with col1:
        event_id = st.text_input("事件编号", value="event_5194986460286423", key="run_event_id")
        max_agents = st.number_input("最大智能体数量", min_value=5, max_value=200, value=10, step=1)
    with col2:
        rounds = st.number_input("仿真轮数", min_value=1, max_value=20, value=5, step=1)
        seed = st.number_input("随机种子", value=42, step=1)
    with col3:
        use_llm = st.checkbox("启用大模型生成", value=True)
        enable_interactions = st.checkbox("启用智能体互动", value=True)

    with st.expander("高级运行参数", expanded=True):
        col4, col5, col6 = st.columns(3)
        with col4:
            interaction_mode = st.selectbox("互动模式", ["none", "kol_first"], index=1, format_func=display_label)
            kol_speaker_limit = st.number_input("关键意见领袖发言上限", min_value=1, max_value=20, value=5, step=1)
        with col5:
            top_k_context_comments = st.number_input("上下文评论数量上限", min_value=1, max_value=10, value=3, step=1)
            enable_emotion_dynamics = st.checkbox("启用情绪动态更新", value=True)
        with col6:
            timeout = st.number_input("超时时间（秒）", min_value=30, max_value=3600, value=300, step=30)
            auto_select_latest = st.checkbox("运行后自动切换到最新结果", value=True)

    params = {
        "event_id": event_id,
        "max_agents": int(max_agents),
        "rounds": int(rounds),
        "seed": int(seed),
        "use_llm": use_llm,
        "enable_interactions": enable_interactions,
        "interaction_mode": interaction_mode,
        "kol_speaker_limit": int(kol_speaker_limit),
        "top_k_context_comments": int(top_k_context_comments),
        "enable_emotion_dynamics": enable_emotion_dynamics,
        "timeout": int(timeout),
        "output_base_dir": base_dir,
        "auto_select_latest": auto_select_latest,
    }
    st.caption("将通过子进程调用 scope/run_multiround_simulation.py，不会在页面启动时自动运行。")
    if st.button("运行仿真", type="primary"):
        run_simulation(params)
        st.rerun()
    show_run_output()


def tab_overview(run_dir: Path | None, data: RunData) -> None:
    st.subheader("仿真运行概览")
    if run_dir is None:
        st.warning("尚未选择运行结果。")
        return
    summary = data.summary
    metrics = data.metrics
    event_id = summary.get("event_id") or data.selected_event.get("event_id") or data.config.get("event_id", "")
    topic = summary.get("topic") or data.selected_event.get("topic", "")
    event_context = data.selected_event.get("event_context", "")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("事件编号", value=str(event_id or "暂无"), disabled=True, key="overview_event_id")
    with col2:
        st.text_input("话题", value=str(topic or "暂无"), disabled=True, key="overview_topic")

    st.text_area("事件背景", value=str(event_context or "暂无"), height=120, disabled=True, key="overview_event_context")

    overview_items = [
        ("动态演化", summary.get("dynamics_enabled", metric_value(summary, metrics, "dynamics_enabled", ""))),
        ("互动启用", summary.get("interaction_enabled", data.config.get("enable_interactions", ""))),
        ("智能体数量", summary.get("total_agents", len(data.agents))),
        ("仿真轮数", summary.get("rounds", metric_value(summary, metrics, "round_id", ""))),
    ]
    rows = st.columns(4)
    for index, (label, value) in enumerate(overview_items):
        with rows[index]:
            metric_card(label, value)

    st.info("本次仿真以热点事件为输入，初始化一批微博用户智能体，在多轮评论区互动中模拟高影响力用户先发声、普通用户观察并响应，以及由互动边驱动的情绪传染与立场演化。")
    with st.expander("动态摘要原始数据"):
        st.json(summary or {})


def tab_agent_profile(data: RunData) -> None:
    st.subheader("智能体群体画像")
    agents = data.agents
    if agents.empty:
        st.warning("缺少智能体初始状态数据，无法展示智能体群体画像。")
        return
    st.caption("智能体由用户画像映射而来，不是随机节点。")
    level_counts = agents.get("memory_user_level", pd.Series(dtype=str)).fillna("未知").value_counts()
    metrics = [
        ("智能体总数", len(agents)),
        ("核心用户数量", int(level_counts.get("core", 0))),
        ("普通用户数量", int(level_counts.get("normal", 0))),
        ("背景用户数量", int(level_counts.get("background", 0))),
        ("平均影响力分数", safe_round(pd.to_numeric(agents.get("influence_score"), errors="coerce").mean(), 3)),
        ("平均易感性分数", safe_round(pd.to_numeric(agents.get("susceptibility_score"), errors="coerce").mean(), 3)),
        ("平均活跃度分数", safe_round(pd.to_numeric(agents.get("activity_score"), errors="coerce").mean(), 3)),
    ]
    cols = st.columns(4)
    for index, (label, value) in enumerate(metrics):
        with cols[index % 4]:
            metric_card(label, value)

    chart_cols = st.columns(2)
    categorical = [
        ("memory_user_level", "用户分层分布"),
        ("verified_type_name", "认证类型分布"),
    ]
    for index, (column, title) in enumerate(categorical):
        with chart_cols[index % 2]:
            if column in agents.columns:
                fig = plot_bar(value_count_frame(agents[column], column), column, "数量", title)
                render_chart(fig, f"缺少字段“{display_label(column)}”，跳过图表。")
            else:
                st.info(f"缺少字段“{display_label(column)}”，跳过图表。")
    if "propagation_role" in agents.columns:
        render_chart(plot_bar(explode_roles(agents["propagation_role"]), "propagation_role", "数量", "传播角色分布"), "缺少传播角色。")
    else:
        st.info("缺少字段“传播角色”，跳过角色分布。")

    numeric_columns = [
        ("influence_score", "影响力分数直方图"),
        ("susceptibility_score", "易感性分数直方图"),
        ("activity_score", "活跃度分数直方图"),
        ("emotion_score", "初始情绪分数分布"),
        ("stance_score", "初始立场分数分布"),
    ]
    for start in range(0, len(numeric_columns), 2):
        cols = st.columns(2)
        for offset, (column, title) in enumerate(numeric_columns[start : start + 2]):
            with cols[offset]:
                render_chart(plot_histogram(agents, column, title), f"缺少字段“{display_label(column)}”，跳过图表。")

    table_cols = [
        "agent_id",
        "user_id",
        "memory_user_level",
        "verified_type_name",
        "propagation_role",
        "influence_score",
        "susceptibility_score",
        "activity_score",
        "emotion_score",
        "stance_score",
        "emotion_label",
        "stance_label",
    ]
    existing = [column for column in table_cols if column in agents.columns]
    st.dataframe(localize_dataframe(round_numeric_columns(agents[existing].copy(), 3)), width="stretch", hide_index=True)


def tab_dynamics(data: RunData, show_round_zero: bool) -> None:
    st.subheader("情绪与立场演化")
    metrics = data.metrics.copy()
    if metrics.empty:
        st.warning("缺少轮次指标数据，无法展示演化趋势。")
        return
    if not show_round_zero and "round_id" in metrics.columns:
        metrics = coerce_numeric(metrics, ["round_id"])
        metrics = metrics[metrics["round_id"] != 0]
    st.caption("情绪分数范围为 [-1, 1]，越低表示越偏负向；立场分数范围为 [-1, 1]，越低表示越偏反对，越高表示越偏支持。")
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_line(metrics, "round_id", "avg_emotion_score", "群体平均情绪随轮次变化"), "缺少平均情绪分数或轮次。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "avg_stance_score", "群体平均立场随轮次变化"), "缺少平均立场分数或轮次。")
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_stacked_ratios(metrics, "round_id", ["positive_ratio", "neutral_ratio", "negative_ratio"], "情绪分布变化"), "缺少情绪比例字段。")
    with cols[1]:
        render_chart(plot_stacked_ratios(metrics, "round_id", ["support_ratio", "neutral_stance_ratio", "oppose_ratio"], "立场分布变化"), "缺少立场比例字段。")

    optional_groups = [
        (["emotion_volatility", "stance_volatility", "polarization_score"], "情绪波动与立场极化"),
        (["avg_abs_emotion_delta", "avg_abs_stance_delta", "max_abs_emotion_delta", "max_abs_stance_delta"], "状态变化强度"),
        (["avg_neighbor_count", "agents_affected_by_neighbors", "avg_neighbor_influence_weight"], "邻居影响指标"),
        (["interaction_count", "avg_interaction_weight", "high_influence_interaction_count"], "每轮互动规模"),
    ]
    for columns, title in optional_groups:
        fig = plot_multi_line(metrics, "round_id", columns, title)
        render_chart(fig, f"字段不足，跳过“{title}”。")
    st.info("极化程度表示立场分布离散程度；受邻居影响的智能体数量表示每轮受到互动边影响的智能体数。")


def tab_network(data: RunData, top_k: int) -> None:
    st.subheader("互动网络与关键节点")
    graph = data.graph
    overview = graph_overview(graph)
    if overview:
        cols = st.columns(4)
        for index, (label, value) in enumerate(overview.items()):
            with cols[index % 4]:
                metric_card(label, value)
        st.caption(f"网络数据来源：{graph_source_label(data.graph_source)}")
    else:
        st.warning("暂无网络数据。")

    centrality = compute_centrality_table(graph, data.agents, data.states)

    st.subheader("网络图展示控制")
    control_cols = st.columns(3)
    with control_cols[0]:
        view_mode = st.selectbox("网络视图模式", ["全局简化图", "关键节点子图", "单个智能体邻域网络"])
        edge_metric = st.selectbox("边筛选依据", ["weight_sum", "interaction_count", "weight"], format_func=display_label)
    with control_cols[1]:
        max_edges = st.number_input("最大显示边数", min_value=20, max_value=300, value=80, step=10)
        node_size_metric = st.selectbox(
            "节点大小依据",
            ["pagerank", "influence_score", "in_degree", "out_degree"],
            index=1,
            format_func=display_label,
        )
    with control_cols[2]:
        label_top_k = st.number_input("显示标签的关键节点数", min_value=0, max_value=30, value=10, step=1)
        center_agent_id = None
        if view_mode == "单个智能体邻域网络" and graph is not None and graph.number_of_nodes():
            agent_ids = sorted(str(node) for node in graph.nodes)
            center_agent_id = st.selectbox("邻域网络中心智能体编号", agent_ids)

    if not centrality.empty:
        display_cols = [
            "agent_id",
            "pagerank",
            "degree_centrality",
            "in_degree",
            "out_degree",
            "betweenness_centrality",
            "influence_score",
            "susceptibility_score",
            "propagation_role",
            "final_emotion_score",
            "final_stance_score",
        ]
        existing = [column for column in display_cols if column in centrality.columns]
        st.dataframe(localize_dataframe(centrality[existing].head(top_k)), width="stretch", hide_index=True)
    else:
        st.info("无法计算中心性指标，可能是网络为空。")

    if graph is not None and graph.number_of_nodes():
        view_graph, view_error = build_network_view_graph(
            graph,
            centrality,
            view_mode,
            int(top_k),
            center_agent_id=center_agent_id,
        )
        if view_error:
            st.warning(view_error)
        if view_graph is not None and view_graph.number_of_nodes():
            ok, error = render_pyvis_network(
                view_graph,
                centrality,
                max_edges=int(max_edges),
                edge_metric=edge_metric,
                node_size_metric=node_size_metric,
                label_top_k=int(label_top_k),
            )
            if not ok:
                st.warning(error)
            st.markdown(role_color_legend_html(), unsafe_allow_html=True)
            st.caption("节点大小表示所选中心性或影响力指标，节点颜色表示传播角色，边宽表示互动强度。为避免视觉拥挤，图中默认仅展示高权重互动边。")

    st.subheader("互动边统计")
    metrics = data.metrics
    interactions = data.interactions
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_line(metrics, "round_id", "interaction_count", "每轮互动次数"), "缺少互动次数。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "avg_interaction_weight", "每轮平均互动权重"), "缺少平均互动权重。")
    if not interactions.empty and "interaction_type" in interactions.columns:
        fig = plot_bar(value_count_frame(interactions["interaction_type"], "interaction_type"), "interaction_type", "数量", "互动类型分布")
        render_chart(fig, "缺少互动类型。")
    else:
        st.info("缺少互动明细或互动类型，跳过互动类型分布。")
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_histogram(interactions, "source_influence_score", "源智能体影响力分数分布"), "缺少源智能体影响力分数。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "high_influence_interaction_count", "高影响力互动次数随轮次变化"), "缺少高影响力互动次数。")


def tab_comments(data: RunData, comment_limit: int) -> None:
    st.subheader("评论流与状态明细")
    reactions = data.reactions
    states = data.states
    interactions = data.interactions

    if not reactions.empty and "round_id" in reactions.columns:
        reactions = coerce_numeric(reactions, ["round_id"])
        rounds = sorted(int(value) for value in reactions["round_id"].dropna().unique())
        selected_round = st.selectbox("评论流轮次", rounds, index=0)
        round_reactions = reactions[reactions["round_id"] == selected_round].copy()
        if "speaker_type" in round_reactions.columns:
            kol_reactions = round_reactions[round_reactions["speaker_type"] == "kol_speaker"]
            regular_reactions = round_reactions[round_reactions["speaker_type"] != "kol_speaker"]
        else:
            kol_reactions = pd.DataFrame()
            regular_reactions = round_reactions
        st.markdown("**关键意见领袖发言者**")
        st.dataframe(
            localize_dataframe(prepare_comment_table(kol_reactions, comment_limit)),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**普通智能体**")
        st.dataframe(
            localize_dataframe(prepare_comment_table(regular_reactions, comment_limit)),
            width="stretch",
            hide_index=True,
        )
        with st.expander("该轮完整评论表"):
            st.dataframe(localize_dataframe(round_numeric_columns(round_reactions, 3)), width="stretch", hide_index=True)
    else:
        st.info("缺少活跃反应数据或轮次，无法展示评论流。")
        selected_round = None

    st.subheader("智能体状态轨迹")
    if has_columns(states, ["agent_id", "round_id"]):
        state_data = coerce_numeric(states, ["round_id", "emotion_score", "stance_score"])
        agent_ids = sorted(state_data["agent_id"].dropna().astype(str).unique())
        selected_agent = st.selectbox("智能体编号", agent_ids)
        agent_rows = state_data[state_data["agent_id"].astype(str) == selected_agent].sort_values("round_id")
        cols = st.columns(2)
        with cols[0]:
            render_chart(plot_line(agent_rows, "round_id", "emotion_score", "情绪分数轨迹"), "缺少情绪分数。")
        with cols[1]:
            render_chart(plot_line(agent_rows, "round_id", "stance_score", "立场分数轨迹"), "缺少立场分数。")
        detail_cols = [
            "round_id",
            "emotion_score",
            "stance_score",
            "emotion_delta",
            "stance_delta",
            "neighbor_count",
            "state_update_reason",
        ]
        st.dataframe(localize_dataframe(round_numeric_columns(agent_rows[[c for c in detail_cols if c in agent_rows.columns]], 3)), width="stretch", hide_index=True)
    else:
        st.info("缺少智能体轮次状态数据或必要字段，无法展示智能体状态轨迹。")
        selected_agent = None

    st.subheader("邻居影响明细")
    if has_columns(interactions, ["target_agent_id", "round_id"]):
        interaction_data = coerce_numeric(interactions, ["round_id", "weight", "source_emotion_score", "source_stance_score", "source_influence_score"])
        target_ids = sorted(interaction_data["target_agent_id"].dropna().astype(str).unique())
        rounds = sorted(int(value) for value in interaction_data["round_id"].dropna().unique())
        col1, col2 = st.columns(2)
        with col1:
            target_id = st.selectbox("目标智能体编号", target_ids)
        with col2:
            neighbor_round = st.selectbox("邻居影响轮次", rounds, index=0 if selected_round is None or selected_round not in rounds else rounds.index(selected_round))
        rows = interaction_data[
            (interaction_data["target_agent_id"].astype(str) == target_id)
            & (interaction_data["round_id"] == neighbor_round)
        ].copy()
        columns = [
            "source_agent_id",
            "interaction_type",
            "weight",
            "source_reaction_text",
            "target_reaction_text",
            "source_emotion_score",
            "source_stance_score",
            "source_influence_score",
        ]
        if rows.empty:
            st.info("该轮次下没有找到指向该智能体的互动边。")
        else:
            if "source_reaction_text" in rows.columns:
                rows["source_reaction_text"] = rows["source_reaction_text"].map(lambda value: truncate_text(value, 120))
            if "target_reaction_text" in rows.columns:
                rows["target_reaction_text"] = rows["target_reaction_text"].map(lambda value: truncate_text(value, 120))
            st.dataframe(localize_dataframe(round_numeric_columns(rows[[c for c in columns if c in rows.columns]], 3)), width="stretch", hide_index=True)
    else:
        st.info("缺少互动明细或必要字段，无法展示邻居影响明细。")


def main() -> None:
    st.set_page_config(page_title="多智能体群体情绪演化仿真面板", layout="wide")
    apply_dashboard_styles()
    st.title("多智能体群体情绪演化仿真面板")

    with st.sidebar:
        st.header("展示控制")
        show_round_zero = st.checkbox("显示第 0 轮", value=True)
        top_k = st.number_input("关键节点数量", min_value=3, max_value=50, value=10, step=1)
        comment_limit = st.number_input("每轮默认评论数", min_value=5, max_value=200, value=30, step=5)

    base_dir = initial_output_dir()
    st.session_state.setdefault("output_base_dir", str(base_dir))
    runs = list_simulation_runs(base_dir)
    selected_run = Path(st.session_state["selected_run_dir"]) if st.session_state.get("selected_run_dir") else (runs[0] if runs else None)
    data = load_run(selected_run)

    tabs = st.tabs([
        "运行与结果选择",
        "仿真运行概览",
        "智能体群体画像",
        "情绪与立场演化",
        "互动网络与关键节点",
        "评论流与状态明细",
    ])
    with tabs[0]:
        tab_run_selection(base_dir, runs, selected_run, data)
    with tabs[1]:
        tab_overview(selected_run, data)
    with tabs[2]:
        tab_agent_profile(data)
    with tabs[3]:
        tab_dynamics(data, bool(show_round_zero))
    with tabs[4]:
        tab_network(data, int(top_k))
    with tabs[5]:
        tab_comments(data, int(comment_limit))

    st.divider()
    st.caption(
        "运行命令：使用可视化服务启动 scope/visualization/simulation_dashboard.py。"
        "依赖：页面框架、数据处理、图表绘制、网络分析和交互网络图组件。"
        "启动后默认读取最新已有仿真结果；可选择已有运行结果，也可配置参数并点击运行仿真。"
        "运行失败时在命令行标准输出和错误输出中查看错误；成功后会自动切换到识别到的新运行结果。"
    )


if __name__ == "__main__":
    main()
