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
    compute_centrality_table,
    graph_overview,
    prepare_comment_table,
    render_pyvis_network,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTED_OUTPUT_DIR = PROJECT_ROOT / "scope" / "outputs" / "simulation" / "multiround"
CLI_CURRENT_OUTPUT_DIR = PROJECT_ROOT / "scope" / "data" / "outputs" / "simulation" / "multiround"
CLI_SCRIPT = PROJECT_ROOT / "scope" / "run_multiround_simulation.py"


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
    documented_runs = list_simulation_runs(DOCUMENTED_OUTPUT_DIR)
    current_runs = list_simulation_runs(CLI_CURRENT_OUTPUT_DIR)
    return DOCUMENTED_OUTPUT_DIR if documented_runs or not current_runs else CLI_CURRENT_OUTPUT_DIR


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
            "message": "启动 CLI 失败。",
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
        st.caption(f"识别到的 run 目录：{result['new_run_dir']}")
    with st.expander("CLI stdout", expanded=False):
        st.code(result.get("stdout", "") or "无 stdout")
    with st.expander("CLI stderr", expanded=not result["ok"]):
        st.code(result.get("stderr", "") or "无 stderr")


def render_chart(fig, missing_message: str) -> None:
    if fig is None:
        st.info(missing_message)
    else:
        st.plotly_chart(fig, width="stretch")


def tab_run_selection(base_dir: Path, runs: list[Path], selected_run: Path | None, data: RunData) -> None:
    st.subheader("结果目录与已有 run")
    output_text = st.text_input("当前输出根目录", value=str(base_dir))
    new_base = Path(output_text)
    if new_base != base_dir:
        st.session_state["output_base_dir"] = str(new_base)
        st.session_state.pop("selected_run_dir", None)
        st.rerun()

    if base_dir == CLI_CURRENT_OUTPUT_DIR and base_dir != DOCUMENTED_OUTPUT_DIR:
        st.info(
            "已检测到当前项目已有结果位于 scope/data/outputs/simulation/multiround。"
            "如需完全按文档目录运行，可将上方路径改为 scope/outputs/simulation/multiround。"
        )

    if runs:
        run_labels = [path.name for path in runs]
        selected_index = 0
        if selected_run is not None:
            for index, path in enumerate(runs):
                if path.resolve() == selected_run.resolve():
                    selected_index = index
                    break
        chosen = st.selectbox("已有 run 选择", run_labels, index=selected_index)
        chosen_path = base_dir / chosen
        if str(chosen_path) != st.session_state.get("selected_run_dir"):
            st.session_state["selected_run_dir"] = str(chosen_path)
            st.rerun()
    else:
        st.warning("当前输出根目录下还没有可展示的 run。可以先运行新仿真，或切换到已有结果目录。")

    st.subheader("文件读取状态")
    show_file_status(data)

    st.subheader("运行新仿真")
    col1, col2, col3 = st.columns(3)
    with col1:
        event_id = st.text_input("event_id", value="event_5194986460286423")
        max_agents = st.number_input("max_agents", min_value=5, max_value=200, value=10, step=1)
    with col2:
        rounds = st.number_input("rounds", min_value=1, max_value=20, value=5, step=1)
        seed = st.number_input("seed", value=42, step=1)
    with col3:
        use_llm = st.checkbox("use_llm", value=True)
        enable_interactions = st.checkbox("enable_interactions", value=True)

    with st.expander("高级运行参数", expanded=True):
        col4, col5, col6 = st.columns(3)
        with col4:
            interaction_mode = st.selectbox("interaction_mode", ["none", "kol_first"], index=1)
            kol_speaker_limit = st.number_input("kol_speaker_limit", min_value=1, max_value=20, value=5, step=1)
        with col5:
            top_k_context_comments = st.number_input("top_k_context_comments", min_value=1, max_value=10, value=3, step=1)
            enable_emotion_dynamics = st.checkbox("enable_emotion_dynamics", value=True)
        with col6:
            timeout = st.number_input("timeout 秒数", min_value=30, max_value=3600, value=300, step=30)
            auto_select_latest = st.checkbox("运行后自动切换到最新 run", value=True)

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
    st.caption("将通过 subprocess.run 调用 scope/run_multiround_simulation.py，不会在页面启动时自动运行。")
    if st.button("运行仿真", type="primary"):
        run_simulation(params)
        st.rerun()
    show_run_output()


def tab_overview(run_dir: Path | None, data: RunData) -> None:
    st.subheader("仿真运行概览")
    if run_dir is None:
        st.warning("尚未选择 run。")
        return
    summary = data.summary
    metrics = data.metrics
    event_id = summary.get("event_id") or data.selected_event.get("event_id") or data.config.get("event_id", "")
    topic = summary.get("topic") or data.selected_event.get("topic", "")
    event_context = data.selected_event.get("event_context", "")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("事件 ID", value=str(event_id or "暂无"), disabled=True)
    with col2:
        st.text_input("话题", value=str(topic or "暂无"), disabled=True)

    st.text_area("事件背景", value=str(event_context or "暂无"), height=120, disabled=True)

    overview_items = [
        ("动态演化", summary.get("dynamics_enabled", metric_value(summary, metrics, "dynamics_enabled", ""))),
        ("互动启用", summary.get("interaction_enabled", data.config.get("enable_interactions", ""))),
        ("Agent 数量", summary.get("total_agents", len(data.agents))),
        ("仿真轮数", summary.get("rounds", metric_value(summary, metrics, "round_id", ""))),
    ]
    rows = st.columns(4)
    for index, (label, value) in enumerate(overview_items):
        with rows[index]:
            metric_card(label, value)

    st.info("本次仿真以热点事件为输入，初始化一批微博用户 Agent，在多轮评论区互动中模拟高影响力用户先发声、普通用户观察并响应，以及由互动边驱动的情绪传染与立场演化。")
    with st.expander("dynamics_summary 原始 JSON"):
        st.json(summary or {})


def tab_agent_profile(data: RunData) -> None:
    st.subheader("Agent 群体画像")
    agents = data.agents
    if agents.empty:
        st.warning("缺少 agent_initial_states 数据，无法展示 Agent 群体画像。")
        return
    st.caption("Agent 由用户画像映射而来，不是随机节点。")
    level_counts = agents.get("memory_user_level", pd.Series(dtype=str)).fillna("未知").value_counts()
    metrics = [
        ("总 Agent 数", len(agents)),
        ("core 数量", int(level_counts.get("core", 0))),
        ("normal 数量", int(level_counts.get("normal", 0))),
        ("background 数量", int(level_counts.get("background", 0))),
        ("平均 influence_score", safe_round(pd.to_numeric(agents.get("influence_score"), errors="coerce").mean(), 3)),
        ("平均 susceptibility_score", safe_round(pd.to_numeric(agents.get("susceptibility_score"), errors="coerce").mean(), 3)),
        ("平均 activity_score", safe_round(pd.to_numeric(agents.get("activity_score"), errors="coerce").mean(), 3)),
    ]
    cols = st.columns(4)
    for index, (label, value) in enumerate(metrics):
        with cols[index % 4]:
            metric_card(label, value)

    chart_cols = st.columns(2)
    categorical = [
        ("memory_user_level", "memory_user_level 分布"),
        ("verified_type_name", "verified_type_name 分布"),
    ]
    for index, (column, title) in enumerate(categorical):
        with chart_cols[index % 2]:
            if column in agents.columns:
                fig = plot_bar(value_count_frame(agents[column], column), column, "数量", title)
                render_chart(fig, f"缺少字段 {column}，跳过图表。")
            else:
                st.info(f"缺少字段 {column}，跳过图表。")
    if "propagation_role" in agents.columns:
        render_chart(plot_bar(explode_roles(agents["propagation_role"]), "propagation_role", "数量", "propagation_role 分布"), "缺少 propagation_role。")
    else:
        st.info("缺少字段 propagation_role，跳过角色分布。")

    numeric_columns = [
        ("influence_score", "influence_score 直方图"),
        ("susceptibility_score", "susceptibility_score 直方图"),
        ("activity_score", "activity_score 直方图"),
        ("emotion_score", "初始 emotion_score 分布"),
        ("stance_score", "初始 stance_score 分布"),
    ]
    for start in range(0, len(numeric_columns), 2):
        cols = st.columns(2)
        for offset, (column, title) in enumerate(numeric_columns[start : start + 2]):
            with cols[offset]:
                render_chart(plot_histogram(agents, column, title), f"缺少字段 {column}，跳过图表。")

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
    st.dataframe(round_numeric_columns(agents[existing].copy(), 3), width="stretch", hide_index=True)


def tab_dynamics(data: RunData, show_round_zero: bool) -> None:
    st.subheader("情绪与立场演化")
    metrics = data.metrics.copy()
    if metrics.empty:
        st.warning("缺少 round_metrics 数据，无法展示演化趋势。")
        return
    if not show_round_zero and "round_id" in metrics.columns:
        metrics = coerce_numeric(metrics, ["round_id"])
        metrics = metrics[metrics["round_id"] != 0]
    st.caption("情绪分数范围为 [-1, 1]，越低表示越偏负向；立场分数范围为 [-1, 1]，越低表示越偏反对，越高表示越偏支持。")
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_line(metrics, "round_id", "avg_emotion_score", "群体平均情绪随轮次变化"), "缺少 avg_emotion_score 或 round_id。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "avg_stance_score", "群体平均立场随轮次变化"), "缺少 avg_stance_score 或 round_id。")
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
    st.info("polarization_score 表示立场分布离散程度；agents_affected_by_neighbors 表示每轮受到互动边影响的 Agent 数。")


def tab_network(data: RunData, top_k: int, max_edges: int) -> None:
    st.subheader("互动网络与关键节点")
    graph = data.graph
    overview = graph_overview(graph)
    if overview:
        cols = st.columns(4)
        for index, (label, value) in enumerate(overview.items()):
            with cols[index % 4]:
                metric_card(label, value)
        st.caption(f"网络数据来源：{data.graph_source}")
    else:
        st.warning("暂无网络数据。")

    centrality = compute_centrality_table(graph, data.agents, data.states)
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
        st.dataframe(centrality[existing].head(top_k), width="stretch", hide_index=True)
    else:
        st.info("无法计算中心性指标，可能是网络为空。")

    if graph is not None and graph.number_of_nodes():
        ok, error = render_pyvis_network(graph, centrality, max_edges=max_edges)
        if not ok:
            st.warning(error)

    st.subheader("互动边统计")
    metrics = data.metrics
    interactions = data.interactions
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_line(metrics, "round_id", "interaction_count", "每轮 interaction_count"), "缺少 interaction_count。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "avg_interaction_weight", "每轮 avg_interaction_weight"), "缺少 avg_interaction_weight。")
    if not interactions.empty and "interaction_type" in interactions.columns:
        fig = plot_bar(value_count_frame(interactions["interaction_type"], "interaction_type"), "interaction_type", "数量", "interaction_type 分布")
        render_chart(fig, "缺少 interaction_type。")
    else:
        st.info("缺少 interactions.csv 或 interaction_type，跳过互动类型分布。")
    cols = st.columns(2)
    with cols[0]:
        render_chart(plot_histogram(interactions, "source_influence_score", "source_influence_score 分布"), "缺少 source_influence_score。")
    with cols[1]:
        render_chart(plot_line(metrics, "round_id", "high_influence_interaction_count", "high_influence_interaction_count 随轮次变化"), "缺少 high_influence_interaction_count。")


def tab_comments(data: RunData, comment_limit: int) -> None:
    st.subheader("评论流与状态明细")
    reactions = data.reactions
    states = data.states
    interactions = data.interactions

    if not reactions.empty and "round_id" in reactions.columns:
        reactions = coerce_numeric(reactions, ["round_id"])
        rounds = sorted(int(value) for value in reactions["round_id"].dropna().unique())
        selected_round = st.selectbox("评论流 round_id", rounds, index=0)
        round_reactions = reactions[reactions["round_id"] == selected_round].copy()
        if "speaker_type" in round_reactions.columns:
            kol_reactions = round_reactions[round_reactions["speaker_type"] == "kol_speaker"]
            regular_reactions = round_reactions[round_reactions["speaker_type"] != "kol_speaker"]
        else:
            kol_reactions = pd.DataFrame()
            regular_reactions = round_reactions
        st.markdown("**KOL speaker**")
        st.dataframe(
            prepare_comment_table(kol_reactions, comment_limit),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Regular agent**")
        st.dataframe(
            prepare_comment_table(regular_reactions, comment_limit),
            width="stretch",
            hide_index=True,
        )
        with st.expander("该轮完整评论表"):
            st.dataframe(round_numeric_columns(round_reactions, 3), width="stretch", hide_index=True)
    else:
        st.info("缺少 active_reactions.jsonl 或 round_id，无法展示评论流。")
        selected_round = None

    st.subheader("Agent 状态轨迹")
    if has_columns(states, ["agent_id", "round_id"]):
        state_data = coerce_numeric(states, ["round_id", "emotion_score", "stance_score"])
        agent_ids = sorted(state_data["agent_id"].dropna().astype(str).unique())
        selected_agent = st.selectbox("agent_id", agent_ids)
        agent_rows = state_data[state_data["agent_id"].astype(str) == selected_agent].sort_values("round_id")
        cols = st.columns(2)
        with cols[0]:
            render_chart(plot_line(agent_rows, "round_id", "emotion_score", "emotion_score 轨迹"), "缺少 emotion_score。")
        with cols[1]:
            render_chart(plot_line(agent_rows, "round_id", "stance_score", "stance_score 轨迹"), "缺少 stance_score。")
        detail_cols = [
            "round_id",
            "emotion_score",
            "stance_score",
            "emotion_delta",
            "stance_delta",
            "neighbor_count",
            "state_update_reason",
        ]
        st.dataframe(round_numeric_columns(agent_rows[[c for c in detail_cols if c in agent_rows.columns]], 3), width="stretch", hide_index=True)
    else:
        st.info("缺少 agent_states_by_round.csv 或必要字段，无法展示 Agent 状态轨迹。")
        selected_agent = None

    st.subheader("邻居影响明细")
    if has_columns(interactions, ["target_agent_id", "round_id"]):
        interaction_data = coerce_numeric(interactions, ["round_id", "weight", "source_emotion_score", "source_stance_score", "source_influence_score"])
        target_ids = sorted(interaction_data["target_agent_id"].dropna().astype(str).unique())
        rounds = sorted(int(value) for value in interaction_data["round_id"].dropna().unique())
        col1, col2 = st.columns(2)
        with col1:
            target_id = st.selectbox("target_agent_id", target_ids)
        with col2:
            neighbor_round = st.selectbox("邻居影响 round_id", rounds, index=0 if selected_round is None or selected_round not in rounds else rounds.index(selected_round))
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
            st.info("该 round 下没有找到指向该 Agent 的互动边。")
        else:
            if "source_reaction_text" in rows.columns:
                rows["source_reaction_text"] = rows["source_reaction_text"].map(lambda value: truncate_text(value, 120))
            if "target_reaction_text" in rows.columns:
                rows["target_reaction_text"] = rows["target_reaction_text"].map(lambda value: truncate_text(value, 120))
            st.dataframe(round_numeric_columns(rows[[c for c in columns if c in rows.columns]], 3), width="stretch", hide_index=True)
    else:
        st.info("缺少 interactions.csv 或必要字段，无法展示邻居影响明细。")


def main() -> None:
    st.set_page_config(page_title="多智能体群体情绪演化仿真面板", layout="wide")
    st.title("多智能体群体情绪演化仿真面板")

    with st.sidebar:
        st.header("展示控制")
        show_round_zero = st.checkbox("显示第 0 轮", value=True)
        top_k = st.number_input("关键节点 Top K", min_value=3, max_value=50, value=10, step=1)
        max_edges = st.number_input("PyVis 最大边数", min_value=10, max_value=1000, value=100, step=10)
        comment_limit = st.number_input("每轮默认评论数", min_value=5, max_value=200, value=30, step=5)

    base_dir = initial_output_dir()
    st.session_state.setdefault("output_base_dir", str(base_dir))
    runs = list_simulation_runs(base_dir)
    selected_run = Path(st.session_state["selected_run_dir"]) if st.session_state.get("selected_run_dir") else (runs[0] if runs else None)
    data = load_run(selected_run)

    tabs = st.tabs([
        "运行与结果选择",
        "仿真运行概览",
        "Agent 群体画像",
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
        tab_network(data, int(top_k), int(max_edges))
    with tabs[5]:
        tab_comments(data, int(comment_limit))

    st.divider()
    st.caption(
        "运行命令：streamlit run scope/visualization/simulation_dashboard.py。"
        "依赖：streamlit、pandas、plotly、networkx、pyvis。"
        "启动后默认读取最新已有仿真结果；可选择已有 run，也可配置参数并点击运行仿真。"
        "运行失败时在 CLI stdout / stderr 中查看错误；成功后会自动切换到识别到的新 run。"
    )


if __name__ == "__main__":
    main()
