from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import streamlit.components.v1 as components

from simulation_dashboard_utils import latest_states, round_numeric_columns, safe_round, truncate_text


def graph_overview(graph: nx.DiGraph | None) -> dict[str, Any]:
    if graph is None:
        return {}
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    return {
        "节点数": node_count,
        "边数": edge_count,
        "网络密度": safe_round(nx.density(graph), 4) if node_count > 1 else 0.0,
        "平均入度": safe_round(sum(in_degrees.values()) / node_count, 3) if node_count else 0.0,
        "平均出度": safe_round(sum(out_degrees.values()) / node_count, 3) if node_count else 0.0,
        "最大入度": max(in_degrees.values(), default=0),
        "最大出度": max(out_degrees.values(), default=0),
    }


def compute_centrality_table(
    graph: nx.DiGraph | None,
    agents: pd.DataFrame,
    agent_states: pd.DataFrame,
) -> pd.DataFrame:
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame()

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    degree_centrality = nx.degree_centrality(graph)
    in_degree_centrality = nx.in_degree_centrality(graph)
    out_degree_centrality = nx.out_degree_centrality(graph)
    try:
        betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    except Exception:
        betweenness = {node: 0.0 for node in graph.nodes}
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except Exception:
        pagerank = {node: 0.0 for node in graph.nodes}

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "agent_id": node,
                "pagerank": pagerank.get(node, 0.0),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "in_degree_centrality": in_degree_centrality.get(node, 0.0),
                "out_degree_centrality": out_degree_centrality.get(node, 0.0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "betweenness_centrality": betweenness.get(node, 0.0),
            }
        )
    table = pd.DataFrame(rows)

    if not agents.empty and "agent_id" in agents.columns:
        agent_cols = [
            column
            for column in [
                "agent_id",
                "influence_score",
                "susceptibility_score",
                "activity_score",
                "memory_user_level",
                "verified_type_name",
                "propagation_role",
            ]
            if column in agents.columns
        ]
        table = table.merge(agents[agent_cols].drop_duplicates("agent_id"), on="agent_id", how="left")

    final_states = latest_states(agent_states)
    if not final_states.empty:
        rename = {
            "emotion_score": "final_emotion_score",
            "stance_score": "final_stance_score",
            "emotion_label": "final_emotion_label",
            "stance_label": "final_stance_label",
        }
        final_cols = ["agent_id", *[column for column in rename if column in final_states.columns]]
        final_states = final_states[final_cols].rename(columns=rename)
        table = table.merge(final_states, on="agent_id", how="left")

    return round_numeric_columns(table.sort_values("pagerank", ascending=False).reset_index(drop=True), 4)


def _node_attr(graph: nx.DiGraph, node: str, key: str, default: Any = "") -> Any:
    return graph.nodes[node].get(key, default)


def _edge_attr(attrs: dict[str, Any], key: str, default: Any = "") -> Any:
    return attrs.get(key, default)


def render_pyvis_network(
    graph: nx.DiGraph,
    centrality: pd.DataFrame,
    max_edges: int = 100,
    height: int = 700,
) -> tuple[bool, str | None]:
    try:
        from pyvis.network import Network
    except Exception as exc:
        return False, f"PyVis 不可用：{exc}"

    if graph.number_of_nodes() == 0:
        return False, "网络为空，暂无可渲染节点。"

    pagerank = {}
    if not centrality.empty and {"agent_id", "pagerank"}.issubset(centrality.columns):
        pagerank = dict(zip(centrality["agent_id"].astype(str), centrality["pagerank"]))

    ranked_edges = sorted(
        graph.edges(data=True),
        key=lambda item: (
            float(item[2].get("weight_sum", item[2].get("weight", 0.0)) or 0.0),
            int(item[2].get("interaction_count", 1) or 1),
        ),
        reverse=True,
    )[:max_edges]
    nodes_to_render = set()
    for source, target, _attrs in ranked_edges:
        nodes_to_render.add(str(source))
        nodes_to_render.add(str(target))
    if not nodes_to_render:
        nodes_to_render = set(str(node) for node in graph.nodes)

    try:
        net = Network(height="650px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
        net.barnes_hut(gravity=-2800, central_gravity=0.25, spring_length=120, spring_strength=0.02)
        for node in nodes_to_render:
            node_id = str(node)
            influence = float(_node_attr(graph, node, "influence_score", 0.0) or 0.0)
            pr = float(pagerank.get(node_id, 0.0) or 0.0)
            size = 12 + min(34, influence * 24 + pr * 450)
            emotion = _node_attr(graph, node, "final_emotion_score", _node_attr(graph, node, "emotion_score", ""))
            stance = _node_attr(graph, node, "final_stance_score", _node_attr(graph, node, "stance_score", ""))
            title = (
                f"agent_id: {node_id}<br>"
                f"memory_user_level: {_node_attr(graph, node, 'memory_user_level', '')}<br>"
                f"propagation_role: {_node_attr(graph, node, 'propagation_role', '')}<br>"
                f"influence_score: {safe_round(influence, 3)}<br>"
                f"susceptibility_score: {_node_attr(graph, node, 'susceptibility_score', '')}<br>"
                f"final_emotion_score: {safe_round(emotion, 3)}<br>"
                f"final_stance_score: {safe_round(stance, 3)}"
            )
            label = node_id[-6:] if len(node_id) > 6 else node_id
            net.add_node(node_id, label=label, title=title, size=size)

        for source, target, attrs in ranked_edges:
            source_id = str(source)
            target_id = str(target)
            if source_id not in nodes_to_render or target_id not in nodes_to_render:
                continue
            weight_sum = float(_edge_attr(attrs, "weight_sum", _edge_attr(attrs, "weight", 1.0)) or 1.0)
            count = int(float(_edge_attr(attrs, "interaction_count", 1) or 1))
            title = (
                f"interaction_count: {count}<br>"
                f"weight_sum: {safe_round(weight_sum, 3)}<br>"
                f"first_round: {_edge_attr(attrs, 'first_round', '')}<br>"
                f"last_round: {_edge_attr(attrs, 'last_round', '')}<br>"
                f"interaction_types: {_edge_attr(attrs, 'interaction_types', _edge_attr(attrs, 'interaction_type', ''))}"
            )
            net.add_edge(source_id, target_id, value=max(weight_sum, 0.1), width=1 + min(7, weight_sum), title=title)

        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "network.html"
            net.write_html(str(html_path), notebook=False, open_browser=False)
            html = html_path.read_text(encoding="utf-8")
        components.html(html, height=height, scrolling=True)
        return True, None
    except Exception as exc:
        return False, f"PyVis 渲染失败：{exc}"


def prepare_comment_table(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "speaker_type",
        "agent_id",
        "memory_user_level",
        "propagation_role",
        "action_type",
        "emotion_label",
        "emotion_intensity",
        "stance_label",
        "stance_intensity",
        "reaction_text",
        "context_comment_count",
        "influenced_by_high_influence",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    existing = [column for column in columns if column in df.columns]
    result = df[existing].copy().head(limit)
    if "reaction_text" in result.columns:
        result["reaction_text"] = result["reaction_text"].map(lambda value: truncate_text(value, 120))
    return round_numeric_columns(result, 3)
