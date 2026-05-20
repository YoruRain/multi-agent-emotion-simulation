from __future__ import annotations
import html
import json
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st

from simulation_dashboard_utils import latest_states, round_numeric_columns, safe_round, truncate_text


ROLE_PRIORITY = [
    "潜在影响者",
    "KOL 敏感型用户",
    "媒体信息跟随者",
    "转发评论者",
    "转发扩散者",
    "原创表达者",
    "普通参与者",
    "低活跃观察者",
]

ROLE_COLORS = {
    "潜在影响者": "#e76f51",
    "KOL 敏感型用户": "#f4a261",
    "媒体信息跟随者": "#8e7cc3",
    "转发评论者": "#4f81bd",
    "转发扩散者": "#6fa8dc",
    "原创表达者": "#2a9d8f",
    "普通参与者": "#7aa6dc",
    "低活跃观察者": "#b7b7b7",
    "未知角色": "#cccccc",
}

ROLE_DISPLAY_LABELS = {
    "KOL 敏感型用户": "关键意见领袖敏感型用户",
}

EDGE_COLORS = {
    "same_round_context": "#9dc3e6",
    "reply": "#4472c4",
    "repost": "#8064a2",
    "influence_candidate": "#b7b7b7",
    "其他": "#cccccc",
}

EDGE_TYPE_PRIORITY = ["reply", "repost", "same_round_context", "influence_candidate"]

EDGE_TYPE_LABELS = {
    "same_round_context": "同轮上下文",
    "reply": "回复",
    "repost": "转发",
    "influence_candidate": "候选影响边",
}


def display_role(role: Any) -> str:
    text = str(role)
    if "," in text or "，" in text:
        parts = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
        return "，".join(ROLE_DISPLAY_LABELS.get(part, part) for part in parts)
    return ROLE_DISPLAY_LABELS.get(text, text)


def display_edge_types(edge_types: Any) -> str:
    tokens = [token.strip() for token in str(edge_types).replace("，", ",").split(",") if token.strip()]
    labels = [EDGE_TYPE_LABELS.get(token, token) for token in tokens]
    return "，".join(labels) if labels else "未知"


def get_primary_role(role_text: Any) -> str:
    if role_text is None:
        return "未知角色"
    normalized = str(role_text).replace("，", ",").replace("；", ",").replace(";", ",")
    roles = [role.strip() for role in normalized.split(",") if role.strip() and role.strip().lower() != "nan"]
    if not roles:
        return "未知角色"
    role_set = set(roles)
    for role in ROLE_PRIORITY:
        if role in role_set:
            return role
    return roles[0] if roles[0] in ROLE_COLORS else "未知角色"


def scale_node_size(value: Any, min_value: float, max_value: float, min_size: int = 16, max_size: int = 48) -> float:
    numeric = _as_float(value)
    if numeric is None:
        return 24.0
    if max_value <= min_value:
        return float((min_size + max_size) / 2)
    ratio = (numeric - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    return round(min_size + ratio * (max_size - min_size), 2)


def short_agent_label(agent_id: Any) -> str:
    node_id = str(agent_id)
    return node_id[-6:] if len(node_id) > 6 else node_id


def role_color_legend_html() -> str:
    items = []
    for role, color in ROLE_COLORS.items():
        items.append(
            "<span style='display:inline-flex;align-items:center;gap:6px;margin:0 14px 8px 0;'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:{color};display:inline-block;'></span>"
            f"<span>{html.escape(display_role(role))}</span>"
            "</span>"
        )
    return (
        "<div style='line-height:1.8;margin-top:8px;'>"
        "<strong>角色颜色图例：</strong><br>"
        + "".join(items)
        + "</div>"
    )


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return default
    return int(numeric)


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
        pagerank = degree_centrality.copy()

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


def _centrality_records(centrality: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if centrality.empty or "agent_id" not in centrality.columns:
        return {}
    records = centrality.copy()
    records["agent_id"] = records["agent_id"].astype(str)
    return {row["agent_id"]: row for row in records.to_dict("records")}


def _node_value(graph: nx.DiGraph, centrality_rows: dict[str, dict[str, Any]], node: Any, key: str, default: Any = "") -> Any:
    node_id = str(node)
    row = centrality_rows.get(node_id, {})
    value = row.get(key, None)
    if value is not None and str(value).lower() != "nan":
        return value
    return _node_attr(graph, node, key, default)


def _metric_value(
    graph: nx.DiGraph,
    centrality_rows: dict[str, dict[str, Any]],
    node: Any,
    metric: str,
) -> float | None:
    value = _as_float(_node_value(graph, centrality_rows, node, metric, None))
    if value is not None:
        return value
    value = _as_float(_node_value(graph, centrality_rows, node, "influence_score", None))
    if value is not None:
        return value
    return None


def _edge_sort_value(attrs: dict[str, Any], edge_metric: str) -> float:
    if edge_metric == "weight_sum":
        return _as_float(attrs.get("weight_sum"), _as_float(attrs.get("weight"), 0.0)) or 0.0
    if edge_metric == "interaction_count":
        return _as_float(attrs.get("interaction_count"), 1.0) or 0.0
    return _as_float(attrs.get("weight"), _as_float(attrs.get("weight_sum"), 0.0)) or 0.0


def _top_edges(graph: nx.DiGraph, max_edges: int, edge_metric: str) -> list[tuple[Any, Any, dict[str, Any]]]:
    ranked_edges = sorted(
        graph.edges(data=True),
        key=lambda item: (
            _edge_sort_value(item[2], edge_metric),
            _as_float(item[2].get("weight_sum"), 0.0) or 0.0,
            _as_float(item[2].get("interaction_count"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    return ranked_edges[:max_edges]


def _ranked_nodes(centrality: pd.DataFrame, nodes: set[str], metric: str, top_k: int) -> set[str]:
    if top_k <= 0 or not nodes:
        return set()
    metric_column = metric if metric in centrality.columns else "pagerank"
    if centrality.empty or "agent_id" not in centrality.columns or metric_column not in centrality.columns:
        return set(list(nodes)[:top_k])
    ranked = centrality.copy()
    ranked["agent_id"] = ranked["agent_id"].astype(str)
    ranked = ranked[ranked["agent_id"].isin(nodes)]
    ranked[metric_column] = pd.to_numeric(ranked[metric_column], errors="coerce").fillna(0.0)
    return set(ranked.sort_values(metric_column, ascending=False)["agent_id"].head(top_k))


def build_network_view_graph(
    graph: nx.DiGraph,
    centrality: pd.DataFrame,
    view_mode: str,
    key_node_count: int,
    center_agent_id: str | None = None,
) -> tuple[nx.DiGraph | None, str | None]:
    if graph.number_of_nodes() == 0:
        return None, "网络为空，暂无可渲染节点。"

    if view_mode == "全局简化图":
        return graph.copy(), None

    if view_mode == "关键节点子图":
        all_nodes = {str(node) for node in graph.nodes}
        key_nodes = _ranked_nodes(centrality, all_nodes, "pagerank", max(1, key_node_count))
        nodes_to_keep: set[Any] = set()
        for node_id in key_nodes:
            if node_id not in graph:
                continue
            nodes_to_keep.add(node_id)
            nodes_to_keep.update(graph.predecessors(node_id))
            nodes_to_keep.update(graph.successors(node_id))
        if not nodes_to_keep:
            return None, "无法识别关键节点，暂无可渲染子图。"
        return graph.subgraph(nodes_to_keep).copy(), None

    if view_mode == "单个智能体邻域网络":
        if not center_agent_id:
            return None, "请选择用于邻域网络的中心智能体编号。"
        if center_agent_id not in graph:
            return None, f"智能体 {center_agent_id} 不在当前网络中。"
        nodes_to_keep = {center_agent_id, *graph.predecessors(center_agent_id), *graph.successors(center_agent_id)}
        return graph.subgraph(nodes_to_keep).copy(), None

    return graph.copy(), None


def _edge_primary_type(attrs: dict[str, Any]) -> str:
    raw_types = attrs.get("interaction_types", attrs.get("interaction_type", ""))
    tokens = [token.strip() for token in str(raw_types).replace("，", ",").split(",") if token.strip()]
    token_set = set(tokens)
    for edge_type in EDGE_TYPE_PRIORITY:
        if edge_type in token_set:
            return edge_type
    return tokens[0] if tokens and tokens[0] in EDGE_COLORS else "其他"


def _edge_width(attrs: dict[str, Any], min_weight: float, max_weight: float) -> float:
    count = _as_int(attrs.get("interaction_count"))
    if count is not None and count > 0:
        return float(1 + min(6, count))
    weight_sum = _as_float(attrs.get("weight_sum"), _as_float(attrs.get("weight"), 1.0)) or 1.0
    if max_weight <= min_weight:
        return 3.0
    ratio = (weight_sum - min_weight) / (max_weight - min_weight)
    return round(1 + max(0.0, min(1.0, ratio)) * 5, 2)


def render_pyvis_network(
    graph: nx.DiGraph,
    centrality: pd.DataFrame,
    max_edges: int = 80,
    edge_metric: str = "weight_sum",
    node_size_metric: str = "pagerank",
    label_top_k: int = 10,
    height: int = 700,
) -> tuple[bool, str | None]:
    try:
        from pyvis.network import Network
    except Exception as exc:
        return False, f"网络图渲染组件不可用：{exc}"

    if graph.number_of_nodes() == 0:
        return False, "网络为空，暂无可渲染节点。"

    centrality_rows = _centrality_records(centrality)
    ranked_edges = _top_edges(graph, max_edges, edge_metric)
    nodes_to_render = {str(node) for node in graph.nodes}
    metric_values = [
        value
        for node in graph.nodes
        if (value := _metric_value(graph, centrality_rows, node, node_size_metric)) is not None
    ]
    min_metric = min(metric_values) if metric_values else 0.0
    max_metric = max(metric_values) if metric_values else 0.0
    label_metric = node_size_metric if not centrality.empty and node_size_metric in centrality.columns else "pagerank"
    labeled_nodes = _ranked_nodes(centrality, nodes_to_render, label_metric, label_top_k)
    edge_weights = [
        _as_float(attrs.get("weight_sum"), _as_float(attrs.get("weight"), 1.0)) or 1.0
        for _source, _target, attrs in ranked_edges
    ]
    min_edge_weight = min(edge_weights) if edge_weights else 0.0
    max_edge_weight = max(edge_weights) if edge_weights else 0.0

    try:
        net = Network(height="650px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.15,
            spring_length=180,
            spring_strength=0.03,
            damping=0.75,
            overlap=0.8,
        )
        net.set_options(
            json.dumps(
                {
                    "interaction": {
                        "hover": True,
                        "navigationButtons": True,
                        "keyboard": True,
                    },
                    "physics": {
                        "enabled": True,
                        "barnesHut": {
                            "gravitationalConstant": -5000,
                            "centralGravity": 0.15,
                            "springLength": 180,
                            "springConstant": 0.03,
                            "damping": 0.75,
                            "avoidOverlap": 0.8,
                        },
                    },
                    "edges": {
                        "smooth": {"enabled": True, "type": "dynamic"},
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                    },
                }
            )
        )
        for node in graph.nodes:
            node_id = str(node)
            influence = _as_float(_node_value(graph, centrality_rows, node, "influence_score", None))
            size_value = _metric_value(graph, centrality_rows, node, node_size_metric)
            size = scale_node_size(size_value, min_metric, max_metric)
            propagation_role = _node_value(graph, centrality_rows, node, "propagation_role", "未知角色")
            primary_role = get_primary_role(propagation_role)
            color = ROLE_COLORS.get(primary_role, ROLE_COLORS["未知角色"])
            emotion = _node_value(graph, centrality_rows, node, "final_emotion_score", _node_attr(graph, node, "emotion_score", ""))
            stance = _node_value(graph, centrality_rows, node, "final_stance_score", _node_attr(graph, node, "stance_score", ""))
            title = (
                f"智能体编号: {node_id}<br>"
                f"主要传播角色: {html.escape(display_role(primary_role))}<br>"
                f"传播角色: {html.escape(display_role(propagation_role or '未知角色'))}<br>"
                f"用户分层: {_node_value(graph, centrality_rows, node, 'memory_user_level', '')}<br>"
                f"影响力分数: {safe_round(influence, 3) if influence is not None else ''}<br>"
                f"易感性分数: {_node_value(graph, centrality_rows, node, 'susceptibility_score', '')}<br>"
                f"页面排名中心性: {_node_value(graph, centrality_rows, node, 'pagerank', '')}<br>"
                f"入度: {_node_value(graph, centrality_rows, node, 'in_degree', '')}<br>"
                f"出度: {_node_value(graph, centrality_rows, node, 'out_degree', '')}<br>"
                f"最终情绪分数: {safe_round(emotion, 3)}<br>"
                f"最终立场分数: {safe_round(stance, 3)}"
            )
            label = short_agent_label(node_id) if node_id in labeled_nodes else " "
            font = {"size": 14} if node_id in labeled_nodes else {"size": 0}
            net.add_node(node_id, label=label, title=title, size=size, color=color, font=font)

        for source, target, attrs in ranked_edges:
            source_id = str(source)
            target_id = str(target)
            if source_id not in nodes_to_render or target_id not in nodes_to_render:
                continue
            weight_sum = _as_float(_edge_attr(attrs, "weight_sum", _edge_attr(attrs, "weight", 1.0)), 1.0) or 1.0
            count = _as_int(_edge_attr(attrs, "interaction_count", 1), 1) or 1
            edge_types = _edge_attr(attrs, "interaction_types", _edge_attr(attrs, "interaction_type", ""))
            title = (
                f"源智能体: {source_id}<br>"
                f"目标智能体: {target_id}<br>"
                f"互动次数: {count}<br>"
                f"累计权重: {safe_round(weight_sum, 3)}<br>"
                f"首次轮次: {_edge_attr(attrs, 'first_round', '')}<br>"
                f"末次轮次: {_edge_attr(attrs, 'last_round', '')}<br>"
                f"互动类型: {display_edge_types(edge_types)}"
            )
            edge_type = _edge_primary_type(attrs)
            net.add_edge(
                source_id,
                target_id,
                value=max(weight_sum, 0.1),
                width=_edge_width(attrs, min_edge_weight, max_edge_weight),
                color=EDGE_COLORS.get(edge_type, EDGE_COLORS["其他"]),
                title=title,
                arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
            )

        network_html = net.generate_html(notebook=False)
        st.iframe(network_html, height=height)
        return True, None
    except Exception as exc:
        return False, f"网络图渲染失败：{exc}"


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
