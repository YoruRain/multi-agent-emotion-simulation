from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

import networkx as nx
import pandas as pd

from utils import DISPLAY_FONT_FAMILY, safe_text, shorten_text


def _as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def annotate_comment_relations(comment_frame: pd.DataFrame) -> pd.DataFrame:
    frame = comment_frame.copy()
    comment_ids = set(frame["comment_id"].tolist())
    frame["_parent_valid"] = frame["parent_id"].isin(comment_ids)
    child_counts = frame.loc[frame["_parent_valid"], "parent_id"].value_counts()
    frame["_child_count_in_set"] = frame["comment_id"].map(child_counts).fillna(0).astype(int)
    frame["_has_child"] = frame["_child_count_in_set"] > 0
    frame["_score"] = (
        pd.to_numeric(frame.get("like_count"), errors="coerce").fillna(0) * 4
        + pd.to_numeric(frame.get("sub_comment_count"), errors="coerce").fillna(0) * 10
        + pd.to_numeric(frame.get("engagement"), errors="coerce").fillna(0)
    )
    return frame


def _build_parent_maps(frame: pd.DataFrame) -> tuple[dict[int, int], dict[int, list[int]]]:
    parent_map: dict[int, int] = {}
    children_map: dict[int, list[int]] = defaultdict(list)
    valid_ids = set(frame["comment_id"].tolist())
    for row in frame.itertuples(index=False):
        comment_id = int(row.comment_id)
        parent_id = getattr(row, "parent_id", None)
        if pd.notna(parent_id):
            try:
                parent_value = int(parent_id)
            except (TypeError, ValueError):
                parent_value = -1
        else:
            parent_value = -1
        parent_map[comment_id] = parent_value
        if parent_value in valid_ids:
            children_map[parent_value].append(comment_id)
    for key, children in children_map.items():
        children.sort()
    return parent_map, children_map


def _compute_roots_and_depths(frame: pd.DataFrame, parent_map: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]:
    comment_ids = set(frame["comment_id"].tolist())
    root_map: dict[int, int] = {}
    depth_map: dict[int, int] = {}

    for comment_id in comment_ids:
        current = comment_id
        depth = 0
        visited: set[int] = set()
        while current in comment_ids and current not in visited:
            visited.add(current)
            parent_id = parent_map.get(current, -1)
            if parent_id not in comment_ids:
                break
            current = parent_id
            depth += 1
        root_map[comment_id] = current
        depth_map[comment_id] = depth

    return root_map, depth_map


def _expand_ancestor_chain(candidate_ids: set[int], parent_map: dict[int, int], valid_ids: set[int]) -> set[int]:
    expanded = set(candidate_ids)
    for comment_id in list(candidate_ids):
        current = comment_id
        visited: set[int] = set()
        while current in valid_ids and current not in visited:
            visited.add(current)
            parent_id = parent_map.get(current, -1)
            if parent_id not in valid_ids:
                break
            expanded.add(parent_id)
            current = parent_id
    return expanded


def sample_comment_subgraph(
    comment_frame: pd.DataFrame,
    max_nodes: int = 60,
    min_likes: int = 0,
    only_high_quality: bool = False,
    relation_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if comment_frame.empty:
        return comment_frame.copy(), {"strategy": "无评论数据可采样"}

    annotated = annotate_comment_relations(comment_frame)
    valid_ids = set(annotated["comment_id"].tolist())
    parent_map, _ = _build_parent_maps(annotated)
    root_map, depth_map = _compute_roots_and_depths(annotated, parent_map)
    annotated["_root_id"] = annotated["comment_id"].map(root_map)
    annotated["_depth"] = annotated["comment_id"].map(depth_map)

    quality_mask = pd.Series(True, index=annotated.index)
    if only_high_quality and "text_quality_label" in annotated.columns:
        quality_mask = annotated["text_quality_label"].fillna("").eq("可分析")

    interaction_mask = (
        pd.to_numeric(annotated["like_count"], errors="coerce").fillna(0).ge(min_likes)
        | annotated["_parent_valid"]
        | annotated["_has_child"]
    )
    candidate_mask = quality_mask & interaction_mask
    if relation_only:
        candidate_mask &= annotated["_parent_valid"] | annotated["_has_child"]

    candidate_ids = set(annotated.loc[candidate_mask, "comment_id"].tolist())
    if not candidate_ids:
        fallback = annotated.sort_values(["_score", "_child_count_in_set"], ascending=False)
        candidate_ids = set(fallback.head(max(10, min(max_nodes, len(fallback))))["comment_id"].tolist())

    selected_pool_ids = _expand_ancestor_chain(candidate_ids, parent_map, valid_ids)

    root_scores = (
        annotated.loc[annotated["comment_id"].isin(selected_pool_ids)]
        .groupby("_root_id")
        .agg(root_score=("_score", "sum"), node_count=("comment_id", "count"))
        .sort_values(["root_score", "node_count"], ascending=False)
    )

    selected_ids: set[int] = set()
    for root_id in root_scores.index:
        thread_ids = set(
            annotated.loc[
                (annotated["_root_id"] == root_id) & (annotated["comment_id"].isin(selected_pool_ids)),
                "comment_id",
            ].tolist()
        )
        if not thread_ids:
            continue
        if len(selected_ids) + len(thread_ids) <= max_nodes or not selected_ids:
            selected_ids.update(thread_ids)
        if len(selected_ids) >= max_nodes:
            break

    if len(selected_ids) > max_nodes:
        trimmed_ids: set[int] = set()
        ordered = annotated.loc[annotated["comment_id"].isin(selected_ids)].sort_values(
            ["_score", "_depth"], ascending=[False, True]
        )
        for row in ordered.itertuples(index=False):
            current_id = int(row.comment_id)
            chain_ids = {current_id}
            current = current_id
            while parent_map.get(current, -1) in selected_ids:
                current = parent_map[current]
                chain_ids.add(current)
            if len(trimmed_ids | chain_ids) <= max_nodes or not trimmed_ids:
                trimmed_ids.update(chain_ids)
            if len(trimmed_ids) >= max_nodes:
                break
        selected_ids = trimmed_ids

    sampled = annotated.loc[annotated["comment_id"].isin(selected_ids)].copy()
    sampled["_parent_in_sample"] = sampled["parent_id"].isin(sampled["comment_id"])
    sampled["_node_type"] = "孤立评论"
    sampled.loc[~sampled["_parent_in_sample"] & sampled["_has_child"], "_node_type"] = "一级评论"
    sampled.loc[sampled["_parent_in_sample"], "_node_type"] = "回复评论"
    sampled = sampled.sort_values(["_depth", "_score"], ascending=[True, False]).reset_index(drop=True)

    summary = {
        "strategy": "按互动强度优先选择线程，并在必要时补齐祖先链路",
        "sampled_nodes": int(len(sampled)),
        "sampled_edges": int(sampled["_parent_in_sample"].sum()),
        "thread_count": int(sampled.loc[~sampled["_parent_in_sample"], "comment_id"].nunique()),
        "high_quality_mode": only_high_quality,
        "relation_only_mode": relation_only,
        "min_likes": min_likes,
        "max_nodes": max_nodes,
    }
    return sampled, summary


def build_comment_graph(sampled_comments: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for _, row in sampled_comments.iterrows():
        graph.add_node(
            int(row["comment_id"]),
            screen_name=safe_text(row.get("screen_name")),
            content=safe_text(row.get("content")),
            like_count=_as_int(row.get("like_count")),
            sub_comment_count=_as_int(row.get("sub_comment_count")),
            text_quality_label=safe_text(row.get("text_quality_label")),
            ip_location=safe_text(row.get("ip_location")),
            node_type=row.get("_node_type", "评论"),
            depth=_as_int(row.get("_depth")),
        )
    for _, row in sampled_comments.iterrows():
        if bool(row.get("_parent_in_sample", False)):
            graph.add_edge(int(row["parent_id"]), int(row["comment_id"]))
    return graph


def _build_hover_html(row: pd.Series) -> str:
    create_time = row.get("create_time")
    create_time_text = create_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(create_time) else "暂无"
    return (
        f"<div style='font-family:{DISPLAY_FONT_FAMILY};line-height:1.6;'>"
        f"<b>{safe_text(row.get('screen_name'))}</b><br>"
        f"评论ID：{row.get('comment_id')}<br>"
        f"类型：{row.get('_node_type')}<br>"
        f"发布时间：{create_time_text}<br>"
        f"点赞：{_as_int(row.get('like_count'))} | 回复数：{_as_int(row.get('sub_comment_count'))}<br>"
        f"文本质量：{safe_text(row.get('text_quality_label'))}<br>"
        f"IP 属地：{safe_text(row.get('ip_location'))}<br>"
        f"内容：{safe_text(row.get('content'))}"
        f"</div>"
    )


def _safe_html_name(title: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(title)).strip("._")
    if not sanitized:
        sanitized = "comment_graph"
    if not sanitized.lower().endswith(".html"):
        sanitized = f"{sanitized}.html"
    return sanitized


def _build_position_map(sampled_comments: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """为评论子图生成“中心聚合 + 向外扩散”的静态坐标。"""
    if sampled_comments.empty:
        return {}

    frame = sampled_comments.copy()
    frame["_score"] = pd.to_numeric(frame.get("_score"), errors="coerce").fillna(0)
    children_map: dict[int, list[int]] = defaultdict(list)
    row_map = {int(row["comment_id"]): row for _, row in frame.iterrows()}

    for _, row in frame.iterrows():
        if bool(row.get("_parent_in_sample", False)):
            children_map[int(row["parent_id"])].append(int(row["comment_id"]))

    for parent_id, child_ids in children_map.items():
        child_ids.sort(key=lambda item: row_map[item].get("_score", 0), reverse=True)

    def subtree_size(node_id: int) -> int:
        child_ids = children_map.get(node_id, [])
        if not child_ids:
            return 1
        return 1 + sum(subtree_size(child_id) for child_id in child_ids)

    def place_branch(node_id: int, start_angle: float, end_angle: float, depth: int) -> None:
        node_radius = 110.0 + depth * 180.0
        node_angle = (start_angle + end_angle) / 2
        positions[node_id] = (
            round(node_radius * math.cos(node_angle), 2),
            round(node_radius * math.sin(node_angle), 2),
        )

        child_ids = children_map.get(node_id, [])
        if not child_ids:
            return

        child_sector = end_angle - start_angle
        child_sizes = [subtree_size(child_id) for child_id in child_ids]
        total_size = sum(child_sizes) or len(child_ids)

        cursor = start_angle
        min_sector = math.radians(12)
        if len(child_ids) * min_sector > child_sector:
            min_sector = child_sector / max(1, len(child_ids))

        for child_id, child_size in zip(child_ids, child_sizes):
            proportional_sector = child_sector * (child_size / total_size)
            allocated_sector = max(proportional_sector, min_sector)
            next_cursor = min(end_angle, cursor + allocated_sector)
            place_branch(child_id, cursor, next_cursor, depth + 1)
            cursor = next_cursor

    roots = frame.loc[~frame["_parent_in_sample"]].copy()
    if roots.empty:
        roots = frame.loc[frame["_depth"].eq(0)].copy()
    roots = roots.sort_values(["_score", "comment_id"], ascending=[False, True])

    root_ids = [int(comment_id) for comment_id in roots["comment_id"].tolist()]
    positions: dict[int, tuple[float, float]] = {}
    if not root_ids:
        return positions

    root_sizes = [subtree_size(root_id) for root_id in root_ids]
    total_root_size = sum(root_sizes) or len(root_ids)
    full_circle = math.tau
    base_cursor = -math.pi / 2
    root_count = len(root_ids)
    desired_root_spacing = 68.0
    root_radius = max(95.0, min(185.0, (root_count * desired_root_spacing) / full_circle))

    for root_id, root_size in zip(root_ids, root_sizes):
        sector = full_circle * (root_size / total_root_size)
        root_angle = base_cursor + sector / 2
        positions[root_id] = (
            round(root_radius * math.cos(root_angle), 2),
            round(root_radius * math.sin(root_angle), 2),
        )
        place_branch(root_id, base_cursor, base_cursor + sector, depth=0)
        positions[root_id] = (
            round(root_radius * math.cos(root_angle), 2),
            round(root_radius * math.sin(root_angle), 2),
        )
        base_cursor += sector

    for comment_id in frame["comment_id"].tolist():
        comment_id_int = int(comment_id)
        if comment_id_int not in positions:
            depth = _as_int(row_map[comment_id_int].get("_depth"))
            fallback_radius = 110.0 + depth * 180.0
            positions[comment_id_int] = (0.0, fallback_radius)

    return positions


def build_pyvis_html(sampled_comments: pd.DataFrame, title: str = "评论回复关系图") -> str:
    if sampled_comments.empty:
        return ""
    try:
        from pyvis.network import Network
    except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失时的兜底分支
        raise RuntimeError("缺少 pyvis 依赖，请先执行 `pip install -r requirements.txt`。") from exc

    network = Network(height="720px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    network.toggle_physics(False)
    color_map = {"一级评论": "#1f4e79", "回复评论": "#d17b0f", "孤立评论": "#5b6c8f"}
    position_map = _build_position_map(sampled_comments)

    for _, row in sampled_comments.iterrows():
        size = 18 + min(_as_int(row.get("like_count")), 60) * 0.25 + min(_as_int(row.get("sub_comment_count")), 12)
        label = shorten_text(row.get("screen_name"), 10)
        node_type = row.get("_node_type", "评论")
        node_x, node_y = position_map.get(int(row["comment_id"]), (0.0, 0.0))
        network.add_node(
            int(row["comment_id"]),
            label=label,
            title=_build_hover_html(row),
            color=color_map.get(node_type, "#5b6c8f"),
            size=size,
            shape="dot",
            font={"face": "Microsoft YaHei", "size": 16, "color": "#1f2d3d"},
            x=node_x,
            y=node_y,
            physics=False,
        )

    for _, row in sampled_comments.iterrows():
        if bool(row.get("_parent_in_sample", False)):
            network.add_edge(int(row["parent_id"]), int(row["comment_id"]), arrows="to", color="#b7c4d6")

    network.set_options(
        """
        var options = {
          "layout": {
            "improvedLayout": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "cubicBezier",
              "forceDirection": "vertical",
              "roundness": 0.28
            }
          },
          "physics": {
            "enabled": false
          }
        }
        """
    )

    html = network.generate_html(name=_safe_html_name(title))
    return html.replace(
        "</head>",
        f"<style>body{{font-family:{DISPLAY_FONT_FAMILY};}} #mynetwork{{border:1px solid #e7edf5;border-radius:12px;}}</style></head>",
    )
