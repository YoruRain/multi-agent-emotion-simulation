from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STANDARD_FILE_PREFIXES = {
    "config": ("config", ".json"),
    "selected_event": ("selected_event", ".json"),
    "dynamics_summary": ("dynamics_summary", ".json"),
    "agent_initial_states": ("agent_initial_states", ".csv"),
    "agent_states_by_round": ("agent_states_by_round", ".csv"),
    "active_reactions": ("active_reactions", ".jsonl"),
    "interactions": ("interactions", ".csv"),
    "network": ("network", ".graphml"),
    "round_metrics": ("round_metrics", ".csv"),
}


def list_simulation_runs(base_dir: Path) -> list[Path]:
    """Return run directories sorted by modified time descending."""

    try:
        if not base_dir.exists():
            return []
        return sorted(
            [path for path in base_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []


def resolve_run_file(run_dir: Path | None, key: str) -> Path | None:
    """Resolve a standard output file, accepting timestamp-suffixed legacy names."""

    if run_dir is None or not run_dir.exists():
        return None
    prefix, suffix = STANDARD_FILE_PREFIXES[key]
    standard = run_dir / f"{prefix}{suffix}"
    if standard.exists():
        return standard
    matches = sorted(
        run_dir.glob(f"{prefix}*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def resolve_run_files(run_dir: Path | None) -> dict[str, Path | None]:
    return {key: resolve_run_file(run_dir, key) for key in STANDARD_FILE_PREFIXES}


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_jsonl(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if isinstance(item, dict):
                    rows.append(item)
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def load_graph(path: Path | None) -> nx.DiGraph | None:
    if path is None or not path.exists():
        return None
    try:
        graph = nx.read_graphml(path)
        return nx.DiGraph(graph)
    except Exception:
        return None


def truncate_text(text: Any, max_len: int = 80) -> str:
    value = "" if pd.isna(text) else str(text)
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def safe_round(value: Any, ndigits: int = 3) -> Any:
    try:
        if pd.isna(value):
            return value
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def round_numeric_columns(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include=["number"]).columns:
        result[column] = result[column].round(ndigits)
    return result


def missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return not df.empty and not missing_columns(df, columns)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def plot_line(df: pd.DataFrame, x: str, y: str, title: str):
    if not has_columns(df, [x, y]):
        return None
    data = coerce_numeric(df, [x, y]).dropna(subset=[x, y])
    if data.empty:
        return None
    fig = px.line(data, x=x, y=y, markers=True, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=360)
    return fig


def plot_multi_line(df: pd.DataFrame, x: str, y_cols: list[str], title: str):
    existing = [column for column in y_cols if column in df.columns]
    if not has_columns(df, [x]) or not existing:
        return None
    data = coerce_numeric(df, [x, *existing]).dropna(subset=[x])
    if data.empty:
        return None
    fig = px.line(data, x=x, y=existing, markers=True, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=360, legend_title_text="")
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    if not has_columns(df, [x, y]):
        return None
    data = df[[x, y]].dropna()
    if data.empty:
        return None
    fig = px.bar(data, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=360)
    return fig


def plot_stacked_ratios(df: pd.DataFrame, x: str, ratio_cols: list[str], title: str):
    existing = [column for column in ratio_cols if column in df.columns]
    if not has_columns(df, [x]) or not existing:
        return None
    data = coerce_numeric(df, [x, *existing])
    melted = data.melt(id_vars=x, value_vars=existing, var_name="指标", value_name="比例")
    melted = melted.dropna(subset=[x, "比例"])
    if melted.empty:
        return None
    fig = px.area(melted, x=x, y="比例", color="指标", title=title, groupnorm="fraction")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=360, yaxis_tickformat=".0%")
    return fig


def plot_histogram(df: pd.DataFrame, column: str, title: str):
    if not has_columns(df, [column]):
        return None
    data = coerce_numeric(df, [column]).dropna(subset=[column])
    if data.empty:
        return None
    fig = px.histogram(data, x=column, nbins=20, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=330)
    return fig


def value_count_frame(series: pd.Series, label: str, count_label: str = "数量") -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[label, count_label])
    counts = series.fillna("未知").astype(str).replace("", "未知").value_counts().reset_index()
    counts.columns = [label, count_label]
    return counts


def explode_roles(series: pd.Series) -> pd.DataFrame:
    roles: list[str] = []
    for value in series.fillna("未知").astype(str):
        parts = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
        roles.extend(parts or ["未知"])
    return value_count_frame(pd.Series(roles), "propagation_role")


def build_graph_from_interactions(interactions: pd.DataFrame, agents: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    if not agents.empty and "agent_id" in agents.columns:
        latest_agents = agents.drop_duplicates("agent_id", keep="last")
        for row in latest_agents.to_dict("records"):
            agent_id = str(row.get("agent_id", ""))
            if not agent_id:
                continue
            graph.add_node(
                agent_id,
                agent_id=agent_id,
                user_id=str(row.get("user_id", "")),
                memory_user_level=str(row.get("memory_user_level", "")),
                verified_type_name=str(row.get("verified_type_name", "")),
                propagation_role=str(row.get("propagation_role", "")),
                influence_score=safe_round(row.get("influence_score", 0.0), 4),
                susceptibility_score=safe_round(row.get("susceptibility_score", 0.0), 4),
                activity_score=safe_round(row.get("activity_score", 0.0), 4),
                final_emotion_score=safe_round(row.get("emotion_score", 0.0), 4),
                final_stance_score=safe_round(row.get("stance_score", 0.0), 4),
            )
    if interactions.empty or missing_columns(interactions, ["source_agent_id", "target_agent_id"]):
        return graph
    for row in interactions.to_dict("records"):
        source = str(row.get("source_agent_id", "")).strip()
        target = str(row.get("target_agent_id", "")).strip()
        if not source or not target:
            continue
        weight = safe_round(row.get("weight", 1.0), 4)
        weight = weight if isinstance(weight, float | int) else 1.0
        if not graph.has_node(source):
            graph.add_node(source, agent_id=source)
        if not graph.has_node(target):
            graph.add_node(target, agent_id=target)
        interaction_type = str(row.get("interaction_type", "influence_candidate"))
        round_id = int(float(row.get("round_id", 0) or 0))
        if graph.has_edge(source, target):
            edge = graph[source][target]
            edge["weight_sum"] = round(float(edge.get("weight_sum", 0.0)) + float(weight), 4)
            edge["interaction_count"] = int(edge.get("interaction_count", 0)) + 1
            edge["first_round"] = min(int(edge.get("first_round", round_id)), round_id)
            edge["last_round"] = max(int(edge.get("last_round", round_id)), round_id)
            edge["weight"] = round(edge["weight_sum"] / edge["interaction_count"], 4)
            types = set(str(edge.get("interaction_types", "")).split(","))
            types.add(interaction_type)
            edge["interaction_types"] = ",".join(sorted(token for token in types if token))
            edge["interaction_type"] = edge["interaction_types"]
        else:
            graph.add_edge(
                source,
                target,
                weight=float(weight),
                weight_sum=float(weight),
                interaction_count=1,
                first_round=round_id,
                last_round=round_id,
                interaction_type=interaction_type,
                interaction_types=interaction_type,
            )
    return graph


def latest_states(agent_states: pd.DataFrame) -> pd.DataFrame:
    if agent_states.empty or missing_columns(agent_states, ["agent_id", "round_id"]):
        return pd.DataFrame()
    data = coerce_numeric(agent_states, ["round_id"])
    data = data.sort_values(["agent_id", "round_id"])
    return data.drop_duplicates("agent_id", keep="last")


def metric_value(summary: dict[str, Any], metrics: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column in summary:
        return summary.get(column, default)
    if not metrics.empty and column in metrics.columns:
        series = metrics[column].dropna()
        if not series.empty:
            return series.iloc[-1]
    return default


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(title=title, height=320, margin=dict(l=20, r=20, t=55, b=20))
    return fig
