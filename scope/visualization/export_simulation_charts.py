from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:  # Seaborn is optional; matplotlib remains the hard dependency.
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "scope" / "data" / "outputs" / "simulation" / "multiround"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scope" / "data" / "outputs" / "img"

STANDARD_FILE_PREFIXES = {
    "agent_initial_states": ("agent_initial_states", ".csv"),
    "agent_states_by_round": ("agent_states_by_round", ".csv"),
    "round_metrics": ("round_metrics", ".csv"),
}

FIELD_LABELS = {
    "memory_user_level": "用户分层",
    "verified_type_name": "认证类型",
    "propagation_role": "传播角色",
    "influence_score": "影响力分数",
    "susceptibility_score": "易感性分数",
    "activity_score": "活跃度分数",
    "emotion_score": "情绪分数",
    "stance_score": "立场分数",
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
}

VALUE_LABELS = {
    "core": "核心用户",
    "normal": "普通用户",
    "background": "背景用户",
    "kol_first": "关键意见领袖优先",
    "none": "无互动",
    "positive_ratio": "正向情绪比例",
    "neutral_ratio": "中性情绪比例",
    "negative_ratio": "负向情绪比例",
    "support_ratio": "支持比例",
    "neutral_stance_ratio": "中立立场比例",
    "oppose_ratio": "反对比例",
}

MODULE_NAMES = {
    "agent-population-profile": "智能体群体画像",
    "emotion-stance-evolution": "情绪与立场演化",
    "comment-flow-state-details": "评论流与状态明细",
}


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    agents: pd.DataFrame
    states: pd.DataFrame
    metrics: pd.DataFrame


@dataclass(frozen=True)
class ChartSpec:
    module_slug: str
    file_slug: str
    title: str
    render: Callable[[RunData, Path, str, int], bool]


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
    text = value.strip()
    if "," in text or "，" in text:
        parts = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
        mapped = [VALUE_LABELS.get(part, part.replace("KOL", "关键意见领袖")) for part in parts]
        return "，".join(mapped)
    return VALUE_LABELS.get(text, text.replace("KOL", "关键意见领袖"))


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def resolve_run_file(run_dir: Path, key: str) -> Path | None:
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


def load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def load_run(run_id: str, input_root: Path) -> RunData:
    run_dir = input_root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"未找到运行结果目录：{run_dir}")
    return RunData(
        run_dir=run_dir,
        agents=load_csv(resolve_run_file(run_dir, "agent_initial_states")),
        states=load_csv(resolve_run_file(run_dir, "agent_states_by_round")),
        metrics=load_csv(resolve_run_file(run_dir, "round_metrics")),
    )


def configure_plot_style() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    font_family = "Microsoft YaHei"
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams.update(
        {
            "font.sans-serif": [font_family, "SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#e5e7eb",
        }
    )
    if sns is not None:
        sns.set_theme(style="whitegrid", font=font_family, rc={"axes.unicode_minus": False})


def palette(size: int) -> list[Any]:
    if sns is not None:
        return list(sns.color_palette("Set2", max(size, 1)))
    base = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]
    return [base[index % len(base)] for index in range(max(size, 1))]


def output_path(output_dir: Path, module_slug: str, file_slug: str, image_format: str) -> Path:
    return output_dir / module_slug / f"{file_slug}.{image_format}"


def save_figure(fig: plt.Figure, path: Path, dpi: int, bottom_margin: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bottom_margin is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return not df.empty and all(column in df.columns for column in columns)


def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    counts = (
        df[column]
        .fillna("未知")
        .astype(str)
        .replace("", "未知")
        .map(display_value)
        .value_counts()
        .reset_index()
    )
    counts.columns = [column, "count"]
    return counts


def explode_role_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    roles: list[str] = []
    for value in df[column].fillna("未知").astype(str):
        parts = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
        roles.extend(display_value(part) for part in (parts or ["未知"]))
    return pd.Series(roles).value_counts().reset_index(name="count").rename(columns={"index": column})


def render_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path, dpi: int) -> bool:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return False
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = palette(len(data))
    ax.bar(data[x_col].astype(str), data[y_col], color=colors)
    ax.set_xlabel(display_label(x_col))
    ax.set_ylabel("数量")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    save_figure(fig, path, dpi)
    return True


def render_histogram(df: pd.DataFrame, column: str, title: str, path: Path, dpi: int) -> bool:
    if not has_columns(df, [column]):
        return False
    data = coerce_numeric(df, [column]).dropna(subset=[column])
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if sns is not None:
        sns.histplot(data=data, x=column, bins=20, kde=True, color="#4c78a8", ax=ax)
    else:
        ax.hist(data[column], bins=20, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.set_xlabel(display_label(column))
    ax.set_ylabel("数量")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
    save_figure(fig, path, dpi)
    return True


def render_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path, dpi: int) -> bool:
    if not has_columns(df, [x_col, y_col]):
        return False
    data = coerce_numeric(df, [x_col, y_col]).dropna(subset=[x_col, y_col]).sort_values(x_col)
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(data[x_col], data[y_col], marker="o", linewidth=2.2, color="#4c78a8")
    ax.set_xlabel(display_label(x_col))
    ax.set_ylabel(display_label(y_col))
    ax.grid(alpha=0.35)
    x_values = pd.to_numeric(data[x_col], errors="coerce").dropna()
    if not x_values.empty and ((x_values - x_values.round()).abs() < 1e-9).all():
        ax.set_xticks(sorted(x_values.astype(int).unique()))
    save_figure(fig, path, dpi)
    return True


def render_multi_line(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, path: Path, dpi: int) -> bool:
    existing = [column for column in y_cols if column in df.columns]
    if not has_columns(df, [x_col]) or not existing:
        return False
    data = coerce_numeric(df, [x_col, *existing]).dropna(subset=[x_col]).sort_values(x_col)
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = palette(len(existing))
    plotted = False
    for color, column in zip(colors, existing):
        series = data[[x_col, column]].dropna()
        if series.empty:
            continue
        ax.plot(series[x_col], series[column], marker="o", linewidth=2, label=display_label(column), color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel(display_label(x_col))
    ax.set_ylabel("指标值")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.35)
    save_figure(fig, path, dpi)
    return True


def render_interaction_scale_dual_axis(df: pd.DataFrame, title: str, path: Path, dpi: int) -> bool:
    count_cols = ["interaction_count", "high_influence_interaction_count"]
    weight_col = "avg_interaction_weight"
    existing = [column for column in [*count_cols, weight_col] if column in df.columns]
    if not has_columns(df, ["round_id"]) or not existing:
        return False
    data = coerce_numeric(df, ["round_id", *existing]).dropna(subset=["round_id"]).sort_values("round_id")
    if data.empty:
        return False

    fig, left_ax = plt.subplots(figsize=(8.8, 5.0))
    right_ax = left_ax.twinx()
    colors = {
        "interaction_count": "#4c78a8",
        "high_influence_interaction_count": "#f58518",
        "avg_interaction_weight": "#54a24b",
    }
    markers = {
        "interaction_count": "o",
        "high_influence_interaction_count": "^",
    }
    plotted = False
    for column in count_cols:
        if column not in data.columns:
            continue
        series = data[["round_id", column]].dropna()
        if series.empty:
            continue
        left_ax.plot(
            series["round_id"],
            series[column],
            marker=markers[column],
            linewidth=2,
            label=display_label(column),
            color=colors[column],
        )
        plotted = True

    if weight_col in data.columns:
        weight_series = data[["round_id", weight_col]].dropna()
        if not weight_series.empty:
            right_ax.plot(
                weight_series["round_id"],
                weight_series[weight_col],
                marker="s",
                linewidth=2,
                linestyle="--",
                label=display_label(weight_col),
                color=colors[weight_col],
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    left_ax.set_xlabel(display_label("round_id"))
    left_ax.set_ylabel("次数")
    right_ax.set_ylabel("权重")
    left_ax.grid(alpha=0.35)
    right_ax.grid(False)
    x_values = pd.to_numeric(data["round_id"], errors="coerce").dropna()
    if not x_values.empty and ((x_values - x_values.round()).abs() < 1e-9).all():
        left_ax.set_xticks(sorted(x_values.astype(int).unique()))

    lines = left_ax.get_lines() + right_ax.get_lines()
    labels = [line.get_label() for line in lines]
    left_ax.legend(lines, labels, loc="best", frameon=False)
    save_figure(fig, path, dpi)
    return True


def render_stacked_ratio(df: pd.DataFrame, x_col: str, ratio_cols: list[str], title: str, path: Path, dpi: int) -> bool:
    existing = [column for column in ratio_cols if column in df.columns]
    if not has_columns(df, [x_col]) or not existing:
        return False
    data = coerce_numeric(df, [x_col, *existing]).dropna(subset=[x_col]).sort_values(x_col)
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = data[x_col].to_numpy()
    y_values = [data[column].fillna(0).to_numpy() for column in existing]
    hatches = ["///", "...", "\\\\\\", "xxx"]
    colors = palette(len(existing))
    areas = ax.stackplot(x, y_values, colors=colors, alpha=0.9)
    legend_handles: list[Patch] = []
    for index, (area, column) in enumerate(zip(areas, existing)):
        hatch = hatches[index % len(hatches)]
        area.set_hatch(hatch)
        area.set_edgecolor("#111827")
        area.set_linewidth(0.45)
        legend_handles.append(
            Patch(
                facecolor=colors[index],
                edgecolor="#111827",
                hatch=hatch,
                label=display_label(column),
                linewidth=0.45,
            )
        )
    ax.set_xlabel(display_label(x_col))
    ax.set_ylabel("比例")
    ax.set_ylim(0, max(1.0, math.ceil(max(sum(values) for values in zip(*y_values)) * 10) / 10))
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(legend_handles),
        frameon=False,
    )
    ax.grid(alpha=0.35)
    save_figure(fig, path, dpi, bottom_margin=0.16)
    return True


def render_trajectory(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    path: Path,
    dpi: int,
    agent_id: str | None = None,
) -> bool:
    if not has_columns(df, ["agent_id", "round_id", y_col]):
        return False
    data = coerce_numeric(df, ["round_id", y_col]).dropna(subset=["agent_id", "round_id", y_col])
    if agent_id:
        data = data[data["agent_id"].astype(str) == str(agent_id)]
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    if agent_id:
        series = data.sort_values("round_id")
        ax.plot(series["round_id"], series[y_col], marker="o", linewidth=2.2, color="#4c78a8", label=str(agent_id))
        ax.legend(loc="best", frameon=False)
    else:
        for _, agent_rows in data.groupby("agent_id"):
            agent_rows = agent_rows.sort_values("round_id")
            ax.plot(agent_rows["round_id"], agent_rows[y_col], color="#9ca3af", alpha=0.35, linewidth=1)
        mean_rows = data.groupby("round_id", as_index=False)[y_col].mean().sort_values("round_id")
        ax.plot(mean_rows["round_id"], mean_rows[y_col], marker="o", linewidth=2.6, color="#e45756", label="群体平均")
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel(display_label("round_id"))
    ax.set_ylabel(display_label(y_col))
    ax.grid(alpha=0.35)
    save_figure(fig, path, dpi)
    return True


def build_chart_specs(agent_id: str | None = None) -> list[ChartSpec]:
    return [
        ChartSpec(
            "agent-population-profile",
            "user-level-distribution",
            "用户分层分布",
            lambda data, path, fmt, dpi: render_bar(value_counts(data.agents, "memory_user_level"), "memory_user_level", "count", "用户分层分布", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "verified-type-distribution",
            "认证类型分布",
            lambda data, path, fmt, dpi: render_bar(value_counts(data.agents, "verified_type_name"), "verified_type_name", "count", "认证类型分布", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "propagation-role-distribution",
            "传播角色分布",
            lambda data, path, fmt, dpi: render_bar(explode_role_counts(data.agents, "propagation_role"), "propagation_role", "count", "传播角色分布", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "influence-score-histogram",
            "影响力分数直方图",
            lambda data, path, fmt, dpi: render_histogram(data.agents, "influence_score", "影响力分数直方图", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "susceptibility-score-histogram",
            "易感性分数直方图",
            lambda data, path, fmt, dpi: render_histogram(data.agents, "susceptibility_score", "易感性分数直方图", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "activity-score-histogram",
            "活跃度分数直方图",
            lambda data, path, fmt, dpi: render_histogram(data.agents, "activity_score", "活跃度分数直方图", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "initial-emotion-score-distribution",
            "初始情绪分数分布",
            lambda data, path, fmt, dpi: render_histogram(data.agents, "emotion_score", "初始情绪分数分布", path, dpi),
        ),
        ChartSpec(
            "agent-population-profile",
            "initial-stance-score-distribution",
            "初始立场分数分布",
            lambda data, path, fmt, dpi: render_histogram(data.agents, "stance_score", "初始立场分数分布", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "group-average-emotion-by-round",
            "群体平均情绪随轮次变化",
            lambda data, path, fmt, dpi: render_line(data.metrics, "round_id", "avg_emotion_score", "群体平均情绪随轮次变化", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "group-average-stance-by-round",
            "群体平均立场随轮次变化",
            lambda data, path, fmt, dpi: render_line(data.metrics, "round_id", "avg_stance_score", "群体平均立场随轮次变化", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "emotion-distribution-change",
            "情绪分布变化",
            lambda data, path, fmt, dpi: render_stacked_ratio(data.metrics, "round_id", ["positive_ratio", "neutral_ratio", "negative_ratio"], "情绪分布变化", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "stance-distribution-change",
            "立场分布变化",
            lambda data, path, fmt, dpi: render_stacked_ratio(data.metrics, "round_id", ["support_ratio", "neutral_stance_ratio", "oppose_ratio"], "立场分布变化", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "emotion-volatility-and-stance-polarization",
            "情绪波动与立场极化",
            lambda data, path, fmt, dpi: render_multi_line(data.metrics, "round_id", ["emotion_volatility", "stance_volatility", "polarization_score"], "情绪波动与立场极化", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "state-change-intensity",
            "状态变化强度",
            lambda data, path, fmt, dpi: render_multi_line(data.metrics, "round_id", ["avg_abs_emotion_delta", "avg_abs_stance_delta", "max_abs_emotion_delta", "max_abs_stance_delta"], "状态变化强度", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "neighbor-influence-metrics",
            "邻居影响指标",
            lambda data, path, fmt, dpi: render_multi_line(data.metrics, "round_id", ["avg_neighbor_count", "agents_affected_by_neighbors", "avg_neighbor_influence_weight"], "邻居影响指标", path, dpi),
        ),
        ChartSpec(
            "emotion-stance-evolution",
            "interaction-scale-by-round",
            "每轮互动规模",
            lambda data, path, fmt, dpi: render_interaction_scale_dual_axis(data.metrics, "每轮互动规模", path, dpi),
        ),
        ChartSpec(
            "comment-flow-state-details",
            "emotion-score-trajectory",
            "情绪分数轨迹",
            lambda data, path, fmt, dpi: render_trajectory(data.states, "emotion_score", "情绪分数轨迹", path, dpi, agent_id),
        ),
        ChartSpec(
            "comment-flow-state-details",
            "stance-score-trajectory",
            "立场分数轨迹",
            lambda data, path, fmt, dpi: render_trajectory(data.states, "stance_score", "立场分数轨迹", path, dpi, agent_id),
        ),
    ]


def export_charts(
    run_id: str,
    input_root: Path,
    output_root: Path,
    image_format: str,
    dpi: int,
    agent_id: str | None,
) -> tuple[Path, list[Path], list[str]]:
    data = load_run(run_id, input_root)
    destination = output_root / run_id
    generated: list[Path] = []
    skipped: list[str] = []
    configure_plot_style()
    for module_slug in MODULE_NAMES:
        (destination / module_slug).mkdir(parents=True, exist_ok=True)
    for spec in build_chart_specs(agent_id):
        path = output_path(destination, spec.module_slug, spec.file_slug, image_format)
        try:
            ok = spec.render(data, path, image_format, dpi)
        except Exception as exc:
            skipped.append(f"{MODULE_NAMES[spec.module_slug]} / {spec.title}：导出失败（{exc}）")
            continue
        if ok:
            generated.append(path)
        else:
            skipped.append(f"{MODULE_NAMES[spec.module_slug]} / {spec.title}：缺少可用数据")
    manifest = {
        "run_id": run_id,
        "run_dir": str(data.run_dir),
        "output_dir": str(destination),
        "image_format": image_format,
        "agent_id": agent_id,
        "generated": [str(path) for path in generated],
        "skipped": skipped,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination, generated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static images for multiround simulation dashboard charts.")
    parser.add_argument("run_id", help="Simulation run id, for example 20260521_150429_0cdb1d.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Simulation output root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Image output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", help="Image format. Default: png.")
    parser.add_argument("--dpi", type=int, default=180, help="Image dpi. Default: 180.")
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Only export trajectories for the given agent id. By default, show all agents and highlight the group mean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_format = "jpg" if args.format == "jpeg" else args.format
    destination, generated, skipped = export_charts(
        run_id=args.run_id,
        input_root=args.input_root,
        output_root=args.output_root,
        image_format=image_format,
        dpi=args.dpi,
        agent_id=args.agent_id,
    )
    print(f"Output directory: {destination}")
    print(f"Generated images: {len(generated)}")
    for path in generated:
        print(f"- {path.relative_to(destination)}")
    if skipped:
        print(f"Skipped charts: {len(skipped)}")
        print("- See manifest.json for details.")


if __name__ == "__main__":
    main()
