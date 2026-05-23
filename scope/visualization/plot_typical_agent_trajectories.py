from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "scope"
    / "data"
    / "outputs"
    / "simulation"
    / "multiround"
    / "20260521_150429_0cdb1d"
    / "agent_states_by_round05211504.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "scope"
    / "data"
    / "outputs"
    / "img"
    / "20260521_150429_0cdb1d"
)

AGENTS = {
    "weibo_user_5539703320": "智能体 A（高影响力）",
    "weibo_user_2189371845": "智能体 B（普通活跃）",
    "weibo_user_5864080129": "智能体 C （观望型）",
}

REQUIRED_COLUMNS = {"agent_id", "round_id", "emotion_score", "stance_score"}


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
            "grid.linewidth": 0.8,
        }
    )


def load_agent_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")

    data = pd.read_csv(path, encoding="utf-8")
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing = "、".join(sorted(missing_columns))
        raise ValueError(f"输入文件缺少必要字段：{missing}")

    selected = data[data["agent_id"].isin(AGENTS)].copy()
    missing_agents = [agent_id for agent_id in AGENTS if agent_id not in set(selected["agent_id"])]
    if missing_agents:
        missing = "、".join(missing_agents)
        raise ValueError(f"输入文件中未找到指定智能体：{missing}")

    for column in ["round_id", "emotion_score", "stance_score"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected = selected.dropna(subset=["round_id", "emotion_score", "stance_score"])
    selected["round_id"] = selected["round_id"].astype(int)
    selected = selected.sort_values(["agent_id", "round_id"])
    return selected


def plot_score_trajectory(
    data: pd.DataFrame,
    score_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    colors = {
        "weibo_user_5539703320": "#dc2626",
        "weibo_user_2189371845": "#16a34a",
        "weibo_user_5864080129": "#2563eb",
    }
    markers = {
        "weibo_user_5539703320": "o",
        "weibo_user_2189371845": "s",
        "weibo_user_5864080129": "^",
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for agent_id, label in AGENTS.items():
        agent_data = data[data["agent_id"] == agent_id]
        ax.plot(
            agent_data["round_id"],
            agent_data[score_column],
            marker=markers[agent_id],
            markersize=6,
            linewidth=2.2,
            color=colors[agent_id],
            label=label,
        )

    ax.axhline(0, color="#6b7280", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("轮次", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, axis="both", linestyle="-", alpha=0.9)
    ax.legend(loc="best", frameon=False, fontsize=11)
    ax.margins(x=0.03)

    rounds = sorted(data["round_id"].unique())
    ax.set_xticks(rounds)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_agent_states(INPUT_FILE)

    emotion_path = OUTPUT_DIR / "typical_agents_emotion_trajectory.png"
    stance_path = OUTPUT_DIR / "typical_agents_stance_trajectory.png"

    plot_score_trajectory(
        data,
        "emotion_score",
        "典型智能体情绪分数变化轨迹",
        "情绪分数",
        emotion_path,
    )
    plot_score_trajectory(
        data,
        "stance_score",
        "典型智能体立场分数变化轨迹",
        "立场分数",
        stance_path,
    )

    print(f"Saved emotion trajectory chart: {emotion_path}")
    print(f"Saved stance trajectory chart: {stance_path}")


if __name__ == "__main__":
    main()
