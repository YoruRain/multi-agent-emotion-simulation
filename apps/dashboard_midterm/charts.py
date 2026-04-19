from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import DISPLAY_FONT_FAMILY, shorten_text


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
COLOR_SEQUENCE = ["#1f4e79", "#2f7e79", "#d17b0f", "#b24c63", "#5b6c8f", "#8d9f4f"]


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16})
    figure.update_layout(height=320)
    return apply_base_layout(figure, title=None)


def apply_base_layout(figure: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        font={"family": DISPLAY_FONT_FAMILY, "size": 13, "color": "#243447"},
        title={"text": title or "", "x": 0.01, "xanchor": "left"},
        colorway=COLOR_SEQUENCE,
        height=height,
        margin={"l": 30, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure


def _count_bar(frame: pd.DataFrame, column: str, title: str, top_n: int | None = None) -> go.Figure:
    if column not in frame.columns or frame.empty:
        return empty_figure("暂无可展示数据")

    counts = frame[column].fillna("未标注").astype(str).value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    chart_data = counts.rename_axis(column).reset_index(name="count")
    figure = px.bar(chart_data, x=column, y="count", text_auto=True)
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="记录数")
    return apply_base_layout(figure, title=title)


def plot_topic_value_distribution(topic_weibo: pd.DataFrame) -> go.Figure:
    return _count_bar(topic_weibo, "topic_value_label_display", "话题价值等级分布")


def plot_trending_type_distribution(topic_weibo: pd.DataFrame) -> go.Figure:
    return _count_bar(topic_weibo, "trending_type_display", "热点类型分布")


def plot_comment_quality_distribution(topic_comment: pd.DataFrame) -> go.Figure:
    return _count_bar(topic_comment, "text_quality_label_display", "评论文本质量分布")


def plot_user_value_distribution(user_info: pd.DataFrame) -> go.Figure:
    if "user_value_label_display" not in user_info.columns and "user_value_label" in user_info.columns:
        user_info = user_info.copy()
        user_info["user_value_label_display"] = user_info["user_value_label"].fillna("未标注")
    return _count_bar(user_info, "user_value_label_display", "用户价值等级分布")


def plot_weibo_engagement_distribution(topic_weibo: pd.DataFrame) -> go.Figure:
    if "engagement" not in topic_weibo.columns or topic_weibo.empty:
        return empty_figure("暂无互动量数据")

    frame = topic_weibo.copy()
    frame["engagement_clip"] = pd.to_numeric(frame["engagement"], errors="coerce").fillna(0).clip(upper=50_000)
    figure = px.histogram(frame, x="engagement_clip", nbins=30)
    figure.update_xaxes(title="互动量（截断至 50,000）")
    figure.update_yaxes(title="微博数量")
    return apply_base_layout(figure, title="热点微博互动量分布")


def plot_comment_time_trend(comment_frame: pd.DataFrame) -> go.Figure:
    if comment_frame.empty or "create_time" not in comment_frame.columns:
        return empty_figure("当前微博暂无评论时间数据")

    trend = comment_frame.dropna(subset=["create_time"]).copy()
    if trend.empty:
        return empty_figure("当前微博暂无评论时间数据")
    trend["time_bucket"] = trend["create_time"].dt.floor("h")
    chart_data = trend.groupby("time_bucket").size().reset_index(name="comment_count")
    figure = px.line(chart_data, x="time_bucket", y="comment_count", markers=True)
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="评论数量")
    return apply_base_layout(figure, title="评论发布时间趋势")


def plot_comment_like_distribution(comment_frame: pd.DataFrame) -> go.Figure:
    if comment_frame.empty or "like_count" not in comment_frame.columns:
        return empty_figure("当前微博暂无评论点赞数据")
    figure = px.histogram(comment_frame, x="like_count", nbins=20)
    figure.update_xaxes(title="评论点赞数")
    figure.update_yaxes(title="评论数量")
    return apply_base_layout(figure, title="评论点赞分布")


def plot_comment_ip_distribution(comment_frame: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if comment_frame.empty or "ip_location" not in comment_frame.columns:
        return empty_figure("暂无 IP 属地数据")
    return _count_bar(comment_frame, "ip_location", "评论用户 IP 属地 Top10", top_n=top_n)


def plot_high_quality_user_comparison(comment_user_frame: pd.DataFrame) -> go.Figure:
    required_columns = {"text_quality_label", "user_follower_count", "user_following_count"}
    if comment_user_frame.empty or not required_columns.issubset(comment_user_frame.columns):
        return empty_figure("暂无可对比的用户画像数据")

    chart = comment_user_frame.copy()
    chart["评论类型"] = chart["text_quality_label"].where(chart["text_quality_label"].eq("可分析"), "其他评论")
    summary = (
        chart.groupby("评论类型")
        .agg(
            平均粉丝数=("user_follower_count", "mean"),
            平均关注数=("user_following_count", "mean"),
        )
        .reset_index()
        .melt(id_vars="评论类型", var_name="指标", value_name="均值")
    )
    if summary.empty:
        return empty_figure("暂无可对比的用户画像数据")
    figure = px.bar(summary, x="评论类型", y="均值", color="指标", barmode="group")
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="平均值")
    return apply_base_layout(figure, title="高质量评论用户 vs 其他评论用户")


def build_post_overview_markdown(selected_weibo: pd.Series) -> str:
    author = selected_weibo.get("screen_name", "未知用户")
    topic = selected_weibo.get("topic_display", "未标注话题")
    create_time = selected_weibo.get("create_time")
    create_time_text = create_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(create_time) else "暂无"
    content = selected_weibo.get("content", "暂无正文")
    return (
        f"<b>话题</b>：{topic}<br><br>"
        f"<b>发布者</b>：@{author}<br><br>"
        f"<b>发布时间</b>：{create_time_text}<br><br>"
        f"<b>微博正文</b>：{content}"
    )


def build_dataset_card_data(
    topic_weibo: pd.DataFrame,
    topic_comment: pd.DataFrame,
    user_info: pd.DataFrame,
    user_weibo: pd.DataFrame,
) -> list[dict[str, Any]]:
    return [
        {"label": "热点微博", "value": len(topic_weibo), "help": "df_topic_weibo / topic_weibo"},
        {"label": "话题评论", "value": len(topic_comment), "help": "df_topic_comment / topic_comment"},
        {"label": "评论用户", "value": len(user_info), "help": "df_user_info / user_info"},
        {"label": "历史微博", "value": len(user_weibo), "help": "df_user_weibo / user_weibo"},
    ]


def style_recent_weibo_table(history_frame: pd.DataFrame) -> pd.DataFrame:
    if history_frame.empty:
        return history_frame

    display_frame = history_frame.copy()
    if "content" in display_frame.columns:
        display_frame["content"] = display_frame["content"].apply(lambda text: shorten_text(text, 72))
    if "create_time" in display_frame.columns:
        display_frame["create_time"] = display_frame["create_time"].dt.strftime("%Y-%m-%d %H:%M")
    return display_frame
