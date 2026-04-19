from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from comment_graph import build_comment_graph, build_pyvis_html, sample_comment_subgraph
from data_loader import (
    DEFAULT_DATA_DIR,
    get_comments_for_weibo,
    get_user_recent_weibo,
    load_processed_data,
    merge_comment_with_user,
)
from utils import (
    DISPLAY_FONT_FAMILY,
    build_comment_option,
    build_weibo_option,
    format_count,
    format_ratio,
    safe_text,
    shorten_text,
)
from charts import (
    PLOTLY_CONFIG,
    build_dataset_card_data,
    build_post_overview_markdown,
    plot_comment_ip_distribution,
    plot_comment_like_distribution,
    plot_comment_quality_distribution,
    plot_comment_time_trend,
    plot_high_quality_user_comparison,
    plot_topic_value_distribution,
    plot_trending_type_distribution,
    plot_user_value_distribution,
    plot_weibo_engagement_distribution,
    style_recent_weibo_table,
)


st.set_page_config(page_title="微博热点评论可视化", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: {DISPLAY_FONT_FAMILY};
    }}
    .block-container {{
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}
    div[data-testid="stMetric"] {{
        background: #f7fafc;
        border: 1px solid #e4ebf3;
        padding: 14px 16px;
        border-radius: 14px;
    }}
    .post-card {{
        background: linear-gradient(180deg, #fbfdff 0%, #f5f8fc 100%);
        border: 1px solid #dfe7f0;
        border-radius: 16px;
        padding: 18px 20px;
        line-height: 1.8;
    }}
    .user-card {{
        background: #ffffff;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        padding: 16px 18px;
        line-height: 1.8;
        box-shadow: 0 6px 18px rgba(31, 78, 121, 0.05);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("基于微博热点事件的评论关系可视化")

st.sidebar.header("数据与筛选")
data_dir = st.sidebar.text_input("数据目录", value=str(DEFAULT_DATA_DIR))

try:
    with st.spinner("正在加载数据并构建展示索引..."):
        datasets = load_processed_data(data_dir)
except Exception as exc:  # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.info(f"请确认目录 `{data_dir}` 中存在 README 中约定的 4 张数据表，或检查文件名兼容映射。")
    st.stop()

topic_weibo = datasets["topic_weibo"]
topic_comment = datasets["topic_comment"]
user_info = datasets["user_info"]
user_weibo = datasets["user_weibo"]
weibo_profile = datasets["weibo_profile"]
topic_summary = datasets["topic_summary"]

if weibo_profile.empty:
    st.warning("热点微博表为空，暂时无法展示。")
    st.stop()

default_weibo = weibo_profile.iloc[0]
topic_options = topic_summary["topic_display"].tolist()
default_topic = default_weibo["topic_display"]
default_topic_index = topic_options.index(default_topic) if default_topic in topic_options else 0
selected_topic = st.sidebar.selectbox("选择话题", topic_options, index=default_topic_index)

topic_weibo_options = weibo_profile.loc[weibo_profile["topic_display"] == selected_topic].copy()
if topic_weibo_options.empty:
    st.warning("当前话题下没有可展示的微博。")
    st.stop()

weibo_id_options = topic_weibo_options["weibo_id"].tolist()
weibo_label_map = {row["weibo_id"]: build_weibo_option(row) for _, row in topic_weibo_options.iterrows()}
default_weibo_id = int(topic_weibo_options.iloc[0]["weibo_id"])
selected_weibo_id = st.sidebar.selectbox(
    "选择微博",
    weibo_id_options,
    index=weibo_id_options.index(default_weibo_id),
    format_func=lambda weibo_id: weibo_label_map.get(weibo_id, str(weibo_id)),
)

selected_weibo = topic_weibo_options.loc[topic_weibo_options["weibo_id"] == selected_weibo_id].iloc[0]
selected_comments = get_comments_for_weibo(topic_comment, int(selected_weibo_id))

max_like_value = (
    int(pd.to_numeric(selected_comments.get("like_count"), errors="coerce").fillna(0).max())
    if not selected_comments.empty
    else 0
)
slider_max = max(20, min(max_like_value, 300))
graph_max_nodes = st.sidebar.slider("评论图最大节点数", min_value=20, max_value=120, value=60, step=10)
graph_min_likes = st.sidebar.slider("评论图最小点赞阈值", min_value=0, max_value=slider_max, value=min(3, slider_max), step=1)
graph_relation_only = st.sidebar.toggle("仅展示存在回复关系的评论", value=True)
graph_high_quality = st.sidebar.toggle("优先高质量评论", value=True)

with st.sidebar.expander("数据文件映射", expanded=False):
    for dataset_name, dataset_path in datasets["dataset_paths"].items():
        st.write(f"`{dataset_name}`")
        st.caption(dataset_path)

summary_columns = st.columns(4)
for column, card in zip(
    summary_columns,
    build_dataset_card_data(topic_weibo=topic_weibo, topic_comment=topic_comment, user_info=user_info, user_weibo=user_weibo),
):
    column.metric(card["label"], format_count(card["value"]), help=card["help"])

tabs = st.tabs(["数据总览", "典型微博", "评论关系网络", "评论与用户联动"])

with tabs[0]:
    st.subheader("数据集规模与整体概况")
    left, right = st.columns(2)
    left.plotly_chart(plot_topic_value_distribution(topic_weibo), use_container_width=True, config=PLOTLY_CONFIG)
    right.plotly_chart(plot_trending_type_distribution(topic_weibo), use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2)
    left.plotly_chart(plot_weibo_engagement_distribution(topic_weibo), use_container_width=True, config=PLOTLY_CONFIG)
    right.plotly_chart(plot_comment_quality_distribution(topic_comment), use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2)
    left.plotly_chart(plot_user_value_distribution(user_info), use_container_width=True, config=PLOTLY_CONFIG)
    right.plotly_chart(plot_comment_ip_distribution(topic_comment), use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("主题概览表", expanded=False):
        display_topic_summary = topic_summary.copy()
        display_topic_summary["avg_engagement"] = display_topic_summary["avg_engagement"].round(2)
        st.dataframe(
            display_topic_summary.rename(
                columns={
                    "topic_display": "话题",
                    "topic_weibo_count": "微博数",
                    "comment_crawled_total": "已采样评论数",
                    "reply_edge_total": "回复边数",
                    "avg_engagement": "平均互动量",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:
    st.subheader("典型热点微博展示")
    post_left, post_right = st.columns([1.45, 1.0])
    post_left.markdown(f"<div class='post-card'>{build_post_overview_markdown(selected_weibo)}</div>", unsafe_allow_html=True)

    with post_right:
        metric_row_1 = st.columns(3)
        metric_row_1[0].metric("总评论数", format_count(selected_weibo.get("comment_count", 0)))
        metric_row_1[1].metric("已采样评论", format_count(selected_weibo.get("comment_crawled_count", len(selected_comments))))
        metric_row_1[2].metric("高质量评论", format_count(selected_weibo.get("comment_hq_count", 0)))

        metric_row_2 = st.columns(3)
        metric_row_2[0].metric("点赞数", format_count(selected_weibo.get("like_count", 0)))
        metric_row_2[1].metric("转发数", format_count(selected_weibo.get("repost_count", 0)))
        metric_row_2[2].metric("高质量占比", format_ratio(selected_weibo.get("hq_comment_ratio_display")))

        metric_row_3 = st.columns(3)
        metric_row_3[0].metric("互动量", format_count(selected_weibo.get("engagement", 0)))
        metric_row_3[1].metric("回复边数", format_count(selected_weibo.get("reply_edge_count", 0)))
        metric_row_3[2].metric("活跃线程数", format_count(selected_weibo.get("active_thread_count", 0)))

    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(plot_comment_time_trend(selected_comments), use_container_width=True, config=PLOTLY_CONFIG)
    chart_right.plotly_chart(plot_comment_like_distribution(selected_comments), use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(
        f"当前案例：话题“{safe_text(selected_weibo.get('topic_display'))}”，发布者 @{safe_text(selected_weibo.get('screen_name'))}，"
        "系统优先按“图结构丰富度 + 已采样评论量 + 互动量”排序推荐。"
    )

with tabs[2]:
    st.subheader("评论回复关系可视化")

    if selected_comments.empty:
        st.warning("所选微博暂无已采样评论，暂时无法构建评论图。")
    else:
        sampled_comments, sample_summary = sample_comment_subgraph(
            comment_frame=selected_comments,
            max_nodes=graph_max_nodes,
            min_likes=graph_min_likes,
            only_high_quality=graph_high_quality,
            relation_only=graph_relation_only,
        )
        graph = build_comment_graph(sampled_comments)
        weak_components = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0

        metric_row = st.columns(4)
        metric_row[0].metric("采样节点数", format_count(sample_summary["sampled_nodes"]))
        metric_row[1].metric("采样边数", format_count(sample_summary["sampled_edges"]))
        metric_row[2].metric("线程数量", format_count(sample_summary["thread_count"]))
        metric_row[3].metric("图中连通分量", format_count(weak_components))

        # st.info(
        #     "采样策略：优先选择高互动、存在回复关系的评论线程；当开启“优先高质量评论”时，先筛出高质量候选，再补齐必要祖先链路，"
        #     "确保答辩展示时既能看到重点评论，也能保留上下文关系。"
        # )

        if sampled_comments.empty:
            st.warning("当前筛选条件下没有可展示的评论子图，请适当放宽点赞阈值或关闭关系过滤。")
        else:
            try:
                graph_html = build_pyvis_html(sampled_comments, title=f"weibo_{selected_weibo_id}_comment_graph")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                graph_html = ""

            if graph_html:
                components.html(graph_html, height=760, scrolling=True)
            else:
                st.warning("评论图 HTML 未成功生成。")

            with st.expander("当前评论子图样本", expanded=False):
                graph_table = sampled_comments[
                    [
                        "comment_id",
                        "screen_name",
                        "content",
                        "like_count",
                        "sub_comment_count",
                        "text_quality_label",
                        "ip_location",
                        "_node_type",
                    ]
                ].copy()
                graph_table["content"] = graph_table["content"].apply(lambda text: shorten_text(text, 60))
                st.dataframe(
                    graph_table.rename(
                        columns={
                            "comment_id": "评论ID",
                            "screen_name": "昵称",
                            "content": "评论内容",
                            "like_count": "点赞数",
                            "sub_comment_count": "回复数",
                            "text_quality_label": "文本质量",
                            "ip_location": "IP属地",
                            "_node_type": "节点类型",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

with tabs[3]:
    st.subheader("评论明细与用户画像联动")
    if selected_comments.empty:
        st.warning("所选微博暂无评论数据。")
    else:
        comment_user_frame = merge_comment_with_user(selected_comments, user_info)

        display_comment_frame = comment_user_frame[
            [
                "comment_id",
                "comment_screen_name",
                "content",
                "like_count",
                "sub_comment_count",
                "text_quality_label",
                "comment_ip_location",
            ]
        ].copy()
        display_comment_frame["content"] = display_comment_frame["content"].apply(lambda text: shorten_text(text, 72))

        st.dataframe(
            display_comment_frame.rename(
                columns={
                    "comment_id": "评论ID",
                    "comment_screen_name": "评论用户",
                    "content": "评论内容",
                    "like_count": "点赞数",
                    "sub_comment_count": "回复数",
                    "text_quality_label": "文本质量",
                    "comment_ip_location": "IP属地",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        comment_options = comment_user_frame["comment_id"].tolist()
        comment_label_map = {
            row["comment_id"]: build_comment_option(
                pd.Series(
                    {
                        "comment_id": row["comment_id"],
                        "screen_name": row["comment_screen_name"],
                        "content": row["content"],
                        "like_count": row["like_count"],
                    }
                )
            )
            for _, row in comment_user_frame.iterrows()
        }

        selected_comment_id = st.selectbox(
            "选择一条评论查看用户画像",
            comment_options,
            format_func=lambda comment_id: comment_label_map.get(comment_id, str(comment_id)),
        )
        selected_comment = comment_user_frame.loc[comment_user_frame["comment_id"] == selected_comment_id].iloc[0]

        user_left, user_right = st.columns([1.0, 1.2])
        verified_flag = bool(selected_comment.get("user_verified")) if pd.notna(selected_comment.get("user_verified")) else False

        user_gender = selected_comment.get('user_gender') or selected_comment.get('comment_gender')
        user_gender = "男" if user_gender == 'm' else "女"


        user_left.markdown(
            (
                "<div class='user-card'>"
                f"<b>评论用户</b>：@{safe_text(selected_comment.get('user_screen_name') or selected_comment.get('comment_screen_name'))}<br>"
                f"<b>性别</b>：{safe_text(user_gender)}<br>"
                f"<b>IP 属地</b>：{safe_text(selected_comment.get('user_ip_location') or selected_comment.get('comment_ip_location'))}<br>"
                f"<b>认证状态</b>：{'是' if verified_flag else '否'}<br>"
                f"<b>认证类型</b>：{safe_text(selected_comment.get('user_verified_type_name'))}<br>"
                f"<b>粉丝数</b>：{format_count(selected_comment.get('user_follower_count'))}<br>"
                f"<b>关注数</b>：{format_count(selected_comment.get('user_following_count'))}<br>"
                f"<b>粉关比</b>：{format_ratio(selected_comment.get('user_follower_following_ratio'))}<br>"
                f"<b>用户等级</b>：{safe_text(selected_comment.get('user_rank_display'))}<br>"
                f"<b>用户价值</b>：{safe_text(selected_comment.get('user_value_label_display'))}<br>"
                f"<b>活跃天数</b>：{format_count(selected_comment.get('user_active_days'))}<br>"
                f"<b>简介</b>：{safe_text(selected_comment.get('user_description'))}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        recent_history = get_user_recent_weibo(user_weibo, int(selected_comment["user_id"]), limit=8)
        user_right.plotly_chart(
            plot_high_quality_user_comparison(comment_user_frame),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        with st.expander("该用户最近历史微博摘要", expanded=True):
            if recent_history.empty:
                st.caption("当前用户在 `user_weibo` 中暂无可展示的历史微博。")
            else:
                st.dataframe(style_recent_weibo_table(recent_history), use_container_width=True, hide_index=True)

st.divider()
with st.expander("实现说明与鲁棒性处理", expanded=False):
    st.markdown(
        f"""
        - 默认数据目录：`{Path(data_dir)}`
        - 文件名兼容：同时兼容 `topic_weibo.parquet` / `df_topic_weibo.parquet` 这类命名
        - 缺失字段处理：统一补齐必要展示列，避免直接因缺列报错
        - 时间字段处理：自动解析 `create_time`、`trending_date`、`registration_time`
        - 评论图边界处理：`parent_id` 为空、非法或不在当前微博评论集合中时，视作一级评论或孤立节点
        - PyVis 嵌入方式：使用 `cdn_resources="in_line"` 生成内联 HTML，再通过 `components.html` 稳定嵌入 Streamlit
        """
    )
