你是一名熟悉 Python、pandas、中文社交媒体数据处理、主题画像构建与数据工程规范的 Coding Agent。请帮我编写一个脚本，用于融合“微博级显式主题分析结果”和“微博级隐式主题聚类结果”，生成“微博级主题融合表”。

## 任务背景

此前已经完成两路主题分析：

1. 显式主题分析：
   基于微博话题标签、关键词词典、转发源话题等显式信号，为微博生成显式主题类别。

2. 隐式主题分析：
   对“原创、可分析、但显式主题类别为空”的微博进行聚类，并人工审阅 k=20 的聚类结果，为每个 cluster_id 编写主题解释。

现在需要将两路结果在微博级别进行融合，生成一张微博级主题融合表。后续会基于该表重新聚合得到最终用户级主题画像。

## 输入文件

1. 微博级显式主题分析结果文件：

data/profile/weibos/subject_profile/user_weibo_topic_signals.parquet

该文件包含字段：

- weibo_id
- user_id
- content
- text_quality
- is_repost
- reposted_weibo_id
- source_content
- has_repost_comment
- user_topics
- source_topics
- explicit_keywords
- explicit_topic_categories
- signal_confidence

2. k=20 的隐式聚类结果文件：

data/profile/weibos/subject_profile/implicit_topic_clustering/implicit_topic_clustering_k20.parquet

该文件包含字段：

- weibo_id
- user_id
- analysis_text
- analysis_text_length
- cluster_id
- distance_to_center

## 输出文件

请将微博级主题融合表保存为：

data/profile/weibos/subject_profile/user_weibo_topic_fusion.parquet

## 请生成的代码文件

只创建一个脚本文件即可：

scripts/weibo_analysis/subject_profile/build_user_weibo_topic_fusion.py

## cluster 主题解释映射

下面是 `CLUSTER_TOPIC_MAPPING` 字典变量，实现了从 cluster_id 到主题解释的映射：

```python
CLUSTER_TOPIC_MAPPING = {
    0: {
        "implicit_topic_label": "小说、影视剧与剧情评价",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "high"
    }, 
    1: {
        "implicit_topic_label": "消费购物、商品信息与商业内容",
        "implicit_topic_category": "广告营销",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    2: {
        "implicit_topic_label": "娱乐舆论、粉圈争议与明星事件",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    3: {
        "implicit_topic_label": "公益活动、账号运营与推广内容",
        "implicit_topic_category": "广告营销",
        "implicit_topic_confidence_level": "medium"
    }, 
    4: {
        "implicit_topic_label": "萌宠、亲子、生活与趣味分享",
        "implicit_topic_category": "日常生活",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    5: {
        "implicit_topic_label": "音乐、演出与文艺娱乐内容",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    6: {
        "implicit_topic_label": "旅行、地点打卡与图文分享",
        "implicit_topic_category": "日常生活",
        "implicit_topic_confidence_level": "high"
    }, 
    7: {
        "implicit_topic_label": "人生感悟、价值思考与自我成长",
        "implicit_topic_category": "情感表达",
        "implicit_topic_confidence_level": "high"
    }, 
    8: {
        "implicit_topic_label": "吐槽、愤怒与社会事件短评",
        "implicit_topic_category": "情感表达",
        "implicit_topic_confidence_level": "medium"
    }, 
    9: {
        "implicit_topic_label": "明星应援、偶像物料与粉圈内容",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "high"
    }, 
    10: {
        "implicit_topic_label": "节日祝福、新年愿望与年度记录",
        "implicit_topic_category": "日常生活",
        "implicit_topic_confidence_level": "high"
    }, 
    11: {
        "implicit_topic_label": "游戏、赛事、票务与线上娱乐互动",
        "implicit_topic_category": "游戏动漫",
        "implicit_topic_confidence_level": "medium"
    }, 
    12: {
        "implicit_topic_label": "粉圈冲突、娱乐八卦与平台舆论",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    13: {
        "implicit_topic_label": "日常疲惫、失眠、通勤与生活烦恼",
        "implicit_topic_category": "情感表达",
        "implicit_topic_confidence_level": "high"
    }, 
    14: {
        "implicit_topic_label": "性别议题、家庭婚恋与社会公平讨论",
        "implicit_topic_category": "社会公共事件",
        "implicit_topic_confidence_level": "high"
    }, 
    15: {
        "implicit_topic_label": "恋爱、CP、追剧与亲密关系讨论",
        "implicit_topic_category": "娱乐文化",
        "implicit_topic_confidence_level": "medium_high"
    }, 
    16: {
        "implicit_topic_label": "睡眠、身体健康与心理压力",
        "implicit_topic_category": "日常生活",
        "implicit_topic_confidence_level": "high"
    }, 
    17: {
        "implicit_topic_label": "饮食、美食与食品健康",
        "implicit_topic_category": "日常生活",
        "implicit_topic_confidence_level": "high"
    }, 
    18: {
        "implicit_topic_label": "碎片化回忆、短句记录与低密度表达",
        "implicit_topic_category": "其他",
        "implicit_topic_confidence_level": "low"
    }, 
    19: {
        "implicit_topic_label": "积极情绪、美好记录与审美表达",
        "implicit_topic_category": "情感表达",
        "implicit_topic_confidence_level": "high"
    }
}
```

如果某个 cluster_id 没有出现在 CLUSTER_TOPIC_MAPPING 中，应将其隐式主题解释设为空，并打印 warning。

## 输出表字段

微博级主题融合表建议包含以下字段：

原显式主题字段：

- weibo_id
- user_id
- content
- text_quality
- is_repost
- reposted_weibo_id
- source_content
- has_repost_comment
- user_topics
- source_topics
- explicit_keywords
- explicit_topic_categories
- signal_confidence

隐式主题字段：

- implicit_cluster_id
- implicit_analysis_text
- implicit_analysis_text_length
- distance_to_center
- implicit_topic_label
- implicit_topic_category
- implicit_topic_confidence_level
- implicit_topic_base_score
- cluster_distance_quantile_group
- distance_factor
- implicit_topic_confidence_score
- implicit_topic_valid

融合主题字段：

- final_topic_categories
- final_topic_labels
- topic_signal_source
- final_topic_confidence

## 字段生成规则

### 一、读取与合并

1. 读取显式主题分析结果 df_explicit。
2. 读取隐式聚类结果 df_implicit。
3. 根据 df_implicit 的 cluster_id，通过 CLUSTER_TOPIC_MAPPING 补充：
   - implicit_topic_label
   - implicit_topic_category
   - implicit_topic_confidence_level
4. 将 df_implicit 与 df_explicit 按 weibo_id 合并。
5. 合并方式应以 df_explicit 为主表，即 left merge。
6. 合并后行数必须等于 df_explicit 行数。


### 二、隐式主题置信度计算

请根据簇级置信度等级和微博级 distance_to_center 共同生成 implicit_topic_confidence_score。

1. 簇级置信度基础分

使用以下映射：

```
CONFIDENCE_BASE = {
    "high": 0.85,
    "medium_high": 0.75,
    "medium": 0.60,
    "low": 0.35,
}
```

如果 implicit_topic_confidence_level 为空或不在上述映射中，则 base_score = 0.0。

2. 簇内距离分位分组

不要直接用全局 distance_to_center，需要在每个 implicit_cluster_id 内部计算距离分位数。

对于每个 cluster_id，计算 distance_to_center 的：

- q25
- q50
- q75
- q90

然后按如下规则给每条微博分配 cluster_distance_quantile_group 和 distance_factor：

- distance <= q25:
  - cluster_distance_quantile_group = "center_core"
  - distance_factor = 1.10

- q25 < distance <= q50:
  - cluster_distance_quantile_group = "center_typical"
  - distance_factor = 1.00

- q50 < distance <= q75:
  - cluster_distance_quantile_group = "middle"
  - distance_factor = 0.90

- q75 < distance <= q90:
  - cluster_distance_quantile_group = "edge"
  - distance_factor = 0.75

- distance > q90:
  - cluster_distance_quantile_group = "far_edge"
  - distance_factor = 0.60

如果 distance_to_center 为空，或 cluster_id 为空，则：
- cluster_distance_quantile_group = ""
- distance_factor = 0.0

3. 隐式主题置信度分数

计算：

implicit_topic_confidence_score = implicit_topic_base_score * distance_factor

然后截断到 [0.0, 1.0] 区间。

4. 隐式主题有效性

生成 implicit_topic_valid，规则为：

implicit_topic_valid = (
    implicit_topic_category 非空
    且 implicit_topic_category != "其他"
    且 implicit_topic_confidence_score >= 0.45
)

如果不满足，则 False。

### 三、融合规则

生成以下字段：

- final_topic_categories
- final_topic_labels
- topic_signal_source
- final_topic_confidence

请按以下优先级处理：

1. 如果 explicit_topic_categories 非空：

- final_topic_categories = explicit_topic_categories
- final_topic_labels = explicit_topic_categories
- topic_signal_source = "显式主题"
- final_topic_confidence = signal_confidence


2. 如果 explicit_topic_categories 为空，但 implicit_topic_valid == True：

- final_topic_categories = [implicit_topic_category]
- final_topic_labels = [implicit_topic_label]
- topic_signal_source = "隐式主题"
- final_topic_confidence = implicit_topic_confidence_score


3. 如果 explicit_topic_categories 为空，且存在 implicit_topic_label，但 implicit_topic_valid == False：

- final_topic_categories = []
- final_topic_labels = [implicit_topic_label]
- topic_signal_source = "隐式低置信度"
- final_topic_confidence = implicit_topic_confidence_score


4. 如果 explicit_topic_categories 为空，且不存在隐式主题结果：

- final_topic_categories = []
- final_topic_labels = []
- topic_signal_source = "no_topic_signal"
- final_topic_confidence = 0.0

注意：

- final_topic_categories 和 final_topic_labels 保存为字符串形式；若存在多值，则使用逗号 `,` 分隔。
- 如果 final_topic_confidence 为空，应设为 0.0。
- final_topic_confidence 应截断到 [0.0, 1.0] 区间。
- 当前隐式聚类只针对显式主题为空的原创微博，因此一般不需要处理显式与隐式冲突。
- 如果未来有显式和隐式同时存在的记录，本脚本当前仍按“显式优先”处理。

## 日志输出

脚本运行时请打印以下信息：

- 显式主题表行数
- 隐式聚类表行数
- 成功匹配到隐式聚类结果的微博数
- cluster_id 未出现在 CLUSTER_TOPIC_MAPPING 中的数量和 ID 列表
- explicit_topic_categories 非空微博数
- implicit_topic_valid 为 True 的微博数
- topic_signal_source 的 value_counts()
- final_topic_categories 非空微博数
- final_topic_coverage，即 final_topic_categories 非空微博数 / 总微博数
- final_topic_confidence 的 describe() 统计
- 输出 parquet 路径
- 输出 csv 预览路径


请根据以上要求完成脚本实现，并在代码末尾添加：

```python
if __name__ == "__main__":
    main()
```