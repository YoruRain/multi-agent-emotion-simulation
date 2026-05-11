## 任务背景

此前已经完成两阶段主题分析：

1. 显式主题分析：
   基于微博话题标签、关键词词典、转发源话题等显式信号生成主题类别。

2. 隐式主题分析：
   对显式主题为空的原创微博，使用 BGE embedding + MiniBatchKMeans 聚类，并人工为 cluster 生成主题解释。

现在已经生成了微博级主题融合表。该表以显式主题为主干，并使用隐式主题补充显式主题缺失的原创微博。

当前任务是：修改已有的用户级主题画像聚合脚本，使其基于微博级主题融合表，生成最终用户级主题画像。

## 现有代码情况

请读取并修改现有文件：

scripts/weibo_analysis/subject_profile/build_user_topic_profile.py

请在现有结构上扩展和调整。

## 新的输入文件

请将默认输入文件从：

data/profile/weibos/subject_profile/user_weibo_topic_signals.parquet

改为：

data/profile/weibos/subject_profile/user_weibo_topic_fusion.parquet

该微博级主题融合表至少包含以下字段：

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

隐式主题相关字段：

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

融合主题相关字段：

- final_topic_categories
- final_topic_labels
- topic_signal_source
- final_topic_confidence

## 新的输出文件

请将最终用户级主题画像保存为：

data/profile/weibos/subject_profile/user_topic_profile_final.parquet

## 最终用户级主题画像字段

请使用以下字段作为最终输出字段：

- user_id
- total_weibo_count
- repost_weibo_count
- original_weibo_count
- top_user_topics
- top_source_topics
- top_all_topics
- top_categories
- top_implicit_topic_labels
- top_final_topic_categories
- final_category_distribution
- public_issue_topic_ratio
- final_public_issue_topic_ratio
- entertainment_topic_ratio
- final_entertainment_topic_ratio
- daily_life_topic_ratio
- final_daily_life_topic_ratio
- repost_topic_dependency
- explicit_topic_coverage
- implicit_valid_topic_coverage
- final_topic_coverage
- topic_source_balance_label
- explicit_topic_profile_reliability
- final_topic_profile_reliability

可以额外保留以下辅助字段，便于检查和后续分析：

- avg_signal_confidence
- avg_final_topic_confidence

如果为了兼容旧代码，也可以保留 top_all_topics、top_categories，但最终输出字段中必须至少包含上面的最低限度字段。

## 字段计算规则

1. user_id

直接按 user_id 分组聚合。

2. top_user_topics

沿用原逻辑，从 user_topics 字段中统计该用户原创微博自写话题标签的 top 10。

注意：

- user_topics 是空值或普通逗号分隔字符串。

3. top_source_topics

沿用原逻辑，从 source_topics 字段中统计该用户转发源微博话题标签 top 10。

4. top_implicit_topic_labels

统计该用户通过隐式聚类获得的有效细主题。

来源字段：

- implicit_topic_label
- implicit_topic_valid

只统计 implicit_topic_valid == True 的记录。

分母使用 total_weibo_count。

输出 top 10，格式与 top_user_topics 保持一致。

5. top_final_topic_categories

统计该用户最终融合后的主题大类。

来源字段：

- final_topic_categories

需要解析 final_topic_categories，其格式为英文逗号分隔字符串或空值

统计 top 10，分母使用 total_weibo_count。

6. final_category_distribution

统计用户最终融合主题类别的完整分布。

来源字段：

- final_topic_categories

格式与现有的 `top_cateories` 相同。

注意：

- 同一条微博若有多个 final_topic_categories，则每个类别都计数一次。
- ratio = category_count / total_weibo_count。
- 类别按 count 降序排列。

7. public_issue_topic_ratio

保留原有显式主题阶段指标。

计算方式：

public_issue_topic_ratio =
该用户 explicit_topic_categories 中包含 PUBLIC_ISSUE_CATEGORIES 任一类别的微博数 / total_weibo_count

它只基于显式主题结果。

8. final_public_issue_topic_ratio

基于最终融合主题计算。

计算方式：

final_public_issue_topic_ratio =
该用户 final_topic_categories 中包含 PUBLIC_ISSUE_CATEGORIES 任一类别的微博数 / total_weibo_count

9. entertainment_topic_ratio

10. final_entertainment_topic_ratio

11. daily_life_topic_ratio

12. final_daily_life_topic_ratio

上述四个字段的计算方式参考第7与第8点。

13. repost_topic_dependency

沿用原逻辑。

14. explicit_topic_coverage

保留原有显式主题覆盖率。

15. implicit_valid_topic_coverage

新增字段，表示有效隐式主题覆盖率。

计算方式：

implicit_valid_topic_coverage =
implicit_topic_valid == True 的微博数 / total_weibo_count

注意：

- 只有隐式主题有效时才计入。
- implicit_topic_valid 是 bool 类型。

16. final_topic_coverage

新增字段，表示显式 + 隐式融合后的最终主题覆盖率。

计算方式：

final_topic_coverage =
final_topic_categories 非空的微博数 / total_weibo_count

17. topic_profile_reliability

保留原有基于显式主题覆盖率的可靠性判断。

可以继续使用原函数 determine_topic_profile_reliability(total_weibo_count, explicit_topic_coverage)。

18. final_topic_profile_reliability

新增字段，基于融合后的主题覆盖率判断最终画像可靠性。

建议规则：

- 如果 total_weibo_count >= 20 且 final_topic_coverage >= 0.5 且 avg_final_topic_confidence >= 0.5：
  - 高可靠

- 如果 total_weibo_count >= 10 且 final_topic_coverage >= 0.3：
  - 中可靠

- 否则：
  - 低可靠

请实现一个新函数：

determine_final_topic_profile_reliability(total_weibo_count, final_topic_coverage, avg_final_topic_confidence) -> str

19. avg_signal_confidence

沿用原逻辑，表示显式主题平均置信度。

20. avg_final_topic_confidence

新增字段，表示用户所有微博 final_topic_confidence 的平均值。

final_topic_confidence 需转换为数值，空值按 0.0 处理。

21. topic_source_balance_label

沿用原逻辑，根据 repost_topic_dependency 判断。

## REQUIRED_COLUMNS

请更新 REQUIRED_COLUMNS，至少包含：

- user_id
- is_repost
- user_topics
- source_topics
- explicit_topic_categories
- signal_confidence
- implicit_topic_label
- implicit_topic_category
- implicit_topic_valid
- implicit_topic_confidence_score
- final_topic_categories
- final_topic_labels
- final_topic_confidence

如果为了计算辅助字段需要，也可以包含：

- topic_signal_source
- text_quality

但不要要求不存在的字段。

## OUTPUT_COLUMNS

请更新 OUTPUT_COLUMNS，至少包含以下字段并按此顺序输出：

- user_id
- total_weibo_count
- repost_weibo_count
- original_weibo_count
- top_user_topics
- top_source_topics
- top_all_topics
- top_categories
- top_implicit_topic_labels
- top_final_topic_categories
- final_category_distribution
- public_issue_topic_ratio
- final_public_issue_topic_ratio
- entertainment_topic_ratio
- final_entertainment_topic_ratio
- daily_life_topic_ratio
- final_daily_life_topic_ratio
- repost_topic_dependency
- explicit_topic_coverage
- implicit_valid_topic_coverage
- final_topic_coverage
- topic_source_balance_label
- explicit_topic_profile_reliability
- final_topic_profile_reliability

建议在这些字段后追加辅助字段：

- avg_signal_confidence
- avg_final_topic_confidence

## 默认路径

请修改默认路径：

DEFAULT_INPUT_PATH =
PROJECT_ROOT / "data" / "profile" / "weibos" / "subject_profile" / "user_weibo_topic_fusion.parquet"

DEFAULT_OUTPUT_PATH =
PROJECT_ROOT / "data" / "profile" / "weibos" / "subject_profile" / "user_topic_profile_final.parquet"

DEFAULT_LOG_FILE_NAME =
"user_topic_profile_final.log"