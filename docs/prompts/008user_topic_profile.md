你是一名熟悉 Python、pandas、中文社交媒体数据处理和数据画像构建的 Coding Agent。请帮我编写一段脚本，根据微博级主题显式信号表，聚合生成“用户级主题画像表”。

## 任务背景

此前已经生成了微博级主题显式信号表，其中每条微博包含用户自写文本中的话题标签、转发源微博中的话题标签、显式主题类别、显式关键词和主题信号置信度等字段。

现在需要进一步按照 user_id 聚合，生成用户级主题画像表，用于描述每个核心用户的长期主题关注方向、公共议题参与程度、娱乐/日常等主题偏好，以及其主题信号是否主要依赖转发源微博。

## 输入文件

微博级主题表路径：

.\data\profile\weibos\subject_profile\user_weibo_topic_signals.parquet

该表至少包含以下字段：

- weibo_id
- user_id
- content
- is_repost
- has_repost_comment
- user_topics
- source_topics
- explicit_keywords
- explicit_topic_categories
- signal_confidence

注意：

1. user_topics、source_topics 字段是字符串形式，存在多值时，会使用逗号 `,` 分隔。
2. explicit_topic_categories 字段目前使用中文类别名。如果存在多值，则使用逗号 `,` 分隔
3. explicit_topic_categories 如果存在多值，则使用英文逗号 `,` 分隔。
4. explicit_topic_categories 当前包含以下类别：

- 社会公共事件
- 政策民生
- 时事政治
- 娱乐文化
- 日常生活
- 游戏动漫
- 情感表达
- 广告营销
- 媒体官方

## 输出文件

请将生成的用户级主题画像表保存到：

.\data\profile\weibos\subject_profile\user_topic_profile.parquet

## 输出字段

用户级主题画像表至少包含以下字段：

- user_id
- top_user_topics
- top_source_topics
- public_issue_topic_ratio
- entertainment_topic_ratio
- daily_life_topic_ratio
- repost_topic_dependency
- explicit_topic_coverage

如果为了质量检查和后续分析方便，也可以额外加入以下辅助字段：

- total_weibo_count
- repost_weibo_count
- original_weibo_count
- has_explicit_category_ratio
- avg_signal_confidence
- topic_source_balance_label

## 字段计算规则

1. total_weibo_count

当前用户在微博级主题表中的微博总数。

2. repost_weibo_count 和 original_weibo_count

- repost_weibo_count：is_repost == True 的微博数量。
- original_weibo_count：is_repost == False 的微博数量。

3. has_explicit_category_ratio：`explicit_topic_categories` 字段非空的微博数占总微博数的比例

4. public_issue_topic_ratio

表示该用户历史微博中具有公共议题属性的微博占比。

公共议题类别定义为：

- 社会公共事件
- 政策民生
- 时事政治
- 媒体官方

计算方式：

public_issue_topic_ratio =
当前用户 explicit_topic_categories 中命中上述公共议题类别的微博数 / 当前用户微博总数

注意：

- 只要一条微博的 explicit_topic_categories 中包含任意一个公共议题类别，就算作公共议题微博。
- 分母使用 total_weibo_count，而不是有显式主题类别的微博数。
- 如果 total_weibo_count 为 0，则结果设为 0.0。

5. entertainment_topic_ratio

表示该用户历史微博中娱乐文化相关微博占比。

“娱乐文化”可涵盖的类别：

- 娱乐文化
- 游戏动漫

计算方式：

entertainment_topic_ratio =
explicit_topic_categories 中命中上述娱乐类别的微博数 / total_weibo_count

6. daily_life_topic_ratio

表示该用户历史微博中日常生活相关微博占比。

计算方式：

daily_life_topic_ratio =
explicit_topic_categories 中包含“日常生活”的微博数 / total_weibo_count

7. repost_topic_dependency

表示该用户主题画像在多大程度上依赖转发源微博，而不是用户自写文本。

建议定义为：

repost_topic_dependency =
has_source_topic_count / (has_user_topic_count + has_source_topic_count)

其中：

- has_user_topic_count：当前用户 user_topics 非空的微博数。
- has_source_topic_count：当前用户 source_topics 非空的微博数。

如果分母为 0，则 repost_topic_dependency = 0.0。

可以额外生成 topic_source_balance_label：

- repost_topic_dependency >= 0.7：转发源主题依赖型
- repost_topic_dependency <= 0.3：自写主题主导型
- 其他：混合主题来源型

8. explicit_topic_coverage

表示该用户的微博中，有多少比例可以通过显式主题类别识别出粗粒度主题。

计算方式：

explicit_topic_coverage =
has_explicit_category_count / total_weibo_count

其中：

- has_explicit_category_count：explicit_topic_categories 非空的微博数。

如果 total_weibo_count 为 0，则结果设为 0.0。

9. avg_signal_confidence

输入表中存在 signal_confidence 字段，计算当前用户的平均 signal_confidence。

## 实现要求

请使用 Python 编写脚本，建议脚本路径为：

.\scripts\weibo_analysis\build_user_topic_profile.py

要求：

- 使用 pandas、pathlib、json、collections.Counter 等常用库。
- 代码需要包含 main() 函数。
- 所有比例字段保留为 float，建议四舍五入到 4 位小数

## 质量检查与日志输出

脚本运行时请打印以下信息：

- 输入微博级主题表行数
- 用户数量
- 输出用户级主题画像表行数
- 平均每个用户微博数
- public_issue_topic_ratio 的 describe() 统计
- entertainment_topic_ratio 的 describe() 统计
- daily_life_topic_ratio 的 describe() 统计
- repost_topic_dependency 的 describe() 统计
- explicit_topic_coverage 的 describe() 统计
- topic_source_balance_label 的 value_counts()
- top_user_topics 为空的用户数量
- top_source_topics 为空的用户数量

【代码结构建议】

建议实现以下函数：

- parse_categories(value) -> list[str]
- count_top_items(list_series, top_n=10) -> list[dict]
- build_user_topic_profile(df) -> pd.DataFrame
- save_outputs(profile_df, output_dir)
- main()

请根据以上要求完成脚本实现，并在代码末尾添加：

```python
if __name__ == "__main__":
    main()
```