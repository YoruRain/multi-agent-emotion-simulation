请在当前项目中新增一个 Python 脚本，用于将得到的微博级情感分析结果按用户进行聚合，生成用户级情绪画像。
请基于微博级输出（路径为 `D:\GraduationProject\data\analysis\weibos\emotion_profile\user_weibo_emotion_analysis.parquet`），按 `user_id` 聚合生成用户级画像表。

需要输出以下字段：

- `user_id`
- `profile_version`
- `analyzable_weibo_count`
- `weighted_weibo_count`
- `pos_ratio`
- `neu_ratio`
- `neg_ratio`
- `avg_polarity_score`
- `median_polarity_score`
- `polarity_std`
- `avg_intensity_score`
- `strong_emotion_ratio`
- `strong_positive_ratio`
- `strong_negative_ratio`
- `dominant_emotion`
- `emotion_profile_type`
- `profile_reliability`
- `emotion_profile_summary`

时间口径：
由于本项目话题微博和评论主要限制在 2025 年，而历史微博中包含 2026 年内容，请支持两种画像：

1. `profile_version = "2025"`：只使用 `year == 2025` 的微博
2. `profile_version = "all"`：使用全部微博

请默认同时生成这两个版本的用户画像，并合并输出到同一个 `df_user_emotion_profile` 中。

`analyzable_weibo_count` 指的是当前 version（"2025" 或 "all"）下微博级情感分析结果数据集中该用户的微博记录数。

`weighted_weibo_count` 可以简单定义为当前 version 下微博级情感分析结果数据集中该用户微博记录的 `text_weight` 字段之和。

聚合要求：

1. `pos_ratio`、`neu_ratio`、`neg_ratio` 使用 `text_weight` 加权统计。
2. `avg_polarity_score` 使用 `text_weight` 加权平均。
3. `median_polarity_score` 使用普通中位数即可。
4. `polarity_std` 使用普通标准差。
5. `avg_intensity_score` 使用加权平均。
6. `strong_emotion_ratio` 表示强情绪微博占比。
   - 默认阈值可以设为 `emotion_intensity_score >= 0.7`
   - 请提供参数 `--strong_threshold`，默认 0.7
7. `strong_positive_ratio`：
   - `emotion_intensity_score >= strong_threshold` 且 `sentiment_label_en == "Positive"`
8. `strong_negative_ratio`：
   - `emotion_intensity_score >= strong_threshold` 且 `sentiment_label_en == "Negative"`



`emotion_profile_type` 请根据用户级指标生成粗粒度类型。可以先使用简单规则：

- 如果 `analyzable_weibo_count < 5`：`样本不足型`
- 如果 `neu_ratio >= 0.6` 且 `avg_intensity_score < 0.4`：`稳定中性型`
- 如果 `neg_ratio >= 0.4` 且 `strong_negative_ratio >= 0.2`：`强消极表达型`
- 如果 `pos_ratio >= 0.4` 且 `strong_positive_ratio >= 0.2`：`强积极表达型`
- 如果 `polarity_std >= 0.5`：`高波动型`
- 如果 `avg_polarity_score > 0.15`：`轻度积极型`
- 如果 `avg_polarity_score < -0.15`：`轻度消极型`
- 否则：`混合表达型`

`profile_reliability` 请根据可分析微博数量生成：

- `analyzable_weibo_count >= 20`：`high`
- `10 <= analyzable_weibo_count < 20`：`medium`
- `5 <= analyzable_weibo_count < 10`：`low`
- `< 5`：`insufficient`

`emotion_profile_summary` 由以下部分构成：

```
整体情绪倾向 + 情绪强度 + 情绪波动 (+ 强积极/强消极倾向补充) + 可靠性提示
```

具体规则为：

1. 整体情绪倾向：

   ```python
   if analyzable_weibo_count < 5:
       tendency_phrase = "该用户历史原创微博样本较少，暂难稳定判断整体情绪倾向"
   elif neu_ratio >= 0.6 and abs(avg_polarity_score) < 0.15:
       tendency_phrase = "该用户历史原创微博整体偏中性"
   elif avg_polarity_score >= 0.25 or pos_ratio >= 0.45:
       tendency_phrase = "该用户历史原创微博整体偏积极"
   elif avg_polarity_score <= -0.25 or neg_ratio >= 0.45:
       tendency_phrase = "该用户历史原创微博整体偏消极"
   elif avg_polarity_score >= 0.10:
       tendency_phrase = "该用户历史原创微博略偏积极"
   elif avg_polarity_score <= -0.10:
       tendency_phrase = "该用户历史原创微博略偏消极"
   else:
       tendency_phrase = "该用户历史原创微博积极、消极与中性表达较为混合"
   ```

2. 情绪强度：

   ```python
   if avg_intensity_score >= 0.65 or strong_emotion_ratio >= 0.35:
       intensity_phrase = "情绪强度较高"
   elif avg_intensity_score >= 0.45 or strong_emotion_ratio >= 0.20:
       intensity_phrase = "情绪强度中等"
   else:
       intensity_phrase = "情绪强度较低"
   ```

3. 情绪波动：

   ```python
   if polarity_std >= 0.65:
       volatility_phrase = "表达波动较大"
   elif polarity_std >= 0.40:
       volatility_phrase = "存在一定情绪波动"
   else:
       volatility_phrase = "表达相对稳定"
   ```

4. （仅在特征明显时补充）：判断“强积极 / 强消极倾向补充”

   ```python
   extra_phrase = ""
   
   if strong_negative_ratio >= 0.25 and neg_ratio >= 0.35:
       extra_phrase = "较容易出现强消极表达"
   elif strong_positive_ratio >= 0.25 and pos_ratio >= 0.35:
       extra_phrase = "较容易出现强积极表达"
   elif pos_ratio >= 0.35 and neg_ratio >= 0.35:
       extra_phrase = "积极与消极表达均有一定比例"
   ```

5. 可靠性提示：

   ```python
   if profile_reliability == "insufficient":
       reliability_phrase = "样本不足，画像仅供参考"
   elif profile_reliability == "low":
       reliability_phrase = "可分析样本较少，画像可靠性有限"
   elif profile_reliability == "medium":
       reliability_phrase = "画像可靠性中等"
   else:
       reliability_phrase = ""

最后把这些短语拼接起来。

输出文件：
请支持 Parquet 输出。保存路径为：`D:\GraduationProject\data\analysis\weibos\emotion_profile\user_emotion_profile.parquet`

代码质量要求：

1. 使用清晰的函数拆分，不要写成一个巨大的 main 函数。
5. 保证脚本可以被命令行直接运行。

请完成脚本的编写，并在 README.md 中添加说明。

