你是一名 Python 数据分析与工程实现助手。请根据现有的微博用户数据，编写一个脚本，用于构建用户级传播画像表 `df_user_propagation_profile`。

请重点保证：字段定义清晰、规则可解释、代码可复现、输出适合后续 Agent 建模使用。

## 一、输入数据

需要读取以下三个 DataFrame，实际路径请在脚本顶部通过常量或 argparse 参数配置：

- `df_user_weibo`: `.\data\cleaned\user_weibo.parquet`
- `df_user_info`: `.\data\high_quality\user_info.parquet`
- `df_source_creator`: `.\data\high_quality\source_creator_info.parquet`

### 1. 用户信息表 `df_user_info`

字段包括：

`['user_id', 'screen_name', 'gender', 'ip_location', 'registration_time', 'account_age_days', 'verified', 'verified_type', 'verified_type_name', 'total_weibo_count', 'follower_count', 'following_count', 'follower_following_ratio', 'user_rank', 'weibo_crawled_count', 'weibo_2025_count', 'weibo_hq_count', 'weibo_hq_ratio', 'original_ratio', 'comment_crawled_count', 'comment_hq_count', 'comment_hq_ratio', 'active_days', 'description', 'user_value',  'user_value_label']`

### 2. 用户微博表 `df_user_weibo`

字段包括：

`['weibo_id', 'user_id', 'screen_name', 'content', 'cleaned_content', 'text_length', 'cleaned_text_length', 'text_quality', 'text_quality_label', 'create_time', 'year', 'month', 'day', 'hour', 'weekday', 'like_count', 'comment_count', 'repost_count', 'engagement', 'is_repost', 'reposted_weibo_id', 'topics', 'at_users']`

注意：该表既包含建模用户自己发布的历史原创微博和转发微博，也包含转发微博对应的源微博。用户转发微博与源微博之间可以通过：

`df_user_weibo.reposted_weibo_id -> df_user_weibo.weibo_id`

建立连接。

### 3. 转发源微博作者信息表 `df_source_creator`

字段包括：

`['user_id', 'screen_name', 'gender', 'ip_location', 'registration_time', 'total_weibo_count', 'follower_count', 'following_count', 'description', 'verified', 'verified_type', 'verified_type_name', 'user_rank']`

认证类型 `verified_type_name` 的所有可能取值为：

`['个人认证', '政府', '企业', '媒体', '校园', '网站', '应用', '团体/机构', '普通用户']`

## 二、输出结果

最终输出用户级传播画像表：

`df_user_propagation_profile`

一行对应一个 `df_user_info` 中的用户。

最终字段严格控制为以下最小字段清单，不要自行添加大量额外字段：

`user_id`  
`weibo_hq_count`  
`active_days`  
`propagation_activity_level`  
`original_ratio`  
`repost_ratio`  
`repost_with_comment_ratio`  
`source_media_ratio`  
`source_government_ratio`  
`source_institution_ratio`  
`source_personal_verified_ratio`  
`source_high_follower_ratio`  
`media_dependency_score`  
`kol_sensitivity_score`  
`avg_engagement`  
`high_engagement_weibo_ratio`  
`influence_score`  
`influence_level`  
`propagation_role`  

请将结果保存为 Parquet 文件，路径为：

`./data/profile/weibos/propagation_profile/user_propagation_profile.parquet`

编写的代码保存至：

`./scripts/weibo_analysis/propagation_profile/user_propagation_profile.py`

## 三、核心处理逻辑

### 1. 识别建模用户

建模用户集合来自 `df_user_info['user_id']`。

从 `df_user_weibo` 中筛选出这些用户自己发布的微博，得到 `user_weibo`：

`user_weibo = df_user_weibo[df_user_weibo['user_id'].isin(df_user_info['user_id'])]`

### 2. 计算原创 / 转发倾向

基于 `user_weibo`，按用户统计：

- 原创微博数量：`is_repost == False`
- 转发微博数量：`is_repost == True`
- 有效转发微博数量：`is_repost == True` 且 `reposted_weibo_id` 非空、非 `-1`
- `original_ratio`
  - 优先使用 `df_user_info` 中已有的 `original_ratio`
  - 若缺失，则根据 `user_weibo` 重新计算
- `repost_ratio = 1 - original_ratio`

### 3. 判断“带有效评语转发”

前期已经做过文本质量分级：

- `text_quality == 0`：空文本
- `text_quality == 1`：抽奖打卡、占位文本、纯话题、收藏指令等
- `text_quality == 2`：低信息量文本
- `text_quality == 3`：可分析文本

对一般的转发微博，使用 `text_quality == 3` 来判定微博“带有有效评语”即可。

对一部分转发微博，内容中可能存在转发链，特征是文本会带有 `//`，例如：

`丑剧盛行！//@依依岁华晚:回复@高贵骆驼追剧:...`

这类转发微博的文本质量等级均为3，但并不能保证用户留下了有效评语。

可以通过：

`text.split("//")`

拆分转发链。拆分后第一个元素表示当前建模用户在转发时留下的评论。如果用户未留下评论，文本通常形如：

`//@咦-果果:看看人家的乒协...`

此时 `split("//")[0]` 为空串。

此外，拆分转发链后第一个元素为一些占位符、功能互动等低信息量的文本也应认为用户“未留下有效评语”，例如：

`转发微博`, `转发`, `1`, 纯@用户等。

这里提供一些正则规则帮助您确认用户所留评论的有效性：
```python
(
    r'^\s*('
    r'[转轉][发發]?(至?微博)?|'
    r'Repost|'
    r'[存码马](住|下|克)?|'
    r'转一个|必须转|'
    r'签到|'
    r'收藏(了)?'
    r')\s*$', '占位/功能互动'
),
(r'^#[^#]+#(\s*#[^#]+#)*$', '纯话题占位'),
(r'\d+', '纯数字'), 
(r'[^\w\u4e00-\u9fff\U00010000-\U0010FFFF]+', '纯符号'), 
(r'[a-zA-Z]+', '纯英文字母'),
(r'(@[^\s@]+\s*)+', '纯@用户')
```

请实现一个函数，例如：

`extract_repost_comment(text: str) -> str`

逻辑如下：

- 若输入为空，返回空串
- 使用每条微博的 `content` 字段
- 使用 `str(text).split("//", 1)[0].strip()` 获取当前用户的转发评语
- 如果结果为空，或结果被上述的低信息量文本模式匹配，说明用户未留下有效评语


计算：

`repost_with_comment_ratio = repost_with_comment_count / valid_repost_count`

若 `valid_repost_count == 0`，则该比例设为 0。

### 4. 构建转发关系表

从 `user_weibo` 中筛选有效转发微博，得到 `user_repost_weibo`：

条件：

`is_repost == True`  
`reposted_weibo_id` 非空  
`reposted_weibo_id != -1`

保留字段至少包括：

`user_id`  
`weibo_id`  
`reposted_weibo_id`  
`cleaned_content`  
`content`  
`text_quality`  
`engagement`

然后用：

`user_repost_weibo.reposted_weibo_id -> df_user_weibo.weibo_id`

连接源微博，得到源微博作者 ID。注意连接时要避免字段名冲突，可以使用 suffix，例如 `_repost` 和 `_source`。

再用源微博作者 ID 连接 `df_source_creator.user_id`，得到源作者认证类型、粉丝数、user_rank 等信息。

### 5. 归并转发源账号类型

根据 `df_source_creator['verified_type_name']` 归并源作者类型：

- `媒体`：`verified_type_name == '媒体'`
- `政府`：`verified_type_name == '政府'`
- `机构`：`verified_type_name in ['企业', '校园', '网站', '应用', '团体/机构']`
- `个人认证`：`verified_type_name == '个人认证'`
- `普通用户`：`verified_type_name == '普通用户'`

按用户计算以下比例：

`source_media_ratio`：转发源作者中媒体账号占比  
`source_government_ratio`：转发源作者中政府账号占比  
`source_institution_ratio`：转发源作者中机构类账号占比  
`source_personal_verified_ratio`：转发源作者中个人认证账号占比

比例的分母为该用户所有成功连接到源作者信息的有效转发数。若某用户没有可连接的源作者信息，则上述比例全部设为 0。

### 6. 计算高粉源账号占比

基于 `df_source_creator['follower_count']` 计算高粉账号阈值。

建议使用数据内部分位数：

`high_follower_threshold = df_source_creator['follower_count'].quantile(0.75)`

在连接后的转发关系表中，如果源作者的 `follower_count >= high_follower_threshold`，则认为该转发源为高粉源账号。

按用户计算：

`source_high_follower_ratio = 高粉源账号转发数 / 成功连接源作者信息的有效转发数`

若分母为 0，则设为 0。

### 7. 计算媒体依赖度与 KOL 敏感度

构建：

`media_dependency_score`

建议公式：

`media_dependency_score = source_media_ratio + 0.7 * source_government_ratio + 0.5 * source_institution_ratio`

然后将结果裁剪到 `[0, 1]` 区间。

构建：

`kol_sensitivity_score`

建议公式：

`kol_sensitivity_score = 0.5 * source_personal_verified_ratio + 0.5 * source_high_follower_ratio`

然后将结果裁剪到 `[0, 1]` 区间。

注意：

- 媒体依赖度主要表示用户是否依赖媒体、政府、机构等组织化信息源。
- KOL 敏感度主要表示用户是否容易转发个人认证账号或高粉账号内容。
- 二者不要混为一谈。

### 8. 计算互动与自身影响力

基于建模用户自己发布的微博 `user_weibo` 计算：

`avg_engagement`：用户微博平均互动量  
`high_engagement_weibo_ratio`：高互动微博占比

高互动微博阈值建议使用 `user_weibo['engagement']` 的 75% 分位数：

`high_engagement_threshold = user_weibo['engagement'].quantile(0.75)`

用户某条微博的 `engagement >= high_engagement_threshold`，则视为高互动微博。

按用户计算：

`high_engagement_weibo_ratio = 高互动微博数 / 用户微博数`

若用户无微博，则设为 0。

构建 `influence_score`。为了避免粉丝数和互动量长尾分布影响，请先进行 `log1p` 处理，并做分位数归一化或 min-max 归一化。

建议实现几个辅助函数：

- `safe_log1p`
- `minmax_normalize`
- `quantile_level`

影响力可以由以下指标组成：

- 粉丝数得分：由 `df_user_info['follower_count']` 经过 `log1p` 和归一化得到
- 互动得分：由 `avg_engagement` 经过 `log1p` 和归一化得到
- 认证得分：`verified == True` 为 1，否则为 0
- user_rank 得分：如果 `user_rank` 可用，则归一化；若缺失，则设为 0

建议公式：

`influence_score = 0.4 * follower_score + 0.3 * engagement_score + 0.2 * verified_score + 0.1 * user_rank_score`

结果裁剪到 `[0, 1]`。

然后根据分位数将 `influence_score` 划分为：

- `low`
- `medium`
- `high`

可以使用 33% 和 67% 分位数作为阈值。

### 9. 传播活跃度等级

基于：

- `weibo_hq_count`
- `active_days`

构建 `propagation_activity_level`。

建议使用分位数规则：

计算 `weibo_hq_count` 和 `active_days` 在 `df_user_info` 中的 33% 和 67% 分位数。

规则：

- 如果 `weibo_hq_count` 和 `active_days` 均较低，则为 `low`
- 如果二者至少一个较高，或整体处于中间水平，则为 `medium`
- 如果二者均较高，则为 `high`

也可以实现为一个综合分：

`activity_score = 0.5 * normalized_weibo_hq_count + 0.5 * normalized_active_days`

再按 33% 和 67% 分位数划分为 `low / medium / high`。

### 10. 传播角色 `propagation_role`

请根据前面得到的用户级指标生成传播角色标签。该字段可以是字符串，多个标签之间用英文逗号 `,` 分隔。

候选标签包括：

- `原创表达者`
- `转发扩散者`
- `转发评论者`
- `媒体信息跟随者`
- `KOL 敏感型用户`
- `普通参与者`
- `潜在影响者`
- `低活跃观察者`

建议规则：

- 若 `propagation_activity_level == 'low'`，加入 `低活跃观察者`
- 若 `original_ratio >= 0.6`，加入 `原创表达者`
- 若 `repost_ratio >= 0.6`，加入 `转发扩散者`
- 若 `repost_with_comment_ratio >= 0.4`，加入 `转发评论者`
- 若 `media_dependency_score >= 0.5`，加入 `媒体信息跟随者`
- 若 `kol_sensitivity_score >= 0.5`，加入 `KOL 敏感型用户`
- 若 `influence_level == 'high'`，加入 `潜在影响者`
- 如果没有命中任何标签，则设为 `普通参与者`

注意：允许多标签，因为真实用户可能同时是“转发扩散者”和“媒体信息跟随者”。



## 四、输出字段口径

最终 `df_user_propagation_profile` 必须只包含以下字段，并按此顺序排列：

`user_id`  
`weibo_hq_count`  
`active_days`  
`propagation_activity_level`  
`original_ratio`  
`repost_ratio`  
`repost_with_comment_ratio`  
`source_media_ratio`  
`source_government_ratio`  
`source_institution_ratio`  
`source_personal_verified_ratio`  
`source_high_follower_ratio`  
`media_dependency_score`  
`kol_sensitivity_score`  
`avg_engagement`  
`high_engagement_weibo_ratio`  
`influence_score`  
`influence_level`  
`propagation_role`  

## 五、代码要求

1. 使用 Python + pandas 实现。

2. 建议使用 argparse，让输入输出路径可配置，例如：

- `--user-info-path`
- `--user-weibo-path`
- `--source-creator-path`
- `--output-path`

3. 需要包含必要的辅助函数，例如：

- `load_dataframe(path)`
- `save_dataframe(df, path)`
- `extract_repost_comment(text)`
- `normalize_verified_type(verified_type_name)`
- `minmax_normalize(series)`
- `assign_level_by_quantile(series)`
- `build_propagation_roles(row)`

4. 注意处理缺失值：

- 数值字段缺失时尽量填 0
- 比例字段缺失时填 0
- 文本字段缺失时填 `None`
- 分母为 0 时比例设为 0

5. 注意不要误把转发源微博当作建模用户自己的微博。

建模用户必须来自 `df_user_info['user_id']`。

6. 代码中添加适量注释，说明每个指标的含义。

7. 脚本运行结束后，打印以下检查信息：

- `df_user_info` 用户数
- 输出画像用户数
- 有效转发记录数
- 成功连接源微博的转发记录数
- 成功连接源作者信息的转发记录数
- `propagation_activity_level` 分布
- `influence_level` 分布
- `propagation_role` 高频统计前 10 项
- 输出文件路径

## 六、质量检查要求

请在脚本最后加入简单的数据质量检查：

1. 输出用户数应与 `df_user_info` 用户数一致。

2. 以下比例字段应位于 `[0, 1]`：

`original_ratio`  
`repost_ratio`  
`repost_with_comment_ratio`  
`source_media_ratio`  
`source_government_ratio`  
`source_institution_ratio`  
`source_personal_verified_ratio`  
`source_high_follower_ratio`  
`media_dependency_score`  
`kol_sensitivity_score`  
`high_engagement_weibo_ratio`  
`influence_score`

3. `propagation_activity_level` 和 `influence_level` 不应为空。

4. `propagation_role` 不应为空。