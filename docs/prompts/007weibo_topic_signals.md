你是一名熟悉 Python、pandas、中文社交媒体数据处理与数据工程规范的 Coding Agent。请帮我编写一段脚本，用于根据核心用户历史微博数据生成“微博级主题显式信号表”。

## 任务背景

我当前需要为核心用户构建主题画像。直接利用已有显式信号，包括微博文本中的话题标签、转发源微博的话题标签、粗粒度主题关键词等，生成一张微博级主题表。

## 输入数据

1. 核心用户微博数据集：
   `D:\GraduationProject\data\high_quality\user_weibo.parquet`

   该数据集仅包含核心用户发布的微博，包括原创微博和转发微博。

2. 用户微博总数据集：
   `D:\GraduationProject\data\cleaned\user_weibo.parquet`

   该数据集包含所有用户发布的微博，以及转发微博对应的原微博记录。后续需要通过核心用户微博中的 `reposted_weibo_id`，在总数据集中查找原微博的信息。

两个数据集的表结构完全一致，至少包含以下字段：

- `weibo_id`
- `user_id`
- `content`
- `topics`
- `is_repost`
- `reposted_weibo_id`
- `text_quality`

其中：

- `topics` 字段记录微博文本中的话题标签。如果存在两个及以上话题标签，则使用英文逗号 `,` 分隔。
- `is_repost` 为布尔类型，表示是否为转发微博。
- `reposted_weibo_id` 表示转发微博对应的原微博 ID；原创微博该字段值为 `-1`。
- `text_quality` 表示文本质量等级，数值越高说明越可分析。

## 输出目标

请针对核心用户微博数据集中的每一条微博，生成微博级主题显式信号表。

输出表应包含以下字段：

- `weibo_id`
- `user_id`
- `content`
- `is_repost`
- `reposted_weibo_id`
- `has_repost_comment`
- `user_topics`
- `source_topics`
- `explicit_keywords`
- `explicit_topic_categories`
- `signal_confidence`

建议输出路径：

`D:\GraduationProject\data\profile\weibos\subject_profile\user_weibo_topic_signals.parquet`

## 字段生成规则

1. `weibo_id`、`user_id`、`content`、`is_repost`、`reposted_weibo_id`

这四个字段直接从核心用户微博表迁移。

2. `has_repost_comment`

该字段表示“转发微博是否包含用户自己写的转发评论”。

注意：该字段仅在 `is_repost == True` 时具有实际语义。对于原创微博，请统一设为 `False`，避免引入空值导致后续统计复杂化。

具体规则：

- 如果 `is_repost == False`：
  - `has_repost_comment = False`
- 如果 `is_repost == True`：
  - 当 `text_quality < 3` 时，认为没有有效转发评论，`has_repost_comment = False`
  - 或者当 `content` 字段以 `//` 开头时，认为没有有效转发评论，`has_repost_comment = False`
  - 其他情况设为 `True`

3. `user_topics`：来自核心用户当前微博记录的 `topics` 字段。

4. `source_topics`：表示转发源微博的话题标签。

   - 生成规则：

     - 如果 `is_repost == False`：`source_topics = None`
       

     - 如果 `is_repost == True`：
       - 使用当前记录的 `reposted_weibo_id`，在用户微博总数据集 `df_all` 中按 `weibo_id` 查找对应原微博
       - 找到后读取原微博的 `topics` 字段并迁移


5. `explicit_keywords`：表示通过粗粒度主题词典在微博文本中命中的显式关键词

   - 请设计一个粗粒度主题词典，至少包含以下类别和关键词。可以在实现时适度补充，但不要过度复杂化：

     - public_event：公共事件 / 社会新闻
       - 警方、通报、事故、地震、火灾、爆炸、车祸、救援、调查、回应、热搜、维权、判决、法院、检察院、公安、消防


     - policy_livelihood：政策民生
       - 政策、医保、社保、教育、就业、工资、房价、租房、学校、医院、高考、考研、公务员、养老金、户口、补贴


     - entertainment_culture：娱乐文化
       - 明星、演员、歌手、演唱会、电影、电视剧、综艺、粉丝、塌房、票房、官宣、剧组、偶像


     - daily_life：日常生活
       - 上班、下班、睡觉、吃饭、天气、早安、晚安、回家、出门、旅游、学习、考试、宿舍、加班、周末


     - game_anime：游戏动漫
       - 游戏、抽卡、皮肤、角色、原神、王者荣耀、崩坏、明日方舟、二次元、动漫、漫画、番剧、cos


     - emotion_expression：情绪表达
       - 无语、崩溃、破防、开心、难过、生气、烦死、震惊、心疼、恶心、离谱、好笑、哭了、累了、焦虑


     - marketing_low_value：营销 / 抽奖 / 低价值传播
       - 抽奖、红包、优惠券、下单、直播间、福利、转发抽奖、中奖、带货、秒杀、返现、链接、推广


     - media_official：媒体 / 官方信息源表达
       - 记者、报道、据悉、来源、官方、声明、发布、人民日报、新华社、央视、澎湃、观察者网、环球时报


- 关键词匹配规则：

  - 使用简单的 substring 匹配即可，不必进行复杂分词。

  - 对每条微博返回命中的关键词列表。

  - 同一个关键词只保留一次。

  - 保留命中顺序，便于人工检查。

  - 匹配文本需要包含以下部分：
    - 原创微博：使用 `content`
    - 转发微博且 `has_repost_comment == True`：优先使用用户自己的 `content`，同时可以将源微博 `content` 作为辅助文本
    - 转发微博且 `has_repost_comment == False`：主要使用源微博 `content` 作为显式信号来源


6. `explicit_topic_categories`：表示通过显式关键词和话题标签推断出的粗粒度主题类别。

- 生成规则：

  - 如果某个类别下的关键词在文本中命中，则加入该类别

  - 同一条微博可以属于多个类别

  - 类别名使用英文，如：
    - public_event
    - policy_livelihood
    - entertainment_culture
    - daily_life
    - game_anime
    - emotion_expression
    - marketing_low_value
    - media_official


- 如果没有任何关键词或话题标签命中，则 `explicit_topic_categories = None`

请把主题词典设计为一个清晰的 Python dict，例如：

```python
TOPIC_KEYWORDS = {
    "public_event": [...],
    "policy_livelihood": [...],
    ...
}
```

7. `signal_confidence`表示当前微博通过显式信号识别主题的置信度，取值范围为 0 到 1 的 float。

- 请使用规则型评分，不需要训练模型。

- 建议规则如下，可根据实现情况微调，但需要在代码注释中说明：

  - 基础分为 0.0。

  - 加分项：

    - `user_topics` 非空：+0.35

    - `source_topics` 非空：+0.20

    - `explicit_keywords` 非空：+0.30

    - `explicit_topic_categories` 非空：+0.15

    - 如果是转发微博且 `has_repost_comment == True`：+0.10


降分项：

- 如果是转发微博且 `has_repost_comment == False`：-0.15
- 如果 `content` 为空或 `text_quality < 3`：-0.10

最后将分数截断到 `[0.0, 1.0]` 区间。

## 数据存储要求

- 输出主表保存为 Parquet。
- user_topics、source_topics、explicit_keywords、explicit_topic_categories 均以字符串形式保存，若存在多值，则用逗号 `,`分隔

## 质量检查与日志输出

脚本运行时请打印以下信息：

- 核心用户微博表行数
- 用户微博总表行数
- 输出表行数
- 原创微博数量
- 转发微博数量
- 有效转发评论数量
- 无有效转发评论数量
- 转发源微博匹配成功数量
- 转发源微博匹配失败数量
- `user_topics` 非空比例
- `source_topics` 非空比例
- `explicit_keywords` 非空比例
- `explicit_topic_categories` 非空比例
- `signal_confidence` 的 `describe()` 统计结果

请进行基本断言：

- 输出表行数必须等于核心用户微博表行数
- 输出表必须包含所有目标字段
- `signal_confidence` 必须都在 `[0, 1]` 区间内

## 代码风格要求

- 使用 Python 编写，优先使用 pandas、pathlib、json、re 等常用库。
- 代码应结构清晰，包含 main() 函数。
- 建议将脚本保存到：
  D:\GraduationProject\scripts\weibo_analysis\build_user_weibo_topic_signals.py

请根据上述要求完成脚本实现，并在代码末尾给出运行入口：

```python
if __name__ == "__main__":
    main()
```