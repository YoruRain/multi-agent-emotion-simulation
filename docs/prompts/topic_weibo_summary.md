# 概况

本阶段需要对 `df_topic_weibo` 中的文本内容（`content` 字段）进行摘要操作，具体策略为：

- 向 `df_topic_weibo` 添加4个新字段：`need_summary`, `summary_text`, `summary_status` 和 `analysis_context`
  - 查看每条记录的 `text_length` 字段，若值小于100，则置 `need_summary = False`；否则 `need_summary = True`
  - 对于 `text_length < 100` 的记录，`summary_text` 直接置为 `None` ；其余记录，待后续大模型 API 返回分析结果后填充进去
  - `text_length < 100` 的记录，`summary_status` 直接置为 `"skipped"`；其余记录，初始值先设为 `"pending"`。若大模型成功返回分析结果，则在存入 `summary_text` 后将 `summary_status` 置为 `"success"`；若为成功返回结果，则置为 `"failed"`
  - 对于 `text_length < 100` 的记录，`analysis_context` 直接置为 `content` 相同的值；其余记录，置为生成摘要后的 `summary_text` 字段
- 生成一份调用 DeepSeek 大语言模型 API 进行摘要生成的程序
- 由于需要摘要的数据量不大，所以不必使用异步运行等策略
- 最好能在程序运行过程中及时将分析好的部分保存，以免程序出错后返工
- 按需要，可以分成多个 Python 程序文件来实现
- 原数据集位置：`D:\GraduationProject\data\cleaned\topic_weibo.parquet`
- 保存数据：`D:\GraduationProject\data\cleaned\topic_weibo_summary.parquet`

- System Prompt：

  > 你是一名微博话题文本摘要助手。
  >
  > 你的任务是：根据输入的话题微博文本，生成一段简洁、客观、信息完整的摘要。
  >
  > 请严格遵循以下要求：
  >
  > 1. 保留事件主体、关键事实、主要行为、争议焦点和结果信息。
  > 2. 摘要必须体现该微博涉及的主要争议点或分歧（如：是否存在违规、是否属实、是否合理等）。
  > 3. 如果原文表达了明确立场、批评、支持、质疑、讽刺、愤怒等态度，可以适度保留其总体倾向。
  > 4. 不要照搬原文中的冗余修饰、感叹、重复表达、营销语或平台提示语。
  > 5. 不得编造原文没有的信息，不得加入常识性扩写或主观评价。
  > 6. 输出应简洁、通顺、便于后续模型快速理解该微博在讨论什么、争议点是什么。
  > 7. 摘要长度控制在 50～120 字之间；
  > 8. 只输出摘要正文，不要输出任何解释、标题、前缀或额外说明。


- User Prompt：

  > 请对下面的话题微博文本生成摘要。
  >
  > 要求：
  > - 用于后续评论的立场与情绪分析
  > - 关注“这条微博在说什么、争议点是什么、作者大致持什么态度”
  > - 只输出一段摘要正文
  >
  > 微博文本：
  > {{content}}