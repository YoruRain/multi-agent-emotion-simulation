请在 `D:\GraduationProject\scripts\weibo_analysis\infer_user_weibo_emotion` 目录下新增若干个 Python 脚本，用于将已经微调保存好的 MacBERT 情绪分类模型迁移到“用户历史微博长期情绪画像”任务中。

目前，请先完成使用 MacBERT 完成情感分析的部分。

背景说明：
我已经将模型保存到了项目目录下：

`./models/macbert_finetuned_sentiment`

> 目前暂时没有保存，请先按照下面的信息编写后续代码

该目录中包含：
- tokenizer 文件：通过 `tokenizer.save_pretrained(save_dir)` 保存
- BERT 底座配置与权重：通过 `bert.bert.save_pretrained(save_dir)` 保存
- 完整模型参数：`model_state.pt`
- 任务配置：`task_config.json`
- 训练与测试用 Notebook：`07Pre-trained Language Model.ipynb`，可以从中读取原模型结构


请注意：原模型结构是一个自定义 PyTorch 模型，大致为：

* 底座：MacBERT / BertModel
* dropout
* linear 分类头
* 输出 3 类情绪分类 logits

请你先根据已有项目文件和 `model_state.pt` 的 key，判断分类头字段名称应如何定义。

如果项目中已有之前训练 notebook 或模型类定义，请尽量复用原定义；如果没有，则在脚本中定义一个兼容的 `MacBertSentimentClassifier` 类，使其能够正确加载 `model_state.pt`。

目标功能：
这个脚本需要输出：

微博级情绪推理结果表

请不要重新训练模型，只做推理与聚合。

输入数据：
脚本应支持读取 Parquet 文件，默认输入设为：

`D:\GraduationProject\data\high_quality\user_weibo.parquet`


输入 DataFrame 包含以下字段，请尽量兼容：

* `weibo_id`
* `user_id`
* `content`
* `text_length`
* `text`
* `text_quality`
* `text_quality_label`
* `create_time`
* `year`
* `is_repost`
* `text_quality`


数据过滤要求：
默认只对可分析微博做推理，默认保留 `text_quality >= 3` 的记录。
请使用 argparse 提供参数：

* `--min_text_quality`，默认值为 3
* `--no_quality_filter`，允许用户关闭质量过滤

模型推理要求：

1. 加载 tokenizer：

   * 使用 `BertTokenizer.from_pretrained(model_dir)`

2. 加载 BERT 底座：

   * 使用 `BertModel.from_pretrained(model_dir)`

3. 加载完整模型权重：

   * 从 `model_state.pt` 读取 `state_dict`
   * 正确加载到自定义分类模型中

4. 推理时：

   * 使用 `torch.no_grad()`
   * 自动选择 `cuda` 或 `cpu`
   * 使用 batch 推理，默认 `batch_size=64`
   * 默认 `max_len` 从 `task_config.json` 读取
   * 但允许通过参数 `max_len` 覆盖
   * 对长微博进行 tokenizer 截断
   * 输出 softmax 概率

输出：以一个微博级情绪分析表形式，包含以下字段：

* `weibo_id`
* `user_id`
* `content`
* `year`
* `is_repost`
* `sentiment_label_en`：`Negative` / `Neutral` / `Positive`
* `sentiment_label`：`消极` / `中性` / `积极`
* `p_neg`
* `p_neu`
* `p_pos`
* `model_confidence`：三类概率最大值
* `polarity_score`：`p_pos - p_neg`
* `model_intensity`：`1 - p_neu`
* `rule_intensity`
* `emotion_intensity_score`
* `text_weight`

其中：

`polarity_score = p_pos - p_neg`

`model_intensity = 1 - p_neu`

`rule_intensity` 请通过规则计算，范围控制在 0 到 1。规则可以包括：

* 感叹号、问号数量，尤其是连续 `！！`、`？？`
* 重复字符，如 “啊啊啊”、“哈哈哈”、“呜呜呜”
* 强语气词，如 “真的”、“太”、“特别”、“极其”、“完全”、“超级”、“绝了”、“离谱”、“爆炸”
* 情绪词，如 “开心”、“难受”、“崩溃”、“恶心”、“震惊”、“无语”、“感动”、“生气”、“失望”、“破防”、“笑死”、“绷不住”
* emoji 或常见网络情绪表达

可以使用简单可解释的打分方式，例如命中若干规则后累加，再 clip 到 0 到 1。

`emotion_intensity_score` 按如下方式计算：

```python
emotion_intensity_score = 0.6 * model_intensity + 0.4 * rule_intensity
```

`text_weight` 用于用户级聚合。请按以下规则设计：

1. 文本长度权重：

   * 文本长度 < 6：`length_weight = 0.3`
   * 6 <= 文本长度 < 15：`length_weight = 0.6`
   * 文本长度 >= 15：`length_weight = 1.0`

2. 原创 / 转发权重：

   * 如果为原创微博（`is_repost = False`）：`source_weight = 1.0`
   * 如果为转发微博（`is_repost = True`）：`source_weight = 0.7`

3. 最终：

   * `text_weight = length_weight * source_weight`

输出文件：
请支持 JSONL 和 Parquet 输出，命名为 `user_weibo_emotion_profile`，目录为：
`D:\GraduationProject\data\analysis\weibos\emotion_profile`

根据输出路径后缀自动判断：

* `.jsonl`：保存 JSONL
* `.parquet`：保存 Parquet

代码质量要求：

1. 使用清晰的函数拆分与文件拆分，不要写成一个巨大的 main 函数。
2. 至少包含以下函数：

   * `load_task_config`
   * `load_model_and_tokenizer`
   * `load_input_dataframe`
   * `prepare_texts`
   * `predict_sentiment`
   * `compute_rule_intensity`
   * `compute_text_weight`
   * `build_weibo_emotion_table`
   * `save_dataframe`
   * `main`
3. 加入必要的异常处理和日志输出。
4. 使用 `tqdm` 显示推理进度。
5. 保证脚本可以被命令行直接运行。
6. 不要重新训练模型。
7. 推理结果应尽量可复现。

请完成相关代码的编写，并提供一份介绍运行方式、相关参数含义的 Markdown 文档，在文档末尾提供一段示例运行命令。
