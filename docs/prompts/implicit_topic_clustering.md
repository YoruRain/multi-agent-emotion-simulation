你是一名熟悉 Python、pandas、中文文本向量表示、scikit-learn 聚类方法和数据工程规范的 Coding Agent。请帮我编写一个脚本，用于对候选微博文本生成 BGE embedding，并使用 MiniBatchKMeans 进行隐式主题聚类实验。

## 任务背景

我此前已经完成了微博级显式主题表，并筛选出一批需要进行隐式主题理解的微博。

候选微博数量约为 16,560 条。它们主要是“原创、可分析、但显式主题类别缺失”的用户表达文本。现在需要使用中文文本 embedding 模型生成语义向量，并使用 MiniBatchKMeans 尝试多个 k 值进行聚类，为后续“主题簇解释”和“微博级隐式主题表生成”做准备。

## 请生成的代码文件

请只创建一个脚本文件即可，路径为：

`D:/GraduationProject/scripts/weibo_analysis/implicit_topic_clustering.py`

该脚本需要完成以下步骤：

1. 读取候选微博数据集
2. 使用 BGE 模型生成 embedding
3. 保存 embedding 缓存文件
4. 对 embedding 执行 MiniBatchKMeans，尝试多个 k
5. 保存每个 k 的聚类标签和距离中心距离
6. 输出聚类规模统计与简单评估指标，供下一阶段选择最终 k

## 输入文件

候选微博数据集路径：

`D:/GraduationProject/data/interim/candidate_weibos_for_implicit_topic.parquet`

候选数据集包含字段：

- weibo_id
- user_id
- content
- text_quality
- is_repost
- explicit_topic_categories
- signal_confidence

其中 content 为需要生成 embedding 和聚类的微博文本。

## 运行环境

本阶段使用到的相关库已安装至环境中，请使用以下命令激活项目环境：

`conda activate D:\GraduationProject\.gp`

## 输出目录

请将所有输出保存到：

`D:/GraduationProject/data/profile/weibos/subject_profile/implicit_topic_clustering/`

## 输出文件要求

1. embedding 文件：

candidate_embeddings_bge_small_zh_v1_5.npy

2. embedding 元信息文件：

candidate_embeddings_meta.json

元信息至少包含：

- embedding_model
- candidate_count
- embedding_dim
- normalize_embeddings
- input_file
- embedding_file
- created_at

3. 不同 k 的聚类结果文件：

例如：

implicit_topic_clustering_k10.parquet
implicit_topic_clustering_k15.parquet
implicit_topic_clustering_k20.parquet
implicit_topic_clustering_k25.parquet
implicit_topic_clustering_k30.parquet

每个文件至少包含：

- weibo_id
- user_id
- analysis_text
- analysis_text_length
- cluster_id
- distance_to_center

其中：

- cluster_id 为 MiniBatchKMeans 分配的聚类簇编号
- distance_to_center 表示当前样本 embedding 到所属簇中心的欧氏距离
- 后续会使用 distance_to_center 选择每个簇的代表微博，因此必须准确计算

5. 聚类评估与规模统计文件：

cluster_eval_summary.csv

字段至少包含：

- k
- candidate_count
- cluster_count
- min_cluster_size
- max_cluster_size
- mean_cluster_size
- median_cluster_size
- inertia
- silhouette_score_sample
- random_state
- batch_size

其中：

- inertia 使用 MiniBatchKMeans 的 inertia_
- silhouette_score_sample 可以抽样计算，避免全量计算过慢
- 如果样本数不足或计算失败，则 silhouette_score_sample 可以设为 NaN，并打印 warning

## 模型要求

使用 BGE 中文 embedding 模型：

BAAI/bge-small-zh-v1.5

我已经在环境中安装了 FlagEmbedding：

`pip install -U FlagEmbedding`

请优先使用 FlagEmbedding 中的 BGEM3FlagModel 或 FlagModel 完成 embedding 生成。考虑到 bge-small-zh-v1.5 是非 M3 embedding 模型，优先尝试使用：

`from FlagEmbedding import FlagModel`

示例思路：

```python
model = FlagModel(
    "BAAI/bge-small-zh-v1.5",
    query_instruction_for_retrieval="",
    use_fp16=True
)

embeddings = model.encode(texts, batch_size=..., max_length=...)
```

如果本地环境中 FlagEmbedding API 存在差异，请在代码中尽量写清楚，并通过注释说明如何调整。

## embedding 生成要求

1. 使用 content 作为 analysis_text。
2. 生成 embedding 前需要做基础清洗：
   - 去除首尾空白字符
   - 将连续空白字符压缩为单个空格
   - analysis_text 长度小于 2 的记录建议删除，因为过短文本聚类意义较弱
3. 生成 embedding 后，建议做 L2 normalize。
   - 因为 MiniBatchKMeans 使用欧氏距离，而文本 embedding 通常更适合基于方向相似性比较。
   - 对 embedding 归一化后，欧氏距离与余弦相似度更接近。
4. embedding 结果必须保存为 .npy 文件，避免后续调参时重复生成。
5. 如果 embedding 文件已经存在，并且用户没有显式要求重新生成，应默认复用缓存。
6. 建议通过 argparse 提供参数：
   - --input
   - --output_dir
   - --model_name
   - --batch_size
   - --max_length
   - --force_rebuild_embeddings
   - --k_values
   - --random_state

【聚类要求】

使用 scikit-learn 的 MiniBatchKMeans。

默认尝试以下 k 值：

10, 15, 20, 25, 30

允许通过命令行参数覆盖，例如：

--k_values 10 15 20 25 30

MiniBatchKMeans 建议参数：

- n_clusters=k
- random_state=42
- batch_size=1024 或 2048
- n_init="auto" 如果当前 scikit-learn 版本支持
- 如果 n_init="auto" 报错，则兼容降级为 n_init=10

请确保代码兼容不同版本 scikit-learn。

## distance_to_center 计算要求

对每个样本：

1. 获取其 cluster_id
2. 获取对应 cluster center
3. 计算 embedding 与该 center 的欧氏距离

示例：

`distance = np.linalg.norm(embedding - cluster_centers[cluster_id])`

将 distance_to_center 保存到每个 k 的聚类结果文件中。

## silhouette_score_sample 计算要求

使用 `sklearn.metrics.silhouette_score`。

由于全量计算可能较慢，请抽样计算：

- 如果样本数 > 3000，则随机抽样 3000 条计算 silhouette_score
- 如果样本数 <= 3000，则全量计算
- metric 可以使用 "euclidean"，因为 embedding 已经 L2 normalize
- 如果某个 k 只有一个簇，或计算异常，则结果设为 NaN

## 日志与质量检查要求

脚本运行时请打印以下信息：

- 输入文件路径
- 原始候选微博行数
- 清洗后候选微博行数
- 删除的空文本 / 过短文本数量
- embedding 模型名称
- embedding shape
- embedding 是否使用缓存
- 每个 k 的聚类输出路径
- 每个 k 的聚类规模统计
- cluster_eval_summary.csv 的保存路径

## 代码结构建议

请将代码组织成清晰函数，至少包括：

- parse_args()
- ensure_output_dir(output_dir)
- clean_text(text)
- load_and_prepare_candidates(input_path)
- load_or_build_embeddings(texts, output_dir, model_name, batch_size, max_length, force_rebuild)
- normalize_embeddings(embeddings)
- run_minibatch_kmeans(embeddings, k, random_state, batch_size)
- compute_distance_to_center(embeddings, labels, centers)
- compute_silhouette_sample(embeddings, labels, random_state, sample_size=3000)
- save_clustering_result(df_candidates, labels, distances, k, output_dir)
- build_cluster_eval_row(...)
- main()

【依赖库】

请使用以下常用库：

- pandas
- numpy
- pathlib
- json
- argparse
- datetime
- tqdm
- sklearn.cluster.MiniBatchKMeans
- sklearn.metrics.silhouette_score
- sklearn.preprocessing.normalize
- FlagEmbedding.FlagModel

不要进行网络请求，除非本地第一次加载 Hugging Face 模型时环境自动下载模型；代码本身不要写主动网络请求逻辑。

## 运行示例

请在代码注释中给出运行示例：

python scripts/weibo_analysis/implicit_topic_clustering.py ^
  --input data/profile/weibos/subject_profile/candidate_weibos_for_implicit_topic.parquet ^
  --output_dir data/profile/weibos/subject_profile/implicit_topic_clustering ^
  --model_name BAAI/bge-small-zh-v1.5 ^
  --batch_size 64 ^
  --max_length 256 ^
  --k_values 10 15 20 25 30

如果使用 PowerShell，也可以给出单行示例。

## 实现重点

这一步不是最终主题解释，而是为下一阶段提供聚类实验结果。

因此，本脚本不需要为 cluster_id 命名，也不需要生成 implicit_topic_label 或 implicit_topic_category。

本脚本只负责：

- 候选文本清洗
- embedding 生成与缓存
- 多组 k 聚类
- 保存 cluster_id 和 distance_to_center
- 输出聚类规模统计

后续我会基于这些聚类结果，选择合适的 k，并为每个聚类簇生成主题解释。

请根据以上要求完成脚本实现，并在代码末尾添加：

```python
if __name__ == "__main__":
    main()
```