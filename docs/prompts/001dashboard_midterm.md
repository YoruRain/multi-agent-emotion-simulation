你是一名经验丰富的 Python 数据可视化与应用开发工程师。请你为我编写一个可运行的可视化原型，技术栈限定为：

- Plotly
- NetworkX
- PyVis
- Streamlit

项目主题与背景：
我的毕业设计是“基于多智能体的社会群体情绪模拟系统的设计与实现”。当前阶段，我已经获得了一批微博相关数据，现在希望通过一个 Python 可视化原型来展示数据基础与建模潜力。这个原型不需要追求复杂后端或生产级部署，但必须结构清晰、界面整洁、可运行、适合现场演示。

我的核心展示目标：
1. 展示一个热点事件下的典型微博主帖信息。
2. 展示该微博下若干条评论及其回复关系，也就是评论树/评论网络结构。
3. 展示评论对应用户的部分属性信息。
4. 通过图表说明该热点事件、评论互动、用户分布等信息。
5. 整体效果要适合演示，强调“数据已经具备支持后续 Agent 建模的基础”。

我当前已有 4 张核心数据表，字段信息如下：

1. df_topic_weibo（话题微博数据）
- weibo_id: 微博唯一标识符
- user_id: 发布微博的用户ID
- screen_name: 微博用户昵称
- gender: 用户性别
- topic: 微博文本提及的话题
- content: 微博文本内容
- text_length: 微博文本长度
- create_time: 微博发布时间
- year, month, day, hour: 时间分解字段
- weekday: 发布时的星期几
- like_count: 微博获得的点赞数
- comment_count: 微博获得的评论数
- repost_count: 微博的转发数
- engagement: 互动参与度
- comment_crawled_count: 爬取的评论数
- comment_hq_count: 高质量评论数
- comment_hq_ratio: 高质量评论占比
- comment_hq_user_count: 发表高质量评论的用户数
- topic_value: 话题价值等级
- topic_value_label: 话题价值等级标签
- trending_date: 上榜日期
- trending_type: 话题类型
- trending_click: 话题点击量

2. df_topic_comment（话题评论数据）
- comment_id: 评论唯一标识符
- weibo_id: 所属微博ID
- parent_id: 父评论ID（用于区分一级评论与回复）
- user_id: 评论用户ID
- screen_name: 评论用户昵称
- gender: 评论用户性别
- content: 评论文本内容
- text_length: 评论文本长度
- text_quality: 评论文本质量评分
- text_quality_label: 评论文本质量等级标签
- create_time: 评论发布时间
- year, month, day, hour: 时间分解字段
- weekday: 评论发布时的星期几
- like_count: 评论点赞数
- sub_comment_count: 该评论的回复数
- engagement: 互动参与度
- ip_location: 评论用户IP属地

3. df_user_info（用户信息数据）
- user_id: 用户唯一标识符
- screen_name: 用户昵称
- gender: 用户性别
- ip_location: 用户IP属地
- registration_time: 账号注册时间
- account_age_days: 账号存在天数
- verified: 是否认证
- verified_type: 认证类型代码
- verified_type_name: 认证类型名称
- total_weibo_count: 用户微博总数
- follower_count: 粉丝数
- following_count: 关注数
- follower_following_ratio: 粉丝关注比
- user_rank: 用户等级
- weibo_crawled_count: 爬取微博数
- weibo_hq_count: 高质量微博数
- weibo_hq_ratio: 高质量微博占比
- original_ratio: 原创微博占比
- comment_crawled_count: 爬取评论数
- comment_hq_count: 高质量评论数
- comment_hq_ratio: 高质量评论占比
- active_days: 用户活跃天数
- description: 个人简介
- user_value: 用户价值评分
- user_value_label: 用户价值等级标签

4. df_user_weibo（用户历史微博数据）
- weibo_id: 微博唯一标识符
- user_id: 用户ID
- screen_name: 用户昵称
- content: 微博文本内容
- text_length: 微博文本长度
- text_quality: 微博文本质量评分
- text_quality_label: 微博文本质量等级标签
- create_time: 发布时间
- year, month, day, hour: 时间分解字段
- weekday: 星期几
- like_count: 点赞数
- comment_count: 评论数
- repost_count: 转发数
- engagement: 互动参与度
- is_repost: 是否转发微博
- reposted_weibo_id: 被转发原微博ID
- topics: 关联话题列表
- at_users: @提及用户列表

请基于以上数据结构，为我编写一个 Streamlit 可视化原型。要求如下：

一、总体要求
1. 代码必须可运行，并尽量模块化。
2. 优先保证“展示效果”和“结构清晰”，不要过度工程化。
3. 必须使用：
   - Plotly：用于统计图表
   - NetworkX：用于构建评论关系图
   - PyVis：用于渲染评论回复网络/树
   - Streamlit：用于搭建交互界面
4. 对于数据量较大的表，要考虑性能问题，不能一次性把全部评论都画成网络图。需要设计合理的筛选与采样逻辑。
5. 代码中必须有清晰注释，便于我后续阅读和修改。
6. 界面语言使用中文。
7. 注意兼容中文文本显示。
8. 不要伪造数据字段，不要臆造不存在的列。若某个字段缺失，请在代码中做健壮性处理。

二、功能页面设计
请至少实现以下几个模块，建议做成一个 Streamlit 多区块页面，或者侧边栏导航。

【模块1：数据总览】
目标：展示数据集规模与整体概况。

建议内容：
- 4 张表的记录数展示
- 热点微博数量、评论数量、用户数量、历史微博数量
- 若干指标卡（st.metric）
- 话题价值等级分布图
- 热点类型分布图
- 热点微博评论数/互动量分布图
- 评论文本质量等级分布图
- 用户价值等级分布图

图表建议：
- Plotly 柱状图、饼图、箱线图、直方图均可
- 要求视觉简洁，不要堆砌过多颜色

【模块2：典型热点微博展示】
目标：选择一个热点事件中的典型微博，展示主帖信息。

建议内容：
- 通过侧边栏下拉框选择 topic 或 weibo_id
- 显示微博正文、发布时间、发布者昵称、互动指标
- 显示该微博的评论总数、已爬取评论数、高质量评论数、高质量评论占比等
- 可补充一个该微博评论时间分布图
- 可补充该微博评论点赞分布图

要求：
- 微博正文要以较好的排版展示
- 主帖指标可以用 metric/card 风格展示

【模块3：评论回复关系可视化】
这是本项目最重要的模块。

目标：
围绕某一条热点微博，展示其评论区中的“评论-回复”关系结构，体现出讨论链条和互动层级。

具体要求：
1. 基于 df_topic_comment 中的 comment_id 和 parent_id 构建有向图。
2. 使用 NetworkX 构建图。
3. 使用 PyVis 渲染交互式网络图。
4. 节点至少应包含以下信息中的若干项：
   - comment_id
   - screen_name
   - content（必要时截断）
   - like_count
   - sub_comment_count
   - text_quality_label
   - ip_location
5. 鼠标悬浮节点时，能够显示较完整的评论信息。
6. 一级评论与回复评论在视觉上要尽量区分。
7. 应考虑以下边界情况：
   - parent_id 为空或无效
   - parent_id 不在当前微博评论集合中
   - 存在孤立节点
8. 由于评论量可能很多，必须设计一个“用于答辩展示的子图抽取策略”，例如：
   - 仅展示某条微博下点赞数最高的前 N 条一级评论及其回复
   - 或仅展示高质量评论构成的子图
   - 或保留高互动评论，并补齐祖先链路
9. 需要在界面上提供筛选参数，例如：
   - 最大评论数量
   - 最小点赞数阈值
   - 是否只看高质量评论
   - 是否只看存在回复关系的评论
10. 若 PyVis 图无法很好嵌入，请实现合理的 Streamlit 嵌入方式，例如导出 HTML 后再嵌入。

【模块4：评论明细与用户信息联动展示】
目标：当我在展示评论时，可以同时看到该评论用户的相关信息。

建议内容：
- 在评论表格中展示：
  - comment_id
  - screen_name
  - content
  - like_count
  - sub_comment_count
  - text_quality_label
  - ip_location
- 同时将 df_topic_comment 与 df_user_info 按 user_id 关联
- 展示对应用户的：
  - screen_name
  - gender
  - ip_location
  - verified / verified_type_name
  - follower_count
  - following_count
  - follower_following_ratio
  - user_rank
  - user_value_label
  - active_days
  - description
- 可以采用如下方式之一：
  1. 选中某条评论后，在右侧展示用户卡片
  2. 选择某位评论用户后，展示其画像信息
  3. 点击评论节点后，在下方同步展示用户详情（若点击联动过难，也可退化为下拉选择）

要求：
- 必须让老师能够看出：评论背后对应的是“可建模的个体用户”
- 用户信息展示不必过多，但必须具有代表性

【模块5：可选增强模块】
如时间允许，可实现以下任意 1~2 个增强功能：
1. 评论发布时间趋势图
2. 评论用户 IP 属地分布图
3. 高质量评论用户 vs 普通评论用户的简单对比
4. 某评论用户的历史微博摘要展示（从 df_user_weibo 中抽取最近若干条）
5. 话题层面的小型统计看板

三、数据处理要求
1. 请先编写数据加载函数，支持从本地文件加载数据。
2. 数据文件格式可能是 CSV / PKL / Parquet，请尽量写得通用，或者至少将读取部分集中封装，方便我自行替换。
3. 对时间字段做统一解析。
4. 合理处理缺失值、异常 parent_id、重复记录。
5. 对展示文本做适度截断，但保留完整信息用于 hover。
6. 对大图进行采样，避免前端卡顿。

四、代码结构要求
请将代码组织得尽量清晰，建议类似如下结构，但不强制：
- app.py                      # Streamlit 主程序
- data_loader.py              # 数据读取与预处理
- visualization.py            # Plotly 图表函数
- comment_graph.py            # NetworkX + PyVis 图构建逻辑
- utils.py                    # 通用工具函数
- requirements.txt            # 依赖列表
- README_usage.md             # 运行说明（可选）

如果你认为单文件更适合演示，也可以采用单文件，但必须逻辑清晰、函数化良好。

五、实现细节要求
1. Streamlit 页面布局要整洁，适合答辩展示。
2. 请优先使用宽屏布局。
3. 适当使用：
   - st.sidebar
   - st.metric
   - st.plotly_chart
   - st.dataframe
   - st.expander
   - st.tabs
4. 对于 PyVis 生成的 HTML，请给出稳定的嵌入方式。
5. 对评论网络图的节点文字不要直接全部铺开，避免过于拥挤；优先采用 hover 展示详细内容。
6. 代码应包含必要的异常处理和提示信息，例如：
   - 数据文件不存在
   - 所选微博无评论
   - 评论关系无法构图
7. 尽量让默认展示就是一个“好看且合理的案例”，避免我运行后还要手动调很多参数。

六、输出要求
请直接输出以下内容：
1. 完整代码
2. requirements.txt 内容
3. 如何运行项目的说明
4. 你对采样策略/评论子图抽取策略的说明
5. 若有必要，请简要说明你做出的关键设计取舍

七、重要约束
1. 不要只给伪代码，必须给出可直接运行或经过少量路径修改即可运行的代码。
2. 不要把所有逻辑都写成空函数。
3. 不要省略关键实现，尤其是：
   - 评论图构建
   - PyVis 嵌入 Streamlit
   - 评论与用户信息关联展示
4. 优先完成一个可用的 MVP，再考虑增强功能。
5. 如果你发现某些交互在 Streamlit 中实现复杂，请优先选择“稳定可展示”的替代方案，而不是为了复杂交互牺牲可运行性。

八、推荐默认实现方案
为了确保项目可顺利落地，我建议你默认采用以下实现思路：
- 使用侧边栏选择一个 topic，再选择该 topic 下的一条 weibo
- 页面上方展示主帖信息与指标
- 中间使用 Plotly 展示若干统计图
- 下方使用 NetworkX + PyVis 展示评论回复关系图
- 最下方展示评论明细表与用户信息表
- 评论图默认仅展示：
  “点赞数靠前且有回复关系的一级评论 + 它们的回复”
  或
  “高质量评论中能构成回复关系的子图”
请你自行选择一个更稳妥、更适合答辩展示的方案，并在代码中实现

九、数据路径
请在代码中预留如下可修改位置：
DATA_DIR = "D:\GraduationProject\data\cleaned"

在目录下存在以下文件名（你可以写成可配置）：
- topic_weibo.parquet
- topic_comment.parquet
- user_info.parquet
- user_weibo.parquet

十、代码风格
1. 使用 Python 3.10+ 语法
2. 添加必要的类型注解
3. 函数命名清晰
4. 注释简洁但到位
5. 不要为了“高级感”写得过于晦涩

请开始完成这一任务。