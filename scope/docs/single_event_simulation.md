# AgentScope 单事件微博用户仿真器

## 目标

本模块用于运行“静态画像驱动的单事件、单轮群体反应模拟”。给定一个 `event_id`，程序会加载该事件、微博用户 Agent 画像、系统提示词和少量历史记忆，调用 AgentScope Agent 生成每个用户的结构化 JSON 反应，并输出基础统计报告。

当前 MVP 不实现 Agent 之间的对话、扩散链路或多轮情绪状态更新。

## 输入文件

默认输入目录为 `scope/data/inputs/`：

- `agent_profiles.jsonl`：每行一个用户 Agent 画像，包含 `agent_id`、`user_id`、`base_identity`、`prompt_profile`、`behavior_parameters`、`metadata`。
- `agent_memories.jsonl`：每行一个用户的历史记忆样本，包含 `memory_user_level` 和 `memories` 列表。
- `agent_sys_prompts.jsonl`：每行一个用户的系统提示词，按 `agent_id` 关联。
- `events.jsonl`：每行一个热点事件，按 `event_id` 检索。

如果某个 Agent 缺少 memories，程序会记录 warning 并继续；如果缺少 `sys_prompt`，程序会用 `prompt_profile` 生成兜底系统提示词。

## 环境与模型配置

当前 `.gp` 环境尚未安装 AgentScope。推荐在项目环境中安装：

```powershell
conda run -p D:\GraduationProject\.gp pip install agentscope
```

模型配置通过环境变量读取，不要在代码中硬编码 API key：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
$env:MODEL_NAME="deepseek-chat"
$env:BASE_URL="https://api.deepseek.com/v1"
```

实现使用 AgentScope 的 `ReActAgent`、`OpenAIChatModel`、`OpenAIChatFormatter` 和 `InMemoryMemory`。真实调用时会优先使用 `structured_model=ReactionSchema` 获取结构化输出；如果输出无法直接解析，会进行一次严格 JSON retry。

## 运行示例

先做 dry run，检查事件消息和记忆片段：

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_single_event_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 1 `
  --memory-user-level core `
  --dry-run
```

正式运行小规模仿真：

```powershell
conda run -p D:\GraduationProject\.gp python scope\run_single_event_simulation.py `
  --event-id event_5177192956301027 `
  --max-agents 5 `
  --memory-user-level core
```

常用参数：

- `--event-id`：必填，指定单个事件。
- `--max-agents`：限制运行 Agent 数量，方便调试。
- `--memory-user-level`：可选过滤值，例如 `core`、`normal`、`background`。
- `--output-dir`：默认 `scope/data/outputs/simulation/single_event`。
- `--overwrite`：覆盖当前 run 输出文件。
- `--resume` / `--no-resume`：是否跳过已完成的 `event_id + agent_id`。
- `--dry-run`：只打印输入摘要，不调用模型。
- `--seed`：传给模型生成参数，便于复现实验。
- `--concurrency`：MVP 暂保留参数，当前仍串行执行。

## 输出文件

每次正式运行会生成一个 run 目录：

```text
scope/data/outputs/simulation/single_event/{run_id}/
```

主要输出：

- `agent_reactions.jsonl`：逐 Agent 写入，单个 Agent 失败不会中断整体运行。
- `summary_report.csv`：事件级汇总统计。
- `memory_user_level_summary.csv`：按记忆用户层级分组统计。
- `influence_level_summary.csv`：按影响力层级分组统计。
- `action_type_summary.csv`：按行为类型分组统计。

`agent_reactions.jsonl` 至少包含：

- `run_id`、`event_id`、`weibo_id`、`topic`
- `agent_id`、`user_id`、`memory_user_level`
- `verified_type_name`、`influence_level`、`propagation_role`
- `participate`、`action_type`
- `emotion_label`、`emotion_intensity`
- `stance_label`、`stance_intensity`
- `reaction_text`、`reason`
- `raw_output`、`parse_status`、`error_message`
- `model_name`、`created_at`

## Agent 输出 JSON

模型输出会被校验为以下结构：

```json
{
  "participate": true,
  "action_type": "comment",
  "emotion_label": "neutral",
  "emotion_intensity": 1,
  "stance_label": "neutral",
  "stance_intensity": 1,
  "reaction_text": "这里写该用户可能发表的微博式反应",
  "reason": "用一句话说明为什么该用户会产生这种反应"
}
```

字段约束：

- `action_type`：`ignore`、`comment`、`repost`、`repost_with_comment`
- `emotion_label`：`positive`、`neutral`、`anger`、`sadness`、`disgust`、`worry`、`surprise`
- `stance_label`：`support`、`against`、`neutral`、`unclear`
- `emotion_intensity`、`stance_intensity`：只能是 `0`、`1`、`2`
- 当 `participate=false` 时，`action_type` 必须为 `ignore`，`reaction_text` 必须为空字符串，两个强度必须为 `0`

## MVP 限制

- 只模拟单个事件。
- Agent 之间暂不交互。
- 不做多轮传播。
- `behavior_parameters` 暂不深度参与状态更新，只用于结果记录和后续扩展。
- 第一版串行运行，`concurrency` 只作为未来并发扩展入口。

## 后续扩展

- 多轮传播与时间步模拟。
- KOL 先发声，普通用户随后响应。
- 根据 `influence_score` 和 `kol_sensitivity_score` 构造影响权重。
- 引入群体情绪状态更新。
- 使用 AgentScope 长期记忆或检索模块做更精细的记忆召回。
