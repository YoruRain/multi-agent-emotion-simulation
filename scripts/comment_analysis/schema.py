"""Prompt text and Pydantic schema for comment sentiment and stance analysis."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MODEL_NAME = "deepseek-chat"

SYSTEM_PROMPT = """
你是一名“社交媒体评论情绪与立场分析助手”，服务于“基于多智能体的社会群体情绪模拟系统”的数据建模阶段。

你的任务是：对单条中文社交媒体评论进行结构化分析，并严格按照给定 Schema 输出结果。

你的分析必须分为三个层次，但只输出最终结构化结果，不输出推理过程：

第 1 层：情绪事件角色层（role_layer）
目标是识别这条评论中：
1. 谁/什么是情绪指向对象（emotion target）
2. 谁/什么是立场指向对象（stance target）
3. 引发情绪或态度的触发因素（cause or stimulus）
4. 目标是否在文本中被明确提到（target_explicit）

第 2 层：评价维度层（appraisal_layer）
目标是判断这条评论背后的评价结构：
1. 关注焦点属于事件、行为、对象，还是混合
2. 责任主要归因于谁
3. 评论者感知到的局面可控性高低
4. 评论是否包含明显的规范违背/道德谴责判断

第 3 层：任务输出层（task_output_layer）
目标是给出最终可用于建模的结果：
1. 情绪标签及强度
2. 立场标签及强度
3. 论证/表达方式
4. 整体置信度

分析原则：
1. 只基于输入内容进行判断，不要凭空补充外部事实。
2. 如果提供了“话题”“源微博摘要”“父评论摘要”，要综合利用；如果为空，则仅依据评论文本本身。
3. “情绪”与“立场”必须区分：
   - 情绪是评论者表达出的情感状态，如愤怒、失望、喜悦。
   - 立场是评论者相对于某个目标的态度，如支持、反对、中立。
   - 两者的 target 可以相同，也可以不同。
4. 若信息不足，不要编造。请使用：
   - 枚举字段中的 "unclear"
   - 文本字段中的空字符串 ""
5. 强度定义：
   - 0 = 无法判断 / 基本无明显表达
   - 1 = 弱
   - 2 = 中
   - 3 = 强
6. confidence 取值范围为 0 到 1：
   - 低于 0.4：很不确定
   - 0.4 到 0.7：中等确定
   - 高于 0.7：较确定
7. 若评论高度依赖上下文、反讽过强、目标缺失或语义残缺，应将 needs_more_context 设为 true。
8. 输出必须是合法 JSON，且必须严格符合给定 schema。
9. 不要输出任何 JSON 之外的文字。
""".strip()

USER_PROMPT = """
请根据既定规则分析下面这条评论，并按 JSON Schema 返回结果。

输入：
{input_json}
""".strip()

SCHEMA_INSTRUCTION = """
输出格式硬性要求：
1. 只能输出一个 JSON object，不要输出 Markdown、解释、注释或多余文本。
2. 顶层必须包含且只包含 JSON Schema 要求的字段：
   comment_id, role_layer, appraisal_layer, task_output_layer, evidence, review_flags。
3. comment_id 必须与输入中的 comment_id 完全一致。
4. 所有字段名必须逐字匹配 Schema。
5. 所有枚举值必须使用 Schema 中给定的英文枚举值，不要翻译成中文。
6. evidence 和 review_flags 是必填对象，不能省略。

target 抽取规则：
1. target_explicit=True 表示评论文本本身明确出现了 target。
2. 只要判定 target_explicit=True，emotion_target_text 和 stance_target_text 中的非空 target text
   必须能在 comment_text 原文中逐字找到，不能为空。
3. comment_text 中出现的名词、代词、指代短语或行为短语，只能视为候选 target，而不是自动视为最终 target。
4. 只有当该候选 target 是评论中真正被评价、被支持、被反对、被归责、被批评或被赞许的对象时，
   才可将其填入 target_text。
5. 若句中出现的人物、群体、说法或经历仅用于举例、转述、引用、说明风险、补充背景或描述过程，
   而真正被评价的是更抽象的行为、决策、事件、观点、规则或后果，则应将后者作为 target。
6. 如果无法从评论原文中抽出明确对象，或目标对象只能依赖 topic 或 source_weibo_summary 推断，
   则优先将 target_explicit 设为 False，并将对应 target_text 设为空字符串 ""。
7. 对依赖背景推断的隐含 target，应适当降低 confidence；若目标不稳，needs_more_context 应设为 true。

target 一致性规则：
1. emotion_target 与 stance_target 可以不同，但只有在评论中确实存在两个不同评价对象时才允许不同。
2. 若评论中只存在一个明确且真实的评价对象，优先让 emotion_target 与 stance_target 保持一致。
3. 若某一 target_type 已明确但 target_text 为空，应谨慎检查该 target 是否只是基于背景推断；若是，则优先将对应 target_explicit 设为 False，并适当降低 confidence。

target_type 类似枚举值的区分：
- behavior：被评价的是某种行为、做法、操作方式、决策方式、处理方式
- policy：被评价的是制度、规则、政策、立场主张、规范性安排
- event：被评价的是某个具体事件或事件结果
- object：被评价的是具体事物、物品、系统或抽象对象，但不强调其行为属性

evidence 抽取规则：
1. evidence 必须尽量直接摘取 comment_text 原文中的最短关键片段。
2. evidence 应尽量短、尽量来自原文、尽量直接支撑最终标签，不要写成摘要、解释、改写或扩写。
3. 若存在多个证据片段，可用中文分号连接。
4. 若评论文本中没有可直接支撑判断的片段，则 evidence 使用空字符串 ""，不要从背景摘要中摘取。

argument_type 判定规则：
1. argument_type 应优先反映评论的主要语用功能，而不是表面句式。
2. 反问句如果主要作用是表达否定评价、质疑或批评，应优先标为 evaluation。
3. 只有评论主要是在寻求信息、等待回答、缺少明确评价倾向时，才标为 question。

needs_more_context 触发条件：
1. 代词/省略严重，无法得知在说谁
2. target 和 stance 都无法稳定确定
3. 明显反讽/隐喻，且正反都说得通
4. 评论只说结果，不说对象，且话题上下文也不足以补足
5. 文本过短，完全没有评价锚点
否则，哪怕对象不够细，只要情绪或立场方向明确，即可把 needs_more_context 设为 False。

unclear 使用规则：
1. 只有在评论文本与提供背景结合后仍无法做出相对稳定判断时，才使用 unclear。
2. 若情绪或立场方向已较明确，但对象不够具体，不应仅因对象不够细就整体输出 unclear。
3. 应优先保留能够稳定判断的字段，只将无法判断的局部字段设为 unclear。

confidence 判定参考：
1. 若 target 明确、evidence 可直接定位、情绪或立场方向清晰，则 confidence 应较高。
2. 若 target 部分依赖背景推断，或 evidence 较弱，但整体方向仍较明确，则 confidence 应为中等。
3. 若存在明显反讽、省略严重、目标不稳、evidence 很弱或多种解释都说得通，则 confidence 应较低。
4. confidence 应与 target_explicit、needs_more_context、evidence 质量保持一致，不要出现明显矛盾。

举例与转述规则：
1. 评论中出现的人物、群体、说法，不一定自动构成 target。
2. 若某人物/对象只是被用来举例、转述、引用或铺垫，而真正被评价的是更抽象的行为、事件、风险或观点，则应将真正被评价的对象作为 target。
3. 不要仅因句中出现某个人物名词，就机械地将其判为 stance_target 或 emotion_target。


必须严格遵守下面的 JSON Schema：
{schema_json}
""".strip()


class RoleLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion_target_type: Literal[
        "event",
        "person",
        "institution",
        "group",
        "policy",
        "media",
        "commenter_self",
        "other_commenter",
        "object",
        "behavior", 
        "unclear",
    ] = Field(..., description="情绪指向对象的类别。若无法判断则为 'unclear'。")

    emotion_target_text: str = Field(
        ..., description="情绪指向对象在文本中的具体表述；若无法判断则为空字符串。"
    )

    stance_target_type: Literal[
        "event",
        "person",
        "institution",
        "group",
        "policy",
        "media",
        "commenter_self",
        "other_commenter",
        "object",
        "behavior", 
        "unclear",
    ] = Field(..., description="立场指向对象的类别。若无法判断则为 'unclear'。")

    stance_target_text: str = Field(
        ..., description="立场指向对象在文本中的具体表述；若无法判断则为空字符串。"
    )

    cause_or_stimulus: str = Field(
        ..., description="引发情绪或态度的触发因素；若不明确则为空字符串。"
    )

    target_explicit: bool = Field(..., description="目标对象是否在评论文本中被明确提及。")


class AppraisalLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_type: Literal[
        "event",
        "behavior",
        "object",
        "mixed",
        "unclear",
    ] = Field(..., description="评论的主要评价焦点：事件、行为、对象、混合或不明确。")

    responsibility: Literal[
        "self",
        "other_individual",
        "institution",
        "media",
        "group",
        "society",
        "environment",
        "unclear",
    ] = Field(..., description="评论中主要归责对象。")

    control: Literal[
        "high",
        "medium",
        "low",
        "unclear",
    ] = Field(..., description="评论者感知到的局面可控性。")

    norm_violation: Literal[
        "high",
        "medium",
        "low",
        "unclear",
    ] = Field(..., description="评论中体现出的规范违背/道德谴责程度。")


class TaskOutputLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion_label: Literal[
        "anger",
        "sadness",
        "fear",
        "joy",
        "disgust",
        "disappointment",
        "surprise",
        "none",
        "mixed",
        "unclear",
    ] = Field(..., description="评论表达的主导情绪标签。")

    emotion_intensity: int = Field(
        ..., ge=0, le=3, description="情绪强度：0=无法判断/基本无明显表达，1=弱，2=中，3=强。"
    )

    stance_label: Literal[
        "favor",
        "against",
        "neutral",
        "mixed",
        "unclear",
    ] = Field(..., description="评论相对于 stance target 的立场标签。")

    stance_intensity: int = Field(
        ..., ge=0, le=3, description="立场强度：0=无法判断/基本无明显表达，1=弱，2=中，3=强。"
    )

    argument_type: Literal[
        "fact",
        "evaluation",
        "causality",
        "sarcasm",
        "venting",
        "appeal",
        "agreement",
        "question",
        "unclear",
    ] = Field(..., description="评论的主要表达/论证方式。")

    confidence: float = Field(..., ge=0.0, le=1.0, description="整体分析置信度。")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion_evidence: str = Field(
        ..., description="支撑情绪判断的关键文本证据；若不足则为空字符串。"
    )

    stance_evidence: str = Field(
        ..., description="支撑立场判断的关键文本证据；若不足则为空字符串。"
    )


class ReviewFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_more_context: bool = Field(..., description="该评论是否明显依赖更多上下文才能稳定判断。")

    low_confidence_reason: str = Field(
        ..., description="若置信度较低，简要说明原因；否则为空字符串。"
    )


class CommentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str = Field(..., description="评论唯一标识。")
    role_layer: RoleLayer = Field(..., description="第1层：情绪事件角色层。")
    appraisal_layer: AppraisalLayer = Field(..., description="第2层：评价维度层。")
    task_output_layer: TaskOutputLayer = Field(..., description="第3层：任务输出层。")
    evidence: Evidence = Field(..., description="关键证据片段。")
    review_flags: ReviewFlags = Field(..., description="复核标记。")


@lru_cache(maxsize=1)
def get_comment_analysis_schema_json() -> str:
    if hasattr(CommentAnalysis, "model_json_schema"):
        schema = CommentAnalysis.model_json_schema()
    else:
        schema = CommentAnalysis.schema()
    return json.dumps(schema, ensure_ascii=False, indent=2)
