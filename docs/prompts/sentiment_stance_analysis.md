# 1 系统提示词（System Prompt）

```
你是一名“社交媒体评论情绪与立场分析助手”，服务于“基于多智能体的社会群体情绪模拟系统”的数据建模阶段。

你的任务是：对单条中文社交媒体评论进行结构化分析，并严格按照给定 JSON Schema 输出结果。

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
```

# 2 用户输入模板（User Prompt Template）

```
请根据既定规则分析下面这条评论，并按 JSON Schema 返回结果。

输入：
{
  "comment_id": "{{comment_id}}",
  "topic": "{{topic}}",
  "source_weibo_summary": "{{source_weibo_summary}}",
  "parent_comment_summary": "{{parent_comment_summary}}",
  "comment_text": "{{comment_text}}"
}
```

实现细节建议：

- `topic`：填评论所属话题名，能帮助模型理解讨论背景
-  `parent_comment_summary`：如果是一级评论可填空字符串；如果是二级评论，建议填被回复评论的简短摘要
-  `comment_text`：原始评论文本

如果某项没有，就传空字符串 `""`。

# 3 Schema

```python
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


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
        "unclear",
    ] = Field(
        ...,
        description="情绪指向对象的类别。若无法判断则为 'unclear'。"
    )

    emotion_target_text: str = Field(
        ...,
        description="情绪指向对象在文本中的具体表述；若无法判断则为空字符串。"
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
        "unclear",
    ] = Field(
        ...,
        description="立场指向对象的类别。若无法判断则为 'unclear'。"
    )

    stance_target_text: str = Field(
        ...,
        description="立场指向对象在文本中的具体表述；若无法判断则为空字符串。"
    )

    cause_or_stimulus: str = Field(
        ...,
        description="引发情绪或态度的触发因素；若不明确则为空字符串。"
    )

    target_explicit: bool = Field(
        ...,
        description="目标对象是否在评论文本中被明确提及。"
    )


class AppraisalLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_type: Literal[
        "event",
        "behavior",
        "object",
        "mixed",
        "unclear",
    ] = Field(
        ...,
        description="评论的主要评价焦点：事件、行为、对象、混合或不明确。"
    )

    responsibility: Literal[
        "self",
        "other_individual",
        "institution",
        "group",
        "society",
        "environment",
        "unclear",
    ] = Field(
        ...,
        description="评论中主要归责对象。"
    )

    control: Literal[
        "high",
        "medium",
        "low",
        "unclear",
    ] = Field(
        ...,
        description="评论者感知到的局面可控性。"
    )

    norm_violation: Literal[
        "high",
        "medium",
        "low",
        "unclear",
    ] = Field(
        ...,
        description="评论中体现出的规范违背/道德谴责程度。"
    )


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
    ] = Field(
        ...,
        description="评论表达的主导情绪标签。"
    )

    emotion_intensity: int = Field(
        ...,
        ge=0,
        le=3,
        description="情绪强度：0=无法判断/基本无明显表达，1=弱，2=中，3=强。"
    )

    stance_label: Literal[
        "favor",
        "against",
        "neutral",
        "mixed",
        "unclear",
    ] = Field(
        ...,
        description="评论相对于 stance target 的立场标签。"
    )

    stance_intensity: int = Field(
        ...,
        ge=0,
        le=3,
        description="立场强度：0=无法判断/基本无明显表达，1=弱，2=中，3=强。"
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
    ] = Field(
        ...,
        description="评论的主要表达/论证方式。"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="整体分析置信度。"
    )


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion_evidence: str = Field(
        ...,
        description="支撑情绪判断的关键文本证据；若不足则为空字符串。"
    )

    stance_evidence: str = Field(
        ...,
        description="支撑立场判断的关键文本证据；若不足则为空字符串。"
    )


class ReviewFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_more_context: bool = Field(
        ...,
        description="该评论是否明显依赖更多上下文才能稳定判断。"
    )

    low_confidence_reason: str = Field(
        ...,
        description="若置信度较低，简要说明原因；否则为空字符串。"
    )


class CommentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str = Field(
        ...,
        description="评论唯一标识。"
    )

    role_layer: RoleLayer = Field(
        ...,
        description="第1层：情绪事件角色层。"
    )

    appraisal_layer: AppraisalLayer = Field(
        ...,
        description="第2层：评价维度层。"
    )

    task_output_layer: TaskOutputLayer = Field(
        ...,
        description="第3层：任务输出层。"
    )

    evidence: Evidence = Field(
        ...,
        description="关键证据片段。"
    )

    review_flags: ReviewFlags = Field(
        ...,
        description="复核标记。"
    )
```

