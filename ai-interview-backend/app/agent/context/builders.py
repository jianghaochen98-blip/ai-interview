from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agent.context.types import PromptContext
from app.agent.prompts.loader import PromptTemplate


class ContextBuilder:
    """将 PromptContext 转换为模板渲染所需的字典"""

    def __init__(self, template: PromptTemplate):
        self.template = template

    def build(
        self,
        ctx: PromptContext,
        debug: bool = False,
        extra_instructions: Optional[List[str]] = None,
        level_hint: Optional[str] = None,
        difficulty_desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建渲染上下文"""
        context: Dict[str, Any] = {}

        if ctx.resume_text:
            context["resume_text"] = ctx.resume_text
        if ctx.parsed_resume:
            context["parsed_resume_json"] = json.dumps(
                ctx.parsed_resume, ensure_ascii=False
            )
        if ctx.target_position:
            context["target_position"] = ctx.target_position
        if ctx.difficulty:
            context["difficulty"] = ctx.difficulty
        if ctx.question_count:
            context["question_count"] = str(ctx.question_count)
        if ctx.question:
            context["question"] = ctx.question
        if ctx.answer:
            context["answer"] = ctx.answer
        if ctx.chat_history:
            context["chat_history_text"] = self._format_history(
                ctx.chat_history
            )
        if ctx.questions_and_scores:
            context["qa_records_text"] = self._format_qa_records(
                ctx.questions_and_scores
            )

        # 注入额外指令
        all_instructions = list(ctx.extra_instructions)
        if extra_instructions:
            all_instructions.extend(extra_instructions)
        if all_instructions:
            context["_extra_instructions"] = all_instructions

        if level_hint:
            context["_level_hint"] = level_hint
        if difficulty_desc:
            context["_difficulty_desc"] = difficulty_desc

        return context

    @staticmethod
    def _format_history(chat_history: List[Dict[str, str]]) -> str:
        lines = []
        for msg in chat_history:
            role_label = "面试官" if msg.get("role") == "interviewer" else "候选人"
            lines.append(f"{role_label}: {msg.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_qa_records(
        questions_and_scores: List[Dict[str, Any]],
    ) -> str:
        lines = []
        for item in questions_and_scores:
            lines.append(f"问题：{item.get('question', '')}")
            lines.append(f"回答：{item.get('answer', '未回答')}")
            lines.append(f"得分：{item.get('score', 'N/A')}")
            lines.append("")
        return "\n".join(lines)


def determine_level_hint(
    target_position: str, template: PromptTemplate
) -> str:
    """根据岗位判断是实习还是正式，返回对应的 level_hint"""
    if not template.level_hints:
        return ""
    is_intern = any(
        kw in target_position.lower()
        for kw in ["实习", "intern"]
    )
    level = "intern" if is_intern else "fulltime"
    return template.get_level_hint(level)


def determine_difficulty_desc(
    difficulty: str, template: PromptTemplate
) -> str:
    """根据难度返回难度描述"""
    if not template.difficulty_map:
        return ""
    return template.difficulty_map.get(
        difficulty, template.difficulty_map.get("medium", "")
    )
