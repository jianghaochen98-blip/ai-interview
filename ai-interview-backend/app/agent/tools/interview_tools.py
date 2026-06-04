from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class InterviewToolContext:
    """工具运行时共享上下文（由调用方在每次 Agent 运行前注入）"""

    def __init__(self):
        self.resume: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []
        self.current_question: Optional[str] = None
        self.questions_list: List[Dict[str, Any]] = []

    def to_json(self) -> str:
        return json.dumps(
            {
                "resume": self.resume,
                "chat_history": self.chat_history,
                "current_question": self.current_question,
            },
            ensure_ascii=False,
        )


_tool_context = InterviewToolContext()


def set_tool_context(
    resume: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    current_question: Optional[str] = None,
    questions_list: Optional[List[Dict[str, Any]]] = None,
):
    global _tool_context
    if resume is not None:
        _tool_context.resume = resume
    if chat_history is not None:
        _tool_context.chat_history = chat_history
    if current_question is not None:
        _tool_context.current_question = current_question
    if questions_list is not None:
        _tool_context.questions_list = questions_list


def get_tool_context() -> InterviewToolContext:
    return _tool_context


class GetResumeContextInput(BaseModel):
    pass


@tool(args_schema=GetResumeContextInput)
def get_resume_context() -> str:
    """获取候选人简历的结构化信息，包含姓名、学历、技能、工作经历、项目经历等"""
    ctx = get_tool_context()
    if not ctx.resume:
        return "暂无可用的简历信息"
    return json.dumps(ctx.resume, ensure_ascii=False)


class GetChatHistoryInput(BaseModel):
    recent_n: int = Field(
        default=6, description="返回最近 N 条对话记录，默认 6 条"
    )


@tool(args_schema=GetChatHistoryInput)
def get_chat_history(recent_n: int = 6) -> str:
    """获取当前面试的对话历史记录"""
    ctx = get_tool_context()
    history = ctx.chat_history[-recent_n:] if ctx.chat_history else []
    if not history:
        return "暂无对话历史"
    lines = []
    for msg in history:
        role = "面试官" if msg.get("role") == "interviewer" else "候选人"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


class GetCurrentQuestionInput(BaseModel):
    pass


@tool(args_schema=GetCurrentQuestionInput)
def get_current_question() -> str:
    """获取当前面试题目"""
    ctx = get_tool_context()
    if not ctx.current_question:
        return "暂无当前题目"
    return ctx.current_question


class GetAllQuestionsInput(BaseModel):
    pass


@tool(args_schema=GetAllQuestionsInput)
def get_all_questions() -> str:
    """获取本次面试所有题目的概览"""
    ctx = get_tool_context()
    if not ctx.questions_list:
        return "暂无题目信息"
    items = []
    for q in ctx.questions_list:
        items.append(
            f"[{q.get('index', '?')}] {q.get('category', '未知分类')}: {q.get('question', '')}"
        )
    return "\n".join(items)


INTERVIEW_TOOLS = [
    get_resume_context,
    get_chat_history,
    get_current_question,
    get_all_questions,
]
