from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptContext:
    """动态上下文，运行时注入到 Prompt 模板中"""

    resume_text: Optional[str] = None
    parsed_resume: Optional[Dict[str, Any]] = None
    target_position: Optional[str] = None
    difficulty: Optional[str] = None
    question_count: Optional[int] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    questions_and_scores: Optional[List[Dict[str, Any]]] = None
    extra_instructions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AgentConfig:
    """Agent 运行时配置"""

    temperature: float = 0.5
    max_tokens: int = 2000
    max_history_messages: int = 6
    stream: bool = False


@dataclass
class HarnessResult:
    """Harness 执行结果"""

    parsed: Dict[str, Any]
    raw_response: str
    messages: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
