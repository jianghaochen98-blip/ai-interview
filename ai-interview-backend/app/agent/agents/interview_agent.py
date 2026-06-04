from __future__ import annotations

import re
from typing import AsyncIterable, Dict, List, Optional

from openai import AsyncOpenAI

from app.agent.context.types import PromptContext
from app.agent.harness import SimpleAgentHarness


class AnswerEvaluatorAgent(SimpleAgentHarness):
    """回答评分 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("answer_evaluator", llm_client=llm_client)

    async def evaluate(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: List[Dict[str, str]],
    ) -> dict:
        ctx = PromptContext(
            parsed_resume=resume_context,
            question=question,
            answer=answer,
            chat_history=chat_history,
        )
        result = await self.run(ctx)
        return result.parsed


class AnswerEvaluatorStreamAgent(SimpleAgentHarness):
    """流式回答评分 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("answer_evaluator_stream", llm_client=llm_client)

    async def evaluate_stream(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: List[Dict[str, str]],
    ) -> AsyncIterable[str]:
        ctx = PromptContext(
            parsed_resume=resume_context,
            question=question,
            answer=answer,
            chat_history=chat_history,
        )
        async for chunk in self.run_stream(ctx):
            yield chunk
