from __future__ import annotations

from typing import List

from openai import AsyncOpenAI

from app.agent.context.types import PromptContext
from app.agent.harness import SimpleAgentHarness


class QuestionGeneratorAgent(SimpleAgentHarness):
    """出题 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("question_generator", llm_client=llm_client)

    async def generate(
        self,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int,
    ) -> list:
        ctx = PromptContext(
            parsed_resume=parsed_resume,
            target_position=target_position,
            difficulty=difficulty,
            question_count=count,
        )
        result = await self.run(ctx)
        return result.parsed
