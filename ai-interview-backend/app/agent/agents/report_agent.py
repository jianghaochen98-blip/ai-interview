from __future__ import annotations

from typing import Dict, List

from openai import AsyncOpenAI

from app.agent.context.types import PromptContext
from app.agent.harness import SimpleAgentHarness


class ReportGeneratorAgent(SimpleAgentHarness):
    """报告生成 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("report_generator", llm_client=llm_client)

    async def generate(
        self,
        parsed_resume: dict,
        target_position: str,
        questions_and_scores: List[Dict],
    ) -> dict:
        ctx = PromptContext(
            parsed_resume=parsed_resume,
            target_position=target_position,
            questions_and_scores=questions_and_scores,
        )
        result = await self.run(ctx)
        return result.parsed
