from __future__ import annotations

from typing import Dict

from openai import AsyncOpenAI

from app.agent.context.types import PromptContext
from app.agent.harness import SimpleAgentHarness


class ResumeParserAgent(SimpleAgentHarness):
    """简历解析 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("resume_parser", llm_client=llm_client)

    async def parse(self, resume_text: str) -> dict:
        ctx = PromptContext(resume_text=resume_text)
        result = await self.run(ctx)
        return result.parsed


class ResumeAnalyzerAgent(SimpleAgentHarness):
    """简历分析 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("resume_analyzer", llm_client=llm_client)

    async def analyze(
        self, parsed_resume: dict, target_position: str
    ) -> dict:
        ctx = PromptContext(
            parsed_resume=parsed_resume,
            target_position=target_position,
        )
        result = await self.run(ctx)
        return result.parsed
