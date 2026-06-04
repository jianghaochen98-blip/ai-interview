import json
import logging
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.agent.adapters.ai_service_adapter import ai_service as _harness_ai

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL
)


class AIService:

    @staticmethod
    async def parse_resume(resume_text: str) -> dict:
        return await _harness_ai.parse_resume(resume_text)

    @staticmethod
    async def analyze_resume(parsed_resume: dict, target_position: str) -> dict:
        return await _harness_ai.analyze_resume(parsed_resume, target_position)

    @staticmethod
    async def generate_questions(
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int
    ) -> list:
        return await _harness_ai.generate_questions(
            parsed_resume, target_position, difficulty, count
        )

    @staticmethod
    async def evaluate_answer(
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
        next_question: Optional[str] = None
    ) -> dict:
        return await _harness_ai.evaluate_answer(
            question=question,
            answer=answer,
            resume_context=resume_context,
            chat_history=chat_history,
            next_question=next_question,
        )

    @staticmethod
    async def evaluate_answer_stream(
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list
    ):
        async for chunk in _harness_ai.evaluate_answer_stream(
            question=question,
            answer=answer,
            resume_context=resume_context,
            chat_history=chat_history,
        ):
            yield chunk

    @staticmethod
    async def generate_report(
        parsed_resume: dict,
        target_position: str,
        questions_and_scores: list
    ) -> dict:
        return await _harness_ai.generate_report(
            parsed_resume, target_position, questions_and_scores
        )
