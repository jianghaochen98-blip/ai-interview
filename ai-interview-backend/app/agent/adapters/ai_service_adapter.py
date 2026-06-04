from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterable, Dict, List, Optional

from openai import AsyncOpenAI

from app.agent.agents.interview_agent import (
    AnswerEvaluatorAgent,
    AnswerEvaluatorStreamAgent,
)
from app.agent.agents.question_agent import QuestionGeneratorAgent
from app.agent.agents.rag_agent import (
    RAGAnswerEvaluatorAgent,
    RAGQuestionGeneratorAgent,
)
from app.agent.agents.report_agent import ReportGeneratorAgent
from app.agent.agents.resume_agent import ResumeAnalyzerAgent, ResumeParserAgent

logger = logging.getLogger(__name__)


class AIService:
    """保持原有 API 不变，内部切换到 Harness 架构"""

    def __init__(self):
        self._llm_client: Optional[AsyncOpenAI] = None
        self._resume_parser: Optional[ResumeParserAgent] = None
        self._resume_analyzer: Optional[ResumeAnalyzerAgent] = None
        self._question_generator: Optional[QuestionGeneratorAgent] = None
        self._rag_question_generator: Optional[RAGQuestionGeneratorAgent] = None
        self._answer_evaluator: Optional[AnswerEvaluatorAgent] = None
        self._rag_answer_evaluator: Optional[RAGAnswerEvaluatorAgent] = None
        self._answer_evaluator_stream: Optional[AnswerEvaluatorStreamAgent] = None
        self._report_generator: Optional[ReportGeneratorAgent] = None

    @property
    def llm_client(self) -> AsyncOpenAI:
        if self._llm_client is None:
            from app.core.config import settings
            self._llm_client = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
            )
        return self._llm_client

    @property
    def resume_parser(self) -> ResumeParserAgent:
        if self._resume_parser is None:
            self._resume_parser = ResumeParserAgent(self.llm_client)
        return self._resume_parser

    @property
    def resume_analyzer(self) -> ResumeAnalyzerAgent:
        if self._resume_analyzer is None:
            self._resume_analyzer = ResumeAnalyzerAgent(self.llm_client)
        return self._resume_analyzer

    @property
    def question_generator(self) -> QuestionGeneratorAgent:
        if self._question_generator is None:
            self._question_generator = QuestionGeneratorAgent(self.llm_client)
        return self._question_generator

    @property
    def rag_question_generator(self) -> RAGQuestionGeneratorAgent:
        if self._rag_question_generator is None:
            self._rag_question_generator = RAGQuestionGeneratorAgent(self.llm_client)
        return self._rag_question_generator

    @property
    def answer_evaluator(self) -> AnswerEvaluatorAgent:
        if self._answer_evaluator is None:
            self._answer_evaluator = AnswerEvaluatorAgent(self.llm_client)
        return self._answer_evaluator

    @property
    def rag_answer_evaluator(self) -> RAGAnswerEvaluatorAgent:
        if self._rag_answer_evaluator is None:
            self._rag_answer_evaluator = RAGAnswerEvaluatorAgent(self.llm_client)
        return self._rag_answer_evaluator

    @property
    def answer_evaluator_stream(self) -> AnswerEvaluatorStreamAgent:
        if self._answer_evaluator_stream is None:
            self._answer_evaluator_stream = AnswerEvaluatorStreamAgent(self.llm_client)
        return self._answer_evaluator_stream

    @property
    def report_generator(self) -> ReportGeneratorAgent:
        if self._report_generator is None:
            self._report_generator = ReportGeneratorAgent(self.llm_client)
        return self._report_generator

    async def parse_resume(self, resume_text: str) -> dict:
        return await self.resume_parser.parse(resume_text)

    async def analyze_resume(
        self, parsed_resume: dict, target_position: str
    ) -> dict:
        return await self.resume_analyzer.analyze(parsed_resume, target_position)

    async def generate_questions(
        self,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int,
    ) -> list:
        return await self.question_generator.generate(
            parsed_resume, target_position, difficulty, count
        )

    async def generate_questions_with_rag(
        self,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int,
    ) -> list:
        return await self.rag_question_generator.generate(
            parsed_resume, target_position, difficulty, count
        )

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
        next_question: Optional[str] = None,
    ) -> dict:
        return await self.answer_evaluator.evaluate(
            question=question,
            answer=answer,
            resume_context=resume_context,
            chat_history=chat_history,
        )

    async def evaluate_answer_with_rag(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
    ) -> dict:
        return await self.rag_answer_evaluator.evaluate(
            question=question,
            answer=answer,
            resume_context=resume_context,
            chat_history=chat_history,
        )

    async def evaluate_answer_stream(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
    ) -> AsyncIterable[str]:
        async for chunk in self.answer_evaluator_stream.evaluate_stream(
            question=question,
            answer=answer,
            resume_context=resume_context,
            chat_history=chat_history,
        ):
            yield chunk

    async def generate_report(
        self,
        parsed_resume: dict,
        target_position: str,
        questions_and_scores: list,
    ) -> dict:
        return await self.report_generator.generate(
            parsed_resume, target_position, questions_and_scores
        )

    def inject_instruction(self, agent_type: str, instruction: str):
        """向指定 Agent 注入额外指令"""
        agent_map = {
            "parser": self.resume_parser,
            "analyzer": self.resume_analyzer,
            "question": self.question_generator,
            "evaluator": self.answer_evaluator,
            "evaluator_stream": self.answer_evaluator_stream,
            "report": self.report_generator,
        }
        agent = agent_map.get(agent_type)
        if agent:
            agent.inject_instruction(instruction)


ai_service = AIService()
