from __future__ import annotations

import json
from typing import List, Optional

from openai import AsyncOpenAI

from app.agent.context.types import PromptContext
from app.agent.harness import SimpleAgentHarness
from app.agent.tools.rag_tools import search_knowledge_base
from app.rag.retriever import retriever


class RAGQuestionGeneratorAgent(SimpleAgentHarness):
    """带 RAG 检索的出题 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("rag_question_generator", llm_client=llm_client)

    async def generate(
        self,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int,
        enable_rag: bool = True,
    ) -> list:
        rag_context = ""
        if enable_rag:
            rag_context = await retriever.search_formatted(
                query=target_position,
                top_k=5,
                difficulty=difficulty,
            )

        ctx = PromptContext(
            parsed_resume=parsed_resume,
            target_position=target_position,
            difficulty=difficulty,
            question_count=count,
        )
        context = self._build_context(ctx)
        context["rag_context"] = rag_context

        messages = self._loader.render_messages(self.template, context)
        response = await self._llm_client.chat.completions.create(
            model=self._llm_model_name,
            messages=messages,
            temperature=self.template.temperature,
            max_tokens=self.template.max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        return self.extract_json(raw)

    @property
    def _llm_model_name(self) -> str:
        from app.core.config import settings
        return settings.DEEPSEEK_MODEL


class RAGAnswerEvaluatorAgent(SimpleAgentHarness):
    """带 RAG 参考的评分 Agent"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__("rag_answer_evaluator", llm_client=llm_client)

    async def evaluate(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
        enable_rag: bool = True,
    ) -> dict:
        rag_context = ""
        if enable_rag:
            rag_context = await retriever.search_formatted(
                query=question,
                top_k=3,
            )

        ctx = PromptContext(
            parsed_resume=resume_context,
            question=question,
            answer=answer,
            chat_history=chat_history,
        )
        context = self._build_context(ctx)
        context["rag_context"] = rag_context

        messages = self._loader.render_messages(self.template, context)
        response = await self._llm_client.chat.completions.create(
            model=self._llm_model_name,
            messages=messages,
            temperature=self.template.temperature,
            max_tokens=self.template.max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        return self.extract_json(raw)

    @property
    def _llm_model_name(self) -> str:
        from app.core.config import settings
        return settings.DEEPSEEK_MODEL
