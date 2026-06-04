from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.retriever import retriever


class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="检索查询文本，如'Python多线程面试题'")
    top_k: int = Field(default=5, description="返回最相似的前K条结果")
    category: Optional[str] = Field(
        default=None,
        description="按分类筛选：project/technical/coding/system-design",
    )
    difficulty: Optional[str] = Field(
        default=None, description="按难度筛选：easy/medium/hard"
    )


@tool(args_schema=SearchKnowledgeInput)
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> str:
    """检索知识库，获取与查询相关的面试题、参考答案、知识点等信息"""
    result = await retriever.search_formatted(
        query=query,
        top_k=top_k,
        category=category,
        difficulty=difficulty,
    )
    return result


class SearchSimilarQuestionsInput(BaseModel):
    query: str = Field(description="用于查找相似题目的查询文本")
    top_k: int = Field(default=3, description="返回最相似的前K道题")


@tool(args_schema=SearchSimilarQuestionsInput)
async def search_similar_questions(query: str, top_k: int = 3) -> str:
    """检索知识库中与给定问题相似的历史面试题，用于参考出题或对比评分"""
    result = await retriever.search_formatted(
        query=query,
        top_k=top_k,
        category="technical",
    )
    return result


RAG_TOOLS = [
    search_knowledge_base,
    search_similar_questions,
]
