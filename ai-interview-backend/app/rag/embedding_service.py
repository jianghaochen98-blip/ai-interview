from __future__ import annotations

import logging
from typing import List, Optional

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 向量化服务（DashScope / OpenAI 兼容 API）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.EMBEDDING_API_KEY or settings.DEEPSEEK_API_KEY
        self.base_url = base_url or settings.EMBEDDING_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def embed_text(self, text: str) -> List[float]:
        """单条文本向量化"""
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """查询文本向量化（与 embed_text 相同，便于未来扩展）"""
        return await self.embed_text(query)


embedding_service = EmbeddingService()
