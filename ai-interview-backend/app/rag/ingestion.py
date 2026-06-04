from __future__ import annotations

import logging
from typing import List, Optional

from app.rag.embedding_service import embedding_service
from app.rag.milvus_client import milvus_store
from app.rag.schemas import KnowledgeItem

logger = logging.getLogger(__name__)


class IngestionService:
    """知识库数据摄入服务"""

    def __init__(self):
        self._store = milvus_store
        self._embedding = embedding_service

    async def ingest_batch(
        self,
        items: List[KnowledgeItem],
    ) -> int:
        """批量摄入知识条目"""
        if not items:
            return 0

        texts = [item.content for item in items]
        embeddings = await self._embedding.embed_texts(texts)

        data = []
        for item, emb in zip(items, embeddings):
            data.append({
                "knowledge_id": item.knowledge_id or 0,
                "content": item.content,
                "category": item.category,
                "difficulty": item.difficulty,
                "source": item.source,
                "embedding": emb,
            })

        ids = self._store.insert(data)
        logger.info(f"摄入 {len(ids)} 条知识到 Milvus")
        return len(ids)

    async def delete_by_knowledge_ids(self, knowledge_ids: List[int]):
        """根据知识ID删除向量"""
        if not knowledge_ids:
            return
        self._store.delete_by_knowledge_ids(knowledge_ids)
        logger.info(f"已删除 {len(knowledge_ids)} 条知识向量")

    def ensure_collection(self):
        """确保 Milvus 集合已创建"""
        self._store.create_collection()


ingestion_service = IngestionService()
