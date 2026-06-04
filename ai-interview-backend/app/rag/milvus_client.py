from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class MilvusStore:
    """Milvus 向量数据库连接管理与集合操作"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name or settings.MILVUS_COLLECTION_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._connected = False
        self._collection: Optional[Collection] = None

    def connect(self):
        """建立 Milvus 连接"""
        if self._connected:
            return
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )
        self._connected = True
        logger.info(f"Milvus 连接成功: {self.host}:{self.port}")

    def disconnect(self):
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False

    def collection_exists(self) -> bool:
        self.connect()
        return utility.has_collection(self.collection_name)

    def create_collection(self, drop_if_exists: bool = False):
        """创建知识库集合"""
        self.connect()

        if self.collection_exists():
            if drop_if_exists:
                utility.drop_collection(self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                self._collection = Collection(self.collection_name)
                self._collection.load()
                return

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="knowledge_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="difficulty",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="AI 面试知识库",
            enable_dynamic_field=True,
        )

        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        self._collection.load()
        logger.info(f"集合创建成功: {self.collection_name} (dim={self.dimension})")

    def get_collection(self) -> Collection:
        if self._collection is None:
            self.connect()
            if not self.collection_exists():
                self.create_collection()
            else:
                self._collection = Collection(self.collection_name)
                self._collection.load()
        return self._collection

    def insert(self, data: List[Dict[str, Any]]) -> List[int]:
        """批量插入向量数据"""
        collection = self.get_collection()
        result = collection.insert(data)
        collection.flush()
        return result.primary_keys

    def delete_by_knowledge_ids(self, knowledge_ids: List[int]):
        """根据 knowledge_id 删除数据"""
        collection = self.get_collection()
        expr = f"knowledge_id in {knowledge_ids}"
        collection.delete(expr)
        collection.flush()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """向量相似度检索"""
        collection = self.get_collection()

        expr_parts = []
        if category:
            expr_parts.append(f'category == "{category}"')
        if difficulty:
            expr_parts.append(f'difficulty == "{difficulty}"')
        expr = " && ".join(expr_parts) if expr_parts else None

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "knowledge_id",
                "content",
                "category",
                "difficulty",
                "source",
            ],
        )

        hits = []
        for result in results[0]:
            if result.score >= min_score:
                hits.append({
                    "id": result.id,
                    "knowledge_id": result.entity.get("knowledge_id"),
                    "content": result.entity.get("content"),
                    "category": result.entity.get("category", ""),
                    "difficulty": result.entity.get("difficulty", ""),
                    "source": result.entity.get("source", ""),
                    "score": result.score,
                })
        return hits

    def count(self) -> int:
        collection = self.get_collection()
        return collection.num_entities


milvus_store = MilvusStore()
