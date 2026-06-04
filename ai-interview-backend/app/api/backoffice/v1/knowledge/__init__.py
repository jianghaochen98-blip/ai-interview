from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete as sql_delete

from app.db.session import get_db
from app.api.backoffice.deps import get_current_admin
from app.models.admin import Admin
from app.models.knowledge_base import KnowledgeItem as KnowledgeItemModel
from app.schemas.knowledge import (
    KnowledgeItemCreate,
    KnowledgeItemListOut,
    KnowledgeItemOut,
    KnowledgeItemUpdate,
    KnowledgeSearchIn,
    KnowledgeSearchResult,
    KnowledgeSearchOut,
    KnowledgeBatchCreate,
)
from app.schemas.response import ApiResponse
from app.rag.ingestion import ingestion_service
from app.rag.retriever import retriever
from app.rag.schemas import KnowledgeItem as RagKnowledgeItem

router = APIRouter()


async def _sync_to_milvus(item: KnowledgeItemModel):
    """将数据库记录同步到 Milvus"""
    rag_item = RagKnowledgeItem(
        knowledge_id=item.id,
        content=item.content,
        category=item.category,
        difficulty=item.difficulty,
        source=item.source,
    )
    await ingestion_service.ingest_batch([rag_item])


@router.get("")
async def list_knowledge(
    page: int = 1,
    per_page: int = 20,
    category: str = None,
    difficulty: str = None,
    keyword: str = None,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    query = select(KnowledgeItemModel).order_by(KnowledgeItemModel.updated_at.desc())
    count_query = select(func.count()).select_from(KnowledgeItemModel)

    if category:
        query = query.where(KnowledgeItemModel.category == category)
        count_query = count_query.where(KnowledgeItemModel.category == category)
    if difficulty:
        query = query.where(KnowledgeItemModel.difficulty == difficulty)
        count_query = count_query.where(KnowledgeItemModel.difficulty == difficulty)
    if keyword:
        query = query.where(
            (KnowledgeItemModel.title.ilike(f"%{keyword}%"))
            | (KnowledgeItemModel.content.ilike(f"%{keyword}%"))
        )
        count_query = count_query.where(
            (KnowledgeItemModel.title.ilike(f"%{keyword}%"))
            | (KnowledgeItemModel.content.ilike(f"%{keyword}%"))
        )

    total = await db.scalar(count_query)
    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    items = result.scalars().all()

    return ApiResponse.success(
        data={
            "items": [KnowledgeItemOut.model_validate(i) for i in items],
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    )


@router.post("")
async def create_knowledge(
    body: KnowledgeItemCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    item = KnowledgeItemModel(
        title=body.title,
        content=body.content,
        category=body.category,
        difficulty=body.difficulty,
        source=body.source,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    await _sync_to_milvus(item)

    return ApiResponse.success(
        data=KnowledgeItemOut.model_validate(item), message="创建成功"
    )


@router.post("/batch")
async def batch_create_knowledge(
    body: KnowledgeBatchCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    if not body.items:
        return ApiResponse.failed(message="items 不能为空", body_code=400)

    db_items = []
    rag_items = []
    for item_data in body.items:
        item = KnowledgeItemModel(
            title=item_data.title,
            content=item_data.content,
            category=item_data.category,
            difficulty=item_data.difficulty,
            source=item_data.source,
        )
        db.add(item)
        db_items.append(item)

    await db.flush()

    for item in db_items:
        rag_items.append(
            RagKnowledgeItem(
                knowledge_id=item.id,
                content=item.content,
                category=item.category,
                difficulty=item.difficulty,
                source=item.source,
            )
        )

    await db.commit()
    await ingestion_service.ingest_batch(rag_items)

    return ApiResponse.success(
        data={"count": len(db_items)}, message=f"批量创建 {len(db_items)} 条成功"
    )


@router.get("/{knowledge_id}")
async def get_knowledge(
    knowledge_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    item = await db.get(KnowledgeItemModel, knowledge_id)
    if not item:
        return ApiResponse.failed(message="知识条目不存在", body_code=404, http_code=404)
    return ApiResponse.success(data=KnowledgeItemOut.model_validate(item))


@router.put("/{knowledge_id}")
async def update_knowledge(
    knowledge_id: int,
    body: KnowledgeItemUpdate,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    item = await db.get(KnowledgeItemModel, knowledge_id)
    if not item:
        return ApiResponse.failed(message="知识条目不存在", body_code=404, http_code=404)

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    await db.commit()
    await db.refresh(item)

    await ingestion_service.delete_by_knowledge_ids([knowledge_id])
    await _sync_to_milvus(item)

    return ApiResponse.success(
        data=KnowledgeItemOut.model_validate(item), message="更新成功"
    )


@router.delete("/{knowledge_id}")
async def delete_knowledge(
    knowledge_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    item = await db.get(KnowledgeItemModel, knowledge_id)
    if not item:
        return ApiResponse.failed(message="知识条目不存在", body_code=404, http_code=404)

    await db.delete(item)
    await db.commit()
    await ingestion_service.delete_by_knowledge_ids([knowledge_id])

    return ApiResponse.success(message="删除成功")


@router.post("/search")
async def search_knowledge(
    body: KnowledgeSearchIn,
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    results = await retriever.search(
        query=body.query,
        top_k=body.top_k,
        category=body.category,
        difficulty=body.difficulty,
        min_score=body.min_score,
    )

    return ApiResponse.success(
        data=KnowledgeSearchOut(
            query=body.query,
            results=[
                KnowledgeSearchResult(
                    knowledge_id=r.knowledge_id,
                    content=r.content,
                    category=r.category,
                    difficulty=r.difficulty,
                    score=r.score,
                )
                for r in results
            ],
        )
    )


@router.post("/rebuild-index")
async def rebuild_index(
    db: AsyncSession = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
):
    """从数据库重建 Milvus 索引"""
    query = select(KnowledgeItemModel).where(
        KnowledgeItemModel.status == "active"
    )
    result = await db.execute(query)
    items = result.scalars().all()

    if not items:
        return ApiResponse.success(message="没有可重建的数据", data={"count": 0})

    rag_items = [
        RagKnowledgeItem(
            knowledge_id=item.id,
            content=item.content,
            category=item.category,
            difficulty=item.difficulty,
            source=item.source,
        )
        for item in items
    ]

    ingestion_service.ensure_collection()
    count = await ingestion_service.ingest_batch(rag_items)
    return ApiResponse.success(message=f"索引重建完成", data={"count": count})
