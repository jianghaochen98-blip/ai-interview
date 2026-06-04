from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.db.base import Base


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(256), nullable=False, default="")
    content = Column(Text, nullable=False)
    category = Column(String(64), nullable=False, default="general", index=True)
    difficulty = Column(String(32), nullable=False, default="medium")
    source = Column(String(256), nullable=False, default="")
    status = Column(String(32), nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
