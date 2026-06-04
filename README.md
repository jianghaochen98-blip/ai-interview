# AI-Interview 智面 - AI 面试模拟平台

基于大模型的智能面试模拟系统，支持简历智能解析、AI 模拟面试、实时流式评分与综合评估报告。集成 **Milvus 向量数据库 RAG** 与 **LangChain Harness Engineering** 架构。

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | FastAPI 0.115 + Uvicorn |
| LLM | DeepSeek（OpenAI 兼容 API） |
| Agent 框架 | LangChain 0.3 + Tool Calling Agent |
| RAG / 向量库 | Milvus 2.5 + DashScope Embedding |
| 数据库 | PostgreSQL 16（SQLAlchemy 2.0 异步） |
| 缓存/队列 | Redis 7 + Celery 5 |
| 用户认证 | JWT（python-jose + passlib） |
| 前端 | Vue 3 + Vite + Pinia + Vue Router + Axios |
| 容器化 | Docker Compose（多服务编排） |
| 部署 | Nginx 反向代理 + 多阶段 Dockerfile |

---

## 核心功能

### 面试流程
- **简历智能解析** - 上传 PDF 简历，AI 自动提取姓名、学历、技能、项目经历等结构化信息
- **简历质量分析** - AI 评分 + 优劣势 + 改进建议，自动识别实习/正式岗位调整评价标准
- **AI 模拟面试** - 根据简历 + 目标岗位 + 难度等级生成针对性题目，多轮逐题问答
- **实时流式评分** - SSE 流式推送 AI 评语，无需等待完整响应
- **综合评估报告** - 面试结束后生成总体评价、优劣势、录用建议

### RAG 知识增强
- **知识库管理** - 后台 CRUD 管理面试题、参考答案、知识点
- **向量检索** - Milvus 存储 Embedding，支持按分类/难度筛选 + 相似度阈值
- **RAG 出题** - 检索知识库中相似岗位参考题，注入 Prompt 生成针对性面试题
- **RAG 评分** - 检索参考答案与评分要点，注入 Prompt 提升评分稳定性与可解释性

### Harness Engineering（Agent 架构）
- **静态提示词** - 8 个 YAML 模板定义角色、约束、输出格式，非开发人员可直接调整 Prompt
- **动态提示词** - 运行时注入 resume/position/difficulty/chat_history 等上下文
- **指令注入** - 仿 Claude Code 的 `inject_instruction` 机制，动态追加运行时规则
- **Tool Calling** - LangChain Agent 循环，支持自主决策调用工具（查简历、查历史、查题库）
- **向后兼容** - 适配器模式，原有 API 调用零改动

---

## 项目结构

```
ai-interview/
├── ai-interview-backend/           # FastAPI 后端
│   ├── app/
│   │   ├── agent/                  # Agent Harness 工程
│   │   │   ├── harness.py          # AgentHarness + ToolCallingHarness + MessageManager
│   │   │   ├── agents/             # 场景 Agent（简历/出题/评分/报告/RAG）
│   │   │   ├── tools/              # LangChain Tools（面试 + RAG 检索）
│   │   │   ├── prompts/static/     # YAML 提示词模板
│   │   │   ├── context/            # PromptContext + ContextBuilder
│   │   │   └── adapters/           # AIService 适配层
│   │   ├── rag/                    # RAG 模块
│   │   │   ├── milvus_client.py    # Milvus 连接管理与向量检索
│   │   │   ├── embedding_service.py # Embedding 向量化服务
│   │   │   ├── retriever.py        # 检索服务
│   │   │   └── ingestion.py        # 数据摄入服务
│   │   ├── api/
│   │   │   ├── client/v1/          # 用户端 API
│   │   │   └── backoffice/v1/      # 管理端 API
│   │   ├── services/client/        # 业务逻辑层
│   │   ├── models/                 # SQLAlchemy 数据模型
│   │   ├── schemas/                # Pydantic 请求/响应模型
│   │   ├── core/                   # 配置、安全、Celery、日志
│   │   ├── route/                  # 路由注册中心
│   │   └── db/                     # 数据库会话管理
│   ├── migrations/                 # Alembic 数据库迁移
│   ├── docker-compose.yml          # 生产环境编排
│   ├── docker-compose.dev.yml      # 开发环境编排
│   ├── Dockerfile
│   └── requirements.txt
├── ai-interview-frontend/          # 用户端（Vue 3 SPA）
└── ai-interview-admin/             # 管理端（Vue 3 SPA）
```

---

## 快速开始

### 环境要求
- Docker 20.10+ / Docker Compose v2+
- Python 3.11+（开发模式）
- Node.js 18+（开发模式）

### 1. 配置环境变量

```bash
cd ai-interview-backend
cp .env.example .env
```

编辑 `.env` 文件，配置必要变量：

```env
# DeepSeek API
DEEPSEEK_API_KEY=sk-your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Embedding（DashScope 或 OpenAI 兼容）
EMBEDDING_API_KEY=sk-your-embedding-key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSION=1536

# 数据库
POSTGRES_USER=demo
POSTGRES_PASSWORD=demo123
POSTGRES_DB=ai_interview

# JWT
SECRET_KEY=your-secret-key-change-in-production
```

### 2. Docker Compose 启动

```bash
# 开发环境
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# 生产环境
docker compose up -d --build

# 带监控面板
docker compose --profile monitoring up -d --build
```

### 3. 数据库初始化

```bash
docker exec -it ai-interview-app alembic upgrade head
```

### 4. 访问服务

| 服务 | 地址 |
|------|------|
| 用户端前端 | http://localhost:3000 |
| 管理端前端 | http://localhost:3001 |
| Client API 文档 | http://localhost:8006/client/docs |
| Backoffice API 文档 | http://localhost:8006/backoffice/docs |
| 健康检查 | http://localhost:8006/api/v1/config/health |

### 5. 前端启动（开发模式）

```bash
cd ai-interview-frontend
npm install && npm run dev

cd ai-interview-admin
npm install && npm run dev
```

---

## API 概览

### 用户端 API (`/api/v1/`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/auth/register` | POST | 用户注册 |
| `/auth/login` | POST | 用户登录 |
| `/auth/refresh` | POST | 刷新 Token |
| `/resumes/upload` | POST | 上传简历 PDF |
| `/resumes/{id}` | GET | 获取简历及 AI 分析 |
| `/interviews/start` | POST | 开始面试 |
| `/interviews/{id}/submit` | POST | 提交回答并获取评分 |
| `/interviews/{id}/submit-stream` | POST | 流式提交回答（SSE） |
| `/interviews/{id}/report` | GET | 获取面试报告 |
| `/interviews` | GET | 面试记录列表 |

### 管理端 API (`/api/v1/backoffice/`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/auth/login` | POST | 管理员登录 |
| `/users` | GET | 用户列表（分页+搜索） |
| `/users/{id}/toggle-active` | PUT | 启用/禁用用户 |
| `/users/stats` | GET | 平台统计数据 |
| `/interviews` | GET | 面试记录列表 |
| `/interviews/{id}` | GET | 面试详情（含消息和报告） |
| `/knowledge` | GET/POST | 知识库列表 / 新增条目 |
| `/knowledge/batch` | POST | 批量导入知识条目 |
| `/knowledge/{id}` | GET/PUT/DELETE | 知识条目 CRUD |
| `/knowledge/search` | POST | 知识库向量检索 |
| `/knowledge/rebuild-index` | POST | 重建 Milvus 索引 |

---

## Docker 服务架构

```
                    ┌─────────────┐
                    │    Nginx    │  :80/:443
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │  :8006
                    └──┬───┬───┬─┘
         ┌─────────────┤   │   ├──────────────┐
         ▼             ▼   │   ▼              ▼
   ┌──────────┐  ┌────────┐│┌─────────┐ ┌──────────┐
   │PostgreSQL│  │  Redis  │││ Celery  │ │  Milvus  │
   │   :5432  │  │  :6379  │││ Worker  │ │  :19530  │
   └──────────┘  └────────┘│└─────────┘ └────┬─────┘
                            │┌─────────┐ ┌───┴───┐
                            ││  Celery │ │ etcd  │
                            ││  Beat   │ │ :2379 │
                            │└─────────┘ └───────┘
                            │               ┌───────┐
                            │               │ MinIO │
                            │               └───────┘
```

---

## RAG 数据流

```
管理员写入知识
       │
       ▼
PostgreSQL (knowledge_items)                          Milvus 向量存储
       │                                                    ▲
       ▼                                                    │
EmbeddingService.embed_texts() ────────────────────────────┘
       │
面试出题/评分时
       │
       ▼
Retriever.search(query, top_k, category, difficulty) ──► 向量检索
       │
       ▼
RAG 上下文注入 Prompt → LLM 生成题目 / 评分
```

---

## Agent Harness 架构

```
┌──────────────────────────────────────────────┐
│               AgentHarness                     │
│  ┌───────────┐  ┌────────────┐  ┌─────────┐  │
│  │ 静态提示词 │  │ 动态上下文  │  │ 额外指令 │  │
│  │ (YAML)    │  │ (Runtime)  │  │(inject) │  │
│  └─────┬─────┘  └─────┬──────┘  └────┬────┘  │
│        └──────┬───────┴───────────────┘       │
│               ▼                                │
│      PromptAssembler.assemble()                │
│               │                                │
│     ┌─────────┴─────────┐                      │
│     ▼                   ▼                      │
│ SimpleHarness    ToolCallingHarness            │
│ (prompt→resp)    (Agent 循环+工具调用)           │
└──────────────────────────────────────────────┘
```

---

## 项目亮点

1. **RAG 知识增强** - Milvus 向量检索 + DashScope Embedding，知识库检索结果注入 Prompt
2. **Harness Engineering** - 仿 Claude Code 架构，YAML 提示词 + 动态上下文 + 指令注入
3. **双端 API** - Client API 与 Backoffice API 共享 Service 层
4. **SSE 流式传输** - AI 评语逐字推送，用户体验流畅
5. **实习/正式差异化** - 自动识别岗位类型调整出题难度与评价标准
6. **Docker 一键部署** - 多服务编排，健康检查+自动重启
7. **统一响应格式** - ApiResponse + 分层异常体系

---

## 技术实现要点

| 特征 | 实现 |
|------|------|
| 简历解析 | pdfplumber 提取文本 → DeepSeek 结构化输出 |
| 面试出题 | Prompt 注入 resume/position/difficulty → LLM 生成 |
| 回答评分 | Prompt 注入 reference_answer/key_points → LLM 评分 |
| 流式输出 | FastAPI StreamingResponse + SSE |
| 向量存储 | Milvus Standalone，COSINE 相似度，IVF_FLAT 索引 |
| Agent 框架 | LangChain Tool Calling Agent，最大 5 次迭代 |
| 异步处理 | Celery + Redis 消息队列 |
| 认证授权 | JWT（access/refresh token 双令牌） |
| 数据迁移 | Alembic 版本控制 |
