from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, Dict, List, Optional

from openai import AsyncOpenAI

from app.agent.context.builders import (
    ContextBuilder,
    determine_difficulty_desc,
    determine_level_hint,
)
from app.agent.context.types import AgentConfig, HarnessResult, PromptContext
from app.agent.prompts.loader import PromptLoader, PromptTemplate, prompt_loader

logger = logging.getLogger(__name__)


class MessageManager:
    """对话历史管理 + 上下文窗口控制"""

    def __init__(self, max_history: int = 6):
        self._messages: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._injected_instructions: List[str] = []

    def add(self, role: str, content: str, **meta):
        self._messages.append({"role": role, "content": content, **meta})

    def get_history(self, window: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = window or self._max_history
        return self._messages[-limit:] if limit > 0 else []

    def get_chat_history(self, window: Optional[int] = None) -> List[Dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.get_history(window)
        ]

    def inject_instruction(self, instruction: str):
        """注入运行时指令（Claude Code 式动态提示词）"""
        self._injected_instructions.append(instruction)

    def clear_instructions(self):
        self._injected_instructions.clear()

    def get_instructions(self) -> List[str]:
        return list(self._injected_instructions)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self):
        self._messages.clear()
        self._injected_instructions.clear()


class AgentHarness(ABC):
    """Agent 基类：静态提示词 + 动态上下文调度"""

    def __init__(
        self,
        prompt_name: str,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[AsyncOpenAI] = None,
    ):
        self.prompt_name = prompt_name
        self.config = config or AgentConfig()
        self._loader: PromptLoader = prompt_loader
        self._template: Optional[PromptTemplate] = None
        self._message_manager = MessageManager(self.config.max_history_messages)
        self._llm_client = llm_client

    @property
    def template(self) -> PromptTemplate:
        if self._template is None:
            self._template = self._loader.load(self.prompt_name)
        return self._template

    @property
    def message_manager(self) -> MessageManager:
        return self._message_manager

    def inject_instruction(self, instruction: str):
        """Claude Code 式动态指令注入"""
        self._message_manager.inject_instruction(instruction)

    def _build_context(self, ctx: PromptContext, **kwargs) -> Dict[str, Any]:
        builder = ContextBuilder(self.template)

        level_hint = None
        if ctx.target_position and self.template.level_hints:
            level_hint = determine_level_hint(
                ctx.target_position, self.template
            )

        difficulty_desc = None
        if ctx.difficulty and self.template.difficulty_map:
            difficulty_desc = determine_difficulty_desc(
                ctx.difficulty, self.template
            )

        return builder.build(
            ctx,
            debug=False,
            extra_instructions=self._message_manager.get_instructions(),
            level_hint=level_hint,
            difficulty_desc=difficulty_desc,
        )

    def _build_messages(
        self, ctx: PromptContext, **kwargs
    ) -> List[Dict[str, str]]:
        context = self._build_context(ctx, **kwargs)
        return self._loader.render_messages(self.template, context)

    @abstractmethod
    async def run(self, ctx: PromptContext, **kwargs) -> HarnessResult:
        """子类实现具体的执行逻辑"""

    @staticmethod
    def extract_json(text: str) -> dict:
        """从 AI 响应中提取 JSON"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)


class SimpleAgentHarness(AgentHarness):
    """基础 Agent：单轮 prompt → response，无 Tool Calling"""

    def __init__(
        self,
        prompt_name: str,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[AsyncOpenAI] = None,
    ):
        super().__init__(prompt_name, config, llm_client)

    async def run(self, ctx: PromptContext, **kwargs) -> HarnessResult:
        messages = self._build_messages(ctx, **kwargs)

        response = await self._llm_client.chat.completions.create(
            model=self._llm_model_name,
            messages=messages,
            temperature=self.template.temperature,
            max_tokens=self.template.max_tokens,
        )

        raw_response = response.choices[0].message.content.strip()

        parsed = {}
        if self.template.expects_json:
            parsed = self.extract_json(raw_response)

        return HarnessResult(
            parsed=parsed,
            raw_response=raw_response,
            messages=messages,
        )

    async def run_stream(
        self, ctx: PromptContext, **kwargs
    ) -> AsyncIterable[str]:
        messages = self._build_messages(ctx, **kwargs)

        stream = await self._llm_client.chat.completions.create(
            model=self._llm_model_name,
            messages=messages,
            temperature=self.template.temperature,
            max_tokens=self.template.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def _llm_model_name(self) -> str:
        from app.core.config import settings
        return settings.DEEPSEEK_MODEL


class ToolCallingHarness(AgentHarness):
    """支持 LangChain Tool Calling 的 Agent Harness"""

    def __init__(
        self,
        prompt_name: str,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[AsyncOpenAI] = None,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(prompt_name, config, llm_client)
        self._tools = tools or []
        self._verbose = verbose
        self._agent_executor = None

    def add_tool(self, tool):
        self._tools.append(tool)
        self._agent_executor = None

    def set_tools(self, tools):
        self._tools = list(tools)
        self._agent_executor = None

    def _build_agent(self):
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from app.core.config import settings

        system_content = self._loader.build_system_prompt(self.template, {})

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_content),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm = ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=self.template.temperature,
            max_tokens=self.template.max_tokens,
        )

        agent = create_tool_calling_agent(llm=llm, tools=self._tools, prompt=prompt)

        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=self._verbose,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        return self._agent_executor

    @property
    def agent_executor(self):
        if self._agent_executor is None:
            self._build_agent()
        return self._agent_executor

    async def run(self, ctx: PromptContext, **kwargs) -> HarnessResult:
        user_content = self._build_user_content(ctx)
        chat_history = self._build_langchain_history(ctx)

        result = await self.agent_executor.ainvoke(
            {"input": user_content, "chat_history": chat_history}
        )

        raw_response = result.get("output", "")
        parsed = {}
        if self.template.expects_json:
            try:
                parsed = self.extract_json(raw_response)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Agent 输出非 JSON: {raw_response[:200]}")
                parsed = {"raw_output": raw_response}

        return HarnessResult(
            parsed=parsed,
            raw_response=raw_response,
            messages=[],
            tool_calls=result.get("intermediate_steps", []),
        )

    async def run_stream(self, ctx: PromptContext, **kwargs) -> AsyncIterable[str]:
        user_content = self._build_user_content(ctx)
        chat_history = self._build_langchain_history(ctx)

        async for event in self.agent_executor.astream_events(
            {"input": user_content, "chat_history": chat_history},
            version="v2",
        ):
            kind = event.get("event", "")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", None)
                if chunk and chunk.content:
                    yield chunk.content

    def _build_user_content(self, ctx: PromptContext) -> str:
        parts = []
        if ctx.parsed_resume:
            parts.append(json.dumps(ctx.parsed_resume, ensure_ascii=False, indent=2))
        if ctx.target_position:
            parts.append(f"目标岗位：{ctx.target_position}")
        if ctx.question:
            parts.append(f"当前问题：{ctx.question}")
        if ctx.answer:
            parts.append(f"候选人回答：{ctx.answer}")
        if ctx.extra_instructions:
            for instr in ctx.extra_instructions:
                parts.append(f"附加要求：{instr}")
        return "\n\n".join(parts)

    def _build_langchain_history(self, ctx: PromptContext) -> list:
        history = ctx.chat_history or []
        return [
            (
                "human" if m.get("role") == "interviewer" else "ai",
                m.get("content", ""),
            )
            for m in history[-self.config.max_history_messages:]
        ]
