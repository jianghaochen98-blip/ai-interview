from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptTemplate:
    """从 YAML 加载的 Prompt 模板"""

    def __init__(self, raw: Dict[str, Any], yaml_path: Path):
        self.name: str = raw["name"]
        self.version: str = raw.get("version", "1.0")
        self.description: str = raw.get("description", "")
        self.temperature: float = raw.get("temperature", 0.5)
        self.max_tokens: int = raw.get("max_tokens", 2000)
        self.expects_json: bool = raw.get("expects_json", True)
        self.static: Dict[str, Any] = raw.get("static", {})
        self.dynamic: Dict[str, str] = raw.get("dynamic", {})
        self._raw = raw
        self._yaml_path = yaml_path

    @property
    def role(self) -> str:
        return self.static.get("role", "")

    @property
    def task(self) -> str:
        return self.static.get("task", "")

    @property
    def constraints(self) -> List[str]:
        return self.static.get("constraints", [])

    @property
    def output_schema(self) -> Optional[Dict[str, Any]]:
        return self.static.get("output_schema")

    @property
    def user_template(self) -> str:
        return self.dynamic.get("user_template", "")

    @property
    def scoring_rubric(self) -> Optional[Dict[str, str]]:
        return self.static.get("scoring_rubric")

    @property
    def level_hints(self) -> Optional[Dict[str, Dict[str, str]]]:
        return self.static.get("level_hints")

    @property
    def difficulty_map(self) -> Optional[Dict[str, str]]:
        return self.static.get("difficulty_map")

    def get_level_hint(self, level: str) -> str:
        if not self.level_hints:
            return ""
        hint_config = self.level_hints.get(level, {})
        return hint_config.get("instruction", "")


class PromptLoader:
    """YAML 模板加载器，支持静态提示词 + 动态上下文渲染"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).parent / "static")
        self._prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptTemplate] = {}

    def load(self, name: str) -> PromptTemplate:
        """加载并缓存指定模板"""
        if name in self._cache:
            return self._cache[name]

        yaml_path = self._prompts_dir / f"{name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt 模板不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        template = PromptTemplate(raw, yaml_path)
        self._cache[name] = template
        logger.info(f"加载 Prompt 模板: {name} (v{template.version})")
        return template

    def build_system_prompt(
        self,
        template: PromptTemplate,
        context: Dict[str, Any],
    ) -> str:
        """构建静态 System Prompt，可注入额外指令"""
        parts: List[str] = []

        if template.role:
            parts.append(template.role)

        if template.task:
            parts.append(template.task)

        # 注入动态指令（Claude Code 式 inject_instruction）
        extra_instructions = context.get("_extra_instructions", [])
        if isinstance(extra_instructions, str):
            extra_instructions = [extra_instructions]
        for instr in extra_instructions:
            parts.append(f"\n【附加指令】{instr}")

        # 注入 level hint
        level_hint = context.get("_level_hint", "")
        if level_hint:
            parts.append(f"\n{level_hint}")

        # 注入 difficulty desc
        difficulty_desc = context.get("_difficulty_desc", "")
        if difficulty_desc:
            parts.append(f"\n面试难度：{difficulty_desc}")

        # 约束
        for constraint in template.constraints:
            parts.append(constraint)

        # 评分标准
        if template.scoring_rubric:
            parts.append("\n评分标准：")
            for key, desc in template.scoring_rubric.items():
                parts.append(f"- {desc}")

        # 输出格式说明
        if template.output_schema:
            parts.append("\n返回格式：")
            parts.append(json.dumps(template.output_schema, ensure_ascii=False))

        return "\n".join(parts)

    def build_user_prompt(
        self,
        template: PromptTemplate,
        context: Dict[str, Any],
    ) -> str:
        """基于 Jinja2 渲染动态 User Prompt"""
        user_template = template.user_template
        if not user_template:
            return ""

        try:
            from jinja2 import Template as Jinja2Template
            jt = Jinja2Template(user_template)
            return jt.render(**context)
        except ImportError:
            result = user_template
            for key, value in context.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                placeholder = f"{{{{ {key} }}}}"
                result = result.replace(placeholder, str(value))
            return result

    def render_messages(
        self,
        template: PromptTemplate,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """渲染完整的 messages 列表"""
        system_content = self.build_system_prompt(template, context)
        user_content = self.build_user_prompt(template, context)

        messages: List[Dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages

    def reload(self, name: Optional[str] = None):
        """重新加载模板（支持热更新）"""
        if name:
            self._cache.pop(name, None)
            self.load(name)
        else:
            self._cache.clear()

    @property
    def loaded_templates(self) -> List[str]:
        return list(self._cache.keys())


prompt_loader = PromptLoader()
