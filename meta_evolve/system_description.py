"""SystemDescription：让 skill 生成器知道在优化什么系统。"""

from __future__ import annotations

from dataclasses import dataclass

from .task_adapter import Task
from .task_profile import TaskProfile


@dataclass
class SystemDescription:
    task_name: str
    domain_description: str  # Task.system_prompt
    profile: TaskProfile  # 8 维任务画像
    initial_code_preview: str  # EVOLVE-BLOCK 前 800 字符
    mutable_scope: str  # 可变范围描述
    evaluator_source: str = ""  # 评估器源码

    def format_for_prompt(self) -> str:
        profile = self.profile
        tags = ", ".join(profile.tags())
        lines = [
            "## 目标系统",
            f"**任务**: {self.task_name}",
            f"**画像**: {tags}",
            f"**可变范围**: {self.mutable_scope}",
            "",
            "**领域描述**:",
            self.domain_description[:600],
            "",
            "**初始代码预览**:",
            f"```python\n{self.initial_code_preview}\n```",
        ]
        if self.evaluator_source:
            lines.append("")
            lines.append("**评分函数预览**:")
            lines.append(f"```python\n{self.evaluator_source[:2000]}\n```")
        return "\n".join(lines)


def build_system_description(task: Task, profile: TaskProfile) -> SystemDescription:
    return SystemDescription(
        task_name=task.name,
        domain_description=task.system_prompt,
        profile=profile,
        initial_code_preview=task.initial_code[:800],
        mutable_scope="EVOLVE-BLOCK 内的 Python 函数",
        evaluator_source=task.evaluator_source,
    )
