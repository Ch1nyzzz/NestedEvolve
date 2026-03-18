"""Minimal bash + execute_code agent for the experimental noa_new flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from noa_new.artifacts import ArtifactStore
from noa_new.tools import MINIMAL_TOOLS, run_bash, run_python

DEFAULT_MODEL = "together_ai/moonshotai/Kimi-K2.5"


def _llm_call_with_tools(**kwargs):
    from utils.llm import llm_call_with_tools

    return llm_call_with_tools(**kwargs)


def build_system_prompt(
    *,
    layer: str,
    workspace: Path,
    target_description: str,
    test_set_path: str | None = None,
) -> str:
    guide = "WORKFLOW.md" if layer.upper() == "L1" else "l2_config/WORKFLOW.md"
    lines = [
        f"你是 {layer.upper()} 优化器。",
        "你只有 bash 和 execute_code 两个工具。",
        f"工作目录: {workspace}",
        f"目标: {target_description}",
        f"详细工作指南: cat {guide}",
    ]
    if layer.upper() == "L1":
        lines.append(
            "你必须把每一步结构化写入 trace.jsonl，并维护 metrics.json 与 pareto.json。"
        )
        if test_set_path:
            lines.append(f"约束: 不碰 test_set ({test_set_path})。")
    else:
        lines.append("你不直接修改 target_system；只通过 l1_config/ 的 6 个槽位纠偏。")
        lines.append("你必须把每个 patch 追加到 l2_patches.jsonl。")
    return "\n".join(lines)


@dataclass
class MinimalAgent:
    workspace: str
    target_description: str
    layer: str = "L1"
    model: str = DEFAULT_MODEL
    max_turns: int = 12
    bash_timeout_sec: int = 300
    python_timeout_sec: int = 600

    def __post_init__(self) -> None:
        self.workspace_path = Path(self.workspace).resolve()
        self.artifacts = ArtifactStore(self.workspace_path)
        self.layer = self.layer.upper()

    def run(self, initial_task: str | None = None) -> dict:
        system_prompt = build_system_prompt(
            layer=self.layer,
            workspace=self.workspace_path,
            target_description=self.target_description,
            test_set_path=str(self.workspace_path / "test_set.pkl"),
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": initial_task
                or "开始工作。先阅读 workflow，再探索目录结构，然后执行最合适的下一步。",
            }
        )
        tool_call_count = 0
        final_text = ""

        for turn in range(1, self.max_turns + 1):
            resp = _llm_call_with_tools(
                messages=messages,
                tools=MINIMAL_TOOLS,
                model=self.model,
                max_tokens=4096,
                temperature=0,
            )
            final_text = resp.text or ""

            if not resp.tool_calls:
                break

            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.text or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            for tc in resp.tool_calls:
                tool_call_count += 1
                output = self._execute_tool(tc.name, tc.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )
        summary = {
            "layer": self.layer,
            "workspace": str(self.workspace_path),
            "model": self.model,
            "turns": turn,
            "tool_calls": tool_call_count,
            "final_text": final_text,
        }
        self.artifacts.write_run_summary(summary)
        return summary

    def _execute_tool(self, name: str, arguments: dict) -> str:
        if name == "bash":
            return run_bash(
                arguments.get("command", ""),
                cwd=self.workspace_path,
                timeout_sec=self.bash_timeout_sec,
            )
        if name == "execute_code":
            return run_python(
                arguments.get("code", ""),
                cwd=self.workspace_path,
                timeout_sec=self.python_timeout_sec,
            )
        return json.dumps(
            {"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False
        )
