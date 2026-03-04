"""Initiator — 读取目标源码，LLM 内省生成 SystemDescription。"""

from __future__ import annotations

import json
import os

from utils.llm import llm_call, resolve_model
from noa.core.protocol import SystemDescription, SourceFile
from noa.core import prompts


def initiate(
    source_dir: str,
    model: str = resolve_model("gpt-4.1-mini"),
    system_description: str = "",
) -> SystemDescription:
    """读取源码 + LLM 内省，返回结构化 SystemDescription。

    不计算 baseline — 由 engine 从首次 observe 的轨迹中推导。
    """
    source_files = collect_sources(source_dir)
    source_context = "\n\n".join(
        f"## File: {sf.path}\n```{'json' if sf.path.endswith('.json') else 'python'}\n{sf.content}\n```"
        for sf in source_files
    )

    prompt = prompts.INITIATOR_PROMPT.format(
        source_code=source_context,
        system_description=system_description or "No additional description.",
        layer_context="",
    )
    resp = llm_call(
        prompt,
        model=model,
        max_tokens=4096,
        temperature=0,
        system=prompts.INITIATOR_SYSTEM,
    )
    parsed = _parse_json(resp.text)

    return SystemDescription(
        workflow_summary=parsed.get("workflow_summary", ""),
        component_names=parsed.get("component_names", []),
        source_files=source_files,
        source_dir=source_dir,
        baseline_score=0,
    )


def collect_sources(source_dir: str) -> list[SourceFile]:
    """收集目录下所有 .py/.json 文件（排除 __pycache__ 和 _noa_adapter.py）。"""
    _EXTENSIONS = {".py", ".json"}
    source_files = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1]
            if ext not in _EXTENSIONS or fname == "_noa_adapter.py":
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, source_dir)
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            source_files.append(SourceFile(path=rel, content=content))
    return source_files


def _parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end].split("\n", 1)[-1]
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1:
        try:
            return json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            pass
    return {}
