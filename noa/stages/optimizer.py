"""Optimizer — LLM 生成 SEARCH/REPLACE diff patch（支持 agentic 多轮）。"""

from __future__ import annotations

import json
import logging

from utils.llm import llm_call, resolve_model
from noa.core.protocol import SystemDescription, Diagnosis, DeltaPatch
from noa.diff_utils import extract_diffs, apply_diffs_in_memory
from noa.core import prompts

log = logging.getLogger(__name__)


def optimize(
    sys_desc: SystemDescription,
    diagnosis: Diagnosis,
    model: str = resolve_model("gpt-4.1-mini"),
    past_attempts: str = "",
    layer_context: str = "",
    trajectory_samples: str = "",
    temperature: float = 0,
) -> DeltaPatch:
    """基于诊断结果生成 SEARCH/REPLACE diff patch，并进行自检评分。"""
    diag_text = json.dumps(diagnosis.failure_patterns, indent=2, ensure_ascii=False)

    prompt = prompts.OPTIMIZER_PROMPT.format(
        system_context=sys_desc.to_context_str(),
        source_code=sys_desc.get_source_context(),
        diagnosis=diag_text,
        past_attempts=past_attempts or "(none)",
        layer_context=layer_context,
        trajectory_samples=trajectory_samples or "(none)",
    )

    resp = llm_call(
        prompt,
        model=model,
        max_tokens=16384,
        temperature=temperature,
        system=prompts.OPTIMIZER_SYSTEM,
    )

    # 解析 diff 块
    print(f"[Optimizer] Raw LLM response ({len(resp.text)} chars):\n{resp.text[:2000]}")
    diffs = extract_diffs(resp.text, sys_desc.source_files)

    # 解析 metadata JSON
    metadata = _parse_metadata(resp.text)
    quality_score = _self_check_quality(
        sys_desc=sys_desc,
        diagnosis_text=diag_text,
        llm_output=resp.text,
        model=model,
    )

    return DeltaPatch(
        diffs=diffs,
        rationale=metadata.get("rationale", ""),
        target_patterns=metadata.get("target_patterns", []),
        quality_score=quality_score,
    )


def optimize_agentic(
    sys_desc: SystemDescription,
    diagnosis: Diagnosis,
    model: str = resolve_model("gpt-4.1-mini"),
    past_attempts: str = "",
    layer_context: str = "",
    max_tool_calls: int = 5,
    trajectory_samples: str = "",
    temperature: float = 0,
) -> DeltaPatch:
    """Agentic optimizer — 多轮 tool-calling 验证 SEARCH block 精确匹配。

    max_tool_calls=0 退化为 optimize()。
    """
    if max_tool_calls <= 0:
        return optimize(
            sys_desc,
            diagnosis,
            model,
            past_attempts,
            layer_context,
            trajectory_samples=trajectory_samples,
            temperature=temperature,
        )

    from noa.stages.agentic import agentic_loop

    diag_text = json.dumps(diagnosis.failure_patterns, indent=2, ensure_ascii=False)
    probe = _OptimizerProbe(sys_desc)

    user_content = prompts.OPTIMIZER_PROMPT.format(
        system_context=sys_desc.to_context_str(),
        source_code=sys_desc.get_source_context(),
        diagnosis=diag_text,
        past_attempts=past_attempts or "(none)",
        layer_context=layer_context,
        trajectory_samples=trajectory_samples or "(none)",
    )
    messages = [
        {"role": "system", "content": prompts.OPTIMIZER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    tools = probe.get_tool_schemas()

    final_text = agentic_loop(
        messages=messages,
        tools=tools,
        tool_executor=probe.execute_tool,
        model=model,
        max_tool_calls=max_tool_calls,
        max_tokens=16384,
        json_retries=0,
        budget_exhausted_prompt="Tool budget exhausted. Output your final patch now.",
    )

    diffs = extract_diffs(final_text, sys_desc.source_files)
    metadata = _parse_metadata(final_text)
    quality_score = _self_check_quality(
        sys_desc=sys_desc, diagnosis_text=diag_text, llm_output=final_text, model=model
    )

    return DeltaPatch(
        diffs=diffs,
        rationale=metadata.get("rationale", ""),
        target_patterns=metadata.get("target_patterns", []),
        quality_score=quality_score,
    )


class _OptimizerProbe:
    """Optimizer agentic 模式工具集 — 验证 SEARCH block、读取源文件、dry-run patch。"""

    def __init__(self, sys_desc: SystemDescription):
        self.sys_desc = sys_desc
        self._file_map = {sf.path: sf.content for sf in sys_desc.source_files}

    def get_tool_schemas(self) -> list[dict]:
        file_list = list(self._file_map.keys())
        return [
            {
                "type": "function",
                "function": {
                    "name": "verify_search_block",
                    "description": "Verify if a SEARCH text block exists verbatim in a source file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "enum": file_list},
                            "search_text": {"type": "string"},
                        },
                        "required": ["file_path", "search_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_source_file",
                    "description": "Read a source file content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "enum": file_list},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dry_run_patch",
                    "description": "Try applying a SEARCH/REPLACE diff in memory and report success/failure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "search_text": {"type": "string"},
                            "replace_text": {"type": "string"},
                        },
                        "required": ["file_path", "search_text", "replace_text"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "verify_search_block":
            fp = arguments.get("file_path", "")
            search = arguments.get("search_text", "")
            content = self._file_map.get(fp, "")
            if not content:
                return json.dumps({"found": False, "error": f"File {fp} not found"})
            found = search in content
            result = {"found": found, "file_path": fp}
            if not found:
                # 提供附近文本帮助 LLM 修正
                lines = content.split("\n")
                search_first_line = search.split("\n")[0].strip() if search else ""
                for i, line in enumerate(lines):
                    if search_first_line and search_first_line in line:
                        start = max(0, i - 1)
                        end = min(len(lines), i + len(search.split("\n")) + 2)
                        result["nearby_lines"] = "\n".join(lines[start:end])
                        result["nearby_start_line"] = start + 1
                        break
            return json.dumps(result, ensure_ascii=False)[:3000]

        elif tool_name == "read_source_file":
            fp = arguments.get("file_path", "")
            content = self._file_map.get(fp, "")
            if not content:
                return json.dumps({"error": f"File {fp} not found"})
            lines = content.split("\n")
            start = int(arguments.get("start_line", 1)) - 1
            end = int(arguments.get("end_line", len(lines)))
            selected = lines[max(0, start) : min(len(lines), end)]
            text = "\n".join(f"{i+start+1:4d} | {ln}" for i, ln in enumerate(selected))
            return text[:4000]

        elif tool_name == "dry_run_patch":
            fp = arguments.get("file_path", "")
            search = arguments.get("search_text", "")
            replace = arguments.get("replace_text", "")
            from noa.core.protocol import DiffBlock

            diff = DiffBlock(file_path=fp, search=search, replace=replace)
            modified = apply_diffs_in_memory(self.sys_desc.source_files, [diff])
            if modified:
                return json.dumps(
                    {"success": True, "files_modified": [sf.path for sf in modified]}
                )
            return json.dumps(
                {
                    "success": False,
                    "reason": "SEARCH block did not match any content in the file",
                }
            )

        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _parse_metadata(text: str) -> dict:
    """从 LLM 输出中提取 metadata JSON（在 diff 块之后）。"""
    # 尝试提取 ```json ... ``` 块
    if "```json" in text:
        start = text.rfind("```json")
        end = text.find("```", start + 7)
        if end != -1:
            block = text[start + 7 : end].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
    # 尝试找最后一个 { ... }
    right = text.rfind("}")
    if right != -1:
        # 从 right 向前找匹配的 {
        depth = 0
        for i in range(right, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i : right + 1])
                except json.JSONDecodeError:
                    break
    return {}


def _self_check_quality(
    *,
    sys_desc: SystemDescription,
    diagnosis_text: str,
    llm_output: str,
    model: str,
) -> float:
    """让 LLM 对 patch 可执行性做 0~1 自评。"""
    review_prompt = (
        "Review the proposed patch output and estimate patch executability quality from 0 to 1.\n"
        "Higher score means: search blocks likely match source, changes are minimal and targeted.\n\n"
        f"Diagnosis:\n{diagnosis_text}\n\n"
        f"Source summary:\n{sys_desc.to_context_str()}\n\n"
        f"Patch output:\n{llm_output[:8000]}\n\n"
        'Output strict JSON: {"quality_score": <float>, "comment": "..."}'
    )
    try:
        resp = llm_call(
            review_prompt,
            model=model,
            max_tokens=256,
            temperature=0,
            system="You are a strict patch reviewer.",
        )
        parsed = _parse_metadata(resp.text)
        score = float(parsed.get("quality_score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0
