"""Optimizer — LLM 生成 SEARCH/REPLACE diff patch。"""

from __future__ import annotations

import json

from utils.llm import llm_call
from noa.core.protocol import SystemDescription, Diagnosis, DeltaPatch
from noa.diff_utils import extract_diffs
from noa.core import prompts


def optimize(
    sys_desc: SystemDescription,
    diagnosis: Diagnosis,
    model: str = "gpt-4.1-mini",
    past_attempts: str = "",
) -> DeltaPatch:
    """基于诊断结果生成 SEARCH/REPLACE diff patch。"""
    diag_text = json.dumps(diagnosis.failure_patterns, indent=2, ensure_ascii=False)

    prompt = prompts.OPTIMIZER_PROMPT.format(
        system_context=sys_desc.to_context_str(),
        source_code=sys_desc.get_source_context(),
        diagnosis=diag_text,
        past_attempts=past_attempts or "(none)",
    )

    resp = llm_call(
        prompt, model=model, max_tokens=16384,
        temperature=0, system=prompts.OPTIMIZER_SYSTEM,
    )

    # 解析 diff 块
    print(f"[Optimizer] Raw LLM response ({len(resp.text)} chars):\n{resp.text[:2000]}")
    diffs = extract_diffs(resp.text, sys_desc.source_files)

    # 解析 metadata JSON
    metadata = _parse_metadata(resp.text)

    return DeltaPatch(
        diffs=diffs,
        rationale=metadata.get("rationale", ""),
        target_patterns=metadata.get("target_patterns", []),
    )


def _parse_metadata(text: str) -> dict:
    """从 LLM 输出中提取 metadata JSON（在 diff 块之后）。"""
    # 尝试提取 ```json ... ``` 块
    if "```json" in text:
        start = text.rfind("```json")
        end = text.find("```", start + 7)
        if end != -1:
            block = text[start + 7:end].strip()
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
                    return json.loads(text[i:right + 1])
                except json.JSONDecodeError:
                    break
    return {}
