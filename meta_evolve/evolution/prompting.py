"""Prompt 构建：为 LLM 变异生成 system/user message。支持完整重写和 SEARCH/REPLACE 两种模式。"""

from __future__ import annotations

import re
from typing import Optional

from ..skills.context import format_skill_context
from ..tasks.loader import Task
from .population import Individual
from .strategy import StrategyParams


def build_mutation_prompt(
    task: Task,
    parent: Individual,
    context_programs: list[Individual],
    params: StrategyParams,
    second_parent: Individual | None = None,
    skill_context: dict | None = None,
) -> tuple[str, str]:
    """构建变异 prompt → (system_message, user_message)。"""

    use_diff = params.diff_vs_rewrite < 0.5

    # === system message ===
    if params.domain_hint < 0.3:
        system_message = "You are an expert programmer. Improve the given code to maximize its score."
    elif params.domain_hint < 0.7:
        system_message = (
            task.system_prompt
            or "You are an expert programmer. Improve the given code to maximize its score."
        )
    else:
        system_message = task.system_prompt or ""
        system_message += "\n\nThink step by step about the mathematical and algorithmic properties of this problem before writing code."

    # === 变异强度描述 ===
    ms = params.mutation_strength
    if ms < 0.3:
        strength_desc = "Fine-tune parameters and constants, keep the algorithm structure unchanged"
    elif ms < 0.6:
        strength_desc = "Moderately modify algorithm logic, introduce new sub-steps or optimize existing ones"
    else:
        strength_desc = "Boldly try a completely new algorithm approach or entirely different implementation strategy"

    # === error_analysis ===
    analysis_section = ""
    if params.error_analysis > 0.5:
        analysis_section = """
## Analysis Requirements
Before writing code, briefly analyze:
1. The main bottlenecks or weaknesses of the current approach
2. How you plan to improve it and the expected effect
3. Then provide the improved code
"""

    # === context 程序 ===
    context_section = ""
    if context_programs:
        context_section = "\n\n## Reference Programs (for inspiration)\n"
        for i, prog in enumerate(context_programs):
            if params.show_context_scores > 0.5:
                context_section += f"\n### Program {i+1} (score: {prog.score:.6f}, rank #{i+1})\n```python\n{prog.code}\n```\n"
            else:
                context_section += f"\n### Program {i+1}\n```python\n{prog.code}\n```\n"

    # === crossover ===
    crossover_section = ""
    if second_parent is not None:
        crossover_section = f"""

## Second Reference Program (score: {second_parent.score:.6f})
Try combining the strengths of both programs.
```python
{second_parent.code}
```
"""

    # === 输出格式指令 ===
    if use_diff:
        format_instruction = """Requirements:
1. Output changes in SEARCH/REPLACE format, multiple blocks allowed
2. Format (strictly follow):
```
<<<< SEARCH
original code to replace (must match exactly)
==== REPLACE
new replacement code
>>>>
```
3. Only modify the parts that need changing
4. Keep function signatures unchanged"""
    else:
        format_instruction = """Requirements:
1. Output only the complete code inside the EVOLVE-BLOCK region
2. Code must be executable Python
3. Wrap output in ```python ... ```
4. Keep function signatures unchanged, only modify the implementation"""

    # === eval details 注入 ===
    eval_section = ""
    if parent.eval_details:
        detail_lines = [f"- {k}: {v}" for k, v in parent.eval_details.items()
                        if not isinstance(v, (dict, list)) or len(str(v)) < 200]
        if detail_lines:
            eval_section = "\n## Evaluation Details\n" + "\n".join(detail_lines) + "\n"

    # === skill context 注入 ===
    skill_section = ""
    if skill_context:
        skill_section = "\n" + format_skill_context(skill_context) + "\n"

    user_message = f"""## Current Program (score: {parent.score:.6f})

```python
{parent.code}
```
{eval_section}{context_section}{crossover_section}{analysis_section}{skill_section}
## Mutation Instructions

Improve the program above. Mutation strength: **{ms:.2f}** — {strength_desc}

{format_instruction}"""

    return system_message, user_message


def _syntax_ok(code: str) -> bool:
    """快速语法检查。"""
    try:
        compile(code, "<check>", "exec")
        return True
    except SyntaxError:
        return False


def extract_code(
    response: str, parent_code: str | None = None, use_diff: bool = False
) -> Optional[str]:
    """从 LLM 回复中提取代码。支持完整代码块和 SEARCH/REPLACE，自动 fallback。"""

    # 先尝试 SEARCH/REPLACE（仅在语法通过时采纳）
    if use_diff and parent_code is not None:
        result = _apply_search_replace(response, parent_code)
        if result is not None and _syntax_ok(result):
            return result

    # fallback: 完整代码块（过滤掉包含 SEARCH/REPLACE 标记的块）
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pat, response, re.DOTALL)
        # 过滤掉 SEARCH/REPLACE 块被 ``` 包裹的情况
        matches = [m for m in matches if "<<<< SEARCH" not in m and ">>>>" not in m]
        if matches:
            code = max(matches, key=len).strip()
            if _syntax_ok(code):
                return code

    # 再 fallback: SEARCH/REPLACE（不限 use_diff 标志）
    if parent_code is not None:
        result = _apply_search_replace(response, parent_code)
        if result is not None and _syntax_ok(result):
            return result

    return None


def _apply_search_replace(response: str, parent_code: str) -> Optional[str]:
    """解析并应用 SEARCH/REPLACE 块。"""
    pattern = r"<<<<\s*SEARCH\s*\n(.*?)\n=+\s*(?:REPLACE)?\s*\n(.*?)\n>>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return None

    result = parent_code
    applied = 0
    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
            applied += 1
        else:
            # 模糊匹配：去掉空白差异重试
            search_normalized = re.sub(r"\s+", " ", search.strip())
            lines = result.split("\n")
            for i, line in enumerate(lines):
                # 找连续匹配的起始行
                chunk = "\n".join(lines[i : i + search.count("\n") + 1])
                if re.sub(r"\s+", " ", chunk.strip()) == search_normalized:
                    result = result.replace(chunk, replace, 1)
                    applied += 1
                    break

    return result if applied > 0 else None
