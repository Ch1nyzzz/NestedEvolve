"""Prompt 构建：为 LLM 变异生成 system/user message。支持完整重写和 SEARCH/REPLACE 两种模式。"""

from __future__ import annotations

import re
from typing import Optional

from .population import Individual
from .strategy import StrategyParams
from .task_adapter import Task


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
        strength_desc = "微调参数和常数，保持算法结构不变"
    elif ms < 0.6:
        strength_desc = "适度修改算法逻辑，可以引入新的子步骤或优化现有步骤"
    else:
        strength_desc = "大胆尝试全新的算法思路或完全不同的实现策略"

    # === error_analysis ===
    analysis_section = ""
    if params.error_analysis > 0.5:
        analysis_section = """
## 分析要求
在写代码之前，请先简要分析:
1. 当前方案的主要瓶颈或不足
2. 你打算如何改进，预期效果
3. 然后给出改进后的代码
"""

    # === context 程序 ===
    context_section = ""
    if context_programs:
        context_section = "\n\n## 参考程序（灵感来源）\n"
        for i, prog in enumerate(context_programs):
            if params.show_context_scores > 0.5:
                context_section += f"\n### 程序 {i+1} (score: {prog.score:.6f}, 排名 #{i+1})\n```python\n{prog.code}\n```\n"
            else:
                context_section += f"\n### 程序 {i+1}\n```python\n{prog.code}\n```\n"

    # === crossover ===
    crossover_section = ""
    if second_parent is not None:
        crossover_section = f"""

## 第二个参考程序 (score: {second_parent.score:.6f})
请尝试结合两个程序的优点。
```python
{second_parent.code}
```
"""

    # === 输出格式指令 ===
    if use_diff:
        format_instruction = """要求：
1. 用 SEARCH/REPLACE 格式输出修改，可以有多个块
2. 格式如下（严格遵守）：
```
<<<< SEARCH
要替换的原始代码（必须精确匹配）
==== REPLACE
替换后的新代码
>>>>
```
3. 只修改需要改动的部分
4. 保持函数签名不变"""
    else:
        format_instruction = """要求：
1. 只输出 EVOLVE-BLOCK 区域内的完整代码
2. 代码必须是可执行的 Python
3. 用 ```python ... ``` 包裹输出
4. 保持函数签名不变，只修改实现"""

    # === skill context 注入 ===
    skill_section = ""
    if skill_context:
        from .skill_orchestrator import format_skill_context

        skill_section = "\n" + format_skill_context(skill_context) + "\n"

    user_message = f"""## 当前程序 (score: {parent.score:.6f})

```python
{parent.code}
```
{context_section}{crossover_section}{analysis_section}{skill_section}
## 变异指令

请基于上述程序进行改进。变异强度: **{ms:.2f}** — {strength_desc}

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
