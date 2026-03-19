"""LLM 种群语义分析器：代码 → 结构化语义特征向量。

低频更新的语义辅助状态。不是精确状态描述，而是 LLM 对种群代码的
粗粒度语义摘要。为 surrogate 提供数值统计量无法捕捉的定性信息。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

import numpy as np

from .llm_client import LLMClient
from .population import Population

# 预定义语义维度和类别
SEMANTIC_DIMENSIONS: dict[str, list[str]] = {
    "strategy_family": [
        "greedy_constructive",
        "mathematical_optimization",
        "heuristic_search",
        "hybrid",
        "other",
    ],
    "diversity_type": [
        "parameter_only",
        "structural",
        "mixed",
    ],
    "convergence_stage": [
        "early_exploration",
        "promising_direction",
        "local_optimum",
        "diminishing_returns",
    ],
    "bottleneck": [
        "solution_quality",
        "constraint_satisfaction",
        "algorithmic_approach",
        "parameter_tuning",
        "none_apparent",
    ],
}

# 总维度 = 5 + 3 + 4 + 5 = 17
SEMANTIC_DIM = sum(len(v) for v in SEMANTIC_DIMENSIONS.values())

ANALYSIS_PROMPT = """你是一个进化优化分析专家。请分析以下种群中 top 候选程序的代码，给出结构化语义判断。

{programs_section}

请用以下 JSON 格式回复（每个字段只能从给定选项中选择）：
{{
    "strategy_family": "greedy_constructive" | "mathematical_optimization" | "heuristic_search" | "hybrid" | "other",
    "diversity_type": "parameter_only" | "structural" | "mixed",
    "convergence_stage": "early_exploration" | "promising_direction" | "local_optimum" | "diminishing_returns",
    "bottleneck": "solution_quality" | "constraint_satisfaction" | "algorithmic_approach" | "parameter_tuning" | "none_apparent"
}}

只输出 JSON，不要其他内容。"""


def _truncate_code(code: str, max_lines: int = 80, keep_lines: int = 30) -> str:
    """如果代码超过 max_lines 行，截断中间保留首尾。"""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    head = lines[:keep_lines]
    tail = lines[-keep_lines:]
    return "\n".join(
        head + [f"\n# ... 省略 {len(lines) - 2 * keep_lines} 行 ...\n"] + tail
    )


def _extract_signatures(code: str) -> list[str]:
    """提取函数/类签名。"""
    sigs = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            sigs.append(stripped)
    return sigs


def _labels_to_onehot(labels: dict[str, str]) -> np.ndarray:
    """语义标签 → one-hot 向量。"""
    vec = []
    for dim_name, categories in SEMANTIC_DIMENSIONS.items():
        value = labels.get(dim_name, categories[-1])  # 默认最后一个
        onehot = [0.0] * len(categories)
        if value in categories:
            onehot[categories.index(value)] = 1.0
        else:
            onehot[-1] = 1.0  # fallback
        vec.extend(onehot)
    return np.array(vec, dtype=np.float32)


class SemanticAnalyzer:
    """低频更新的语义辅助状态。"""

    def __init__(self, llm: LLMClient, analyze_interval: int = 5):
        self.llm = llm
        self.analyze_interval = analyze_interval
        self.cache: dict[tuple[int, str], np.ndarray] = {}
        self._last_by_task: dict[int, np.ndarray] = {}

    @staticmethod
    def _population_fingerprint(population: Population) -> str:
        """基于种群 top-5 的 id + score 生成指纹。"""
        top5 = population.top_k(5)
        data = "|".join(f"{ind.id}:{ind.score:.6f}" for ind in top5)
        return hashlib.md5(data.encode()).hexdigest()[:12]

    async def analyze(
        self,
        population: Population,
        task_id: int,
        global_sem: "asyncio.Semaphore | None" = None,
    ) -> np.ndarray:
        """用 LLM 分析种群 top-K 程序，返回语义特征向量 (17 维)。"""
        fp = self._population_fingerprint(population)
        cache_key = (task_id, fp)

        if cache_key in self.cache:
            return self.cache[cache_key]

        top_programs = population.top_k(5)
        if not top_programs:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)

        # 构建分析 prompt
        programs_section = ""
        for i, ind in enumerate(top_programs):
            code = _truncate_code(ind.code)
            sigs = _extract_signatures(ind.code)
            sig_line = "函数签名: " + ", ".join(sigs) if sigs else ""
            programs_section += (
                f"\n[程序 {i+1} (score: {ind.score:.6f})]\n"
                f"```python\n{code}\n```\n{sig_line}\n"
            )

        prompt = ANALYSIS_PROMPT.format(programs_section=programs_section)

        async def _do_generate():
            return await self.llm.generate(
                system_msg="你是进化优化分析专家。只输出 JSON。",
                user_msg=prompt,
                temperature=0.1,
            )

        try:
            if global_sem is not None:
                async with global_sem:
                    response = await _do_generate()
            else:
                response = await _do_generate()
            # 提取 JSON
            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if json_match:
                labels = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            vec = _labels_to_onehot(labels)
            self.cache[cache_key] = vec
            self._last_by_task[task_id] = vec
            return vec

        except Exception as e:
            print(f"  [semantic] 分析失败: {e}")
            return self.get_cached(task_id)

    def get_cached(self, task_id: int) -> np.ndarray:
        """返回该 task_id 最近的缓存语义向量，若无缓存返回全零。"""
        return self._last_by_task.get(task_id, np.zeros(SEMANTIC_DIM, dtype=np.float32))
