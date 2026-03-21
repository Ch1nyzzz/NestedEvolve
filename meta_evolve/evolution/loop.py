"""内层进化循环：batch iteration + island model。"""

from __future__ import annotations

import asyncio
import difflib
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..integrations.llm import LLMClient
from ..tasks.loader import Task, assemble_program
from .population import Individual, Population
from .prompting import build_mutation_prompt, extract_code
from .search_policies import apply_policy_overrides, select_context, select_parent
from .strategy import StrategyParams

_MAX_IN_FLIGHT = 8

# 代码去重：(task_name, code_hash) → score
_eval_cache: dict[tuple[str, int], float] = {}
_CACHE_MAX = 2000


@dataclass
class EvolveResult:
    final_best_score: float
    final_population: Population
    score_trajectory: list[float] = field(default_factory=list)
    step_records: list[dict] = field(default_factory=list)
    best_eval_details: dict = field(default_factory=dict)


def _syntax_check(code: str) -> str | None:
    try:
        compile(code, "<check>", "exec")
        return None
    except SyntaxError as e:
        return f"syntax: {e.msg} line {e.lineno}"


def _summarize_diff(parent_code: str, child_code: str | None, max_lines: int = 8) -> str | None:
    if not child_code:
        return None
    diff = list(
        difflib.unified_diff(
            parent_code.splitlines(),
            child_code.splitlines(),
            lineterm="",
            n=1,
        )
    )
    if not diff:
        return None
    return "\n".join(diff[:max_lines])


def _classify_outcome(
    score: float,
    parent_score: float,
    pre_best: float,
    success: bool,
    error: str | None,
    code_changed: bool,
) -> str:
    if error is not None:
        err = error.lower()
        if "syntax" in err:
            return "syntax_error"
        if "timeout" in err:
            return "timeout"
        return "runtime_error"
    if not success:
        return "invalid"
    if not code_changed:
        return "no_change"
    if score > pre_best + 1e-9:
        return "global_improvement"
    if score > parent_score + 1e-9:
        return "local_improvement"
    if score < parent_score - 1e-9:
        return "regression"
    return "no_effect"


@dataclass
class _WorkerResult:
    """单个 worker 的结果 + 发射时的快照。"""

    code: str | None
    score: float
    error: str | None
    island_id: int
    parent_id: str
    parent_score: float
    parent_code: str
    pre_best: float
    eval_details: dict | None = None


_MAX_RETRIES = 2


async def _mutation_worker(
    llm: LLMClient,
    task: Task,
    population: Population,
    params: StrategyParams,
    island_id: int,
    sem: asyncio.Semaphore | None = None,
    rng: np.random.RandomState | None = None,
    skill_context: dict | None = None,
    search_policy: dict | None = None,
) -> _WorkerResult:
    """单个变异 worker，带 error-aware retry + 可插拔搜索策略。"""
    _rng = rng if rng is not None else np.random
    policy = search_policy or {}

    # 应用 search_policy 的数值覆盖
    effective_params = apply_policy_overrides(params, policy) if policy else params

    pre_best = population.best().score if population.best() else float("-inf")
    parent = select_parent(population, policy, effective_params, island_id, _rng)
    context = select_context(population, policy, effective_params, island_id)

    second_parent = None
    xover_rate = policy.get("crossover_rate", effective_params.crossover_rate)
    if _rng.random() < xover_rate and population.size() >= 2:
        second_parent = select_parent(
            population, {"parent_selection": "random"}, effective_params, island_id, _rng
        )
        if second_parent.id == parent.id:
            second_parent = None

    # 额外 prompt 内容
    merged_skill_context = skill_context
    supplement = policy.get("prompt_supplement")
    if supplement:
        merged_skill_context = dict(skill_context or {})
        merged_skill_context.setdefault("positive", {})["_search_policy"] = {
            "formatted": f"### Search Policy Guidance\n{supplement}",
        }

    sys_msg, user_msg = build_mutation_prompt(
        task, parent, context, effective_params, second_parent,
        skill_context=merged_skill_context,
    )
    use_diff = effective_params.diff_vs_rewrite < 0.5
    temperature = effective_params.llm_temperature
    max_tokens = effective_params.get_max_tokens()

    def _make_result(code, score, error, eval_details=None):
        return _WorkerResult(
            code=code, score=score, error=error,
            island_id=island_id, parent_id=parent.id,
            parent_score=parent.score, parent_code=parent.code,
            pre_best=pre_best, eval_details=eval_details,
        )

    code = None
    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            retry_msg = user_msg
            if attempt > 0 and last_error:
                retry_msg = (
                    user_msg
                    + f"\n\n## Previous Attempt Failed\nError: {last_error}\nAvoid the same issue and regenerate improved code."
                )
            if sem is not None:
                async with sem:
                    response = await llm.generate(
                        sys_msg, retry_msg, temperature=temperature, max_tokens=max_tokens
                    )
            else:
                response = await llm.generate(
                    sys_msg, retry_msg, temperature=temperature, max_tokens=max_tokens
                )
            code = extract_code(response, parent_code=parent.code, use_diff=use_diff)
        except Exception as e:
            last_error = str(e)
            continue

        if code is None:
            last_error = "no_code"
            continue
        break

    if code is None:
        return _make_result(None, 0.0, last_error or "no_code")

    # 语法检查失败时也 retry（把 syntax error 注入 prompt 重试一次）
    full_program = assemble_program(code, task.full_initial_code)
    syntax_err = _syntax_check(full_program)
    if syntax_err is not None and last_error != "syntax_retried":
        try:
            retry_msg = (
                user_msg
                + f"\n\n## Previous Attempt Failed\nSyntax error: {syntax_err}\nFix the syntax issue and regenerate the complete code."
            )
            if sem is not None:
                async with sem:
                    response = await llm.generate(
                        sys_msg, retry_msg, temperature=temperature, max_tokens=max_tokens
                    )
            else:
                response = await llm.generate(
                    sys_msg, retry_msg, temperature=temperature, max_tokens=max_tokens
                )
            retry_code = extract_code(response, parent_code=parent.code, use_diff=use_diff)
            if retry_code:
                retry_full = assemble_program(retry_code, task.full_initial_code)
                if _syntax_check(retry_full) is None:
                    code = retry_code
                    full_program = retry_full
                    syntax_err = None
        except Exception:
            pass
    if syntax_err is not None:
        return _make_result(None, 0.0, syntax_err)

    cache_key = (task.name, hash(code))
    if cache_key in _eval_cache:
        return _make_result(code, _eval_cache[cache_key], None)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(full_program)
        tmp_path = f.name

    eval_details = None
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, task.evaluate_fn, tmp_path)
        score = result.get("combined_score", 0.0)
        error = result.get("error")
        eval_details = {k: v for k, v in result.items() if k not in ("combined_score", "error")}
    except Exception as e:
        score = 0.0
        error = str(e)
    finally:
        os.unlink(tmp_path)

    if len(_eval_cache) < _CACHE_MAX:
        _eval_cache[cache_key] = score

    return _make_result(code, score, error, eval_details=eval_details)


async def run_inner_loop(
    task: Task,
    params: StrategyParams,
    n_iterations: int = 20,
    batch_size: int = 8,
    llm: LLMClient | None = None,
    population: Population | None = None,
    max_pop_size: int = 30,
    skill_context: dict | None = None,
    iteration_label: str | None = None,
    iteration_offset: int = 0,
    search_policy: dict | None = None,
) -> EvolveResult:
    """Batch iteration 进化循环。

    每个 iteration 并行发 batch_size 个 candidate，全部完成后统一更新种群。
    """
    if llm is None:
        llm = LLMClient()

    num_islands = params.num_islands
    migration_interval = max(2, n_iterations // (num_islands + 1))
    sem = asyncio.Semaphore(_MAX_IN_FLIGHT)

    if population is None:
        population = Population()
        for isl in range(num_islands):
            ind = Individual(
                id=Population.make_id(),
                code=task.initial_code,
                score=task.baseline_score,
                generation=0,
                island_id=isl,
            )
            population.add(ind)

    score_trajectory = []
    step_records = []
    global_step = 0

    for iteration in range(n_iterations):
        # 并行发 batch_size 个 worker
        tasks = []
        for i in range(batch_size):
            island_id = i % num_islands
            t = _mutation_worker(
                llm, task, population, params, island_id,
                sem=sem, skill_context=skill_context,
                search_policy=search_policy,
            )
            tasks.append(t)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理 batch 结果，统一更新种群
        iter_successes = 0
        iter_best = population.best().score if population.best() else float("-inf")

        for r in results:
            global_step += 1
            if isinstance(r, Exception):
                print(f"  [iter {iteration+1}/{n_iterations}] worker exception: {r}")
                continue
            if r.error:
                print(f"    [worker-err] score={r.score:.4f} err={r.error[:200]}")

            success = (
                r.error is None
                and r.code is not None
                and np.isfinite(r.score)
            )
            code_changed = bool(r.code is not None and r.code != r.parent_code)
            admitted = False

            if success:
                gen = (
                    max(
                        (ind.generation for ind in population.individuals.values()),
                        default=0,
                    )
                    + 1
                )
                new_ind = Individual(
                    id=Population.make_id(),
                    code=r.code,
                    score=r.score,
                    generation=gen,
                    island_id=r.island_id,
                    eval_details=r.eval_details,
                )
                population.add(new_ind)
                admitted = True
                iter_successes += 1

            outcome_type = _classify_outcome(
                r.score, r.parent_score, r.pre_best,
                success, r.error, code_changed,
            )

            step_records.append({
                "step": global_step,
                "iteration": iteration + iteration_offset,
                "score": r.score,
                "best_so_far": population.best().score if population.best() else float("-inf"),
                "error": r.error,
                "island_id": r.island_id,
                "success": success,
                "parent_id": r.parent_id,
                "parent_score": r.parent_score,
                "parent_best": r.pre_best,
                "admitted": admitted,
                "outcome_type": outcome_type,
                "code_snippet": (r.code or r.parent_code)[:200],
                "diff_summary": _summarize_diff(r.parent_code, r.code),
                "code_changed": code_changed,
                "eval_details": r.eval_details,
            })

        # batch 结束后统一 prune
        population.prune(max_pop_size, params.elite_ratio, num_islands)

        # 迁移
        if num_islands > 1 and (iteration + 1) % migration_interval == 0:
            population.migrate(params.migration_rate, num_islands)

        cur_best = population.best().score if population.best() else float("-inf")
        score_trajectory.append(cur_best)

        n_failed = batch_size - iter_successes
        label = iteration_label or f"{iteration+1}/{n_iterations}"
        print(
            f"  [iter {label}] "
            f"best={cur_best:.6f} success={iter_successes}/{batch_size} "
            f"failed={n_failed}"
        )

    best_ind = population.best()
    return EvolveResult(
        final_best_score=best_ind.score if best_ind else float("-inf"),
        final_population=population,
        score_trajectory=score_trajectory,
        step_records=step_records,
        best_eval_details=best_ind.eval_details if best_ind else {},
    )
