"""内层进化循环：streaming pipeline + island model。

保持 max_in_flight 个 worker 恒定在飞。
谁先完成就更新 population，立刻补一个新 worker（能看到最新种群）。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .llm_client import LLMClient
from .population import Individual, Population
from .prompt_builder import build_mutation_prompt, extract_code
from .state import extract_numeric_state
from .strategy import StrategyParams
from .task_adapter import Task, assemble_program

SEMANTIC_STATE_DIM = 17
_MAX_IN_FLIGHT = 5

# 代码去重：(task_name, code_hash) → score
_eval_cache: dict[tuple[str, int], float] = {}
_CACHE_MAX = 2000


@dataclass
class EvolveResult:
    final_best_score: float
    initial_state: np.ndarray
    final_state: np.ndarray
    final_population: Population
    score_trajectory: list[float] = field(default_factory=list)
    state_trajectory: list[np.ndarray] = field(default_factory=list)
    transitions: list[tuple[np.ndarray, np.ndarray, int, float]] = field(
        default_factory=list
    )
    step_records: list[dict] = field(default_factory=list)


def _individual_fitness_label(score: float, pre_mean: float, success: bool) -> float:
    """基于 pre-update stats 计算 label。"""
    if not success:
        return -0.1
    return score - pre_mean


def _syntax_check(code: str) -> str | None:
    try:
        compile(code, "<check>", "exec")
        return None
    except SyntaxError as e:
        return f"syntax: {e.msg} line {e.lineno}"


@dataclass
class _WorkerResult:
    """单个 worker 的结果 + 发射时的快照。"""

    code: str | None
    score: float
    error: str | None
    island_id: int
    # 发射时记录的 pre-update 快照
    pre_best: float
    pre_mean: float
    pre_state: np.ndarray


async def _mutation_worker(
    llm: LLMClient,
    task: Task,
    population: Population,
    params: StrategyParams,
    island_id: int,
    global_sem: asyncio.Semaphore | None = None,
    rng: np.random.RandomState | None = None,
    skill_context: dict | None = None,
) -> _WorkerResult:
    """单个变异 worker。发射时快照 population 状态。

    rng: 可选的固定随机数生成器，用于 common random numbers。
    """
    _rng = rng if rng is not None else np.random

    # --- 发射时快照 ---
    pre_best = population.best().score if population.best() else 0.0
    pre_mean = float(population.scores().mean()) if population.size() > 0 else 0.0
    pre_state = extract_numeric_state(population)

    parent = population.select_parent(
        params.selection_temperature, params.exploration_rate, island_id, _rng
    )
    context = population.select_context(
        params.context_size, params.diversity_weight, island_id
    )

    # crossover
    second_parent = None
    if _rng.random() < params.crossover_rate and population.size() >= 2:
        second_parent = population.select_parent(
            params.selection_temperature, 1.0 - params.parent_best_bias, island_id, _rng
        )
        if second_parent.id == parent.id:
            second_parent = None

    sys_msg, user_msg = build_mutation_prompt(
        task, parent, context, params, second_parent, skill_context=skill_context
    )
    use_diff = params.diff_vs_rewrite < 0.5
    temperature = params.llm_temperature
    max_tokens = params.get_max_tokens()

    def _make_result(code, score, error):
        return _WorkerResult(
            code=code,
            score=score,
            error=error,
            island_id=island_id,
            pre_best=pre_best,
            pre_mean=pre_mean,
            pre_state=pre_state,
        )

    try:
        if global_sem is not None:
            async with global_sem:
                response = await llm.generate(
                    sys_msg, user_msg, temperature=temperature, max_tokens=max_tokens
                )
        else:
            response = await llm.generate(
                sys_msg, user_msg, temperature=temperature, max_tokens=max_tokens
            )
        code = extract_code(response, parent_code=parent.code, use_diff=use_diff)
    except Exception as e:
        return _make_result(None, 0.0, str(e))

    if code is None:
        return _make_result(None, 0.0, "no_code")

    # 代码去重
    cache_key = (task.name, hash(code))
    if cache_key in _eval_cache:
        return _make_result(code, _eval_cache[cache_key], None)

    full_program = assemble_program(code, task.full_initial_code)

    # 语法预检
    syntax_err = _syntax_check(full_program)
    if syntax_err is not None:
        return _make_result(None, 0.0, syntax_err)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(full_program)
        tmp_path = f.name

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, task.evaluate_fn, tmp_path)
        score = result.get("combined_score", 0.0)
        error = result.get("error")
    except Exception as e:
        score = 0.0
        error = str(e)
    finally:
        os.unlink(tmp_path)

    if len(_eval_cache) < _CACHE_MAX:
        _eval_cache[cache_key] = score

    return _make_result(code, score, error)


async def run_inner_loop(
    task: Task,
    params: StrategyParams,
    n_steps: int = 20,
    llm: LLMClient | None = None,
    population: Population | None = None,
    semantic_analyzer=None,
    max_pop_size: int = 30,
    analyze_interval: int = 5,
    max_in_flight: int = _MAX_IN_FLIGHT,
    global_sem: asyncio.Semaphore | None = None,
    rng_seed: int | None = None,
    skill_context: dict | None = None,
) -> EvolveResult:
    """Streaming pipeline 进化循环。

    保持 max_in_flight 个 worker 恒定在飞。
    每完成一个 → 更新 population → 立刻补一个新 worker（看到最新种群）。
    """
    if llm is None:
        llm = LLMClient()

    num_islands = params.num_islands
    migration_interval = max(3, n_steps // (num_islands + 1))

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

    phi_vec = params.to_vector()
    semantic_state = np.zeros(SEMANTIC_STATE_DIM, dtype=np.float32)

    numeric_state = extract_numeric_state(population)
    if semantic_analyzer is not None:
        try:
            semantic_state = await semantic_analyzer.analyze(
                population, task.task_id, global_sem
            )
        except Exception:
            semantic_state = semantic_analyzer.get_cached(task.task_id)
    initial_state = np.concatenate([numeric_state, semantic_state])

    score_trajectory = []
    state_trajectory = []
    transitions = []
    step_records = []

    # --- streaming: 发满 → 完成一个补一个 ---
    next_worker_id = 0  # 下一个要发射的 worker 编号
    completed = 0  # 已完成的 worker 数
    in_flight: set[asyncio.Task] = set()
    semantic_counter = 0

    # common random numbers: 每个 worker 用确定性 rng（如果指定了 seed）
    base_seed = rng_seed

    def _launch_one() -> asyncio.Task | None:
        nonlocal next_worker_id
        if next_worker_id >= n_steps:
            return None
        island_id = next_worker_id % num_islands
        worker_rng = None
        if base_seed is not None:
            worker_rng = np.random.RandomState(base_seed + next_worker_id)
        t = asyncio.create_task(
            _mutation_worker(
                llm,
                task,
                population,
                params,
                island_id,
                global_sem,
                worker_rng,
                skill_context=skill_context,
            )
        )
        next_worker_id += 1
        return t

    # 发满初始 batch
    for _ in range(min(max_in_flight, n_steps)):
        t = _launch_one()
        if t:
            in_flight.add(t)

    while in_flight:
        # 等任意一个完成
        done, in_flight = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED
        )

        for finished_task in done:
            r: _WorkerResult = finished_task.result()
            completed += 1

            # 语义分析（周期性）
            if (
                semantic_analyzer is not None
                and semantic_counter % analyze_interval == 0
            ):
                try:
                    semantic_state = await semantic_analyzer.analyze(
                        population, task.task_id, global_sem
                    )
                except Exception:
                    semantic_state = semantic_analyzer.get_cached(task.task_id)
            semantic_counter += 1

            s_t = np.concatenate([r.pre_state, semantic_state])
            state_trajectory.append(s_t)

            success = (
                r.error is None
                and r.code is not None
                and np.isfinite(r.score)
                and r.score > 0
            )
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
                )
                population.add(new_ind)
                population.prune(max_pop_size, params.elite_ratio, num_islands)
                cur_best = population.best().score
                print(
                    f"  [{completed}/{n_steps}] score={r.score:.6f} (best={cur_best:.6f}) [island {r.island_id}]"
                )
            else:
                print(f"  [{completed}/{n_steps}] 失败: {(r.error or '?')[:80]}")

            # 迁移
            if num_islands > 1 and completed % migration_interval == 0:
                population.migrate(params.migration_rate, num_islands)

            # transition label 基于 pre-update stats
            fitness_label = _individual_fitness_label(r.score, r.pre_mean, success)
            transitions.append(
                (phi_vec.copy(), s_t.copy(), task.task_id, fitness_label)
            )

            new_best = population.best().score if population.best() else 0.0
            score_trajectory.append(new_best)

            # 记录 step record
            step_records.append(
                {
                    "step": completed,
                    "score": r.score,
                    "best_so_far": new_best,
                    "error": r.error,
                    "island_id": r.island_id,
                    "success": success,
                }
            )

            # 立刻补一个新 worker（看到最新 population）
            new_task = _launch_one()
            if new_task:
                in_flight.add(new_task)

    final_numeric = extract_numeric_state(population)
    final_state = np.concatenate([final_numeric, semantic_state])

    return EvolveResult(
        final_best_score=population.best().score if population.best() else 0.0,
        initial_state=initial_state,
        final_state=final_state,
        final_population=population,
        score_trajectory=score_trajectory,
        state_trajectory=state_trajectory,
        transitions=transitions,
        step_records=step_records,
    )
