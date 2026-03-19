"""通用适配层：将 φ 向量映射到任意进化框架的 config，统一接口。

目标框架只需实现 TargetSystem 协议（一个 run 方法），
meta_evolve 就能优化它。
"""

from __future__ import annotations

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


from .strategy import StrategyParams


# ============================================================
# 通用协议
# ============================================================


class TargetSystem(ABC):
    """任何可优化的进化框架都实现这个接口。"""

    @abstractmethod
    def run(self, config_overrides: dict, benchmark: str, n_steps: int) -> float:
        """跑进化，返回 best score。"""

    @abstractmethod
    def default_config(self) -> dict:
        """返回框架默认配置。"""

    async def run_segment(
        self,
        task,
        n_steps: int,
        skill_context: dict | None = None,
        population=None,
        initial_best_score: float | None = None,
    ):
        """分段运行，注入 skill 上下文。默认 fallback 到 run()。"""
        from .trajectory import SegmentResult

        score = self.run({}, task.name, n_steps)
        init_score = (
            task.baseline_score if initial_best_score is None else initial_best_score
        )
        return SegmentResult(
            best_score=score, initial_score=init_score, skill_context_used=False
        )


# ============================================================
# φ → 通用语义参数
# ============================================================


@dataclass
class UniversalParams:
    """框架无关的通用进化策略参数。
    从 StrategyParams 映射而来，再由各框架 adapter 转成框架 config。
    """

    # 选择
    exploration_rate: float = 0.3
    selection_pressure: float = 0.8  # 0=均匀 1=强偏好最优
    diversity_weight: float = 0.3
    elite_ratio: float = 0.2

    # 变异
    mutation_strength: float = 0.5
    llm_temperature: float = 0.7
    max_tokens: int = 4096
    diff_vs_rewrite: float = 0.3
    error_analysis: float = 0.5
    crossover_rate: float = 0.0

    # Prompt
    num_context: int = 3
    show_scores: float = 0.7
    domain_hint: float = 0.5

    # 结构
    num_islands: int = 3
    migration_rate: float = 0.2

    @staticmethod
    def from_strategy(p: StrategyParams) -> UniversalParams:
        """StrategyParams → UniversalParams (直接映射)。"""
        return UniversalParams(
            exploration_rate=p.exploration_rate,
            selection_pressure=1.0 - p.exploration_rate * (1.0 - p.parent_best_bias),
            diversity_weight=p.diversity_weight,
            elite_ratio=p.elite_ratio,
            mutation_strength=p.mutation_strength,
            llm_temperature=p.llm_temperature,
            max_tokens=p.get_max_tokens(),
            diff_vs_rewrite=p.diff_vs_rewrite,
            error_analysis=p.error_analysis,
            crossover_rate=p.crossover_rate,
            num_context=p.context_size,
            show_scores=p.show_context_scores,
            domain_hint=p.domain_hint,
            num_islands=p.num_islands,
            migration_rate=p.migration_rate,
        )


# ============================================================
# 框架 Adapters
# ============================================================


class NativeAdapter(TargetSystem):
    """适配我们自己的 evolve_loop（同进程，直接调用）。"""

    def __init__(
        self,
        llm_model: str = "together_ai/MiniMaxAI/MiniMax-M2.5",
        params: StrategyParams | None = None,
        llm=None,
    ):
        self.llm_model = llm_model
        self.params = params or StrategyParams()
        self._llm = llm  # 可复用的 LLM client

    def default_config(self) -> dict:
        return StrategyParams().__dict__

    def run(self, config_overrides: dict, benchmark: str, n_steps: int) -> float:
        import asyncio
        from .llm_client import LLMClient
        from .evolve_loop import run_inner_loop
        from .task_adapter import PHASE1_TASKS, load_task

        tid = PHASE1_TASKS.index(benchmark) if benchmark in PHASE1_TASKS else 0
        task = load_task(benchmark, tid)
        params = StrategyParams(
            **{
                k: v
                for k, v in config_overrides.items()
                if k in StrategyParams.__dataclass_fields__
            }
        )
        llm = LLMClient(model=self.llm_model)
        result = asyncio.run(
            run_inner_loop(task=task, params=params, n_steps=n_steps, llm=llm)
        )
        return result.final_best_score

    async def run_segment(
        self,
        task,
        n_steps: int,
        skill_context: dict | None = None,
        population=None,
        initial_best_score: float | None = None,
    ):
        """分段运行 evolve_loop，注入 skill 上下文。"""
        from .llm_client import LLMClient
        from .evolve_loop import run_inner_loop
        from .trajectory import SegmentResult, StepRecord

        llm = self._llm or LLMClient(model=self.llm_model)
        result = await run_inner_loop(
            task=task,
            params=self.params,
            n_steps=n_steps,
            llm=llm,
            population=population,
            skill_context=skill_context,
        )
        # 转换 step_records → StepRecord
        records = []
        initial_best = (
            task.baseline_score if initial_best_score is None else initial_best_score
        )
        prev_best = initial_best
        for rec in result.step_records:
            cur_best = rec.get("best_so_far", 0.0)
            records.append(
                StepRecord(
                    step=rec["step"],
                    score=rec["score"],
                    delta_score=cur_best - prev_best,
                    error_summary=rec.get("error"),
                )
            )
            prev_best = cur_best

        # 提取 population snapshot
        pop = result.final_population
        pop_snapshot = None
        island_stats = None
        if pop and pop.size() > 0:
            pop_snapshot = [
                {
                    "code": ind.code[:500],
                    "score": ind.score,
                    "island_id": ind.island_id,
                    "id": ind.id,
                }
                for ind in pop.all_sorted()
            ]
            # 按岛统计
            island_stats = {}
            for isl in pop.active_islands():
                members = pop.island_members(isl)
                scores = [m.score for m in members]
                island_stats[isl] = {
                    "size": len(members),
                    "best": max(scores),
                    "mean": sum(scores) / len(scores),
                }

        # 提取错误详情（从原始 step_records 获取更多信息）
        error_details = []
        for raw in result.step_records:
            if raw.get("error"):
                error_details.append(
                    {
                        "step": raw["step"],
                        "error_full": raw["error"],
                        "island_id": raw.get("island_id"),
                        "score": raw.get("score", 0.0),
                    }
                )

        seg = SegmentResult(
            best_score=result.final_best_score,
            initial_score=initial_best,
            trajectory=records,
            skill_context_used=skill_context is not None,
            population_snapshot=pop_snapshot,
            error_details=error_details or None,
            island_stats=island_stats,
        )
        # 附加 population 以便跨 segment 复用
        seg.population = result.final_population
        return seg


class OpenEvolveAdapter(TargetSystem):
    """适配 OpenEvolve（subprocess 调用）。"""

    def __init__(self, root: str | Path = "openevolve"):
        self.root = Path(root).resolve()

    def default_config(self) -> dict:
        return {
            "llm": {"temperature": 0.7, "top_p": None, "max_tokens": 4096},
            "database": {
                "population_size": 1000,
                "num_islands": 5,
                "elite_selection_ratio": 0.1,
                "exploration_ratio": 0.2,
                "exploitation_ratio": 0.7,
                "migration_interval": 50,
                "migration_rate": 0.1,
                "feature_bins": 10,
            },
            "prompt": {
                "num_top_programs": 3,
                "num_diverse_programs": 2,
                "use_template_stochasticity": True,
                "include_artifacts": True,
            },
            "evaluator": {"cascade_evaluation": True},
        }

    @staticmethod
    def phi_to_config(u: UniversalParams) -> dict:
        """UniversalParams → OpenEvolve config overrides。"""
        exploration = u.exploration_rate
        return {
            "llm": {
                "temperature": u.llm_temperature,
                "max_tokens": u.max_tokens,
            },
            "database": {
                "elite_selection_ratio": u.elite_ratio,
                "exploration_ratio": min(exploration, 0.4),
                "exploitation_ratio": max(1.0 - exploration - u.elite_ratio, 0.3),
                "num_islands": u.num_islands,
                "migration_rate": u.migration_rate,
            },
            "prompt": {
                "num_top_programs": u.num_context,
                "num_diverse_programs": max(1, int(u.num_context * u.diversity_weight)),
                "include_artifacts": u.error_analysis > 0.5,
            },
        }

    def run(self, config_overrides: dict, benchmark: str, n_steps: int) -> float:
        """用 subprocess 跑 OpenEvolve。"""
        bench_dir = self.root / "examples" / benchmark
        if not bench_dir.exists():
            raise FileNotFoundError(
                f"OpenEvolve benchmark {benchmark} not found at {bench_dir}"
            )

        # 写临时 config
        import tempfile

        cfg = self.default_config()
        _deep_update(cfg, config_overrides)
        cfg["max_iterations"] = n_steps

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            cfg_path = f.name

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openevolve",
                    str(bench_dir),
                    "--config",
                    cfg_path,
                ],
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(self.root),
            )
            return _parse_best_score(proc.stdout)
        except subprocess.TimeoutExpired:
            return 0.0
        finally:
            Path(cfg_path).unlink(missing_ok=True)


class AdaEvolveAdapter(TargetSystem):
    """适配 SkyDiscover AdaEvolve（subprocess 调用）。"""

    def __init__(self, root: str | Path = "skydiscover"):
        self.root = Path(root).resolve()

    def default_config(self) -> dict:
        return {
            "database": {
                "strategy": "adaevolve",
                "population_size": 20,
                "num_islands": 2,
                "decay": 0.9,
                "intensity_min": 0.15,
                "intensity_max": 0.5,
                "use_adaptive_search": True,
                "use_ucb_selection": True,
                "use_migration": True,
                "migration_interval": 15,
                "migration_count": 5,
                "archive_elite_ratio": 0.2,
                "pareto_weight": 0.4,
                "fitness_weight": 1.0,
                "novelty_weight": 0.0,
                "use_dynamic_islands": True,
                "use_paradigm_breakthrough": True,
            },
            "llm": {"temperature": 0.7, "max_tokens": 4096},
        }

    @staticmethod
    def phi_to_config(u: UniversalParams) -> dict:
        """UniversalParams → AdaEvolve config overrides。"""
        return {
            "database": {
                "num_islands": u.num_islands,
                "decay": 0.7 + 0.25 * u.selection_pressure,  # [0.7, 0.95]
                "intensity_min": 0.05
                + 0.25 * (1.0 - u.exploration_rate),  # 高探索→低 min
                "intensity_max": 0.3 + 0.5 * u.exploration_rate,  # 高探索→高 max
                "migration_interval": max(5, int(50 * (1.0 - u.migration_rate))),
                "migration_count": max(1, int(10 * u.migration_rate)),
                "archive_elite_ratio": u.elite_ratio,
                "fitness_weight": u.selection_pressure,
                "novelty_weight": u.diversity_weight * 0.5,
                "pareto_weight": 1.0 - u.selection_pressure,
                "use_paradigm_breakthrough": u.mutation_strength > 0.5,
            },
            "llm": {
                "temperature": u.llm_temperature,
                "max_tokens": u.max_tokens,
            },
        }

    def run(self, config_overrides: dict, benchmark: str, n_steps: int) -> float:
        bench_dir = self.root / "benchmarks" / "math" / benchmark
        if not bench_dir.exists():
            raise FileNotFoundError(
                f"AdaEvolve benchmark {benchmark} not found at {bench_dir}"
            )

        import tempfile

        cfg = self.default_config()
        _deep_update(cfg, config_overrides)
        cfg["max_iterations"] = n_steps

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            cfg_path = f.name

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "skydiscover",
                    "run",
                    str(bench_dir / "initial_program.py"),
                    str(bench_dir / "evaluator" / "evaluator.py"),
                    "-c",
                    str(bench_dir / "config.yaml"),
                    "--override",
                    cfg_path,
                    "-s",
                    "adaevolve",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(self.root),
            )
            return _parse_best_score(proc.stdout)
        except subprocess.TimeoutExpired:
            return 0.0
        finally:
            Path(cfg_path).unlink(missing_ok=True)


# ============================================================
# 工具函数
# ============================================================


def _deep_update(base: dict, overrides: dict):
    """递归合并 overrides 到 base。"""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _parse_best_score(stdout: str) -> float:
    """从框架 stdout 中解析 best score。"""
    best = 0.0
    for line in stdout.strip().split("\n"):
        # 常见格式: "best_score: 0.xxx" 或 "combined_score: 0.xxx"
        for key in ("best_score", "combined_score", "final_best"):
            if key in line:
                try:
                    val = float(line.split(key)[-1].strip().strip(":=").split()[0])
                    best = max(best, val)
                except (ValueError, IndexError):
                    pass
    return best


# ============================================================
# 注册表
# ============================================================

ADAPTERS: dict[str, type[TargetSystem]] = {
    "native": NativeAdapter,
    "openevolve": OpenEvolveAdapter,
    "adaevolve": AdaEvolveAdapter,
}


def get_adapter(name: str, **kwargs) -> TargetSystem:
    """按名字获取适配器。"""
    cls = ADAPTERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return cls(**kwargs)
