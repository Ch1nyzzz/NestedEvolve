"""Benchmark 适配层：将 skydiscover benchmark 统一为 Task 接口。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

# baseline 缓存文件（带线程锁）
_BASELINE_CACHE_PATH = Path(__file__).resolve().parent / "baseline_cache.json"
_CACHE_LOCK = threading.Lock()

BENCHMARKS_ROOT = (
    Path(__file__).resolve().parent.parent / "skydiscover" / "benchmarks" / "math"
)

# Phase 1: 11 个单实例任务（5 train + 6 test，按字母序分配 task_id）
PHASE1_TASKS = sorted(
    [
        "circle_packing",
        "circle_packing_rect",
        "erdos_min_overlap",
        "first_autocorr_ineq",
        "heilbronn_triangle",
        "matmul",
        "second_autocorr_ineq",
        "signal_processing",
        "sums_diffs_finite_sets",
        "third_autocorr_ineq",
        "uncertainty_ineq",
    ]
)

EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


@dataclass
class Task:
    name: str
    task_id: int
    evaluate_fn: Callable[[str], dict]
    initial_code: str  # EVOLVE-BLOCK 内的可进化代码
    full_initial_code: str  # 完整初始程序（含 fixed 部分）
    system_prompt: str
    baseline_score: float
    eval_timeout: int


def _find_evaluator(bench_dir: Path) -> Path:
    """探测 evaluator 路径：优先根目录，否则子目录。"""
    root_eval = bench_dir / "evaluator.py"
    sub_eval = bench_dir / "evaluator" / "evaluator.py"
    if root_eval.exists():
        return root_eval
    if sub_eval.exists():
        return sub_eval
    raise FileNotFoundError(f"No evaluator found in {bench_dir}")


def _load_evaluator_fn(evaluator_path: Path, timeout: int) -> Callable[[str], dict]:
    """动态加载 evaluator 模块，返回 evaluate(program_path) 函数。"""
    spec = importlib.util.spec_from_file_location("evaluator", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def evaluate_with_timeout(program_path: str) -> dict:
        """在 subprocess 中运行 evaluator，加 timeout 保护。"""
        script = f"""
import sys, json
# jax 兼容: 新版移除了 jax.tree_map，补回来
try:
    import jax
    if not hasattr(jax, 'tree_map'):
        jax.tree_map = jax.tree_util.tree_map
except ImportError:
    pass
sys.path.insert(0, {str(evaluator_path.parent)!r})
import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", {str(evaluator_path)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.evaluate({"{program_path!r}"})
print(json.dumps(result))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script.replace("{program_path!r}", repr(program_path)))
            tmp = f.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return {"combined_score": 0.0, "error": proc.stderr[-500:]}
            import json

            for line in reversed(proc.stdout.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
            return {"combined_score": 0.0, "error": "no JSON output"}
        except subprocess.TimeoutExpired:
            return {"combined_score": 0.0, "error": "timeout"}
        except Exception as e:
            return {"combined_score": 0.0, "error": str(e)}
        finally:
            os.unlink(tmp)

    return evaluate_with_timeout


def _extract_evolve_block(code: str) -> str:
    """提取 EVOLVE-BLOCK 区域的代码。"""
    lines = code.split("\n")
    in_block = False
    block_lines = []
    for line in lines:
        if EVOLVE_START in line:
            in_block = True
            continue
        if EVOLVE_END in line:
            in_block = False
            continue
        if in_block:
            block_lines.append(line)
    if not block_lines:
        raise ValueError("No EVOLVE-BLOCK found in initial program")
    return "\n".join(block_lines)


def _get_fixed_code(full_code: str) -> str:
    """提取 EVOLVE-BLOCK 外的 fixed 代码部分。"""
    lines = full_code.split("\n")
    fixed_lines = []
    in_block = False
    for line in lines:
        if EVOLVE_START in line:
            in_block = True
            fixed_lines.append("{EVOLVE_BLOCK}")
            continue
        if EVOLVE_END in line:
            in_block = False
            continue
        if not in_block:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def assemble_program(evolve_code: str, full_initial_code: str) -> str:
    """将 EVOLVE-BLOCK 代码拼回完整程序。"""
    fixed = _get_fixed_code(full_initial_code)
    block = f"{EVOLVE_START}\n{evolve_code}\n{EVOLVE_END}"
    return fixed.replace("{EVOLVE_BLOCK}", block)


def _load_baseline_cache() -> dict:
    """读取 baseline 缓存（线程安全）。"""
    with _CACHE_LOCK:
        if _BASELINE_CACHE_PATH.exists():
            try:
                return json.loads(_BASELINE_CACHE_PATH.read_text())
            except json.JSONDecodeError:
                return {}
    return {}


def _save_baseline_cache(cache: dict):
    """写入 baseline 缓存（线程安全，原子写）。"""
    with _CACHE_LOCK:
        tmp = _BASELINE_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2))
        tmp.replace(_BASELINE_CACHE_PATH)


def load_task(benchmark_name: str, task_id: int) -> Task:
    """加载单个 benchmark 为 Task（baseline 带缓存）。"""
    bench_dir = BENCHMARKS_ROOT / benchmark_name
    if not bench_dir.exists():
        raise FileNotFoundError(f"Benchmark {benchmark_name} not found at {bench_dir}")

    # 读取 config
    config_path = bench_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    system_prompt = config.get("prompt", {}).get("system_message", "")
    eval_timeout = config.get("evaluator", {}).get("timeout", 360)

    # 读取初始程序
    init_path = bench_dir / "initial_program.py"
    full_code = init_path.read_text()
    evolve_code = _extract_evolve_block(full_code)

    # 加载 evaluator
    evaluator_path = _find_evaluator(bench_dir)
    evaluate_fn = _load_evaluator_fn(evaluator_path, eval_timeout)

    # baseline: 优先读缓存，miss 时 eval 后原子合并写入
    cache = _load_baseline_cache()
    if benchmark_name in cache:
        baseline_score = cache[benchmark_name]
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(full_code)
            tmp_path = f.name
        try:
            result = evaluate_fn(tmp_path)
            baseline_score = result.get("combined_score", 0.0)
        finally:
            os.unlink(tmp_path)
        # 加锁：重新读最新缓存 → 合并 → 写入（避免覆盖其他线程的写入）
        with _CACHE_LOCK:
            fresh = {}
            if _BASELINE_CACHE_PATH.exists():
                try:
                    fresh = json.loads(_BASELINE_CACHE_PATH.read_text())
                except json.JSONDecodeError:
                    pass
            fresh[benchmark_name] = baseline_score
            tmp_f = _BASELINE_CACHE_PATH.with_suffix(".tmp")
            tmp_f.write_text(json.dumps(fresh, indent=2))
            tmp_f.replace(_BASELINE_CACHE_PATH)

    return Task(
        name=benchmark_name,
        task_id=task_id,
        evaluate_fn=evaluate_fn,
        initial_code=evolve_code,
        full_initial_code=full_code,
        system_prompt=system_prompt,
        baseline_score=baseline_score,
        eval_timeout=eval_timeout,
    )


def load_task_parallel(names: list[str], max_workers: int = 8) -> list[Task]:
    """并行加载多个 benchmark（baseline eval 用线程池并发）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    task_ids = {
        name: PHASE1_TASKS.index(name) if name in PHASE1_TASKS else 0 for name in names
    }
    results: dict[str, Task] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(load_task, name, task_ids[name]): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                errors[name] = str(e)

    for name, err in errors.items():
        print(f"[WARN] 跳过 {name}: {err}")

    # 保持传入顺序
    return [results[n] for n in names if n in results]


def load_all_tasks() -> list[Task]:
    """加载所有 Phase 1 benchmark，按字母序分配 task_id。"""
    return load_task_parallel(PHASE1_TASKS)


def split_tasks(
    tasks: list[Task], train_ratio: float = 0.75, seed: int = 42
) -> tuple[list[Task], list[Task]]:
    """按 train/test 划分任务。"""
    import random

    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]
