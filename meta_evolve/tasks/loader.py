"""Benchmark 适配层：自动发现并加载 benchmark，统一为 Task 接口。"""

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

from ..config import PACKAGE_ROOT, cache_path

# baseline 缓存文件（带线程锁）
_BASELINE_CACHE_PATH = cache_path("baseline_cache.json")
_CACHE_LOCK = threading.Lock()

# 默认 benchmark 根目录：meta_evolve/benchmarks
DEFAULT_BENCHMARKS_ROOT = PACKAGE_ROOT / "benchmarks"


def _discover_tasks(benchmarks_root: Path) -> list[str]:
    """自动发现 benchmarks_root 下所有有效 benchmark（含 initial_program.py）。"""
    tasks = []
    if not benchmarks_root.exists():
        return tasks
    for entry in sorted(benchmarks_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        # 顶层目录本身就是 benchmark（如 arc_benchmark）
        if (entry / "initial_program.py").exists():
            tasks.append(entry.name)
        else:
            # 子目录结构（如 math/circle_packing）
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "initial_program.py").exists():
                    # 用 "category/name" 格式
                    tasks.append(f"{entry.name}/{sub.name}")
    return tasks


def get_benchmarks_root(root: str | Path | None = None) -> Path:
    """获取 benchmark 根目录，支持外部传入。"""
    if root is not None:
        p = Path(root).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Benchmarks root not found: {p}")
        return p
    return DEFAULT_BENCHMARKS_ROOT


def _resolve_bench_dir(benchmarks_root: Path, benchmark_name: str) -> Path:
    """将 benchmark_name 解析为实际目录路径。

    支持两种格式:
      - "circle_packing" → 在所有子目录中搜索
      - "math/circle_packing" → 直接定位
    """
    # 带 "/" 的直接定位
    if "/" in benchmark_name:
        bench_dir = benchmarks_root / benchmark_name
        if bench_dir.exists():
            return bench_dir
        raise FileNotFoundError(f"Benchmark {benchmark_name} not found at {bench_dir}")

    # 先检查顶层
    top = benchmarks_root / benchmark_name
    if top.exists() and (top / "initial_program.py").exists():
        return top

    # 搜索子目录
    for category in sorted(benchmarks_root.iterdir()):
        if not category.is_dir():
            continue
        candidate = category / benchmark_name
        if candidate.exists() and (candidate / "initial_program.py").exists():
            return candidate

    raise FileNotFoundError(
        f"Benchmark '{benchmark_name}' not found under {benchmarks_root}"
    )


# 兼容旧代码：PHASE1_TASKS 改为动态发现
def _get_phase1_tasks() -> list[str]:
    return _discover_tasks(DEFAULT_BENCHMARKS_ROOT)


PHASE1_TASKS = _get_phase1_tasks()

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
    evaluator_source: str = ""  # 评估器源码，供 skill 生成器理解评分逻辑


def _find_evaluator(bench_dir: Path) -> Path:
    """探测 evaluator 路径：优先根目录，否则子目录。"""
    root_eval = bench_dir / "evaluator.py"
    sub_eval = bench_dir / "evaluator" / "evaluator.py"
    if root_eval.exists():
        return root_eval
    if sub_eval.exists():
        return sub_eval
    raise FileNotFoundError(f"No evaluator found in {bench_dir}")


_WORKER_SCRIPT = '''\
import sys, json, os, signal, io

# jax 兼容
try:
    import jax
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
except ImportError:
    pass

evaluator_path = sys.argv[1]
sys.path.insert(0, os.path.dirname(evaluator_path))

import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["evaluator"] = mod
spec.loader.exec_module(mod)

# 保存真实 stdout 用于 JSON 通信，eval 期间压制被评估代码的 print
_real_stdout = sys.stdout

# 就绪信号
_real_stdout.write("READY\\n")
_real_stdout.flush()

# 循环接收 program_path，eval，返回 JSON
for line in sys.stdin:
    program_path = line.strip()
    if not program_path:
        continue
    try:
        sys.stdout = io.StringIO()  # 压制被评估代码的 print
        result = mod.evaluate(program_path)
        sys.stdout = _real_stdout
        _real_stdout.write(json.dumps(result) + "\\n")
    except Exception as e:
        sys.stdout = _real_stdout
        _real_stdout.write(json.dumps({"combined_score": 0.0, "error": str(e)}) + "\\n")
    _real_stdout.flush()
'''


class _EvalWorker:
    """常驻 evaluator subprocess，启动时加载一次 evaluator，之后复用。"""

    def __init__(self, evaluator_path: Path, timeout: int):
        self._evaluator_path = str(evaluator_path)
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_alive(self):
        """确保 worker 存活，挂了就重启。"""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._start()

    def _start(self):
        """启动 worker subprocess。"""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER_SCRIPT, self._evaluator_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # 等待 READY 信号
        try:
            ready = self._proc.stdout.readline().strip()
            if ready != "READY":
                raise RuntimeError(f"Worker failed to start: {ready}")
        except Exception:
            self._proc.kill()
            self._proc = None
            raise

    def evaluate(self, program_path: str) -> dict:
        """发送 program_path 给 worker，读取 JSON 结果。"""
        with self._lock:
            self._ensure_alive()
            try:
                self._proc.stdin.write(program_path + "\n")
                self._proc.stdin.flush()

                # 带 timeout 读结果
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(self._proc.stdout, selectors.EVENT_READ)
                events = sel.select(timeout=self._timeout)
                sel.close()

                if not events:
                    # 超时：杀掉 worker（下次调用会自动重启）
                    self._proc.kill()
                    self._proc = None
                    return {"combined_score": 0.0, "error": "timeout"}

                line = self._proc.stdout.readline().strip()
                if not line:
                    self._proc.kill()
                    self._proc = None
                    return {"combined_score": 0.0, "error": "worker died"}
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    # stdout 被污染（被评估程序可能有 print），跳过非 JSON 行
                    for _ in range(5):
                        extra = self._proc.stdout.readline().strip()
                        if extra:
                            try:
                                return json.loads(extra)
                            except json.JSONDecodeError:
                                continue
                    return {"combined_score": 0.0, "error": f"non-json output: {line[:200]}"}
            except Exception as e:
                # worker 异常，杀掉让下次重启
                if self._proc is not None:
                    self._proc.kill()
                    self._proc = None
                return {"combined_score": 0.0, "error": str(e)}

    def shutdown(self):
        """关闭 worker。"""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None


class _EvalWorkerPool:
    """多 worker 池，支持并发评估。"""

    def __init__(self, evaluator_path: Path, timeout: int, size: int = 4):
        self._workers = [_EvalWorker(evaluator_path, timeout) for _ in range(size)]
        self._sem = threading.Semaphore(size)
        self._idx = 0
        self._idx_lock = threading.Lock()

    def evaluate(self, program_path: str) -> dict:
        self._sem.acquire()
        try:
            with self._idx_lock:
                idx = self._idx % len(self._workers)
                self._idx += 1
            return self._workers[idx].evaluate(program_path)
        finally:
            self._sem.release()

    def shutdown(self):
        for w in self._workers:
            w.shutdown()


def _load_evaluator_fn(evaluator_path: Path, timeout: int) -> Callable[[str], dict]:
    """返回 evaluate(program_path) 函数。

    使用常驻 worker subprocess 池：启动时加载一次 evaluator，
    后续复用，保留进程隔离同时去掉重复启动开销。
    """
    pool = _EvalWorkerPool(evaluator_path, timeout, size=4)
    return pool.evaluate


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
        _write_baseline_cache(cache)


def _write_baseline_cache(cache: dict):
    """原子写 baseline 缓存。调用方负责加锁。"""
    tmp = _BASELINE_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(_BASELINE_CACHE_PATH)


def load_task(
    benchmark_name: str,
    task_id: int = 0,
    benchmarks_root: Path | None = None,
) -> Task:
    """加载单个 benchmark 为 Task（baseline 带缓存）。"""
    root = get_benchmarks_root(benchmarks_root)
    bench_dir = _resolve_bench_dir(root, benchmark_name)

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
    evaluator_source = evaluator_path.read_text()

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
            _write_baseline_cache(fresh)

    return Task(
        name=benchmark_name,
        task_id=task_id,
        evaluate_fn=evaluate_fn,
        initial_code=evolve_code,
        full_initial_code=full_code,
        system_prompt=system_prompt,
        baseline_score=baseline_score,
        eval_timeout=eval_timeout,
        evaluator_source=evaluator_source,
    )


def load_task_parallel(
    names: list[str],
    max_workers: int = 8,
    benchmarks_root: Path | None = None,
) -> list[Task]:
    """并行加载多个 benchmark（baseline eval 用线程池并发）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, Task] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(load_task, name, i, benchmarks_root): name
            for i, name in enumerate(names)
        }
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


def load_all_tasks(benchmarks_root: Path | None = None) -> list[Task]:
    """自动发现并加载所有 benchmark。"""
    root = get_benchmarks_root(benchmarks_root)
    names = _discover_tasks(root)
    return load_task_parallel(names, benchmarks_root=root)


def discover_tasks(benchmarks_root: str | Path | None = None) -> list[str]:
    """公开接口：发现指定目录下所有可用 benchmark 名称。"""
    root = get_benchmarks_root(benchmarks_root)
    return _discover_tasks(root)


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
