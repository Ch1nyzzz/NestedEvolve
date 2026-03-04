"""Subprocess Runner — 在隔离子进程中运行优化层，避免模块缓存问题。"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap


def run_layer_subprocess(
    noa_dir: str,
    project_root: str,
    target_source_dir: str,
    dataset_pickle_path: str,
    layer_level: int = 1,
    max_steps: int = 20,
    n_samples: int = 20,
    model: str = "gpt-4.1-mini",
    timeout: int = 1200,
    isolate_source: bool = False,
    eval_n_samples: int = 20,
    max_llm_calls: int = 80,
    max_evals: int = 12,
    max_no_improve_steps: int = 5,
    max_tool_calls: int = 10,
    observer_tool_calls: int = 8,
) -> dict:
    """在子进程中运行 NOptimizer，返回结果 dict。

    Args:
        noa_dir: noa/ 目录路径（可能是临时目录中的修改版本）
        project_root: 项目根目录（用于 import utils/ 等）
        target_source_dir: 目标系统源码目录
        dataset_pickle_path: pickle 序列化的 dataset 文件路径
        layer_level: 层级（1=L1, 2=L2, ...）
        max_steps: planner 最大步数
        n_samples: 每次 observe 的采样数
        model: LLM 模型名
        timeout: 子进程超时秒数
        isolate_source: 若为 True，在临时副本上运行，不污染原始目录

    Returns:
        {"final_score": float, "baseline_score": float, ...}
        失败时返回 {"final_score": 0, "error": str}
    """
    # 隔离模式：复制 target_source_dir 到临时目录
    temp_source_dir = None
    if isolate_source:
        import shutil

        temp_source_dir = tempfile.mkdtemp(prefix="noa_layer_isolated_")
        isolated_dir = os.path.join(
            temp_source_dir, os.path.basename(target_source_dir)
        )
        shutil.copytree(target_source_dir, isolated_dir)
        target_source_dir = isolated_dir

    noa_parent = os.path.dirname(os.path.abspath(noa_dir))

    # 创建临时文件用于传递结果
    result_fd, result_file = tempfile.mkstemp(prefix="noa_result_", suffix=".json")
    os.close(result_fd)

    runner_code = textwrap.dedent(f"""\
        import sys, os, json, pickle

        noa_parent = {noa_parent!r}
        project_root = {project_root!r}
        for p in [noa_parent, project_root]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from noa.engine import NOptimizer
        from noa.auto_adapter import auto_adapt

        with open({dataset_pickle_path!r}, "rb") as f:
            dataset = pickle.load(f)

        l0_source_dir = {target_source_dir!r}
        adapter, target_factory = auto_adapt(l0_source_dir, model={model!r})

        eval_module_path = os.path.join(l0_source_dir, "evaluate.py")
        if os.path.exists(eval_module_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_target_eval", eval_module_path)
            eval_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_mod)
            evaluate_batch = eval_mod.evaluate_batch
            score_fn = eval_mod.f1_score
        else:
            raise RuntimeError(f"evaluate.py not found at {{l0_source_dir}}")

        optimizer = NOptimizer(
            source_dir=l0_source_dir,
            target_factory=target_factory,
            dataset=dataset,
            eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=4),
            max_steps={max_steps},
            n_samples={n_samples},
            eval_n_samples={eval_n_samples},
            model={model!r},
            score_fn=score_fn,
            max_llm_calls={max_llm_calls},
            max_evals={max_evals},
            max_no_improve_steps={max_no_improve_steps},
            max_tool_calls={max_tool_calls},
            observer_max_tool_calls={observer_tool_calls},
        )
        result = optimizer.run()

        # DiffBlock → dict 序列化
        serialized_history = []
        for h in result.get("history", []):
            sh = dict(h)
            sh["diffs"] = [
                {{"file_path": d.file_path, "search": d.search, "replace": d.replace}}
                if hasattr(d, "file_path") else d
                for d in h.get("diffs", [])
            ]
            serialized_history.append(sh)
        result["history"] = serialized_history

        with open({result_file!r}, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    """)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{noa_parent}{os.pathsep}{project_root}"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        if os.path.exists(result_file):
            os.remove(result_file)
        return {"final_score": 0, "error": f"Subprocess timed out ({timeout}s)"}
    finally:
        if temp_source_dir and os.path.isdir(temp_source_dir):
            import shutil

            shutil.rmtree(temp_source_dir, ignore_errors=True)

    # 优先从临时文件读取结果
    try:
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    finally:
        if os.path.exists(result_file):
            os.remove(result_file)

    # 子进程失败 — 提取有意义的错误信息
    _NOISE_KEYWORDS = (
        "Loading weights",
        "Materializing",
        "BertModel",
        "LOAD REPORT",
        "UNEXPECTED",
        "Notes:",
        "All model checkpoint",
        "Some weights of",
        "You should probably TRAIN",
        "was not used when initializing",
        "Key ",
        "---+---",
        "Status",
        "embeddings.position_ids",
    )
    stderr = proc.stderr.strip()
    if stderr:
        lines = [
            ln
            for ln in stderr.splitlines()
            if not any(kw in ln for kw in _NOISE_KEYWORDS)
        ]
        error_msg = "\n".join(lines[-50:]) if lines else ""
    else:
        error_msg = ""

    if proc.returncode != 0 and not error_msg:
        error_msg = f"Subprocess exited with code {proc.returncode}"

    if not error_msg:
        error_msg = proc.stdout.strip()[-2000:] or "Unknown error"

    return {"final_score": 0, "error": error_msg[:2000]}


# Backward compatibility alias
run_l1_subprocess = run_layer_subprocess


def serialize_dataset(dataset: list, cache_dir: str | None = None) -> str:
    """将 dataset 序列化为 pickle 文件，返回路径。"""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "dataset.pkl")
        if os.path.exists(path):
            return path
    else:
        fd, path = tempfile.mkstemp(prefix="noa_dataset_", suffix=".pkl")
        os.close(fd)

    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    return path
