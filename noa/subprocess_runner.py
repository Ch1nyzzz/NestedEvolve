"""Subprocess Runner — 在隔离子进程中运行优化层，避免模块缓存问题。"""

from __future__ import annotations

import json
import os
import pickle
import hashlib
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
    timeout: int = 4800,
    isolate_source: bool = False,
    eval_n_samples: int = 20,
    max_llm_calls: int = 80,
    max_evals: int = 12,
    max_no_improve_steps: int = 5,
    random_seed: int | None = None,
) -> dict:
    """在子进程中运行 NOptimizer，返回结果 dict。"""
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

    # 若 noa_dir 不叫 "noa"（如 sandbox candidate），创建临时 symlink 确保 import 正确
    temp_noa_link_dir = None
    noa_basename = os.path.basename(os.path.abspath(noa_dir))
    if noa_basename != "noa":
        temp_noa_link_dir = tempfile.mkdtemp(prefix="noa_import_link_")
        os.symlink(os.path.abspath(noa_dir), os.path.join(temp_noa_link_dir, "noa"))
        noa_parent = temp_noa_link_dir

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
            eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=50),
            max_steps={max_steps},
            n_samples={n_samples},
            eval_n_samples={eval_n_samples},
            model={model!r},
            score_fn=score_fn,
            max_llm_calls={max_llm_calls},
            max_evals={max_evals},
            max_no_improve_steps={max_no_improve_steps},
        )
        _seed = {random_seed!r}
        if _seed is not None:
            import random as _rng
            _rng.seed(_seed)

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
        if temp_noa_link_dir and os.path.isdir(temp_noa_link_dir):
            import shutil

            shutil.rmtree(temp_noa_link_dir, ignore_errors=True)
        return {"final_score": 0, "error": f"Subprocess timed out ({timeout}s)"}
    finally:
        if temp_source_dir and os.path.isdir(temp_source_dir):
            import shutil

            shutil.rmtree(temp_source_dir, ignore_errors=True)
        if temp_noa_link_dir and os.path.isdir(temp_noa_link_dir):
            import shutil

            shutil.rmtree(temp_noa_link_dir, ignore_errors=True)

    try:
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    finally:
        if os.path.exists(result_file):
            os.remove(result_file)

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

    # 结构化错误分类，帮助 L2 诊断
    error_type = "unknown"
    if "SyntaxError" in error_msg or "IndentationError" in error_msg:
        error_type = "syntax_error"
    elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
        error_type = "import_error"
    elif "timed out" in error_msg:
        error_type = "timeout"
    elif proc.returncode != 0:
        error_type = "runtime_crash"

    return {
        "final_score": 0,
        "error": error_msg[:2000],
        "error_type": error_type,
        "returncode": proc.returncode,
    }


# Backward compatibility alias
run_l1_subprocess = run_layer_subprocess


def serialize_dataset(dataset: list, cache_dir: str | None = None) -> str:
    """将 dataset 序列化为 pickle 文件，返回路径。"""
    dataset_blob = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(dataset_blob).hexdigest()[:16]

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"dataset_{digest}.pkl")
    else:
        fd, path = tempfile.mkstemp(prefix="noa_dataset_", suffix=".pkl")
        os.close(fd)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(dataset_blob)
    return path
