"""Subprocess Runner — 在隔离子进程中运行优化层，避免模块缓存问题。

统一结果协议: 所有返回 dict 必须包含 status 字段：
  - success: 正常完成，result_file 有效
  - timeout: 超时被杀，结果可能不完整
  - crash: 运行时崩溃
  - partial: result_file 存在但进程异常退出（如 stderr 有 tqdm 输出）
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
import textwrap

from noa.runtime import current_scope, mini_l1_log_path, run_subprocess, scope_env


def _classify_error(error_msg: str, returncode: int | None) -> str:
    lower_error = error_msg.lower()
    if "SyntaxError" in error_msg or "IndentationError" in error_msg:
        return "syntax_error"
    if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
        return "import_error"
    if "timed out" in lower_error:
        return "timeout"
    if any(
        token in lower_error
        for token in (
            "connection error",
            "api connection",
            "connection reset",
            "remoteprotocolerror",
            "server disconnected",
            "temporarily unavailable",
            "rate limit",
            "429",
        )
    ):
        return "llm_connection_error"
    if returncode not in (None, 0):
        return "runtime_crash"
    return "unknown"


def _extract_meaningful_error(proc) -> str:
    """从 stderr/stdout 提取有意义的错误信息，过滤掉进度条等噪音。"""
    stderr = proc.stderr_tail.strip()
    # 过滤 tqdm / 进度条噪音行
    if stderr:
        meaningful_lines = []
        for line in stderr.split("\n"):
            stripped = line.strip()
            # 跳过 tqdm 进度条、空行、纯数字百分比行
            if not stripped:
                continue
            if any(indicator in stripped for indicator in ("%|", "it/s", "s/it", "\r")):
                continue
            if stripped.startswith("[") and "%" in stripped[:20]:
                continue
            meaningful_lines.append(stripped)
        stderr = "\n".join(meaningful_lines).strip()

    if stderr:
        return stderr[:2000]

    stdout = proc.stdout_tail.strip()
    if stdout:
        return stdout[:2000]

    return proc.error or f"Subprocess exited with code {proc.returncode}"


def run_layer_subprocess(
    noa_dir: str,
    project_root: str,
    target_source_dir: str,
    dataset_pickle_path: str,
    layer_level: int = 1,
    max_steps: int = 20,
    n_samples: int = 20,
    model: str = "gpt-4.1-mini",
    timeout: int = 28800,
    isolate_source: bool = False,
    max_llm_calls: int = 80,
    max_no_improve_steps: int = 5,
    random_seed: int | None = None,
    train_pool_pickle_path: str | None = None,
    val_set_pickle_path: str | None = None,
    test_set_pickle_path: str | None = None,
    train_sample_size: int = 25,
    top_k: int = 3,
    attempt_index: int = 0,
) -> dict:
    """在子进程中运行 NOptimizer，返回结构化结果 dict。

    返回字段:
      status: success | timeout | crash | partial
      timed_out: bool
      returncode: int | None
      result_written: bool — result_file 是否有效写入
      log_path: str
      duration_sec: float
      final_score: float
      error: str (仅失败时)
      error_type: str (仅失败时)
    """
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

    temp_noa_link_dir = None
    noa_basename = os.path.basename(os.path.abspath(noa_dir))
    if noa_basename != "noa":
        temp_noa_link_dir = tempfile.mkdtemp(prefix="noa_import_link_")
        os.symlink(os.path.abspath(noa_dir), os.path.join(temp_noa_link_dir, "noa"))
        noa_parent = temp_noa_link_dir

    result_fd, result_file = tempfile.mkstemp(prefix="noa_result_", suffix=".json")
    os.close(result_fd)

    runner_code = textwrap.dedent(
        f"""\
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

        train_pool = None
        val_set = None
        test_set = None
        _train_path = {train_pool_pickle_path!r}
        _val_path = {val_set_pickle_path!r}
        _test_path = {test_set_pickle_path!r}
        if _train_path and os.path.exists(_train_path):
            with open(_train_path, "rb") as f:
                train_pool = pickle.load(f)
        if _val_path and os.path.exists(_val_path):
            with open(_val_path, "rb") as f:
                val_set = pickle.load(f)
        if _test_path and os.path.exists(_test_path):
            with open(_test_path, "rb") as f:
                test_set = pickle.load(f)

        l0_source_dir = {target_source_dir!r}
        adapter, target_factory = auto_adapt(l0_source_dir, model={model!r})

        eval_module_path = os.path.join(l0_source_dir, "evaluate.py")
        if os.path.exists(eval_module_path):
            import importlib.util

            spec = importlib.util.spec_from_file_location("_target_eval", eval_module_path)
            eval_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_mod)
            evaluate_batch = eval_mod.evaluate_batch
            score_fn = (
                getattr(eval_mod, "score_example", None)
                or getattr(eval_mod, "mrr_score", None)
                or getattr(eval_mod, "f1_score", None)
                or getattr(eval_mod, "exact_match", None)
            )
            if score_fn is None:
                raise RuntimeError(
                    f"evaluate.py at {{l0_source_dir}} has no supported score function "
                    f"(expected one of score_example, mrr_score, f1_score, exact_match)"
                )
        else:
            raise RuntimeError(f"evaluate.py not found at {{l0_source_dir}}")

        eval_workers = max(1, int(os.getenv("NOA_EVAL_MAX_WORKERS", "100")))

        optimizer = NOptimizer(
            source_dir=l0_source_dir,
            target_factory=target_factory,
            dataset=dataset,
            eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=eval_workers),
            max_steps={max_steps},
            n_samples={n_samples},
            model={model!r},
            score_fn=score_fn,
            max_llm_calls={max_llm_calls},
            max_no_improve_steps={max_no_improve_steps},
            train_pool=train_pool,
            val_set=val_set,
            test_set=test_set,
            train_sample_size={train_sample_size},
            top_k={top_k},
        )
        _seed = {random_seed!r}
        if _seed is not None:
            import random as _rng
            _rng.seed(_seed)

        result = optimizer.run()

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
    """
    )

    env = scope_env()
    env["PYTHONPATH"] = f"{noa_parent}{os.pathsep}{project_root}"
    scope = current_scope()
    log_path = mini_l1_log_path(scope.spawn_id or f"layer_{layer_level}", attempt_index)

    try:
        proc = run_subprocess(
            [sys.executable, "-c", runner_code],
            timeout_sec=timeout,
            kind=f"layer_subprocess_L{layer_level}",
            env=env,
            log_path=log_path,
            result_path=result_file,
            stream_output=True,
            heartbeat_message=f"L{layer_level} optimizer subprocess",
        )

        result_exists = os.path.exists(result_file)
        result_size = os.path.getsize(result_file) if result_exists else 0
        result_written = result_exists and result_size > 0

        print(
            f"[SubprocessRunner] returncode={proc.returncode} ok={proc.ok} "
            f"timed_out={proc.timed_out} result_written={result_written} "
            f"result_size={result_size} duration={proc.duration_sec:.0f}s"
        )

        # 构建统一结果
        base_result = {
            "timed_out": proc.timed_out,
            "returncode": proc.returncode,
            "result_written": result_written,
            "log_path": log_path,
            "duration_sec": proc.duration_sec,
        }

        # 优先读 result_file
        if result_written:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                payload.update(base_result)
                if proc.ok:
                    payload["status"] = "success"
                elif proc.timed_out:
                    # 结果写出了但进程被超时杀掉 — 部分结果
                    payload["status"] = "partial"
                    payload["warning"] = (
                        f"timed_out after {proc.duration_sec:.0f}s but result_file valid"
                    )
                else:
                    payload["status"] = "partial"
                    payload["warning"] = (
                        f"returncode={proc.returncode} but result_file valid"
                    )
                return payload
            except (json.JSONDecodeError, ValueError):
                pass  # result_file 损坏，走 error 路径

        # 没有有效 result_file — 提取有意义的错误信息
        error_msg = _extract_meaningful_error(proc)

        if proc.timed_out:
            status = "timeout"
            error_type = "timeout"
            error_msg = (
                f"Subprocess timed out after {proc.duration_sec:.0f}s "
                f"(limit={timeout}s). Result file not written. "
                f"Last meaningful output: {error_msg[:500]}"
            )
        else:
            status = "crash"
            error_type = _classify_error(error_msg, proc.returncode)

        base_result.update(
            {
                "status": status,
                "final_score": 0,
                "error": error_msg[:2000],
                "error_type": error_type,
            }
        )
        return base_result
    finally:
        if os.path.exists(result_file):
            os.remove(result_file)
        if temp_source_dir and os.path.isdir(temp_source_dir):
            import shutil

            shutil.rmtree(temp_source_dir, ignore_errors=True)
        if temp_noa_link_dir and os.path.isdir(temp_noa_link_dir):
            import shutil

            shutil.rmtree(temp_noa_link_dir, ignore_errors=True)


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
