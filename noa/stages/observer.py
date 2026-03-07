"""Observer — 收集执行轨迹（支持 agentic 多轮决策）。"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shlex
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

from utils.llm import DEFAULT_MODEL
from noa.core.protocol import Trajectory
from noa.stages.agentic import agentic_loop

log = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
}
_TRAJECTORY_EXTS = {".json", ".jsonl", ".ndjson"}
_TRAJECTORY_NAME_HINTS = (
    "trajectory",
    "trajectories",
    "trace",
    "history",
    "result",
    "eval",
)
_TRAJECTORY_KEY_HINTS = (
    '"question"',
    '"prediction"',
    '"ground_truth"',
    '"f1"',
    '"intermediate"',
    '"eval_details"',
    '"history"',
)
_MAX_FILE_SCAN = 40000
_MAX_DEPTH = 8
_TRAJECTORY_SCHEMA_VERSION = "2.0"
_MIN_INTERMEDIATE_COVERAGE = 0.95
_MAX_TOPUP_ROUNDS = 3

AGENTIC_OBSERVER_SYSTEM = (
    "You are an autonomous observer for optimization systems. "
    "Your job is to gather high-quality trajectories with minimum unnecessary compute. "
    "You can inspect existing artifacts, run targeted experiments, then finalize your collection plan."
)

AGENTIC_OBSERVER_PROMPT = """\
You need {n_samples} trajectories for the current optimization cycle.

Context:
- source_dir: {source_dir}
- dataset_size: {dataset_size}
- seed: {seed}
- search_roots: {search_roots}

Workflow:
1. Inspect existing trajectory artifacts first
2. Reuse trajectories if they are available and useful
3. If insufficient, decide how to run experiments to generate trajectories
4. Finish by calling `finalize_observation`

Rules:
- Prefer existing trajectories when quality/coverage is sufficient
- Ensure reused trajectories keep component-level `intermediate` outputs
- If reused records miss `intermediate`, replay those questions before diagnosis
- If existing data is stale or missing, run new observation
- Keep tool use focused (avoid redundant calls)
- Call `finalize_observation` when you have a concrete decision
{layer_context}"""


def observe_agentic(
    target,
    dataset: list,
    n_samples: int = 20,
    seed: int = 42,
    *,
    score_fn,
    model: str = DEFAULT_MODEL,
    source_dir: str | None = None,
    search_roots: list[str] | None = None,
    max_tool_calls: int = 10,
    persist_root: str | None = None,
    layer_context: str = "",
    layer_context_obj=None,
    required_intermediate_keys: list[str] | None = None,
    stats: dict | None = None,
) -> list[Trajectory]:
    """Agentic observe：LLM 多轮决定“复用已有轨迹”或“运行实验获取新轨迹”。"""
    probe = _ObservationProbe(
        target=target,
        dataset=dataset,
        score_fn=score_fn,
        source_dir=source_dir,
        search_roots=search_roots,
        layer_context=layer_context_obj,
        required_intermediate_keys=required_intermediate_keys,
    )

    messages = [
        {"role": "system", "content": AGENTIC_OBSERVER_SYSTEM},
        {
            "role": "user",
            "content": AGENTIC_OBSERVER_PROMPT.format(
                n_samples=n_samples,
                source_dir=source_dir or "(unknown)",
                dataset_size=len(dataset),
                seed=seed,
                search_roots=probe.search_roots,
                layer_context=layer_context,
            ),
        },
    ]
    tools = probe.get_tool_schemas()

    def _parse_decision(text: str) -> dict | None:
        parsed = _parse_json_like(text)
        if isinstance(parsed, dict) and parsed.get("strategy"):
            probe.final_decision = parsed
            return parsed
        return None

    try:
        agentic_loop(
            messages=messages,
            tools=tools,
            tool_executor=probe.execute_tool,
            model=model,
            max_tool_calls=max_tool_calls,
            parse_fn=lambda text: _parse_decision(text),
            early_stop_fn=lambda: probe.final_decision is not None,
            no_tool_call_prompt="Continue using tools if needed, then finalize with finalize_observation.",
            json_retries=0,
            budget_exhausted_prompt="Finalize your observation strategy now.",
            stats=stats,
        )
    except Exception:
        log.warning("Agentic observe failed, fallback to fresh observe", exc_info=True)

    trajectories = probe.materialize(n_samples=n_samples, seed=seed)
    if not trajectories:
        trajectories = _run_fresh_observe(
            target, dataset, n_samples=n_samples, seed=seed, score_fn=score_fn
        )

    _persist_trajectories(
        trajectories,
        persist_root=persist_root or probe.persist_root,
        tag="agentic_observe",
        meta={
            "mode": "agentic",
            "n_samples": n_samples,
            "seed": seed,
            "decision": probe.final_decision or {},
            "replay_count": probe.replay_count,
            "replay_failed": probe.replay_failed,
        },
    )
    return trajectories


class _ObservationProbe:
    """供 agentic observer 调用的工具集合。"""

    def __init__(
        self,
        *,
        target,
        dataset: list,
        score_fn,
        source_dir: str | None,
        search_roots: list[str] | None,
        layer_context=None,
        required_intermediate_keys: list[str] | None = None,
    ):
        self.target = target
        self.dataset = dataset
        self.score_fn = score_fn
        self.source_dir = os.path.abspath(source_dir) if source_dir else None
        self.search_roots = _normalize_roots(search_roots, self.source_dir)
        self.persist_root = _default_persist_root(self.search_roots, self.source_dir)
        self.layer_context = layer_context
        self.required_intermediate_keys = [
            k for k in (required_intermediate_keys or []) if isinstance(k, str) and k
        ]
        self._dataset_by_question = {}
        for ex in dataset:
            q = getattr(ex, "question", None)
            a = getattr(ex, "answer", None)
            if isinstance(q, str) and q and isinstance(a, str):
                self._dataset_by_question[q] = a

        self._loaded: dict[str, list[Trajectory]] = {}
        self._fresh_runs: list[list[Trajectory]] = []
        self._cached_candidates: list[dict] | None = None
        self._command_book: dict[str, str] = {}
        self.final_decision: dict | None = None
        self.replay_count = 0
        self.replay_failed = 0

    def get_tool_schemas(self) -> list[dict]:
        # 高层 (level >= 2) 禁用命令执行工具
        allow_commands = (
            self.layer_context is None or getattr(self.layer_context, "level", 1) < 2
        )
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "list_trajectory_files",
                    "description": "Search project roots for reusable trajectory/trace/result files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 30,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "load_trajectory_file",
                    "description": "Load trajectories from one artifact file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_items": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 500,
                                "default": 100,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]
        if allow_commands:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "discover_experiment_commands",
                            "description": "Discover runnable experiment scripts to generate trajectories if needed.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "limit": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 100,
                                        "default": 20,
                                    },
                                },
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "run_experiment_command",
                            "description": "Run one discovered experiment command.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "timeout_sec": {
                                        "type": "integer",
                                        "minimum": 10,
                                        "maximum": 3600,
                                        "default": 240,
                                    },
                                },
                                "required": ["command"],
                            },
                        },
                    },
                ]
            )
        schemas.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "run_dataset_observe",
                        "description": "Run fresh observation directly on current target/dataset.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "n_samples": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 500,
                                    "default": 20,
                                },
                                "seed": {"type": "integer", "default": 42},
                            },
                            "required": ["n_samples", "seed"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "finalize_observation",
                        "description": "Finalize decision for which trajectories to return.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "strategy": {
                                    "type": "string",
                                    "enum": [
                                        "auto",
                                        "use_file",
                                        "use_fresh",
                                        "use_mixed",
                                    ],
                                    "default": "auto",
                                },
                                "path": {"type": "string"},
                                "run_index": {"type": "integer", "default": -1},
                                "n_samples": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 500,
                                },
                                "seed": {"type": "integer"},
                            },
                        },
                    },
                },
            ]
        )
        return schemas

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "list_trajectory_files":
                result = self.list_trajectory_files(
                    limit=int(arguments.get("limit", 30))
                )
            elif tool_name == "load_trajectory_file":
                result = self.load_trajectory_file(
                    path=str(arguments.get("path", "")),
                    max_items=int(arguments.get("max_items", 100)),
                )
            elif tool_name == "discover_experiment_commands":
                result = self.discover_experiment_commands(
                    limit=int(arguments.get("limit", 20))
                )
            elif tool_name == "run_experiment_command":
                result = self.run_experiment_command(
                    command=str(arguments.get("command", "")),
                    timeout_sec=int(arguments.get("timeout_sec", 240)),
                )
            elif tool_name == "run_dataset_observe":
                result = self.run_dataset_observe(
                    n_samples=int(arguments.get("n_samples", 20)),
                    seed=int(arguments.get("seed", 42)),
                )
            elif tool_name == "finalize_observation":
                result = self.finalize_observation(arguments)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception:
            result = {"error": traceback.format_exc()[-1000:]}

        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > 6000:
            text = text[:6000] + "... (truncated)"
        return text

    def list_trajectory_files(self, limit: int = 30) -> dict:
        if self._cached_candidates is None:
            self._cached_candidates = _find_trajectory_candidates(
                self.search_roots, limit=max(100, limit)
            )
        candidates = self._cached_candidates[:limit]
        return {
            "count": len(candidates),
            "search_roots": self.search_roots,
            "candidates": candidates,
        }

    def load_trajectory_file(self, path: str, max_items: int = 100) -> dict:
        resolved = _resolve_path(path, self.search_roots)
        if not resolved or not os.path.isfile(resolved):
            return {"error": f"File not found: {path}"}

        trajectories = _load_trajectories_from_file(resolved, max_items=max_items)
        self._loaded[resolved] = trajectories
        coverage = _intermediate_coverage(trajectories, self.required_intermediate_keys)
        return {
            "path": resolved,
            "loaded": len(trajectories),
            "intermediate_coverage": round(coverage, 4),
            "preview": _trajectory_preview(trajectories, limit=3),
        }

    def discover_experiment_commands(self, limit: int = 20) -> dict:
        commands = _discover_experiment_commands(self.search_roots, limit=limit)
        self._command_book = {item["command"]: item["cwd"] for item in commands}
        return {"count": len(commands), "commands": commands}

    def run_experiment_command(self, command: str, timeout_sec: int = 240) -> dict:
        if command not in self._command_book:
            return {
                "error": "Command not allowed. Call discover_experiment_commands first and use one listed command.",
                "known_commands": list(self._command_book.keys())[:20],
            }

        cwd = self._command_book[command]
        started_at = time.time()
        try:
            proc = subprocess.run(
                shlex.split(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            duration = round(time.time() - started_at, 2)
            self._cached_candidates = None  # invalidate cache
            return {
                "command": command,
                "cwd": cwd,
                "returncode": proc.returncode,
                "duration_sec": duration,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "cwd": cwd,
                "error": f"Timeout after {timeout_sec}s",
            }

    def run_dataset_observe(self, n_samples: int, seed: int) -> dict:
        trajectories = _run_fresh_observe(
            self.target,
            self.dataset,
            n_samples=n_samples,
            seed=seed,
            score_fn=self.score_fn,
            required_intermediate_keys=self.required_intermediate_keys,
        )
        self._fresh_runs.append(trajectories)
        return {
            "run_index": len(self._fresh_runs) - 1,
            "n": len(trajectories),
            "preview": _trajectory_preview(trajectories, limit=3),
        }

    def finalize_observation(self, decision: dict) -> dict:
        strategy = decision.get("strategy", "auto")
        self.final_decision = {
            "strategy": strategy,
            "path": decision.get("path"),
            "run_index": decision.get("run_index", -1),
            "n_samples": decision.get("n_samples"),
            "seed": decision.get("seed"),
        }
        return {"ok": True, "decision": self.final_decision}

    def materialize(self, n_samples: int, seed: int) -> list[Trajectory]:
        decision = self.final_decision or {"strategy": "auto"}
        strategy = decision.get("strategy", "auto")
        chosen: list[Trajectory] = []

        if strategy == "use_file":
            path = _resolve_path(str(decision.get("path") or ""), self.search_roots)
            if path:
                if path not in self._loaded:
                    self._loaded[path] = _load_trajectories_from_file(
                        path, max_items=max(2 * n_samples, 50)
                    )
                chosen = list(self._loaded.get(path, []))

        elif strategy == "use_fresh":
            run_index = int(decision.get("run_index", -1))
            if self._fresh_runs:
                if -len(self._fresh_runs) <= run_index < len(self._fresh_runs):
                    chosen = list(self._fresh_runs[run_index])
                else:
                    chosen = list(self._fresh_runs[-1])
            if not chosen:
                chosen = _run_fresh_observe(
                    self.target,
                    self.dataset,
                    n_samples=int(decision.get("n_samples") or n_samples),
                    seed=int(decision.get("seed") or seed),
                    score_fn=self.score_fn,
                    required_intermediate_keys=self.required_intermediate_keys,
                )

        elif strategy == "use_mixed":
            file_part = self._best_loaded()
            if not file_part:
                self.list_trajectory_files(limit=20)
                for c in self._cached_candidates or []:
                    loaded = _load_trajectories_from_file(
                        c["path"], max_items=n_samples
                    )
                    if loaded:
                        file_part = loaded
                        self._loaded[c["path"]] = loaded
                        break
            fresh = (
                self._fresh_runs[-1]
                if self._fresh_runs
                else _run_fresh_observe(
                    self.target,
                    self.dataset,
                    n_samples=n_samples,
                    seed=seed,
                    score_fn=self.score_fn,
                    required_intermediate_keys=self.required_intermediate_keys,
                )
            )
            half = max(1, n_samples // 2)
            chosen = list(file_part[:half]) + list(fresh[: max(0, n_samples - half)])

        if not chosen:
            # auto / fallback
            loaded = self._best_loaded()
            if loaded:
                chosen = loaded
            elif self._fresh_runs:
                chosen = list(self._fresh_runs[-1])
            else:
                chosen = _run_fresh_observe(
                    self.target,
                    self.dataset,
                    n_samples=n_samples,
                    seed=seed,
                    score_fn=self.score_fn,
                    required_intermediate_keys=self.required_intermediate_keys,
                )
        chosen = _shrink_to_n(chosen, n_samples=n_samples, seed=seed)
        chosen = self._repair_missing_intermediate(chosen)
        chosen = self._topup_intermediate_coverage(
            chosen, n_samples=n_samples, seed=seed
        )
        return chosen

    def _repair_missing_intermediate(
        self, trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        repaired: list[Trajectory] = []
        replay_cache: dict[str, Trajectory] = {}

        for t in trajectories:
            if _is_intermediate_complete(
                t.intermediate, self.required_intermediate_keys
            ):
                t.intermediate_complete = True
                if not t.trace_source:
                    t.trace_source = "reused"
                repaired.append(t)
                continue

            cached = replay_cache.get(t.question)
            if cached is not None:
                repaired.append(cached)
                continue

            replayed = self._replay_trajectory(t)
            replay_cache[t.question] = replayed
            repaired.append(replayed)

        return repaired

    def _topup_intermediate_coverage(
        self, trajectories: list[Trajectory], *, n_samples: int, seed: int
    ) -> list[Trajectory]:
        current = trajectories
        rounds = 0
        while rounds < _MAX_TOPUP_ROUNDS:
            cov = _intermediate_coverage(current, self.required_intermediate_keys)
            if cov >= _MIN_INTERMEDIATE_COVERAGE:
                break
            have = sum(
                1
                for t in current
                if _is_intermediate_complete(
                    t.intermediate, self.required_intermediate_keys
                )
            )
            required = int(math.ceil(_MIN_INTERMEDIATE_COVERAGE * max(1, n_samples)))
            need = max(1, required - have)
            fresh = _run_fresh_observe(
                self.target,
                self.dataset,
                n_samples=need,
                seed=seed + 1000 + rounds,
                score_fn=self.score_fn,
                required_intermediate_keys=self.required_intermediate_keys,
            )
            for t in fresh:
                t.trace_source = "fresh"
                t.intermediate_complete = _is_intermediate_complete(
                    t.intermediate, self.required_intermediate_keys
                )
            current = _prioritize_complete_trajectories(
                current + fresh,
                n_samples=n_samples,
                seed=seed + rounds,
                required_intermediate_keys=self.required_intermediate_keys,
            )
            rounds += 1
        return current

    def _replay_trajectory(self, t: Trajectory) -> Trajectory:
        try:
            result = self.target(t.question)
            pred = str(getattr(result, "answer", "") or "")
            gt = t.ground_truth or self._dataset_by_question.get(t.question, "")
            score = float(self.score_fn(pred, gt)) if gt else float(t.f1)
            intermediate = getattr(result, "intermediate", {})
            if not isinstance(intermediate, dict):
                intermediate = {}
            complete = _is_intermediate_complete(
                intermediate, self.required_intermediate_keys
            )
            if complete:
                self.replay_count += 1
            else:
                self.replay_failed += 1
            return Trajectory(
                question=t.question,
                ground_truth=gt or t.ground_truth,
                prediction=pred,
                f1=score,
                intermediate=intermediate,
                error=None,
                trace_source="replayed",
                intermediate_complete=complete,
            )
        except Exception:
            self.replay_failed += 1
            return Trajectory(
                question=t.question,
                ground_truth=t.ground_truth,
                prediction=t.prediction,
                f1=float(t.f1),
                intermediate=t.intermediate if isinstance(t.intermediate, dict) else {},
                error=t.error or traceback.format_exc()[-1000:],
                trace_source=t.trace_source or "reused",
                intermediate_complete=False,
            )

    def _best_loaded(self) -> list[Trajectory]:
        best: list[Trajectory] = []
        best_score: tuple[float, int] = (-1.0, -1)
        for trajectories in self._loaded.values():
            score = (
                _intermediate_coverage(trajectories, self.required_intermediate_keys),
                len(trajectories),
            )
            if score > best_score:
                best = trajectories
                best_score = score
        return list(best)


def _run_fresh_observe(
    target,
    dataset: list,
    n_samples: int,
    seed: int,
    score_fn,
    required_intermediate_keys: list[str] | None = None,
) -> list[Trajectory]:
    rng = random.Random(seed)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    trajectories: list[Trajectory] = []
    for ex in sampled:
        try:
            result = target(ex.question)
            score = score_fn(result.answer, ex.answer)
            trajectories.append(
                Trajectory(
                    question=ex.question,
                    ground_truth=ex.answer,
                    prediction=result.answer,
                    f1=float(score),
                    intermediate=result.intermediate
                    if isinstance(result.intermediate, dict)
                    else {},
                    trace_source="fresh",
                    intermediate_complete=_is_intermediate_complete(
                        result.intermediate
                        if isinstance(result.intermediate, dict)
                        else {},
                        required_intermediate_keys,
                    ),
                )
            )
        except Exception:
            trajectories.append(
                Trajectory(
                    question=ex.question,
                    ground_truth=ex.answer,
                    prediction="",
                    f1=0.0,
                    error=traceback.format_exc(),
                    trace_source="fresh",
                    intermediate_complete=False,
                )
            )
    return trajectories


def _persist_trajectories(
    trajectories: list[Trajectory],
    *,
    persist_root: str | None,
    tag: str,
    meta: dict | None = None,
) -> None:
    if not trajectories:
        return
    root = persist_root or os.getcwd()
    root = os.path.abspath(root)
    if os.path.basename(root) == ".noa_cache":
        cache_dir = os.path.join(root, "trajectories")
    else:
        cache_dir = os.path.join(root, ".noa_cache", "trajectories")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{ts}.json"
        path = os.path.join(cache_dir, filename)
        meta_payload = dict(meta or {})
        replay_count = sum(1 for t in trajectories if t.trace_source == "replayed")
        coverage = _intermediate_coverage(trajectories)
        meta_payload.update(
            {
                "schema_version": _TRAJECTORY_SCHEMA_VERSION,
                "intermediate_coverage": round(coverage, 4),
                "contains_replayed": replay_count > 0,
                "replay_count": replay_count,
            }
        )
        payload = {
            "tag": tag,
            "created_at": datetime.now().isoformat(),
            "count": len(trajectories),
            "meta": meta_payload,
            "trajectories": [_trajectory_to_dict(t) for t in trajectories],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.debug("Failed to persist trajectories", exc_info=True)


def _trajectory_to_dict(t: Trajectory) -> dict:
    return {
        "question": t.question,
        "ground_truth": t.ground_truth,
        "prediction": t.prediction,
        "f1": t.f1,
        "intermediate": t.intermediate,
        "error": t.error,
        "trace_source": t.trace_source,
        "intermediate_complete": bool(t.intermediate_complete),
    }


def _shrink_to_n(
    trajectories: list[Trajectory], n_samples: int, seed: int
) -> list[Trajectory]:
    if len(trajectories) <= n_samples:
        return trajectories
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(trajectories)), n_samples))
    return [trajectories[i] for i in idxs]


def _normalize_roots(
    search_roots: list[str] | None, source_dir: str | None
) -> list[str]:
    roots: list[str] = []
    if search_roots:
        roots.extend(search_roots)
    if source_dir:
        roots.append(source_dir)
        p = Path(source_dir).resolve()
        for parent in [p.parent, p.parent.parent]:
            roots.append(str(parent))
    roots.append(os.getcwd())

    normalized: list[str] = []
    seen = set()
    for root in roots:
        if not root:
            continue
        r = os.path.abspath(root)
        if not os.path.isdir(r):
            continue
        if r in seen:
            continue
        seen.add(r)
        normalized.append(r)
    return normalized


def _default_persist_root(search_roots: list[str], source_dir: str | None) -> str:
    for root in search_roots:
        if os.path.isdir(os.path.join(root, ".git")):
            return root
        if os.path.isdir(os.path.join(root, ".noa_cache")):
            return root
    if source_dir:
        p = Path(source_dir).resolve()
        for parent in [p, *p.parents]:
            if (parent / ".git").exists() or (parent / ".noa_cache").exists():
                return str(parent)
    return os.getcwd()


def _find_trajectory_candidates(
    search_roots: list[str], limit: int = 100
) -> list[dict]:
    candidates: list[dict] = []
    scanned = 0
    for root in search_roots:
        root_path = Path(root)
        for dirpath, dirnames, filenames in os.walk(root):
            scanned += len(filenames)
            if scanned > _MAX_FILE_SCAN:
                break

            rel_depth = len(Path(dirpath).relative_to(root_path).parts)
            if rel_depth > _MAX_DEPTH:
                dirnames[:] = []
                continue

            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _TRAJECTORY_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                score = _trajectory_candidate_score(fpath)
                if score <= 0:
                    continue
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                candidates.append(
                    {
                        "path": os.path.abspath(fpath),
                        "size": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(
                            timespec="seconds"
                        ),
                        "score": score,
                    }
                )
            if len(candidates) >= limit * 4:
                break
        if scanned > _MAX_FILE_SCAN:
            break

    candidates.sort(key=lambda c: (c["score"], c["mtime"]), reverse=True)
    return candidates[:limit]


def _trajectory_candidate_score(path: str) -> int:
    lower = path.lower()
    base = os.path.basename(lower)
    ext = os.path.splitext(base)[1]
    if ext not in _TRAJECTORY_EXTS:
        return 0

    score = 1
    for hint in _TRAJECTORY_NAME_HINTS:
        if hint in base:
            score += 2
    if ".noa_cache/trajectories/" in lower:
        score += 4
    if "/.noa_runs/" in lower:
        score += 2
    if "/openevolve_output/" in lower:
        score += 1
    if base.startswith("results") and ext == ".json":
        score -= 3

    sniff = _quick_sniff(path)
    if '"intermediate"' in sniff:
        score += 2
    else:
        score -= 2
    for hint in _TRAJECTORY_KEY_HINTS:
        if hint in sniff:
            score += 1
    return score


def _quick_sniff(path: str, max_bytes: int = 4096) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes).lower()
    except OSError:
        return ""


def _resolve_path(path: str, search_roots: list[str]) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    for root in search_roots:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _load_trajectories_from_file(path: str, max_items: int = 100) -> list[Trajectory]:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jsonl", ".ndjson"}:
            data: list = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= max_items * 5:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parsed = _parse_json_like(line)
                    if parsed is not None:
                        data.append(parsed)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
    except Exception:
        return []

    raw_records = _extract_trajectory_records(data)
    trajectories: list[Trajectory] = []
    seen = set()
    for raw in raw_records:
        t = _record_to_trajectory(raw)
        if t is None:
            continue
        sig = (t.question, t.prediction, round(float(t.f1), 4))
        if sig in seen:
            continue
        seen.add(sig)
        trajectories.append(t)
        if len(trajectories) >= max_items:
            break
    return trajectories


def _extract_trajectory_records(data) -> list[dict]:
    records: list[dict] = []

    def walk(node, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(node, dict):
            if _is_trajectory_like(node):
                records.append(node)
                return

            # 常见容器字段优先
            for key in (
                "trajectories",
                "trajectory",
                "eval_details",
                "details",
                "history",
                "rounds",
                "trace",
                "traces",
                "samples",
            ):
                val = node.get(key)
                if isinstance(val, (list, dict)):
                    walk(val, depth + 1)

            # 补充遍历（容错）
            for val in node.values():
                if isinstance(val, (list, dict)):
                    walk(val, depth + 1)

        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item, depth + 1)

    walk(data, 0)
    return records


def _is_trajectory_like(item: dict) -> bool:
    if "question" not in item:
        return False
    if "f1" in item or "prediction" in item or "pred" in item:
        return True
    if "score" in item and ("answer" in item or "ground_truth" in item):
        return True
    return False


def _record_to_trajectory(item: dict) -> Trajectory | None:
    question = str(item.get("question", "")).strip()
    if not question:
        return None

    gt = _pick_str(
        item.get("ground_truth"),
        item.get("gold"),
        item.get("target"),
        item.get("answer"),
        item.get("gt"),
    )
    pred = _pick_str(
        item.get("prediction"),
        item.get("pred"),
        item.get("model_answer"),
        item.get("response"),
        item.get("answer_pred"),
        item.get("answer"),
    )
    f1 = _to_float(item.get("f1"))
    if f1 is None:
        score = _to_float(item.get("score"))
        if score is not None:
            f1 = score / 100.0 if score > 1 else score
    if f1 is None:
        f1 = 0.0

    intermediate = item.get("intermediate")
    if not isinstance(intermediate, dict):
        intermediate = {}
    for k in ("_artifacts", "artifacts", "retrieval", "reasoning"):
        if k in item and k not in intermediate:
            intermediate[k] = item[k]
    trace_source = _pick_str(item.get("trace_source")) or "reused"
    intermediate_complete = _coerce_bool(item.get("intermediate_complete"))
    if intermediate_complete is None:
        intermediate_complete = _is_intermediate_complete(intermediate, None)

    err = _pick_str(item.get("error"), item.get("traceback"), item.get("exception"))
    return Trajectory(
        question=question,
        ground_truth=gt,
        prediction=pred,
        f1=float(f1),
        intermediate=intermediate,
        error=err or None,
        trace_source=trace_source,
        intermediate_complete=bool(intermediate_complete),
    )


def _pick_str(*vals) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _coerce_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"true", "1", "yes", "y"}:
            return True
        if lv in {"false", "0", "no", "n"}:
            return False
    return None


def _is_intermediate_complete(
    intermediate: dict, required_keys: list[str] | None
) -> bool:
    if not isinstance(intermediate, dict) or not intermediate:
        return False
    if required_keys:
        present = [k for k in required_keys if k in intermediate]
        return len(present) >= max(1, min(2, len(required_keys)))
    return True


def _intermediate_coverage(
    trajectories: list[Trajectory], required_keys: list[str] | None = None
) -> float:
    if not trajectories:
        return 0.0
    complete = sum(
        1
        for t in trajectories
        if _is_intermediate_complete(t.intermediate, required_keys)
    )
    return complete / len(trajectories)


def _prioritize_complete_trajectories(
    trajectories: list[Trajectory],
    *,
    n_samples: int,
    seed: int,
    required_intermediate_keys: list[str] | None,
) -> list[Trajectory]:
    if len(trajectories) <= n_samples:
        return trajectories

    rng = random.Random(seed)
    complete = [
        t
        for t in trajectories
        if _is_intermediate_complete(t.intermediate, required_intermediate_keys)
    ]
    incomplete = [
        t
        for t in trajectories
        if not _is_intermediate_complete(t.intermediate, required_intermediate_keys)
    ]
    rng.shuffle(complete)
    rng.shuffle(incomplete)
    kept = complete[:n_samples]
    if len(kept) < n_samples:
        kept.extend(incomplete[: n_samples - len(kept)])
    return kept


def _to_float(v) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _trajectory_preview(trajectories: list[Trajectory], limit: int = 3) -> list[dict]:
    preview = []
    for t in trajectories[:limit]:
        preview.append(
            {
                "question": t.question[:120],
                "f1": round(float(t.f1), 4),
                "has_error": bool(t.error),
                "trace_source": t.trace_source,
                "intermediate_complete": bool(t.intermediate_complete),
            }
        )
    return preview


def _discover_experiment_commands(
    search_roots: list[str], limit: int = 20
) -> list[dict]:
    commands: list[dict] = []
    seen = set()
    for root in search_roots:
        root_path = Path(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = len(Path(dirpath).relative_to(root_path).parts)
            if rel_depth > 4:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

            for fname in filenames:
                lower = fname.lower()
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, root)
                cmd = None
                if lower.startswith("run_") and lower.endswith(".py"):
                    cmd = f"python3 {rel}"
                elif (
                    lower.startswith("run_") or "experiment" in lower
                ) and lower.endswith(".sh"):
                    cmd = f"bash {rel}"
                elif lower in {"run_l1.py", "run_nested.py", "run_baseline.py"}:
                    cmd = f"python3 {rel}"

                if not cmd:
                    continue
                key = (cmd, root)
                if key in seen:
                    continue
                seen.add(key)
                commands.append({"command": cmd, "cwd": root})
                if len(commands) >= limit:
                    return commands
    return commands


def _parse_json_like(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left : right + 1])
        except Exception:
            pass
    return None
