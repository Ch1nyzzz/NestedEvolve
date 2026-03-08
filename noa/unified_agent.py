"""UnifiedOptimizerAgent — 持久诊断优化 agent，完整 tool matrix + namespace 隔离。"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

from noa.core.protocol import (
    LayerContext,
    PatchOp,
    PatchValidationError,
    StructuredPatch,
    SystemDescription,
)
from noa.patch_protocol import apply_patch_ops, validate_patch_ops
from noa.core.protocol import OptimizationBudget
from noa.sandbox_manager import SandboxManager
from noa.stages.agentic import agentic_loop
from noa.stages.initiator import collect_sources
from noa.tools.probe import ComponentProbe
from noa.trajectory_store import TrajectoryStore
from noa.unified_prompts import UNIFIED_AGENT_INITIAL, UNIFIED_AGENT_SYSTEM

log = logging.getLogger(__name__)
# 确保 log 输出到 stdout
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


class UnifiedOptimizerAgent:
    """Unified optimizer agent — 完整工具集 + namespace 前缀。"""

    def __init__(
        self,
        *,
        sys_desc: SystemDescription,
        source_dir: str,
        target_factory: Callable,
        target,
        dataset: list,
        eval_fn,
        score_fn,
        model: str,
        layer_context: LayerContext,
        budget: OptimizationBudget,
        sandbox_manager: SandboxManager,
        trajectory_store: TrajectoryStore,
        component_probe: ComponentProbe | None = None,
        observer_search_roots: list[str] | None = None,
        noa_dir: str | None = None,
        project_root: str | None = None,
        dataset_pickle_path: str | None = None,
        spawn_config: dict | None = None,
        train_pool: list | None = None,
        test_set: list | None = None,
        train_sample_size: int = 25,
        n_samples: int = 30,
        top_k: int = 3,
    ):
        self.sys_desc = sys_desc
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.target = target
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.score_fn = score_fn
        self.model = model
        self.layer_context = layer_context
        self.budget = budget
        self.sandbox = sandbox_manager
        self.traj_store = trajectory_store
        self.component_probe = component_probe
        self.observer_search_roots = observer_search_roots
        self.noa_dir = noa_dir
        self.project_root = project_root
        self.dataset_pickle_path = dataset_pickle_path
        self.spawn_config = spawn_config or {}
        self.train_pool = train_pool or dataset
        self.test_set = test_set or dataset
        self.train_sample_size = train_sample_size
        self.n_samples = n_samples
        self.top_k = top_k

        self.prefix = "target" if layer_context.level <= 1 else "optimizer"
        self._file_map: dict[str, str] = {}
        self._refresh_file_map()

        # Agent 状态
        self._current_score = 0.0
        self._best_score = 0.0
        self._baseline_score = 0.0
        self._candidate_scores: dict[str, float] = {}
        self._accepted_patches = 0
        self._step_count = 0
        self._trajectories = []
        self._diagnosis = None
        self._history: list[dict] = []
        self._episode_counter = 0
        self._spawn_noa_modified = False
        self._in_escape_mode = False
        self._escape_resolved = False
        self._candidate_meta: dict[str, dict] = {}
        # 当轮训练样本 (每次 observe 时从 train_pool 随机抽取)
        self._current_train_samples: list = []
        # Top-K 候选池
        self._top_candidates: list[dict] = []  # [{label, score, ops, rationale}]

    def _refresh_file_map(self):
        """刷新源文件映射，包括 readable_roots 中的只读文件。"""
        self._file_map = {sf.path: sf.content for sf in self.sys_desc.source_files}
        # 加载 readable_roots 中不在 source_dir 内的目录（只读）
        source_abs = os.path.abspath(self.source_dir)
        for root_dir in self.layer_context.readable_roots:
            root_abs = os.path.abspath(root_dir)
            if root_abs == source_abs or root_abs.startswith(source_abs + os.sep):
                continue  # 已在 source_files 中
            for sf in collect_sources(root_abs):
                key = f"@readonly/{sf.path}"
                self._file_map[key] = sf.content

    def _compute_baseline(self):
        """用 test_set 跑一次 baseline 分数。"""
        try:
            result = self.eval_fn(self.target, self.test_set)
            score = result["score"]
            self._baseline_score = score * 100 if score <= 1 else score
            self._current_score = self._baseline_score
            print(
                f"[Baseline] test_set ({len(self.test_set)} samples) score={self._baseline_score:.2f}"
            )
        except Exception as e:
            log.error(f"[Baseline] failed: {e}")
            self._baseline_score = 0.0
            self._current_score = 0.0

    def _final_eval_top_candidates(self) -> dict | None:
        """用 test_set 对 top-K 候选做 full eval，commit 最优的那个。"""
        if not self._top_candidates:
            return None
        print(
            f"\n[FinalEval] Evaluating top-{len(self._top_candidates)} candidates on test_set ({len(self.test_set)} samples)..."
        )
        best_label, best_score = None, self._baseline_score
        for cand in self._top_candidates:
            label = cand["label"]
            candidate_dir = os.path.join(self.sandbox._candidates_dir, label)
            if not os.path.isdir(candidate_dir):
                print(f"  [FinalEval] {label}: candidate_dir missing, skip")
                continue
            result = self.sandbox.eval_in_sandbox(
                candidate_dir,
                self.target_factory,
                self.eval_fn,
                self.test_set,
                len(self.test_set),
                seed=42,
            )
            score = result.get("score", 0)
            final_score = score * 100 if score <= 1 else score
            print(
                f"  [FinalEval] {label}: test_score={final_score:.2f} (train_score={cand['score']:.2f})"
            )
            if final_score > best_score:
                best_score = final_score
                best_label = label
        if best_label:
            print(
                f"  [FinalEval] Best: {best_label} score={best_score:.2f}, committing..."
            )
            self.sandbox.accept_candidate(best_label)
            self.sys_desc.source_files = collect_sources(self.source_dir)
            self._refresh_file_map()
            self.target = self.target_factory(self.source_dir)
            self._best_score = best_score
            self._current_score = best_score
            self._accepted_patches += 1
            self._history.append(
                {
                    "action": "final_eval_commit",
                    "label": best_label,
                    "test_score": round(best_score, 2),
                    "baseline_score": round(self._baseline_score, 2),
                }
            )
            return {"label": best_label, "test_score": best_score}
        else:
            print(
                f"  [FinalEval] No candidate beats baseline ({self._baseline_score:.2f})"
            )
            return None

    def run(self) -> dict:
        """运行 unified agent 循环。"""
        # 用 test_set 跑 baseline
        self._compute_baseline()

        layer_label = self.layer_context.layer_id
        writable_name = os.path.basename(self.layer_context.writable_root.rstrip("/"))

        system_prompt = UNIFIED_AGENT_SYSTEM.format(
            layer_label=layer_label,
            target_description=self.sys_desc.workflow_summary,
            writable_root=writable_name,
            prefix=self.prefix,
            budget_summary=json.dumps(self.budget.to_summary(), indent=1),
        )

        file_list = ", ".join(sf.path for sf in self.sys_desc.source_files)
        initial_prompt = UNIFIED_AGENT_INITIAL.format(
            state_summary=f"Baseline score: {self._baseline_score:.2f} (on test_set, {len(self.test_set)} samples). No observations yet.",
            system_context=self.sys_desc.to_context_str(),
            source_files_list=file_list,
            layer_context=self.layer_context.to_prompt_context(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]
        tools = self._build_tool_schemas()

        loop_stats: dict = {}
        agentic_loop(
            messages=messages,
            tools=tools,
            tool_executor=self._dispatch_tool,
            model=self.model,
            max_tool_calls=self.budget.max_steps * 4,
            max_tokens=16384,
            early_stop_fn=self._should_stop,
            stats=loop_stats,
            no_tool_call_prompt=(
                "You must continue the optimization loop by calling tools. "
                "After analyze, you should: (1) checkpoint_candidate with the patch ops, "
                "(2) eval_candidate to measure the score, (3) accept_candidate if improved, "
                "or try a different patch. "
                "REMINDER: You MUST call spawn_sublayer before finish."
            ),
        )
        self.budget.llm_calls_used += loop_stats.get("llm_calls", 0)

        return self._build_result()

    def _should_stop(self) -> bool:
        if self._spawn_noa_modified:
            return True
        if self._in_escape_mode:
            return self._escape_resolved
        if not self.budget.reached_limit():
            return False
        # 预算耗尽 + 可spawn → 进入逃生模式（持续放行直到 spawn/finish）
        if self.layer_context and self.layer_context.can_spawn_sublayer():
            self._in_escape_mode = True
            return False
        return True

    def _build_result(self) -> dict:
        # Final eval: 用 test_set 对 top-K 候选做完整评估并 commit 最优
        final_eval = self._final_eval_top_candidates()
        # final_score 只取 final eval 的 test_set 结果（如果有 commit），否则回退到 baseline
        if final_eval and final_eval.get("test_score") is not None:
            committed_score = final_eval["test_score"]
        else:
            committed_score = self._baseline_score
        return {
            "final_score": committed_score,
            "baseline_score": self._baseline_score,
            "iterations": self._step_count,
            "accepted": self._accepted_patches,
            "history": self._history,
            "steps": self._step_count,
            "budget_usage": self.budget.to_summary(),
            "spawn_restart": self._spawn_noa_modified,
            "final_eval": final_eval,
            "top_candidates": [
                {"label": c["label"], "train_score": c["score"]}
                for c in self._top_candidates
            ],
        }

    # --- Tool Schema Builder ---

    def _build_tool_schemas(self) -> list[dict]:
        """构建带命名空间前缀的工具集。"""
        tools = []
        p = self.prefix

        file_list = list(self._file_map.keys())

        # PatchOp schema — 让 LLM 知道精确结构
        _PATCH_OP_SCHEMA = {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": [
                        "update",
                        "create",
                        "delete",
                        "insert_after",
                        "insert_before",
                    ],
                },
                "file_path": {
                    "type": "string",
                    "description": f"File path, one of: {', '.join(file_list)}",
                },
                "search": {
                    "type": "string",
                    "description": "Exact text to find (for update op)",
                },
                "replace": {
                    "type": "string",
                    "description": "Replacement text (for update op)",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content (for create op)",
                },
                "anchor": {
                    "type": "string",
                    "description": "Anchor text (for insert_after/insert_before)",
                },
                "new_lines": {
                    "type": "string",
                    "description": "Lines to insert (for insert_after/insert_before)",
                },
            },
            "required": ["op", "file_path"],
        }

        # A. 观察与探索
        tools.append(
            _tool(f"{p}__list_source_files", "List all modifiable source files.", {})
        )
        tools.append(
            _tool(
                f"{p}__read_source_file",
                "Read a source file (optionally a range).",
                {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                required=["path"],
            )
        )
        tools.append(
            _tool(
                f"{p}__search_text",
                "Search for text in source files.",
                {
                    "query": {"type": "string"},
                    "glob": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.py')",
                    },
                },
                required=["query"],
            )
        )
        tools.append(
            _tool(
                f"{p}__inspect_artifacts",
                "Inspect artifact files matching pattern.",
                {
                    "pattern": {"type": "string"},
                },
                required=["pattern"],
            )
        )

        # B. 执行与评估
        tools.append(
            _tool(
                f"{p}__run_observe",
                "Collect trajectories.",
                {
                    "n_samples": {"type": "integer", "default": 20},
                    "seed": {"type": "integer"},
                },
            )
        )
        tools.append(
            _tool(
                f"{p}__run_smoke",
                "Quick health check (3 samples).",
                {
                    "seed": {"type": "integer"},
                },
            )
        )
        tools.append(
            _tool(
                f"{p}__run_eval",
                "Evaluate system on current train samples (NOT test set).",
                {
                    "mode": {
                        "type": "string",
                        "enum": ["cheap", "full"],
                        "default": "full",
                    },
                    "n_samples": {"type": "integer"},
                    "seed": {"type": "integer"},
                },
            )
        )
        tools.append(
            _tool(
                f"{p}__run_reproduce",
                "Reproduce a specific failure case.",
                {
                    "case_id": {"type": "string"},
                },
                required=["case_id"],
            )
        )

        # C. 诊断
        tools.append(
            _tool(
                f"{p}__analyze",
                "Run error diagnosis on collected trajectories.",
                {
                    "top_n": {"type": "integer", "default": 10},
                },
            )
        )
        tools.append(
            _tool(
                f"{p}__compare_episodes",
                "Compare two observation episodes.",
                {
                    "ep_a": {"type": "string"},
                    "ep_b": {"type": "string"},
                },
                required=["ep_a", "ep_b"],
            )
        )

        # D. 编辑与验证
        tools.append(
            _tool(
                f"{p}__apply_patch",
                "Apply a structured patch (sandbox only, does NOT commit).",
                {
                    "ops": {
                        "type": "array",
                        "items": _PATCH_OP_SCHEMA,
                        "description": "List of PatchOp dicts",
                    },
                    "rationale": {"type": "string"},
                },
                required=["ops", "rationale"],
            )
        )
        tools.append(
            _tool(
                f"{p}__verify_search_block",
                "Verify if text exists verbatim in a file.",
                {
                    "path": {"type": "string"},
                    "text": {"type": "string"},
                },
                required=["path", "text"],
            )
        )
        tools.append(
            _tool(
                f"{p}__dry_run_patch",
                "Dry-run validate a patch without applying.",
                {
                    "ops": {"type": "array", "items": _PATCH_OP_SCHEMA},
                },
                required=["ops"],
            )
        )

        # E. 沙盒与候选管理
        tools.append(
            _tool(
                f"{p}__snapshot",
                "Save current state as named snapshot.",
                {
                    "label": {"type": "string"},
                },
                required=["label"],
            )
        )
        tools.append(
            _tool(
                f"{p}__restore",
                "Restore to a snapshot.",
                {
                    "label": {"type": "string"},
                },
                required=["label"],
            )
        )
        tools.append(
            _tool(
                f"{p}__checkpoint_candidate",
                "Create a candidate from clean snapshot + patch.",
                {
                    "label": {"type": "string"},
                    "ops": {"type": "array", "items": _PATCH_OP_SCHEMA},
                    "rationale": {"type": "string"},
                },
                required=["label", "ops", "rationale"],
            )
        )
        tools.append(
            _tool(
                f"{p}__eval_candidate",
                "Evaluate a candidate in sandbox.",
                {
                    "label": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["cheap", "full"],
                        "default": "full",
                    },
                    "n_samples": {"type": "integer"},
                    "seed": {"type": "integer"},
                },
                required=["label"],
            )
        )
        tools.append(
            _tool(
                f"{p}__accept_candidate",
                "Add a candidate to the top-K pool if it beats baseline. Best candidate is committed via final test eval.",
                {
                    "label": {"type": "string"},
                },
                required=["label"],
            )
        )

        # F. 状态与控制
        tools.append(_tool("get_state", "Get current optimization state summary.", {}))
        tools.append(
            _tool(
                "get_history",
                "Get optimization history.",
                {
                    "last_n": {"type": "integer", "default": 10},
                },
            )
        )
        tools.append(_tool("get_budget_status", "Get budget usage.", {}))
        tools.append(
            _tool(
                "finish",
                "End optimization.",
                {
                    "summary": {"type": "string"},
                },
                required=["summary"],
            )
        )

        # G. 嵌套层 spawn（仅当可以 spawn 时暴露）
        if self.layer_context and self.layer_context.can_spawn_sublayer():
            tools.append(
                _tool(
                    "spawn_sublayer",
                    "Spawn a higher-layer meta-optimizer (L2) to optimize YOUR code (the noa/ framework). "
                    "Use this when you've exhausted easy gains on the target and believe the optimization "
                    "framework itself can be improved. The child optimizer will modify noa/ code and run "
                    "mini-L1 evaluations to validate changes. This is expensive — use sparingly.",
                    {
                        "reason": {
                            "type": "string",
                            "description": "Why spawning L2 is needed now",
                        },
                    },
                    required=["reason"],
                )
            )

        # L1 独有：组件微实验
        if self.layer_context.level <= 1 and self.component_probe:
            comp_names = (
                list(self.component_probe.components.keys())
                or self.component_probe.workflow
            )
            tools.append(
                _tool(
                    f"{p}__run_component",
                    "Run a single component.",
                    {
                        "component_name": {"type": "string", "enum": comp_names},
                        "inputs": {"type": "object"},
                    },
                    required=["component_name", "inputs"],
                )
            )
            tools.append(
                _tool(
                    f"{p}__run_from",
                    "Run pipeline from a component to end.",
                    {
                        "start_component": {"type": "string", "enum": comp_names},
                        "inputs": {"type": "object"},
                    },
                    required=["start_component", "inputs"],
                )
            )

        return tools

    # --- Tool Dispatcher ---

    # 消耗 step budget 的重量级工具
    _HEAVY_TOOLS = frozenset(
        {
            "run_observe",
            "run_eval",
            "analyze",
            "eval_candidate",
        }
    )

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """解析命名空间前缀，分发执行。"""
        # 逃生窗口：预算耗尽后只允许 spawn 或结束
        if self._in_escape_mode and self.budget.reached_limit():
            if name not in ("spawn_sublayer", "finish"):
                return json.dumps(
                    {
                        "error": "Budget exhausted. Only spawn_sublayer or finish allowed.",
                        "hint": "You have consecutive failures. Use spawn_sublayer to let L2 optimize the framework, or call finish.",
                    }
                )
        # 只读控制工具不计入 step budget
        if name == "get_state":
            return self._tool_get_state()
        if name == "get_history":
            return self._tool_get_history(args)
        if name == "get_budget_status":
            return json.dumps(self.budget.to_summary(), indent=1)
        if name == "finish":
            # 未 spawn 过且有权 spawn → 拒绝 finish，强制先 spawn
            if (
                self.layer_context
                and self.layer_context.can_spawn_sublayer()
                and self.budget.spawn_calls_used == 0
            ):
                return json.dumps(
                    {
                        "error": "Cannot finish before spawning. You MUST call spawn_sublayer at least once before finishing.",
                        "hint": "Call spawn_sublayer now to let L2 optimize the framework, then you can finish.",
                    }
                )
            self._escape_resolved = True
            self.budget.step_count = self.budget.max_steps  # 触发停止
            return json.dumps(
                {"status": "finished", "summary": args.get("summary", "")}
            )
        if name == "spawn_sublayer":
            self._escape_resolved = True
            self._step_count += 1
            self.budget.step_count += 1
            log.info(
                "[dispatch] SPAWN tool step=%d reason=%s",
                self._step_count,
                args.get("reason", "")[:200],
            )
            try:
                result = self._tool_spawn_sublayer(args)
                result_dict = json.loads(result) if isinstance(result, str) else result
                if result_dict.get("ok"):
                    self.budget.spawn_calls_used += 1
                    self.layer_context.spawn_calls_used += 1
                    if result_dict.get("noa_modified"):
                        self._spawn_noa_modified = True
                return (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, ensure_ascii=False, default=str)
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)[:500]})

        # 带前缀的工具 — 剥离前缀
        expected_prefix = self.prefix + "__"
        if not name.startswith(expected_prefix):
            return json.dumps(
                {
                    "error": f"Invalid tool namespace: {name}, expected prefix '{expected_prefix}'"
                }
            )

        tool_name = name[len(expected_prefix) :]

        # 仅重量级工具消耗 step budget
        if tool_name in self._HEAVY_TOOLS:
            self._step_count += 1
            self.budget.step_count += 1
            log.info(
                "[dispatch] HEAVY tool=%s step=%d/%d args=%s",
                tool_name,
                self._step_count,
                self.budget.max_steps,
                json.dumps(args, ensure_ascii=False, default=str)[:500],
            )
        else:
            log.info(
                "[dispatch] light tool=%s args=%s",
                tool_name,
                json.dumps(args, ensure_ascii=False, default=str)[:300],
            )

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = handler(args)
            return (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
        except Exception as e:
            return json.dumps({"error": str(e)[:500]})

    # --- Tool Implementations ---

    def _tool_list_source_files(self, args: dict) -> dict:
        self._refresh_file_map()
        return {"files": list(self._file_map.keys()), "count": len(self._file_map)}

    def _tool_read_source_file(self, args: dict) -> str:
        path = args.get("path", "")
        content = self._file_map.get(path)
        if content is None:
            # 尝试 normpath
            for k, v in self._file_map.items():
                if os.path.normpath(k) == os.path.normpath(path):
                    content = v
                    break
        if content is None:
            return json.dumps({"error": f"File not found: {path}"})
        lines = content.split("\n")
        start = int(args.get("start_line", 1)) - 1
        end = int(args.get("end_line", len(lines)))
        selected = lines[max(0, start) : min(len(lines), end)]
        text = "\n".join(f"{i+start+1:4d} | {ln}" for i, ln in enumerate(selected))
        return text[:8000]

    def _tool_search_text(self, args: dict) -> dict:
        query = args.get("query", "")
        glob_filter = args.get("glob", "")
        results = []
        for path, content in self._file_map.items():
            if glob_filter and not _glob_match(path, glob_filter):
                continue
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    results.append(
                        {"file": path, "line": i + 1, "text": line.strip()[:200]}
                    )
                    if len(results) >= 50:
                        return {"results": results, "truncated": True}
        return {"results": results, "truncated": False}

    def _tool_inspect_artifacts(self, args: dict) -> dict:
        if self.component_probe:
            return self.component_probe.inspect_artifacts(
                args.get("pattern", "*"), self.source_dir
            )
        # L2 fallback: 直接扫描 source_dir
        import fnmatch

        pattern = args.get("pattern", "*")
        artifacts = []
        for root, _, files in os.walk(self.source_dir):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    fpath = os.path.join(root, f)
                    rel = os.path.relpath(fpath, self.source_dir)
                    try:
                        size = os.path.getsize(fpath)
                    except OSError:
                        size = 0
                    artifacts.append({"path": rel, "size": size})
                    if len(artifacts) >= 50:
                        return {
                            "count": len(artifacts),
                            "artifacts": artifacts,
                            "truncated": True,
                        }
        return {"count": len(artifacts), "artifacts": artifacts}

    def _tool_run_observe(self, args: dict) -> dict:
        import random as _random

        from noa.stages.observer import observe_agentic

        # 每轮从 train_pool 随机抽取 train_sample_size 条 (seed 随 episode 递增)
        obs_seed = 42 + self._episode_counter
        rng = _random.Random(obs_seed)
        n_samples = min(self.train_sample_size, len(self.train_pool))
        self._current_train_samples = rng.sample(self.train_pool, n_samples)
        seed = obs_seed
        trajectories = observe_agentic(
            self.target,
            self._current_train_samples,
            n_samples=n_samples,
            seed=seed,
            score_fn=self.score_fn,
            model=self.model,
            source_dir=self.source_dir,
            search_roots=self.observer_search_roots,
            max_tool_calls=8,
            layer_context=self.layer_context.to_prompt_context()
            if self.layer_context
            else "",
            layer_context_obj=self.layer_context,
            required_intermediate_keys=(
                None
                if self.layer_context and self.layer_context.level >= 2
                else getattr(self.sys_desc, "component_names", None)
            ),
        )
        self._trajectories = trajectories or []
        # 保存到 trajectory store
        self._episode_counter += 1
        ep_id = f"ep_{self._episode_counter}"
        if trajectories:
            mean = sum(t.f1 for t in trajectories) / len(trajectories) * 100
            self.traj_store.save_episode(
                ep_id, trajectories, {"step": self._step_count, "mean": mean}
            )
            print(
                f"\n[Observe] episode={ep_id} count={len(trajectories)} mean_score={mean:.2f}"
            )
            self._history.append(
                {
                    "action": "observe",
                    "step": self._step_count,
                    "episode_id": ep_id,
                    "n_trajectories": len(trajectories),
                    "mean_f1": round(mean, 2),
                }
            )
            return {
                "episode_id": ep_id,
                "count": len(trajectories),
                "mean_score": round(mean, 2),
            }
        self._history.append(
            {
                "action": "observe",
                "step": self._step_count,
                "episode_id": ep_id,
                "n_trajectories": 0,
                "mean_f1": 0,
            }
        )
        return {"episode_id": ep_id, "count": 0, "mean_score": 0}

    def _tool_run_smoke(self, args: dict) -> dict:
        import random

        seed = int(args.get("seed", 42))
        rng = random.Random(seed)
        pool = (
            self._current_train_samples
            if self._current_train_samples
            else self.train_pool
        )
        samples = rng.sample(pool, min(3, len(pool)))
        try:
            result = self.eval_fn(self.target, samples)
            return {"ok": True, "score": result["score"]}
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}

    def _tool_run_eval(self, args: dict) -> dict:
        # 优化过程中只用当轮 train 样本，不碰 test_set（test_set 仅用于 baseline + final eval）
        samples = (
            self._current_train_samples
            if self._current_train_samples
            else self.train_pool
        )
        self.budget.evals_used += 1
        try:
            result = self.eval_fn(self.target, samples)
            score = result["score"]
            new_score = score * 100 if score <= 1 else score
            if new_score > self._best_score:
                self._best_score = new_score
                self.budget.no_improve_count = 0
            else:
                self.budget.no_improve_count += 1
            self._current_score = new_score
            print(
                f"\n[Eval] score={self._current_score:.2f} (baseline={self._baseline_score:.2f}) [train samples, n={len(samples)}]"
            )
            self._history.append(
                {
                    "action": "eval",
                    "step": self._step_count,
                    "score": round(new_score, 2),
                    "n_samples": len(samples),
                }
            )
            return {
                "ok": True,
                "score": score,
                "details_count": len(result.get("details", [])),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}

    def _tool_run_reproduce(self, args: dict) -> dict:
        if self.component_probe:
            return self.component_probe.reproduce(args.get("case_id", ""))
        # L2 fallback: 直接调用 target
        case_id = args.get("case_id", "")
        try:
            result = self.target(case_id)
            return {
                "ok": True,
                "case_id": case_id,
                "output": str(getattr(result, "answer", result))[:2000],
            }
        except Exception as e:
            return {"ok": False, "case_id": case_id, "error": str(e)[:500]}

    def _tool_analyze(self, args: dict) -> dict:
        if not self._trajectories:
            return {"error": "No trajectories. Run observe first."}
        from noa.stages.analyzer import analyze_incremental

        top_n = int(args.get("top_n", 10))
        diagnosis, _ = analyze_incremental(
            self.sys_desc,
            self._trajectories,
            model=self.model,
            failure_threshold=None,
            past_attempts="",
            pool=None,
            top_n=top_n,
            probe=self.component_probe,
            max_tool_calls=6,
            layer_context=self.layer_context.to_prompt_context(),
        )
        self._diagnosis = diagnosis
        self.budget.llm_calls_used += 1
        print(f"\n[Analyze] Summary: {diagnosis.summary[:200]}")
        for i, pat in enumerate(diagnosis.failure_patterns[:top_n]):
            print(f"  Pattern {i+1}: {pat}")
        self._history.append(
            {
                "action": "analyze",
                "step": self._step_count,
                "patterns": [str(p)[:150] for p in diagnosis.failure_patterns[:top_n]],
            }
        )
        return {
            "summary": diagnosis.summary,
            "patterns": diagnosis.failure_patterns[:top_n],
            "pattern_count": len(diagnosis.failure_patterns),
        }

    def _tool_compare_episodes(self, args: dict) -> dict:
        return self.traj_store.compare_episodes(args["ep_a"], args["ep_b"])

    def _tool_apply_patch(self, args: dict) -> dict:
        ops = _parse_ops(args.get("ops", []))
        # 写保护：拒绝对只读文件的修改
        for op in ops:
            if op.file_path.startswith("@readonly/"):
                return {
                    "ok": False,
                    "errors": [
                        {"message": f"Cannot modify read-only file: {op.file_path}"}
                    ],
                }
        print(
            f"\n[ApplyPatch] ops={len(ops)} rationale={args.get('rationale', '')[:100]}"
        )
        for i, op in enumerate(ops):
            search_preview = repr(op.search[:60]) if op.search else "''"
            print(f"  op[{i}]: {op.op} file={op.file_path} search={search_preview}")
        source_files = self.sys_desc.source_files
        modified, errors, deleted_paths = apply_patch_ops(source_files, ops)
        if errors:
            print(f"  [ApplyPatch] FAILED: {[_err_dict(e) for e in errors]}")
            return {"ok": False, "errors": [_err_dict(e) for e in errors]}
        mod_map = {sf.path: sf for sf in modified}
        existing_paths = {sf.path for sf in source_files}
        # 过滤 delete，替换 update
        new_list = [
            mod_map.get(sf.path, sf)
            for sf in source_files
            if sf.path not in deleted_paths
        ]
        # 追加 create
        new_files = [sf for path, sf in mod_map.items() if path not in existing_paths]
        self.sys_desc.source_files = new_list + new_files
        # 写入沙盒（修改和新建）
        for sf in modified:
            fpath = os.path.join(self.source_dir, sf.path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(sf.content)
        # 删除沙盒文件
        for dp in deleted_paths:
            fpath = os.path.join(self.source_dir, dp)
            if os.path.exists(fpath):
                os.remove(fpath)
        self._refresh_file_map()
        print(
            f"  [ApplyPatch] OK modified={[sf.path for sf in modified]} deleted={list(deleted_paths)}"
        )
        self._history.append(
            {
                "action": "apply_patch",
                "step": self._step_count,
                "diffs": [
                    {
                        "file_path": op.file_path,
                        "search": (op.search or "")[:100],
                        "replace": (op.replace or "")[:100],
                    }
                    for op in ops
                ],
            }
        )
        return {
            "ok": True,
            "modified_files": [sf.path for sf in modified],
            "deleted_files": list(deleted_paths),
        }

    def _tool_verify_search_block(self, args: dict) -> dict:
        path = args.get("path", "")
        text = args.get("text", "")
        content = self._file_map.get(path, "")
        if not content:
            return {"found": False, "error": f"File not found: {path}"}
        found = text in content
        result = {"found": found, "file_path": path}
        if not found:
            # 附近文本提示
            first_line = text.split("\n")[0].strip() if text else ""
            for i, line in enumerate(content.split("\n")):
                if first_line and first_line in line:
                    lines = content.split("\n")
                    start = max(0, i - 1)
                    end = min(len(lines), i + text.count("\n") + 3)
                    result["nearby_lines"] = "\n".join(lines[start:end])
                    break
        return result

    def _tool_dry_run_patch(self, args: dict) -> dict:
        ops = _parse_ops(args.get("ops", []))
        errors = validate_patch_ops(self.sys_desc.source_files, ops)
        if errors:
            return {"valid": False, "errors": [_err_dict(e) for e in errors]}
        return {"valid": True, "op_count": len(ops)}

    def _tool_snapshot(self, args: dict) -> dict:
        path = self.sandbox.snapshot(args["label"])
        return {"ok": True, "label": args["label"], "path": path}

    def _tool_restore(self, args: dict) -> dict:
        self.sandbox.restore(args["label"])
        self.sys_desc.source_files = collect_sources(self.source_dir)
        self._refresh_file_map()
        self.target = self.target_factory(self.source_dir)
        return {"ok": True, "label": args["label"]}

    def _tool_checkpoint_candidate(self, args: dict) -> dict:
        raw_ops = args.get("ops", [])
        ops = _parse_ops(raw_ops)
        print(
            f"\n[Checkpoint] label={args['label']} ops={len(ops)} raw_ops_type={type(raw_ops).__name__} raw_ops_len={len(raw_ops) if isinstance(raw_ops, (list, str)) else 'N/A'} rationale={args.get('rationale', '')[:100]}"
        )
        if not ops:
            print(f"  [Checkpoint] WARNING: 0 ops! raw_ops={repr(str(raw_ops))[:200]}")
            return {
                "ok": False,
                "errors": [
                    {
                        "message": "No ops provided. You must pass the patch ops to checkpoint_candidate."
                    }
                ],
            }
        for i, op in enumerate(ops):
            search_preview = repr(op.search[:60]) if op.search else "''"
            print(f"  op[{i}]: {op.op} file={op.file_path} search={search_preview}")
        ops_summary = [
            {
                "op": op.op,
                "file_path": op.file_path,
                "search": (op.search or "")[:100],
                "replace": (op.replace or "")[:100],
            }
            for op in ops
        ]
        patch = StructuredPatch(ops=ops, rationale=args.get("rationale", ""))
        candidate_dir, errors = self.sandbox.checkpoint_candidate(args["label"], patch)
        if errors:
            print(f"  [Checkpoint] FAILED: {[_err_dict(e) for e in errors]}")
            # 从 checkpoint 基线读取正确版本的文件（而非 working copy）
            base_dir = self.sandbox._get_checkpoint_base()
            base_files = {sf.path: sf.content for sf in collect_sources(base_dir)}
            enriched = []
            for e in errors:
                d = _err_dict(e)
                if e.code == "search_not_found" and e.op_index < len(ops):
                    needle = ops[e.op_index].search
                    first_line = needle.split("\n")[0].strip() if needle else ""
                    content = base_files.get(e.file_path, "")
                    if first_line and content:
                        for j, line in enumerate(content.split("\n")):
                            if first_line in line:
                                lines = content.split("\n")
                                start, end = max(0, j - 2), min(len(lines), j + 5)
                                d["nearby"] = "\n".join(
                                    f"{k+1}: {lines[k]}" for k in range(start, end)
                                )
                                break
                enriched.append(d)
            self._history.append(
                {
                    "action": "checkpoint_candidate",
                    "step": self._step_count,
                    "label": args["label"],
                    "ok": False,
                    "rationale": args.get("rationale", "")[:300],
                    "ops": ops_summary,
                    "errors": enriched,
                }
            )
            return {"ok": False, "errors": enriched}
        print(f"  [Checkpoint] OK dir={candidate_dir}")
        self._candidate_meta[args["label"]] = {
            "label": args["label"],
            "rationale": args.get("rationale", "")[:500],
            "ops": ops_summary,
            "step": self._step_count,
        }
        self._history.append(
            {
                "action": "checkpoint_candidate",
                "step": self._step_count,
                "label": args["label"],
                "ok": True,
                "rationale": args.get("rationale", "")[:300],
                "ops": ops_summary,
            }
        )
        return {"ok": True, "label": args["label"], "candidate_dir": candidate_dir}

    def _update_top_candidates(self, label: str, score: float):
        """更新 top-K 候选池。"""
        meta = self._candidate_meta.get(label, {})
        entry = {
            "label": label,
            "score": score,
            "ops": meta.get("ops", []),
            "rationale": meta.get("rationale", ""),
        }
        # 检查是否已存在同 label 的候选
        self._top_candidates = [c for c in self._top_candidates if c["label"] != label]
        self._top_candidates.append(entry)
        # 按分数降序排序，保留 top-K
        self._top_candidates.sort(key=lambda c: c["score"], reverse=True)
        self._top_candidates = self._top_candidates[: self.top_k]

    def _tool_eval_candidate(self, args: dict) -> dict:
        label = args["label"]
        candidate_dir = os.path.join(self.sandbox._candidates_dir, label)
        if not os.path.isdir(candidate_dir):
            return {"error": f"Candidate not found: {label}"}
        mode = args.get("mode", "full")
        # cheap 和 full 都在当轮 train 样本上评估
        val_data = (
            self._current_train_samples
            if self._current_train_samples
            else self.train_pool
        )
        eval_n = len(val_data)
        eval_seed = 42
        self.budget.evals_used += 1
        result = self.sandbox.eval_in_sandbox(
            candidate_dir,
            self.target_factory,
            self.eval_fn,
            val_data,
            eval_n,
            seed=eval_seed,
        )
        score = result.get("score", result.get("mean_score", 0))
        # 归一化到 0~100，与 run_eval 保持一致
        normalized_score = score * 100 if score <= 1 else score
        self._candidate_scores[label] = normalized_score
        # 注意: 不在 eval 阶段加入 top_candidates，只在 accept_candidate 时加入
        best_score = max(self._best_score, self._baseline_score)
        candidate_meta = self._candidate_meta.get(label, {})
        accepted = normalized_score > best_score
        # 检测 subprocess 崩溃（L2 场景）
        subprocess_errors = result.get("subprocess_errors", [])
        if subprocess_errors:
            print(
                f"\n[EvalCandidate] label={label} score={normalized_score:.2f} "
                f"SUBPROCESS ERRORS ({len(subprocess_errors)}):"
            )
            for err in subprocess_errors[:2]:
                print(f"  {err[:200]}")
        else:
            print(
                f"\n[EvalCandidate] label={label} score={normalized_score:.2f} (baseline={self._baseline_score:.2f}, best={self._best_score:.2f}, current={self._current_score:.2f})"
            )
        # 合并 error 信息
        eval_error = result.get("error")
        if not eval_error and subprocess_errors:
            eval_error = f"subprocess_crash: {subprocess_errors[0][:300]}"
        self._history.append(
            {
                "action": "eval_candidate",
                "step": self._step_count,
                "label": label,
                "mode": mode,
                "rationale": candidate_meta.get("rationale", ""),
                "ops": candidate_meta.get("ops", []),
                "before_score": round(best_score, 2),
                "after_score": round(normalized_score, 2),
                "accepted": accepted,
                "details_count": len(result.get("details", [])),
                "error": eval_error,
            }
        )
        return result

    def _tool_accept_candidate(self, args: dict) -> dict:
        """接受候选 — 加入 top-K 池，不直接 commit（final eval 时再 commit 最优的）。"""
        label = args["label"]
        cand_score = self._candidate_scores.get(label, 0)
        best_train_score = max(
            (c["score"] for c in self._top_candidates), default=self._baseline_score
        )
        print(
            f"\n[AcceptCandidate] label={label} candidate_score={cand_score:.2f} "
            f"baseline={self._baseline_score:.2f} best_train={best_train_score:.2f}"
        )
        # 检查是否已在 top-K 中
        in_top = any(c["label"] == label for c in self._top_candidates)
        if not in_top and cand_score <= self._baseline_score:
            print(
                f"  [AcceptCandidate] REJECTED: candidate {cand_score:.2f} <= baseline {self._baseline_score:.2f}"
            )
            self.budget.no_improve_count += 1
            candidate_meta = self._candidate_meta.get(label, {})
            self._history.append(
                {
                    "action": "reject_candidate",
                    "step": self._step_count,
                    "label": label,
                    "rationale": candidate_meta.get("rationale", ""),
                    "ops": candidate_meta.get("ops", []),
                    "reason": f"score {cand_score:.2f} <= baseline {self._baseline_score:.2f}",
                }
            )
            return {
                "ok": False,
                "error": f"Candidate score {cand_score:.2f} not better than baseline {self._baseline_score:.2f}. Not added to top-{self.top_k} pool.",
            }
        # 加入 top-K 池（已在 eval_candidate 时更新过，这里确认）
        self._update_top_candidates(label, cand_score)
        self.budget.no_improve_count = 0
        if cand_score > self._best_score:
            self._best_score = cand_score
        self._current_score = cand_score
        candidate_meta = self._candidate_meta.get(label, {})
        self._history.append(
            {
                "action": "accept_candidate",
                "label": label,
                "step": self._step_count,
                "rationale": candidate_meta.get("rationale", ""),
                "ops": candidate_meta.get("ops", []),
                "score": cand_score,
            }
        )
        top_labels = [c["label"] for c in self._top_candidates]
        print(
            f"  [AcceptCandidate] Added to top-{self.top_k} pool. "
            f"Current pool: {top_labels}"
        )
        return {"ok": True, "label": label, "top_candidates": top_labels}

    def _tool_spawn_sublayer(self, args: dict) -> dict:
        """Spawn L2 meta-optimizer to optimize noa/ framework code."""
        from types import SimpleNamespace

        from noa.engine import NOptimizer
        from noa.subprocess_runner import run_layer_subprocess, serialize_dataset

        if not self.layer_context or not self.layer_context.can_spawn_sublayer():
            return {"ok": False, "error": "Cannot spawn: depth/budget limit reached"}
        if not self.noa_dir or not self.project_root:
            return {
                "ok": False,
                "error": "Cannot spawn: noa_dir or project_root not set",
            }

        child_level = self.layer_context.level + 1
        noa_dir = self.noa_dir
        project_root = self.project_root

        print(f"\n[Spawn] Starting L{child_level} meta-optimizer (noa_dir={noa_dir})")

        # dataset pickle
        dpp = self.dataset_pickle_path
        cache_dir = os.path.join(project_root, ".noa_cache")
        if not dpp:
            dpp = serialize_dataset(self.dataset, cache_dir=cache_dir)

        # 序列化 train_pool / test_set 供 mini-L1 subprocess 使用
        train_pool_pp = (
            serialize_dataset(self.train_pool, cache_dir=cache_dir)
            if self.train_pool
            else None
        )
        test_set_pp = (
            serialize_dataset(self.test_set, cache_dir=cache_dir)
            if self.test_set
            else None
        )
        _train_sample_size = self.train_sample_size
        _top_k = self.top_k

        # child target_factory: 每个 question 跑一次 mini-L1 subprocess
        ml1 = self.spawn_config.get("mini_l1", {})

        def child_target_factory(noa_source_dir):
            def target(question):
                run_seed = hash(question) & 0x7FFFFFFF
                result = run_layer_subprocess(
                    noa_dir=noa_source_dir,
                    project_root=project_root,
                    target_source_dir=self.source_dir,
                    dataset_pickle_path=dpp,
                    layer_level=1,
                    max_steps=ml1.get("max_steps", 8),
                    n_samples=ml1.get("n_samples", 10),
                    max_llm_calls=ml1.get("max_llm_calls", 40),
                    max_evals=ml1.get("max_evals", 4),
                    max_no_improve_steps=ml1.get("max_no_improve_steps", 3),
                    model=self.model,
                    isolate_source=True,
                    random_seed=run_seed,
                    train_pool_pickle_path=train_pool_pp,
                    test_set_pickle_path=test_set_pp,
                    train_sample_size=_train_sample_size,
                    top_k=_top_k,
                )
                return SimpleNamespace(
                    answer=str(result.get("final_score", 0)),
                    intermediate=result,
                )

            return target

        # child score_fn — 直接用 final_score 原始值（已是 0-100 尺度）
        # prediction = str(final_score), ground_truth 不使用
        def child_score_fn(prediction: str, ground_truth: str) -> float:
            try:
                return float(str(prediction).strip()) / 100.0
            except (TypeError, ValueError):
                return 0.0

        # child eval_fn — 与 child_score_fn 保持同一尺度
        def child_eval_fn(target, dataset):
            scores, details, errors = [], [], []
            for ex in dataset:
                kwargs = {"question": ex.question}
                if getattr(ex, "context", ""):
                    kwargs["context"] = ex.context
                result = target(**kwargs)
                raw = float(result.answer) if result.answer else 0.0
                s = raw / 100.0  # final_score 是 0-100，归一化到 0-1
                scores.append(s)
                detail = {"question": ex.question, "f1": round(s, 4), "raw_score": raw}
                inter = getattr(result, "intermediate", {}) or {}
                if inter.get("error"):
                    detail["error"] = inter["error"][:500]
                    detail["error_type"] = inter.get("error_type", "unknown")
                    errors.append(inter["error"][:500])
                if inter.get("history"):
                    detail["accepted"] = inter.get("accepted", 0)
                    detail["steps"] = inter.get("steps", 0)
                details.append(detail)
            avg = sum(scores) / len(scores) if scores else 0.0
            out = {"score": avg, "details": details}
            if errors:
                out["subprocess_errors"] = errors
            return out

        # child dataset — 每次 eval 跑 1 个完整 L1
        child_dataset = [
            SimpleNamespace(question="opt_run_1", answer="0"),
        ]

        # parent history for L2 context
        parent_history = [
            {
                "iteration": i + 1,
                "accepted": True,
                "label": h.get("label", ""),
                "step": h.get("step", 0),
            }
            for i, h in enumerate(self._history)
            if h.get("action") == "accept_candidate"
        ]
        parent_summary = (
            f"Initial: {self._baseline_score:.2f}, Current: {self._current_score:.2f}, "
            f"Delta: {self._current_score - self._baseline_score:+.2f}, "
            f"Accepted: {self._accepted_patches}"
        )

        child_layer_context = LayerContext(
            layer_id=f"L{child_level}",
            level=child_level,
            writable_root=noa_dir,
            readable_roots=[noa_dir, self.source_dir],
            parent_history=parent_history,
            parent_summary=parent_summary,
            max_depth=self.layer_context.max_depth,
            max_spawn_calls=max(0, self.layer_context.max_spawn_calls - 1),
        )

        l2_cfg = self.spawn_config.get("l2", {})
        try:
            child = NOptimizer(
                source_dir=noa_dir,
                target_factory=child_target_factory,
                dataset=child_dataset,
                eval_fn=child_eval_fn,
                max_steps=l2_cfg.get("max_steps", 12),
                n_samples=l2_cfg.get("n_samples", 2),
                model=self.model,
                score_fn=child_score_fn,
                max_llm_calls=l2_cfg.get("max_llm_calls", 60),
                max_evals=l2_cfg.get("max_evals", 8),
                max_no_improve_steps=l2_cfg.get("max_no_improve_steps", 4),
                layer_context=child_layer_context,
                observer_search_roots=[noa_dir],
            )
            child_result = child.run()
        except Exception as e:
            log.error(f"[Spawn] L{child_level} crashed: {e}", exc_info=True)
            return {"ok": False, "error": f"L{child_level} crashed: {str(e)[:500]}"}

        child_accepted = int(child_result.get("accepted", 0) or 0)
        child_spawn_restart = bool(child_result.get("spawn_restart"))
        noa_modified = child_accepted > 0 or child_spawn_restart

        self._history.append(
            {
                "action": "spawn_sublayer",
                "step": self._step_count,
                "child_level": child_level,
                "child_score": child_result.get("final_score", 0),
                "child_accepted": child_accepted,
                "noa_modified": noa_modified,
            }
        )

        print(
            f"[Spawn] L{child_level} done: score={child_result.get('final_score', 0):.2f}, "
            f"accepted={child_accepted}, noa_modified={noa_modified}"
        )

        return {
            "ok": True,
            "noa_modified": noa_modified,
            "child_score": child_result.get("final_score", 0),
            "child_accepted": child_accepted,
            "child_steps": child_result.get("steps", 0),
        }

    def _tool_run_component(self, args: dict) -> dict:
        if not self.component_probe:
            return {"error": "No component probe available"}
        return self.component_probe.run_component(
            args["component_name"], args.get("inputs", {})
        )

    def _tool_run_from(self, args: dict) -> dict:
        if not self.component_probe:
            return {"error": "No component probe available"}
        return self.component_probe.run_from(
            args["start_component"], args.get("inputs", {})
        )

    def _tool_get_state(self) -> str:
        recent_candidate_events = [
            h
            for h in self._history
            if h.get("action")
            in (
                "checkpoint_candidate",
                "eval_candidate",
                "reject_candidate",
                "accept_candidate",
            )
        ][-10:]
        state = {
            "baseline_score": self._baseline_score,
            "current_score": self._current_score,
            "accepted_patches": self._accepted_patches,
            "step_count": self._step_count,
            "has_trajectories": bool(self._trajectories),
            "trajectory_count": len(self._trajectories),
            "has_diagnosis": self._diagnosis is not None,
            "snapshots": self.sandbox.list_snapshots(),
            "episodes": self.traj_store.list_episodes(),
            "recent_candidate_events": recent_candidate_events,
            "top_candidates": [
                {"label": c["label"], "train_score": c["score"]}
                for c in self._top_candidates
            ],
            "train_sample_size": len(self._current_train_samples),
        }
        return json.dumps(state, indent=1)

    def _tool_get_history(self, args: dict) -> str:
        n = int(args.get("last_n", 10))
        return json.dumps(self._history[-n:], indent=1, default=str)


# --- Helpers ---


def _tool(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
    """构建 OpenAI function calling 格式工具定义。"""
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": schema},
    }


def _parse_ops(raw_ops) -> list[PatchOp]:
    """从 LLM 输出的 dict 列表解析为 PatchOp 对象。支持字符串形式的 JSON。"""
    if isinstance(raw_ops, str):
        raw_ops = json.loads(raw_ops)
    if not isinstance(raw_ops, list):
        raw_ops = [raw_ops]
    ops = []
    for raw in raw_ops:
        op = PatchOp(
            op=raw.get("op", "update"),
            file_path=raw.get("file_path", ""),
            search=raw.get("search", ""),
            replace=raw.get("replace", ""),
            occurrence=int(raw.get("occurrence", 1)),
            must_be_unique=bool(raw.get("must_be_unique", True)),
            context_before=raw.get("context_before", ""),
            context_after=raw.get("context_after", ""),
            content=raw.get("content", ""),
            anchor=raw.get("anchor", ""),
            anchor_occurrence=int(raw.get("anchor_occurrence", 1)),
            anchor_must_be_unique=bool(raw.get("anchor_must_be_unique", True)),
            new_lines=raw.get("new_lines", ""),
        )
        ops.append(op)
    return ops


def _err_dict(e: PatchValidationError) -> dict:
    return {
        "op_index": e.op_index,
        "code": e.code,
        "file_path": e.file_path,
        "message": e.message,
    }


def _glob_match(path: str, pattern: str) -> bool:
    """简单 glob 匹配。"""
    import fnmatch

    return fnmatch.fnmatch(path, pattern)
