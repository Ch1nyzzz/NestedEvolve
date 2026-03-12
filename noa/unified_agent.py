"""UnifiedOptimizerAgent — 持久诊断优化 agent，完整 tool matrix + namespace 隔离。"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
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
from noa.runtime import bind_scope, record_run_event
from noa.sandbox_manager import SandboxManager
from noa.stages.agentic import agentic_loop
from noa.stages.initiator import collect_sources
from noa.tools.probe import ComponentProbe
from noa.trajectory_store import TrajectoryStore
from noa.unified_prompts import UNIFIED_AGENT_INITIAL, UNIFIED_AGENT_SYSTEM

# 默认总墙钟预算 (秒): 8 小时
DEFAULT_WALL_BUDGET_SEC = 28800
# 子操作预留余量 (秒): 给上层处理结果留时间
_DEADLINE_MARGIN_SEC = 120
# 拒绝启动 eval 的最低剩余时间 (秒)
_MIN_REMAINING_FOR_EVAL_SEC = 300

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
        initial_baseline_score: float | None = None,
        wall_budget_sec: float = DEFAULT_WALL_BUDGET_SEC,
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
        self._initial_baseline_score = initial_baseline_score

        # 统一 deadline: 绝对单调时间 (monotonic)
        self._wall_budget_sec = wall_budget_sec
        self._deadline = time.monotonic() + wall_budget_sec

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
        self._final_eval_result: dict | None = None
        self._final_eval_trigger: str | None = None
        self._spawn_ready_baseline_score: float | None = None
        # 当轮训练样本 (每次 observe 时从 train_pool 随机抽取)
        self._current_train_samples: list = []
        # 当轮 train 样本上的 baseline 分数 (每次 observe 时重新计算)
        self._current_train_baseline: float = 0.0
        # Top-K 候选池
        self._top_candidates: list[dict] = []  # [{label, score, ops, rationale}]
        # compact 计数器
        self._compact_chunk_counter = 0

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

    def _write_detail_file(self, filename: str, payload: dict | list) -> str:
        """写 payload 到 source_dir/.noa_meta/{filename}，返回可被 read_source_file 读取的路径。"""
        meta_dir = os.path.join(self.source_dir, ".noa_meta")
        os.makedirs(meta_dir, exist_ok=True)
        fpath = os.path.join(meta_dir, filename)
        text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        rel = ".noa_meta/" + filename
        self._file_map[rel] = text
        return rel

    def _push_history(self, entry: dict) -> None:
        self._history.append(entry)
        action = entry.get("action")
        if not action:
            return
        payload = {k: v for k, v in entry.items() if k != "action"}
        record_run_event(action, **payload)

    def _compact_messages(self, messages: list[dict]) -> list[dict]:
        """压缩旧 messages: 原始记录存文件 + LLM 生成摘要 + 保留最近 KEEP_RECENT 条。
        L2+ 层迭代少、上下文轻，跳过 compact。
        """
        # L2+ 层不需要 compact — 每次 spawn 都是全新 agent，messages 量小
        if self.layer_context and self.layer_context.level >= 2:
            return messages
        KEEP_RECENT = 10
        if len(messages) <= KEEP_RECENT + 2:
            return messages

        head = messages[:2]  # system + initial
        tail = messages[-KEEP_RECENT:]
        old = messages[2:-KEEP_RECENT]

        # 确保 tail 的第一条不是孤立的 tool response
        while tail and tail[0].get("role") == "tool" and old:
            old.append(tail.pop(0))

        if not tail:
            return messages

        # 1) 保存原始 messages 到 detail file，供 agent 按需读取
        self._compact_chunk_counter += 1
        chunk_id = self._compact_chunk_counter
        # 保存完整消息（包括 tool_calls 和 tool_call_id）
        raw_records = []
        for m in old:
            rec = {"role": m.get("role", ""), "content": m.get("content", "")[:4000]}
            if m.get("tool_calls"):
                rec["tool_calls"] = [
                    {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"][:500],
                    }
                    for tc in m["tool_calls"]
                    if isinstance(tc, dict) and "function" in tc
                ]
            if m.get("tool_call_id"):
                rec["tool_call_id"] = m["tool_call_id"]
            raw_records.append(rec)
        raw_file = self._write_detail_file(
            f"compact_raw_{chunk_id}.json",
            raw_records,
        )

        # 2) LLM 生成结构化摘要
        summary = self._generate_compact_summary(old, chunk_id, raw_file)

        compact_msg = {
            "role": "user",
            "content": summary,
        }
        return head + [compact_msg] + tail

    def _generate_compact_summary(
        self, old_messages: list[dict], chunk_id: int, raw_file: str
    ) -> str:
        """用 LLM 对被压缩的消息生成结构化摘要；失败时 fallback 到规则提取。"""
        # 提取关键事件用于 LLM 摘要输入
        events = []
        for msg in old_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                # agent 的推理 — 截取核心部分
                if content.strip():
                    events.append(f"[Agent reasoning] {content[:600]}")
            elif role == "tool":
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        slim = {
                            k: data[k]
                            for k in (
                                "action",
                                "score",
                                "candidate_score",
                                "ok",
                                "error",
                                "label",
                                "episode_id",
                                "summary",
                                "count",
                                "candidate_label",
                                "next_action",
                                "top_patterns",
                                "mean_score",
                                "detail_file",
                            )
                            if k in data
                        }
                        if slim:
                            events.append(
                                f"[Tool result] {json.dumps(slim, ensure_ascii=False)}"
                            )
                            continue
                except (json.JSONDecodeError, TypeError):
                    pass
                events.append(f"[Tool result] {content[:400]}")

        events_text = "\n".join(events[:40])

        # 构建 LLM 摘要请求
        prompt = (
            "Summarize the following optimization session events into a concise structured digest.\n"
            "Focus on: (1) what was observed/diagnosed, (2) what patches were tried and their results, "
            "(3) key insights and lessons learned, (4) current state.\n"
            "Output format:\n"
            "## Observations\n- ...\n"
            "## Patches Attempted\n- patch_label: result (score)\n"
            "## Key Insights\n- ...\n"
            "## Current State\nscore=X, baseline=Y\n\n"
            f"Events:\n{events_text}"
        )

        try:
            from utils.llm import llm_call

            resp = llm_call(
                prompt,
                model=self.model,
                max_tokens=800,
                temperature=0,
                system="You are a concise summarizer for optimization logs. Output only the structured digest.",
            )
            digest = resp.text.strip() if resp.text else ""
            if len(digest) > 100:
                # 保存摘要到文件
                summary_file = self._write_detail_file(
                    f"compact_summary_{chunk_id}.md", {"summary": digest}
                )
                return (
                    f"[Context Compact] Earlier {len(old_messages)} messages compressed.\n\n"
                    f"{digest}\n\n"
                    f"Full raw messages: read_source_file(path='{raw_file}')\n"
                    f"Summary file: read_source_file(path='{summary_file}')"
                )
        except Exception as e:
            log.warning(
                "[compact] LLM summary failed: %s, falling back to rule-based",
                str(e)[:200],
            )

        # Fallback: 规则提取
        return self._rule_based_compact_summary(old_messages, raw_file)

    def _rule_based_compact_summary(
        self, old_messages: list[dict], raw_file: str
    ) -> str:
        """规则提取摘要 — LLM 失败时的 fallback。"""
        summary_parts = []
        for msg in old_messages:
            if msg.get("role") == "assistant":
                text = msg.get("content", "").strip()
                if text:
                    summary_parts.append(f"[Agent] {text[:300]}")
            elif msg.get("role") == "tool":
                content = msg.get("content", "")
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        slim = {
                            k: data[k]
                            for k in (
                                "action",
                                "score",
                                "candidate_score",
                                "ok",
                                "error",
                                "label",
                                "episode_id",
                                "summary",
                                "count",
                            )
                            if k in data
                        }
                        if slim:
                            summary_parts.append(json.dumps(slim, ensure_ascii=False))
                            continue
                except (json.JSONDecodeError, TypeError):
                    pass
                summary_parts.append(
                    content[:400] + "..." if len(content) > 400 else content
                )
        return (
            f"[Context Compact] Earlier {len(old_messages)} messages compressed. "
            f"Key events:\n"
            + "\n".join(summary_parts[:30])
            + f"\n\nFull raw messages: read_source_file(path='{raw_file}')"
        )

    def _compute_baseline(self):
        """用 test_set 跑一次 baseline 分数。"""
        if self._initial_baseline_score is not None:
            self._baseline_score = float(self._initial_baseline_score)
            self._current_score = self._baseline_score
            print(f"[Baseline] using precomputed score={self._baseline_score:.2f}")
            return
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

    def _final_eval_top_candidates(self, trigger: str = "finish") -> dict | None:
        """用 test_set 对 top-K 候选做 full eval，commit 最优的那个。
        L2+ 层: mini-L1 的 eval_candidate 已在 test_set 上做过 final eval，
        跳过冗余 FinalEval，直接用已有分数 commit 最优候选。
        """
        if not self._top_candidates:
            self._push_history(
                {
                    "action": "final_eval_skipped",
                    "step": self._step_count,
                    "trigger": trigger,
                    "reason": "no_top_candidates",
                    "baseline_score": round(self._baseline_score, 2),
                }
            )
            return None

        # L2+ 层: eval_candidate 返回的分数已经是 mini-L1 的 test_set final eval，
        # 再跑一次 FinalEval 只会引入随机噪声。直接用已有分数 commit 最优。
        if self.layer_context and self.layer_context.level >= 2:
            best_cand = max(self._top_candidates, key=lambda c: c["score"])
            best_label = best_cand["label"]
            best_score = best_cand["score"]
            if best_score > self._baseline_score:
                print(
                    f"\n[FinalEval] L2+ skip re-eval: committing {best_label} "
                    f"(eval_score={best_score:.2f}, baseline={self._baseline_score:.2f})"
                )
                self.sandbox.accept_candidate(best_label)
                self.sys_desc.source_files = collect_sources(self.source_dir)
                self._refresh_file_map()
                self.target = self.target_factory(self.source_dir)
                self._best_score = best_score
                self._current_score = best_score
                self._accepted_patches += 1
                self._push_history(
                    {
                        "action": "final_eval_commit",
                        "step": self._step_count,
                        "trigger": trigger,
                        "label": best_label,
                        "test_score": round(best_score, 2),
                        "baseline_score": round(self._baseline_score, 2),
                        "l2_skip_reeval": True,
                    }
                )
                return {"label": best_label, "test_score": best_score}
            else:
                print(
                    f"\n[FinalEval] L2+ skip re-eval: no candidate beats baseline "
                    f"(best={best_score:.2f}, baseline={self._baseline_score:.2f})"
                )
                self._push_history(
                    {
                        "action": "final_eval_no_commit",
                        "step": self._step_count,
                        "trigger": trigger,
                        "baseline_score": round(self._baseline_score, 2),
                        "top_candidate_count": len(self._top_candidates),
                        "l2_skip_reeval": True,
                    }
                )
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
            self._push_history(
                {
                    "action": "final_eval_commit",
                    "step": self._step_count,
                    "trigger": trigger,
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
            self._push_history(
                {
                    "action": "final_eval_no_commit",
                    "step": self._step_count,
                    "trigger": trigger,
                    "baseline_score": round(self._baseline_score, 2),
                    "top_candidate_count": len(self._top_candidates),
                }
            )
            return None

    def _run_final_eval_once(self, trigger: str) -> dict | None:
        """只执行一次 held-out final eval，供 pre-spawn 和 finish 共享。"""
        if self._final_eval_trigger is not None:
            return self._final_eval_result
        self._final_eval_trigger = trigger
        self._final_eval_result = self._final_eval_top_candidates(trigger=trigger)
        return self._final_eval_result

    def _prepare_spawn_baseline(self) -> float:
        """spawn 前先固化一次 test-set 结果，供 L2 直接复用。"""
        if self._spawn_ready_baseline_score is not None:
            return self._spawn_ready_baseline_score

        print("\n[SpawnPrep] Running pre-spawn test-set final eval...")
        final_eval = self._run_final_eval_once(trigger="pre_spawn")
        if final_eval and final_eval.get("test_score") is not None:
            self._spawn_ready_baseline_score = float(final_eval["test_score"])
        else:
            self._spawn_ready_baseline_score = self._baseline_score
        print(f"[SpawnPrep] L2 baseline seed={self._spawn_ready_baseline_score:.2f}")
        return self._spawn_ready_baseline_score

    def _write_parent_context_artifact(
        self, parent_summary: str, baseline_score: float
    ) -> str | None:
        """将完整父层上下文写入只读 JSON，供 L2 按需读取。"""
        if not self.source_dir:
            return None
        meta_dir = os.path.join(self.source_dir, ".noa_meta")
        os.makedirs(meta_dir, exist_ok=True)
        artifact_path = os.path.join(meta_dir, "l1_parent_context.json")
        payload = {
            "parent_summary": parent_summary,
            "baseline_score": baseline_score,
            "current_score": self._current_score,
            "accepted_patches": self._accepted_patches,
            "step_count": self._step_count,
            "top_candidates": deepcopy(self._top_candidates),
            "final_eval": deepcopy(self._final_eval_result),
            "history": deepcopy(self._history),
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        rel_path = os.path.relpath(artifact_path, self.source_dir)
        return f"@readonly/{rel_path}"

    def _run_mini_l1_subprocess(
        self,
        *,
        noa_source_dir: str,
        project_root: str,
        dataset_pickle_path: str,
        train_pool_pickle_path: str | None,
        test_set_pickle_path: str | None,
        question: str,
        ml1_cfg: dict,
        train_sample_size: int,
        top_k: int,
    ) -> dict:
        """运行 mini-L1 subprocess，对瞬时失败做有限重试。"""
        from noa.subprocess_runner import run_layer_subprocess

        retries = int(ml1_cfg.get("subprocess_retries", 1))
        timeout = self._nested_mini_l1_timeout(ml1_cfg)
        transient_types = {"timeout", "llm_connection_error", "unknown"}
        last_result: dict | None = None

        for attempt in range(retries + 1):
            run_seed = hash((question, attempt)) & 0x7FFFFFFF
            result = run_layer_subprocess(
                noa_dir=noa_source_dir,
                project_root=project_root,
                target_source_dir=self.source_dir,
                dataset_pickle_path=dataset_pickle_path,
                layer_level=1,
                max_steps=ml1_cfg.get("max_steps", 8),
                n_samples=ml1_cfg.get("n_samples", 10),
                max_llm_calls=ml1_cfg.get("max_llm_calls", 999999),
                max_evals=ml1_cfg.get("max_evals", 4),
                max_no_improve_steps=ml1_cfg.get("max_no_improve_steps", 3),
                model=self.model,
                isolate_source=True,
                random_seed=run_seed,
                train_pool_pickle_path=train_pool_pickle_path,
                test_set_pickle_path=test_set_pickle_path,
                train_sample_size=train_sample_size,
                top_k=top_k,
                timeout=timeout,
                attempt_index=attempt,
            )
            if not result.get("error"):
                if attempt > 0:
                    result["subprocess_retry_count"] = attempt
                return result
            last_result = result
            error_type = result.get("error_type", "unknown")
            if attempt >= retries or error_type not in transient_types:
                break
            print(
                f"[Spawn] mini-L1 subprocess retry {attempt + 1}/{retries} after {error_type}"
            )

        if last_result is None:
            last_result = {
                "final_score": 0,
                "error": "mini-L1 subprocess failed without result",
                "error_type": "unknown",
            }
        last_result["subprocess_retry_count"] = retries
        return last_result

    def _remaining_sec(self) -> float:
        """返回距离 deadline 的剩余秒数 (可能为负)。"""
        return self._deadline - time.monotonic()

    def _nested_mini_l1_timeout(self, ml1_cfg: dict | None = None) -> int:
        """为嵌套 mini-L1 评估选择超时，基于剩余预算动态计算。

        优先级: 剩余预算 - margin > 配置显式值 > 层级默认值。
        确保 mini_l1_timeout <= 剩余预算 - margin。
        """
        cfg = ml1_cfg if ml1_cfg is not None else self.spawn_config.get("mini_l1", {})

        # 基于剩余时间计算上限
        remaining = self._remaining_sec()
        budget_limit = max(0, int(remaining - _DEADLINE_MARGIN_SEC))

        # 配置显式值
        raw = cfg.get("timeout")
        if raw is not None:
            configured = int(raw)
        elif self.layer_context and self.layer_context.level >= 2:
            configured = 14400  # 4h default for L2
        else:
            configured = 14400  # 4h default for L1

        # 取两者中较小的
        timeout = min(configured, budget_limit)
        if timeout < _MIN_REMAINING_FOR_EVAL_SEC:
            log.warning(
                "[timeout] mini_l1 timeout=%ds < minimum %ds, remaining=%.0fs",
                timeout,
                _MIN_REMAINING_FOR_EVAL_SEC,
                remaining,
            )
        return max(timeout, 0)

    def _nested_eval_timeout(self) -> float | None:
        """候选评估超时，从剩余预算动态计算。

        L1: 基于剩余预算 (不再返回 None)
        L2+: mini_l1_timeout + 60s margin, 但不超过剩余预算
        """
        remaining = self._remaining_sec()
        budget_limit = max(0, remaining - _DEADLINE_MARGIN_SEC)

        if not self.layer_context or self.layer_context.level < 2:
            # L1: sandbox eval 用 run_forked, 给它剩余预算
            return budget_limit if budget_limit > 0 else 600.0

        # L2+: 外层 sandbox 需要比内层 mini-L1 更长
        inner = self._nested_mini_l1_timeout()
        desired = float(inner + 60)
        return min(desired, budget_limit)

    def _check_time_for_eval(self, operation: str = "eval") -> str | None:
        """检查是否有足够时间启动评估。不够则返回错误消息。"""
        remaining = self._remaining_sec()
        if remaining < _MIN_REMAINING_FOR_EVAL_SEC:
            msg = (
                f"Insufficient time for {operation}: "
                f"{remaining:.0f}s remaining < {_MIN_REMAINING_FOR_EVAL_SEC}s minimum. "
                f"Call finish to end the optimization loop."
            )
            log.warning("[deadline] %s", msg)
            return msg
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

        # 墙钟超时：从统一 deadline 计算，留 margin 给 final eval
        wall_timeout = max(0, self._remaining_sec() - _DEADLINE_MARGIN_SEC)

        loop_stats: dict = {}
        agentic_loop(
            messages=messages,
            tools=tools,
            tool_executor=self._dispatch_tool,
            model=self.model,
            max_tool_calls=self.budget.max_steps * 20,
            max_tokens=16384,
            early_stop_fn=self._should_stop,
            stats=loop_stats,
            wall_timeout_sec=wall_timeout,
            no_tool_call_prompt=(
                "You must continue the optimization loop by calling tools. "
                "After analyze, you should: (1) checkpoint_candidate with the patch ops, "
                "(2) for deterministic fixes: accept_candidate(skip_eval=true, skip_eval_reason=...), "
                "for behavioral changes: eval_candidate then accept_candidate if improved, "
                "or try a different patch. "
                "REMINDER: You MUST call spawn_sublayer before finish."
            ),
            compact_fn=self._compact_messages,
            compact_threshold=60,
        )
        self.budget.llm_calls_used += loop_stats.get("llm_calls", 0)

        # Loop 结束后，若未 spawn 且有权 spawn，自动触发 L2
        if (
            self.layer_context
            and self.layer_context.can_spawn_sublayer()
            and self.budget.spawn_calls_used == 0
        ):
            log.info(
                "[unified_agent] agentic_loop 结束但未 spawn，自动触发 spawn_sublayer"
            )
            result = self._tool_spawn_sublayer({})
            if result.get("ok"):
                self.budget.spawn_calls_used += 1
                self.layer_context.spawn_calls_used += 1
                if result.get("child_accepted", 0) > 0 or result.get("noa_modified"):
                    self._spawn_noa_modified = True

        return self._build_result()

    def _should_stop(self) -> bool:
        if self._spawn_noa_modified:
            return True
        if self._in_escape_mode:
            return self._escape_resolved
        # deadline 检查: 剩余时间不够做任何有意义的操作时强制停止
        if self._remaining_sec() < _MIN_REMAINING_FOR_EVAL_SEC:
            log.warning(
                "[should_stop] deadline approaching: %.0fs remaining, forcing stop",
                self._remaining_sec(),
            )
            return True
        if not self.budget.reached_limit():
            return False
        # 预算耗尽 + 可spawn → 进入逃生模式（持续放行直到 spawn/finish）
        if self.layer_context and self.layer_context.can_spawn_sublayer():
            self._in_escape_mode = True
            return False
        return True

    def _auto_accept_unadded_candidates(self):
        """自动将已 eval 且超过 train baseline 但未在 top-K 池中的候选加入池。
        修复: 最后一步 eval 的候选因 agentic loop 结束而无法被 LLM 调用 accept_candidate。
        """
        top_labels = {c["label"] for c in self._top_candidates}
        # L1: 用当轮 train baseline；L2+: 用 test_set baseline
        train_bl = (
            self._current_train_baseline
            if self.layer_context.level <= 1 and self._current_train_baseline > 0
            else self._baseline_score
        )
        for label, score in self._candidate_scores.items():
            if label not in top_labels and score > train_bl:
                log.info(
                    "[auto_accept] %s score=%.2f > train_baseline=%.2f, adding to top-K",
                    label,
                    score,
                    train_bl,
                )
                self._update_top_candidates(label, score)

    def _build_result(self) -> dict:
        # 自动补救: 将已 eval 但未 accept 的优质候选加入 top-K 池
        self._auto_accept_unadded_candidates()
        # Final eval: 用 test_set 对 top-K 候选做完整评估并 commit 最优
        final_eval = self._run_final_eval_once(trigger="finish")
        # final_score 只取 final eval 的 test_set 结果（如果有 commit），否则回退到 baseline
        if final_eval and final_eval.get("test_score") is not None:
            committed_score = final_eval["test_score"]
        else:
            committed_score = self._baseline_score

        # L2 spawn 的 mini-L1 已含 test eval，直接复用 child_score
        if self._spawn_noa_modified:
            for h in reversed(self._history):
                if h.get("action") == "spawn_sublayer":
                    committed_score = h.get("child_score", committed_score)
                    log.info("[build_result] 复用 L2 child_score=%.2f", committed_score)
                    break

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
                        "overwrite",
                    ],
                    "description": "update=SEARCH/REPLACE, overwrite=replace entire file with content, create=new file, delete=remove file",
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
                f"{p}__fix_patch",
                "Fix a broken candidate: replace its patch ops with corrected ones, re-validate (including syntax check), and re-checkpoint. Use after eval_candidate returns score=0 with subprocess errors.",
                {
                    "label": {
                        "type": "string",
                        "description": "Candidate label to fix",
                    },
                    "ops": {
                        "type": "array",
                        "items": _PATCH_OP_SCHEMA,
                        "description": "Corrected patch ops",
                    },
                    "rationale": {"type": "string"},
                },
                required=["label", "ops"],
            )
        )
        tools.append(
            _tool(
                f"{p}__accept_candidate",
                (
                    "Add a candidate to the top-K pool. Best candidate is committed via final test eval. "
                    "For DETERMINISTIC fixes (obvious bugs like wrong regex, incorrect index, clear logic errors), "
                    "set skip_eval=true to skip expensive evaluation. "
                    "For BEHAVIORAL changes (prompt rewording, parameter tuning, heuristic changes), "
                    "you MUST call eval_candidate first."
                ),
                {
                    "label": {"type": "string"},
                    "skip_eval": {
                        "type": "boolean",
                        "description": "Skip eval for deterministic bug fixes. Default false.",
                        "default": False,
                    },
                    "skip_eval_reason": {
                        "type": "string",
                        "description": "Required when skip_eval=true. Explain why this patch is a deterministic fix that doesn't need evaluation.",
                    },
                },
                required=["label"],
            )
        )

        # E2. 候选合并
        tools.append(
            _tool(
                f"{p}__merge_candidates",
                "Merge multiple candidates into one combined candidate. "
                "Use this to STACK patches that fix different issues — instead of picking only the best single patch, "
                "combine several beneficial patches for a larger improvement. "
                "The merged candidate can then be evaluated and accepted normally.",
                {
                    "label": {
                        "type": "string",
                        "description": "Label for the merged candidate",
                    },
                    "source_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels of candidates to merge (must be existing checkpoint'd candidates)",
                    },
                },
                required=["label", "source_labels"],
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

    def _commit_best_before_observe(self):
        """新一轮 observe 前，commit 当前 top-K 最优候选到 source_dir 作为新 base。"""
        if self._episode_counter == 0 or not self._top_candidates:
            return
        best_cand = max(self._top_candidates, key=lambda c: c["score"])
        if best_cand["score"] <= self._baseline_score:
            return
        best_label = best_cand["label"]
        self.sandbox.accept_candidate(best_label)
        self.sys_desc.source_files = collect_sources(self.source_dir)
        self._refresh_file_map()
        self.target = self.target_factory(self.source_dir)
        self._baseline_score = best_cand["score"]
        self._top_candidates.clear()
        self._candidate_scores.clear()
        print(
            f"\n[Observe] Committed best candidate '{best_label}' "
            f"(score={best_cand['score']:.2f}) as new base for next round"
        )

    def _tool_run_observe(self, args: dict) -> dict:
        import random as _random

        # 新一轮前先 commit 上一轮最优，让所有后续操作（包括 L2）基于最新 base
        self._commit_best_before_observe()

        # L2+: 直接从 parent_history 合成 Trajectory，不跑 mini-L1
        if (
            self.layer_context
            and self.layer_context.level >= 2
            and self.layer_context.parent_history
        ):
            return self._observe_from_parent_history()

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
        # 计算当轮 train 样本上的 baseline 分数（未修改的原始 pipeline）
        try:
            base_result = self.eval_fn(self.target, self._current_train_samples)
            base_score = base_result.get("score", 0)
            self._current_train_baseline = (
                base_score * 100 if base_score <= 1 else base_score
            )
            print(
                f"\n[Observe] train_baseline={self._current_train_baseline:.2f} "
                f"(on {len(self._current_train_samples)} train samples)"
            )
        except Exception as e:
            log.warning(f"[Observe] failed to compute train baseline: {e}")
            self._current_train_baseline = 0.0
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
            self._push_history(
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
        self._push_history(
            {
                "action": "observe",
                "step": self._step_count,
                "episode_id": ep_id,
                "n_trajectories": 0,
                "mean_f1": 0,
            }
        )
        return {"episode_id": ep_id, "count": 0, "mean_score": 0}

    def _observe_from_parent_history(self) -> dict:
        """L2+: 返回摘要统计 + 指引 LLM 用 read_source_file 读完整历史。
        第二轮+: 返回上一轮 eval/reject 摘要 + mini_l1 文件指针。"""
        # 如果有上一轮的 eval/reject 记录，用它们代替静态 parent_history
        eval_records = [
            h
            for h in self._history
            if h.get("action") in ("eval_candidate", "reject_candidate")
        ]
        if eval_records:
            self._episode_counter += 1
            ep_id = f"ep_{self._episode_counter}"
            self._push_history(
                {
                    "action": "observe",
                    "step": self._step_count,
                    "episode_id": ep_id,
                    "source": "own_eval_history",
                    "n_eval_records": len(eval_records),
                }
            )
            # 收集 mini_l1 文件指针
            mini_l1_files = [
                f".noa_meta/mini_l1_{h['label']}.json"
                for h in eval_records
                if h.get("label")
                and f".noa_meta/mini_l1_{h['label']}.json" in self._file_map
            ]
            slim_records = [
                {
                    k: h[k]
                    for k in (
                        "action",
                        "step",
                        "label",
                        "after_score",
                        "before_score",
                        "accepted",
                        "error",
                    )
                    if k in h
                }
                for h in eval_records
            ]
            result = {
                "episode_id": ep_id,
                "source": "own_eval_history",
                "eval_summary": slim_records,
            }
            if mini_l1_files:
                result["mini_l1_detail_files"] = mini_l1_files
                result["hint"] = "read_source_file to inspect mini-L1 details"
            return result

        # 第一轮：沿用现有逻辑读 parent_history
        history = self.layer_context.parent_history

        n_evals = sum(1 for h in history if h.get("action") == "eval_candidate")
        n_accepted = sum(
            1
            for h in history
            if h.get("action") == "eval_candidate" and h.get("accepted")
        )
        n_analyzes = sum(1 for h in history if h.get("action") == "analyze")

        context_file = None
        for path in self.layer_context.parent_context_files:
            if "l1_parent_context" in path:
                context_file = path
                break

        self._trajectories = []
        self._episode_counter += 1
        ep_id = f"ep_{self._episode_counter}"

        print(
            f"\n[Observe-L2] parent_history: {len(history)} records "
            f"(evals={n_evals}, accepted={n_accepted}, analyzes={n_analyzes})"
        )
        self._push_history(
            {
                "action": "observe",
                "step": self._step_count,
                "episode_id": ep_id,
                "n_records": len(history),
                "n_evals": n_evals,
                "n_accepted": n_accepted,
                "source": "parent_history",
            }
        )
        result = {
            "episode_id": ep_id,
            "source": "parent_history",
            "n_records": len(history),
            "n_evals": n_evals,
            "n_accepted": n_accepted,
            "n_analyzes": n_analyzes,
        }
        if context_file:
            result["full_history_file"] = context_file
            result["hint"] = (
                f"Use read_source_file(path='{context_file}') to inspect "
                f"the complete untruncated L1 optimization history."
            )
        return result

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
            self._push_history(
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
        for i, pat in enumerate(diagnosis.failure_patterns[:5]):
            print(f"  Pattern {i+1}: {pat}")

        # 精简 patterns 用于 history 和返回值
        slim_patterns = [
            {
                "pattern": p.get("pattern", "") if isinstance(p, dict) else str(p),
                "root_cause": (p.get("root_cause", "") if isinstance(p, dict) else ""),
                "affected_file": (
                    p.get("affected_file", "") if isinstance(p, dict) else ""
                ),
                "severity": (p.get("severity", "") if isinstance(p, dict) else ""),
            }
            for p in diagnosis.failure_patterns[:5]
        ]
        self._push_history(
            {
                "action": "analyze",
                "step": self._step_count,
                "patterns": slim_patterns,
            }
        )
        # 完整 patterns 写入 detail 文件
        ep_id = self._episode_counter
        detail_file = self._write_detail_file(
            f"diagnosis_ep_{ep_id}.json",
            diagnosis.failure_patterns[:top_n],
        )
        return {
            "summary": diagnosis.summary[:1000],
            "top_patterns": slim_patterns,
            "total_pattern_count": len(diagnosis.failure_patterns),
            "detail_file": detail_file,
            "hint": f"read_source_file(path='{detail_file}') for all patterns",
        }

    def _tool_compare_episodes(self, args: dict) -> dict:
        return self.traj_store.compare_episodes(args["ep_a"], args["ep_b"])

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
        import ast as _ast

        ops = _parse_ops(args.get("ops", []))
        errors = validate_patch_ops(self.sys_desc.source_files, ops)
        if errors:
            return {"valid": False, "errors": [_err_dict(e) for e in errors]}
        # 模拟 apply 后对 .py 文件做语法检查
        patched, apply_errors, _ = apply_patch_ops(self.sys_desc.source_files, ops)
        if apply_errors:
            return {"valid": False, "errors": [_err_dict(e) for e in apply_errors]}
        syntax_errors = []
        for sf in patched:
            if sf.path.endswith(".py"):
                try:
                    _ast.parse(sf.content, filename=sf.path)
                except SyntaxError as e:
                    syntax_errors.append(
                        {
                            "file": sf.path,
                            "line": e.lineno,
                            "message": str(e),
                        }
                    )
        if syntax_errors:
            return {"valid": False, "syntax_errors": syntax_errors}
        return {"valid": True, "op_count": len(ops)}

    def _tool_fix_patch(self, args: dict) -> dict:
        """修复已 checkpoint 的 candidate：用新 ops 替换原有 patch 并重新验证。"""
        import ast as _ast

        label = args.get("label", "")
        new_ops = _parse_ops(args.get("ops", []))
        if not label or not new_ops:
            return {"ok": False, "error": "label and ops are required"}
        # 验证 ops 能匹配原始文件
        errors = validate_patch_ops(self.sys_desc.source_files, new_ops)
        if errors:
            return {"ok": False, "errors": [_err_dict(e) for e in errors]}
        # 模拟 apply 后语法检查
        patched, apply_errors, _ = apply_patch_ops(self.sys_desc.source_files, new_ops)
        if apply_errors:
            return {"ok": False, "errors": [_err_dict(e) for e in apply_errors]}
        syntax_errors = []
        for sf in patched:
            if sf.path.endswith(".py"):
                try:
                    _ast.parse(sf.content, filename=sf.path)
                except SyntaxError as e:
                    syntax_errors.append(
                        {
                            "file": sf.path,
                            "line": e.lineno,
                            "message": str(e),
                        }
                    )
        if syntax_errors:
            return {"ok": False, "syntax_errors": syntax_errors}
        # 重新 checkpoint（覆盖旧的）
        patch = StructuredPatch(
            ops=new_ops, rationale=args.get("rationale", f"fix for {label}")
        )
        candidate_dir, cp_errors = self.sandbox.checkpoint_candidate(label, patch)
        if cp_errors:
            return {"ok": False, "errors": [_err_dict(e) for e in cp_errors]}
        # 更新 meta
        self._candidate_meta[label] = {
            "rationale": args.get("rationale", f"fix for {label}"),
            "ops": [{"op": o.op, "file_path": o.file_path} for o in new_ops],
        }
        return {"ok": True, "label": label, "op_count": len(new_ops)}

    def _tool_snapshot(self, args: dict) -> dict:
        path = self.sandbox.snapshot(args["label"])
        return {"ok": True, "label": args["label"], "path": path}

    def _tool_restore(self, args: dict) -> dict:
        self.sandbox.restore(args["label"])
        self.sys_desc.source_files = collect_sources(self.source_dir)
        self._refresh_file_map()
        self.target = self.target_factory(self.source_dir)
        return {"ok": True, "label": args["label"]}

    def _tool_merge_candidates(self, args: dict) -> dict:
        """合并多个候选 patch 为一个组合候选。"""
        label = args["label"]
        source_labels = args.get("source_labels", [])
        if len(source_labels) < 2:
            return {"ok": False, "error": "Need at least 2 source candidates to merge."}

        print(f"\n[MergeCandidates] label={label} sources={source_labels}")
        candidate_dir, conflicts = self.sandbox.merge_candidates(label, source_labels)
        if not candidate_dir:
            print(f"  [MergeCandidates] FAILED: {conflicts}")
            return {"ok": False, "errors": conflicts}

        # 收集源候选的元数据
        merged_ops = []
        merged_rationales = []
        for src in source_labels:
            meta = self._candidate_meta.get(src, {})
            merged_ops.extend(meta.get("ops", []))
            if meta.get("rationale"):
                merged_rationales.append(f"[{src}] {meta['rationale']}")

        self._candidate_meta[label] = {
            "label": label,
            "rationale": f"Merged from {', '.join(source_labels)}: "
            + "; ".join(merged_rationales)[:500],
            "ops": merged_ops,
            "step": self._step_count,
            "merged_from": source_labels,
        }
        self._push_history(
            {
                "action": "merge_candidates",
                "step": self._step_count,
                "label": label,
                "source_labels": source_labels,
                "ok": True,
                "conflicts": conflicts,
            }
        )

        # 附上源候选的已知分数供 LLM 参考
        source_scores = {}
        for src in source_labels:
            if src in self._candidate_scores:
                source_scores[src] = round(self._candidate_scores[src], 2)

        result: dict = {
            "ok": True,
            "label": label,
            "candidate_dir": candidate_dir,
            "source_scores": source_scores,
            "next_action": (
                f"Call eval_candidate(label='{label}') to score the merged candidate. "
                f"If it scores lower than individual candidates, accept the best single one instead."
            ),
        }
        if conflicts:
            result["conflicts"] = conflicts
            result["note"] = (
                "Some file regions had overlapping changes. "
                "The first candidate's version was used for conflicts. "
                "Eval the merge to verify correctness."
            )
            print(
                f"  [MergeCandidates] OK with {len(conflicts)} conflicts: {conflicts}"
            )
        else:
            print("  [MergeCandidates] OK, clean merge")
        return result

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
        full_ops = [
            {
                "op": op.op,
                "file_path": op.file_path,
                "search": op.search,
                "replace": op.replace,
                "content": op.content,
                "anchor": op.anchor,
                "new_lines": op.new_lines,
                "occurrence": op.occurrence,
                "must_be_unique": op.must_be_unique,
                "context_before": op.context_before,
                "context_after": op.context_after,
            }
            for op in ops
        ]
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
        detail_file = self._write_detail_file(
            f"checkpoint_{args['label']}.json",
            {
                "label": args["label"],
                "rationale": args.get("rationale", ""),
                "ops": full_ops,
            },
        )
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
            self._push_history(
                {
                    "action": "checkpoint_candidate",
                    "step": self._step_count,
                    "label": args["label"],
                    "ok": False,
                    "rationale": args.get("rationale", "")[:300],
                    "ops": ops_summary,
                    "detail_file": detail_file,
                    "errors": enriched,
                }
            )
            return {"ok": False, "errors": enriched}
        print(f"  [Checkpoint] OK dir={candidate_dir}")
        self._candidate_meta[args["label"]] = {
            "label": args["label"],
            "rationale": args.get("rationale", "")[:500],
            "ops": ops_summary,
            "detail_file": detail_file,
            "step": self._step_count,
        }
        self._push_history(
            {
                "action": "checkpoint_candidate",
                "step": self._step_count,
                "label": args["label"],
                "ok": True,
                "rationale": args.get("rationale", "")[:300],
                "ops": ops_summary,
                "detail_file": detail_file,
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
        # 时间检查: 拒绝启动必定超时的评估
        time_err = self._check_time_for_eval("eval_candidate")
        if time_err:
            return {"error": time_err, "error_type": "insufficient_time"}
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
        timeout_sec = self._nested_eval_timeout()
        self.budget.evals_used += 1
        result = self.sandbox.eval_in_sandbox(
            candidate_dir,
            self.target_factory,
            self.eval_fn,
            val_data,
            eval_n,
            seed=eval_seed,
            timeout_sec=timeout_sec,
        )
        score = result.get("score", result.get("mean_score", 0))
        # 归一化到 0~100，与 run_eval 保持一致
        normalized_score = score * 100 if score <= 1 else score
        self._candidate_scores[label] = normalized_score
        # 注意: 不在 eval 阶段加入 top_candidates，只在 accept_candidate 时加入
        # L1: 用当轮 train 样本的 baseline 比较；L2+: 用 test_set baseline（L2 不走 observe 抽样）
        train_baseline = (
            self._current_train_baseline
            if self.layer_context.level <= 1 and self._current_train_baseline > 0
            else self._baseline_score
        )
        candidate_meta = self._candidate_meta.get(label, {})
        accepted = normalized_score > train_baseline
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
                f"\n[EvalCandidate] label={label} score={normalized_score:.2f} (train_baseline={train_baseline:.2f}, best={self._best_score:.2f}, current={self._current_score:.2f})"
            )
        # 合并 error 信息
        eval_error = result.get("error")
        if not eval_error and subprocess_errors:
            eval_error = f"subprocess_crash: {subprocess_errors[0][:1000]}"
        self._push_history(
            {
                "action": "eval_candidate",
                "step": self._step_count,
                "label": label,
                "mode": mode,
                "rationale": candidate_meta.get("rationale", ""),
                "ops": candidate_meta.get("ops", []),
                "before_score": round(train_baseline, 2),
                "after_score": round(normalized_score, 2),
                "accepted": accepted,
                "details_count": len(result.get("details", [])),
                "error": eval_error,
            }
        )
        # 完整 details + subprocess_errors 写入 detail 文件
        detail_payload = {}
        details = result.get("details", [])
        if details:
            detail_payload["details"] = details
        if subprocess_errors:
            detail_payload["subprocess_errors"] = subprocess_errors
        detail_file = None
        if detail_payload:
            detail_file = self._write_detail_file(f"eval_{label}.json", detail_payload)

        # L2: 提取 mini-L1 信息写独立文件
        mini_l1_file = None
        if self.layer_context and self.layer_context.level >= 2 and details:
            mini_l1_data = []
            for d in details:
                if d.get("mini_l1_history"):
                    mini_l1_data.append(
                        {
                            "question": d.get("question"),
                            "raw_score": d.get("raw_score"),
                            "accepted": d.get("accepted"),
                            "steps": d.get("steps"),
                            "score_trajectory": d.get("score_trajectory", []),
                            "history": d["mini_l1_history"],
                        }
                    )
            if mini_l1_data:
                mini_l1_file = self._write_detail_file(
                    f"mini_l1_{label}.json", mini_l1_data
                )

        # 精简返回值
        if normalized_score > train_baseline:
            next_action = (
                f"Score {normalized_score:.2f} > train_baseline {train_baseline:.2f}. "
                f"Call accept_candidate(label='{label}') to add to top-K pool."
            )
        else:
            next_action = (
                f"Score {normalized_score:.2f} <= train_baseline {train_baseline:.2f}. "
                f"Analyze why and try a different patch."
            )
        slim_result = {
            "candidate_label": label,
            "candidate_score": normalized_score,
            "baseline_score": train_baseline,
            "next_action": next_action,
            "error": eval_error,
        }
        if subprocess_errors:
            slim_result["subprocess_error_count"] = len(subprocess_errors)
            slim_result["subprocess_error_preview"] = subprocess_errors[0][:200]
        if detail_file:
            slim_result["detail_file"] = detail_file
        if mini_l1_file:
            slim_result["mini_l1_detail_file"] = mini_l1_file
        return slim_result

    def _tool_accept_candidate(self, args: dict) -> dict:
        """接受候选 — 加入 top-K 池，不直接 commit（final eval 时再 commit 最优的）。"""
        label = args["label"]
        skip_eval = bool(args.get("skip_eval", False))
        skip_reason = args.get("skip_eval_reason", "")

        # skip_eval 模式: 确定性修复，跳过 eval 直接加入 top-K
        if skip_eval:
            if not skip_reason:
                return {
                    "ok": False,
                    "error": "skip_eval=true requires skip_eval_reason explaining why eval is unnecessary.",
                }
            # 检查 candidate 是否存在（必须先 checkpoint）
            candidate_dir = os.path.join(self.sandbox._candidates_dir, label)
            if not os.path.isdir(candidate_dir):
                return {
                    "ok": False,
                    "error": f"Candidate not found: {label}. Call checkpoint_candidate first.",
                }
            # 用 baseline 作为虚拟分数（final test eval 会重新测量真实分数）
            train_baseline = (
                self._current_train_baseline
                if self.layer_context.level <= 1 and self._current_train_baseline > 0
                else self._baseline_score
            )
            virtual_score = train_baseline + 0.01
            self._candidate_scores[label] = virtual_score
            self._update_top_candidates(label, virtual_score)
            candidate_meta = self._candidate_meta.get(label, {})
            print(
                f"\n[AcceptCandidate] label={label} SKIP_EVAL (deterministic fix) "
                f"reason={skip_reason[:100]}"
            )
            self._push_history(
                {
                    "action": "accept_candidate",
                    "label": label,
                    "step": self._step_count,
                    "rationale": candidate_meta.get("rationale", ""),
                    "ops": candidate_meta.get("ops", []),
                    "score": virtual_score,
                    "skip_eval": True,
                    "skip_eval_reason": skip_reason,
                }
            )
            top_labels = [c["label"] for c in self._top_candidates]
            return {
                "ok": True,
                "label": label,
                "skip_eval": True,
                "top_candidates": top_labels,
            }

        # 正常模式: 需要先 eval_candidate
        cand_score = self._candidate_scores.get(label, 0)
        train_baseline = (
            self._current_train_baseline
            if self.layer_context.level <= 1 and self._current_train_baseline > 0
            else self._baseline_score
        )
        best_train_score = max(
            (c["score"] for c in self._top_candidates), default=train_baseline
        )
        print(
            f"\n[AcceptCandidate] label={label} candidate_score={cand_score:.2f} "
            f"train_baseline={train_baseline:.2f} best_train={best_train_score:.2f}"
        )
        in_top = any(c["label"] == label for c in self._top_candidates)
        if not in_top and cand_score <= train_baseline:
            print(
                f"  [AcceptCandidate] REJECTED: candidate {cand_score:.2f} <= train_baseline {train_baseline:.2f}"
            )
            self.budget.no_improve_count += 1
            candidate_meta = self._candidate_meta.get(label, {})
            self._push_history(
                {
                    "action": "reject_candidate",
                    "step": self._step_count,
                    "label": label,
                    "rationale": candidate_meta.get("rationale", ""),
                    "ops": candidate_meta.get("ops", []),
                    "reason": f"score {cand_score:.2f} <= train_baseline {train_baseline:.2f}",
                }
            )
            return {
                "ok": False,
                "error": f"Candidate score {cand_score:.2f} not better than train_baseline {train_baseline:.2f}. Not added to top-{self.top_k} pool.",
            }
        self._update_top_candidates(label, cand_score)
        self.budget.no_improve_count = 0
        if cand_score > self._best_score:
            self._best_score = cand_score
        self._current_score = cand_score
        candidate_meta = self._candidate_meta.get(label, {})
        self._push_history(
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
        from noa.subprocess_runner import serialize_dataset

        if not self.layer_context or not self.layer_context.can_spawn_sublayer():
            return {"ok": False, "error": "Cannot spawn: depth/budget limit reached"}
        if not self.noa_dir or not self.project_root:
            return {
                "ok": False,
                "error": "Cannot spawn: noa_dir or project_root not set",
            }
        # 时间检查: spawn 需要至少 10 分钟
        remaining = self._remaining_sec()
        if remaining < 600:
            return {
                "ok": False,
                "error": f"Insufficient time for spawn: {remaining:.0f}s remaining < 600s minimum",
            }

        child_level = self.layer_context.level + 1
        noa_dir = self.noa_dir
        project_root = self.project_root
        spawn_baseline_score = self._prepare_spawn_baseline()
        spawn_id = f"{self.layer_context.layer_id.lower()}_step_{self._step_count}"

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
                result = self._run_mini_l1_subprocess(
                    noa_source_dir=noa_source_dir,
                    project_root=project_root,
                    dataset_pickle_path=dpp,
                    train_pool_pickle_path=train_pool_pp,
                    test_set_pickle_path=test_set_pp,
                    question=question,
                    ml1_cfg=ml1,
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
                    # mini-L1 history: 精简版，只保留关键字段
                    raw_history = inter.get("history", [])
                    detail["mini_l1_history"] = [
                        {
                            k: h[k]
                            for k in (
                                "action",
                                "step",
                                "label",
                                "after_score",
                                "before_score",
                                "score",
                                "accepted",
                                "error",
                                "rationale",
                            )
                            if k in h
                        }
                        for h in raw_history
                    ]
                    detail["score_trajectory"] = [
                        round(h.get("after_score", h.get("score", 0)), 2)
                        for h in raw_history
                        if h.get("action")
                        in ("eval_candidate", "accept_candidate", "reject_candidate")
                    ]
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

        # parent history for L2 context — 传完整历史，L2 直接分析无需 re-observe
        parent_history = list(self._history)  # 全量 copy
        parent_current_score = (
            self._spawn_ready_baseline_score
            if self._spawn_ready_baseline_score is not None
            else self._current_score
        )
        parent_summary = (
            f"Initial: {self._baseline_score:.2f}, Current: {parent_current_score:.2f}, "
            f"Delta: {parent_current_score - self._baseline_score:+.2f}, "
            f"Accepted: {self._accepted_patches}, Steps: {self._step_count}"
        )
        parent_context_artifact = self._write_parent_context_artifact(
            parent_summary=parent_summary,
            baseline_score=spawn_baseline_score,
        )

        child_layer_context = LayerContext(
            layer_id=f"L{child_level}",
            level=child_level,
            writable_root=noa_dir,
            readable_roots=[noa_dir, self.source_dir],
            parent_history=parent_history,
            parent_summary=parent_summary,
            parent_context_files=(
                [parent_context_artifact] if parent_context_artifact else []
            ),
            max_depth=self.layer_context.max_depth,
            max_spawn_calls=max(0, self.layer_context.max_spawn_calls - 1),
        )

        l2_cfg = self.spawn_config.get("l2", {})
        # 传播 deadline: 子层预算 = 剩余时间 - margin
        child_wall_budget = max(0, self._remaining_sec() - _DEADLINE_MARGIN_SEC)
        log.info(
            "[spawn] child wall_budget=%.0fs (parent remaining=%.0fs)",
            child_wall_budget,
            self._remaining_sec(),
        )
        try:
            with bind_scope(layer_id=child_layer_context.layer_id, spawn_id=spawn_id):
                child = NOptimizer(
                    source_dir=noa_dir,
                    target_factory=child_target_factory,
                    dataset=child_dataset,
                    eval_fn=child_eval_fn,
                    max_steps=l2_cfg.get("max_steps", 12),
                    n_samples=l2_cfg.get("n_samples", 2),
                    model=self.model,
                    score_fn=child_score_fn,
                    max_llm_calls=l2_cfg.get("max_llm_calls", 999999),
                    max_evals=l2_cfg.get("max_evals", 8),
                    max_no_improve_steps=l2_cfg.get("max_no_improve_steps", 4),
                    layer_context=child_layer_context,
                    observer_search_roots=[noa_dir],
                    initial_baseline_score=spawn_baseline_score,
                    wall_budget_sec=child_wall_budget,
                )
                child_result = child.run()
        except Exception as e:
            log.error(f"[Spawn] L{child_level} crashed: {e}", exc_info=True)
            return {"ok": False, "error": f"L{child_level} crashed: {str(e)[:500]}"}

        child_accepted = int(child_result.get("accepted", 0) or 0)
        child_spawn_restart = bool(child_result.get("spawn_restart"))
        noa_modified = child_accepted > 0 or child_spawn_restart

        self._push_history(
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
        _SLIM_KEYS = (
            "action",
            "step",
            "label",
            "score",
            "after_score",
            "before_score",
            "accepted",
            "error",
            "episode_id",
            "rationale",
        )
        slim = [{k: h[k] for k in _SLIM_KEYS if k in h} for h in self._history[-n:]]
        # 完整历史写 detail 文件
        detail_file = self._write_detail_file("full_history.json", self._history)
        return json.dumps(
            {
                "recent": slim,
                "total_count": len(self._history),
                "detail_file": detail_file,
                "hint": f"read_source_file(path='{detail_file}') for full history",
            },
            indent=1,
            default=str,
        )


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
