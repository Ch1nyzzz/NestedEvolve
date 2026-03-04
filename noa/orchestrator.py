"""Orchestrator — workspace 管理 + NOptimizer 薄入口，递归由 spawn_sublayer 驱动。"""

from __future__ import annotations

import os
from types import SimpleNamespace

from noa.core.protocol import LayerContext
from noa.engine import NOptimizer
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from noa.workspace import WorkspaceManager
from utils.llm import resolve_model


class SubprocessTarget:
    """通用子进程目标包装 — 服务任意层级的评估。"""

    def __init__(
        self,
        noa_dir: str,
        project_root: str,
        target_source_dir: str,
        dataset_pickle_path: str,
        layer_level: int = 1,
        max_steps: int = 20,
        n_samples: int = 20,
        model: str = resolve_model("gpt-4.1-mini"),
        eval_n_samples: int = 20,
        max_llm_calls: int = 80,
        max_evals: int = 12,
        max_no_improve_steps: int = 5,
        max_tool_calls: int = 10,
        observer_tool_calls: int = 8,
    ):
        self.noa_dir = noa_dir
        self.project_root = project_root
        self.target_source_dir = target_source_dir
        self.dataset_pickle_path = dataset_pickle_path
        self.layer_level = layer_level
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.model = model
        self.eval_n_samples = eval_n_samples
        self.max_llm_calls = max_llm_calls
        self.max_evals = max_evals
        self.max_no_improve_steps = max_no_improve_steps
        self.max_tool_calls = max_tool_calls
        self.observer_tool_calls = observer_tool_calls

    def __call__(self, question: str) -> SimpleNamespace:
        result = run_layer_subprocess(
            noa_dir=self.noa_dir,
            project_root=self.project_root,
            target_source_dir=self.target_source_dir,
            dataset_pickle_path=self.dataset_pickle_path,
            layer_level=self.layer_level,
            max_steps=self.max_steps,
            n_samples=self.n_samples,
            eval_n_samples=self.eval_n_samples,
            model=self.model,
            max_llm_calls=self.max_llm_calls,
            max_evals=self.max_evals,
            max_no_improve_steps=self.max_no_improve_steps,
            max_tool_calls=self.max_tool_calls,
            observer_tool_calls=self.observer_tool_calls,
            isolate_source=True,
        )
        if result.get("error"):
            print(f"[SubprocessTarget] subprocess error: {result['error'][:200]}")
        return SimpleNamespace(
            answer=str(result.get("final_score", 0)), intermediate=result
        )


# Backward compatibility alias
L1Target = SubprocessTarget


class Orchestrator:
    """简化的嵌套优化器入口 — workspace 管理 + L1 NOptimizer 启动。

    递归嵌套由 NOptimizer planner 的 spawn_sublayer action 驱动，
    不再硬编码 L1/L2 调度。
    """

    def __init__(
        self,
        source_dir: str,
        target_factory,
        dataset: list,
        eval_fn,
        score_fn,
        *,
        # L1 参数
        l1_max_steps: int = 20,
        l1_n_samples: int = 20,
        l1_eval_n_samples: int = 20,
        l1_model: str = resolve_model("gpt-4.1-mini"),
        l1_max_llm_calls: int = 80,
        l1_max_evals: int = 12,
        l1_max_no_improve_steps: int = 5,
        l1_max_tool_calls: int = 10,
        l1_observer_tool_calls: int = 8,
        l1_optimizer_tool_calls: int = 5,
        # 递归参数
        max_depth: int = 3,
        max_spawn_calls: int = 2,
        # 兼容旧参数（忽略）
        l2_max_steps: int = 12,
        l2_n_samples: int | None = None,
        l2_model: str = resolve_model("gpt-4.1-mini"),
        l2_max_llm_calls: int = 60,
        l2_max_evals: int = 8,
        l2_max_no_improve_steps: int = 4,
        l2_observer_tool_calls: int = 8,
        max_l2_rounds: int = 2,
        max_l1_rounds: int | None = None,
        orchestrator_model: str | None = None,
        orchestrator_max_llm_calls: int = 24,
        max_orchestrator_steps: int | None = None,
    ):
        self.source_dir = os.path.abspath(source_dir)
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.score_fn = score_fn

        self.l1_max_steps = l1_max_steps
        self.l1_n_samples = l1_n_samples
        self.l1_eval_n_samples = l1_eval_n_samples
        self.l1_model = l1_model
        self.l1_max_llm_calls = l1_max_llm_calls
        self.l1_max_evals = l1_max_evals
        self.l1_max_no_improve_steps = l1_max_no_improve_steps
        self.l1_max_tool_calls = l1_max_tool_calls
        self.l1_observer_tool_calls = l1_observer_tool_calls
        self.l1_optimizer_tool_calls = l1_optimizer_tool_calls

        self.max_depth = max_depth
        self.max_spawn_calls = max_spawn_calls

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(self.project_root, ".noa_cache")
        self.dataset_pickle_path = serialize_dataset(dataset, cache_dir=cache_dir)

    def run(self) -> dict:
        ws = WorkspaceManager(self.project_root, self.source_dir)
        ws_noa, ws_source = ws.setup()
        print(f"[Orchestrator] Workspace ready: {ws.run_dir}")

        rounds = []

        # Round 0: in-process L1
        layer_context = LayerContext(
            layer_id="L1",
            level=1,
            writable_root=ws_source,
            readable_roots=[ws_source],
            parent_history=[],
            max_depth=self.max_depth,
            max_spawn_calls=self.max_spawn_calls,
        )

        optimizer = NOptimizer(
            source_dir=ws_source,
            target_factory=self.target_factory,
            dataset=self.dataset,
            eval_fn=self.eval_fn,
            max_steps=self.l1_max_steps,
            n_samples=self.l1_n_samples,
            eval_n_samples=self.l1_eval_n_samples,
            model=self.l1_model,
            score_fn=self.score_fn,
            max_llm_calls=self.l1_max_llm_calls,
            max_evals=self.l1_max_evals,
            max_no_improve_steps=self.l1_max_no_improve_steps,
            max_tool_calls=self.l1_max_tool_calls,
            observer_max_tool_calls=self.l1_observer_tool_calls,
            optimizer_max_tool_calls=self.l1_optimizer_tool_calls,
            layer_context=layer_context,
            noa_dir=ws_noa,
            dataset_pickle_path=self.dataset_pickle_path,
        )
        result = optimizer.run()
        ws.snapshot("after_round_0")
        rounds.append({"round": 0, "l1_result": result})

        # Subsequent rounds: subprocess L1 restart after spawn_sublayer modified noa/
        round_num = 1
        while result.get("spawn_restart") and round_num <= self.max_spawn_calls:
            print(
                f"\n[Orchestrator] === Round {round_num}: Restarting L1 via subprocess (noa/ modified) ==="
            )
            result = run_layer_subprocess(
                noa_dir=ws_noa,
                project_root=self.project_root,
                target_source_dir=ws_source,
                dataset_pickle_path=self.dataset_pickle_path,
                layer_level=1,
                max_steps=self.l1_max_steps,
                n_samples=self.l1_n_samples,
                eval_n_samples=self.l1_eval_n_samples,
                model=self.l1_model,
                max_llm_calls=self.l1_max_llm_calls,
                max_evals=self.l1_max_evals,
                max_no_improve_steps=self.l1_max_no_improve_steps,
                max_tool_calls=self.l1_max_tool_calls,
                observer_tool_calls=self.l1_observer_tool_calls,
            )
            ws.snapshot(f"after_round_{round_num}")
            rounds.append({"round": round_num, "l1_result": result})
            round_num += 1

        # 取最后一轮的分数
        last = rounds[-1]["l1_result"]
        return {
            "final_score": last.get("final_score", 0),
            "baseline_score": rounds[0]["l1_result"].get("baseline_score", 0),
            "rounds": rounds,
            "total_rounds": len(rounds),
            "schedule_trace": [],
            "orchestrator_usage": {},
            "run_dir": ws.run_dir,
            "snapshots": ws.list_snapshots(),
        }
