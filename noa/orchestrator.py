"""Orchestrator — workspace 管理 + NOptimizer 入口，递归由 spawn_sublayer 驱动。"""

from __future__ import annotations

import os

from noa.core.protocol import LayerContext
from noa.engine import NOptimizer
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from noa.workspace import WorkspaceManager
from utils.llm import DEFAULT_MODEL


class Orchestrator:
    """嵌套优化器入口 — workspace 管理 + L1 NOptimizer 启动。

    递归嵌套由 NOptimizer 的 spawn_sublayer 工具驱动。
    """

    def __init__(
        self,
        source_dir: str,
        target_factory,
        dataset: list,
        eval_fn,
        score_fn,
        *,
        l1_max_steps: int = 20,
        l1_n_samples: int = 20,
        l1_eval_n_samples: int = 20,
        l1_model: str = DEFAULT_MODEL,
        l1_max_llm_calls: int = 80,
        l1_max_evals: int = 12,
        l1_max_no_improve_steps: int = 5,
        max_depth: int = 3,
        max_spawn_calls: int = 2,
        spawn_config: dict | None = None,
        val_set: list | None = None,
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

        self.max_depth = max_depth
        self.max_spawn_calls = max_spawn_calls
        self.spawn_config = spawn_config or {}
        self.val_set = val_set

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(self.project_root, ".noa_cache")
        self.dataset_pickle_path = serialize_dataset(dataset, cache_dir=cache_dir)

    def run(self) -> dict:
        ws = WorkspaceManager(self.project_root, self.source_dir)
        ws_noa, ws_source = ws.setup()
        print(f"[Orchestrator] Workspace ready: {ws.run_dir}")

        rounds = []

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
            layer_context=layer_context,
            noa_dir=ws_noa,
            dataset_pickle_path=self.dataset_pickle_path,
            spawn_config=self.spawn_config,
            val_set=self.val_set,
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
            )
            ws.snapshot(f"after_round_{round_num}")
            rounds.append({"round": round_num, "l1_result": result})
            round_num += 1

        last = rounds[-1]["l1_result"]
        return {
            "final_score": last.get("final_score", 0),
            "baseline_score": rounds[0]["l1_result"].get("baseline_score", 0),
            "rounds": rounds,
            "total_rounds": len(rounds),
            "run_dir": ws.run_dir,
            "snapshots": ws.list_snapshots(),
        }
