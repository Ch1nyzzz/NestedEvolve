"""Orchestrator — workspace 管理 + NOptimizer 入口，递归由 spawn_sublayer 驱动。"""

from __future__ import annotations

import os

from noa.core.protocol import LayerContext
from noa.engine import NOptimizer
from noa.runtime import update_current_run
from noa.subprocess_runner import serialize_dataset
from noa.workspace import WorkspaceManager
from utils.llm import (
    DEFAULT_MODEL,
    infer_provider_from_model,
    resolve_model,
    rpm_limit_for_provider,
)


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
        l1_model: str = DEFAULT_MODEL,
        l1_max_llm_calls: int = 999999,
        l1_max_no_improve_steps: int = 5,
        max_depth: int = 3,
        max_spawn_calls: int = 2,
        spawn_config: dict | None = None,
        train_pool: list | None = None,
        val_set: list | None = None,
        test_set: list | None = None,
        train_sample_size: int = 25,
        top_k: int = 3,
        test_eval_fn=None,
    ):
        self.source_dir = os.path.abspath(source_dir)
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.test_eval_fn = test_eval_fn
        self.score_fn = score_fn

        self.l1_max_steps = l1_max_steps
        self.l1_n_samples = l1_n_samples

        self.l1_model = l1_model
        self.l1_max_llm_calls = l1_max_llm_calls
        self.l1_max_no_improve_steps = l1_max_no_improve_steps

        self.max_depth = max_depth
        self.max_spawn_calls = max_spawn_calls
        self.spawn_config = spawn_config or {}
        self.train_pool = train_pool
        self.val_set = val_set
        self.test_set = test_set
        self.train_sample_size = train_sample_size
        self.top_k = top_k

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(self.project_root, ".noa_cache")
        self.dataset_pickle_path = serialize_dataset(dataset, cache_dir=cache_dir)
        # 序列化 train_pool / val_set / test_set 供 subprocess 使用
        self.train_pool_pickle_path = (
            serialize_dataset(train_pool, cache_dir=cache_dir) if train_pool else None
        )
        self.val_set_pickle_path = (
            serialize_dataset(val_set, cache_dir=cache_dir) if val_set else None
        )
        self.test_set_pickle_path = (
            serialize_dataset(test_set, cache_dir=cache_dir) if test_set else None
        )

    def run(self) -> dict:
        ws = WorkspaceManager(self.project_root, self.source_dir)
        ws_noa, ws_source = ws.setup()
        print(f"[Orchestrator] Workspace ready: {ws.run_dir}")
        resolved_model = resolve_model(self.l1_model)
        provider = infer_provider_from_model(resolved_model)
        update_current_run(
            run_id=ws.run_id,
            run_dir=ws.run_dir,
            pid=os.getpid(),
            source_dir=ws_source,
            target_name=os.path.basename(self.source_dir.rstrip("/")),
            model=resolved_model,
            llm_provider=provider,
            llm_rpm_limit=rpm_limit_for_provider(provider),
            status="running",
            active_layer="L1",
            active_spawn_id=None,
        )

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
            model=self.l1_model,
            score_fn=self.score_fn,
            max_llm_calls=self.l1_max_llm_calls,
            max_no_improve_steps=self.l1_max_no_improve_steps,
            layer_context=layer_context,
            observer_search_roots=[ws_noa],
            noa_dir=ws_noa,
            dataset_pickle_path=self.dataset_pickle_path,
            spawn_config=self.spawn_config,
            train_pool=self.train_pool,
            val_set=self.val_set,
            test_set=self.test_set,
            train_sample_size=self.train_sample_size,
            top_k=self.top_k,
            test_eval_fn=self.test_eval_fn,
        )
        result = optimizer.run()
        ws.snapshot("after_round_0")
        rounds.append({"round": 0, "l1_result": result})

        # L2 spawn 的 mini-L1 已包含 test eval，不再重跑 L1。
        # final_score 由 _build_result 自动取 max(L1 final, L2 child_score)。

        best = max(rounds, key=lambda r: r["l1_result"].get("final_score", 0))
        payload = {
            "final_score": best["l1_result"].get("final_score", 0),
            "baseline_score": rounds[0]["l1_result"].get("baseline_score", 0),
            "rounds": rounds,
            "total_rounds": len(rounds),
            "run_dir": ws.run_dir,
            "snapshots": ws.list_snapshots(),
        }
        update_current_run(
            status="completed",
            active_layer=None,
            active_spawn_id=None,
        )
        return payload
