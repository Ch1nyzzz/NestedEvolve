"""NOptimizer — unified agentic loop 驱动的优化引擎。"""

from __future__ import annotations

import os
from typing import Callable

from noa.core.protocol import LayerContext, OptimizationBudget, SystemDescription
from noa.runtime import bind_scope, configure_layer_logging, update_current_run
from noa.stages.initiator import initiate
from utils.llm import DEFAULT_MODEL


class NOptimizer:
    """Unified agentic loop 驱动的嵌套优化器。"""

    def __init__(
        self,
        source_dir: str,
        target_factory: Callable[[str], object],
        dataset: list,
        eval_fn,
        *,
        max_steps: int = 20,
        n_samples: int = 20,
        model: str = DEFAULT_MODEL,
        system_description: str = "",
        score_fn,
        max_llm_calls: int = 120,
        max_no_improve_steps: int = 5,
        layer_context: LayerContext | None = None,
        observer_search_roots: list[str] | None = None,
        noa_dir: str | None = None,
        dataset_pickle_path: str | None = None,
        spawn_config: dict | None = None,
        train_pool: list | None = None,
        val_set: list | None = None,
        test_set: list | None = None,
        train_sample_size: int = 25,
        top_k: int = 3,
        initial_baseline_score: float | None = None,
        wall_budget_sec: float | None = None,
        test_eval_fn=None,
    ):
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.test_eval_fn = test_eval_fn
        self.n_samples = n_samples

        self.model = model
        self.system_description = system_description
        self.score_fn = score_fn
        self.train_pool = train_pool
        self.val_set = val_set
        self.test_set = test_set
        self.train_sample_size = train_sample_size
        self.top_k = top_k
        self.initial_baseline_score = initial_baseline_score
        self.wall_budget_sec = wall_budget_sec

        max_spawn = layer_context.max_spawn_calls if layer_context else 2
        self.budget = OptimizationBudget(
            max_steps=max_steps,
            max_llm_calls=max_llm_calls,
            max_no_improve_steps=max_no_improve_steps,
            target_delta=float("inf"),
            max_spawn_calls=max_spawn,
        )
        self.layer_context = layer_context
        self.observer_search_roots = observer_search_roots
        self.noa_dir = noa_dir
        self.dataset_pickle_path = dataset_pickle_path
        self.spawn_config = spawn_config or {}

        self.target = target_factory(source_dir)
        self.sys_desc: SystemDescription | None = None
        self.history: list[dict] = []
        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(self) -> dict:
        """运行 unified optimizer agent，返回优化结果。"""
        from noa.sandbox_manager import SandboxManager
        from noa.trajectory_store import TrajectoryStore
        from noa.tools.probe import ComponentProbe
        from noa.unified_agent import UnifiedOptimizerAgent

        layer_context = self.layer_context or LayerContext(
            layer_id="L1",
            level=1,
            writable_root=self.source_dir,
            readable_roots=[self.source_dir],
            parent_history=[],
        )
        with bind_scope(layer_id=layer_context.layer_id):
            configure_layer_logging(layer_context.layer_id)
            update_current_run(status="running", active_layer=layer_context.layer_id)

            print("\n[NOA] === Initiate ===")
            self.sys_desc = initiate(
                source_dir=self.source_dir,
                model=self.model,
                system_description=self.system_description,
            )
            print(f"[NOA] Workflow: {self.sys_desc.workflow_summary}")

            layer_label = f"l{layer_context.level}"
            target_name = os.path.basename(self.source_dir.rstrip("/"))
            ws_dir = os.path.join(
                self._project_root, ".noa_runs", f"{target_name}_unified_{layer_label}"
            )
            sandbox = SandboxManager(self.source_dir, ws_dir, layer_context)
            sandbox.save_accepted_snapshot()
            traj_store = TrajectoryStore(os.path.join(ws_dir, "trajectories"))

            probe = None
            try:
                p = ComponentProbe(self.target, self.sys_desc)
                if p.components:
                    probe = p
            except Exception:
                pass

            budget = OptimizationBudget(
                **{
                    k: getattr(self.budget, k)
                    for k in OptimizationBudget.__dataclass_fields__
                }
            )

            # wall_budget_sec: 显式传入则用，否则用默认 4h
            from noa.unified_agent import DEFAULT_WALL_BUDGET_SEC

            _wbs = (
                self.wall_budget_sec
                if self.wall_budget_sec is not None
                else DEFAULT_WALL_BUDGET_SEC
            )

            agent = UnifiedOptimizerAgent(
                sys_desc=self.sys_desc,
                source_dir=self.source_dir,
                target_factory=self.target_factory,
                target=self.target,
                dataset=self.dataset,
                eval_fn=self.eval_fn,
                score_fn=self.score_fn,
                model=self.model,
                layer_context=layer_context,
                budget=budget,
                sandbox_manager=sandbox,
                trajectory_store=traj_store,
                component_probe=probe,
                observer_search_roots=self.observer_search_roots,
                noa_dir=self.noa_dir,
                project_root=self._project_root,
                dataset_pickle_path=self.dataset_pickle_path,
                spawn_config=self.spawn_config,
                train_pool=self.train_pool,
                val_set=self.val_set,
                test_set=self.test_set,
                train_sample_size=self.train_sample_size,
                n_samples=self.n_samples,
                top_k=self.top_k,
                initial_baseline_score=self.initial_baseline_score,
                wall_budget_sec=_wbs,
                test_eval_fn=self.test_eval_fn,
            )

            result = agent.run()
            self.history = result.get("history", [])
            update_current_run(active_layer=layer_context.layer_id, status="running")
            return result

    def __call__(self, **kwargs) -> dict:
        return self.run()
