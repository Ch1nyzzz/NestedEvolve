"""NOptimizer — unified agentic loop 驱动的优化引擎。"""

from __future__ import annotations

import os
from typing import Callable

from noa.core.protocol import LayerContext, OptimizationBudget, SystemDescription
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
        eval_n_samples: int = 20,
        model: str = DEFAULT_MODEL,
        system_description: str = "",
        score_fn,
        max_llm_calls: int = 120,
        max_evals: int = 20,
        max_no_improve_steps: int = 5,
        layer_context: LayerContext | None = None,
        observer_search_roots: list[str] | None = None,
        noa_dir: str | None = None,
        dataset_pickle_path: str | None = None,
        spawn_config: dict | None = None,
        val_set: list | None = None,
    ):
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.n_samples = n_samples
        self.eval_n_samples = eval_n_samples
        self.model = model
        self.system_description = system_description
        self.score_fn = score_fn

        max_spawn = layer_context.max_spawn_calls if layer_context else 2
        self.budget = OptimizationBudget(
            max_steps=max_steps,
            max_llm_calls=max_llm_calls,
            max_evals=max_evals,
            max_no_improve_steps=max_no_improve_steps,
            target_delta=float("inf"),
            max_spawn_calls=max_spawn,
        )
        self.layer_context = layer_context
        self.observer_search_roots = observer_search_roots
        self.noa_dir = noa_dir
        self.dataset_pickle_path = dataset_pickle_path
        self.spawn_config = spawn_config or {}
        self.val_set = val_set

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

        print("\n[NOA] === Initiate ===")
        self.sys_desc = initiate(
            source_dir=self.source_dir,
            model=self.model,
            system_description=self.system_description,
        )
        print(f"[NOA] Workflow: {self.sys_desc.workflow_summary}")

        layer_context = self.layer_context or LayerContext(
            layer_id="L1",
            level=1,
            writable_root=self.source_dir,
            readable_roots=[self.source_dir],
            parent_history=[],
        )

        layer_label = f"l{layer_context.level}"
        ws_dir = os.path.join(self._project_root, ".noa_runs", f"unified_{layer_label}")
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
            val_set=self.val_set,
        )

        result = agent.run()
        self.history = result.get("history", [])
        return result

    def __call__(self, **kwargs) -> dict:
        return self.run()
