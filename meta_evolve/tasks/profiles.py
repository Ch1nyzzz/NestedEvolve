"""TaskProfile：预定义维度的任务画像。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TaskProfile:
    """8 维任务画像，用于 skill 路由。"""

    name: str
    eval_cost: Literal["fast", "medium", "slow"] = "medium"
    signal_density: Literal["dense", "sparse", "binary"] = "dense"
    search_space: Literal["continuous", "combinatorial", "code_structural"] = (
        "continuous"
    )
    decomposable: Literal["modular", "monolithic"] = "monolithic"
    multimodal: Literal["unimodal", "few_modes", "highly_multimodal"] = "few_modes"
    constraint_type: Literal["none", "hard", "soft"] = "none"
    objective_type: Literal["maximize", "minimize", "satisfy"] = "maximize"
    problem_scale: Literal["small", "medium", "large"] = "medium"

    def tags(self) -> list[str]:
        """返回所有维度值作为 tag 列表（用于 skill trigger 匹配）。"""
        return [
            self.eval_cost,
            self.signal_density,
            self.search_space,
            self.decomposable,
            self.multimodal,
            self.constraint_type,
            self.objective_type,
            self.problem_scale,
        ]


# ============================================================
# 预定义 11 个 skydiscover 任务的画像
# ============================================================

TASK_PROFILES: dict[str, TaskProfile] = {
    "circle_packing": TaskProfile(
        name="circle_packing",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="highly_multimodal",
        constraint_type="hard",
        objective_type="maximize",
        problem_scale="medium",
    ),
    "circle_packing_rect": TaskProfile(
        name="circle_packing_rect",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="highly_multimodal",
        constraint_type="hard",
        objective_type="maximize",
        problem_scale="medium",
    ),
    "erdos_min_overlap": TaskProfile(
        name="erdos_min_overlap",
        eval_cost="medium",
        signal_density="sparse",
        search_space="combinatorial",
        decomposable="monolithic",
        multimodal="few_modes",
        constraint_type="hard",
        objective_type="minimize",
        problem_scale="large",
    ),
    "first_autocorr_ineq": TaskProfile(
        name="first_autocorr_ineq",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="few_modes",
        constraint_type="soft",
        objective_type="maximize",
        problem_scale="small",
    ),
    "heilbronn_triangle": TaskProfile(
        name="heilbronn_triangle",
        eval_cost="medium",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="highly_multimodal",
        constraint_type="hard",
        objective_type="maximize",
        problem_scale="medium",
    ),
    "matmul": TaskProfile(
        name="matmul",
        eval_cost="fast",
        signal_density="dense",
        search_space="code_structural",
        decomposable="modular",
        multimodal="few_modes",
        constraint_type="none",
        objective_type="maximize",
        problem_scale="large",
    ),
    "second_autocorr_ineq": TaskProfile(
        name="second_autocorr_ineq",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="few_modes",
        constraint_type="soft",
        objective_type="maximize",
        problem_scale="small",
    ),
    "signal_processing": TaskProfile(
        name="signal_processing",
        eval_cost="medium",
        signal_density="dense",
        search_space="code_structural",
        decomposable="modular",
        multimodal="few_modes",
        constraint_type="soft",
        objective_type="maximize",
        problem_scale="medium",
    ),
    "sums_diffs_finite_sets": TaskProfile(
        name="sums_diffs_finite_sets",
        eval_cost="fast",
        signal_density="sparse",
        search_space="combinatorial",
        decomposable="monolithic",
        multimodal="unimodal",
        constraint_type="hard",
        objective_type="maximize",
        problem_scale="medium",
    ),
    "third_autocorr_ineq": TaskProfile(
        name="third_autocorr_ineq",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="few_modes",
        constraint_type="soft",
        objective_type="maximize",
        problem_scale="small",
    ),
    "uncertainty_ineq": TaskProfile(
        name="uncertainty_ineq",
        eval_cost="fast",
        signal_density="dense",
        search_space="continuous",
        decomposable="monolithic",
        multimodal="few_modes",
        constraint_type="hard",
        objective_type="maximize",
        problem_scale="small",
    ),
}


def get_task_profile(task_name: str) -> TaskProfile:
    """获取任务画像，找不到时返回默认画像。"""
    if task_name in TASK_PROFILES:
        return TASK_PROFILES[task_name]
    return TaskProfile(name=task_name)
