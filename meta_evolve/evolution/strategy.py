"""策略参数化：将搜索策略编码为连续向量 φ。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrategyParams:
    # === 种群管理 ===
    exploration_rate: float = 0.3  # [0,1] 随机 vs 精英采样
    selection_temperature: float = 1.0  # (0,∞) softmax 温度
    elite_ratio: float = 0.2  # [0,1] 精英保留比例
    mutation_strength: float = 0.5  # [0,1] 变异激进程度
    context_size: int = 3  # {1..5} 给 LLM 的灵感程序数
    diversity_weight: float = 0.3  # [0,1] 选择中的多样性权重

    # === Prompt 策略 ===
    error_analysis: float = 0.5  # [0,1] 是否要求先分析再改
    diff_vs_rewrite: float = 0.3  # [0,1] 局部 diff(0) vs 重写(1)
    show_context_scores: float = 0.7  # [0,1] context 是否带分数排名
    domain_hint: float = 0.5  # [0,1] 领域知识详细程度

    # === LLM 控制 ===
    llm_temperature: float = 0.7  # [0.1, 1.5] LLM 采样温度
    max_tokens_ratio: float = 0.8  # [0,1] 输出长度预算

    # === 评估策略 ===
    parent_best_bias: float = 0.7  # [0,1] 选 parent 偏最优(1) vs 偏多样(0)
    crossover_rate: float = 0.2  # [0,1] 双 parent 组合概率

    # === 种群结构 ===
    num_islands: int = 3  # {1..8} 岛屿数量
    migration_rate: float = 0.2  # [0,1] 每次迁移的比例

    PHI_DIM: int = 16

    @classmethod
    def from_config(cls, config: dict | None = None) -> StrategyParams:
        """从 config['strategy'] 构造参数，忽略 dataclass 外字段。"""
        strategy_cfg = (config or {}).get("strategy", {})
        return cls(
            **{
                key: value
                for key, value in strategy_cfg.items()
                if key in cls.__dataclass_fields__ and key != "PHI_DIM"
            }
        )

    def to_vector(self) -> np.ndarray:
        """→ 16 维连续向量。"""

        def inv_sigmoid(x: float) -> float:
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return float(np.log(x / (1 - x)))

        def inv_softplus(x: float) -> float:
            x = max(x, 1e-6)
            return float(np.log(np.exp(x) - 1))

        return np.array(
            [
                inv_sigmoid(self.exploration_rate),
                inv_softplus(self.selection_temperature),
                inv_sigmoid(self.elite_ratio),
                inv_sigmoid(self.mutation_strength),
                float(self.context_size),
                inv_sigmoid(self.diversity_weight),
                inv_sigmoid(self.error_analysis),
                inv_sigmoid(self.diff_vs_rewrite),
                inv_sigmoid(self.show_context_scores),
                inv_sigmoid(self.domain_hint),
                inv_softplus(self.llm_temperature),
                inv_sigmoid(self.max_tokens_ratio),
                inv_sigmoid(self.parent_best_bias),
                inv_sigmoid(self.crossover_rate),
                float(self.num_islands),
                inv_sigmoid(self.migration_rate),
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> StrategyParams:
        """连续向量 → StrategyParams。"""

        def sigmoid(x: float) -> float:
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

        def softplus(x: float) -> float:
            return float(np.log1p(np.exp(np.clip(x, -20, 20))))

        return cls(
            exploration_rate=sigmoid(v[0]),
            selection_temperature=max(softplus(v[1]), 1e-3),
            elite_ratio=sigmoid(v[2]),
            mutation_strength=sigmoid(v[3]),
            context_size=int(np.clip(round(v[4]), 1, 5)),
            diversity_weight=sigmoid(v[5]),
            error_analysis=sigmoid(v[6]),
            diff_vs_rewrite=sigmoid(v[7]),
            show_context_scores=sigmoid(v[8]),
            domain_hint=sigmoid(v[9]),
            llm_temperature=np.clip(softplus(v[10]), 0.1, 1.5),
            max_tokens_ratio=sigmoid(v[11]),
            parent_best_bias=sigmoid(v[12]),
            crossover_rate=sigmoid(v[13]),
            num_islands=int(np.clip(round(v[14]), 1, 8)),
            migration_rate=sigmoid(v[15]),
        )

    def clamp(self) -> StrategyParams:
        """强制约束到合法范围。"""
        return StrategyParams(
            exploration_rate=np.clip(self.exploration_rate, 0.0, 1.0),
            selection_temperature=max(self.selection_temperature, 1e-3),
            elite_ratio=np.clip(self.elite_ratio, 0.05, 0.95),
            mutation_strength=np.clip(self.mutation_strength, 0.0, 1.0),
            context_size=int(np.clip(self.context_size, 1, 5)),
            diversity_weight=np.clip(self.diversity_weight, 0.0, 1.0),
            error_analysis=np.clip(self.error_analysis, 0.0, 1.0),
            diff_vs_rewrite=np.clip(self.diff_vs_rewrite, 0.0, 1.0),
            show_context_scores=np.clip(self.show_context_scores, 0.0, 1.0),
            domain_hint=np.clip(self.domain_hint, 0.0, 1.0),
            llm_temperature=np.clip(self.llm_temperature, 0.1, 1.5),
            max_tokens_ratio=np.clip(self.max_tokens_ratio, 0.0, 1.0),
            parent_best_bias=np.clip(self.parent_best_bias, 0.0, 1.0),
            crossover_rate=np.clip(self.crossover_rate, 0.0, 1.0),
            num_islands=int(np.clip(self.num_islands, 1, 8)),
            migration_rate=np.clip(self.migration_rate, 0.0, 1.0),
        )

    def get_max_tokens(self) -> int:
        """返回 LLM 输出 max tokens（需覆盖推理模型的 thinking tokens）。"""
        return 8192
