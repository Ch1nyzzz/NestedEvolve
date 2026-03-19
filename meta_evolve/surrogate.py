"""Task-Conditioned MLP 代理模型 g_ψ(φ, s_t, z_task) → Δfitness。"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurrogateModel:
    def __init__(
        self,
        phi_dim: int = 6,
        numeric_state_dim: int = 8,
        semantic_state_dim: int = 17,
        n_tasks: int = 11,
        task_embed_dim: int = 8,
        hidden: int = 64,
        buffer_size: int = 5000,
        lr: float = 1e-3,
    ):
        self.phi_dim = phi_dim
        self.state_dim = numeric_state_dim + semantic_state_dim
        self.task_embed_dim = task_embed_dim
        self.n_tasks = n_tasks

        self.task_embedding = nn.Embedding(n_tasks, task_embed_dim)
        total_input = phi_dim + self.state_dim + task_embed_dim
        self.net = MLP(total_input, hidden, 1)

        # 输入归一化（running mean/std）
        self.input_dim = phi_dim + self.state_dim  # task_embed 不归一化
        self.register_running = True
        self._running_mean = np.zeros(self.input_dim, dtype=np.float32)
        self._running_var = np.ones(self.input_dim, dtype=np.float32)
        self._running_count = 0

        all_params = list(self.net.parameters()) + list(
            self.task_embedding.parameters()
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

        self.buffer: deque[tuple[np.ndarray, np.ndarray, int, float]] = deque(
            maxlen=buffer_size
        )

    def _normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """用 running stats 归一化 [φ, state] 部分。"""
        mean = torch.tensor(self._running_mean, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(self._running_var, dtype=torch.float32) + 1e-8)
        if raw.dim() == 1:
            return (raw - mean) / std
        return (raw - mean.unsqueeze(0)) / std.unsqueeze(0)

    def _update_running_stats(self, phi_state: np.ndarray):
        """Welford online 更新 mean/var。"""
        for x in phi_state:
            self._running_count += 1
            delta = x - self._running_mean
            self._running_mean += delta / self._running_count
            delta2 = x - self._running_mean
            self._running_var += (
                delta * delta2 - self._running_var
            ) / self._running_count

    def _build_input(
        self,
        phi: torch.Tensor,
        state: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """拼接 [φ, state, task_embed] 为模型输入（归一化）。"""
        task_idx = torch.tensor([task_id], dtype=torch.long)
        task_embed = self.task_embedding(task_idx).squeeze(0)
        raw = torch.cat([phi, state])
        normed = self._normalize(raw)
        return torch.cat([normed, task_embed])

    def predict(self, phi: np.ndarray, state: np.ndarray, task_id: int) -> float:
        """g_ψ(φ, s_t, z_task) → E[Δfitness]。"""
        with torch.no_grad():
            phi_t = torch.tensor(phi, dtype=torch.float32)
            state_t = torch.tensor(state, dtype=torch.float32)
            x = self._build_input(phi_t, state_t, task_id)
            return self.net(x.unsqueeze(0)).item()

    def predict_differentiable(
        self,
        phi_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """可微版本。只对 phi_tensor 保持梯度，state 和 task_embed detach。"""
        task_idx = torch.tensor([task_id], dtype=torch.long)
        task_embed = self.task_embedding(task_idx).squeeze(0).detach()
        state_detached = state_tensor.detach()
        # 归一化（保持 phi 梯度）
        raw = torch.cat([phi_tensor, state_detached])
        mean = torch.tensor(self._running_mean, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(self._running_var, dtype=torch.float32) + 1e-8)
        normed = (raw - mean) / std
        x = torch.cat([normed, task_embed])
        return self.net(x.unsqueeze(0)).squeeze()

    def compute_gradient(
        self, phi: np.ndarray, state: np.ndarray, task_id: int
    ) -> np.ndarray:
        """∇_φ g_ψ(φ, s_t, z_task) — 只对 φ 求梯度。"""
        phi_t = torch.tensor(phi, dtype=torch.float32, requires_grad=True)
        state_t = torch.tensor(state, dtype=torch.float32)
        pred = self.predict_differentiable(phi_t, state_t, task_id)
        pred.backward()
        return phi_t.grad.numpy()

    def update(self, transitions: list[tuple], epochs: int = 10) -> float:
        """用 transitions 更新 surrogate (MSE loss)。

        Args:
            transitions: [(φ, s_t, task_id, Δfitness), ...]
            epochs: 训练轮数

        Returns:
            最终 loss
        """
        # 更新 running stats（先于 buffer 判断，确保早期样本也被统计）
        new_phi_state = np.array([np.concatenate([d[0], d[1]]) for d in transitions])
        self._update_running_stats(new_phi_state)

        # 加入 buffer
        for t in transitions:
            self.buffer.append(t)

        if len(self.buffer) < 10:
            return float("inf")

        # 构建数据集
        data = list(self.buffer)
        phis = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
        states = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
        task_ids = torch.tensor([d[2] for d in data], dtype=torch.long)
        targets = torch.tensor([d[3] for d in data], dtype=torch.float32)

        # 归一化 [phi, state]
        raw = torch.cat([phis, states], dim=1)
        normed = self._normalize(raw)

        loss_val = 0.0
        batch_size = min(64, len(data))

        for _ in range(epochs):
            indices = torch.randperm(len(data))[:batch_size]
            b_normed = normed[indices]
            b_tid = task_ids[indices]
            b_target = targets[indices]

            task_embeds = self.task_embedding(b_tid)
            x = torch.cat([b_normed, task_embeds], dim=1)
            pred = self.net(x).squeeze(-1)

            loss = nn.functional.mse_loss(pred, b_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()

        return loss_val

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "task_embedding": self.task_embedding.state_dict(),
            "running_mean": self._running_mean.copy(),
            "running_var": self._running_var.copy(),
            "running_count": self._running_count,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.task_embedding.load_state_dict(state["task_embedding"])
        if "running_mean" in state:
            self._running_mean = np.array(state["running_mean"], dtype=np.float32)
            self._running_var = np.array(state["running_var"], dtype=np.float32)
            self._running_count = state["running_count"]
