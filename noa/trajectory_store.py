"""TrajectoryStore — 持久化轨迹存储，支持查询、对比。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from noa.core.protocol import Trajectory


class TrajectoryStore:
    """持久化的轨迹存储，支持查询、对比、回放。"""

    def __init__(self, store_dir: str):
        self.store_dir = os.path.abspath(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)

    def _episode_path(self, episode_id: str) -> str:
        return os.path.join(self.store_dir, f"{episode_id}.json")

    def save_episode(
        self, episode_id: str, trajectories: list[Trajectory], metadata: dict
    ) -> None:
        """保存一组轨迹为一个 episode。"""
        data = {
            "episode_id": episode_id,
            "metadata": metadata,
            "trajectories": [asdict(t) for t in trajectories],
        }
        with open(self._episode_path(episode_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def load_episode(self, episode_id: str) -> tuple[list[Trajectory], dict]:
        """加载 episode，返回 (trajectories, metadata)。"""
        path = self._episode_path(episode_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Episode 不存在: {episode_id}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        trajectories = [
            Trajectory(
                **{k: v for k, v in t.items() if k in Trajectory.__dataclass_fields__}
            )
            for t in data.get("trajectories", [])
        ]
        return trajectories, data.get("metadata", {})

    def query(self, filter_fn=None, limit: int = 10) -> list[str]:
        """查询 episode_ids，可选 filter_fn(metadata) -> bool。"""
        results = []
        for fname in sorted(os.listdir(self.store_dir)):
            if not fname.endswith(".json"):
                continue
            episode_id = fname[:-5]
            if filter_fn is not None:
                try:
                    _, meta = self.load_episode(episode_id)
                    if not filter_fn(meta):
                        continue
                except Exception:
                    continue
            results.append(episode_id)
            if len(results) >= limit:
                break
        return results

    def compare_episodes(self, ep_a: str, ep_b: str) -> dict:
        """对比两个 episode 的分数分布、失败类型等。"""
        trajs_a, meta_a = self.load_episode(ep_a)
        trajs_b, meta_b = self.load_episode(ep_b)

        scores_a = [t.f1 for t in trajs_a]
        scores_b = [t.f1 for t in trajs_b]
        mean_a = sum(scores_a) / len(scores_a) if scores_a else 0
        mean_b = sum(scores_b) / len(scores_b) if scores_b else 0

        failures_a = sum(1 for s in scores_a if s < 0.5)
        failures_b = sum(1 for s in scores_b if s < 0.5)

        errors_a = sum(1 for t in trajs_a if t.error)
        errors_b = sum(1 for t in trajs_b if t.error)

        return {
            "ep_a": {
                "id": ep_a,
                "count": len(trajs_a),
                "mean": mean_a,
                "failures": failures_a,
                "errors": errors_a,
            },
            "ep_b": {
                "id": ep_b,
                "count": len(trajs_b),
                "mean": mean_b,
                "failures": failures_b,
                "errors": errors_b,
            },
            "delta_mean": mean_b - mean_a,
            "delta_failures": failures_b - failures_a,
        }

    def list_episodes(self) -> list[str]:
        """列出所有 episode id。"""
        return [
            fname[:-5]
            for fname in sorted(os.listdir(self.store_dir))
            if fname.endswith(".json")
        ]
