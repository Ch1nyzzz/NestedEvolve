"""SandboxManager — candidate 级快照、恢复和 checkpoint。"""

from __future__ import annotations

import os
import shutil

from noa.core.protocol import PatchValidationError, StructuredPatch
from noa.patch_protocol import apply_patch_ops
from noa.stages.initiator import collect_sources


class SandboxManager:
    """管理 candidate-level 的快照、恢复和 checkpoint。"""

    def __init__(self, source_dir: str, workspace_dir: str, layer_context=None):
        self.source_dir = os.path.abspath(source_dir)
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.layer_context = layer_context
        self._snapshots_dir = os.path.join(workspace_dir, "_sandbox_snapshots")
        self._candidates_dir = os.path.join(workspace_dir, "_sandbox_candidates")
        os.makedirs(self._snapshots_dir, exist_ok=True)
        os.makedirs(self._candidates_dir, exist_ok=True)
        self._accepted_snapshot: str | None = None

    def save_accepted_snapshot(self) -> str:
        """保存当前 source_dir 为 accepted 快照（rolling baseline）。"""
        self._accepted_snapshot = "accepted"
        return self.snapshot("accepted")

    def _get_checkpoint_base(self) -> str:
        """返回 checkpoint 应该复制的基础目录。"""
        if self._accepted_snapshot:
            snap = os.path.join(self._snapshots_dir, self._accepted_snapshot)
            if os.path.isdir(snap):
                return snap
        return self.source_dir

    def snapshot(self, label: str) -> str:
        """保存当前 source_dir 状态为命名快照，返回快照路径。"""
        snap_dir = os.path.join(self._snapshots_dir, label)
        if os.path.exists(snap_dir):
            shutil.rmtree(snap_dir)
        shutil.copytree(self.source_dir, snap_dir)
        return snap_dir

    def restore(self, label: str) -> None:
        """从快照恢复 source_dir 到指定状态。"""
        snap_dir = os.path.join(self._snapshots_dir, label)
        if not os.path.isdir(snap_dir):
            raise FileNotFoundError(f"快照不存在: {label}")
        # 清空 source_dir 并从快照复制
        if os.path.exists(self.source_dir):
            shutil.rmtree(self.source_dir)
        shutil.copytree(snap_dir, self.source_dir)

    def checkpoint_candidate(
        self, label: str, patch: StructuredPatch
    ) -> tuple[str, list[PatchValidationError]]:
        """在干净快照上应用 patch，写入独立 candidate 目录。

        不修改 source_dir 本体。返回 (candidate_dir, errors)。
        """
        candidate_dir = os.path.join(self._candidates_dir, label)
        if os.path.exists(candidate_dir):
            shutil.rmtree(candidate_dir)
        shutil.copytree(self._get_checkpoint_base(), candidate_dir)

        # 收集候选目录的源文件
        source_files = collect_sources(candidate_dir)
        modified, errors, _ = apply_patch_ops(source_files, patch.ops)
        if errors:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            return "", errors

        # 将修改写入候选目录
        for sf in modified:
            fpath = os.path.join(candidate_dir, sf.path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(sf.content)

        # 处理 delete ops
        for op in patch.ops:
            if op.op == "delete":
                fpath = os.path.join(candidate_dir, os.path.normpath(op.file_path))
                if os.path.exists(fpath):
                    os.remove(fpath)

        return candidate_dir, []

    def eval_in_sandbox(
        self,
        candidate_dir: str,
        target_factory,
        eval_fn,
        dataset,
        n_samples: int,
        seed: int = 42,
    ) -> dict:
        """在 candidate 沙盒中运行评估，返回结构化结果。"""
        import random

        rng = random.Random(seed)
        sampled = rng.sample(dataset, min(n_samples, len(dataset)))
        try:
            target = target_factory(candidate_dir)
            result = eval_fn(target, sampled)
            return {
                "ok": True,
                "score": result["score"],
                "details": result.get("details", []),
            }
        except Exception as e:
            return {"ok": False, "score": 0.0, "error": str(e)[:500]}

    def accept_candidate(self, label: str) -> None:
        """将 candidate 提交到 source_dir。这是唯一的 commit 路径。"""
        candidate_dir = os.path.join(self._candidates_dir, label)
        if not os.path.isdir(candidate_dir):
            raise FileNotFoundError(f"候选不存在: {label}")
        if os.path.exists(self.source_dir):
            shutil.rmtree(self.source_dir)
        shutil.copytree(candidate_dir, self.source_dir)
        # 刷新 accepted 快照（rolling baseline）
        self.save_accepted_snapshot()

    def cleanup_candidate(self, label: str) -> None:
        """清理 candidate 目录。"""
        candidate_dir = os.path.join(self._candidates_dir, label)
        if os.path.isdir(candidate_dir):
            shutil.rmtree(candidate_dir, ignore_errors=True)

    def list_snapshots(self) -> list[str]:
        """列出所有快照标签。"""
        if not os.path.isdir(self._snapshots_dir):
            return []
        return sorted(os.listdir(self._snapshots_dir))
