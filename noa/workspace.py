"""Workspace Manager — 原始代码不可变，所有修改在 workspace 副本上进行。"""

from __future__ import annotations

import os
import shutil
from datetime import datetime


class WorkspaceManager:
    """管理 .noa_runs/<run_id>/ 下的 workspace 和 snapshots。

    - workspace/: 运行期间的工作副本，L1/L2 的 commit_to_source 写入这里
    - snapshots/: workspace 在关键时间点的只读快照
    """

    def __init__(self, project_root: str, source_dir: str, run_id: str | None = None):
        self.project_root = os.path.abspath(project_root)
        self.source_dir = os.path.abspath(source_dir)
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self._run_dir = os.path.join(self.project_root, ".noa_runs", self._run_id)
        self._workspace_dir = os.path.join(self._run_dir, "workspace")
        self._snapshots_dir = os.path.join(self._run_dir, "snapshots")

        os.makedirs(self._workspace_dir, exist_ok=True)
        os.makedirs(self._snapshots_dir, exist_ok=True)

    def setup(self) -> tuple[str, str]:
        """复制原始代码到 workspace，拍 init 快照，返回 (ws_noa_dir, ws_source_dir)。"""
        noa_dir = os.path.join(self.project_root, "noa")
        ws_noa = os.path.join(self._workspace_dir, "noa")
        ws_source = os.path.join(
            self._workspace_dir,
            os.path.relpath(self.source_dir, self.project_root),
        )

        # 复制 noa/ 和 source_dir 到 workspace
        if os.path.exists(ws_noa):
            shutil.rmtree(ws_noa)
        shutil.copytree(noa_dir, ws_noa)

        os.makedirs(os.path.dirname(ws_source), exist_ok=True)
        if os.path.exists(ws_source):
            shutil.rmtree(ws_source)
        shutil.copytree(self.source_dir, ws_source)

        # 复制中间目录的 __init__.py，确保包结构完整
        # 例如 source_dir = project_root/target_systems/hotpotqa_rag
        # 需要复制 target_systems/__init__.py 到 workspace/target_systems/__init__.py
        rel_source = os.path.relpath(self.source_dir, self.project_root)
        parts = rel_source.split(os.sep)
        for i in range(len(parts) - 1):
            orig_init = os.path.join(self.project_root, *parts[: i + 1], "__init__.py")
            ws_init = os.path.join(self._workspace_dir, *parts[: i + 1], "__init__.py")
            if os.path.exists(orig_init) and not os.path.exists(ws_init):
                shutil.copy2(orig_init, ws_init)

        # 同时复制 utils/ 以保证 workspace 内 import 可用
        utils_dir = os.path.join(self.project_root, "utils")
        ws_utils = os.path.join(self._workspace_dir, "utils")
        if os.path.isdir(utils_dir):
            if os.path.exists(ws_utils):
                shutil.rmtree(ws_utils)
            shutil.copytree(utils_dir, ws_utils)

        self.snapshot("init")
        return ws_noa, ws_source

    def snapshot(self, label: str) -> str:
        """将当前 workspace 复制为只读快照，返回快照路径。"""
        snap_dir = os.path.join(self._snapshots_dir, label)
        if os.path.exists(snap_dir):
            shutil.rmtree(snap_dir)
        shutil.copytree(self._workspace_dir, snap_dir)
        return snap_dir

    def list_snapshots(self) -> list[str]:
        """列出所有快照标签。"""
        if not os.path.isdir(self._snapshots_dir):
            return []
        return sorted(os.listdir(self._snapshots_dir))

    @property
    def run_dir(self) -> str:
        return self._run_dir

    @property
    def run_id(self) -> str:
        return self._run_id
