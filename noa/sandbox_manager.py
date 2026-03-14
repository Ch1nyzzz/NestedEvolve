"""SandboxManager — candidate 级快照、恢复和 checkpoint。"""

from __future__ import annotations

import difflib
import logging
import os
import shutil

from noa.core.protocol import PatchValidationError, StructuredPatch
from noa.patch_protocol import apply_patch_ops
from noa.runtime import bind_scope, run_forked
from noa.stages.initiator import collect_sources

log = logging.getLogger(__name__)


def _copytree_fast(src: str, dst: str) -> str:
    """使用 hardlink 快速复制目录树，失败时回退到普通复制。

    注意: hardlink 文件共享 inode，写入前必须先 os.unlink() 断开链接。
    仅用于 checkpoint_candidate/merge_candidates（写入路径已做 unlink 处理）。
    """
    try:
        return shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        log.debug("[copytree_fast] hardlink failed, fallback to regular copy")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        return shutil.copytree(src, dst)


def _merge_file_versions(
    base: str, candidates: list[tuple[str, str]]
) -> tuple[str, list[str]]:
    """行级合并多个候选对同一文件的修改。

    Args:
        base: 基线文件内容
        candidates: [(label, content), ...] 各候选的修改版本

    Returns:
        (merged_content, conflicts) — conflicts 为空表示无冲突
    """
    base_lines = base.splitlines(keepends=True)

    # 收集每个候选的变更区域: (base_start, base_end, new_lines, label)
    all_regions: list[tuple[int, int, list[str], str]] = []
    for label, content in candidates:
        mod_lines = content.splitlines(keepends=True)
        sm = difflib.SequenceMatcher(None, base_lines, mod_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                all_regions.append((i1, i2, mod_lines[j1:j2], label))

    all_regions.sort(key=lambda r: (r[0], r[1]))

    merged: list[str] = []
    conflicts: list[str] = []
    prev_end = 0
    i = 0

    while i < len(all_regions):
        start, end, new_lines, label = all_regions[i]
        # 检查后续区域是否与当前重叠
        group = [all_regions[i]]
        j = i + 1
        group_end = end
        while j < len(all_regions) and all_regions[j][0] < group_end:
            group.append(all_regions[j])
            group_end = max(group_end, all_regions[j][1])
            j += 1

        # 添加 base 中未修改的部分
        merged.extend(base_lines[prev_end:start])

        if len(group) == 1:
            # 无冲突，直接应用
            merged.extend(new_lines)
        else:
            # 多个候选修改了重叠区域
            labels = list(dict.fromkeys(r[3] for r in group))
            conflicts.append(
                f"lines {start + 1}-{group_end}: conflict between {', '.join(labels)}"
            )
            # 取第一个候选的版本（按 label 排序确保确定性）
            merged.extend(group[0][2])

        prev_end = group_end
        i = j

    merged.extend(base_lines[prev_end:])
    return "".join(merged), conflicts


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
        _copytree_fast(self._get_checkpoint_base(), candidate_dir)

        # 收集候选目录的源文件
        source_files = collect_sources(candidate_dir)
        modified, errors, _ = apply_patch_ops(source_files, patch.ops)
        if errors:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            return "", errors

        # 将修改写入候选目录（先 unlink 断开 hardlink，避免污染源）
        for sf in modified:
            fpath = os.path.join(candidate_dir, sf.path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            if os.path.exists(fpath):
                os.unlink(fpath)
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
        timeout_sec: float | None = None,
        inline: bool = False,
    ) -> dict:
        """在 candidate 沙盒中运行评估，返回结构化结果。

        Args:
          inline: True 时直接在当前线程执行（适用于 ThreadPoolExecutor 并行场景，
                  避免线程+fork 在 macOS 上死锁）。
        返回字段:
          ok, status, score, error, error_type, timed_out, duration_sec
        """
        import random
        import time as _time

        rng = random.Random(seed)
        sampled = rng.sample(dataset, min(n_samples, len(dataset)))
        if timeout_sec is None:
            timeout_sec = float(os.getenv("NOA_SANDBOX_EVAL_TIMEOUT_SEC", "600"))

        def _evaluate():
            target = target_factory(candidate_dir)
            return eval_fn(target, sampled)

        if inline:
            started = _time.time()
            try:
                result = _evaluate()
                duration = round(_time.time() - started, 3)
                out = {
                    "ok": True,
                    "status": "success",
                    "score": result["score"],
                    "details": result.get("details", []),
                    "timed_out": False,
                    "duration_sec": duration,
                }
                if result.get("subprocess_errors"):
                    out["subprocess_errors"] = result["subprocess_errors"]
                return out
            except Exception as e:
                duration = round(_time.time() - started, 3)
                return {
                    "ok": False,
                    "status": "crash",
                    "score": 0.0,
                    "error": str(e)[:500],
                    "error_type": "runtime_crash",
                    "timed_out": False,
                    "duration_sec": duration,
                }

        with bind_scope(candidate_label=os.path.basename(candidate_dir)):
            managed = run_forked(
                _evaluate,
                timeout_sec=timeout_sec,
                kind="sandbox_eval",
                heartbeat_message=f"evaluating {os.path.basename(candidate_dir)}",
            )

        if not managed.ok:
            status = "timeout" if managed.timed_out else "crash"
            error_msg = managed.error[:500] if managed.error else "Unknown error"
            if managed.timed_out:
                error_msg = (
                    f"Sandbox eval timed out after {managed.duration_sec:.0f}s "
                    f"(limit={timeout_sec:.0f}s). {error_msg}"
                )
            return {
                "ok": False,
                "status": status,
                "score": 0.0,
                "error": error_msg,
                "error_type": "timeout"
                if managed.timed_out
                else (managed.error_type or "runtime_crash"),
                "timed_out": managed.timed_out,
                "duration_sec": managed.duration_sec,
            }

        result = managed.payload or {}
        out = {
            "ok": True,
            "status": "success",
            "score": result["score"],
            "details": result.get("details", []),
            "timed_out": False,
            "duration_sec": managed.duration_sec,
        }
        if result.get("subprocess_errors"):
            out["subprocess_errors"] = result["subprocess_errors"]
        return out

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

    def merge_candidates(
        self, label: str, source_labels: list[str]
    ) -> tuple[str, list[str]]:
        """合并多个候选的 patch 到一个组合候选。

        对每个源候选，找出相对 baseline 的文件差异并合并：
        - 仅一个候选修改的文件 → 直接复制
        - 多个候选修改同一文件 → 行级合并，冲突区域取第一个候选版本

        Returns:
            (candidate_dir, conflicts) — candidate_dir 为空字符串表示失败
        """
        base_dir = self._get_checkpoint_base()
        candidate_dir = os.path.join(self._candidates_dir, label)
        if os.path.exists(candidate_dir):
            shutil.rmtree(candidate_dir)
        _copytree_fast(base_dir, candidate_dir)

        # 校验源候选是否存在
        missing = []
        for src in source_labels:
            src_dir = os.path.join(self._candidates_dir, src)
            if not os.path.isdir(src_dir):
                missing.append(src)
        if missing:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            return "", [f"Candidate not found: {m}" for m in missing]

        # 收集每个文件被哪些候选修改了
        # rel_path -> [(label, content)]
        file_changes: dict[str, list[tuple[str, str]]] = {}
        # rel_path -> [label] — 记录哪些候选删除了哪些基线文件
        file_deletions: dict[str, list[str]] = {}

        # 收集基线文件集合
        base_files: set[str] = set()
        for root, _, files in os.walk(base_dir):
            for fname in files:
                rel_path = os.path.relpath(os.path.join(root, fname), base_dir)
                base_files.add(rel_path)

        for src_label in source_labels:
            src_dir = os.path.join(self._candidates_dir, src_label)
            src_files: set[str] = set()
            for root, _, files in os.walk(src_dir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(src_path, src_dir)
                    src_files.add(rel_path)
                    base_path = os.path.join(base_dir, rel_path)

                    with open(src_path, encoding="utf-8", errors="replace") as f:
                        src_content = f.read()
                    base_content = ""
                    if os.path.exists(base_path):
                        with open(base_path, encoding="utf-8", errors="replace") as f:
                            base_content = f.read()

                    if src_content != base_content:
                        file_changes.setdefault(rel_path, []).append(
                            (src_label, src_content)
                        )

            # 检测该候选删除了哪些基线文件
            for rel_path in base_files - src_files:
                file_deletions.setdefault(rel_path, []).append(src_label)

        # 处理文件删除：如果所有候选都删除了某文件，从合并结果中移除
        # 如果只有部分候选删除，保留该文件（保守策略）
        for rel_path, deleting_labels in file_deletions.items():
            if len(deleting_labels) == len(source_labels):
                dest = os.path.join(candidate_dir, rel_path)
                if os.path.exists(dest):
                    os.unlink(dest)

        # 合并文件变更（先 unlink 断开 hardlink）
        conflicts: list[str] = []
        for rel_path, changes in file_changes.items():
            dest = os.path.join(candidate_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                os.unlink(dest)

            if len(changes) == 1:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(changes[0][1])
            else:
                # 读取 base 内容做行级合并
                base_path = os.path.join(base_dir, rel_path)
                base_content = ""
                if os.path.exists(base_path):
                    with open(base_path, encoding="utf-8", errors="replace") as f:
                        base_content = f.read()
                merged, file_conflicts = _merge_file_versions(base_content, changes)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(merged)
                for c in file_conflicts:
                    conflicts.append(f"{rel_path} {c}")

        return candidate_dir, conflicts

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
