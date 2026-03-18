"""Workspace bootstrap and artifact helpers for noa_new."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from noa_new.schema_defs import (
    L1_SLOT_TEMPLATES,
    L2_PATCH_SCHEMA,
    TRACE_SCHEMA,
    default_metrics,
)
from noa_new.workflows import L1_WORKFLOW, L2_WORKFLOW


class ArtifactStore:
    """Manage the workspace files expected by noa_new."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    def bootstrap_l1(
        self,
        *,
        target_system_dir: str | Path | None = None,
        train_pool_path: str | Path | None = None,
        val_set_path: str | Path | None = None,
        test_set_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> dict[str, str]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if target_system_dir:
            self._copy_tree(
                target_system_dir, self.workspace / "target_system", overwrite
            )
        self._copy_file(train_pool_path, self.workspace / "train_pool.pkl", overwrite)
        self._copy_file(val_set_path, self.workspace / "val_set.pkl", overwrite)
        self._copy_file(test_set_path, self.workspace / "test_set.pkl", overwrite)

        self._write_text(self.workspace / "WORKFLOW.md", L1_WORKFLOW, overwrite)
        self._write_json(self.workspace / "metrics.json", default_metrics(), overwrite)
        self._write_json(self.workspace / "pareto.json", [], overwrite)
        self._ensure_file(self.workspace / "trace.jsonl", overwrite)

        schema_dir = self.workspace / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(schema_dir / "trace_schema.json", TRACE_SCHEMA, overwrite)
        self._write_json(
            schema_dir / "l2_patch_schema.json", L2_PATCH_SCHEMA, overwrite
        )

        cfg_dir = self.workspace / "l1_config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for name, content in L1_SLOT_TEMPLATES.items():
            self._write_text(cfg_dir / name, content, overwrite)

        manifest = {
            "layer": "L1",
            "workspace": str(self.workspace),
            "workflow": "WORKFLOW.md",
            "target_system": "target_system",
            "artifacts": ["trace.jsonl", "metrics.json", "pareto.json"],
            "schema_dir": "schemas",
            "slots": sorted(L1_SLOT_TEMPLATES),
        }
        self._write_json(
            self.workspace / "workspace_manifest.json", manifest, overwrite
        )
        return manifest

    def bootstrap_l2(
        self,
        *,
        framework_dir: str | Path | None = None,
        target_system_dir: str | Path | None = None,
        parent_context_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> dict[str, str]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if framework_dir:
            self._copy_tree(framework_dir, self.workspace / "noa", overwrite)
        if target_system_dir:
            self._copy_tree(
                target_system_dir, self.workspace / "target_system", overwrite
            )
        self._copy_file(
            parent_context_path, self.workspace / "parent_context.json", overwrite
        )

        self._write_json(self.workspace / "metrics.json", default_metrics(), overwrite)
        self._write_json(self.workspace / "pareto.json", [], overwrite)
        self._ensure_file(self.workspace / "trace.jsonl", overwrite)
        self._ensure_file(self.workspace / "l2_patches.jsonl", overwrite)

        schema_dir = self.workspace / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(schema_dir / "trace_schema.json", TRACE_SCHEMA, overwrite)
        self._write_json(
            schema_dir / "l2_patch_schema.json", L2_PATCH_SCHEMA, overwrite
        )

        cfg_dir = self.workspace / "l1_config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for name, content in L1_SLOT_TEMPLATES.items():
            self._write_text(cfg_dir / name, content, overwrite)

        l2_dir = self.workspace / "l2_config"
        l2_dir.mkdir(parents=True, exist_ok=True)
        self._write_text(l2_dir / "WORKFLOW.md", L2_WORKFLOW, overwrite)

        manifest = {
            "layer": "L2",
            "workspace": str(self.workspace),
            "workflow": "l2_config/WORKFLOW.md",
            "framework": "noa",
            "patch_log": "l2_patches.jsonl",
        }
        self._write_json(self.workspace / "l2_manifest.json", manifest, overwrite)
        return manifest

    def append_trace(self, payload: dict) -> Path:
        return self._append_jsonl(self.workspace / "trace.jsonl", payload)

    def update_metrics(self, payload: dict) -> Path:
        path = self.workspace / "metrics.json"
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(payload)
        self._write_json(path, existing, overwrite=True)
        return path

    def write_pareto(self, payload: list[dict]) -> Path:
        path = self.workspace / "pareto.json"
        self._write_json(path, payload, overwrite=True)
        return path

    def append_l2_patch(self, payload: dict) -> Path:
        return self._append_jsonl(self.workspace / "l2_patches.jsonl", payload)

    def write_run_summary(self, payload: dict) -> Path:
        out_dir = self.workspace / ".noa_new"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "last_run.json"
        self._write_json(path, payload, overwrite=True)
        return path

    def _append_jsonl(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def _copy_tree(self, source: str | Path, dest: Path, overwrite: bool) -> None:
        source_path = Path(source).resolve()
        if not source_path.exists():
            return
        if dest.exists():
            if not overwrite:
                return
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)

    def _copy_file(
        self, source: str | Path | None, dest: Path, overwrite: bool
    ) -> None:
        if not source:
            return
        source_path = Path(source).resolve()
        if not source_path.exists():
            return
        if dest.exists() and not overwrite:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

    def _write_text(self, path: Path, content: str, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_json(self, path: Path, payload: object, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _ensure_file(self, path: Path, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
