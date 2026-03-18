from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from noa_new.agent import MinimalAgent
from noa_new.artifacts import ArtifactStore


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_bootstrap_l1_creates_expected_files(tmp_path: Path):
    target_dir = tmp_path / "source_target"
    _write(target_dir / "pipeline.py", "print('hello')\n")
    train = _write(tmp_path / "train.pkl", "train")
    val = _write(tmp_path / "val.pkl", "val")
    test = _write(tmp_path / "test.pkl", "test")

    store = ArtifactStore(tmp_path / "workspace")
    manifest = store.bootstrap_l1(
        target_system_dir=target_dir,
        train_pool_path=train,
        val_set_path=val,
        test_set_path=test,
    )

    workspace = Path(manifest["workspace"])
    assert (workspace / "target_system" / "pipeline.py").exists()
    assert (workspace / "WORKFLOW.md").exists()
    assert (workspace / "trace.jsonl").exists()
    assert (workspace / "metrics.json").exists()
    assert (workspace / "pareto.json").exists()
    assert (workspace / "schemas" / "trace_schema.json").exists()
    assert (workspace / "schemas" / "l2_patch_schema.json").exists()
    assert (workspace / "l1_config" / "system_prompt.md").exists()
    metrics = json.loads((workspace / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["current_strategy"] == "baseline"


def test_bootstrap_l2_copies_framework_and_parent_context(tmp_path: Path):
    framework_dir = tmp_path / "framework_source"
    _write(framework_dir / "__init__.py", "")
    _write(framework_dir / "engine.py", "ENGINE = True\n")
    target_dir = tmp_path / "target_source"
    _write(target_dir / "pipeline.py", "PIPE = True\n")
    parent_context = _write(tmp_path / "parent.json", '{"history": []}\n')

    store = ArtifactStore(tmp_path / "workspace")
    manifest = store.bootstrap_l2(
        framework_dir=framework_dir,
        target_system_dir=target_dir,
        parent_context_path=parent_context,
    )

    workspace = Path(manifest["workspace"])
    assert (workspace / "noa" / "engine.py").exists()
    assert (workspace / "target_system" / "pipeline.py").exists()
    assert (workspace / "parent_context.json").exists()
    assert (workspace / "trace.jsonl").exists()
    assert (workspace / "metrics.json").exists()
    assert (workspace / "pareto.json").exists()
    assert (workspace / "schemas" / "trace_schema.json").exists()
    assert (workspace / "l1_config" / "loop_policy.md").exists()
    assert (workspace / "l2_patches.jsonl").exists()
    assert (workspace / "l2_config" / "WORKFLOW.md").exists()


def test_agent_runs_tool_call_then_finishes(monkeypatch, tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.bootstrap_l1()

    responses = iter(
        [
            SimpleNamespace(
                text="",
                tool_calls=[
                    SimpleNamespace(
                        id="call-1", name="bash", arguments={"command": "pwd"}
                    )
                ],
            ),
            SimpleNamespace(text="done", tool_calls=[]),
        ]
    )

    monkeypatch.setattr(
        "noa_new.agent._llm_call_with_tools", lambda **_: next(responses)
    )

    agent = MinimalAgent(
        workspace=str(tmp_path),
        target_description="Improve target accuracy",
        max_turns=3,
    )
    result = agent.run()

    assert result["final_text"] == "done"
    assert result["tool_calls"] == 1
    summary = json.loads(
        (tmp_path / ".noa_new" / "last_run.json").read_text(encoding="utf-8")
    )
    assert summary["tool_calls"] == 1
