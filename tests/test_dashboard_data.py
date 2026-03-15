from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dashboard.data import (
    RunRecord,
    build_run_snapshot,
    classify_children,
    is_active_run,
    pick_default_run,
)
from noa.core.protocol import (
    LayerContext,
    OptimizationBudget,
    SourceFile,
    SystemDescription,
)
from noa.sandbox_manager import SandboxManager
from noa.trajectory_store import TrajectoryStore
from noa.unified_agent import UnifiedOptimizerAgent


class DashboardDataTests(unittest.TestCase):
    def test_pick_default_run_prefers_latest_active_run(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        runs = [
            RunRecord(
                run_id="older-completed",
                run_dir=Path("/tmp/older-completed"),
                current_run={
                    "status": "completed",
                    "updated_at": (now - timedelta(seconds=5)).isoformat(),
                },
            ),
            RunRecord(
                run_id="newer-running",
                run_dir=Path("/tmp/newer-running"),
                current_run={
                    "status": "running",
                    "updated_at": (now - timedelta(seconds=10)).isoformat(),
                },
            ),
        ]

        self.assertEqual(
            pick_default_run(runs, now=now),
            "newer-running",
        )

    def test_is_active_run_requires_running_status_and_recent_update(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(
            is_active_run(
                {
                    "status": "running",
                    "updated_at": (now - timedelta(seconds=20)).isoformat(),
                },
                now=now,
                active_window_sec=60,
            )
        )
        self.assertFalse(
            is_active_run(
                {
                    "status": "completed",
                    "updated_at": (now - timedelta(seconds=20)).isoformat(),
                },
                now=now,
                active_window_sec=60,
            )
        )

    def test_classify_children_filters_stale_or_dead_pids(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        children = [
            {"pid": 101, "started_at": (now - timedelta(seconds=4)).isoformat()},
            {"pid": 202, "started_at": (now - timedelta(seconds=40)).isoformat()},
            {"pid": 303, "started_at": (now - timedelta(seconds=4)).isoformat()},
        ]
        heartbeats = [
            {"pid": 101, "last_heartbeat_at": (now - timedelta(seconds=3)).isoformat()},
            {
                "pid": 202,
                "last_heartbeat_at": (now - timedelta(seconds=30)).isoformat(),
            },
            {"pid": 303, "last_heartbeat_at": (now - timedelta(seconds=3)).isoformat()},
        ]

        active, stale = classify_children(
            children,
            heartbeats,
            now=now,
            stale_after_sec=15,
            pid_exists=lambda pid: pid in {101, 202},
        )

        self.assertEqual([item["pid"] for item in active], [101])
        self.assertCountEqual([item["pid"] for item in stale], [202, 303])

    def test_snapshot_infers_together_provider_from_observed_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run-1"
            (run_dir / "state").mkdir(parents=True)
            (run_dir / "logs" / "llm").mkdir(parents=True)

            now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
            (run_dir / "state" / "current_run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "status": "running",
                        "updated_at": now.isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "state" / "children.json").write_text("[]", encoding="utf-8")
            (run_dir / "state" / "heartbeats.json").write_text("[]", encoding="utf-8")
            (run_dir / "logs" / "llm" / "req.jsonl").write_text(
                json.dumps(
                    {
                        "event": "request_started",
                        "request_id": "req-1",
                        "model": "together_ai/MiniMaxAI/MiniMax-M2.5",
                        "resolved_model": "together_ai/MiniMaxAI/MiniMax-M2.5",
                        "timestamp": now.timestamp(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = build_run_snapshot(
                run_dir, now=now, pid_exists=lambda _pid: False
            )
            self.assertEqual(snapshot["configured_provider"], "together")
            self.assertEqual(snapshot["configured_rpm_limit"], 55)

    def test_snapshot_builds_train_validation_test_evaluation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run-1"
            source_dir = run_dir / "workspace" / "target_systems" / "pubmedqa"
            meta_dir = source_dir / ".noa_meta"
            (run_dir / "state").mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
            (run_dir / "state" / "current_run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "status": "running",
                        "updated_at": now.isoformat(),
                        "source_dir": str(source_dir),
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "state" / "children.json").write_text("[]", encoding="utf-8")
            (run_dir / "state" / "heartbeats.json").write_text("[]", encoding="utf-8")

            (meta_dir / "eval_patch_a.json").write_text(
                json.dumps(
                    {
                        "details": [
                            {
                                "id": "PubMedQA_train_1",
                                "question": "Train question",
                                "ground_truth": "yes",
                                "prediction": "Answer: yes",
                                "extracted": "yes",
                                "accuracy": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (meta_dir / "validation_eval_step_3_patch_a.json").write_text(
                json.dumps(
                    {
                        "details": [
                            {
                                "id": "PubMedQA_val_2",
                                "question": "Validation question",
                                "ground_truth": "no",
                                "prediction": "Answer: no",
                                "extracted": "no",
                                "accuracy": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (meta_dir / "test_eval_step_4_patch_a.json").write_text(
                json.dumps(
                    {
                        "details": [
                            {
                                "id": "PubMedQA_test_3",
                                "question": "Test question",
                                "ground_truth": "maybe",
                                "prediction": "Answer: maybe",
                                "extracted": "maybe",
                                "accuracy": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            events = [
                {
                    "ts": now.isoformat(),
                    "event": "eval_candidate",
                    "label": "patch_a",
                    "dataset_split": "train",
                    "before_score": 40.0,
                    "after_score": 60.0,
                    "accepted": True,
                    "rationale": "Improve train behavior",
                    "ops": [{"op": "update", "file_path": "config.py"}],
                    "detail_file": ".noa_meta/eval_patch_a.json",
                },
                {
                    "ts": (now + timedelta(seconds=1)).isoformat(),
                    "event": "validation_eval",
                    "label": "patch_a",
                    "dataset_split": "validation",
                    "baseline_score": 55.0,
                    "val_score": 61.0,
                    "accepted": True,
                    "rationale": "Validate candidate",
                    "ops": [{"op": "update", "file_path": "config.py"}],
                    "detail_file": ".noa_meta/validation_eval_step_3_patch_a.json",
                },
                {
                    "ts": (now + timedelta(seconds=2)).isoformat(),
                    "event": "test_eval_candidate",
                    "label": "patch_a",
                    "dataset_split": "test",
                    "baseline_score": 58.0,
                    "test_score": 64.0,
                    "accepted": True,
                    "rationale": "Held-out final check",
                    "ops": [{"op": "update", "file_path": "config.py"}],
                    "detail_file": ".noa_meta/test_eval_step_4_patch_a.json",
                },
            ]
            (run_dir / "state" / "run_events.jsonl").write_text(
                "\n".join(json.dumps(item) for item in events) + "\n",
                encoding="utf-8",
            )

            snapshot = build_run_snapshot(
                run_dir, now=now, pid_exists=lambda _pid: False
            )
            eval_rows = snapshot["evaluation_rows"]
            self.assertEqual(
                [row["split"] for row in eval_rows], ["test", "validation", "train"]
            )
            self.assertEqual(
                eval_rows[0]["detail_rows"][0]["question"], "Test question"
            )
            self.assertEqual(eval_rows[1]["detail_rows"][0]["split"], "validation")
            self.assertEqual(eval_rows[2]["detail_rows"][0]["split"], "train")


class UnifiedAgentBatchEvalTests(unittest.TestCase):
    def _make_agent(self, tmp_path: Path) -> UnifiedOptimizerAgent:
        source_dir = tmp_path / "target"
        noa_dir = tmp_path / "noa"
        source_dir.mkdir()
        noa_dir.mkdir()
        (source_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (noa_dir / "engine.py").write_text("FRAMEWORK = 1\n", encoding="utf-8")

        sys_desc = SystemDescription(
            workflow_summary="dummy workflow",
            component_names=["dummy"],
            source_files=[SourceFile(path="app.py", content="VALUE = 1\n")],
            source_dir=str(source_dir),
            baseline_score=0,
        )
        layer_context = LayerContext(
            layer_id="L1",
            level=1,
            writable_root=str(source_dir),
            readable_roots=[str(source_dir)],
            parent_history=[],
            max_depth=3,
            max_spawn_calls=1,
        )
        budget = OptimizationBudget(
            max_steps=5,
            max_llm_calls=10,
            max_no_improve_steps=5,
            target_delta=float("inf"),
            max_spawn_calls=1,
        )
        sandbox = SandboxManager(
            str(source_dir),
            str(tmp_path / "workspace"),
            layer_context,
        )
        sandbox.save_accepted_snapshot()

        return UnifiedOptimizerAgent(
            sys_desc=sys_desc,
            source_dir=str(source_dir),
            target_factory=lambda path: {"path": path},
            target={"path": str(source_dir)},
            dataset=[],
            eval_fn=lambda target, dataset: {"score": 0.0, "details": []},
            score_fn=lambda prediction, ground_truth: 0.0,
            model="dummy-model",
            layer_context=layer_context,
            budget=budget,
            sandbox_manager=sandbox,
            trajectory_store=TrajectoryStore(str(tmp_path / "trajectories")),
            component_probe=None,
            observer_search_roots=[str(source_dir)],
            noa_dir=str(noa_dir),
            project_root=str(tmp_path),
            dataset_pickle_path=str(tmp_path / "dataset.pkl"),
            spawn_config={"mini_l1": {}, "l2": {}},
            train_pool=[],
            test_set=[],
            train_sample_size=1,
            n_samples=1,
            top_k=3,
        )

    def test_eval_candidate_auto_batches_pending_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(Path(tmp_dir))
            agent._active_analysis_round = 1
            agent._current_train_baseline = 50.0
            agent.train_pool = [{"question": "q1"}, {"question": "q2"}]

            agent._tool_checkpoint_candidate(
                {
                    "label": "cand_a",
                    "rationale": "first hypothesis",
                    "ops": [
                        {
                            "op": "update",
                            "file_path": "app.py",
                            "search": "VALUE = 1",
                            "replace": "VALUE = 2",
                        }
                    ],
                }
            )
            agent._tool_checkpoint_candidate(
                {
                    "label": "cand_b",
                    "rationale": "second hypothesis",
                    "ops": [
                        {
                            "op": "update",
                            "file_path": "app.py",
                            "search": "VALUE = 1",
                            "replace": "VALUE = 3",
                        }
                    ],
                }
            )

            seen_labels: list[str] = []

            def fake_eval_in_sandbox(candidate_dir, *args, **kwargs):
                label = Path(candidate_dir).name
                seen_labels.append(label)
                score = 0.81 if label == "cand_a" else 0.67
                return {"ok": True, "score": score, "details": []}

            agent.sandbox.eval_in_sandbox = fake_eval_in_sandbox

            result = agent._tool_eval_candidate({"label": "cand_a"})

            self.assertTrue(result["ok"])
            self.assertEqual(result["mode"], "auto_batch")
            self.assertEqual(set(result["batch_labels"]), {"cand_a", "cand_b"})
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(set(seen_labels), {"cand_a", "cand_b"})
            self.assertEqual(result["results"]["cand_a"]["candidate_score"], 81.0)
            self.assertEqual(result["results"]["cand_b"]["candidate_score"], 67.0)

    def test_eval_candidate_stays_single_when_only_one_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(Path(tmp_dir))
            agent._active_analysis_round = 1
            agent._current_train_baseline = 50.0
            agent.train_pool = [{"question": "q1"}]

            agent._tool_checkpoint_candidate(
                {
                    "label": "cand_only",
                    "rationale": "only hypothesis",
                    "ops": [
                        {
                            "op": "update",
                            "file_path": "app.py",
                            "search": "VALUE = 1",
                            "replace": "VALUE = 9",
                        }
                    ],
                }
            )

            seen_labels: list[str] = []

            def fake_eval_in_sandbox(candidate_dir, *args, **kwargs):
                seen_labels.append(Path(candidate_dir).name)
                return {"ok": True, "score": 0.74, "details": []}

            agent.sandbox.eval_in_sandbox = fake_eval_in_sandbox

            result = agent._tool_eval_candidate({"label": "cand_only"})

            self.assertTrue(result["ok"])
            self.assertEqual(result["candidate_label"], "cand_only")
            self.assertNotEqual(result.get("mode"), "auto_batch")
            self.assertEqual(seen_labels, ["cand_only"])

    def test_eval_candidate_history_records_train_detail_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(Path(tmp_dir))
            agent._current_train_baseline = 50.0
            agent.train_pool = [{"question": "q1"}]

            agent._tool_checkpoint_candidate(
                {
                    "label": "cand_eval",
                    "rationale": "candidate rationale",
                    "ops": [
                        {
                            "op": "update",
                            "file_path": "app.py",
                            "search": "VALUE = 1",
                            "replace": "VALUE = 2",
                        }
                    ],
                }
            )

            agent.sandbox.eval_in_sandbox = lambda *args, **kwargs: {
                "ok": True,
                "score": 0.75,
                "details": [
                    {
                        "id": "PubMedQA_train_1",
                        "question": "Q",
                        "prediction": "Answer: yes",
                        "accuracy": 1.0,
                    }
                ],
            }

            result = agent._tool_eval_candidate({"label": "cand_eval"})

            self.assertEqual(result["candidate_score"], 75.0)
            history_entry = agent._history[-1]
            self.assertEqual(history_entry["action"], "eval_candidate")
            self.assertEqual(history_entry["dataset_split"], "train")
            self.assertTrue(
                history_entry["detail_file"].startswith(".noa_meta/eval_cand_eval")
            )

    def test_commit_best_before_observe_records_validation_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(Path(tmp_dir))
            candidate_dir = Path(agent.sandbox._candidates_dir) / "cand_val"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            (candidate_dir / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
            agent._episode_counter = 1
            agent._baseline_score = 60.0
            agent.val_set = [{"question": "vq"}]
            agent._top_candidates = [
                {
                    "label": "cand_val",
                    "score": 70.0,
                    "ops": [{"op": "update", "file_path": "app.py"}],
                    "rationale": "validate this candidate",
                }
            ]
            agent.eval_fn = lambda *args, **kwargs: {
                "score": 0.55,
                "details": [
                    {
                        "id": "PubMedQA_val_1",
                        "question": "Validation Q",
                        "prediction": "Answer: no",
                        "accuracy": 1.0,
                    }
                ],
            }

            agent._commit_best_before_observe()

            validation_entries = [
                entry
                for entry in agent._history
                if entry.get("action") == "validation_eval"
            ]
            self.assertEqual(len(validation_entries), 1)
            self.assertEqual(validation_entries[0]["dataset_split"], "validation")
            self.assertTrue(
                validation_entries[0]["detail_file"].startswith(
                    ".noa_meta/validation_eval_step_"
                )
            )


if __name__ == "__main__":
    unittest.main()
