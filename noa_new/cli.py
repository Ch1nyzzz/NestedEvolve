"""CLI for bootstrapping and running noa_new."""

from __future__ import annotations

import argparse
import json

from noa_new.agent import DEFAULT_MODEL, MinimalAgent
from noa_new.artifacts import ArtifactStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental minimal NOA runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    _common_bootstrap_args(bootstrap)

    run = subparsers.add_parser("run")
    _common_bootstrap_args(run)
    run.add_argument("--target-description", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument("--task", default=None)
    return parser


def _common_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--layer", choices=["L1", "L2"], default="L1")
    parser.add_argument("--target-system-dir")
    parser.add_argument("--framework-dir")
    parser.add_argument("--train-pool")
    parser.add_argument("--val-set")
    parser.add_argument("--test-set")
    parser.add_argument("--parent-context")
    parser.add_argument("--overwrite", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    store = ArtifactStore(args.workspace)
    if args.layer == "L1":
        manifest = store.bootstrap_l1(
            target_system_dir=args.target_system_dir,
            train_pool_path=args.train_pool,
            val_set_path=args.val_set,
            test_set_path=args.test_set,
            overwrite=args.overwrite,
        )
    else:
        manifest = store.bootstrap_l2(
            framework_dir=args.framework_dir,
            target_system_dir=args.target_system_dir,
            parent_context_path=args.parent_context,
            overwrite=args.overwrite,
        )

    if args.command == "bootstrap":
        print(
            json.dumps({"ok": True, "manifest": manifest}, ensure_ascii=False, indent=2)
        )
        return 0

    agent = MinimalAgent(
        workspace=args.workspace,
        layer=args.layer,
        target_description=args.target_description,
        model=args.model or DEFAULT_MODEL,
        max_turns=args.max_turns,
    )
    result = agent.run(initial_task=args.task)
    print(
        json.dumps(
            {"ok": True, "manifest": manifest, "result": result},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
