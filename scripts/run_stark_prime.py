"""NOA 统一入口 — STaRK-Prime target system。

配置尽量对齐 Optimas 论文中的 STaRK-Prime 系统：
  - RelationScorer / TextScorer: Claude 3 Haiku prompt optimization
  - Aggregator: relation_weight, text_weight in {0.1, 1.0}

运行前置条件：
  - Python 3.11 环境更稳妥（官方 STARK 依赖要求 <3.12）
  - 安装 `stark-qa` / `gdown`
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from noa import Orchestrator
from noa.auto_adapter import auto_adapt
from scripts.archive_trajectories import archive_and_reset_trajectory_dir
from target_systems.stark_prime.evaluate import evaluate_batch, mrr
from utils.data import load_stark_prime_splits


class _TeeWriter:
    def __init__(self, file, orig):
        self._file = file
        self._orig = orig

    def write(self, s):
        self._orig.write(s)
        self._file.write(s)
        self._file.flush()

    def flush(self):
        self._orig.flush()
        self._file.flush()


def _setup_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "noa.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeWriter(log_file, sys.__stdout__)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    for name in ("noa", "noa.unified_agent", "noa.stages", "noa.engine"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(fh)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(fh)

    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print(f"Logs -> {log_path}")


def main():
    load_dotenv()
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/stark_prime_nested.json"
    )
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    data = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    nest = cfg.get("nesting", {})
    spawn = cfg.get("spawn", {})
    history = cfg.get("history", {})
    output = cfg.get("output")

    from utils.llm import resolve_model

    model = resolve_model(opt.get("model", "moonshotai/Kimi-K2.5"), opt.get("provider"))
    metric = data.get("metric", "mrr")
    split_mode = data.get("split_mode", "paper")
    project_root = Path(__file__).resolve().parent.parent
    source_dir = str(project_root / "target_systems" / "stark_prime")
    stark_root = data.get("root") or str(
        project_root / "target_systems" / "stark_prime" / "artifacts"
    )

    _setup_logging(project_root / "logs")

    if history.get("archive_trajectories_on_start", True):
        cleanup_result = archive_and_reset_trajectory_dir(
            str(project_root),
            once=history.get("archive_once", True),
        )
        print(f"Trajectory cleanup: {cleanup_result}")

    print(f"Loading STaRK-Prime data (split_mode={split_mode}, metric={metric})...")
    trainset, valset, testset = load_stark_prime_splits(
        root=stark_root,
        split_mode=split_mode,
    )
    print(
        f"Train set: {len(trainset)} examples | Val set: {len(valset)} examples | "
        f"Test set: {len(testset)} examples"
    )

    print("Setting up adapter...")
    adapter, target_factory = auto_adapt(source_dir=source_dir, model=model)

    orch = Orchestrator(
        source_dir=source_dir,
        target_factory=target_factory,
        dataset=trainset,
        eval_fn=lambda t, d: evaluate_batch(t, d, metric=metric, max_workers=5),
        score_fn=mrr,
        l1_max_steps=opt.get("max_steps", 10),
        l1_n_samples=opt.get("n_samples", 10),
        l1_model=model,
        l1_max_llm_calls=opt.get("max_llm_calls", 150),
        l1_max_no_improve_steps=opt.get("max_no_improve_steps", 4),
        max_depth=nest.get("max_depth", 2),
        max_spawn_calls=nest.get("max_spawn_calls", 1),
        spawn_config=spawn,
        train_pool=trainset,
        test_set=testset,
        train_sample_size=data.get("train_sample_size", 20),
    )
    results = orch.run()

    print(f"\n{'=' * 60}")
    print(f"Baseline {metric.upper()}:  {results['baseline_score']:.2f}")
    print(f"Final {metric.upper()}:     {results['final_score']:.2f}")
    print(
        f"Improvement:               "
        f"{results['final_score'] - results['baseline_score']:+.2f}"
    )
    print(f"Total rounds:              {results['total_rounds']}")
    print(f"{'=' * 60}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
