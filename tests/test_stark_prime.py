import json
import sys
import types

fake_llm = types.ModuleType("utils.llm")
fake_llm.resolve_model = lambda model: model
fake_llm.llm_call = lambda *args, **kwargs: None
sys.modules.setdefault("utils.llm", fake_llm)

from target_systems.stark_prime.components import Aggregator  # noqa: E402
from target_systems.stark_prime.config import AggregatorConfig, SystemConfig  # noqa: E402
from target_systems.stark_prime.evaluate import hit_at_1, mrr  # noqa: E402
from target_systems.stark_prime.pipeline import StarkPrimePipeline  # noqa: E402


def test_aggregator_matches_upstream_weighted_sum():
    agg = Aggregator(AggregatorConfig(relation_weight=1.0, text_weight=0.1))
    out = agg.forward(
        question="q",
        emb_scores=[0.5, 0.4, 0.3, 0.2, 0.1],
        relation_scores='{"1": 1, "2": 0.5, "3": 0, "4": 0, "5": 0}',
        text_scores='{"1": 0.5, "2": 1, "3": 0, "4": 0, "5": 0}',
    )
    assert out["final_scores"] == [1.55, 1.0, 0.3, 0.2, 0.1]


def test_mrr_and_hit_at_1():
    candidate_ids = [10, 11, 12, 13, 14]
    final_scores = [0.1, 0.2, 0.9, 0.3, 0.4]
    answer_ids = [13]

    assert mrr(candidate_ids, final_scores, answer_ids) == 1.0 / 3.0
    assert hit_at_1(candidate_ids, final_scores, answer_ids) == 0.0


def test_pipeline_serializes_final_scores_without_llm_calls(monkeypatch):
    cfg = SystemConfig()
    pipe = StarkPrimePipeline(cfg)

    monkeypatch.setattr(
        pipe.components[0][1],
        "forward",
        lambda **_: {"relation_scores": '{"1": 0, "2": 0, "3": 0, "4": 0, "5": 1}'},
    )
    monkeypatch.setattr(
        pipe.components[1][1],
        "forward",
        lambda **_: {"text_scores": '{"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}'},
    )

    result = pipe(
        question="q",
        relation_info="[]",
        text_info="[]",
        emb_scores=[0, 0, 0, 0, 0],
        candidate_ids=[1, 2, 3, 4, 5],
    )
    assert json.loads(result.answer) == [0.0, 0.0, 0.0, 0.0, 0.1]
    assert result.intermediate["aggregator"]["final_scores"][-1] == 0.1
