import sys
from dataclasses import dataclass, field

sys.path.insert(0, "")

from pipeline import STaRKPrimePipeline


@dataclass
class AdapterResult:
    answer: str
    prediction: dict = field(default_factory=dict)
    intermediate: dict = field(default_factory=dict)


class AutoAdapter:
    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = STaRKPrimePipeline()
        return self._pipeline

    def __call__(self, question: str, query_id: int | None = None) -> AdapterResult:
        result = self.pipeline(question=question, query_id=query_id)
        return AdapterResult(
            answer=result.answer,
            prediction=result.prediction,
            intermediate=result.intermediate,
        )

    def get_components(self):
        return {name: comp for name, comp in self.pipeline.components}
