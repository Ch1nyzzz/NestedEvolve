import sys
from dataclasses import dataclass, field

sys.path.insert(0, "")

from pipeline import PubMedQAPipeline


@dataclass
class AdapterResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class AutoAdapter:
    def __init__(self):
        self.pipeline = PubMedQAPipeline()

    def __call__(self, question: str, context: str = "") -> AdapterResult:
        result = self.pipeline(question=question, context=context)
        return AdapterResult(answer=result.answer, intermediate=result.intermediate)
