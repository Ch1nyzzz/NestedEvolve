import sys
from dataclasses import dataclass, field

sys.path.insert(0, "")

from pipeline import RAGPipeline


@dataclass
class AdapterResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class AutoAdapter:
    def __init__(self):
        self.pipeline = RAGPipeline()

    def __call__(self, question: str) -> AdapterResult:
        result = self.pipeline(question)
        return AdapterResult(answer=result.answer, intermediate=result.intermediate)
