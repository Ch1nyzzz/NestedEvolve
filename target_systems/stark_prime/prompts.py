"""STaRK-Prime 默认 Prompt 模板。

来源：Optimas 官方 `examples/systems/stark/bio_system.py` 中的 DSPy signature docstring。
"""

RELATION_SCORER = (
    "Given a question and a list of 5 entities with their relational information, "
    "assign each entity a relevance score (between 0 and 1) based on how well "
    "its relations match the information in the question."
)

TEXT_SCORER = (
    "Given a question and a list of 5 entities with their property information, "
    "assign each entity a relevance score between 0 and 1 based on how well "
    "its properties match the requirements described in the question."
)
