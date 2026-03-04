"""默认 Prompt 模板 — 来自 OPTIMAS DSPy Signature docstring。"""

QUESTION_REWRITER = """Rephrase the question to make it clearer and easier to answer.
If the question appears malformed, ambiguous, or logically inconsistent (e.g., asking about two identical items, missing context, or apparent typos), flag it with [AMBIGUOUS] prefix and suggest a clarification."""

INFO_EXTRACTOR = (
    "Extract 3-5 search keywords from the query. "
    "Output ONLY comma-separated entity names or core phrases. "
    "Do NOT add descriptions, categories, or labels (e.g. NO 'nationality', 'director', 'film genre'). "
    "No markdown, no headers, no bullet points, no numbering.\n"
    "If the query is marked [AMBIGUOUS], extract alternative keyword interpretations.\n"
    "Example: Albert Einstein, theory of relativity, Nobel Prize physics"
)

HINT_GENERATOR = (
    "Generate useful hints to answer the query based on the retrieved content."
)

ANSWER_GENERATOR = "Given some hints, directly answer the query with a short answer."
