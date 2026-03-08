"""PubMedQA 默认 Prompt 模板。"""

SYSTEM_PROMPT = "You are a scientist."

CONTEXT_ANALYST = (
    "You are supposed to summarize the key information from the given context "
    "to answer the provided question."
)

PROBLEM_SOLVER = (
    "You are supposed to provide a solution to a given problem "
    "based on the provided summary."
)

FORMAT_YESNO = (
    "Always conclude the last line of your response should be of the following format: "
    "'Answer: $VALUE' (without quotes) where VALUE is either 'yes' or 'no' or 'maybe'."
)
