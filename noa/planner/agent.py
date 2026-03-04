"""Planner LLM agent for selecting next action (支持多轮 tool-calling)."""

from __future__ import annotations

import json
import logging

from utils.llm import llm_call, llm_call_with_tools, resolve_model
from noa.planner import prompts
from noa.planner.protocol import PlannerDecision, PlannerState

log = logging.getLogger(__name__)


class _PlannerProbe:
    """Planner 多轮模式的工具集 — 查询状态细节辅助决策。"""

    def __init__(self, state: PlannerState):
        self.state = state

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_history_detail",
                    "description": "Get detailed info about a specific step in history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "step_index": {
                                "type": "integer",
                                "description": "Index in history (negative for recent)",
                            },
                        },
                        "required": ["step_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_diagnosis_detail",
                    "description": "Get current diagnosis failure patterns detail.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_budget_status",
                    "description": "Get detailed budget usage status.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "get_history_detail":
            idx = int(arguments.get("step_index", -1))
            history = self.state.history
            if not history:
                return json.dumps({"info": "No history yet"})
            if -len(history) <= idx < len(history):
                return json.dumps(history[idx], ensure_ascii=False, default=str)[:3000]
            return json.dumps(
                {"error": f"Index {idx} out of range, history has {len(history)} items"}
            )
        elif tool_name == "get_diagnosis_detail":
            if self.state.diagnosis is None:
                return json.dumps({"info": "No diagnosis available"})
            patterns = self.state.diagnosis.failure_patterns[:5]
            return json.dumps(
                {"patterns": patterns, "summary": self.state.diagnosis.summary},
                ensure_ascii=False,
            )[:3000]
        elif tool_name == "get_budget_status":
            return json.dumps(self.state.budget.to_summary(), ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


class PlannerAgent:
    def __init__(
        self, model: str = resolve_model("gpt-4.1-mini"), max_tool_calls: int = 0
    ):
        self.model = model
        self.max_tool_calls = max_tool_calls

    def decide(self, state: PlannerState) -> PlannerDecision:
        prompt_text = prompts.PLANNER_PROMPT.format(
            layer=state.layer,
            state_summary=json.dumps(state.summary(), ensure_ascii=False, indent=2),
            recent_history=json.dumps(state.history[-5:], ensure_ascii=False, indent=2)
            if state.history
            else "(none)",
            layer_context=state.layer_context.to_prompt_context()
            if state.layer_context
            else "",
        )

        if self.max_tool_calls <= 0:
            return self._decide_single(prompt_text)

        return self._decide_agentic(prompt_text, state)

    def _decide_single(self, prompt: str) -> PlannerDecision:
        """原有单轮决策逻辑。"""
        tries = 0
        last_text = ""
        while tries < 3:
            tries += 1
            resp = llm_call(
                prompt,
                model=self.model,
                max_tokens=512,
                temperature=0,
                system=prompts.PLANNER_SYSTEM,
            )
            last_text = resp.text or ""
            parsed = _parse_json_like(last_text)
            if isinstance(parsed, dict):
                action = str(parsed.get("action", "")).strip()
                if action:
                    return PlannerDecision(
                        action=action,  # type: ignore[arg-type]
                        params=parsed.get("params") or {},
                        reason=str(parsed.get("reason", ""))[:500],
                        expected_gain=float(parsed.get("expected_gain", 0.0) or 0.0),
                        risk=str(parsed.get("risk", ""))[:500],
                    )
            prompt += "\n\nPrevious output was invalid JSON. Output strict JSON only."

        from noa.planner.guardrails import fallback_action

        return PlannerDecision(
            action=_heuristic_action_from_text(last_text)
            or fallback_action(
                type(
                    "S",
                    (),
                    {"trajectories": [], "diagnosis": None, "candidate_patch": None},
                )()
            ),
            params={},
            reason=f"fallback after invalid outputs: {last_text[:200]}",
        )

    def _decide_agentic(self, prompt: str, state: PlannerState) -> PlannerDecision:
        """多轮 tool-calling 决策。"""
        probe = _PlannerProbe(state)
        messages = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        tools = probe.get_tool_schemas()
        calls_remaining = self.max_tool_calls

        while calls_remaining > 0:
            resp = llm_call_with_tools(
                messages, tools, model=self.model, max_tokens=512, temperature=0
            )

            if resp.tool_calls:
                tc_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.text or "",
                        "tool_calls": tc_dicts,
                    }
                )
                for tc in resp.tool_calls:
                    result_text = probe.execute_tool(tc.name, tc.arguments)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                    )
                    calls_remaining -= 1
                    if calls_remaining <= 0:
                        break
            else:
                decision = _try_parse_decision(resp.text or "")
                if decision is not None:
                    return decision
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Output strict JSON only with action, params, reason, expected_gain, risk.",
                    }
                )
                calls_remaining -= 1

        # 预算耗尽，强制结论
        messages.append(
            {
                "role": "user",
                "content": "Tool budget exhausted. Output your decision as strict JSON now.",
            }
        )
        resp = llm_call_with_tools(
            messages, tools=[], model=self.model, max_tokens=512, temperature=0
        )
        decision = _try_parse_decision(resp.text or "")
        if decision is not None:
            return decision

        return PlannerDecision(
            action=_heuristic_action(state),  # type: ignore[arg-type]
            params={},
            reason="fallback after agentic planner exhaustion",
        )


def _try_parse_decision(text: str) -> PlannerDecision | None:
    parsed = _parse_json_like(text)
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "")).strip()
        if action:
            return PlannerDecision(
                action=action,  # type: ignore[arg-type]
                params=parsed.get("params") or {},
                reason=str(parsed.get("reason", ""))[:500],
                expected_gain=float(parsed.get("expected_gain", 0.0) or 0.0),
                risk=str(parsed.get("risk", ""))[:500],
            )
    return None


def _heuristic_action(state: PlannerState) -> str:
    if not state.trajectories:
        return "observe"
    if state.diagnosis is None:
        return "analyze"
    if state.candidate_patch is None or not state.candidate_patch.diffs:
        return "propose_patch"
    # 优先 spawn_sublayer 而非 stop
    if (
        state.no_improve_steps >= 2
        and state.layer_context is not None
        and state.layer_context.can_spawn_sublayer()
    ):
        return "spawn_sublayer"
    if state.no_improve_steps >= state.budget.max_no_improve_steps:
        return "stop"
    return "evaluate_patch"


def _heuristic_action_from_text(text: str) -> str | None:
    """从无效 JSON 文本中尝试提取 action 名。"""
    for action in (
        "observe",
        "analyze",
        "propose_patch",
        "evaluate_patch",
        "spawn_sublayer",
        "stop",
    ):
        if action in text.lower():
            return action
    return None


def _parse_json_like(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left : right + 1])
        except Exception:
            return None
    return None
