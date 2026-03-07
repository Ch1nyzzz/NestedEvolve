"""ComponentProbe — 包装目标系统，提供单组件执行能力。供 Analyzer 做微实验。"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from noa.core.protocol import SystemDescription


class ComponentProbe:
    """包装目标系统，提供单组件 / 部分管道执行能力。"""

    def __init__(self, adapter, sys_desc: SystemDescription, timeout: int = 30):
        self.workflow = sys_desc.component_names
        self.timeout = timeout
        self.components = self._extract_components(adapter)

    def _extract_components(self, adapter) -> dict[str, object]:
        """从 adapter 提取组件。优先 get_components()，fallback 到反射。"""
        if hasattr(adapter, "get_components"):
            return adapter.get_components()
        # fallback：尝试常见属性名
        for attr in ("pipeline", "_pipeline", "pipe"):
            obj = getattr(adapter, attr, None)
            if obj is None:
                continue
            if hasattr(obj, "components"):
                comps = obj.components
                if isinstance(comps, dict):
                    return comps
                # list of (name, comp) tuples
                if isinstance(comps, (list, tuple)):
                    return {name: comp for name, comp in comps}
            # 按 workflow 名称在 pipeline 上查找属性
            found = {}
            for name in self.workflow:
                comp = getattr(obj, name, None)
                if comp is not None and hasattr(comp, "forward"):
                    found[name] = comp
            if found:
                return found
        return {}

    def run_component(self, name: str, inputs: dict) -> dict:
        """运行单个组件，返回输出。线程安全的超时保护。"""
        if name not in self.components:
            return {
                "error": f"Component '{name}' not found. Available: {list(self.components.keys())}"
            }
        comp = self.components[name]
        if not isinstance(inputs, dict):
            inputs = {"input": inputs}
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(comp.forward, **inputs)
            try:
                result = future.result(timeout=self.timeout)
                return result if isinstance(result, dict) else {"output": result}
            except FuturesTimeoutError:
                return {"error": f"Component '{name}' timed out after {self.timeout}s"}
            except Exception:
                return {"error": traceback.format_exc()[-500:]}

    def run_from(self, start_component: str, inputs: dict) -> dict:
        """从指定组件开始运行后续管道，返回最终输出。"""
        if start_component not in self.workflow:
            return {
                "error": f"Component '{start_component}' not in workflow {self.workflow}"
            }
        start_idx = self.workflow.index(start_component)
        current = inputs if isinstance(inputs, dict) else {"input": inputs}
        for name in self.workflow[start_idx:]:
            if name not in self.components:
                return {
                    "error": f"Component '{name}' not available",
                    "partial": current,
                }
            result = self.run_component(
                name, current if isinstance(current, dict) else {"input": current}
            )
            if "error" in result:
                return {"error": result["error"], "failed_at": name, "partial": current}
            current.update(result)
        return current

    def reproduce(self, case_id: str, inputs: dict | None = None) -> dict:
        """用指定输入重现失败，case_id 为 question 或唯一标识。"""
        if inputs is None:
            inputs = {"question": case_id}
        # 依次运行所有组件，收集每步输出
        results = {}
        current = inputs if isinstance(inputs, dict) else {"input": inputs}
        for name in self.workflow:
            if name not in self.components:
                results[name] = {"skipped": True}
                continue
            step_result = self.run_component(name, current)
            results[name] = step_result
            if "error" in step_result:
                return {
                    "error": step_result["error"],
                    "failed_at": name,
                    "steps": results,
                }
            current.update(step_result)
        return {"steps": results, "final": current}

    def inspect_artifacts(self, pattern: str, base_dir: str | None = None) -> dict:
        """查看 eval 产生的 artifact 文件，按 pattern 匹配。"""
        import glob
        import os

        search_dir = base_dir or os.path.dirname(
            os.path.abspath(getattr(self, "_source_dir", ".") or ".")
        )
        matches = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
        artifacts = []
        for path in matches[:20]:
            try:
                size = os.path.getsize(path)
                rel = os.path.relpath(path, search_dir)
                preview = ""
                if size < 10000 and path.endswith((".json", ".jsonl", ".txt", ".log")):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        preview = f.read(2000)
                artifacts.append({"path": rel, "size": size, "preview": preview[:2000]})
            except Exception:
                continue
        return {"count": len(matches), "artifacts": artifacts}

    def get_tool_schemas(self) -> list[dict]:
        """返回 OpenAI function calling 格式的工具定义。"""
        comp_names = list(self.components.keys()) if self.components else self.workflow
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_component",
                    "description": f"Run a single component with given inputs. Available components: {comp_names}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "component_name": {
                                "type": "string",
                                "description": "Name of the component to run",
                                "enum": comp_names,
                            },
                            "inputs": {
                                "type": "object",
                                "description": "Input dict to pass as **kwargs to component.forward()",
                            },
                        },
                        "required": ["component_name", "inputs"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_from",
                    "description": f"Run the pipeline from a given component to the end. Workflow order: {self.workflow}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_component": {
                                "type": "string",
                                "description": "Component to start from",
                                "enum": comp_names,
                            },
                            "inputs": {
                                "type": "object",
                                "description": "Input dict for the start component",
                            },
                        },
                        "required": ["start_component", "inputs"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """分发执行工具调用，返回结果文本。"""
        if tool_name == "run_component":
            result = self.run_component(
                arguments["component_name"], arguments.get("inputs", {})
            )
        elif tool_name == "run_from":
            result = self.run_from(
                arguments["start_component"], arguments.get("inputs", {})
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        # 截断过长结果
        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > 3000:
            text = text[:3000] + "... (truncated)"
        return text
