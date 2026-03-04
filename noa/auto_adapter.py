"""Auto Adapter — LLM 读取目标源码，自动生成 __call__ 适配层 + target_factory。"""

from __future__ import annotations

import importlib.util
import os
import sys

from utils.llm import llm_call, resolve_model
from noa.core import prompts

_ADAPTER_FILENAME = "_noa_adapter.py"
_MAX_RETRIES = 3


def auto_adapt(
    source_dir: str,
    entry_hint: str = "",
    model: str = resolve_model("gpt-4.1-mini"),
    force: bool = False,
) -> tuple[object, callable]:
    """扫描 source_dir 源码，用 LLM 生成 Adapter 类。

    Returns:
        (adapter_instance, target_factory) — adapter 实现 __call__，
        target_factory(dir) 从任意目录加载并返回新 adapter。
    """
    source_dir = os.path.abspath(source_dir)
    adapter_path = os.path.join(source_dir, _ADAPTER_FILENAME)

    # 缓存命中
    if not force and os.path.exists(adapter_path):
        print(f"[AutoAdapter] 加载已有 adapter: {adapter_path}")
        adapter = _load_adapter(adapter_path, source_dir)
        factory = _make_factory(adapter_path, source_dir)
        return adapter, factory

    # 扫描源码
    source_code = _collect_source(source_dir)
    if not source_code:
        raise FileNotFoundError(f"在 {source_dir} 下未找到 .py 文件")

    # LLM 生成 + 验证重试
    hint_section = f"## Entry Hint\n{entry_hint}" if entry_hint else ""
    user_prompt = prompts.AUTO_ADAPTER_PROMPT.format(
        source_code=source_code,
        entry_hint=hint_section,
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        print(f"[AutoAdapter] 生成 adapter（第 {attempt} 次）...")
        resp = llm_call(
            user_prompt,
            model=model,
            max_tokens=16384,
            temperature=0.2,
            system=prompts.AUTO_ADAPTER_SYSTEM,
        )
        code = _clean_code(resp.text)

        try:
            adapter = _exec_adapter(code, source_dir)
            _validate(adapter)
            with open(adapter_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[AutoAdapter] 已保存 adapter -> {adapter_path}")
            factory = _make_factory(adapter_path, source_dir)
            return adapter, factory
        except Exception as e:
            print(f"[AutoAdapter] 验证失败: {e}")
            if attempt < _MAX_RETRIES:
                user_prompt += (
                    f"\n\n## Previous Error\n{e}\nFix the code and try again."
                )
            else:
                raise RuntimeError(
                    f"auto_adapt 在 {_MAX_RETRIES} 次尝试后仍失败: {e}"
                ) from e


def make_target_factory(source_dir: str) -> callable:
    """为已有 adapter 创建 target_factory。

    target_factory(dir) 从指定目录用 importlib 全新加载 adapter。
    """
    adapter_path = os.path.join(os.path.abspath(source_dir), _ADAPTER_FILENAME)
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"未找到 adapter 文件: {adapter_path}")
    return _make_factory(adapter_path, source_dir)


def _make_factory(adapter_path: str, original_source_dir: str) -> callable:
    """创建 target_factory 闭包。"""
    adapter_filename = os.path.basename(adapter_path)
    original_source_dir = os.path.abspath(original_source_dir)

    def target_factory(target_dir: str) -> object:
        """从 target_dir 加载 adapter，全新实例化。"""
        target_dir = os.path.abspath(target_dir)
        fpath = os.path.join(target_dir, adapter_filename)
        if not os.path.exists(fpath):
            import shutil

            shutil.copy2(adapter_path, fpath)

        # 清除目标系统模块缓存 + 重定向包路径，确保从 target_dir 加载
        saved = _isolate_target_modules(original_source_dir, target_dir)

        try:
            # 确保 target_dir 和父目录在 sys.path 最前面
            parent = os.path.dirname(target_dir)
            for p in [target_dir, parent]:
                if p in sys.path:
                    sys.path.remove(p)
                sys.path.insert(0, p)

            # 重新建立包上下文和子模块别名（_isolate 会清除之前的别名）
            aliases = _ensure_package_imported(target_dir)

            spec = importlib.util.spec_from_file_location("_noa_adapter", fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # 别名仅在 exec_module 期间需要

            # 立即清理别名，避免污染后续 import（如 sentence_transformers）
            for alias in aliases:
                sys.modules.pop(alias, None)

            cls = getattr(mod, "AutoAdapter", None)
            if cls is None:
                raise ValueError(f"adapter 文件 {fpath} 中未定义 AutoAdapter")
            return cls()
        finally:
            # 恢复包路径（已加载的对象引用不受影响）
            _restore_target_modules(saved)

    return target_factory


def _isolate_target_modules(source_dir: str, new_dir: str) -> dict:
    """清除目标系统模块缓存，重定向父包 __path__ 使新目录优先。"""
    source_dir = os.path.abspath(source_dir)
    new_dir = os.path.abspath(new_dir)
    source_parent = os.path.dirname(source_dir)
    new_parent = os.path.dirname(new_dir)

    saved = {"removed": {}, "redirects": {}}

    # 1. 清除 source_dir 和 temp 目录下的模块
    for name in list(sys.modules):
        mod = sys.modules[name]
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        abs_file = os.path.abspath(mod_file)
        if abs_file.startswith(source_dir + os.sep) or "noa_eval_" in abs_file:
            saved["removed"][name] = sys.modules.pop(name)

    # 2. 重定向父包 __path__（使 new_dir 的父目录优先）
    for name, mod in sys.modules.items():
        mod_path = getattr(mod, "__path__", None)
        if not mod_path:
            continue
        try:
            path_list = list(mod_path)
        except TypeError:
            continue
        for p in path_list:
            if os.path.abspath(p) == source_parent:
                saved["redirects"][name] = path_list
                mod.__path__ = [new_parent] + path_list
                break

    importlib.invalidate_caches()
    return saved


def _restore_target_modules(saved: dict):
    """恢复包路径。"""
    for name, original_path in saved.get("redirects", {}).items():
        mod = sys.modules.get(name)
        if mod and hasattr(mod, "__path__"):
            mod.__path__ = original_path


def _collect_source(source_dir: str) -> str:
    """收集目录下所有 .py 文件内容（排除 _noa_adapter.py）。"""
    parts = []
    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            if not fname.endswith(".py") or fname == _ADAPTER_FILENAME:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, source_dir)
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            parts.append(f"# === {rel} ===\n{content}")
    return "\n\n".join(parts)


def _clean_code(text: str) -> str:
    """去除 LLM 输出中可能包裹的 markdown 代码块。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _exec_adapter(code: str, source_dir: str) -> object:
    """动态执行生成的代码，返回 AutoAdapter 实例。"""
    source_dir = os.path.abspath(source_dir)
    parent = os.path.dirname(source_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    # 确保 source_dir 作为 Python 包被正确导入，并创建顶级别名
    aliases = _ensure_package_imported(source_dir)

    # 注入 __file__ 使 adapter 可用 os.path.dirname(__file__) 定位自身目录
    adapter_file = os.path.join(source_dir, _ADAPTER_FILENAME)
    namespace: dict = {"__file__": adapter_file}
    exec(code, namespace)  # 别名仅在 exec 期间需要（解析 from pipeline import 等）

    # exec 完成后立即清理别名，避免污染后续 import（如 sentence_transformers）
    for alias in aliases:
        sys.modules.pop(alias, None)

    adapter_cls = namespace.get("AutoAdapter")
    if adapter_cls is None:
        raise ValueError("生成的代码中未定义 AutoAdapter 类")
    return adapter_cls()


def _ensure_package_imported(source_dir: str):
    """确保 source_dir 作为 Python 包被导入，并为子模块创建顶级别名。

    例如 source_dir = project_root/target_systems/hotpotqa_rag:
    - 导入 target_systems.hotpotqa_rag 包
    - 创建别名: sys.modules['pipeline'] -> sys.modules['target_systems.hotpotqa_rag.pipeline']
    这样 adapter 中 `from pipeline import RAGPipeline` 能正确解析相对导入。
    """
    source_dir = os.path.abspath(source_dir)

    # 向上查找有 __init__.py 的包层级
    parts = []
    current = source_dir
    while os.path.exists(os.path.join(current, "__init__.py")):
        parts.append(os.path.basename(current))
        current = os.path.dirname(current)

    if not parts:
        return []

    # current 是包根的父目录，需在 sys.path 中
    if current not in sys.path:
        sys.path.insert(0, current)

    # 逐级导入包
    parts.reverse()
    pkg_name = ".".join(parts)
    try:
        __import__(pkg_name)
    except ImportError:
        return []

    # 导入包内所有子模块（使相对导入在包上下文中正确解析）
    for fname in os.listdir(source_dir):
        if (
            fname.endswith(".py")
            and fname != "__init__.py"
            and fname != _ADAPTER_FILENAME
        ):
            mod_name = fname[:-3]
            full_name = f"{pkg_name}.{mod_name}"
            if full_name not in sys.modules:
                try:
                    __import__(full_name)
                except ImportError:
                    pass

    # 为包内模块创建顶级别名（使 `from pipeline import X` 等短名可用）
    # 返回创建的别名列表，调用方加载完后应清理以避免污染 sys.modules
    aliases_created = []
    prefix = pkg_name + "."
    for name, mod in list(sys.modules.items()):
        if name.startswith(prefix):
            short = name[len(prefix) :]
            if "." not in short and short not in sys.modules:
                sys.modules[short] = mod
                aliases_created.append(short)
    return aliases_created


def _load_adapter(adapter_path: str, source_dir: str) -> object:
    """从缓存的 adapter 文件加载 AutoAdapter 实例。"""
    with open(adapter_path, encoding="utf-8") as f:
        code = f.read()
    return _exec_adapter(code, source_dir)


def _validate(adapter: object) -> None:
    """验证 adapter 实现了 __call__ 接口。get_components 为可选增强。"""
    if not callable(adapter):
        raise TypeError("adapter 不可调用，缺少 __call__ 方法")
    if hasattr(adapter, "get_components"):
        comps = adapter.get_components()
        if not isinstance(comps, dict):
            print("[AutoAdapter] 警告: get_components() 未返回 dict，忽略")
        else:
            print(
                f"[AutoAdapter] get_components() 返回 {len(comps)} 个组件: {list(comps.keys())}"
            )
