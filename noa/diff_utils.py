"""Diff 工具 — SEARCH/REPLACE 解析、内存应用、临时目录管理。"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

from noa.core.protocol import SourceFile, DiffBlock
from noa.patch_protocol import apply_patch_ops, validate_patch_ops  # noqa: F401 — re-export


def _normalize_diff_path(path: str) -> str | None:
    """规范化 diff 中的文件路径。拒绝含 ../ 的路径，去除 ./ 前缀。"""
    if ".." in path.split("/") or ".." in path.split(os.sep):
        return None
    # 去除 ./ 前缀和连续 /
    normalized = os.path.normpath(path)
    if normalized.startswith(os.sep):
        return None  # 拒绝绝对路径
    return normalized


def extract_diffs(text: str, source_files: list[SourceFile]) -> list[DiffBlock]:
    """从 LLM 输出解析 SEARCH/REPLACE 块。

    支持格式:
        ## File: prompts.py
        <<<<<<< SEARCH
        原始代码
        =======
        替换代码
        >>>>>>> REPLACE
    """
    diffs: list[DiffBlock] = []

    # 按 ## File: xxx 分段
    file_sections = re.split(r"^## File:\s*(.+)$", text, flags=re.MULTILINE)

    # file_sections: ['前导文本', 'filename1', 'content1', 'filename2', 'content2', ...]
    sections: list[tuple[str | None, str]] = []

    if len(file_sections) >= 3:
        # 有明确的 ## File: 标记
        for i in range(1, len(file_sections), 2):
            fname = file_sections[i].strip()
            content = file_sections[i + 1] if i + 1 < len(file_sections) else ""
            sections.append((fname, content))
    else:
        # 没有 ## File: 标记，整体解析
        sections.append((None, text))

    # 解析每段中的 SEARCH/REPLACE 块
    diff_pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
        re.DOTALL,
    )

    for file_path, section_text in sections:
        for match in diff_pattern.finditer(section_text):
            search = match.group(1).rstrip("\n")
            replace = match.group(2).rstrip("\n")

            resolved_path = file_path
            if resolved_path is None:
                resolved_path = _find_file_for_search(search, source_files)

            if resolved_path:
                normalized = _normalize_diff_path(resolved_path)
                if normalized is None:
                    print(f"[DiffUtils] 拒绝非法路径: {resolved_path}")
                    continue
                diffs.append(
                    DiffBlock(
                        file_path=normalized,
                        search=search,
                        replace=replace,
                    )
                )

    return diffs


def _find_file_for_search(search: str, source_files: list[SourceFile]) -> str | None:
    """在 source_files 中找到包含 search 文本的文件。"""
    for sf in source_files:
        if search in sf.content:
            return sf.path
    # 尝试行级匹配
    search_lines = search.split("\n")
    for sf in source_files:
        file_lines = sf.content.split("\n")
        for i in range(len(file_lines) - len(search_lines) + 1):
            if file_lines[i : i + len(search_lines)] == search_lines:
                return sf.path
    return None


def apply_diffs_in_memory(
    source_files: list[SourceFile],
    diffs: list[DiffBlock],
) -> list[SourceFile]:
    """在内存中应用 diff，返回修改后的文件列表。

    只返回被修改的文件。不修改原始 source_files。
    """
    # 按文件分组
    file_contents: dict[str, str] = {sf.path: sf.content for sf in source_files}
    modified: set[str] = set()

    for diff in diffs:
        if diff.file_path not in file_contents:
            print(f"[DiffUtils] 警告: 文件 {diff.file_path} 不在 source_files 中，跳过")
            continue

        content = file_contents[diff.file_path]
        # 先尝试直接字符串替换
        if diff.search in content:
            file_contents[diff.file_path] = content.replace(
                diff.search, diff.replace, 1
            )
            modified.add(diff.file_path)
            continue

        # 尝试行级匹配（容忍尾部空白差异）
        new_content, applied = _apply_linewise(content, diff.search, diff.replace)
        if applied:
            file_contents[diff.file_path] = new_content
            modified.add(diff.file_path)
        else:
            print(f"[DiffUtils] 警告: 在 {diff.file_path} 中未匹配到 SEARCH 块，跳过")

    return [SourceFile(path=p, content=file_contents[p]) for p in modified]


def _apply_linewise(content: str, search: str, replace: str) -> tuple[str, bool]:
    """行级匹配应用 diff。"""
    lines = content.split("\n")
    search_lines = search.split("\n")
    replace_lines = replace.split("\n")

    for i in range(len(lines) - len(search_lines) + 1):
        if lines[i : i + len(search_lines)] == search_lines:
            lines[i : i + len(search_lines)] = replace_lines
            return "\n".join(lines), True

    # 尝试 strip 后匹配
    stripped_search = [ln.rstrip() for ln in search_lines]
    for i in range(len(lines) - len(search_lines) + 1):
        if [ln.rstrip() for ln in lines[i : i + len(search_lines)]] == stripped_search:
            lines[i : i + len(search_lines)] = replace_lines
            return "\n".join(lines), True

    return content, False


def write_to_temp_dir(
    modified_files: list[SourceFile],
    source_dir: str,
) -> str:
    """将修改后的文件写入临时目录，保持完整目录结构。

    先复制整个 source_dir，再覆盖修改的文件。返回临时目录路径。
    """
    temp_dir = tempfile.mkdtemp(prefix="noa_eval_")

    # 复制整个源码目录
    target_dir = os.path.join(temp_dir, os.path.basename(source_dir))
    shutil.copytree(source_dir, target_dir)

    # 覆盖修改的文件
    for sf in modified_files:
        fpath = os.path.join(target_dir, sf.path)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(sf.content)

    return target_dir


def commit_to_source(
    modified_files: list[SourceFile],
    source_dir: str,
    layer_context=None,
) -> None:
    """接受 patch 后，将修改写回原始文件。有 layer_context 时检查写权限。"""
    for sf in modified_files:
        fpath = os.path.join(source_dir, sf.path)
        if layer_context is not None and not layer_context.check_write_permission(
            fpath
        ):
            raise PermissionError(f"Layer {layer_context.layer_id} 无权写入: {fpath}")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(sf.content)


def cleanup_temp_dir(temp_dir: str) -> None:
    """清理临时目录。"""
    parent = os.path.dirname(temp_dir)
    if parent and os.path.basename(parent).startswith("noa_eval_"):
        shutil.rmtree(parent, ignore_errors=True)
    elif os.path.basename(temp_dir).startswith("noa_eval_"):
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
