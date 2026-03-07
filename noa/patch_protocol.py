"""结构化 patch 操作 — apply_patch_ops / validate_patch_ops。"""

from __future__ import annotations

import os

from noa.core.protocol import PatchOp, PatchValidationError, SourceFile


def validate_patch_ops(
    source_files: list[SourceFile], ops: list[PatchOp]
) -> list[PatchValidationError]:
    """干运行验证，返回结构化错误列表（空=通过）。"""
    file_map = {sf.path: sf.content for sf in source_files}
    errors: list[PatchValidationError] = []

    for i, op in enumerate(ops):
        # 路径安全检查
        normalized = os.path.normpath(op.file_path)
        if ".." in normalized.split(os.sep) or normalized.startswith(os.sep):
            errors.append(
                PatchValidationError(
                    op_index=i,
                    code="path_traversal",
                    file_path=op.file_path,
                    message=f"非法路径: {op.file_path}",
                )
            )
            continue

        if op.op == "create":
            if normalized in file_map:
                errors.append(
                    PatchValidationError(
                        op_index=i,
                        code="file_already_exists",
                        file_path=op.file_path,
                        message=f"文件已存在: {op.file_path}",
                    )
                )
            continue

        if op.op == "delete":
            if normalized not in file_map:
                errors.append(
                    PatchValidationError(
                        op_index=i,
                        code="file_not_found",
                        file_path=op.file_path,
                        message=f"文件不存在: {op.file_path}",
                    )
                )
            continue

        # update / insert_after / insert_before 都需要文件存在
        content = file_map.get(normalized)
        if content is None:
            errors.append(
                PatchValidationError(
                    op_index=i,
                    code="file_not_found",
                    file_path=op.file_path,
                    message=f"文件不存在: {op.file_path}",
                )
            )
            continue

        if op.op == "update":
            err = _validate_search(i, op, content)
            if err:
                errors.append(err)

        elif op.op in ("insert_after", "insert_before"):
            err = _validate_anchor(i, op, content)
            if err:
                errors.append(err)

    return errors


def apply_patch_ops(
    source_files: list[SourceFile], ops: list[PatchOp]
) -> tuple[list[SourceFile], list[PatchValidationError], set[str]]:
    """在内存中应用 ops，返回 (modified_files, errors, deleted_paths)。

    事务性：任一 op 失败时全部不做，modified_files 为空。
    """
    errors = validate_patch_ops(source_files, ops)
    if errors:
        return [], errors, set()

    file_map = {sf.path: sf.content for sf in source_files}
    modified: set[str] = set()

    for i, op in enumerate(ops):
        normalized = os.path.normpath(op.file_path)

        if op.op == "create":
            file_map[normalized] = op.content
            modified.add(normalized)

        elif op.op == "delete":
            file_map.pop(normalized, None)
            modified.add(normalized)

        elif op.op == "update":
            content = file_map[normalized]
            new_content = _apply_update(op, content)
            if new_content is None:
                return (
                    [],
                    [
                        PatchValidationError(
                            op_index=i,
                            code="search_not_found",
                            file_path=op.file_path,
                            message="运行时 search 未匹配（validate 通过但 apply 失败）",
                        )
                    ],
                    set(),
                )
            file_map[normalized] = new_content
            modified.add(normalized)

        elif op.op == "insert_after":
            content = file_map[normalized]
            new_content = _apply_insert(op, content, after=True)
            if new_content is None:
                return (
                    [],
                    [
                        PatchValidationError(
                            op_index=i,
                            code="anchor_not_found",
                            file_path=op.file_path,
                            message="运行时 anchor 未匹配",
                        )
                    ],
                    set(),
                )
            file_map[normalized] = new_content
            modified.add(normalized)

        elif op.op == "insert_before":
            content = file_map[normalized]
            new_content = _apply_insert(op, content, after=False)
            if new_content is None:
                return (
                    [],
                    [
                        PatchValidationError(
                            op_index=i,
                            code="anchor_not_found",
                            file_path=op.file_path,
                            message="运行时 anchor 未匹配",
                        )
                    ],
                    set(),
                )
            file_map[normalized] = new_content
            modified.add(normalized)

    # 只返回被修改的文件（delete 的文件不返回）
    deleted_paths = {p for p in modified if p not in file_map}
    result = []
    for path in modified:
        if path in file_map:
            result.append(SourceFile(path=path, content=file_map[path]))
    return result, [], deleted_paths


# --- 内部辅助 ---


def _find_rstrip_occurrences(content: str, needle: str) -> list[int]:
    """rstrip 后行级匹配，返回每个匹配处原始内容的字符偏移。"""
    content_lines = content.split("\n")
    needle_lines = [ln.rstrip() for ln in needle.split("\n")]
    n = len(needle_lines)
    positions = []
    char_offset = 0
    for i in range(len(content_lines)):
        if (
            i + n <= len(content_lines)
            and [ln.rstrip() for ln in content_lines[i : i + n]] == needle_lines
        ):
            positions.append(char_offset)
        char_offset += len(content_lines[i]) + 1
    return positions


def _matched_span_length(content: str, pos: int, search: str) -> int:
    """精确匹配时返回 len(search)；fuzzy 时返回原始内容中对应行的实际字符数。"""
    if content[pos : pos + len(search)] == search:
        return len(search)
    # fuzzy 匹配：按实际可见行数计算 span（尾部换行不算额外行）
    n_lines = search.rstrip("\n").count("\n") + 1
    end = pos
    for _ in range(n_lines):
        next_nl = content.find("\n", end)
        if next_nl == -1:
            end = len(content)
            break
        end = next_nl + 1
    # 最后一行不含尾部换行
    if search and not search.endswith("\n") and end > pos and content[end - 1] == "\n":
        end -= 1
    return end - pos


def _find_occurrences(content: str, needle: str) -> list[int]:
    """查找 needle 在 content 中所有出现的起始位置。"""
    positions = []
    start = 0
    while True:
        idx = content.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _disambiguate(
    content: str, needle: str, positions: list[int], op: PatchOp
) -> int | None:
    """用 context_before/context_after 消歧，返回匹配的位置索引，None 表示无法消歧。"""
    if not op.context_before and not op.context_after:
        return None
    for pos in positions:
        if op.context_before:
            before_start = max(0, pos - len(op.context_before) - 50)
            before_text = content[before_start:pos]
            if op.context_before not in before_text:
                continue
        if op.context_after:
            after_end = min(
                len(content), pos + len(needle) + len(op.context_after) + 50
            )
            after_text = content[pos + len(needle) : after_end]
            if op.context_after not in after_text:
                continue
        return pos
    return None


def _validate_search(i: int, op: PatchOp, content: str) -> PatchValidationError | None:
    """验证 update op 的 search 是否能在 content 中正确匹配。"""
    if not op.search:
        return PatchValidationError(
            op_index=i,
            code="search_not_found",
            file_path=op.file_path,
            message="search 为空",
        )
    positions = _find_occurrences(content, op.search)
    if not positions:
        # fuzzy fallback: rstrip 行级匹配
        positions = _find_rstrip_occurrences(content, op.search)
        if not positions:
            return PatchValidationError(
                op_index=i,
                code="search_not_found",
                file_path=op.file_path,
                message="search 文本未在文件中找到",
            )
    if op.must_be_unique and len(positions) > 1:
        # 尝试 context_before/after 消歧
        disamb = _disambiguate(content, op.search, positions, op)
        if disamb is None:
            return PatchValidationError(
                op_index=i,
                code="search_not_unique",
                file_path=op.file_path,
                message=f"search 在文件中出现 {len(positions)} 次，需唯一或提供 context 消歧",
            )
    elif not op.must_be_unique and op.occurrence > len(positions):
        return PatchValidationError(
            op_index=i,
            code="search_not_found",
            file_path=op.file_path,
            message=f"search 仅出现 {len(positions)} 次，但 occurrence={op.occurrence}",
        )
    return None


def _validate_anchor(i: int, op: PatchOp, content: str) -> PatchValidationError | None:
    """验证 insert_after/insert_before op 的 anchor 是否能正确匹配。"""
    if not op.anchor:
        return PatchValidationError(
            op_index=i,
            code="anchor_not_found",
            file_path=op.file_path,
            message="anchor 为空",
        )
    positions = _find_occurrences(content, op.anchor)
    if not positions:
        return PatchValidationError(
            op_index=i,
            code="anchor_not_found",
            file_path=op.file_path,
            message="anchor 文本未在文件中找到",
        )
    if op.anchor_must_be_unique and len(positions) > 1:
        return PatchValidationError(
            op_index=i,
            code="anchor_not_unique",
            file_path=op.file_path,
            message=f"anchor 在文件中出现 {len(positions)} 次，需唯一",
        )
    elif not op.anchor_must_be_unique and op.anchor_occurrence > len(positions):
        return PatchValidationError(
            op_index=i,
            code="anchor_not_found",
            file_path=op.file_path,
            message=f"anchor 仅出现 {len(positions)} 次，但 anchor_occurrence={op.anchor_occurrence}",
        )
    return None


def _resolve_position(
    content: str,
    needle: str,
    op_unique: bool,
    occurrence: int,
    context_before: str,
    context_after: str,
) -> int | None:
    """解析匹配位置，返回起始 index 或 None。"""
    positions = _find_occurrences(content, needle)
    if not positions:
        positions = _find_rstrip_occurrences(content, needle)
    if not positions:
        return None

    if op_unique and len(positions) == 1:
        return positions[0]

    if op_unique and len(positions) > 1:
        # 尝试 context 消歧
        if context_before or context_after:
            for pos in positions:
                if context_before:
                    before_start = max(0, pos - len(context_before) - 50)
                    if context_before not in content[before_start:pos]:
                        continue
                if context_after:
                    after_end = min(
                        len(content), pos + len(needle) + len(context_after) + 50
                    )
                    if context_after not in content[pos + len(needle) : after_end]:
                        continue
                return pos
        return None

    # 非唯一模式，按 occurrence 选
    idx = occurrence - 1
    if 0 <= idx < len(positions):
        return positions[idx]
    return None


def _apply_update(op: PatchOp, content: str) -> str | None:
    """应用 update op，返回新 content 或 None。"""
    pos = _resolve_position(
        content,
        op.search,
        op.must_be_unique,
        op.occurrence,
        op.context_before,
        op.context_after,
    )
    if pos is None:
        return None
    actual_span = _matched_span_length(content, pos, op.search)
    return content[:pos] + op.replace + content[pos + actual_span :]


def _apply_insert(op: PatchOp, content: str, after: bool) -> str | None:
    """应用 insert_after / insert_before op，返回新 content 或 None。"""
    pos = _resolve_position(
        content,
        op.anchor,
        op.anchor_must_be_unique,
        op.anchor_occurrence,
        "",
        "",  # anchor 不使用 context 消歧
    )
    if pos is None:
        return None
    if after:
        insert_pos = pos + len(op.anchor)
    else:
        insert_pos = pos
    return content[:insert_pos] + op.new_lines + content[insert_pos:]
