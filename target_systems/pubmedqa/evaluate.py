"""Exact Match 评估指标 — PubMedQA yes/no/maybe 三分类。"""

import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm


DEFAULT_MAX_WORKERS = 25
DEFAULT_GLOBAL_MAX_WORKERS = 50
_GLOBAL_SLOT_DIR = Path(tempfile.gettempdir()) / "noa_pubmedqa_eval_slots"


def extract_answer_yesno(text: str) -> str:
    """从模型回复中提取 yes/no/maybe 答案。"""
    pattern = r"(?i)\b(yes|no|maybe)\b"
    match = re.search(pattern, text)
    return match.group(1).lower() if match else text.strip().lower()


def exact_match(prediction: str, ground_truth: str) -> float:
    """精确匹配 — 提取 yes/no/maybe 后比较。"""
    pred = extract_answer_yesno(prediction)
    gt = ground_truth.strip().lower()
    return 1.0 if pred == gt else 0.0


def _worker_limits(requested_workers: int) -> tuple[int, int, int]:
    per_eval_limit = max(1, int(os.getenv("NOA_EVAL_MAX_WORKERS", DEFAULT_MAX_WORKERS)))
    global_limit = max(
        per_eval_limit,
        int(os.getenv("NOA_GLOBAL_EVAL_MAX_WORKERS", DEFAULT_GLOBAL_MAX_WORKERS)),
    )
    resolved_workers = max(1, min(requested_workers, per_eval_limit, global_limit))
    return resolved_workers, per_eval_limit, global_limit


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_slot(slot_path: Path) -> bool:
    try:
        owner = slot_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    pid_text = owner.split(":", 1)[0]
    if not pid_text.isdigit() or _pid_alive(int(pid_text)):
        return False
    try:
        slot_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def _acquire_global_eval_slot(global_limit: int):
    _GLOBAL_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    owner = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
    start_idx = (os.getpid() + threading.get_ident()) % global_limit
    slot_path = None

    while slot_path is None:
        for offset in range(global_limit):
            idx = (start_idx + offset) % global_limit
            candidate = _GLOBAL_SLOT_DIR / f"slot-{idx:03d}.lock"
            try:
                fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                _cleanup_stale_slot(candidate)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(owner)
            slot_path = candidate
            break
        if slot_path is None:
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            slot_path.unlink()
        except FileNotFoundError:
            pass


def evaluate_batch(
    pipeline,
    dataset: list,
    max_workers: int = 25,
) -> dict:
    """并行评估 pipeline，返回 accuracy + per_example 详情。"""
    if not dataset:
        return {"score": 0.0, "details": []}

    resolved_workers, _, global_limit = _worker_limits(max_workers)

    def _eval_one(example):
        with _acquire_global_eval_slot(global_limit):
            result = pipeline(question=example.question, context=example.context)
        score = exact_match(result.answer, example.answer)
        return {
            "id": example.id,
            "question": example.question,
            "ground_truth": example.answer,
            "prediction": result.answer,
            "extracted": extract_answer_yesno(result.answer),
            "accuracy": score,
        }

    details = []
    with ThreadPoolExecutor(max_workers=min(resolved_workers, len(dataset))) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in dataset}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating"
        ):
            details.append(future.result())

    accuracy = sum(d["accuracy"] for d in details) / len(details) if details else 0.0
    return {"score": round(accuracy * 100, 2), "details": details}
