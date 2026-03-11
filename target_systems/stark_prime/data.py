"""STaRK-Prime 数据准备。

尽量对齐 Optimas 官方实现：
  - 数据定义参考 `examples/datasets/stark_prime.py`
  - 系统定义参考 `examples/systems/stark/bio_system.py`

实现说明：
  - 论文正文/附录与官方代码在 split / metric 上存在不一致。
  - 这里默认用 `split_mode="paper"`，优先贴近论文附录的原始 split；
    若需要严格复现官方仓库里用于示例运行的子集，可用 `split_mode="optimas_repo"`。
"""

from __future__ import annotations

import ast
import json
import os
import os.path as osp
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

EMBED_MODEL = "text-embedding-ada-002"
EMBED_QUERY_TOKEN = "1MshwJttPZsHEM2cKA5T13SIrsLeBEdyU"
EMBED_DOC_TOKEN = "16EJvCMbgkVrQ0BuIBvLBp-BYPaye-Edy"


@dataclass
class StarkPrimeExample:
    question: str
    answer_ids: list[int]
    candidate_ids: list[int]
    relation_info: str
    text_info: str
    emb_scores: list[float]
    id: str


def _require_module(import_name: str, package_name: str | None = None):
    try:
        return __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        pkg = package_name or import_name
        raise RuntimeError(
            f"Missing dependency '{pkg}'. "
            f"STaRK-Prime preprocessing follows the official STARK stack and "
            f"requires this package to be installed."
        ) from exc


def download_hf_folder(
    repo: str,
    folder: str,
    *,
    repo_type: str = "dataset",
    save_as_folder: str | None = None,
) -> str:
    files = list_repo_files(repo, repo_type=repo_type)
    folder_files = [f for f in files if f.startswith(folder + "/")]

    for file in folder_files:
        path = hf_hub_download(repo, file, repo_type=repo_type)
        if save_as_folder:
            local_path = os.path.join(save_as_folder, os.path.relpath(file, folder))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not osp.exists(local_path):
                shutil.copy2(path, local_path)
        else:
            save_as_folder = osp.dirname(osp.dirname(path))

    if save_as_folder is None:
        raise RuntimeError(f"Unable to download folder {folder} from {repo}")
    return save_as_folder


class STaRKDataset:
    def __init__(
        self,
        name: str,
        root: str,
        *,
        human_generated_eval: bool = False,
    ):
        self.name = name
        self.root = root
        self.human_generated_eval = human_generated_eval
        self.dataset_root = osp.join(root, name)
        self._download()

        query_dir = osp.join(self.dataset_root, "stark_qa")
        filename = (
            "stark_qa_human_generated_eval.csv"
            if human_generated_eval
            else "stark_qa.csv"
        )
        self.qa_csv_path = osp.join(query_dir, filename)

        self.data = pd.read_csv(self.qa_csv_path)
        self.indices = sorted(self.data["id"].tolist())
        self.split_dir = osp.join(self.dataset_root, "split")
        self.split_indices = self.get_idx_split()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = self.data[self.data["id"] == self.indices[idx]].iloc[0]
        answer_ids = ast.literal_eval(row["answer_ids"])
        return row["query"], row["id"], answer_ids, None

    def get_idx_split(self, test_ratio: float = 1.0) -> dict[str, torch.Tensor]:
        if self.human_generated_eval:
            return {"human_generated_eval": torch.LongTensor(self.indices)}

        split_idx = {}
        for split in ["train", "val", "test", "test-0.1"]:
            with open(
                osp.join(self.split_dir, f"{split}.index"), encoding="utf-8"
            ) as f:
                ids = [int(i) for i in f.read().split()]
            split_idx[split] = torch.LongTensor([self.indices.index(i) for i in ids])

        if test_ratio < 1.0:
            split_idx["test"] = split_idx["test"][
                : int(len(split_idx["test"]) * test_ratio)
            ]
        return split_idx

    def _download(self):
        self.dataset_root = download_hf_folder(
            repo="snap-stanford/stark",
            folder=f"qa/{self.name}",
            repo_type="dataset",
            save_as_folder=self.dataset_root,
        )


def _ensure_embeddings(root: str, dataset: str = "prime") -> tuple[str, str]:
    emb_dir = osp.join(root, "emb", dataset, EMBED_MODEL)
    query_emb_dir = osp.join(emb_dir, "query")
    node_emb_dir = osp.join(emb_dir, "doc")
    query_emb_path = osp.join(query_emb_dir, "query_emb_dict.pt")
    node_emb_path = osp.join(node_emb_dir, "candidate_emb_dict.pt")

    if osp.exists(query_emb_path) and osp.exists(node_emb_path):
        return query_emb_dir, node_emb_dir

    gdown = _require_module("gdown")
    os.makedirs(query_emb_dir, exist_ok=True)
    os.makedirs(node_emb_dir, exist_ok=True)

    query_url = f"https://drive.google.com/uc?id={EMBED_QUERY_TOKEN}"
    node_url = f"https://drive.google.com/uc?id={EMBED_DOC_TOKEN}"
    gdown.download(query_url, query_emb_path, quiet=False)
    gdown.download(node_url, node_emb_path, quiet=False)
    return query_emb_dir, node_emb_dir


def _get_vss_topk(vss, query: str, query_id: int) -> dict[int, float]:
    scores = vss(query, query_id)
    cleaned = {int(k): float(v) for k, v in scores.items()}
    return dict(sorted(cleaned.items(), key=lambda x: x[1], reverse=True))


def _filter_candidates(
    candidates: list[int],
    answer_ids: list[int],
    *,
    rng: random.Random,
    n_answers: int = 1,
) -> list[int]:
    answers = [cid for cid in candidates if cid in answer_ids]
    if len(answers) >= n_answers:
        selected = rng.sample(answers, n_answers)
    else:
        selected = [rng.choice(answer_ids)]

    remaining = [cid for cid in candidates if cid not in selected]
    non_selected = rng.sample(remaining, min(len(remaining), 5 - len(selected)))
    while len(selected) + len(non_selected) < 5:
        non_selected.append(rng.choice(candidates))
    result = selected + non_selected
    rng.shuffle(result)
    return result


def _truncate_infos(info_dict: dict[int, dict[str, Any]], limit: int = 1024):
    def truncate(text: str) -> str:
        return text[:limit] if len(text) > limit else text

    for value in info_dict.values():
        value["text_info"] = [truncate(t) for t in value["text_info"]]
        value["relation_info"] = [truncate(t) for t in value["relation_info"]]
    return info_dict


def _resolve_split_indices(
    qa_dataset: STaRKDataset,
    *,
    split_mode: str,
) -> dict[str, list[int]]:
    raw = qa_dataset.get_idx_split()
    if split_mode == "optimas_repo":
        return {
            "train": [int(i) for i in raw["train"][:250]],
            "val": [int(i) for i in raw["val"][:25]],
            "test": [int(i) for i in raw["test"][:100]],
        }

    # Paper appendix states 495 / 51 / 96. We approximate this with the
    # original train / val splits and prefer `test-0.1` when present.
    paper_test = raw.get("test-0.1", raw["test"])
    return {
        "train": [int(i) for i in raw["train"]],
        "val": [int(i) for i in raw["val"]],
        "test": [int(i) for i in paper_test],
    }


def _build_processed_examples(
    *,
    root: str,
    split_mode: str,
    seed: int = 42,
    max_workers: int = 16,
) -> dict[str, list[StarkPrimeExample]]:
    stark_qa = _require_module("stark_qa", "stark-qa")
    vss_mod = _require_module("stark_qa.models", "stark-qa")

    load_skb = getattr(stark_qa, "load_skb")
    VSS = getattr(vss_mod, "VSS")

    rng = random.Random(seed)
    qa_root = osp.join(root, "qa")
    processed_root = osp.join(root, "processed")
    os.makedirs(processed_root, exist_ok=True)

    qa_dataset = STaRKDataset("prime", root=qa_root)
    split_indices = _resolve_split_indices(qa_dataset, split_mode=split_mode)

    query_emb_dir, node_emb_dir = _ensure_embeddings(root, "prime")
    skb = load_skb("prime")

    def get_rel_info(node_id: int) -> str:
        return (
            f"- name: {skb[node_id].name}\n"
            f"- type: {skb[node_id].type}\n"
            f"{skb.get_rel_info(node_id)}"
        )

    def get_text_info(node_id: int) -> str:
        return skb.get_doc_info(node_id, add_rel=False, compact=False)

    candidate_id_paths = {
        split: osp.join(processed_root, f"{split_mode}_{split}_ids.json")
        for split in ("train", "val", "test")
    }
    filtered_paths = {
        split: osp.join(processed_root, f"{split_mode}_filtered_{split}_ids.json")
        for split in ("train", "val", "test")
    }

    if not all(osp.exists(path) for path in candidate_id_paths.values()):
        vss = VSS(skb, query_emb_dir, node_emb_dir, emb_model=EMBED_MODEL, device="cpu")
        for split, indices in split_indices.items():
            candidate_dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _get_vss_topk, vss, qa_dataset[i][0], qa_dataset[i][1]
                    ): i
                    for i in indices
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"STaRK VSS {split}",
                ):
                    i = futures[future]
                    candidate_dict[i] = future.result()
            with open(candidate_id_paths[split], "w", encoding="utf-8") as f:
                json.dump(candidate_dict, f)

    candidate_dicts = {
        split: json.load(open(path, encoding="utf-8"))
        for split, path in candidate_id_paths.items()
    }

    if not all(osp.exists(path) for path in filtered_paths.values()):
        for split, indices in split_indices.items():
            result = {}
            for i in tqdm(indices, desc=f"STaRK features {split}"):
                candidate_id_dict = candidate_dicts[split]
                top_candidates = list(candidate_id_dict[str(i)].keys())[:5]
                top_candidates = [int(cid) for cid in top_candidates]
                answer_ids = qa_dataset[i][2]
                filtered = _filter_candidates(top_candidates, answer_ids, rng=rng)
                result[i] = {
                    "candidate_ids": filtered,
                    "similarity": [
                        candidate_id_dict[str(i)][str(cid)] for cid in filtered
                    ],
                    "relation_info": [get_rel_info(cid) for cid in filtered],
                    "text_info": [get_text_info(cid) for cid in filtered],
                }
            with open(filtered_paths[split], "w", encoding="utf-8") as f:
                json.dump(result, f)

    raw_results = {
        split: json.load(open(path, encoding="utf-8"))
        for split, path in filtered_paths.items()
    }
    results = {
        split: _truncate_infos({int(k): v for k, v in value.items()})
        for split, value in raw_results.items()
    }

    def build_examples(info: dict[int, dict[str, Any]]) -> list[StarkPrimeExample]:
        examples = []
        for i in info:
            examples.append(
                StarkPrimeExample(
                    question=qa_dataset[i][0],
                    id=str(qa_dataset[i][1]),
                    answer_ids=[int(x) for x in qa_dataset[i][2]],
                    candidate_ids=[int(x) for x in info[i]["candidate_ids"]],
                    relation_info=json.dumps(
                        info[i]["relation_info"], indent=2, ensure_ascii=False
                    ),
                    text_info=json.dumps(
                        info[i]["text_info"], indent=2, ensure_ascii=False
                    ),
                    emb_scores=[round(float(v), 2) for v in info[i]["similarity"]],
                )
            )
        return examples

    return {split: build_examples(info) for split, info in results.items()}


def dataset_engine(
    *,
    root: str | None = None,
    split_mode: str = "paper",
) -> tuple[list[StarkPrimeExample], list[StarkPrimeExample], list[StarkPrimeExample]]:
    base_root = root or str(Path(__file__).resolve().parent / "artifacts")
    splits = _build_processed_examples(root=base_root, split_mode=split_mode)
    trainset, valset, testset = splits["train"], splits["val"], splits["test"]
    print(
        f"[STaRK-Prime] Loaded {len(trainset)} train, {len(valset)} val, "
        f"{len(testset)} test examples (split_mode={split_mode})."
    )
    return trainset, valset, testset


def load_stark_prime(
    split: str = "train",
    *,
    root: str | None = None,
    split_mode: str = "paper",
) -> list[StarkPrimeExample]:
    trainset, valset, testset = dataset_engine(root=root, split_mode=split_mode)
    if split == "train":
        return trainset
    if split == "val":
        return valset
    if split == "test":
        return testset
    raise ValueError(f"Unsupported split: {split}")
