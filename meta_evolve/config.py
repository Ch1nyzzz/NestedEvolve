"""Configuration loading and runtime path helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
CACHE_DIR = PACKAGE_ROOT / "cache"


def ensure_runtime_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def load_config(path: str | None = None) -> dict:
    config_path = Path(path) if path is not None else CONFIGS_DIR / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_repo_root() -> Path:
    return REPO_ROOT


def artifact_path(name: str) -> Path:
    ensure_runtime_dirs()
    return ARTIFACTS_DIR / name


def cache_path(name: str) -> Path:
    ensure_runtime_dirs()
    return CACHE_DIR / name


def shared_skill_library_path() -> Path:
    return artifact_path("skill_library_shared.json")


def shared_skill_artifact_path() -> Path:
    return artifact_path("skill_task_artifacts.json")


def managed_skills_dir() -> Path:
    return PACKAGE_ROOT / "managed_skills"
