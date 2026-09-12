# =============================================================================
# src/project_config.py
# ASU LLM Evaluation — shared paths, config.yaml access, API keys, pricing
#
# Every module reads settings through here so paths, .env loading and
# config parsing live in one place. API keys are checked lazily (only when a
# client is actually created), so modules can be imported — e.g. by the unit
# tests — without any keys configured.
# =============================================================================

import os

import yaml
from dotenv import load_dotenv

PROJECT_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH         = os.path.join(PROJECT_ROOT, "config.yaml")
ENV_PATH            = os.path.join(PROJECT_ROOT, ".env")
KNOWLEDGE_BASE_DIR  = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
RESULTS_DIR         = os.path.join(PROJECT_ROOT, "results")

# Real environment variables (e.g. GitHub Actions secrets) take precedence
# over .env — load_dotenv never overrides variables that are already set.
load_dotenv(dotenv_path=ENV_PATH)


def load_config() -> dict:
    """Read config.yaml. Re-read on every call so threshold edits apply immediately."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"config.yaml not found at: {CONFIG_PATH}")

    for section in ("quality_gates", "evaluation"):
        if section not in cfg:
            raise KeyError(f"config.yaml is missing the '{section}' section.")
    return cfg


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"{name} not found. Set it as an environment variable "
            f"or add it to the .env file at {ENV_PATH}."
        )
    return value


def token_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """
    Cost of one call from the per-1M-token prices in config.yaml → pricing.
    Returns None when the model has no configured price.
    """
    prices = (load_config().get("pricing") or {}).get(model)
    if not prices:
        return None
    return (
        (input_tokens or 0) * prices.get("input", 0.0)
        + (output_tokens or 0) * prices.get("output", 0.0)
    ) / 1_000_000
