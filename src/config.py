"""Configuration loading for the MM-LLM benchmark pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class AppConfig:
    """Fully resolved runtime configuration."""

    dataset_dir: Path
    metadata_path: Path
    output_dir: Path
    split: str
    limit_qa: int | None
    reuse_cache: bool
    backend: str
    model_name: str
    api_key: str | None
    api_base_url: str | None
    temperature: float
    max_tokens: int
    timeout_seconds: int
    prompt_version: str
    hf_device_map: str
    hf_dtype: str
    hf_attn_implementation: str | None
    hf_trust_remote_code: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def load_dotenv_defaults(env_path: Path = ENV_PATH) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without extra dependencies."""

    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def resolve_repo_path(path_value: str | None) -> Path | None:
    """Resolve a possibly relative path against the repository root."""

    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def build_cli_defaults() -> dict[str, Any]:
    """Build CLI defaults from .env, keeping backward compatibility."""

    env_values = load_dotenv_defaults()
    backend = env_values.get("LLM_BACKEND", "").strip()
    if not backend:
        use_api = parse_bool(env_values.get("LLM_USE_API"), default=False)
        backend = "openai_compatible" if use_api else "hf_local"

    output_dir = env_values.get("MM_LLM_OUTPUT_DIR") or env_values.get("LLM_OUTPUT_DIR") or "runs/mm_llm/latest"
    openai_model = env_values.get("OPENAI_MODEL_USE", "").strip()
    llm_model = env_values.get("LLM_MODEL_NAME", "").strip()
    model_name = openai_model if backend == "openai_compatible" and openai_model else llm_model

    return {
        "dataset_dir": env_values.get("MAIN_DATASET_DIR", "datasets/out"),
        "metadata_path": env_values.get("MAIN_METADATA_PATH") or None,
        "output_dir": output_dir,
        "split": env_values.get("RUN_SPLIT", "test"),
        "limit_qa": parse_optional_int(env_values.get("RUN_LIMIT_QA")),
        "reuse_cache": parse_bool(env_values.get("RUN_REUSE_CACHE"), default=True),
        "backend": backend,
        "model_name": model_name,
        "api_key": env_values.get("OPENAI_API_KEY") or env_values.get("LLM_API_KEY") or None,
        "api_base_url": env_values.get("LLM_API_BASE_URL") or None,
        "temperature": float(env_values.get("LLM_TEMPERATURE", "0")),
        "max_tokens": int(env_values.get("LLM_MAX_TOKENS", "128")),
        "timeout_seconds": int(env_values.get("LLM_TIMEOUT_SECONDS", "60")),
        "prompt_version": env_values.get("RUN_PROMPT_VERSION", "chartqa_direct_v1"),
        "hf_device_map": env_values.get("HF_DEVICE_MAP", "auto"),
        "hf_dtype": env_values.get("HF_DTYPE", "auto"),
        "hf_attn_implementation": env_values.get("HF_ATTN_IMPLEMENTATION") or None,
        "hf_trust_remote_code": parse_bool(env_values.get("HF_TRUST_REMOTE_CODE"), default=False),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MM-LLM direct-QA benchmark."""

    defaults = build_cli_defaults()
    parser = argparse.ArgumentParser(
        description="Run an MM-LLM direct-QA baseline over the generated chart dataset."
    )
    parser.add_argument("--dataset-dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--metadata-path", type=str, default=defaults["metadata_path"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    parser.add_argument(
        "--split",
        type=str,
        default=defaults["split"],
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit-qa", type=int, default=defaults["limit_qa"])
    parser.add_argument("--backend", type=str, default=defaults["backend"], choices=["openai_compatible", "hf_local"])
    parser.add_argument("--model-name", type=str, default=defaults["model_name"])
    parser.add_argument("--api-key", type=str, default=defaults["api_key"])
    parser.add_argument("--api-base-url", type=str, default=defaults["api_base_url"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--timeout-seconds", type=int, default=defaults["timeout_seconds"])
    parser.add_argument("--prompt-version", type=str, default=defaults["prompt_version"])
    parser.add_argument("--hf-device-map", type=str, default=defaults["hf_device_map"])
    parser.add_argument("--hf-dtype", type=str, default=defaults["hf_dtype"])
    parser.add_argument("--hf-attn-implementation", type=str, default=defaults["hf_attn_implementation"])
    parser.add_argument(
        "--hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_true",
        default=defaults["hf_trust_remote_code"],
    )
    parser.add_argument(
        "--no-hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_false",
    )
    parser.add_argument(
        "--reuse-cache",
        dest="reuse_cache",
        action="store_true",
        default=defaults["reuse_cache"],
    )
    parser.add_argument(
        "--no-reuse-cache",
        dest="reuse_cache",
        action="store_false",
    )
    return parser.parse_args()


def build_app_config(args: argparse.Namespace) -> AppConfig:
    """Resolve CLI arguments into a validated AppConfig."""

    dataset_dir = resolve_repo_path(args.dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset_dir must not be empty.")

    metadata_path = resolve_repo_path(args.metadata_path) if args.metadata_path else dataset_dir / "metadata.jsonl"
    if metadata_path is None:
        raise ValueError("metadata_path must not be empty.")

    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must not be empty.")

    model_name = args.model_name.strip()
    if not model_name:
        raise ValueError("model_name must not be empty. Set LLM_MODEL_NAME in .env or pass --model-name.")

    return AppConfig(
        dataset_dir=dataset_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        split=args.split,
        limit_qa=args.limit_qa,
        reuse_cache=bool(args.reuse_cache),
        backend=args.backend,
        model_name=model_name,
        api_key=args.api_key.strip() if isinstance(args.api_key, str) and args.api_key.strip() else None,
        api_base_url=(
            args.api_base_url.strip() if isinstance(args.api_base_url, str) and args.api_base_url.strip() else None
        ),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_seconds=int(args.timeout_seconds),
        prompt_version=args.prompt_version,
        hf_device_map=args.hf_device_map,
        hf_dtype=args.hf_dtype,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_trust_remote_code=bool(args.hf_trust_remote_code),
    )
