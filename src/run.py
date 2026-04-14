"""Entry point for the MM-LLM direct-QA benchmark pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from src.config import build_app_config, parse_args
from src.pipeline.runner import run_pipeline


def main() -> None:
    """Run the configured MM-LLM benchmark pipeline."""

    args = parse_args()
    config = build_app_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
