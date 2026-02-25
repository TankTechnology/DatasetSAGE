#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasetsage.io import load_query_paper, load_usage_records  # noqa: E402
from datasetsage.pipeline import DatasetSAGEPipeline, PipelineConfig  # noqa: E402
from datasetsage.llm import LLMConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact DatasetSAGE demo.")
    parser.add_argument(
        "--records",
        required=True,
        help="Path to usage-record json file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Path to query paper json file.",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Closed-loop rounds (paper default: 2).")
    parser.add_argument("--top-k", type=int, default=30, help="Final recommendation size (paper default: 30).")
    parser.add_argument("--chat-model", default="qwen-plus", help="LLM model for agent reasoning.")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for RAG retrieval.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing API key.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output json path. Prints to stdout when empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_usage_records(args.records)
    query = load_query_paper(args.query)

    try:
        pipeline = DatasetSAGEPipeline(
            records,
            PipelineConfig(
                rounds=args.rounds,
                top_k=args.top_k,
                llm=LLMConfig(
                    chat_model=args.chat_model,
                    embedding_model=args.embedding_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                ),
            ),
        )
        result = pipeline.run(query)
    except RuntimeError as exc:
        message = str(exc)
        if "Missing API key" in message:
            raise SystemExit(
                f"{message}\n\n"
                "Setup example:\n"
                "  export OPENAI_API_KEY='your_key'\n"
                "  python scripts/run_demo.py\n"
            ) from exc
        raise

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote result to {output_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
