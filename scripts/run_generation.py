from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deccp_w2s.generation import run_model_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="data/deccp_prompts.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_path}. "
            "Run `python scripts/download_dataset.py` first."
        )

    prompts_df = pd.read_csv(prompts_path)

    results_df = run_model_to_csv(
        model_name=args.model,
        prompts_df=prompts_df,
        output_file=args.output,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Saved {len(results_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
