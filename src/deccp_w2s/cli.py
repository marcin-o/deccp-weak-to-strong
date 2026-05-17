from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deccp_w2s.analysis import analyze_judged_results, print_analysis
from deccp_w2s.dataset import save_deccp_prompts
from deccp_w2s.generation import run_model_to_csv


def add_download_dataset_parser(subparsers) -> None:
    parser = subparsers.add_parser("download-dataset")
    parser.add_argument("--dataset", default="augmxnt/deccp")
    parser.add_argument("--output", default="data/deccp_prompts.csv")
    parser.set_defaults(func=handle_download_dataset)


def add_generate_parser(subparsers) -> None:
    parser = subparsers.add_parser("generate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="data/deccp_prompts.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(func=handle_generate)


def add_analyze_parser(subparsers) -> None:
    parser = subparsers.add_parser("analyze")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--specificity-output", default=None)
    parser.set_defaults(func=handle_analyze)


def handle_download_dataset(args) -> None:
    prompts_df = save_deccp_prompts(
        output_file=args.output,
        dataset_name=args.dataset,
    )

    print(f"Saved {len(prompts_df)} prompts to {args.output}")
    print(prompts_df["split"].value_counts().to_string())


def handle_generate(args) -> None:
    prompts_path = Path(args.prompts)

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_path}. "
            "Run `python -m deccp_w2s.cli download-dataset` first."
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


def handle_analyze(args) -> None:
    summary_df, specificity_summary = analyze_judged_results(args.input)
    print_analysis(summary_df, specificity_summary)

    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_output, index=False)
        print(f"Saved summary to {args.summary_output}")

    if args.specificity_output:
        Path(args.specificity_output).parent.mkdir(parents=True, exist_ok=True)
        specificity_summary.to_csv(args.specificity_output, index=False)
        print(f"Saved specificity summary to {args.specificity_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deccp-w2s",
        description="Baseline pipeline for DeCCP weak-to-strong steering experiments.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_download_dataset_parser(subparsers)
    add_generate_parser(subparsers)
    add_analyze_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
