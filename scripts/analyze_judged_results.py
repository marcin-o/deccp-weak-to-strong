from __future__ import annotations

import argparse
from pathlib import Path

from deccp_w2s.analysis import analyze_judged_results, print_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--specificity-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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


if __name__ == "__main__":
    main()
