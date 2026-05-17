from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "refusal",
    "ccp_propaganda",
    "answers_question",
    "factual_specificity",
    "success",
]


def load_judged_results(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def analyze_judged_results(input_file: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_judged_results(input_file)
    total = len(df)

    non_refusals = df[df["refusal"] == 0]
    propaganda_among_non_refusals = (
        non_refusals["ccp_propaganda"].mean() if len(non_refusals) > 0 else 0
    )

    summary_df = pd.DataFrame(
        [
            {
                "file": str(input_file),
                "total_rows": total,
                "refusals": int(df["refusal"].sum()),
                "refusal_rate": df["refusal"].mean(),
                "ccp_propaganda": int(df["ccp_propaganda"].sum()),
                "ccp_propaganda_rate": df["ccp_propaganda"].mean(),
                "answers_question": int(df["answers_question"].sum()),
                "answers_question_rate": df["answers_question"].mean(),
                "success": int(df["success"].sum()),
                "success_rate": df["success"].mean(),
                "failed": int(total - df["success"].sum()),
                "failed_rate": 1 - df["success"].mean(),
                "ccp_propaganda_among_non_refusals": propaganda_among_non_refusals,
            }
        ]
    )

    specificity_summary = (
        df["factual_specificity"].value_counts().sort_index().reset_index()
    )
    specificity_summary.columns = ["factual_specificity", "count"]
    specificity_summary["rate"] = specificity_summary["count"] / total

    return summary_df, specificity_summary


def print_analysis(summary_df: pd.DataFrame, specificity_summary: pd.DataFrame) -> None:
    row = summary_df.iloc[0]
    total = int(row["total_rows"])

    print(f"File: {row['file']}")
    print(f"Total rows: {total}")
    print()

    print(
        f"Refusals: {int(row['refusals'])} / {total} "
        f"({row['refusal_rate'] * 100:.2f}%)"
    )
    print(
        f"CCP propaganda: {int(row['ccp_propaganda'])} / {total} "
        f"({row['ccp_propaganda_rate'] * 100:.2f}%)"
    )
    print(
        f"Answers question: {int(row['answers_question'])} / {total} "
        f"({row['answers_question_rate'] * 100:.2f}%)"
    )
    print(
        f"Success: {int(row['success'])} / {total} "
        f"({row['success_rate'] * 100:.2f}%)"
    )
    print(
        f"Failed: {int(row['failed'])} / {total} "
        f"({row['failed_rate'] * 100:.2f}%)"
    )
    print(
        "CCP propaganda among non-refusals: "
        f"{row['ccp_propaganda_among_non_refusals'] * 100:.2f}%"
    )
    print()
    print("Factual specificity:")
    print(specificity_summary.to_string(index=False))
