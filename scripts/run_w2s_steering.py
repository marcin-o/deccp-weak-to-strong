from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deccp_w2s.steering import run_w2s_to_csv


MODEL_DIR_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen_0_5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen_1_5b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen_14b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_7b",
}


def model_dir_name(model_id: str) -> str:
    if model_id in MODEL_DIR_MAP:
        return MODEL_DIR_MAP[model_id]
    return model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weak-to-strong steering via logit-difference decoding."
    )
    parser.add_argument("--strong-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--weak-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter dir for the weak model. "
        "Defaults to results/steered/<weak_dir>/adapters.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength. 0 = strong model unchanged; higher pushes "
        "harder toward the weak expert.",
    )
    parser.add_argument("--prompts", default="data/deccp_prompts.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to results/steered/<strong_dir>_w2s/raw_responses.csv.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
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

    weak_dir = model_dir_name(args.weak_model)
    strong_dir = model_dir_name(args.strong_model)

    adapter_dir = args.adapter_dir or f"results/steered/{weak_dir}/adapters"
    output = args.output or f"results/steered/{strong_dir}_w2s/raw_responses.csv"

    print(f"=== Weak-to-strong steering ===")
    print(f"-> strong : {args.strong_model}")
    print(f"-> expert : {args.weak_model} + LoRA ({adapter_dir})")
    print(f"-> amateur: {args.weak_model} (base)")
    print(f"-> alpha  : {args.alpha}")
    print(f"-> output : {output}")

    results_df = run_w2s_to_csv(
        strong_model_id=args.strong_model,
        weak_model_id=args.weak_model,
        adapter_dir=adapter_dir,
        prompts_df=prompts_df,
        output_file=output,
        alpha=args.alpha,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )

    print(f"=== SUCCESS: saved {len(results_df)} rows to {output} ===")


if __name__ == "__main__":
    main()
