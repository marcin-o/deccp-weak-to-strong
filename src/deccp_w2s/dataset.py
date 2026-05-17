from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_deccp_prompts(dataset_name: str = "augmxnt/deccp") -> pd.DataFrame:
    dataset = load_dataset(dataset_name)

    rows = []
    global_prompt_id = 0

    for split_name, split_data in dataset.items():
        for split_prompt_id, example in enumerate(split_data):
            rows.append(
                {
                    "prompt_id": global_prompt_id,
                    "split": split_name,
                    "split_prompt_id": split_prompt_id,
                    "prompt": example["text"],
                }
            )
            global_prompt_id += 1

    return pd.DataFrame(rows)


def save_deccp_prompts(
    output_file: str | Path = "data/deccp_prompts.csv",
    dataset_name: str = "augmxnt/deccp",
) -> pd.DataFrame:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_df = load_deccp_prompts(dataset_name=dataset_name)
    prompts_df.to_csv(output_path, index=False)

    return prompts_df
