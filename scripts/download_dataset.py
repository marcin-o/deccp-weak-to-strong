from deccp_w2s.dataset import save_deccp_prompts


def main() -> None:
    prompts_df = save_deccp_prompts("data/deccp_prompts.csv")
    print(f"Saved {len(prompts_df)} prompts to data/deccp_prompts.csv")
    print(prompts_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
