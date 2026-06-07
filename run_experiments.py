"""Training pipeline entry point.

Runs the full experiment matrix and writes result CSVs into results/.
The report notebook then loads these CSVs; it does not retrain.

Usage
-----
    python run_experiments.py
"""

import os

from src import config
from src.data import load_all_data
from src.experiment import run_all_experiments, summarise, build_stats_table


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    df_all = load_all_data()
    print(f"Loaded {len(df_all)} rows across {df_all['lang'].nunique()} languages.")

    print("Running experiments (this is the long step)...")
    store = run_all_experiments(df_all)

    results_df = store.results_frame()
    predictions_df = store.predictions_frame()
    summary_df = summarise(results_df)
    stats_df = build_stats_table(results_df)

    results_df.to_csv(os.path.join(config.RESULTS_DIR, "results_full.csv"), index=False)
    summary_df.to_csv(os.path.join(config.RESULTS_DIR, "summary_df.csv"), index=False)
    stats_df.to_csv(os.path.join(config.RESULTS_DIR, "stats_df.csv"), index=False)
    predictions_df.to_csv(os.path.join(config.RESULTS_DIR, "predictions_df.csv"), index=False)

    print("Saved results to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()