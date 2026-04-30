"""
Prepare borough-level summary outputs for the website.

This script does three main things:
- Load and validate the raw `borough_summary.csv` table
- Derive extra borough and London indicators used in charts
- Export cleaned CSV and JSON outputs for frontend use.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "data" / "borough_summary.csv").exists():
    BASE_DIR = SCRIPT_DIR
else:
    BASE_DIR = SCRIPT_DIR.parent

INPUT_PATH = BASE_DIR / "data" / "borough_summary.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns that must exist in the input borough summary table.
EXPECTED_COLUMNS = [
    "borough",
    "inner_outer",
    "total_chargers",
    "total_population",
    "chargers_per_10k",
    "lsoa_count",
    "lsoa_with_charger",
    "pct_with_charger",
]

EXPECTED_ROW_COUNT = 33

#Convert a DataFrame to JSON-safe records
def serialise_records(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    cleaned_records = []

    for record in records:
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        cleaned_records.append(cleaned)

    return cleaned_records


#Validate the raw borough summary table and return a cleaned copy
#security checks
def validate_and_clean_borough_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    #Keep only the expected schema so downstream outputs are consistent
    cleaned = df[EXPECTED_COLUMNS].copy()
    issues: list[str] = []

    # Standardise text fields first to reduce matching and duplication issues
    cleaned["borough"] = cleaned["borough"].astype("string").str.strip()
    cleaned["inner_outer"] = cleaned["inner_outer"].astype("string").str.strip()

    numeric_columns = [
        "total_chargers",
        "total_population",
        "chargers_per_10k",
        "lsoa_count",
        "lsoa_with_charger",
        "pct_with_charger",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Borough-level London summaries should contain 33 rows.
    if len(cleaned) != EXPECTED_ROW_COUNT:
        issues.append(
            f"Expected {EXPECTED_ROW_COUNT} borough rows, found {len(cleaned)} rows."
        )

    # Report any missing values rather than hiding them.
    missing_value_counts = cleaned.isna().sum()
    missing_value_counts = missing_value_counts[missing_value_counts > 0]
    if not missing_value_counts.empty:
        issues.append(
            "Missing values found: "
            + ", ".join(
                f"{column} ({count})" for column, count in missing_value_counts.items()
            )
        )

    duplicated_boroughs = cleaned.loc[
        cleaned["borough"].duplicated(keep=False), "borough"
    ].tolist()
    if duplicated_boroughs:
        issues.append(
            "Duplicated borough names found: "
            + ", ".join(sorted(set(duplicated_boroughs)))
        )

    allowed_zones = {"Inner", "Outer"}
    invalid_zones = cleaned.loc[
        ~cleaned["inner_outer"].isin(allowed_zones), ["borough", "inner_outer"]
    ]
    if not invalid_zones.empty:
        issues.append(
            "Unexpected inner_outer values: "
            + "; ".join(
                f"{row.borough}={row.inner_outer}"
                for row in invalid_zones.itertuples(index=False)
            )
        )

    for column in numeric_columns:
        negative_rows = cleaned.loc[cleaned[column] < 0, "borough"].tolist()
        if negative_rows:
            issues.append(
                f"Negative values found in {column}: {', '.join(negative_rows)}"
            )

    invalid_pct_rows = cleaned.loc[
        (cleaned["pct_with_charger"] < 0) | (cleaned["pct_with_charger"] > 100),
        ["borough", "pct_with_charger"],
    ]
    if not invalid_pct_rows.empty:
        issues.append(
            "pct_with_charger outside 0-100: "
            + "; ".join(
                f"{row.borough}={row.pct_with_charger}"
                for row in invalid_pct_rows.itertuples(index=False)
            )
        )

    invalid_lsoa_rows = cleaned.loc[
        cleaned["lsoa_with_charger"] > cleaned["lsoa_count"],
        ["borough", "lsoa_with_charger", "lsoa_count"],
    ]
    if not invalid_lsoa_rows.empty:
        issues.append(
            "lsoa_with_charger exceeds lsoa_count: "
            + "; ".join(
                f"{row.borough} ({int(row.lsoa_with_charger)} > {int(row.lsoa_count)})"
                for row in invalid_lsoa_rows.itertuples(index=False)
            )
        )

    invalid_population_rows = cleaned.loc[
        cleaned["total_population"] <= 0, ["borough", "total_population"]
    ]
    if not invalid_population_rows.empty:
        issues.append(
            "Non-positive total_population values: "
            + "; ".join(
                f"{row.borough}={int(row.total_population)}"
                for row in invalid_population_rows.itertuples(index=False)
            )
        )

    # Derived fields use in later charts and summaries
    cleaned["pct_no_charger"] = 100 - cleaned["pct_with_charger"]
    cleaned["lsoa_without_charger"] = cleaned["lsoa_count"] - cleaned["lsoa_with_charger"]
    cleaned["_chargers_per_10k_raw"] = cleaned["chargers_per_10k"]
    cleaned["_pct_with_charger_raw"] = cleaned["pct_with_charger"]
    cleaned["_pct_no_charger_raw"] = cleaned["pct_no_charger"]

    for column in ["total_chargers", "total_population", "lsoa_count", "lsoa_with_charger"]:
        non_integer_rows = cleaned.loc[
            cleaned[column].notna() & ~np.isclose(cleaned[column] % 1, 0),
            ["borough", column],
        ]
        if not non_integer_rows.empty:
            issues.append(
                f"Non-integer values found in {column}: "
                + "; ".join(
                    f"{row.borough}={getattr(row, column)}"
                    for row in non_integer_rows.itertuples(index=False)
                )
            )

    integer_columns = [
        "total_chargers",
        "total_population",
        "lsoa_count",
        "lsoa_with_charger",
        "lsoa_without_charger",
    ]
    for column in integer_columns:
        if cleaned[column].isna().any():
            issues.append(
                f"Cannot fully cast {column} to integer because missing values remain."
            )
        else:
            cleaned[column] = cleaned[column].round().astype(int)

    # Final defensive checks for derived columns
    if cleaned["pct_no_charger"].isna().any():
        issues.append(
            "pct_no_charger contains missing values because pct_with_charger is incomplete."
        )

    if cleaned["lsoa_without_charger"].lt(0).any():
        issues.append("lsoa_without_charger contains negative values.")

    # Standardise frontend display precision.
    cleaned["chargers_per_10k"] = cleaned["chargers_per_10k"].round(1)
    cleaned["pct_with_charger"] = cleaned["pct_with_charger"].round(1)
    cleaned["pct_no_charger"] = cleaned["pct_no_charger"].round(1)

    return cleaned, issues

#Create London-wide summary metrics and key borough extremes
def create_borough_key_metrics(df: pd.DataFrame, validation_issues: list[str]) -> dict:
    london_total_chargers = int(df["total_chargers"].sum())
    london_total_population = int(df["total_population"].sum())
    london_total_lsoas = int(df["lsoa_count"].sum())
    london_lsoas_with_charger = int(df["lsoa_with_charger"].sum())

    london_chargers_per_10k = london_total_chargers / london_total_population * 10000
    london_pct_lsoas_with_charger = london_lsoas_with_charger / london_total_lsoas * 100
    london_pct_lsoas_without_charger = 100 - london_pct_lsoas_with_charger

    # use raw values for identifying true maxima/minima
    borough_highest_density = df.loc[df["_chargers_per_10k_raw"].idxmax()]
    borough_lowest_density = df.loc[df["_chargers_per_10k_raw"].idxmin()]
    borough_highest_coverage = df.loc[df["_pct_with_charger_raw"].idxmax()]
    borough_lowest_coverage = df.loc[df["_pct_with_charger_raw"].idxmin()]

    return {
        "total_boroughs": int(df["borough"].nunique()),
        "london_total_chargers": london_total_chargers,
        "london_total_population": london_total_population,
        "london_chargers_per_10k": round(london_chargers_per_10k, 1),
        "london_total_lsoas": london_total_lsoas,
        "london_lsoas_with_charger": london_lsoas_with_charger,
        "london_pct_lsoas_with_charger": round(london_pct_lsoas_with_charger, 1),
        "london_pct_lsoas_without_charger": round(london_pct_lsoas_without_charger, 1),
        "borough_highest_density": {
            "borough": borough_highest_density["borough"],
            "inner_outer": borough_highest_density["inner_outer"],
            "chargers_per_10k": float(borough_highest_density["chargers_per_10k"]),
            "total_chargers": int(borough_highest_density["total_chargers"]),
            "pct_with_charger": float(borough_highest_density["pct_with_charger"]),
        },
        "borough_lowest_density": {
            "borough": borough_lowest_density["borough"],
            "inner_outer": borough_lowest_density["inner_outer"],
            "chargers_per_10k": float(borough_lowest_density["chargers_per_10k"]),
            "total_chargers": int(borough_lowest_density["total_chargers"]),
            "pct_with_charger": float(borough_lowest_density["pct_with_charger"]),
        },
        "borough_highest_coverage": {
            "borough": borough_highest_coverage["borough"],
            "inner_outer": borough_highest_coverage["inner_outer"],
            "pct_with_charger": float(borough_highest_coverage["pct_with_charger"]),
            "pct_no_charger": float(borough_highest_coverage["pct_no_charger"]),
        },
        "borough_lowest_coverage": {
            "borough": borough_lowest_coverage["borough"],
            "inner_outer": borough_lowest_coverage["inner_outer"],
            "pct_with_charger": float(borough_lowest_coverage["pct_with_charger"]),
            "pct_no_charger": float(borough_lowest_coverage["pct_no_charger"]),
        },
        "data_quality_issues": validation_issues,
    }

#Build a borough density ranking table
def create_borough_ranking_chargers_per_10k(
    df: pd.DataFrame, london_chargers_per_10k: float
) -> pd.DataFrame:
    ranking = (
        df.sort_values("_chargers_per_10k_raw", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    ranking["density_rank"] = range(1, len(ranking) + 1)
    ranking["rank_group"] = "middle"
    ranking.loc[ranking["density_rank"] <= 5, "rank_group"] = "top_5"
    ranking.loc[ranking["density_rank"] > len(ranking) - 5, "rank_group"] = "bottom_5"
    ranking["london_chargers_per_10k"] = round(london_chargers_per_10k, 1)

    return ranking[
        [
            "density_rank",
            "borough",
            "inner_outer",
            "total_chargers",
            "total_population",
            "chargers_per_10k",
            "lsoa_count",
            "lsoa_with_charger",
            "pct_with_charger",
            "pct_no_charger",
            "rank_group",
            "london_chargers_per_10k",
        ]
    ]

#build a table for borough coverage gap view
def create_borough_coverage_gap(df: pd.DataFrame) -> pd.DataFrame:
    coverage_gap = (
        df.sort_values("_pct_no_charger_raw", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    coverage_gap["coverage_gap_rank"] = range(1, len(coverage_gap) + 1)

    return coverage_gap[
        [
            "coverage_gap_rank",
            "borough",
            "inner_outer",
            "lsoa_count",
            "lsoa_with_charger",
            "lsoa_without_charger",
            "pct_with_charger",
            "pct_no_charger",
            "chargers_per_10k",
        ]
    ]

#Aggregate borough rows into Inner + Outer London summaries.
def create_borough_inner_outer_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("inner_outer", dropna=False)
        .agg(
            borough_count=("borough", "count"),
            total_chargers=("total_chargers", "sum"),
            total_population=("total_population", "sum"),
            total_lsoas=("lsoa_count", "sum"),
            lsoas_with_charger=("lsoa_with_charger", "sum"),
            mean_chargers_per_10k=("_chargers_per_10k_raw", "mean"),
            median_chargers_per_10k=("_chargers_per_10k_raw", "median"),
            mean_pct_with_charger=("_pct_with_charger_raw", "mean"),
            median_pct_with_charger=("_pct_with_charger_raw", "median"),
        )
        .reset_index()
    )

    summary["weighted_chargers_per_10k"] = (
        summary["total_chargers"] / summary["total_population"] * 10000
    )
    summary["weighted_pct_with_charger"] = (
        summary["lsoas_with_charger"] / summary["total_lsoas"] * 100
    )
    summary["weighted_pct_no_charger"] = 100 - summary["weighted_pct_with_charger"]

    for column in [
        "mean_chargers_per_10k",
        "median_chargers_per_10k",
        "mean_pct_with_charger",
        "median_pct_with_charger",
        "weighted_chargers_per_10k",
        "weighted_pct_with_charger",
        "weighted_pct_no_charger",
    ]:
        summary[column] = summary[column].round(1)

    return summary.sort_values("inner_outer").reset_index(drop=True)


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)


def main() -> None:
    borough_summary = pd.read_csv(INPUT_PATH)
    processed_summary, validation_issues = validate_and_clean_borough_summary(
        borough_summary
    )

    # Create all chart + card datasets used by the borough analysis section
    key_metrics = create_borough_key_metrics(processed_summary, validation_issues)
    borough_ranking = create_borough_ranking_chargers_per_10k(
        processed_summary, key_metrics["london_chargers_per_10k"]
    )
    coverage_gap = create_borough_coverage_gap(processed_summary)
    inner_outer_summary = create_borough_inner_outer_summary(processed_summary)

    # Remove temporary raw sorting columns before exporting the cleaned master table
    processed_summary_output = processed_summary.drop(
        columns=["_chargers_per_10k_raw", "_pct_with_charger_raw", "_pct_no_charger_raw"]
    )

    # Export both CSV and JSON so the data
    processed_summary_output.to_csv(OUTPUT_DIR / "borough_summary_processed.csv", index=False)

    write_json(
        OUTPUT_DIR / "borough_summary_processed.json",
        serialise_records(processed_summary_output),
    )
    write_json(OUTPUT_DIR / "borough_key_metrics.json", key_metrics)
    write_json(
        OUTPUT_DIR / "borough_ranking_chargers_per_10k.json",
        serialise_records(borough_ranking),
    )
    write_json(
        OUTPUT_DIR / "borough_coverage_gap.json",
        serialise_records(coverage_gap),
    )
    write_json(
        OUTPUT_DIR / "borough_inner_outer_summary.json",
        serialise_records(inner_outer_summary),
    )

    print("\n Output files ")
    print(OUTPUT_DIR / "borough_summary_processed.csv")
    print(OUTPUT_DIR / "borough_summary_processed.json")
    print(OUTPUT_DIR / "borough_key_metrics.json")
    print(OUTPUT_DIR / "borough_ranking_chargers_per_10k.json")
    print(OUTPUT_DIR / "borough_coverage_gap.json")
    print(OUTPUT_DIR / "borough_inner_outer_summary.json")


if __name__ == "__main__":
    main()