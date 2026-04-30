'''
Reproducible borough-level Census context processing for the project

this script:
- Reads an existing borough EV charging summary table
- Reads borough-level Census 2021 TS045 car ownership data
- Reads borough-level Census 2021 TS044 housing type data
- Cleans borough names so that all three datasets can be merged safely
- Aggregates raw Census categories into interpretable borough indicators
- Produces updated borough summary CSV outputs for website charts

note:
This script only creates borough-level CSV outputs and does not modify any
LSOA-level GeoJSON files
'''

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Using paths relative to this script keeps the workflow reproducible across machines
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

TS045_PATH = DATA_DIR / "TS045_car_ownership.csv"
TS044_PATH = DATA_DIR / "TS044_housing_type.csv"
BOROUGH_SUMMARY_PATH = DATA_DIR / "borough_summary.csv"

CAR_OUTPUT_PATH = DATA_DIR / "car_ownership_borough.csv"
HOUSING_OUTPUT_PATH = DATA_DIR / "housing_type_borough.csv"
BOROUGH_UPDATED_OUTPUT_PATH = DATA_DIR / "borough_updated_summary.csv"
INNER_OUTER_UPDATED_OUTPUT_PATH = DATA_DIR / "inner_outer_updated_summary.csv"

# London borough-level outputs should contain 33 rows in this project
#(32 boroughs + City of London).
EXPECTED_BOROUGH_COUNT = 33


def print_step(message: str) -> None:
    print(f"\n=== {message} ===")


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())

##this helper removes punctuation, standardises spacing, and normalises common prefixes
def borough_key(value: str) -> str:
    text = normalize_spaces(value)
    text = text.replace("&", "and")
    text = text.replace("-", " ")
    text = re.sub(r"^london borough of ", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^royal borough of ", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^westminster city$", "westminster", text, flags=re.IGNORECASE)
    text = re.sub(r"^city of westminster$", "westminster", text, flags=re.IGNORECASE)
    text = re.sub(r"^city of london corporation$", "city of london", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

#Create a simplified matching key for Census category labels.
def category_key(value: str) -> str:
    text = normalize_spaces(value)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


#Build a lookup table from cleaned borough keys to canonical borough names
def build_borough_lookup(borough_names: list[str]) -> dict[str, str]:
    lookup = {borough_key(name): normalize_spaces(name) for name in borough_names}

    aliases = {
        "barking dagenham": "Barking and Dagenham",
        "barking and dagenham": "Barking and Dagenham",
        "hammersmith fulham": "Hammersmith and Fulham",
        "hammersmith and fulham": "Hammersmith and Fulham",
        "kensington chelsea": "Kensington and Chelsea",
        "kensington and chelsea": "Kensington and Chelsea",
        "kingston upon thames": "Kingston upon Thames",
        "richmond upon thames": "Richmond upon Thames",
        "city of london": "City of London",
        "westminster": "Westminster",
    }

    for alias, canonical in aliases.items():
        if canonical in borough_names:
            lookup[borough_key(alias)] = canonical

    return lookup

#Return the canonical borough name whenever a known alias is found
def clean_borough_name(value: str, lookup: dict[str, str]) -> str:
    cleaned = normalize_spaces(value)
    return lookup.get(borough_key(cleaned), cleaned)

# Fail early if an input file is missing required columns
def require_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")

# Match a raw Census category label to one cleaned output category
# entry rule is (output_column_name, matching_function). 
def match_category(raw_value: str, rules: list[tuple[str, callable]]) -> str | None:
    text = category_key(raw_value)
    for output_name, matcher in rules:
        if matcher(text):
            return output_name
    return None

# This table defines the borough list and the canonical borough naming convention used by later merge steps.
def load_borough_summary() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    print_step("Loading borough_summary.csv")
    borough_summary = pd.read_csv(BOROUGH_SUMMARY_PATH)
    require_columns(
        borough_summary,
        [
            "borough",
            "inner_outer",
            "total_chargers",
            "total_population",
            "chargers_per_10k",
            "lsoa_count",
            "lsoa_with_charger",
            "pct_with_charger",
        ],
        "borough_summary.csv",
    )

    borough_summary["borough"] = borough_summary["borough"].astype("string").map(normalize_spaces)
    borough_names = borough_summary["borough"].tolist()
    lookup = build_borough_lookup(borough_names)
    borough_summary["borough"] = borough_summary["borough"].map(lambda value: clean_borough_name(value, lookup))

    print(f"borough_summary row count: {len(borough_summary)}")
    return borough_summary, borough_names, lookup

#Keep London borough rows only.
def filter_london_rows(
    df: pd.DataFrame,
    code_column: str,
    borough_column: str,
    borough_names: list[str],
    lookup: dict[str, str],
    dataset_name: str,
) -> pd.DataFrame:
    working = df.copy()
    working[borough_column] = working[borough_column].astype("string").map(normalize_spaces)
    working["borough"] = working[borough_column].map(lambda value: clean_borough_name(value, lookup))

    london_by_code = working[working[code_column].astype("string").str.startswith("E09", na=False)].copy()
    if london_by_code["borough"].nunique() > 0:
        print(f"{dataset_name}: filtered London rows using {code_column} prefix E09.")
        return london_by_code

    london_by_name = working[working["borough"].isin(set(borough_names))].copy()
    print(f"{dataset_name}: E09 filtering failed, fell back to borough name matching.")
    return london_by_name


#Clean borough-level TS045 car ownership data
def process_ts045(borough_names: list[str], lookup: dict[str, str]) -> pd.DataFrame:
    print_step("Processing TS045 car ownership")
    df = pd.read_csv(TS045_PATH)
    require_columns(
        df,
        [
            "Lower tier local authorities Code",
            "Lower tier local authorities",
            "Car or van availability (5 categories)",
            "Observation",
        ],
        "TS045_car_ownership.csv",
    )

    london = filter_london_rows(
        df,
        "Lower tier local authorities Code",
        "Lower tier local authorities",
        borough_names,
        lookup,
        "TS045",
    )
    print(f"TS045 London borough count: {london['borough'].nunique()}")

    # Print available raw categories so any future mismatch is easy to debug
    available_categories = sorted(london["Car or van availability (5 categories)"].dropna().unique().tolist())
    print("TS045 available categories:")
    for category in available_categories:
        print(f"- {category}")

    # Convert observations to numeric and drop "does not apply" rows
    london["Observation"] = pd.to_numeric(london["Observation"], errors="coerce")
    london = london[
        ~london["Car or van availability (5 categories)"].astype("string").str.contains("does not apply", case=False, na=False)
    ].copy()

    #Map raw Census labels into a smaller set of ready-to-analysis categories
    rules = [
        ("no_car_households", lambda text: "no cars or vans" in text),
        ("one_car_households", lambda text: "1 car" in text),
        ("two_car_households", lambda text: "2 cars" in text),
        ("three_plus_car_households", lambda text: "3 or more" in text),
    ]
    london["category_clean"] = london["Car or van availability (5 categories)"].map(
        lambda value: match_category(value, rules)
    )

    unmatched = sorted(
        london.loc[london["category_clean"].isna(), "Car or van availability (5 categories)"].dropna().unique().tolist()
    )
    if unmatched:
        print("TS045 unmatched categories:")
        for category in unmatched:
            print(f"- {category}")
        raise ValueError("TS045 required category matching failed.")

    # Pivot from long-format Census rows into one row per borough
    pivot = (
        london.pivot_table(
            index=["Lower tier local authorities Code", "borough"],
            columns="category_clean",
            values="Observation",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"Lower tier local authorities Code": "lad_code"})
    )

    base_columns = [
        "no_car_households",
        "one_car_households",
        "two_car_households",
        "three_plus_car_households",
    ]
    for column in base_columns:
        if column not in pivot.columns:
            raise ValueError(f"TS045 output is missing expected category column: {column}")
        pivot[column] = pivot[column].round().astype(int)

    # Build derived car ownership indicators
    pivot["households_total"] = pivot[base_columns].sum(axis=1)
    pivot["car_owning_households"] = (
        pivot["one_car_households"] + pivot["two_car_households"] + pivot["three_plus_car_households"]
    )
    pivot["pct_no_car_households"] = pivot["no_car_households"] / pivot["households_total"] * 100
    pivot["pct_car_owning_households"] = pivot["car_owning_households"] / pivot["households_total"] * 100

    #the 3 or more cars or vans category is treated as exactly 3
    pivot["cars_or_vans_minimum_estimate"] = (
        pivot["one_car_households"]
        + pivot["two_car_households"] * 2
        + pivot["three_plus_car_households"] * 3
    )
    pivot["cars_or_vans_per_100_households_min"] = (
        pivot["cars_or_vans_minimum_estimate"] / pivot["households_total"] * 100
    )

    for column in [
        "pct_no_car_households",
        "pct_car_owning_households",
        "cars_or_vans_per_100_households_min",
    ]:
        pivot[column] = pivot[column].round(2)

    pivot["borough"] = pivot["borough"].map(lambda value: clean_borough_name(value, lookup))
    pivot = pivot[
        [
            "lad_code",
            "borough",
            "households_total",
            "no_car_households",
            "one_car_households",
            "two_car_households",
            "three_plus_car_households",
            "car_owning_households",
            "pct_no_car_households",
            "pct_car_owning_households",
            "cars_or_vans_minimum_estimate",
            "cars_or_vans_per_100_households_min",
        ]
    ].sort_values("borough").reset_index(drop=True)

    pivot.to_csv(CAR_OUTPUT_PATH, index=False)
    print(f"car ownership output row count: {len(pivot)}")
    print(f"Saved: {CAR_OUTPUT_PATH}")
    return pivot

#Clean borough-level TS044 housing type data.
def process_ts044(borough_names: list[str], lookup: dict[str, str]) -> pd.DataFrame:
    print_step("Processing TS044 housing type")
    df = pd.read_csv(TS044_PATH)
    require_columns(
        df,
        [
            "Lower tier local authorities Code",
            "Lower tier local authorities",
            "Accommodation type (8 categories)",
            "Observation",
        ],
        "TS044_housing_type.csv",
    )

    london = filter_london_rows(
        df,
        "Lower tier local authorities Code",
        "Lower tier local authorities",
        borough_names,
        lookup,
        "TS044",
    )
    print(f"TS044 London borough count: {london['borough'].nunique()}")

    # Print raw category labels for transparency
    available_categories = sorted(london["Accommodation type (8 categories)"].dropna().unique().tolist())
    print("TS044 available categories:")
    for category in available_categories:
        print(f"- {category}")

    london["Observation"] = pd.to_numeric(london["Observation"], errors="coerce")

    # transform detailed accommodation categories into grouped housing indicators
    rules = [
        ("semi_detached_households", lambda text: "semi detached" in text),
        ("detached_households", lambda text: "detached" in text and "semi detached" not in text),
        ("terraced_households", lambda text: "terraced" in text),
        ("purpose_built_flat_households", lambda text: "purpose built" in text),
        ("converted_shared_house_households", lambda text: "converted or shared house" in text),
        ("other_converted_building_households", lambda text: "another converted building" in text),
        ("commercial_building_households", lambda text: "commercial building" in text),
        ("caravan_temporary_households", lambda text: "caravan" in text or "temporary" in text),
    ]
    london["category_clean"] = london["Accommodation type (8 categories)"].map(
        lambda value: match_category(value, rules)
    )

    unmatched = sorted(
        london.loc[london["category_clean"].isna(), "Accommodation type (8 categories)"].dropna().unique().tolist()
    )
    if unmatched:
        print("TS044 unmatched categories:")
        for category in unmatched:
            print(f"- {category}")
        raise ValueError("TS044 required category matching failed.")

    # Pivot from long format into one row per borough
    pivot = (
        london.pivot_table(
            index=["Lower tier local authorities Code", "borough"],
            columns="category_clean",
            values="Observation",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"Lower tier local authorities Code": "lad_code"})
    )

    raw_columns = [
        "detached_households",
        "semi_detached_households",
        "terraced_households",
        "purpose_built_flat_households",
        "converted_shared_house_households",
        "other_converted_building_households",
        "commercial_building_households",
        "caravan_temporary_households",
    ]
    for column in raw_columns:
        if column not in pivot.columns:
            raise ValueError(f"TS044 output is missing expected category column: {column}")
        pivot[column] = pivot[column].round().astype(int)

    #Combine detailed categories into broader analytical groups
    pivot["house_households"] = (
        pivot["detached_households"]
        + pivot["semi_detached_households"]
        + pivot["terraced_households"]
    )
    pivot["flat_households"] = (
        pivot["purpose_built_flat_households"]
        + pivot["converted_shared_house_households"]
        + pivot["other_converted_building_households"]
        + pivot["commercial_building_households"]
    )
    pivot["other_households"] = pivot["caravan_temporary_households"]
    pivot["households_total_housing"] = (
        pivot["house_households"] + pivot["flat_households"] + pivot["other_households"]
    )
    pivot["pct_houses"] = pivot["house_households"] / pivot["households_total_housing"] * 100
    pivot["pct_flats"] = pivot["flat_households"] / pivot["households_total_housing"] * 100
    pivot["pct_other_housing"] = pivot["other_households"] / pivot["households_total_housing"] * 100

    for column in ["pct_houses", "pct_flats", "pct_other_housing"]:
        pivot[column] = pivot[column].round(2)

    pivot["borough"] = pivot["borough"].map(lambda value: clean_borough_name(value, lookup))
    pivot = pivot[
        [
            "lad_code",
            "borough",
            "households_total_housing",
            "detached_households",
            "semi_detached_households",
            "terraced_households",
            "purpose_built_flat_households",
            "converted_shared_house_households",
            "other_converted_building_households",
            "commercial_building_households",
            "caravan_temporary_households",
            "house_households",
            "flat_households",
            "other_households",
            "pct_houses",
            "pct_flats",
            "pct_other_housing",
        ]
    ].sort_values("borough").reset_index(drop=True)

    pivot.to_csv(HOUSING_OUTPUT_PATH, index=False)
    print(f"housing type output row count: {len(pivot)}")
    print(f"Saved: {HOUSING_OUTPUT_PATH}")
    return pivot

#Print borough name mismatches before a merge and return missing names
def report_name_differences(reference: set[str], other: set[str], label: str) -> list[str]:
    missing = sorted(reference - other)
    extra = sorted(other - reference)
    print(f"missing boroughs after {label}: {missing if missing else 'None'}")
    if missing or extra:
        print(f"extra boroughs in {label}: {extra if extra else 'None'}")
        print("suggested reason: borough name mismatch or unexpected filtering issue")
    return missing

#Merge EV charging summary data with borough-level Census context
def merge_context(
    borough_summary: pd.DataFrame,
    car_df: pd.DataFrame,
    housing_df: pd.DataFrame,
) -> pd.DataFrame:
    print_step("Merging borough-level EV summary with Census context")
    reference_boroughs = set(borough_summary["borough"])
    missing_ts045 = report_name_differences(reference_boroughs, set(car_df["borough"]), "TS045 merge")
    missing_ts044 = report_name_differences(reference_boroughs, set(housing_df["borough"]), "TS044 merge")

    duplicated = sorted(
        set(borough_summary.loc[borough_summary["borough"].duplicated(), "borough"].tolist())
        | set(car_df.loc[car_df["borough"].duplicated(), "borough"].tolist())
        | set(housing_df.loc[housing_df["borough"].duplicated(), "borough"].tolist())
    )
    print(f"duplicated borough names: {duplicated if duplicated else 'None'}")

    merged = borough_summary.merge(car_df, on="borough", how="left", validate="one_to_one")
    merged = merged.merge(
        housing_df.drop(columns=["lad_code"]),
        on="borough",
        how="left",
        validate="one_to_one",
    )

    # Derived comparison fields used in charts and summary cards
    merged["pct_zero_charger_lsoas"] = 100 - merged["pct_with_charger"]
    merged["chargers_per_1000_car_owning_households"] = (
        merged["total_chargers"] / merged["car_owning_households"] * 1000
    )

    #missing values here would indicate a failed or incomplete merge.
    major_required = [
        "households_total",
        "car_owning_households",
        "households_total_housing",
        "house_households",
        "flat_households",
    ]
    missing_required = [column for column in major_required if merged[column].isna().any()]
    if missing_required:
        raise ValueError(f"Major merge fields contain missing values: {missing_required}")

    if len(merged) != EXPECTED_BOROUGH_COUNT:
        raise ValueError(
            f"Final borough_updated_summary.csv has {len(merged)} rows, expected {EXPECTED_BOROUGH_COUNT}."
        )

    if missing_ts045 or missing_ts044:
        raise ValueError("Major merge issue detected: borough mismatches remain after merging.")

    # TS045 and TS044 describe related but process household totals separately
    difference_series = (merged["households_total"] - merged["households_total_housing"]).abs()
    print(f"maximum difference between TS045 households_total and TS044 households_total_housing: {int(difference_series.max())}")

    merged = merged[
        [
            "borough",
            "inner_outer",
            "total_chargers",
            "total_population",
            "chargers_per_10k",
            "lsoa_count",
            "lsoa_with_charger",
            "pct_with_charger",
            "pct_zero_charger_lsoas",
            "households_total",
            "no_car_households",
            "one_car_households",
            "two_car_households",
            "three_plus_car_households",
            "car_owning_households",
            "pct_no_car_households",
            "pct_car_owning_households",
            "cars_or_vans_minimum_estimate",
            "cars_or_vans_per_100_households_min",
            "chargers_per_1000_car_owning_households",
            "households_total_housing",
            "house_households",
            "flat_households",
            "other_households",
            "pct_houses",
            "pct_flats",
            "pct_other_housing",
        ]
    ].sort_values("borough").reset_index(drop=True)

    merged["pct_zero_charger_lsoas"] = merged["pct_zero_charger_lsoas"].round(2)
    merged["chargers_per_1000_car_owning_households"] = merged[
        "chargers_per_1000_car_owning_households"
    ].round(2)

    merged.to_csv(BOROUGH_UPDATED_OUTPUT_PATH, index=False)
    print(f"final merged borough count: {len(merged)}")
    print(f"Saved: {BOROUGH_UPDATED_OUTPUT_PATH}")
    return merged

#Build a summary row for Whole London, Inner London, or Outer London, to create a compact comparison CSV for higher level charts.
def summarize_area(area_name: str, subset: pd.DataFrame) -> dict:
    total_chargers = int(subset["total_chargers"].sum())
    total_population = int(subset["total_population"].sum())
    lsoa_count = int(subset["lsoa_count"].sum())
    lsoa_with_charger = int(subset["lsoa_with_charger"].sum())
    households_total = int(subset["households_total"].sum())
    no_car_households = int(subset["no_car_households"].sum())
    car_owning_households = int(subset["car_owning_households"].sum())
    households_total_housing = int(subset["households_total_housing"].sum())
    house_households = int(subset["house_households"].sum())
    flat_households = int(subset["flat_households"].sum())

    pct_with_charger = lsoa_with_charger / lsoa_count * 100 if lsoa_count else pd.NA
    pct_zero_charger_lsoas = 100 - pct_with_charger if pd.notna(pct_with_charger) else pd.NA
    pct_no_car_households = no_car_households / households_total * 100 if households_total else pd.NA
    pct_car_owning_households = car_owning_households / households_total * 100 if households_total else pd.NA
    pct_houses = house_households / households_total_housing * 100 if households_total_housing else pd.NA
    pct_flats = flat_households / households_total_housing * 100 if households_total_housing else pd.NA
    chargers_per_10k = total_chargers / total_population * 10000 if total_population else pd.NA
    chargers_per_1000_car_owning_households = (
        total_chargers / car_owning_households * 1000 if car_owning_households else pd.NA
    )

    return {
        "area": area_name,
        "total_chargers": total_chargers,
        "total_population": total_population,
        "lsoa_count": lsoa_count,
        "lsoa_with_charger": lsoa_with_charger,
        "pct_with_charger": round(float(pct_with_charger), 2),
        "pct_zero_charger_lsoas": round(float(pct_zero_charger_lsoas), 2),
        "households_total": households_total,
        "no_car_households": no_car_households,
        "car_owning_households": car_owning_households,
        "pct_no_car_households": round(float(pct_no_car_households), 2),
        "pct_car_owning_households": round(float(pct_car_owning_households), 2),
        "households_total_housing": households_total_housing,
        "house_households": house_households,
        "flat_households": flat_households,
        "pct_houses": round(float(pct_houses), 2),
        "pct_flats": round(float(pct_flats), 2),
        "chargers_per_10k": round(float(chargers_per_10k), 2),
        "chargers_per_1000_car_owning_households": round(
            float(chargers_per_1000_car_owning_households), 2
        ),
    }

#Create whole-city and Inner/Outer London comparison summaries
def create_inner_outer_updated_summary(merged: pd.DataFrame) -> pd.DataFrame:
    print_step("Creating Whole London / Inner London / Outer London summary")
    summary = pd.DataFrame(
        [
            summarize_area("Whole London", merged),
            summarize_area("Inner London", merged[merged["inner_outer"] == "Inner"]),
            summarize_area("Outer London", merged[merged["inner_outer"] == "Outer"]),
        ]
    )
    summary.to_csv(INNER_OUTER_UPDATED_OUTPUT_PATH, index=False)
    print(f"Saved: {INNER_OUTER_UPDATED_OUTPUT_PATH}")
    return summary

def main() -> None:
    borough_summary, borough_names, lookup = load_borough_summary()
    car_ownership = process_ts045(borough_names, lookup)
    housing_type = process_ts044(borough_names, lookup)
    merged = merge_context(borough_summary, car_ownership, housing_type)
    create_inner_outer_updated_summary(merged)

    print_step("Output file paths")
    print(CAR_OUTPUT_PATH)
    print(HOUSING_OUTPUT_PATH)
    print(BOROUGH_UPDATED_OUTPUT_PATH)
    print(INNER_OUTER_UPDATED_OUTPUT_PATH)


if __name__ == "__main__":
    main()