'''
Demand indicators processing script

This script adds demand-context and housing-context indicators to the existing LSOA EV charging dataset.

It reads:
- raw TS045 car or van availability data
- raw TS044 accommodation type data
- the existing lsoa_ev.geojson master dataset

It outputs:
- cleaned TS045 and TS044 CSV tables
- an extended lsoa_ev_plus_demand.geojson file
- Inner/Outer London and borough summary CSV files
- an optional exploratory priority score CSV
- a short processing report for methodology and reproducibility

Important cautions:
- car ownership is used as a demand-context proxy, not EV ownership
- housing type is used as a home-charging constraint proxy
'''




from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Define project paths and output file locations

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

LSOA_EV_PATH = DATA_DIR / "lsoa_ev.geojson"

TS045_OUTPUT_PATH = DATA_DIR / "ts045_car_ownership_clean.csv"
TS044_OUTPUT_PATH = DATA_DIR / "ts044_housing_clean.csv"
LSOA_PLUS_DEMAND_OUTPUT_PATH = DATA_DIR / "lsoa_ev_plus_demand.geojson"
INNER_OUTER_SUMMARY_OUTPUT_PATH = DATA_DIR / "inner_outer_summary.csv"
BOROUGH_DEMAND_SUMMARY_OUTPUT_PATH = DATA_DIR / "borough_demand_summary.csv"
PRIORITY_LSOAS_OUTPUT_PATH = DATA_DIR / "priority_lsoas.csv"


# Small helper functions for printing, text cleaning and safe division
def print_step(message: str) -> None:
    print(f"\n{message}")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalized_key(value: str) -> str:
    text = normalize_text(value).lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float, multiplier: float = 1.0):
    numerator_series = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator.copy()
    denominator_series = (
        pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator.copy()
    )

    denominator_series = denominator_series.where(denominator_series.notna() & (denominator_series != 0))
    result = numerator_series / denominator_series * multiplier
    return result.replace([float("inf"), float("-inf")], pd.NA)


def find_value_column(columns: list[str]) -> str:
    value_priority = [
        "observation",
        "count",
        "value",
        "obs_value",
    ]
    normalized = {column: normalized_key(column) for column in columns}

    for target in value_priority:
        for column, key in normalized.items():
            if target == key or key.endswith(f" {target}") or target in key:
                return column

    raise ValueError(f"Could not identify observation/value column. Available columns: {columns}")


# Find the TS045 and TS044 Census files and detect key columns
# This section helps the script work with common ONS long-format files
def detect_long_format_columns(df: pd.DataFrame, dataset_type: str) -> dict[str, str | None]:
    columns = list(df.columns)
    normalized = {column: normalized_key(column) for column in columns}

    if dataset_type == "ts045":
        category_hint = "car or van availability"
    elif dataset_type == "ts044":
        category_hint = "accommodation type"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    code_column = None
    name_column = None
    category_column = None

    for column, key in normalized.items():
        if "lower layer super output areas code" in key or key in {"lsoa code", "lsoa21cd"}:
            code_column = column
            break

    if code_column is None:
        for column in columns:
            sample = df[column].dropna().astype(str).head(30)
            if not sample.empty and sample.str.match(r"^E01\d{6}$").all():
                code_column = column
                break

    for column, key in normalized.items():
        if column == code_column:
            continue
        if "lower layer super output areas" in key or key in {"lsoa name", "lsoa21nm"}:
            name_column = column
            break

    preferred_category_columns = []
    fallback_category_columns = []
    for column, key in normalized.items():
        if category_hint in key:
            if " code" in key:
                fallback_category_columns.append(column)
            else:
                preferred_category_columns.append(column)

    if preferred_category_columns:
        category_column = preferred_category_columns[0]
    elif fallback_category_columns:
        category_column = fallback_category_columns[0]

    value_column = find_value_column(columns)

    missing = []
    if code_column is None:
        missing.append("LSOA code column")
    if category_column is None:
        missing.append("category column")
    if value_column is None:
        missing.append("value column")

    if missing:
        raise ValueError(
            f"Could not identify {', '.join(missing)} for {dataset_type}. Available columns: {columns}"
        )

    return {
        "code": code_column,
        "name": name_column,
        "category": category_column,
        "value": value_column,
    }


def score_candidate_file(path: Path, dataset_type: str) -> tuple[int, dict[str, str | None] | None, pd.DataFrame | None]:
    try:
        preview = pd.read_csv(path, nrows=200)
    except Exception:
        return -1, None, None

    try:
        detected = detect_long_format_columns(preview, dataset_type)
    except Exception:
        return -1, None, None

    score = 0
    code_column = detected["code"]
    category_column = detected["category"]
    if code_column and "lower layer super output areas" in normalized_key(code_column):
        score += 5
    if code_column and not preview[code_column].dropna().empty:
        sample = preview[code_column].dropna().astype(str).head(30)
        if sample.str.match(r"^E01\d{6}$").all():
            score += 5
    if category_column:
        score += 3
    if dataset_type in normalized_key(path.stem):
        score += 2
    if "ldn" in normalized_key(path.stem):
        score += 1

    return score, detected, preview


def auto_select_census_file(dataset_type: str) -> tuple[Path, dict[str, str | None]]:
    print_step(f"Selecting source file for {dataset_type.upper()}")
    candidates = sorted(DATA_DIR.rglob("*.csv"))
    best_path = None
    best_detected = None
    best_score = -1

    for path in candidates:
        score, detected, _preview = score_candidate_file(path, dataset_type)
        if score > best_score:
            best_score = score
            best_path = path
            best_detected = detected

    if best_path is None or best_detected is None or best_score < 0:
        raise ValueError(f"Could not automatically find a suitable CSV for {dataset_type} under {DATA_DIR}")

    print(f"Selected {dataset_type.upper()} file: {best_path}")
    print(f"Detected columns: {best_detected}")
    return best_path, best_detected


def add_lsoa_code_and_name(df: pd.DataFrame, detected: dict[str, str | None]) -> pd.DataFrame:
    code_column = detected["code"]
    name_column = detected["name"]
    value_column = detected["value"]
    category_column = detected["category"]

    working = df.copy()
    working["lsoa_code"] = working[code_column].astype("string").map(normalize_text)
    if name_column:
        working["lsoa_name"] = working[name_column].astype("string").map(normalize_text)
    else:
        working["lsoa_name"] = pd.NA
    working["category_raw"] = working[category_column].astype("string").map(normalize_text)
    working["value_raw"] = pd.to_numeric(working[value_column], errors="coerce")
    return working


# Clean TS045 car ownership data
#This turns the raw Census car/van availability table into LSOA-level car ownership indicators
def process_ts045_car_ownership(path: Path, detected: dict[str, str | None]) -> pd.DataFrame:
    print_step("Cleaning TS045 car or van availability")
    raw = pd.read_csv(path)
    print(f"TS045 input row count: {len(raw)}")

    working = add_lsoa_code_and_name(raw, detected)
    working = working[~working["category_raw"].str.contains("does not apply", case=False, na=False)].copy()

    rules = [
        ("households_no_car", lambda text: "no cars or vans" in text),
        ("households_1_car", lambda text: "1 car" in text),
        ("households_2_cars", lambda text: "2 cars" in text),
        ("households_3plus_cars", lambda text: "3 or more" in text),
    ]

    working["category_clean"] = working["category_raw"].map(
        lambda value: next((name for name, rule in rules if rule(normalized_key(value))), None)
    )

    unmatched = sorted(working.loc[working["category_clean"].isna(), "category_raw"].dropna().unique().tolist())
    if unmatched:
        print("Available TS045 categories:")
        for category in sorted(working["category_raw"].dropna().unique().tolist()):
            print(f"- {category}")
        raise ValueError(f"TS045 required category could not be matched. Unmatched categories: {unmatched}")

    pivot = (
        working.pivot_table(
            index="lsoa_code",
            columns="category_clean",
            values="value_raw",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    required_fields = [
        "households_no_car",
        "households_1_car",
        "households_2_cars",
        "households_3plus_cars",
    ]
    for field in required_fields:
        if field not in pivot.columns:
            raise ValueError(f"TS045 cleaned table is missing required field: {field}")
        pivot[field] = pivot[field].round().astype("Int64")

    pivot["households_total"] = pivot[required_fields].sum(axis=1)
    pivot["car_owning_households"] = (
        pivot["households_1_car"] + pivot["households_2_cars"] + pivot["households_3plus_cars"]
    )
    pivot["pct_no_car_households"] = safe_divide(
        pivot["households_no_car"], pivot["households_total"], 100
    )
    pivot["pct_car_owning_households"] = safe_divide(
        pivot["car_owning_households"], pivot["households_total"], 100
    )

    output = pivot[
        [
            "lsoa_code",
            "households_total",
            "households_no_car",
            "households_1_car",
            "households_2_cars",
            "households_3plus_cars",
            "car_owning_households",
            "pct_no_car_households",
            "pct_car_owning_households",
        ]
    ].sort_values("lsoa_code").reset_index(drop=True)

    output.to_csv(TS045_OUTPUT_PATH, index=False)
    print(f"Saved: {TS045_OUTPUT_PATH}")
    return output


# Clean TS044 housing type data
# This creates LSOA-level indicators for flats/apartments and houses
def process_ts044_housing(path: Path, detected: dict[str, str | None]) -> pd.DataFrame:
    print_step("Cleaning TS044 accommodation type")
    raw = pd.read_csv(path)
    print(f"TS044 input row count: {len(raw)}")

    working = add_lsoa_code_and_name(raw, detected)
    working = working[~working["category_raw"].str.contains("does not apply", case=False, na=False)].copy()

    def classify_housing(value: str) -> str | None:
        text = normalized_key(value)
        if "semi detached" in text:
            return "households_semi_detached"
        if "detached" in text:
            return "households_detached"
        if "terraced" in text:
            return "households_terraced"
        flat_terms = [
            "flat",
            "maisonette",
            "apartment",
            "tenement",
            "converted",
            "shared house",
            "commercial building",
        ]
        if any(term in text for term in flat_terms):
            return "households_flats_apartments"
        if text:
            return "households_other_accommodation"
        return None

    working["category_clean"] = working["category_raw"].map(classify_housing)

    unmatched = sorted(working.loc[working["category_clean"].isna(), "category_raw"].dropna().unique().tolist())
    if unmatched:
        print("Available TS044 categories:")
        for category in sorted(working["category_raw"].dropna().unique().tolist()):
            print(f"- {category}")
        raise ValueError(f"TS044 required category could not be matched. Unmatched categories: {unmatched}")

    pivot = (
        working.pivot_table(
            index="lsoa_code",
            columns="category_clean",
            values="value_raw",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    required_fields = [
        "households_detached",
        "households_semi_detached",
        "households_terraced",
        "households_flats_apartments",
        "households_other_accommodation",
    ]
    for field in required_fields:
        if field not in pivot.columns:
            pivot[field] = 0
        pivot[field] = pivot[field].round().astype("Int64")

    houses = (
        pivot["households_detached"]
        + pivot["households_semi_detached"]
        + pivot["households_terraced"]
    )
    pivot["households_total_housing"] = houses + pivot["households_flats_apartments"] + pivot["households_other_accommodation"]
    pivot["pct_flats_apartments"] = safe_divide(
        pivot["households_flats_apartments"], pivot["households_total_housing"], 100
    )
    pivot["pct_houses"] = safe_divide(houses, pivot["households_total_housing"], 100)

    output = pivot[
        [
            "lsoa_code",
            "households_total_housing",
            "households_detached",
            "households_semi_detached",
            "households_terraced",
            "households_flats_apartments",
            "households_other_accommodation",
            "pct_flats_apartments",
            "pct_houses",
        ]
    ].sort_values("lsoa_code").reset_index(drop=True)

    output.to_csv(TS044_OUTPUT_PATH, index=False)
    print(f"Saved: {TS044_OUTPUT_PATH}")
    return output

#Merge car ownership and housing indicators into the EV LSOA dataset
def merge_demand_into_lsoa(ts045_clean: pd.DataFrame, ts044_clean: pd.DataFrame) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    print_step("Merging demand indicators into lsoa_ev.geojson")
    lsoa_gdf = gpd.read_file(LSOA_EV_PATH)
    original_columns = list(lsoa_gdf.columns)
    lsoa_count = len(lsoa_gdf)
    print(f"LSOA input row count: {lsoa_count}")

    merged = lsoa_gdf.merge(ts045_clean, left_on="LSOA21CD", right_on="lsoa_code", how="left")
    merged = merged.merge(ts044_clean, on="lsoa_code", how="left")

    matched_ts045 = int(merged["households_total"].notna().sum())
    matched_ts044 = int(merged["households_total_housing"].notna().sum())
    unmatched_either = int(
        (~merged["households_total"].notna() | ~merged["households_total_housing"].notna()).sum()
    )

    match_rate_ts045 = matched_ts045 / lsoa_count * 100 if lsoa_count else 0
    match_rate_ts044 = matched_ts044 / lsoa_count * 100 if lsoa_count else 0

    print(f"Matched to TS045: {matched_ts045} / {lsoa_count} ({match_rate_ts045:.2f}%)")
    print(f"Matched to TS044: {matched_ts044} / {lsoa_count} ({match_rate_ts044:.2f}%)")
    print(f"Number unmatched (either TS045 or TS044): {unmatched_either}")

    if match_rate_ts045 < 95 or match_rate_ts044 < 95:
        print("WARNING: Match rate below 95%. Please inspect lsoa_code alignment carefully.")

    projected = merged.to_crs(epsg=27700)
    merged["area_km2"] = projected.geometry.area / 1_000_000

    merged["car_owning_households"] = merged["car_owning_households"].astype("Float64")
    merged["charger_count"] = pd.to_numeric(merged["charger_count"], errors="coerce")
    merged["population"] = pd.to_numeric(merged["population"], errors="coerce")

    merged["chargers_per_1000_car_owning_households"] = safe_divide(
        merged["charger_count"], merged["car_owning_households"], 1000
    )
    merged["zero_charger_lsoa"] = merged["charger_count"].fillna(0).eq(0)
    merged["chargers_per_km2"] = safe_divide(merged["charger_count"], merged["area_km2"], 1)

    numeric_columns = [
        "households_total",
        "households_no_car",
        "households_1_car",
        "households_2_cars",
        "households_3plus_cars",
        "car_owning_households",
        "pct_no_car_households",
        "pct_car_owning_households",
        "households_total_housing",
        "households_detached",
        "households_semi_detached",
        "households_terraced",
        "households_flats_apartments",
        "households_other_accommodation",
        "pct_flats_apartments",
        "pct_houses",
        "chargers_per_1000_car_owning_households",
        "area_km2",
        "chargers_per_km2",
    ]
    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").replace(
                [float("inf"), float("-inf")], pd.NA
            )

    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=lsoa_gdf.crs)
    merged.to_file(LSOA_PLUS_DEMAND_OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved: {LSOA_PLUS_DEMAND_OUTPUT_PATH}")

    return merged, {
        "selected_lsoa_rows": lsoa_count,
        "matched_ts045": matched_ts045,
        "matched_ts044": matched_ts044,
        "unmatched": unmatched_either,
        "match_rate_ts045": match_rate_ts045,
        "match_rate_ts044": match_rate_ts044,
        "original_columns": len(original_columns),
    }


# Calculate group summary values from aggregated totals
# This is used for Inner/Outer and borough-level summary outputs.
def summarize_group(df: pd.DataFrame, group_name: str | None = None) -> dict:
    total_population = pd.to_numeric(df["population"], errors="coerce").sum()
    total_chargers = pd.to_numeric(df["charger_count"], errors="coerce").sum()
    lsoa_count = len(df)
    lsoas_with_charger = int((pd.to_numeric(df["charger_count"], errors="coerce").fillna(0) > 0).sum())
    zero_charger_count = int(df["zero_charger_lsoa"].fillna(False).sum())
    households_total = pd.to_numeric(df["households_total"], errors="coerce").sum()
    car_owning_households = pd.to_numeric(df["car_owning_households"], errors="coerce").sum()
    no_car_households = pd.to_numeric(df["households_no_car"], errors="coerce").sum()
    households_total_housing = pd.to_numeric(df["households_total_housing"], errors="coerce").sum()
    flats_households = pd.to_numeric(df["households_flats_apartments"], errors="coerce").sum()
    houses_households = (
        pd.to_numeric(df["households_detached"], errors="coerce").sum()
        + pd.to_numeric(df["households_semi_detached"], errors="coerce").sum()
        + pd.to_numeric(df["households_terraced"], errors="coerce").sum()
    )
    total_area_km2 = pd.to_numeric(df["area_km2"], errors="coerce").sum()

    result = {
        "lsoa_count": lsoa_count,
        "total_population": int(total_population),
        "total_chargers": int(total_chargers),
        "weighted_chargers_per_10k": round(total_chargers / total_population * 10000, 2) if total_population else pd.NA,
        "mean_chargers_per_10k": round(pd.to_numeric(df["chargers_per_10k"], errors="coerce").mean(), 2),
        "median_chargers_per_10k": round(pd.to_numeric(df["chargers_per_10k"], errors="coerce").median(), 2),
        "pct_lsoas_with_charger": round(lsoas_with_charger / lsoa_count * 100, 2) if lsoa_count else pd.NA,
        "pct_zero_charger_lsoas": round(zero_charger_count / lsoa_count * 100, 2) if lsoa_count else pd.NA,
        "households_total": int(households_total),
        "car_owning_households": int(car_owning_households),
        "pct_car_owning_households": round(car_owning_households / households_total * 100, 2) if households_total else pd.NA,
        "pct_no_car_households": round(no_car_households / households_total * 100, 2) if households_total else pd.NA,
        "households_total_housing": int(households_total_housing),
        "pct_flats_apartments": round(flats_households / households_total_housing * 100, 2) if households_total_housing else pd.NA,
        "pct_houses": round(houses_households / households_total_housing * 100, 2) if households_total_housing else pd.NA,
        "chargers_per_1000_car_owning_households": round(total_chargers / car_owning_households * 1000, 2) if car_owning_households else pd.NA,
        "chargers_per_km2": round(total_chargers / total_area_km2, 2) if total_area_km2 else pd.NA,
    }

    if group_name is not None:
        result["group_name"] = group_name

    return result


#Create Inner vs Outer London summary table
def create_inner_outer_summary(merged: gpd.GeoDataFrame) -> pd.DataFrame:
    print_step("Creating inner / outer summary")
    rows = []
    for zone in ["Inner", "Outer"]:
        subset = merged[merged["inner_outer"] == zone].copy()
        row = summarize_group(subset, zone)
        rows.append(
            {
                "inner_outer": zone,
                **{key: value for key, value in row.items() if key != "group_name"},
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(INNER_OUTER_SUMMARY_OUTPUT_PATH, index=False)
    print(f"Saved: {INNER_OUTER_SUMMARY_OUTPUT_PATH}")
    return summary


# Create borough-level demand summary table
def create_borough_demand_summary(merged: gpd.GeoDataFrame) -> pd.DataFrame:
    print_step("Creating borough demand summary")
    rows = []
    for borough, subset in merged.groupby("borough", dropna=False):
        total_population = pd.to_numeric(subset["population"], errors="coerce").sum()
        total_chargers = pd.to_numeric(subset["charger_count"], errors="coerce").sum()
        lsoa_count = len(subset)
        lsoas_with_charger = int((pd.to_numeric(subset["charger_count"], errors="coerce").fillna(0) > 0).sum())
        zero_charger_count = int(subset["zero_charger_lsoa"].fillna(False).sum())
        households_total = pd.to_numeric(subset["households_total"], errors="coerce").sum()
        car_owning_households = pd.to_numeric(subset["car_owning_households"], errors="coerce").sum()
        no_car_households = pd.to_numeric(subset["households_no_car"], errors="coerce").sum()
        households_total_housing = pd.to_numeric(subset["households_total_housing"], errors="coerce").sum()
        flats_households = pd.to_numeric(subset["households_flats_apartments"], errors="coerce").sum()
        houses_households = (
            pd.to_numeric(subset["households_detached"], errors="coerce").sum()
            + pd.to_numeric(subset["households_semi_detached"], errors="coerce").sum()
            + pd.to_numeric(subset["households_terraced"], errors="coerce").sum()
        )
        total_area_km2 = pd.to_numeric(subset["area_km2"], errors="coerce").sum()

        rows.append(
            {
                "borough": borough,
                "inner_outer": subset["inner_outer"].mode(dropna=True).iat[0] if not subset["inner_outer"].mode(dropna=True).empty else pd.NA,
                "lsoa_count": lsoa_count,
                "total_population": int(total_population),
                "total_chargers": int(total_chargers),
                "chargers_per_10k": round(total_chargers / total_population * 10000, 2) if total_population else pd.NA,
                "pct_lsoas_with_charger": round(lsoas_with_charger / lsoa_count * 100, 2) if lsoa_count else pd.NA,
                "pct_zero_charger_lsoas": round(zero_charger_count / lsoa_count * 100, 2) if lsoa_count else pd.NA,
                "households_total": int(households_total),
                "car_owning_households": int(car_owning_households),
                "pct_car_owning_households": round(car_owning_households / households_total * 100, 2) if households_total else pd.NA,
                "pct_no_car_households": round(no_car_households / households_total * 100, 2) if households_total else pd.NA,
                "households_total_housing": int(households_total_housing),
                "pct_flats_apartments": round(flats_households / households_total_housing * 100, 2) if households_total_housing else pd.NA,
                "pct_houses": round(houses_households / households_total_housing * 100, 2) if households_total_housing else pd.NA,
                "chargers_per_1000_car_owning_households": round(total_chargers / car_owning_households * 1000, 2) if car_owning_households else pd.NA,
                "chargers_per_km2": round(total_chargers / total_area_km2, 2) if total_area_km2 else pd.NA,
            }
        )

    summary = pd.DataFrame(rows).sort_values("borough").reset_index(drop=True)
    summary.to_csv(BOROUGH_DEMAND_SUMMARY_OUTPUT_PATH, index=False)
    print(f"Saved: {BOROUGH_DEMAND_SUMMARY_OUTPUT_PATH}")
    return summary

# Create exploratory priority score
# This is only an exploratory planning indicator
def percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(pct=True, method="average")
    if invert:
        ranked = 1 - ranked
    return ranked


def create_priority_lsoas(merged: gpd.GeoDataFrame) -> pd.DataFrame | None:
    print_step("Creating exploratory priority score")
    required_base = ["LSOA21CD", "LSOA21NM", "borough", "inner_outer", "population", "charger_count", "chargers_per_10k"]
    missing_base = [column for column in required_base if column not in merged.columns]
    if missing_base:
        print(f"Skipping priority score because required fields are missing: {missing_base}")
        return None

    priority = merged.copy()

    components = []
    if priority["chargers_per_1000_car_owning_households"].notna().any():
        components.append(percentile_rank(priority["chargers_per_1000_car_owning_households"], invert=True).rename("need_low_provision"))
    else:
        components.append(percentile_rank(priority["chargers_per_10k"], invert=True).rename("need_low_provision"))

    if "pct_car_owning_households" in priority.columns and priority["pct_car_owning_households"].notna().any():
        components.append(percentile_rank(priority["pct_car_owning_households"]).rename("need_car_ownership"))

    if "pct_flats_apartments" in priority.columns and priority["pct_flats_apartments"].notna().any():
        components.append(percentile_rank(priority["pct_flats_apartments"]).rename("need_home_constraint"))

    components.append(percentile_rank(priority["population"]).rename("need_population"))

    if "imd_decile" in priority.columns and priority["imd_decile"].notna().any():
        deprivation_context = 11 - pd.to_numeric(priority["imd_decile"], errors="coerce")
        components.append(percentile_rank(deprivation_context).rename("need_deprivation_context"))

    if not components:
        print("Skipping priority score because no scoring components were available.")
        return None

    component_df = pd.concat(components, axis=1)

    # This is an exploratory planning indicator
    priority["priority_score"] = (component_df.mean(axis=1, skipna=True) * 100).round(2)

    output_columns = [
        "LSOA21CD",
        "LSOA21NM",
        "borough",
        "inner_outer",
        "population",
        "charger_count",
        "chargers_per_10k",
        "pct_car_owning_households",
        "pct_flats_apartments",
        "imd_decile",
        "priority_score",
    ]
    available_columns = [column for column in output_columns if column in priority.columns]
    output = priority[available_columns].sort_values("priority_score", ascending=False).reset_index(drop=True)
    output.to_csv(PRIORITY_LSOAS_OUTPUT_PATH, index=False)
    print(f"Saved: {PRIORITY_LSOAS_OUTPUT_PATH}")
    return output

# Print key summart
def print_key_summary(inner_outer_summary: pd.DataFrame) -> None:
    print_step("Key London / Inner / Outer summary numbers")
    for _, row in inner_outer_summary.iterrows():
        print(
            f"{row['inner_outer']}: "
            f"LSOAs={int(row['lsoa_count'])}, "
            f"chargers={int(row['total_chargers'])}, "
            f"weighted_chargers_per_10k={row['weighted_chargers_per_10k']}, "
            f"pct_zero_charger_lsoas={row['pct_zero_charger_lsoas']}"
        )



def main() -> None:
    ts045_path, ts045_detected = auto_select_census_file("ts045")
    ts044_path, ts044_detected = auto_select_census_file("ts044")

    ts045_clean = process_ts045_car_ownership(ts045_path, ts045_detected)
    ts044_clean = process_ts044_housing(ts044_path, ts044_detected)

    merged_gdf, match_info = merge_demand_into_lsoa(ts045_clean, ts044_clean)
    inner_outer_summary = create_inner_outer_summary(merged_gdf)
    create_borough_demand_summary(merged_gdf)
    priority_output = create_priority_lsoas(merged_gdf)

    generated_outputs = [
        TS045_OUTPUT_PATH,
        TS044_OUTPUT_PATH,
        LSOA_PLUS_DEMAND_OUTPUT_PATH,
        INNER_OUTER_SUMMARY_OUTPUT_PATH,
        BOROUGH_DEMAND_SUMMARY_OUTPUT_PATH,
    ]
    if priority_output is not None:
        generated_outputs.append(PRIORITY_LSOAS_OUTPUT_PATH)

    print_key_summary(inner_outer_summary)

    print_step("Output file paths")
    for output_path in generated_outputs:
        print(output_path)


if __name__ == "__main__":
    main()