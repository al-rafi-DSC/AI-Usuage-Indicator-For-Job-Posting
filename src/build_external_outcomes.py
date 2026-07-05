import re
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, STATE_ECONOMIC_OUTCOMES_FILE


STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Washington, D.C.": "DC",
    "District of Columbia": "DC",
}


def clean_money(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\[[^\]]+\]", "", str(value))
    text = text.replace("$", "").replace(",", "").strip()
    return pd.to_numeric(text, errors="coerce")


def build_state_outcomes(income_html: Path, gdp_html: Path) -> pd.DataFrame:
    income_tables = pd.read_html(income_html)
    household = income_tables[1][["States and D.C.", "2023"]].copy()
    household.columns = ["state_name", "median_household_income"]
    household["median_household_income"] = household["median_household_income"].apply(clean_money)

    per_capita = income_tables[4][["States and D.C.", "2023"]].copy()
    per_capita.columns = ["state_name", "per_capita_income"]
    per_capita["state_name"] = per_capita["state_name"].str.replace(r"\[[^\]]+\]", "", regex=True)
    per_capita["per_capita_income"] = per_capita["per_capita_income"].apply(clean_money)

    gdp = pd.read_html(gdp_html)[1].copy()
    gdp.columns = ["_".join([str(part) for part in col if str(part) != "nan"]).strip("_") for col in gdp.columns]
    gdp = gdp.rename(
        columns={
            "State or federal district_State or federal district": "state_name",
            "Population (2025)[2]_Population (2025)[2]": "state_population",
            "Nominal GDP per capita_2024": "gdp_per_capita",
        }
    )
    gdp = gdp[["state_name", "state_population", "gdp_per_capita"]]
    gdp["gdp_per_capita"] = gdp["gdp_per_capita"].apply(clean_money)
    gdp["state_population"] = pd.to_numeric(gdp["state_population"], errors="coerce")

    outcomes = household.merge(per_capita, on="state_name", how="inner").merge(gdp, on="state_name", how="left")
    outcomes["region"] = outcomes["state_name"].map(STATE_TO_ABBR)
    outcomes = outcomes.dropna(subset=["region"])
    outcomes["year"] = 2023
    outcomes["country"] = "United States"
    outcomes["source"] = "Wikipedia tables citing ACS income and BEA state GDP"
    return outcomes[
        [
            "region",
            "state_name",
            "country",
            "year",
            "median_household_income",
            "per_capita_income",
            "gdp_per_capita",
            "state_population",
            "source",
        ]
    ]


if __name__ == "__main__":
    outcomes = build_state_outcomes(Path("work/income.html"), Path("work/gdp.html"))
    DATA_DIR.mkdir(exist_ok=True)
    outcomes.to_csv(STATE_ECONOMIC_OUTCOMES_FILE, index=False)
    print(f"Wrote {len(outcomes)} rows to {STATE_ECONOMIC_OUTCOMES_FILE}")
