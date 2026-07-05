import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from src.ai_indicator import add_ai_indicator
from src.config import FIGURE_DIR
from src.data_io import ensure_output_dirs, load_economic_outcomes, load_job_postings, save_table


def aggregate_ai_indicator(postings: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["company", "sector", "country", "region", "year"]
    salary_agg = ("posted_salary", "mean") if "posted_salary" in postings.columns else ("ai_score", "mean")
    aggregated = (
        postings.groupby(group_cols, dropna=False)
        .agg(
            total_postings=("posting_id", "count"),
            ai_postings=("ai_posting", "sum"),
            avg_ai_score=("ai_score", "mean"),
            avg_posted_salary=salary_agg,
        )
        .reset_index()
    )
    aggregated["ai_adoption_rate"] = aggregated["ai_postings"] / aggregated["total_postings"]
    return aggregated.sort_values(["year", "country", "sector", "company"])


def build_sector_country_panel(aggregated: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    panel_ai = (
        aggregated.groupby(["sector", "country", "region", "year"], dropna=False)
        .agg(
            total_postings=("total_postings", "sum"),
            ai_postings=("ai_postings", "sum"),
            avg_ai_score=("avg_ai_score", "mean"),
            avg_posted_salary=("avg_posted_salary", "mean"),
        )
        .reset_index()
    )
    panel_ai["ai_adoption_rate"] = panel_ai["ai_postings"] / panel_ai["total_postings"]

    if {"region", "year"}.issubset(outcomes.columns):
        panel = panel_ai.merge(outcomes, on=["region", "year"], how="left")
        if "country_x" in panel.columns:
            panel = panel.rename(columns={"country_x": "country"})
            panel = panel.drop(columns=[col for col in ["country_y"] if col in panel.columns])
    else:
        panel = panel_ai.merge(outcomes, on=["sector", "country", "year"], how="left")
    return panel.sort_values(["year", "country", "sector"])


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        col
        for col in [
            "avg_posted_salary",
            "median_household_income",
            "per_capita_income",
            "gdp_per_capita",
            "state_population",
            "employment",
            "avg_wage",
            "productivity_index",
            "funding_usd_mn",
        ]
        if col in panel.columns
    ]
    rows = []
    regression_data = panel.dropna(subset=["ai_adoption_rate"]).copy()

    for outcome in outcomes:
        data = regression_data.dropna(subset=[outcome])
        if len(data) < 6 or data["ai_adoption_rate"].nunique() < 2:
            rows.append(
                {
                    "outcome": outcome,
                    "coef_ai_adoption_rate": None,
                    "p_value": None,
                    "r_squared": None,
                    "n_obs": len(data),
                    "note": "Not enough variation for regression",
                }
            )
            continue

        if outcome == "avg_posted_salary":
            formula = f"{outcome} ~ ai_adoption_rate + C(sector) + C(region) + C(year)"
            note = "OLS with sector, region, and year fixed effects"
        else:
            formula = f"{outcome} ~ ai_adoption_rate + C(sector) + C(year)"
            note = "OLS with sector and year fixed effects; state outcomes vary at region level"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")

        rows.append(
            {
                "outcome": outcome,
                "coef_ai_adoption_rate": model.params.get("ai_adoption_rate"),
                "p_value": model.pvalues.get("ai_adoption_rate"),
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def make_figures(panel: pd.DataFrame) -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    sector = (
        panel.groupby("sector", as_index=False)["ai_adoption_rate"]
        .mean()
        .sort_values("ai_adoption_rate", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=sector, x="ai_adoption_rate", y="sector", color="#2F80ED")
    plt.xlabel("Average AI adoption rate")
    plt.ylabel("Sector")
    plt.title("AI Adoption in Job Postings by Sector")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ai_adoption_by_sector.png", dpi=180)
    plt.close()

    country = (
        panel.groupby("region", as_index=False)["ai_adoption_rate"]
        .mean()
        .sort_values("ai_adoption_rate", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=country, x="region", y="ai_adoption_rate", color="#27AE60")
    plt.xlabel("State/region")
    plt.ylabel("Average AI adoption rate")
    plt.title("AI Adoption in Job Postings by Country")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ai_adoption_by_country.png", dpi=180)
    plt.close()

    y_col = "gdp_per_capita" if "gdp_per_capita" in panel.columns else "productivity_index"
    scatter_data = panel.dropna(subset=[y_col, "ai_adoption_rate"])
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=scatter_data,
        x="ai_adoption_rate",
        y=y_col,
        scatter_kws={"s": 70, "alpha": 0.75},
        line_kws={"color": "#EB5757"},
    )
    plt.xlabel("AI adoption rate")
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title("AI Adoption and External Economic Outcomes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ai_vs_productivity.png", dpi=180)
    plt.close()


def run_pipeline() -> None:
    ensure_output_dirs()

    postings = load_job_postings()
    outcomes = load_economic_outcomes()

    postings_with_ai = add_ai_indicator(postings)
    aggregated = aggregate_ai_indicator(postings_with_ai)
    panel = build_sector_country_panel(aggregated, outcomes)
    regression_results = run_regressions(panel)

    save_table(postings_with_ai, "job_postings_with_ai_indicator.csv")
    save_table(aggregated, "ai_adoption_by_company_sector_country_year.csv")
    save_table(panel, "merged_ai_economic_panel.csv")
    save_table(regression_results, "regression_results.csv")
    make_figures(panel)

    print("Analysis complete.")
    print("Tables saved in outputs/tables/")
    print("Figures saved in outputs/figures/")
