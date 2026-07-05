# Lightcast AI Job Postings Final Project

This repository contains a reproducible analysis pipeline for the University of Milan Lightcast Laboratory final project. It builds an AI usage indicator from real job postings, aggregates it by company, sector, geography, and year, and explores its relationship with external economic outcomes.

The current empirical version uses public LinkedIn job postings from Hugging Face as a substitute for restricted Lightcast data. The code is also compatible with a future Lightcast export if the same core fields are provided.

## Data

Main job postings dataset:

- `data/job_postings.csv`
- Source: `xanderios/linkedin-job-postings`
- URL: https://huggingface.co/datasets/xanderios/linkedin-job-postings
- License tag: MIT
- Repository file: GitHub-friendly real subset of 10,028 postings selected from the full public dataset

External economic outcomes:

- `data/us_state_economic_outcomes.csv`
- Built from public state income and GDP tables
- Variables include median household income, per-capita income, GDP per capita, and population

More details are in `DATA_SOURCES.md`.

## Project Structure

```text
.
|-- data/
|   |-- job_postings.csv
|   |-- us_state_economic_outcomes.csv
|   |-- job_postings_sample.csv
|   `-- economic_outcomes_sample.csv
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- src/
|   |-- ai_indicator.py
|   |-- analysis.py
|   |-- build_external_outcomes.py
|   |-- config.py
|   `-- data_io.py
|-- tests/
|   `-- test_ai_indicator.py
|-- DATA_SOURCES.md
|-- main.py
|-- requirements.txt
`-- README.md
```

## What the Pipeline Does

1. Loads and normalizes the real LinkedIn job-postings dataset.
2. Infers broad sector groups from job title, description, and skills text.
3. Builds an AI usage indicator using NLP-style weighted keyword matching.
4. Aggregates AI adoption by company, sector, U.S. state, country, and year.
5. Merges the aggregated postings panel with external state economic indicators.
6. Runs baseline OLS models and exports presentation-ready tables and charts.

## Run the Analysis

```bash
pip install -r requirements.txt
python main.py
```

Optional, only needed if rebuilding the external state outcome file from the downloaded HTML pages:

```bash
python -m src.build_external_outcomes
```

## Main Outputs

- `outputs/tables/job_postings_with_ai_indicator.csv`
- `outputs/tables/ai_adoption_by_company_sector_country_year.csv`
- `outputs/tables/merged_ai_economic_panel.csv`
- `outputs/tables/regression_results.csv`
- `outputs/figures/ai_adoption_by_sector.png`
- `outputs/figures/ai_adoption_by_country.png`
- `outputs/figures/ai_vs_productivity.png`

## Methodology Summary

The AI indicator is based on normalized job-title, description, and skills text. A posting is classified as AI-related when it contains high-confidence AI terms such as `machine learning`, `deep learning`, `generative AI`, `LLM`, `natural language processing`, `computer vision`, `MLOps`, or related AI tools.

Each posting receives:

- `ai_score`: weighted intensity score from 0 to 1
- `ai_posting`: binary indicator equal to 1 when `ai_score >= 0.2`
- `matched_ai_terms`: matched terms for auditability

The core adoption measure is:

```text
AI adoption rate = AI-related postings / total postings
```

The baseline model estimates relationships such as:

```text
outcome = beta0 + beta1 * AI adoption rate + fixed effects + error
```

## Note on Lightcast

The professor's preferred source is Lightcast job postings. Because Lightcast data is restricted, this repository uses a public LinkedIn postings dataset to implement the same workflow. If Lightcast data becomes available, replace `data/job_postings.csv` with a Lightcast export containing posting ID, company, sector, geography, date, title, description, and skills.
