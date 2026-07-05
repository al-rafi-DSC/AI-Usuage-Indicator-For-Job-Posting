# Data Sources

## Job Postings

Primary postings data comes from the public Hugging Face dataset:

- Dataset: `xanderios/linkedin-job-postings`
- File used: `job_postings.csv`
- License tag: MIT
- Source URL: https://huggingface.co/datasets/xanderios/linkedin-job-postings

The dataset contains LinkedIn job postings with fields such as job ID, company ID, title, description, location, listed time, salary fields, work type, and skills description. The project converts these fields into the analysis schema used by the pipeline.

## External Economic Outcomes

The state-level external outcome file `data/us_state_economic_outcomes.csv` is built from public web tables:

- U.S. state income table: https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_income
- U.S. state GDP table: https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP

These tables cite official sources such as the American Community Survey and the U.S. Bureau of Economic Analysis. Variables used include median household income, per-capita income, GDP per capita, and state population.

## Important Note

This is a public-data substitute for the course's preferred Lightcast data. The methodology and code are built so a Lightcast export can replace `data/job_postings.csv` later, while keeping the same AI indicator, aggregation, and regression pipeline.
