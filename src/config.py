from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

JOB_POSTINGS_FILE = DATA_DIR / "job_postings.csv"
ECONOMIC_OUTCOMES_FILE = DATA_DIR / "economic_outcomes.csv"
STATE_ECONOMIC_OUTCOMES_FILE = DATA_DIR / "us_state_economic_outcomes.csv"

SAMPLE_JOB_POSTINGS_FILE = DATA_DIR / "job_postings_sample.csv"
SAMPLE_ECONOMIC_OUTCOMES_FILE = DATA_DIR / "economic_outcomes_sample.csv"

AI_SCORE_THRESHOLD = 0.20
