import pandas as pd

from src.config import (
    ECONOMIC_OUTCOMES_FILE,
    FIGURE_DIR,
    JOB_POSTINGS_FILE,
    SAMPLE_ECONOMIC_OUTCOMES_FILE,
    SAMPLE_JOB_POSTINGS_FILE,
    STATE_ECONOMIC_OUTCOMES_FILE,
    TABLE_DIR,
)

US_STATE_ABBRS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_job_postings() -> pd.DataFrame:
    path = JOB_POSTINGS_FILE if JOB_POSTINGS_FILE.exists() else SAMPLE_JOB_POSTINGS_FILE
    postings = pd.read_csv(path)
    postings = normalize_job_posting_schema(postings)
    postings["posting_date"] = pd.to_datetime(postings["posting_date"], errors="coerce")
    postings["year"] = postings["posting_date"].dt.year
    return postings


def load_economic_outcomes() -> pd.DataFrame:
    if STATE_ECONOMIC_OUTCOMES_FILE.exists():
        return pd.read_csv(STATE_ECONOMIC_OUTCOMES_FILE)
    path = ECONOMIC_OUTCOMES_FILE if ECONOMIC_OUTCOMES_FILE.exists() else SAMPLE_ECONOMIC_OUTCOMES_FILE
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, filename: str) -> None:
    ensure_output_dirs()
    df.to_csv(TABLE_DIR / filename, index=False)


def normalize_job_posting_schema(postings: pd.DataFrame) -> pd.DataFrame:
    if {"posting_id", "company", "sector", "country", "region", "posting_date"}.issubset(postings.columns):
        return postings

    if "job_id" not in postings.columns or "title" not in postings.columns:
        raise ValueError("Unsupported job postings schema. Expected Lightcast-style columns or LinkedIn job_postings.csv.")

    normalized = pd.DataFrame()
    normalized["posting_id"] = postings["job_id"]
    normalized["company"] = postings.get("company_id", "unknown_company").fillna("unknown_company").astype(str)
    normalized["job_title"] = postings["title"].fillna("")
    normalized["description"] = postings.get("description", "").fillna("")
    normalized["skills"] = postings.get("skills_desc", "").fillna("")
    normalized["sector"] = postings.apply(infer_sector, axis=1)
    normalized["country"] = "United States"
    normalized["region"] = postings.get("location", "").fillna("").apply(extract_us_state)
    normalized["posting_date"] = pd.to_datetime(
        postings.get("listed_time", postings.get("original_listed_time")),
        unit="ms",
        errors="coerce",
    )
    normalized["posted_salary"] = postings[["med_salary", "min_salary", "max_salary"]].apply(
        lambda row: row.dropna().mean() if row.notna().any() else None,
        axis=1,
    )
    normalized["work_type"] = postings.get("formatted_work_type", "")
    normalized["experience_level"] = postings.get("formatted_experience_level", "")
    normalized["source_dataset"] = "xanderios/linkedin-job-postings"
    return normalized


def extract_us_state(location: object) -> str:
    if pd.isna(location):
        return "Unknown"
    text = str(location).strip()
    if "," in text:
        state = text.split(",")[-1].strip().upper()
        if state in US_STATE_ABBRS:
            return state
    return "Unknown"


def infer_sector(row: pd.Series) -> str:
    text = f"{row.get('title', '')} {row.get('description', '')} {row.get('skills_desc', '')}".lower()
    sector_terms = {
        "Technology": ("software", "developer", "engineer", "data", "cloud", "cyber", "machine learning", "ai ", "it "),
        "Healthcare": ("health", "nurse", "clinical", "medical", "patient", "pharma", "therapy"),
        "Finance": ("finance", "bank", "accounting", "audit", "tax", "investment", "risk analyst"),
        "Manufacturing": ("manufacturing", "factory", "production", "machinist", "warehouse", "quality"),
        "Retail": ("retail", "store", "sales associate", "merchandising", "customer service"),
        "Education": ("teacher", "school", "university", "education", "instructor", "student"),
        "Energy": ("energy", "renewable", "utility", "solar", "wind", "oil", "gas"),
    }
    for sector, terms in sector_terms.items():
        if any(term in text for term in terms):
            return sector
    return "Other Services"
