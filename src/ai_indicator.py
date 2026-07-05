import re
from dataclasses import dataclass

import pandas as pd

from src.config import AI_SCORE_THRESHOLD


@dataclass(frozen=True)
class AiTermGroup:
    weight: float
    terms: tuple[str, ...]


AI_TERM_GROUPS = (
    AiTermGroup(
        weight=0.45,
        terms=(
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "generative ai",
            "genai",
            "large language model",
            "llm",
            "natural language processing",
            "nlp",
            "computer vision",
        ),
    ),
    AiTermGroup(
        weight=0.30,
        terms=(
            "tensorflow",
            "pytorch",
            "scikit learn",
            "keras",
            "hugging face",
            "transformer model",
            "prompt engineering",
            "mlops",
            "model deployment",
            "neural network",
        ),
    ),
    AiTermGroup(
        weight=0.20,
        terms=(
            "predictive modelling",
            "predictive modeling",
            "recommendation system",
            "classification model",
            "chatbot",
            "data science",
            "data scientist",
            "automation model",
        ),
    ),
    AiTermGroup(
        weight=0.10,
        terms=(
            "python",
            "r programming",
            "spark",
            "cloud ai",
            "azure ai",
            "aws sagemaker",
            "vertex ai",
        ),
    ),
)


def normalize_text(value: object) -> str:
    """Normalize text while preserving enough signal for keyword matching."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_text_blob(row: pd.Series) -> str:
    fields = ("job_title", "description", "skills")
    return " ".join(normalize_text(row.get(field, "")) for field in fields)


def score_ai_usage(text: str) -> tuple[float, list[str]]:
    normalized = normalize_text(text)
    matched_terms: list[str] = []
    score = 0.0

    for group in AI_TERM_GROUPS:
        group_matches = [term for term in group.terms if _contains_term(normalized, term)]
        if group_matches:
            matched_terms.extend(group_matches)
            score += group.weight

    return min(score, 1.0), sorted(set(matched_terms))


def add_ai_indicator(postings: pd.DataFrame) -> pd.DataFrame:
    required = {"job_title", "description", "skills"}
    missing = required.difference(postings.columns)
    if missing:
        raise ValueError(f"Missing required job posting columns: {sorted(missing)}")

    enriched = postings.copy()
    blobs = enriched.apply(build_text_blob, axis=1)
    scored = blobs.apply(score_ai_usage)

    enriched["ai_score"] = scored.apply(lambda item: item[0])
    enriched["matched_ai_terms"] = scored.apply(lambda item: "; ".join(item[1]))
    enriched["ai_posting"] = (enriched["ai_score"] >= AI_SCORE_THRESHOLD).astype(int)
    return enriched


def _contains_term(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None
