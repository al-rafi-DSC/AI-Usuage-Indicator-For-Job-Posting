import pandas as pd

from src.ai_indicator import add_ai_indicator, score_ai_usage


def test_score_ai_usage_detects_high_confidence_terms():
    score, terms = score_ai_usage("Machine learning engineer with NLP and PyTorch experience")

    assert score >= 0.5
    assert "machine learning" in terms
    assert "nlp" in terms
    assert "pytorch" in terms


def test_add_ai_indicator_flags_ai_posting():
    postings = pd.DataFrame(
        [
            {
                "posting_id": 1,
                "job_title": "Data Scientist",
                "description": "Build deep learning models for computer vision.",
                "skills": "Python; TensorFlow",
            },
            {
                "posting_id": 2,
                "job_title": "Accountant",
                "description": "Prepare invoices and monthly financial statements.",
                "skills": "Excel; accounting",
            },
        ]
    )

    result = add_ai_indicator(postings)

    assert result.loc[0, "ai_posting"] == 1
    assert result.loc[1, "ai_posting"] == 0
