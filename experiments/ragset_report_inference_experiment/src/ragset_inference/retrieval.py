import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data import LABEL_COLUMNS


class GoldRetriever:
    """Small local retriever over the gold reports.

    Multilingual: TF-IDF operates on character n-grams which work
    across scripts (Latin, CJK, Arabic, etc.). For best results,
    pair with a multilingual embedding model in production.
    """

    def __init__(self, gold: pd.DataFrame, top_k: int = 8):
        self.gold = gold.reset_index(drop=True)
        self.top_k = min(top_k, len(self.gold))
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
            max_features=20000,
            token_pattern=r"[\w\W]+",
        )
        self.matrix = self.vectorizer.fit_transform(
            self.gold["Report"].astype(str)
        )

    def retrieve(self, report: str) -> list[dict]:
        q = self.vectorizer.transform([report])
        scores = cosine_similarity(q, self.matrix)[0]
        idxs = scores.argsort()[::-1][:self.top_k]
        return [
            {
                "study_id": str(self.gold.iloc[i]["StudyInstanceUID"]),
                "report": str(self.gold.iloc[i]["Report"]),
                "labels": {
                    label: int(self.gold.iloc[i][label])
                    for label in LABEL_COLUMNS
                },
            }
            for i in idxs
        ]

    def retrieve_excluding(self, report: str, exclude_study_id: str) -> list[dict]:
        """Retrieve gold examples, excluding a specific study ID.

        Prevents data leakage: when evaluating on a held-out gold report,
        that report must not appear in its own retrieval context.
        """
        results = self.retrieve(report)
        filtered = [r for r in results if r["study_id"] != exclude_study_id]
        return filtered
