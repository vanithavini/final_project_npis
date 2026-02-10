import pandas as pd

class ArticleRankingEngine:
    def __init__(self, score_column="popularity_percentage"):
        self.score_column = score_column

    def rank_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank articles by popularity score (descending)
        """
        ranked_df = df.sort_values(
            by=self.score_column,
            ascending=False
        ).reset_index(drop=True)

        ranked_df["rank"] = ranked_df.index + 1
        return ranked_df

    def rank_by_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank articles within each priority group
        """
        priority_order = {
            "High Priority": 0,
            "Medium Priority": 1,
            "Non-Priority": 2
        }

        df["priority_order"] = df["priority_label"].map(priority_order)

        ranked_df = df.sort_values(
            by=["priority_order", self.score_column],
            ascending=[True, False]
        ).reset_index(drop=True)

        ranked_df["rank"] = ranked_df.index + 1
        return ranked_df.drop(columns=["priority_order"])

