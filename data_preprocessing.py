# src/data_preprocessing.py

import pandas as pd
import re
from typing import Optional


class NewsDataPreprocessor:
    """
    Data preprocessing pipeline for News Popularity Intelligence System.
    Handles real-world column inconsistencies safely.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def _clean_text(text: Optional[str]) -> str:
        """
        Minimal, transformer-friendly text cleaning.
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def load_data(self) -> pd.DataFrame:
        """
        Load raw dataset and normalize column names.
        """
        df = pd.read_csv(self.input_path)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        required_columns = {"title", "description"}
        missing_cols = required_columns - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean, merge, and deduplicate text fields.
        """
        df = df.copy()

        df["title_clean"] = df["title"].apply(self._clean_text)
        df["description_clean"] = df["description"].apply(self._clean_text)

        df["full_text"] = (
            df["title_clean"] + " [SEP] " + df["description_clean"]
        )

        # Drop empty & duplicate articles
        df = df[df["full_text"].str.strip().astype(bool)]
        df.drop_duplicates(subset=["full_text"], inplace=True)

        df.reset_index(drop=True, inplace=True)
        return df

    def save_data(self, df: pd.DataFrame):
        """
        Save processed dataset.
        """
        df.to_csv(self.output_path, index=False)

    def run_pipeline(self):
        """
        Run full preprocessing pipeline.
        """
        df = self.load_data()
        df_cleaned = self.preprocess(df)
        self.save_data(df_cleaned)

        print("✅ Preprocessing completed") 
        print("📊 Final dataset shape:", df_cleaned.shape)
        print("💾 Saved to:", self.output_path)


if __name__ == "__main__":

    input_path = r"C:\Vanitha\Documents\GUVI\final_project\news_popularity_system\data\raw\news.csv"
    output_path = r"C:\Vanitha\Documents\GUVI\final_project\news_popularity_system\data\processed\news_cleaned.csv"

    preprocessor = NewsDataPreprocessor(
        input_path=input_path,
        output_path=output_path
    )

    preprocessor.run_pipeline() 

