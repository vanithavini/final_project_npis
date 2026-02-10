import numpy as np
import textstat
from textblob import TextBlob
import re


class PopularitySignals:
    def __init__(self):
        self.urgency_keywords = [
            "breaking", "urgent", "just in", "alert",
            "now", "today", "immediately", "developing"
        ]

    def emotional_intensity(self, text):
        blob = TextBlob(text)
        return abs(blob.sentiment.polarity)

    def subjectivity(self, text):
        blob = TextBlob(text)
        return blob.sentiment.subjectivity

    def urgency_score(self, text):
        text_lower = text.lower()
        return sum(1 for word in self.urgency_keywords if word in text_lower) / len(self.urgency_keywords)

    def lexical_diversity(self, text):
        words = re.findall(r"\w+", text.lower())
        if not words:
            return 0
        return len(set(words)) / len(words)

    def readability_score(self, text):
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return 0

    def length_score(self, text):
        words = text.split()
        if len(words) < 50:
            return 0.3
        elif len(words) < 150:
            return 1.0
        elif len(words) < 300:
            return 0.8
        else:
            return 0.5

