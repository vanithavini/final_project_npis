import numpy as np
from src.popularity_signals import PopularitySignals


class PopularityScorer:
    def __init__(self):
        self.signals = PopularitySignals()

        # Editorial weights 
        self.weights = {
            "emotion": 0.25,
            "urgency": 0.20,
            "lexical": 0.15,
            "readability": 0.15,
            "length": 0.10,
            "subjectivity": 0.15
        }

    def score_article(self, text):
        scores = {
            "emotion": self.signals.emotional_intensity(text),
            "urgency": self.signals.urgency_score(text),
            "lexical": self.signals.lexical_diversity(text),
            "readability": self.signals.readability_score(text) / 100,  # normalize
            "length": self.signals.length_score(text),
            "subjectivity": self.signals.subjectivity(text)
        }

        weighted_score = sum(scores[k] * self.weights[k] for k in scores)
        final_score = np.clip(weighted_score * 100, 0, 100)

        return final_score, scores

    def priority_label(self, score):
        if score < 35:
            return "Non-Priority"
        elif score < 80:
            return "Medium Priority"
        else:
            return "High Priority"


