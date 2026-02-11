class ExplainabilityEngine:
    def __init__(self, weights: dict):
        self.weights = weights

    def explain(self, signal_scores: dict) -> dict:
        """
        Returns contribution breakdown for each signal
        """
        contributions = {}

        for signal, score in signal_scores.items():
            weight = self.weights.get(signal, 0)
            contributions[signal] = round(score * weight * 100, 2)

        return contributions

    def top_reasons(self, contributions: dict, top_k: int = 3):
        """
        Return top contributing reasons
        """
        sorted_signals = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_signals[:top_k]

