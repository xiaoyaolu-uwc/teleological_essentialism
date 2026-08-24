from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def classify(self, texts: list[str]) -> list[dict]:
        """Classify a batch of passage texts.

        Args:
            texts: raw passage strings, no metadata

        Returns:
            list of {"tag": str, "reasoning": str}, same length and order as texts
        """
