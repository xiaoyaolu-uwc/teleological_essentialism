from archive.prompt_phase.base import ModelAdapter


class BertModel(ModelAdapter):
    def __init__(self, model_path: str):
        raise NotImplementedError("BertModel is not yet implemented")

    def classify(self, texts: list[str]) -> list[dict]:
        raise NotImplementedError
