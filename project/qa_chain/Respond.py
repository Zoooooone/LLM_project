from ..llm.call_llm import get_completion


class Respond:
    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.1,
            history: list = [],
            max_tokens: int = 2048
    ):
        self.model = model
        self.temperature = temperature
        self.history = history
        self.max_tokens = max_tokens

    def clear_history(self):
        self.history.clear()
        return self.history

    def change_history_length(self, history_len: int = 3):
        n = len(self.history)
        if history_len > n:
            return self.history
        else:
            return self.history[-history_len:]

    def answer(
            self,
            question: str = None,
            model: str = None,
            temperature: float = None,
            max_tokens: int = None,
            prompt: str = None
    ):
        if model is None:
            model = self.model
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if prompt is None:
            prompt = f"please answer the question. {question}"

        result = get_completion(
            prompt=prompt,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens
        )
        self.history.append((question, result))

        return self.history
