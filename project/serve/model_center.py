from typing import Any

from ..llm.call_llm import get_completion
from ..qa_chain.QAChainSelf import QAChainSelf
from ..qa_chain.ChatQAChainSelf import ChatQAChainSelf
from ..qa_chain.Respond import Respond


class ModelCenter:
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}
        self.respond_self = {}

    def chat_qa_chain_self_answer(
            self,
            question: str,
            history: list[tuple[str, str]] = None,
            model: str = "gpt-3.5-turbo",
            embedding: str = "openai",
            temperature: float = 0.1,
            top_k: int = 4,
            history_len: int = 3,
    ):
        history = history if history is not None else []
        if question is None or len(question) == 0:
            return "", history
        try: 
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = \
                    ChatQAChainSelf(
                        model=model,
                        temperature=temperature,
                        top_k=top_k,
                        history=history,
                        embedding=embedding
                )
            chain = self.chat_qa_chain_self[(model, embedding)]
            chain.change_history_length(history_len)
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, history
        
    def qa_chain_self_answer(
            self,
            question: str,
            history: list[tuple[str, str]] = None,
            model: str = "gpt-3.5-turbo",
            embedding: str = "openai",
            temperature: float = 0.1,
            top_k: int = 4,
    ):
        history = history if history is not None else []
        if question is None or len(question) == 0:
            return "", history
        try: 
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = \
                    QAChainSelf(
                        model=model,
                        temperature=temperature,
                        top_k=top_k,
                        embedding=embedding
                )
            chain = self.qa_chain_self[(model, embedding)]
            answer = chain.answer(question=question, temperature=temperature, top_k=top_k)
            history.append((question, answer))
            return "", history
        except Exception as e:
            return e, history

    def clear_history(self):
        if len(self.chat_qa_chain_self):
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()
        if len(self.respond_self):
            for chain in self.respond_self.values():
                chain.clear_history()

    def format_chat_prompt(self, question, history):
        prompt = ""
        for turn in history:
            user, bot = turn
            prompt = f"{prompt} \n User: {user} \n Assistant: {bot}"
        prompt = f"{prompt} \n User: {question} \n Assistant: "
        return prompt

    def respond_self_answer(
            self,
            question: str = None,
            history: list[tuple[str, str]] = None,
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.1,
            history_len: int = 4,
            max_tokens: int = 2048
    ):
        history = history if history is not None else []
        if question is None or len(question) == 0:
            return "", history
        try:
            if model not in self.respond_self:
                self.respond_self[model] = \
                    Respond(
                        model=model,
                        temperature=temperature,
                        history=history,
                        max_tokens=max_tokens
                )
            chain = self.respond_self[model]
            chain.change_history_length(history_len)
            formatted_prompt = self.format_chat_prompt(question, chain.history)
            return "", chain.answer(question=question, temperature=temperature, model=model, max_tokens=max_tokens, prompt=formatted_prompt)
        except Exception as e:
            return e, history
