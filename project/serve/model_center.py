from typing import Any

from ..llm.call_llm import get_completion
from ..qa_chain.QAChainSelf import QAChainSelf
from ..qa_chain.ChatQAChainSelf import ChatQAChainSelf


class ModelCenter:
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(
            self,
            question: str,
            history: list[str] = None,
            model: str = "openai",
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
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)[1]
        except Exception as e:
            return e, history
        
    def qa_chain_self_answer(
            self,
            question: str,
            history: list[str] = None,
            model: str = "openai",
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


def format_chat_prompt(question, history):
    prompt = ""
    for turn in history:
        user, bot = turn
        prompt = f"{prompt} \n User: {user} \n Assistant: {bot}"
    prompt = f"{prompt} \n User: {question} \n Assistant: "
    return prompt


def respond(
        model: Any,
        question: str = None,
        history: list[str] = None,
        history_len: int = 4,
        temperature: float = 0.1,
        max_tokens: int = 2048
):
    history = history if history is not None else []
    if question is None or len(question) == 0:
        return "", history
    try:
        history = history[-history_len:] if history_len else []
        formatted_prompt = format_chat_prompt(question, history)
        bot_answer = get_completion(
            prompt=formatted_prompt,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens
        )
        history.append((question, bot_answer))
        return "", history
    except Exception as e:
        return e, history
