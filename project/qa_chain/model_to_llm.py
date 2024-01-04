from langchain.chat_models import ChatOpenAI
from ..llm.call_llm import parse_llm_api_key


def model_to_llm(
        model: str = None,
        temperature: float = 0.0,
        api_key: str = None
):
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        if api_key is None:
            api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key
        )
    else:
        raise ValueError(f"model {model} not support")
        
    return llm
