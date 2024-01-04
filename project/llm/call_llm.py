import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


def parse_llm_api_key(model: str, env_file: dict = None):
    if env_file is None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        api_key = env_file["OPENAI_API_KEY"]
        if api_key is None:
            raise ValueError("API Key doesn't exist")
        return api_key
    else:
        raise ValueError(f"model{model} not support")


def get_completion(
        prompt: str, 
        model: str, 
        temperature=0.1,
        api_key=None,
        max_tokens=2048
):
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return get_completion_gpt(prompt, model, temperature, api_key, max_tokens)
    else:
        return "Wrong model"
    

def get_completion_gpt(
        prompt: str, 
        model: str, 
        temperature: float,
        api_key: str,
        max_tokens: int
):
    if api_key is None:
        api_key = parse_llm_api_key("openai")
    openai.api_key = api_key
    messages = [{"role": "user", "content": prompt}]
    response = OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
