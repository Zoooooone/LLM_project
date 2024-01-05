import gradio as gr
from typing import Any

from ..llm.call_llm import get_completion
from ..database.create_db import create_db_info

from ..qa_chain.QAChainSelf import QAChainSelf
from ..qa_chain.ChatQAChainSelf import ChatQAChainSelf

LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
}
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
EMBEDDING_MODEL_LIST = ["openai"]
INIT_EMBEDDING_MODEL = "openai"


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
    

def interface():
    model_center = ModelCenter()
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            # img
            
            with gr.Column(scale=2):
                gr.Markdown("<h1><center>Local LLM Chatbot</center></h1>")

            # img_2

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=400,
                    show_copy_button=True,
                    show_share_button=True,
                    # avatar_images
                )
                question = gr.Textbox(label="Question")

                with gr.Row():
                    db_with_history_btn = gr.Button("Chat database with history")
                    db_without_history_btn = gr.Button("Chat database without history")
                    llm_btn = gr.Button("Chat with LLM")
                with gr.Row():
                    clear = gr.ClearButton(
                        components=[chatbot],
                        value="Clear console"
                    )

            with gr.Column(scale=1):
                file = gr.File(
                    label="Please select the knowledge database directory",
                    file_count="directory",
                    file_types=[".txt", ".md", ".docx", ".pdf"]
                )
                with gr.Row():
                    init_db = gr.Button("knowledge database embedding")

                model_argument = gr.Accordion("parameters", open=False)
                with model_argument:
                    temperature = gr.Slider(
                        value=0.01,
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        label="llm temperature",
                        interactive=True
                    )
                    top_k = gr.Slider(
                        value=3,
                        minimum=1,
                        maximum=10,
                        label="top k similar contents when searching",
                        interactive=True
                    )
                    history_len = gr.Slider(
                        value=3,
                        minimum=0,
                        maximum=5,
                        step=1,
                        label="history length",
                        interactive=True
                    )

                model_select = gr.Accordion("model selection")
                with model_select:
                    llm = gr.Dropdown(
                        LLM_MODEL_LIST,
                        label="LLM",
                        value="gpt-3.5-turbo",
                        interactive=True
                    )
                    embeddings = gr.Dropdown(
                        EMBEDDING_MODEL_LIST,
                        label="Embedding models",
                        value=INIT_EMBEDDING_MODEL
                    )
        
            # database initialization
            init_db.click(
                create_db_info,
                inputs=[file, embeddings],
                outputs=[question]
            )

            # chat_qa_chain
            db_with_history_btn.click(
                model_center.chat_qa_chain_self_answer,
                inputs=[question, chatbot, llm, embeddings, temperature, top_k, history_len],
                outputs=[question, chatbot]
            )

            # qa_chain
            db_without_history_btn.click(
                model_center.qa_chain_self_answer,
                inputs=[question, chatbot, llm, embeddings, temperature, top_k],
                outputs=[question, chatbot]
            )

            # respond
            llm_btn.click(
                respond,
                inputs=[question, chatbot, llm, history_len, temperature],
                outputs=[question, chatbot],
                show_progress="minimal"
            )

            # submit
            question.submit(
                respond,
                inputs=[question, chatbot, llm, history_len, temperature],
                outputs=[question, chatbot],
                show_progress="hidden"
            )

            # clear history
            clear.click(model_center.clear_history)
        
        gr.Markdown("""
            1. Please upload your files at first.
            2. Database initialization may cost lot of time, please wait for a while.
            3. If there are some bugs, they will be shown in text box.         
        """)

    gr.close_all()
    demo.launch()


interface()
