import gradio as gr

from .model_center import ModelCenter
from ..database.create_db import create_db_info

LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
}
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
EMBEDDING_MODEL_LIST = ["openai"]
INIT_EMBEDDING_MODEL = "openai"


def return_2():
    return [[2]]


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
                msg = gr.Textbox(label="Prompt")

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
                outputs=[msg]
            )

            # chat_qa_chain
            db_with_history_btn.click(
                model_center.chat_qa_chain_self_answer,
                inputs=[msg, chatbot, llm, embeddings, temperature, top_k, history_len],
                outputs=[msg, chatbot]
            )

            # qa_chain
            db_without_history_btn.click(
                model_center.qa_chain_self_answer,
                inputs=[msg, chatbot, llm, embeddings, temperature, top_k],
                outputs=[msg, chatbot]
            )

            # respond
            llm_btn.click(
                model_center.respond_self_answer,
                inputs=[msg, chatbot, llm, temperature, history_len],
                outputs=[msg, chatbot]
            )

            # submit
            msg.submit(
                model_center.respond_self_answer,
                inputs=[msg, chatbot, llm, temperature, history_len],
                outputs=[msg, chatbot]
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
