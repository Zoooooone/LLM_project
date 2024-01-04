from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pathlib import Path

from .get_vectordb import get_vectordb
from .model_to_llm import model_to_llm


class QAChainSelf:
    current_file_path = Path(__file__).resolve()
    current_dir_path = current_file_path.parent
    template_path = current_dir_path / "template.txt"
    DEFAULT_TEMPLATE = template_path.read_text()

    def __init__(
            self, 
            model: str = "gpt-3.5-turbo", 
            temperature: float = 0.1, 
            top_k: int = 4, 
            file_path: str = None,
            persist_path: str = None,
            api_key: str = None,
            embedding: str = "openai",
            embedding_key: str = None,
            template: str = DEFAULT_TEMPLATE
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        
        self.vector_db = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.api_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], 
            template=self.template
        )

        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
        )

    def answer(
            self, 
            question: str = None,
            temperature: float = None,
            top_k: int = None
    ):
        if not question:
            return ""
        if temperature is None:
            temperature = self.temperature
        if top_k is None:
            top_k = self.top_k

        result = self.qa_chain({
            "query": question,
            "temperature": temperature,
            "top_k": top_k
        })
        return result["result"]


if __name__ == "__main__":
    qa = QAChainSelf()

    q1 = "Can you briefly introduce the deep learning with no more than 15 words?"
    q2 = "How to learn it?"
    q3 = "Who is Yao Ming?"

    a1 = qa.answer(q1)
    a2 = qa.answer(q2)
    a3 = qa.answer(q3)
    
    for q, a in zip([q1, q2, q3], [a1, a2, a3]):
        print(f"question: {q} \n")
        print(f"answer: {a} \n\n")
