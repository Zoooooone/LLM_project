from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from .model_to_llm import model_to_llm
from .get_vectordb import get_vectordb


class ChatQAChainSelf:
    def __init__(
            self, 
            model: str = "gpt-3.5-turbo", 
            temperature: float = 0.1, 
            top_k: int = 4,
            history: list = [], 
            file_path: str = None,
            persist_path: str = None,
            api_key: str = None,
            embedding: str = "openai",
            embedding_key: str = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.history = history
        self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        
        self.vector_db = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.api_key)

        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever
        )

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
            "question": question,
            "chat_history": self.history
        })
        result = result["answer"]
        self.history.append((question, result))

        return self.history


if __name__ == "__main__":
    qa = ChatQAChainSelf(
        model="gpt-3.5-turbo",
        top_k=3
    )

    q1 = "Can you briefly introduce the deep learning with no more than 15 words?"
    q2 = "How to learn it?"
    q3 = "Who is Yao Ming?"

    a1 = qa.answer(q1)[-1][-1]
    a2 = qa.answer(q2)[-1][-1]
    a3 = qa.answer(q3)[-1][-1]
    
    for q, a in zip([q1, q2, q3], [a1, a2, a3]):
        print(f"question: {q} \n")
        print(f"answer: {a} \n\n")

    print(qa.change_history_length(4))
    print(qa.change_history_length(1))

    qa.clear_history()
    print(qa.history)
