import os
import openai
import sys
sys.path.append("project")

from database.create_db import load_db

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


class QA_Chain_With_History:
    def __init__(self, model, temperature, top_k, history):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.history = history
        
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        self.vectordb = load_db()
    
    def get_answer(self, question):
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
        )
        result = qa_chain({"question": question, "chat_history": self.history})
        answer = result["answer"]
        self.history.append((question, answer))

        return self.history


if __name__ == "__main__":
    qa = QA_Chain_With_History(
        model="gpt-4",
        temperature=0,
        top_k=3,
        history=[]
    )
    
    q1 = "Can you briefly introduce the deep learning with no more than 15 words?"
    q2 = "How to learn it?"
    q3 = "Who is Yao Ming?"

    qa.get_answer(q1)
    qa.get_answer(q2)
    qa.get_answer(q3)

    for question, answer in qa.history:
        print(f"Q: {question}")
        print(f"A: {answer}")
        print('\n')
