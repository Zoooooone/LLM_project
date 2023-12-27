import os
import openai
import sys
sys.path.append("project")

from database.create_db import load_db

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
 
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


class QA_Chain:
    def __init__(self, model, temperature, top_k, history):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.history = history
        
        self.llm = ChatOpenAI(model_name=self.model, temperature=self.temperature)
        self.vectordb = load_db()

    def get_answer(self, question):
        template = """
            Use the following context enclosed by ``` to answer the final question enclosed by '''. \
            If you don't know the answer, just say you don't know, don't try to make it upcase. \
            You should make your answer as detailed and specific as possible without going off topic. \
            If the answer is long, break it up as appropriate to improve the reading experience of the answer.
            ```{context}```
            Question: '''{question}'''
        """
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        result = qa_chain({"query": question})
        answer = result["result"]
        self.history.append((question, answer))

        return self.history

    def get_llm_answer(self, question):
        prompt_template = f"please answer the question enclosed by ```. ```{question}```"

        return self.llm.predict(prompt_template)


if __name__ == "__main__":
    qa = QA_Chain(
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
        print("\n")
