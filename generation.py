import os 
import retrieval
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')


ret = retrieval.retriever
llm =  ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

template = """You are a fitness trainer chatbot that helps in giving fitness/gym advice to the people.
Use the following pieces of context to answer the fitness related or gym related questions.
{context}
Question: {question}

Helpful Answer:"""
prompt_template = PromptTemplate.from_template(template=template)

set_ret = RunnableParallel(
    {"context": ret, "question": RunnablePassthrough()} 
)

rag_chain = set_ret |  prompt_template | llm | StrOutputParser()




