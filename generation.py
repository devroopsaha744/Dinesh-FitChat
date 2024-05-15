import os 
import retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')


ret = retrieval.retriever
llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-latest', google_api_key=google_api_key)

template = """You are a doctor chatbot that helps in treating illness of patients.
Use the following pieces of context to answer the question at the end and try to cure the
illness of the patients.
{context}
Question: {question}

Helpful Answer:"""
prompt_template = PromptTemplate.from_template(template=template)

set_ret = RunnableParallel(
    {"context": ret, "question": RunnablePassthrough()} 
)

rag_chain = set_ret |  prompt_template | llm | StrOutputParser()




