import os 
import retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')


ret = retrieval.retriever
llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-latest', google_api_key=google_api_key)



qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = ret
)

