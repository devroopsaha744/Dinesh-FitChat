# %%
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
# %%
google_api_key = os.getenv('GOOGLE_API_KEY')
pinecone_key = os.getenv("PINECONE_API_KEY")

# %%
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= google_api_key)

# %%
data_dir = r"C:\Users\devro\OneDrive\Desktop\Projects\DrChat\data\Medical.pdf"
loader = PyPDFLoader(data_dir )

# %%
docs = loader.load()

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,)
texts = text_splitter.split_documents(docs)

# %%
index_name = "chikitsak"
vectorstore = PineconeVectorStore.from_documents(texts, index_name= index_name, embedding= embeddings)

# %%
retriever = vectorstore.as_retriever()


