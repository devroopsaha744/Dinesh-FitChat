# %%
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
# %%
huggingface_api_key =  os.getenv('HF_TOKEN')
pinecone_key = os.getenv("PINECONE_API_KEY")

# %%
embeddings = embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key= huggingface_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# %%
data_dir = "data/Gym Book.pdf"
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
index_name = "dinesh"
vectorstore = PineconeVectorStore.from_documents(texts, index_name= index_name, embedding= embeddings)

# %%
retriever = vectorstore.as_retriever()


