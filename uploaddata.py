import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import warnings
import uuid
import time

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
qdrant_url = "https://fefb1c85-6465-4013-ac35-937a3dc7acae.eastus-0.azure.cloud.qdrant.io:6333"
qdrant_api_key = "Q-7DPiZWi4xCHr42qkTZH8mwuu626H5uzx9BvQMuR2gkcSr9hibFrA"

warnings.filterwarnings("ignore")

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split the document and generate embeddings
def process_pdfs(uploaded_files):
    docs = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        text = load_pdf(uploaded_file)
        docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(docs)

    # Generate embeddings
    points = get_embedding(text_chunks)

    # Store embeddings in Qdrant
    vectorstore = Qdrant.from_documents(
        documents=text_chunks,
        embedding=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5'),
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="custom_agent-2",
        force_recreate=True
    )

    return vectorstore

# Function to generate embeddings for text chunks
def get_embedding(text_chunks, model_name='BAAI/bge-small-en-v1.5'):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    points = []
    
    embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in text_chunks])

    for idx, chunk in enumerate(text_chunks):
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": embeddings[idx],
            "payload": {"text": chunk.page_content, "source": chunk.metadata["source"]}
        })

    return points
