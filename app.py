import streamlit as st
from uploaddata import process_pdfs
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_chat import message

# Set page configuration
st.set_page_config(
    page_title="Custom Chatbot",
    layout="wide",  # Use wide layout for better presentation
)

# Custom CSS for chat bubbles, input box, and alignment
st.markdown("""
    <style>
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 60%;
    }
    .user-bubble {
        background-color: #E2F0D9;
        color: #000;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #F0F0F0;
        color: #000;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    /* Centering the input box and reducing its size */
    .stTextInput > div > input {
        width: 50%; /* Adjust the width to make it smaller */
        margin: 0 auto; /* Center it */
    }
    </style>
""", unsafe_allow_html=True)

qdrant_url = "https://fefb1c85-6465-4013-ac35-937a3dc7acae.eastus-0.azure.cloud.qdrant.io:6333"
qdrant_api_key = "Q-7DPiZWi4xCHr42qkTZH8mwuu626H5uzx9BvQMuR2gkcSr9hibFrA"
groq_api_key = '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'

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Initialize vector store from existing collection
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="custom-chatbot",
    embeddings=embeddings_model,
)

# Initialize retriever
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5},
)

# Initialize ChatGroq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama3-8b-8192",
)

# Create a custom prompt template (optional)
prompt_template = """
You are a helpful assistant that uses the provided context to answer the user's question.
If you don't know the answer, say that you don't know.

Context: {context}
Question: {question}
Helpful Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create a conversational retrieval chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# Sidebar
with st.sidebar:
    st.markdown("### 👋 Hey I'm your Personal Assistant")
    
    # Navigation Menu
    menu = ["🤖 Chatbot", "🔗📚Upload New Documents"]
    choice = st.selectbox("Navigate", menu, index=0)

# Home Page - Chatbot Interaction
if choice == "🤖 Chatbot":

    # Streamlit UI
    st.title("🤖 Let's Chat with Your Data")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

    # User input
    def get_user_input():
        return st.text_input("Ask Me anything:", key="input", placeholder="Type your question here... After Press Enter ▶️")


    user_question = get_user_input()

    if user_question:
        # Generate response
        with st.spinner("Thinking...!"):
            try:
                result = qa_chain({"question": user_question})
                answer = result["answer"]
                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                answer = "I'm sorry, but I couldn't process your request."
                st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)  # Start container div for chat bubbles
    
        # Loop through chat history in reverse to show the latest first
        for question, answer in reversed(st.session_state.chat_history):
            # User's question first (right aligned)
            st.markdown(f'<div class="chat-bubble user-bubble">👤: {question}</div>', unsafe_allow_html=True)
        
            # Bot's response next (left aligned)
            st.markdown(f'<div class="chat-bubble bot-bubble">🤖: {answer}</div>', unsafe_allow_html=True)
    
        st.markdown('</div>', unsafe_allow_html=True)  # End container div for chat bubbles
    
# Document Upload Page
elif choice == "🔗📚Upload New Documents":
    st.title("Kindly upload your document if you have a new version")
   
    # File uploader to allow users to upload new documents
    uploaded_files = st.file_uploader("Choose a document", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:  # The file uploader returns a list when multiple files are uploaded
    # Process each uploaded PDF file
        for uploaded_file in uploaded_files:
            
            if uploaded_file.type == "application/pdf":
                vectorstore = process_pdfs([uploaded_file])  # Send the uploaded file as a list
                st.success(f"Your data from {uploaded_file.name} has been processed and stored in Qdrant DB. You can now interact with the chatbot!")
            else:
                st.error(f"{uploaded_file.name} is not a PDF file. Please upload PDF files only.")
    else:
        st.info("Please upload a PDF file to proceed.")
