import streamlit as st
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from utils import convert_pdf_to_images,extract_text_from_pdf
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings


# Load environment variables
load_dotenv()


# Initialize embeddings
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="💬",
    layout="wide"
)

# App Header
st.markdown("<h1 style='text-align: center; color:#1E3A8A;'>PDF Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#6B7280;'>Upload PDFs and have a conversation about their content</p>", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    with st.expander("API Settings", expanded=True):
        api_key = st.text_input("Enter your GROQ AI API Key:", type="password", 
                                  help="Your API key will not be stored after the session ends")
    
    if api_key:
        session_id = st.text_input("Session ID:", value="default", 
                                   help="Use different session IDs for different conversations")
        
        with st.expander("Upload Documents", expanded=True):
            uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    else:
        uploaded_files = None  # No file uploads if API key is missing

    st.markdown("### How to use")
    st.markdown("""
    1. Enter your GROQ API key
    2. Upload PDF documents
    3. Ask questions about the content
    """)

# Initialize session state store and other variables
if "store" not in st.session_state:
    st.session_state.store = {}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "images" not in st.session_state:
    st.session_state.images = None

# Main functionality when API key is provided
if api_key:
    llm = ChatGroq(model="qwen-2.5-32b", api_key=api_key)
    
    # Process uploaded PDFs
    if uploaded_files:
        if st.session_state.vector_store is None:
            st.session_state.images = []  # Initialize as an empty list if not already set
            with st.spinner("Processing documents... This may take a moment."):
                documents = []
                for uploaded_file in uploaded_files:
                    temppdf = './temp.pdf'
                    docs = []
                    try:
                        with open(temppdf, "wb") as file:
                            file.write(uploaded_file.getvalue())                        
                        loader = PyPDFLoader(temppdf)
                        docs = loader.load()
                        documents.extend(docs)
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        splitted_docs = text_splitter.split_documents(documents)
                        st.session_state.vector_store = FAISS.from_documents(splitted_docs, embeddings)
                        st.session_state.retriever = st.session_state.vector_store.as_retriever()
                    except Exception as e:
                        st.info(f"Trying our best OCR to extract text....")
                        t = extract_text_from_pdf(temppdf)
                        docs = [Document(page_content=t, metadata={'source': uploaded_file.name})]
                        print(docs)
                        documents.extend(docs)
                                        # Split documents for RAG
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        splitted_docs = text_splitter.split_documents(documents)
                        
                        st.session_state.vector_store = FAISS.from_documents(splitted_docs, embeddings)
                        st.session_state.retriever = st.session_state.vector_store.as_retriever()
                    # Convert the PDF to images and append them to the images list
                    st.session_state.images = convert_pdf_to_images(temppdf)

        # History-aware retriever chain configuration
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever_chain = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)
        
        # QA Chain configuration
        system_prompt = (
            "You are a helpful PDF assistant that provides accurate and concise responses based on the document content. "
            "Use the following retrieved context to answer the question. "
            "If the answer isn't in the context, say 'I don't have enough information about that in the documents you provided.' "
            "Keep your answers conversational but concise and informative. "
            "Format important information in **bold** when appropriate.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever_chain, question_answer_chain)
        
        # Function to manage session history per session ID
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Display PDF previews in the sidebar if available
        with st.sidebar:
            with st.expander("Document Previews", expanded=False):
                if st.session_state.images:
                    for idx, img in enumerate(st.session_state.images):
                        st.image(img, caption=f"Page {idx+1}", use_container_width=True)
        
        # --- Chat Display and Input ---
        # Render existing chat history
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])
        
        # Chat input and clear button in the same row
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.chat_input("Ask about your documents:")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                if session_id in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                st.rerun()
        
        # Handle user input
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.spinner("Thinking..."):
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
            
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            st.chat_message("assistant").write(response['answer'])
            st.rerun()
    else:
        st.info("Please upload PDF documents to begin the conversation.")
else:
    # Welcome screen when no API key is provided
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: "black"; border-radius: 1rem; margin: 2rem 0">
            <img src="https://img.icons8.com/fluency/96/000000/chat.png" style="width: 64px; margin-bottom: 1rem">
            <h2>Welcome to PDF Chat Assistant</h2>
            <p>Enter your GROQ API key in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.warning("⚠️ Please enter your GROQ API Key to use the application")
