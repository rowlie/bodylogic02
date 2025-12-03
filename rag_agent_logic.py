# -*- coding: utf-8 -*-
"""RAG Agent Logic for Streamlit (rag_agent_logic.py)"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import pinecone
import os

# ============================================================================
# CONFIG
# ============================================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_INDEX_NAME = "my-index"

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_vectorstore():
    """
    Initialize Pinecone vectorstore using environment variables.
    Make sure PINECONE_API_KEY and PINECONE_ENV are set.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENV")  # e.g., "us-east1-gcp"

    if not api_key or not environment:
        raise ValueError("Missing PINECONE_API_KEY or PINECONE_ENV environment variables.")

    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(VECTOR_STORE_INDEX_NAME)

    # Use sentence-transformers model for embeddings
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Pinecone vectorstore wrapper for LangChain
    vectorstore = Pinecone(index, embeddings_model.encode, "text")

    return vectorstore

def initialize_chain(vectorstore):
    """
    Create a conversational retrieval chain with memory.
    """
    # Chat LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    # Memory for conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Prompt template
    prompt_template = """
    You are a helpful AI assistant. Use the following context to answer the question.
    Context: {context}
    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

# ============================================================================
# CHAT HELPER
# ============================================================================

def chat_with_rag_and_tools(chain, user_question: str):
    """
    Ask a question to the RAG chain and return answer + sources.
    """
    if chain is None:
        raise RuntimeError("Chain not initialized. Call initialize_chain() first.")

    result = chain({"question": user_question})
    answer = result.get("answer", "")
    sources = result.get("source_documents", [])
    return answer, sources
