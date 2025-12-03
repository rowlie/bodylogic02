# -*- coding: utf-8 -*-
"""Streamlit App Interface"""

import streamlit as st
import os
from rag_agent_logic import initialize_vectorstore, initialize_chain, chat_with_rag_and_tools

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

if "chain" not in st.session_state:
    st.session_state.chain = None

if "memory" not in st.session_state:
    st.session_state.memory = None

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

    if not all([api_key, pinecone_key]):
        st.warning("⚠️ Missing critical environment variables!")
        st.info(
            """
            Please ensure you have set these in your Streamlit Cloud secrets:
            - `OPENAI_API_KEY`
            - `PINECONE_API_KEY`
            - `PINECONE_ENVIRONMENT` (optional)
            """
        )
    else:
        st.success("✅ All critical API keys configured")

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chain_initialized = False
        st.session_state.chain = None
        st.session_state.memory = None
        st.rerun()

# ----------------------------
# Main Application
# ----------------------------
st.title("🤖 Body Logic RAG Chat")
st.markdown("Ask questions about your YouTube QA data! Memory and RAG retrieval are active.")

# Initialize chain if not already
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG Agent..."):
            vectorstore = initialize_vectorstore()
            chain, memory = initialize_chain(vectorstore)
            st.session_state.chain = chain
            st.session_state.memory = memory
            st.session_state.chain_initialized = True
            st.success("✅ Agent successfully initialized! Ask your question.")
    except Exception as e:
        st.error(f"❌ Failed to initialize chain: {str(e)}")
        st.stop()

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Thinking..."):
                response, sources = chat_with_rag_and_tools(st.session_state.chain, prompt)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"❌ Error during response generation: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("💡 Memory and tools are active. Ask a follow-up question!")
