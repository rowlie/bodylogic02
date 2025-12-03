# -*- coding: utf-8 -*-
"""Streamlit App Interface (streamlit_app.py)"""

import streamlit as st
import os

# Import your RAG logic
from rag_agent_logic import initialize_vectorstore, initialize_chain, chat_with_rag_and_tools

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# ============================================================================
# SESSION STATE INIT
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

if "chain" not in st.session_state:
    st.session_state.chain = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")  # Optional

    if not all([api_key, pinecone_key, pinecone_env]):
        st.warning("⚠️ Missing critical environment variables!")
        st.info(
            """
            Please ensure you have set these in your Streamlit Cloud secrets:
            - `OPENAI_API_KEY`
            - `PINECONE_API_KEY`
            - `PINECONE_ENV`
            - `LANGCHAIN_API_KEY` (optional)
            """
        )
    else:
        st.success("✅ All critical API keys configured")

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chain_initialized = False
        st.session_state.chain = None
        st.rerun()

    st.divider()
    st.caption(
        """
**Demo Prompts**

**Pinecone RAG:**  
• *What is the most dangerous type of fat?*

**Tools:**  
• *I am 80kg, male, light activity, want to lose weight — what are my calories and protein targets?*

**Memory:**  
1. *I weigh 75kg*  
2. *What was my weight again?*
"""
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🤖 Welcome to Body Logic")
st.markdown(
    "Ask anything about your fitness goals—tools, memory, and RAG retrieval are active!"
)

# Initialize RAG chain if not done yet
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG Agent..."):
            vectorstore = initialize_vectorstore()
            st.session_state.chain = initialize_chain(vectorstore)
            st.session_state.chain_initialized = True
            st.success("Agent successfully initialized! Ask me a question.")
    except Exception as e:
        st.error(f"❌ Failed to initialize chain. Details: {str(e)}")
        st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Thinking..."):
                answer, sources = chat_with_rag_and_tools(st.session_state.chain, prompt)
                
                # Format answer and sources nicely
                formatted_sources = "\n\n".join([f"- {doc.metadata.get('source', 'Unknown source')}" for doc in sources])
                response_text = f"{answer}\n\n**Sources:**\n{formatted_sources}" if sources else answer

                message_placeholder.markdown(response_text)

                # Add assistant response to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

        except Exception as e:
            error_msg = f"❌ Error during response generation. Details: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# Footer
st.divider()
st.caption("💡 Tip: Memory and tools are active. Try a follow-up question!")
