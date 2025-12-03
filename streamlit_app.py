# -*- coding: utf-8 -*-
"""Streamlit App Interface (streamlit_app.py)"""

import streamlit as st
import os

# Import your chain logic from the separate file
from rag_agent_logic import initialize_chain, chat_with_rag_and_tools

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for conversation history display
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    # Check if required API keys are set
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")  # Optional

    if not all([api_key, pinecone_key]):
        st.warning("⚠️ Missing critical environment variables!")
        st.info(
            """
            Please ensure you have set these in your Streamlit Cloud secrets:
            - `OPENAI_API_KEY`
            - `PINECONE_API_KEY`
            - `LANGCHAIN_API_KEY` (optional, for LangSmith)
            """
        )
    else:
        st.success("✅ All critical API keys configured")

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        if "agent_memory" in st.session_state:
            del st.session_state["agent_memory"]
        st.session_state.chain_initialized = False
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

# Initialize chain once
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG Agent..."):
            initialize_chain()
            st.session_state.chain_initialized = True
            st.success("Agent successfully initialized! Ask me a question.")
    except Exception as e:
        st.error(f"❌ Failed to initialize chain. Check environment variables: {str(e)}")
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
                response = chat_with_rag_and_tools(prompt)
                message_placeholder.markdown(response)

                # Add assistant response to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
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
