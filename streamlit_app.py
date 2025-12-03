# streamlit_app.py
import streamlit as st
from rag_agent_logic import initialize_vectorstore, initialize_chain, chat_with_rag_and_tools
import os

st.set_page_config(page_title="YouTube QA Assistant", page_icon="🤖")

# ----------------------------
# Pinecone credentials
# ----------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# ----------------------------
# Initialize vector store and chain
# ----------------------------
vectorstore = initialize_vectorstore(PINECONE_API_KEY, PINECONE_ENV)
chain = initialize_chain(vectorstore)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("YouTube QA Assistant")
st.write("Ask questions about your YouTube video transcripts!")

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer, sources = chat_with_rag_and_tools(chain, user_question)
        st.markdown(f"**Answer:** {answer}")

        if sources:
            st.markdown("**Sources:**")
            for doc in sources:
                st.markdown(f"- {doc.page_content[:200]}...")
