# rag_agent_logic.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import pinecone

# ----------------------------
# Config
# ----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_INDEX_NAME = "youtube-qa-index"  # Your existing Pinecone index

# ----------------------------
# Initialize Pinecone vector store
# ----------------------------
def initialize_vectorstore(api_key: str, environment: str):
    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(VECTOR_STORE_INDEX_NAME)
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectorstore = Pinecone(index, embeddings_model.encode, "text")
    return vectorstore

# ----------------------------
# Initialize LangChain LLM + Memory
# ----------------------------
def initialize_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt_template = """
    You are a helpful AI assistant. Use the following context to answer the question.
    Context: {context}
    Question: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# ----------------------------
# Chat helper
# ----------------------------
def chat_with_rag_and_tools(chain, user_question: str):
    result = chain({"question": user_question})
    answer = result.get("answer", "")
    sources = result.get("source_documents", [])
    return answer, sources
