# RAG pipeline using LangChain and GroqCloud API
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration and Prompting
load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful customer support and healthcare information assistant.\n"
    "Answer the user question using ONLY the provided context below.\n"
    "If the context is insufficient or does not contain the answer, explicitly state: "
    "'The provided documents do not contain sufficient information to answer this question.'\n"
    "Do not speculate, assume, or invent details not present in the context.\n"
    "Keep your response concise, polite, and directly addressing the query.\n\n"
    "Context:\n{context}"
)

_retriever = None
_llm = None


# Knowledge Base Ingestion
def load_knowledge_base_documents() -> List[Any]:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    documents = []

    wellness_path = os.path.join(data_dir, "wellness.txt")
    if os.path.exists(wellness_path):
        documents.extend(TextLoader(wellness_path, encoding="utf-8").load())

    med_pdf_path = os.path.join(data_dir, "medication_safety.pdf")
    if os.path.exists(med_pdf_path):
        documents.extend(PyPDFLoader(med_pdf_path).load())

    faq_docx_path = os.path.join(data_dir, "faq.docx")
    if os.path.exists(faq_docx_path):
        documents.extend(Docx2txtLoader(faq_docx_path).load())

    return documents


# Vector Store and Retrieval Setup
def get_embedding_model():
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            return FakeEmbeddings(size=384)


def initialize_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    docs = load_knowledge_base_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(chunks, embeddings)
    _retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return _retriever


# GroqCloud LLM Initialization
def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "250"))

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is missing or unconfigured. Please set it in .env file.")

    _llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=max_tokens
    )
    return _llm


# Pipeline Execution
def run_rag_pipeline(question: str) -> Dict[str, Any]:
    start_time = time.perf_counter()

    retriever = initialize_retriever()
    retrieved_docs = retriever.invoke(question)

    context_chunks = []
    sources = []
    for doc in retrieved_docs:
        context_chunks.append(doc.page_content)
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page_info = doc.metadata.get("page", None)
        sources.append({"source": source_name, "page": page_info})

    combined_context = "\n\n---\n\n".join(context_chunks)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    llm = get_llm()
    chain = prompt_template | llm | StrOutputParser()
    answer_text = chain.invoke({"context": combined_context, "question": question})

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "answer": answer_text.strip(),
        "sources": sources,
        "latency_ms": latency_ms
    }
