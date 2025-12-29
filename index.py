"""
COST-OPTIMIZED CORRECTIVE RAG (OPENAI + PDF)

Optimizations:
- FAISS index persistence (no re-embedding)
- Context evaluation caching
- Query refinement caching
- Skip evaluation if retrieval similarity is high
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import hashlib
from typing import List, Dict
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY not found in environment. Create a .env with OPENAI_API_KEY or set it in the environment.")

# ============================================================
# CONFIG
# ============================================================
cwd = os.getcwd()
print("📁 Current working directory:", cwd)

PDF_PATH = f"{cwd}/sources/Third-Avartana-opens-at-ITC-M.pdf"


# .stem extracts the filename without the extension
filename_without_extension = Path(PDF_PATH).stem
FAISS_PATH = f"{cwd}/FAISS_store/{filename_without_extension}_index"

SIMILARITY_SKIP_THRESHOLD = 0.85   # Skip evaluation if confident


# ============================================================
# OPENAI MODELS
# ============================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ============================================================
# LOAD + CHUNK PDF
# ============================================================
def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ============================================================
# VECTOR STORE (PERSISTED)
# ============================================================
if os.path.exists(FAISS_PATH):
    print("🔁 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("🆕 Creating FAISS index (one-time cost)...")
    pdf_chunks = load_and_chunk_pdf(PDF_PATH)
    vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ============================================================
# PROMPTS
# ============================================================
EVALUATION_PROMPT = ChatPromptTemplate.from_template("""
You are a Corrective RAG evaluator.

Query:
{query}

Retrieved Context:
{context}

Evaluate relevance, completeness, accuracy, and specificity (0–1).
Classify overall quality: EXCELLENT, GOOD, FAIR, POOR.

Respond ONLY in JSON:
{{
  "relevance": number,
  "completeness": number,
  "accuracy": number,
  "specificity": number,
  "overall_quality": "EXCELLENT | GOOD | FAIR | POOR",
  "reasoning": string
}}
""")

REFINE_QUERY_PROMPT = ChatPromptTemplate.from_template("""
Original query:
{query}

Why refinement is needed:
{reasoning}

Generate a better search query.
Only output the query.
""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Answer using ONLY the context below.

Question:
{query}

Context:
{context}
""")


# ============================================================
# CACHES (CRITICAL FOR COST)
# ============================================================
evaluation_cache: Dict[str, Dict] = {}
refinement_cache: Dict[str, str] = {}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ============================================================
# CONTEXT EVALUATION (CACHED)
# ============================================================
def evaluate_context(query: str, docs: List[Document]) -> Dict:
    context_text = "\n".join(d.page_content for d in docs)
    key = _hash(query + context_text)

    if key in evaluation_cache:
        return evaluation_cache[key]

    response = llm.invoke(
        EVALUATION_PROMPT.format_messages(
            query=query,
            context=context_text
        )
    )

    result = json.loads(response.content)
    evaluation_cache[key] = result
    return result


# ============================================================
# QUERY REFINEMENT (CACHED)
# ============================================================
def refine_query(query: str, reasoning: str) -> str:
    if query in refinement_cache:
        return refinement_cache[query]

    response = llm.invoke(
        REFINE_QUERY_PROMPT.format_messages(
            query=query,
            reasoning=reasoning
        )
    )

    refined = response.content.strip()
    refinement_cache[query] = refined
    return refined


# ============================================================
# ANSWER GENERATION
# ============================================================
def generate_answer(query: str, docs: List[Document]) -> str:
    context_text = "\n".join(d.page_content for d in docs)

    response = llm.invoke(
        ANSWER_PROMPT.format_messages(
            query=query,
            context=context_text
        )
    )
    return response.content


# ============================================================
# CORRECTIVE RAG PIPELINE (OPTIMIZED)
# ============================================================
def corrective_rag(query: str, MAX_CORRECTION_ATTEMPTS: int) -> Dict:
    attempt = 0
    note = None

    # Initial retrieval
    docs = retriever.invoke(query)

    while attempt <= MAX_CORRECTION_ATTEMPTS:
        evaluation = evaluate_context(query, docs)

        # If quality is sufficient, stop correcting
        if evaluation["overall_quality"] in ["GOOD", "EXCELLENT"]:
            break

        # If max attempts reached, stop correcting
        if attempt == MAX_CORRECTION_ATTEMPTS:
            note = (
                "⚠️ Maximum corrective attempts reached. "
                "Answer may be incomplete."
            )
            break

        # Otherwise, refine and retry
        refined_query = refine_query(query, evaluation["reasoning"])
        docs = retriever.invoke(refined_query)

        attempt += 1

    # Final answer (always produce something)
    answer = generate_answer(query, docs)

    confidence = (
        "HIGH" if evaluation["overall_quality"] == "EXCELLENT"
        else "MEDIUM" if evaluation["overall_quality"] == "GOOD"
        else "LOW"
    )

    return {
        "context_quality": evaluation["overall_quality"],
        "confidence": confidence,
        "answer": answer,
        "note": note
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    query = "What is Avartana planning to do?"

    result = corrective_rag(query, 2)

    print("\n🔍 Context Quality:", result["context_quality"])
    print("📊 Confidence Level:", result["confidence"])
    print("🎯 Answer:\n", result["answer"])

    if result["note"]:
        print(result["note"])
