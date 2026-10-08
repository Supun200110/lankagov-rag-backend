import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ==================================================
# 1. Load environment variables
# ==================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env"
    )


# ==================================================
# 2. Create FastAPI application
# ==================================================

app = FastAPI(
    title="LankaGov AI API",
    description="Sri Lankan Government Services RAG API",
    version="1.0.0"
)


# ==================================================
# 3. Enable CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 4. Load embedding model
# ==================================================

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==================================================
# 5. Connect to ChromaDB
# ==================================================

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="lankagov_passport"
)


# ==================================================
# 6. Create Retriever
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ==================================================
# 7. Connect to OpenRouter
# ==================================================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# ==================================================
# 8. RAG Prompt
# ==================================================

prompt = ChatPromptTemplate.from_template("""

You are LankaGov AI, a Sri Lankan Government
Services Assistant.

Answer the user's question ONLY using the
provided government document context.

Rules:

1. Do not invent information.
2. Do not use information outside the provided context.
3. If the answer cannot be found in the context,
   say that the information could not be found
   in the available official documents.
4. Give a clear and simple answer.
5. Do not make assumptions.

Government document context:

{context}

User question:

{question}

""")


# ==================================================
# 9. Request model
# ==================================================

class ChatRequest(BaseModel):

    question: str


# ==================================================
# 10. Root endpoint
# ==================================================

@app.get("/")
def home():

    return {
        "message": "🇱🇰 LankaGov AI API is running",
        "version": "1.0.0"
    }


# ==================================================
# 11. Chat endpoint
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    # ----------------------------------------------
    # Retrieve relevant documents
    # ----------------------------------------------

    documents = retriever.invoke(question)


    if not documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the available official documents."
            ),
            "sources": []
        }


    # ----------------------------------------------
    # Create context
    # ----------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # ----------------------------------------------
    # Create RAG prompt
    # ----------------------------------------------

    messages = prompt.format_messages(
        context=context,
        question=question
    )


    # ----------------------------------------------
    # Generate AI answer
    # ----------------------------------------------

    response = llm.invoke(messages)


    # ----------------------------------------------
    # Prepare sources
    # ----------------------------------------------

    sources = []

    seen_sources = set()


    for document in documents:

        source = document.metadata.get(
            "source_file",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            None
        )

        page_number = (
            page + 1
            if page is not None
            else None
        )


        source_key = (
            source,
            page_number
        )


        if source_key not in seen_sources:

            sources.append({
                "document": source,
                "page": page_number
            })

            seen_sources.add(source_key)


    # ----------------------------------------------
    # Return response
    # ----------------------------------------------

    return {
        "answer": response.content,
        "sources": sources
    }