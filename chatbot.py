import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# --------------------------------------------------
# 2. Load the same embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to existing ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="lankagov_passport"
)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 5. Connect to OpenRouter
# --------------------------------------------------

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# --------------------------------------------------
# 6. Create RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are LankaGov AI, a Sri Lankan Government Services
assistant.

Answer the user's question ONLY using the provided
government document context.

Rules:
- Do not invent information.
- Do not make assumptions.
- If the answer is not available in the context,
  say that the information could not be found in
  the available official documents.
- Give a clear and simple answer.
- Mention the source document when possible.

Government document context:
{context}

User question:
{question}
""")


# --------------------------------------------------
# 7. Ask a question
# --------------------------------------------------

def ask_question(question):

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    if not documents:
        return "I could not find relevant information in the available government documents."

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # Ask the LLM
    response = llm.invoke(messages)

    return response.content, documents


# --------------------------------------------------
# 8. Start chatbot
# --------------------------------------------------

print("\n🇱🇰 LankaGov AI Assistant")
print("--------------------------------")
print("Ask a question about Sri Lankan passport services.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye! 👋")
        break

    try:

        answer, documents = ask_question(question)

        print("\n🤖 LankaGov AI:")
        print(answer)

        print("\n📚 Sources:")

        shown_sources = set()

        for document in documents:

            source = document.metadata.get(
                "source_file",
                "Unknown source"
            )

            page = document.metadata.get(
                "page",
                "Unknown page"
            )

            source_key = (source, page)

            if source_key not in shown_sources:

                print(
                    f"- {source} | Page {page + 1}"
                )

                shown_sources.add(source_key)

        print("\n" + "-" * 50)

    except Exception as e:

        print("\n❌ Error:")
        print(e)