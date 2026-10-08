import os

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ==================================================
# 1. Page Configuration
# ==================================================

st.set_page_config(
    page_title="LankaGov AI",
    page_icon="🇱🇰",
    layout="wide"
)


# ==================================================
# 2. Load Environment Variables
# ==================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found.")
    st.stop()


# ==================================================
# 3. Application Header
# ==================================================

st.title("🇱🇰 LankaGov AI")

st.subheader(
    "Sri Lankan Government Services Assistant"
)

st.write(
    "Ask questions about Sri Lankan government services "
    "using information from official government documents."
)


# ==================================================
# 4. Load Embedding Model
# ==================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ==================================================
# 5. Connect to ChromaDB
# ==================================================

@st.cache_resource
def load_vector_store():

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="lankagov_passport"
    )


vector_store = load_vector_store()


# ==================================================
# 6. Create Retriever
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ==================================================
# 7. Connect to OpenRouter
# ==================================================

@st.cache_resource
def load_llm():

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )


llm = load_llm()


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
2. Do not use information that is not present
   in the provided context.
3. If the answer cannot be found in the context,
   clearly say that the information could not be
   found in the available official documents.
4. Give a clear and simple answer.
5. Do not make assumptions.
6. Mention relevant source documents when possible.

Government document context:

{context}

User question:

{question}

""")


# ==================================================
# 9. RAG Function
# ==================================================

def ask_lankagov(question):

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    if not documents:

        return (
            "I could not find relevant information "
            "in the available official documents.",
            []
        )

    # Combine retrieved document content
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(messages)

    return response.content, documents


# ==================================================
# 10. Question Input
# ==================================================

st.divider()

st.markdown("### 💬 Ask your question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What documents are required "
        "to apply for a passport?"
    ),
    height=100
)


# ==================================================
# 11. Ask Button
# ==================================================

if st.button(
    "🔍 Ask LankaGov",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching government documents..."
        ):

            try:

                answer, documents = ask_lankagov(
                    question
                )

                # ==================================
                # Answer
                # ==================================

                st.divider()

                st.markdown("### 🤖 Answer")

                st.write(answer)


                # ==================================
                # Sources
                # ==================================

                st.markdown("### 📚 Sources")

                shown_sources = set()

                for document in documents:

                    source = document.metadata.get(
                        "source_file",
                        "Unknown source"
                    )

                    page = document.metadata.get(
                        "page",
                        None
                    )

                    if page is not None:

                        page_number = page + 1

                        source_key = (
                            source,
                            page_number
                        )

                        if source_key not in shown_sources:

                            st.write(
                                f"📄 {source} — "
                                f"Page {page_number}"
                            )

                            shown_sources.add(
                                source_key
                            )

                    else:

                        st.write(
                            f"📄 {source}"
                        )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )


# ==================================================
# 12. Sidebar
# ==================================================

with st.sidebar:

    st.header("🇱🇰 LankaGov AI")

    st.write(
        "AI-powered assistant for understanding "
        "Sri Lankan government services."
    )

    st.divider()

    st.markdown("### 📌 Current Services")

    st.write("🛂 Passport Services")

    st.divider()

    st.markdown("### ⚙️ Technology")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• LangChain")
    st.write("• ChromaDB")
    st.write("• Hugging Face")
    st.write("• OpenRouter")

    st.divider()

    st.caption(
        "Information is generated from the "
        "available government documents."
    )