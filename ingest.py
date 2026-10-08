from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

DATA_FOLDER = Path("data/passport")
CHROMA_FOLDER = "chroma_db"


# --------------------------------------------------
# 2. Load PDF documents
# --------------------------------------------------

documents = []

pdf_files = list(DATA_FOLDER.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")


for pdf_file in pdf_files:

    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pdf_documents = loader.load()

    # Add the file name to metadata
    for document in pdf_documents:
        document.metadata["source_file"] = pdf_file.name

    documents.extend(pdf_documents)


print(f"Total pages loaded: {len(documents)}")


# --------------------------------------------------
# 3. Split documents into smaller chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 4. Create embedding model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# 5. Store embeddings in ChromaDB
# --------------------------------------------------

print("Creating ChromaDB...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_FOLDER,
    collection_name="lankagov_passport"
)


print("===================================")
print("RAG ingestion completed successfully!")
print("Documents:", len(documents))
print("Chunks:", len(chunks))
print("Vector database:", CHROMA_FOLDER)
print("===================================")