import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document as DocxDocument

# Config
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
DATA_SOURCES_DIR = Path("data_sources")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
N_RESULTS = 21


#Split the text in chunks"
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP should be smaller than CHUNK_SIZE.")

    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step

        return chunks


def read_file_pages(file_path: Path) -> list[tuple[int, str]]:
    """
    Return a list(page number, text).
    PDF  -  read page by page and txt - only one page
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]


    if suffix==".txt" :
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return [(1, content)]

    print(f"Format not supported, the file ignored: {file_path.name}")


    if suffix == ".docx" :
        doc = DocxDocument(str(file_path))
        parts = [
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        ]
    return []

def index_file(collection, model, file_path: Path) -> int:
    """split the file and save the chunks in ChromaDB."""
    records = []


    for page_number, page_text in read_file_pages(file_path):
        for chunk_number, chunk in enumerate(chunk_text(page_text), start=1):
            records.append((chunk, page_number, chunk_number))

    # delete the duplicates
    collection.delete(where={"source": file_path.name})

    if not records:
        print(f"Was not found any indexed text {file_path.name}.")
        return 0

    documents = [record[0] for record in records]
    metadatas = [
        {
            "source": file_path.name,
            "page": int(record[1]),
            "chunk": int(record[2]),
        }
        for record in records
    ]
    ids = [
        f"{file_path.name}:p{page_number}:c{chunk_number}"
        for _, page_number, chunk_number in records
    ]
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f" {file_path.name}: {len(documents)} chunk-uri.")
    return len(documents)

#Indexing  the files from data_source

def index_data_sources(collection, model) -> None:

    if not DATA_SOURCES_DIR.exists():
        DATA_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        print(
            f"The folder created '{DATA_SOURCES_DIR}'. "
             )
        return

    files = sorted(path for path in DATA_SOURCES_DIR.iterdir() if path.is_file())
    if not files:
        print(f"The folder '{DATA_SOURCES_DIR}' doesn't contain files.")
        return

    for file_path in files:
        try:
            index_file(collection, model, file_path)
        except Exception as error:
            print(f"The file couldn't be indexed{file_path.name}: {error}")

def answer_from_file(client, collection, model, filename: str, question: str) -> str:
    "send the context to LLM."
    filename = Path(filename.strip()).name

    # Check the already exsting chunks from source
    stored = collection.get(
        where={"source": filename},
        include=["metadatas"],
        limit=1,
        )

    if not stored.get("ids"):
        return (
            f"Doesn't exist indexed data in the file '{filename}'. "

        )

    query_embedding = model.encode(question, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
        where={"source": filename},
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return f" no relevant data in '{filename}'."

    context_parts = []
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", filename)
        page = metadata.get("page", "unknown")
        chunk = metadata.get("chunk", "unknown")
        context_parts.append(
            f"[Source: {source}; page: {page}; chunk: {chunk}]\n{document}"
        )
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=[
            {
                "role": "system",
                "content": ("""
  1.Use only the information from the paragraphs
  2.Don't use invented information, pages, or other text
  3.Respond in English or Romanian, depending on the language used by the client
  4.If the paragraph doesn't contain the answer you have to write this very clear
  5.Mention the source file, number of page if this exists at each of the important information
  6.If the Number of the page doesn't exist, mention that the page number doesn't exist
  7.At the end add a section: "Source" with the name of the file, pages and the rows of the paragraphs used """
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File requested: {filename}\n"
                    f"Question: {question}\n\n"
                    f"Paragraphs retrieved:\n{context}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or "LLM doesn't return a response."


def main():
    load_dotenv()

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    model_name = os.getenv("OPEN_ROUTER_MODEL_NAME")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    model = SentenceTransformer(MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Index of all files .
    index_data_sources(collection, model)

    print("\nWrite 'quit' at the end")
    while True:
        filename = input("\nName of file (din data_sources): ").strip()
        if filename.lower() == "quit":
            break
        if not filename:
            print("Insert the name of the file")
            continue

        question = input("Question: ").strip()
        if not question:
            print("Make a question.")
            continue

        try:
            answer = answer_from_file(client, collection, model, filename, question)
            print(f"\nAgent> {answer}")
        except Exception as error:
            print(f"The answer cannot be added: {error}")


if __name__ == "__main__":
    main()