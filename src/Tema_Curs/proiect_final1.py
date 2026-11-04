"""Cerinta:
Cand este rulat din terminal, incarca toate fisierele dintr-un folder, si le indexeaza in baza de date ChromaDB, similar cum am facut la curs.
Folositi o strategie de chunk-ing la alegere, pentru toate documentele indexate. Parsati documentele folosind orice librarie de parsare documente, puteti sa le convertiti in format text simplu, sau markdown.
Pentru a converti fisierele in text, pe pagini, puteti folosi:
pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
Acceptati documente de tipul .txt, .pdf .
Generati metadate pentru fiecare chunk, in care puneti un "source", care este numele documentului, "page", care este pagina documentului de unde a fost extras acel chunk, si un chunk number. Puteti numerota chunk-urile in orice fel doriti, puteti sa le numarati de la 1 la n, cand le generati pe fiecare.

Puteti folositi collection.upsert pentru a insera documentele in Chroma DB.
Primul program trebuie doar sa treaca prin toate documentele si sa le indexeze."""

from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


DATA_SOURCES_DIR = Path("data_sources_project")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "text_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
N_RESULTS = 3

#1.citim toate fisierele din folderul 'data_source_project' in functie de extensia fisierului
def read_file_pages(file_path: Path) -> list[tuple[int, str]]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return [
            (int(page_number), str(page.extract_text()) or "")
            for page_number, page in enumerate(reader.pages, start=1)

        ]

    if suffix == ".txt":
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return [(1, content)]

    raise ValueError(f"Format de fișier nesuportat: {suffix}")

#2. definim chunk_text ca o lista de chunckuri (impartim textul in chunkuri)
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        # se parcurge textul  utilizand start ca index de start
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # updatare index cu un chunk
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


#3. Indexam un singur fisier in ChromaDB
def index_file(collection, model, file_path: Path) -> int:

    records = []

    # Obtinem chunk-urile pentru fiecare pagina
    for page_number, page_text in read_file_pages(file_path):

        for chunk_number, chunk in enumerate(
            chunk_text(page_text),
            start=1
        ):

            records.append(
                (chunk, page_number, chunk_number)
            )





#4.Stergem datele preexistente in baza de date
    collection.delete(where={"source": file_path.name})

    if not records:
        f"Nu s-a gasit niciun text indexat{file_path.name}"
        return 0


#5.Generam metadatele
    ids = []
    documents = []
    metadatas = []

    for record in records:
        chunk = record[0]
        page_number = record[1]
        chunk_number = record[2]

        documents.append(chunk)

        metadatas.append({
            "source": file_path.name,
            "page": page_number,
            "chunk": chunk_number,
        })

        ids.append(
            f"{file_path.name}:p{page_number}:c{chunk_number}"

        )
    #pentru testare
    print(file_path.name, ":",documents, page_number, chunk_number)



    #6. adaugam datele in ChromaDB
    collection.upsert(

        ids=ids,
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(), #vectorizarea
        documents=documents,
        metadatas=metadatas,
    )
    print(f" {file_path.name}: {len(documents)} chunk-uri indexate.")
    return len(documents)

#7. Indexam toate fisierele din folder
def index_data_sources(collection, model) -> None:

    if not DATA_SOURCES_DIR.exists():

        DATA_SOURCES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Folderul a fost creat: {DATA_SOURCES_DIR}"
        )

        return

    files = sorted(
        path
        for path in DATA_SOURCES_DIR.iterdir()
        if path.is_file()
    )

    if not files:

        print(
            f"Folderul {DATA_SOURCES_DIR} nu contine fisiere."
        )

        return

    for file_path in files:

        try:
            index_file(collection, model, file_path)

        except Exception as error:

            print(
                f"Eroare la fisierul {file_path.name}: {error}"
            )



def main():
    load_dotenv()

    model = SentenceTransformer(MODEL_NAME)

    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    index_data_sources(collection, model)

if __name__ == "__main__":
    main()
