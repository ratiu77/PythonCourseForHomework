"""Cerinta:
Cand este rulat din terminal, incarca toate fisierele dintr-un folder, si le indexeaza in baza de date ChromaDB, similar cum am facut la curs.
Folositi o strategie de chunk-ing la alegere, pentru toate documentele indexate. Parsati documentele folosind orice librarie de parsare documente, puteti sa le convertiti in format text simplu, sau markdown.
Pentru a converti fisierele in text, pe pagini, puteti folosi:
pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
Acceptati documente de tipul .txt, .pdf .
Generati metadate pentru fiecare chunk, in care puneti un "source", care este numele documentului, "page", care este pagina documentului de unde a fost extras acel chunk, si un chunk number. Puteti numerota chunk-urile in orice fel doriti, puteti sa le numarati de la 1 la n, cand le generati pe fiecare.

Puteti folositi collection.upsert pentru a insera documentele in Chroma DB.
Primul program trebuie doar sa treaca prin toate documentele si sa le indexeze."""

from pathlib import Path #pentrul lucrul cu fisiere
from pypdf import PdfReader #pentru extragerea textului din pdf
from sentence_transformers import SentenceTransformer #pentru transformarea textului in vectori - embeddings
import chromadb #pentru stocarea vectorilor si indexarea lor


DATA_SOURCES_DIR = Path("data_sources_project")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "text_chunks"

CHUNK_SIZE = 800 #dimensiunea maxima a unui chunk
CHUNK_OVERLAP = 120 #numarul de caractere care se suprapun intre doua chunk-uri consecutive


#1.citim toate fisierele din folderul 'data_source_project' in functie de extensia fisierului
def read_file_pages(file_path: Path) -> list[tuple[int, str]]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)

        ]

    if suffix == ".txt":
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return [(1, content)]

    raise ValueError(f"Format de fișier nesuportat: {suffix}")

#2. definim chunk_text ca o lista de chunckuri (impartim textul in chunkuri)
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0 # pozitia de la care incepe extragerea primului chunk

    while start < len(text):
        # se parcurge textul  utilizand start ca index de start
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # pastram o parte din textutul anterior in urmatorul chunk
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


#3. Indexam un singur fisier in ChromaDB
def index_file(collection, model, file_path: Path) -> int:

    records = []

    # Obtinem chunk-urile pentru fiecare pagina, apelam functia read_file_pages pentru a extrage datele din document
    for page_number, page_text in read_file_pages(file_path):

    #impartim textul fiecarei pagini in chunkuri si le numerotam si le adaugam in records
        for chunk_number, chunk in enumerate(
            chunk_text(page_text),
            start=1
        ):

            records.append(
                (chunk, page_number, chunk_number)
            )





#4.Stergem datele preexistente in baza de date
    collection.delete(where={"source": file_path.name})

#verificam daca documentul a generat cel putin un chunk
    if not records:
        print(f"Nu s-a gasit niciun text indexat{file_path.name}")
        return 0


#5.Generam metadatele asociate fiecarui chunk
    ids = []
    documents = []
    metadatas = []

    for record in records:
        chunk = record[0]
        page_number = record[1]
        chunk_number = record[2]

        documents.append(chunk) # adaugam textul chunkului in documents

        metadatas.append({
            "source": file_path.name,
            "page": page_number,
            "chunk": chunk_number,
        })

        ids.append(
            f"{file_path.name}:p{page_number}:c{chunk_number}"

        )
    #pentru testare
    print(
    f"Fisier: {file_path.name} | "
    f"Numar chunk-uri: {len(documents)}"
)


    #6. adaugam datele in ChromaDB
    #Transformam toate chunk-urile in embeddings folosind modelul SentenceTransformer
    #upsert adauga inregistrarile noi sau actualizeaza inregistrarile
    collection.upsert(

        ids=ids,
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(), #normalizeaza vectorii pentru comparatii de similaritate
        documents=documents,
        metadatas=metadatas,
    )
    print(f" {file_path.name}: {len(documents)} chunk-uri indexate.")
    return len(documents)

#7. Indexam toate fisierele din folder
def index_data_sources(collection, model) -> None:

    if not DATA_SOURCES_DIR.exists():

        DATA_SOURCES_DIR.mkdir(
            parents=True,#permite crearea si a folderelor parinte
            exist_ok=True #evita eroarea daca folderul exista deja
        )

        print(
            f"Folderul a fost creat: {DATA_SOURCES_DIR}"
        )
         # Oprim functia deoarece folderul nou creat nu contine documente
        return

    files = sorted(
        path
        for path in DATA_SOURCES_DIR.iterdir()
        if path.is_file()  and path.suffix.lower() in [".pdf", ".txt"]
    )

    if not files:

        print(
            f"Folderul {DATA_SOURCES_DIR} nu contine fisiere."
        )



    for file_path in files:

        try:
            index_file(collection, model, file_path)

        except Exception as error:

            print(
                f"Eroare la fisierul {file_path.name}: {error}"
            )



def main():

    model = SentenceTransformer(MODEL_NAME) # init model

    # cream un client chromaDB care salveaza baza de date pe disc
    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


    # Apelam functia care parcurge folderul si indexeaza toate documentele
    index_data_sources(collection, model)

if __name__ == "__main__":
    main()
