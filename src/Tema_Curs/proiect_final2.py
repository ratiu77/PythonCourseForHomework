# Creati al doilea program, care este un agent LLM, care raspunde la intrebarile utilizatorului, din terminal.
# El are la dispozitie un tool, numit RAG search. Implementati acest tool, care primeste intrebarea utilizatorului, si returneaza un string simplu, care va fi contextul pentru agent. Validati si input-ul pentru tool, si output-ul, acel string simplu, folosind pydantic.
#
# Pentru tool-ul RAG Search, puteti sa-l construiti folosind Chroma DB, semantic sau hybrid search in documentele indexate in colectia aleasa din baza de date.
# Folosind acea intrebare se face semantic/hybrid search in Chroma DB, selectati 10 rezultate relevante din baza de date, si le returnati din tool, ca un string simplu. Conectati tool-ul la un agent LLM (agent construit folosind pachetul python openai), si output-ul de la tool, puneti-l in mesajele agentului, pentru a putea raspunde la intrebarea utilizatorului.
# Folosind logica de a construi un agent de la curs, construiti agentul, si tool-ul care are acces la informatiile din baza de date.
# Oferiti raspuns utilizatorului, in terminal.
#
# Puneti in acel folder cel putin 3 pdf-uri sau fisiere text, pentru a le procesa si a folosi RAG search, si puneti intrebari agentului, legat de informatii din acele fisiere.
# Folositi pydantic pentru a valida intrebarea data de utilizator, ca string.
#
# Pentru baza de date Chroma DB, intr-un proiect de python, manage-uit cu uv, unde ati rulat deja uv add chromadb, puteti rula urmatoarea comanda din terminal pentru a porni o baza de date chroma, locala:
# uv run chroma run --path chroma_data --port 8000
#
# Si pentru a va conecta la ea, puteti folosi urmatorul cod python?
#
# import chromadb
#
# client = chromadb.HttpClient(host="localhost", port=8000)
# collection = client.get_or_create_collection("my_collection")


import chromadb
import os
import json
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from openai import OpenAI#clasa folosita pentru a crea un client API
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi #importa clasa BM250kapi BM25 un algoritm de cautare textuala
import re #folosit pentru Regex, cautare  si identificarea anumitor tipare dintr-un text

#pentru rularea aplicatiei, ChromaDB porneste un server local
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("my_collection")

print("Conectat cu succes!")
print("Heartbeat:", client.heartbeat())
print("Colecție:", collection.name)

DATA_SOURCES_DIR = Path(__file__).parent / "data_sources_project_2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "text_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
HYBRID_SEARCH_RESULT_COUNT = 10

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using a knowledge base
of indexed documents. ALWAYS call search_file before answering any
factual or informational question, even if you think you know the answer.
Answer only from the retrieved context. If the context does not contain
the answer, say that the information was not found in the documents.
Do not use your o   wn general knowledge. Always mention the source
(file name and page) from the context."""

#Validare pydantic input si output pentru tool
#Input - intrebarea utilizatorului

## Configurăm Pydantic să respingă câmpurile suplimentare
class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

#intrebarea trebuie sa fie de tip text si sa contina cel putin un caracter
    question: str = Field(
        min_length=1,
        max_length=3000,
        description="Question about the contents of the indexed documents",
    )

#validarea contextului

class SearchResponse(BaseModel):
    """Output-ul tool-ului: un string folosit ca si context pentru agent."""

    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        min_length=1,
        description="Context retrieved from the indexed documents",
    )


# Citim fisierele pdf si le introducem intr-o lista de tupluri cu numarul paginii si textul extras (indexarea documentelor)
def read_file_pages(file_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))

    pages=[]
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages



# Tokenizarea textului

def tokenize(text: str) -> list[str]:
    #regex identifică secvențele de caractere care formează cuvinte și returnează o listă cu acestea.
    #folosita pentru cautarea BM25
    return re.findall(r"\b\w+\b", text.lower())


# Impartim textul in chunkuri
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0 #pozitia de la care incepe extragerea primului fragment

    while start < len(text):
    # se parcurge textul  utilizand start ca index de start
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    # urmatorul chunk va contine si o parte din caracterele primului chunk (zona comuna)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

#Indexam un fisier

def index_file(collection, model, file_path: Path) -> int:

    records = []

    #   # Obtinem chunk-urile pentru fiecare pagina, apelam functia read_file_pages pentru a extrage datele din document
    for page_number, page_text in read_file_pages(file_path):

        # Împărțim textul paginii curente în fragmente.
        page_chunks = chunk_text(page_text)

        # Numerotăm fragmentele din pagina curentă începând de la 1.
        for chunk_number, chunk in enumerate( page_chunks, start=1):


            records.append(
                (chunk, page_number, chunk_number)
            )
#Stergem datele preexistente in baza de date
    collection .delete(where={"source": file_path.name})

    if not records:
        print (f"Nu s-a gasit niciun text indexat{file_path.name}")
        return 0

#Generam metadata
    ids=[]  #identificatorul unic pentru fiecare fragment
    documents = [] # textul fragmentelor
    metadatas = [] #informatii despre fragmente

    for chunk, page_number, chunk_number in records:

        documents.append(chunk) # adaugam textul fragmentului in lista de documente

        metadatas.append({
            "source": file_path.name,
            "page": page_number,
            "chunk": chunk_number,

        })
        ids.append (
            f"{file_path.name}:p{page_number}:c{chunk_number}")


#Adaugam fragmentele in baza de date vectoriala ChromaDB
    collection.upsert(

        ids=ids,
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(), #normalizarea vectorilor
        documents=documents,
        metadatas=metadatas,
    )
    print(f" {file_path.name}: {len(documents)} chunk-uri indexate.")

    #returnam numarul de fragmente indexate
    return len(documents)

#Indexam toate fisierele din folder
def index_data_sources(collection, model) -> None:

    if not DATA_SOURCES_DIR.exists():

        DATA_SOURCES_DIR.mkdir(
            parents=True, #permite crearea și a folderelor părinte lipsă.
            exist_ok=True #evită eroarea dacă folderul există deja
        )

        print(
            f"Folderul a fost creat"
        )

        return

#  Obținem fișierele din folder și le sortăm
    files = sorted(
        path
        for path in DATA_SOURCES_DIR.iterdir()
        if path.is_file() #exclude subfolderele
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

#Incarcarea fragmentelor indexate in ChromaDB
def load_indexed_documents(collection):

    result = collection.get(
        include=["documents", "metadatas"]
    )

    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    ids = result["ids"]

    return documents, metadatas, ids


#  Cautarea RAG (hibrida: semantica + BM25)
def rag_search(collection, model, bm25, documents, metadatas, ids, args:str,) -> str:
    """
        Primeste argumentele tool-ului (JSON string), valideaza input-ul(Intrebarea),
        face hybrid search si returneaza un string simplu validat cu pydantic, construieste contextul.
        """

    #Transformăm șirul JSON în dicționar Python și validăm
    request = SearchRequest.model_validate(json.loads(args))
    question= request.question

    #in cazul in care nu exista docuemnte idnexate
    if not documents:
        return SearchResponse(context="Nu exista document indexat").context


    #Cautarea semantica Chroma DB - transformam intrebarea intr-un embedding
    question_vector = model.encode(question, normalize_embeddings=True).tolist()

    # folosim 2 rezultate si le combinam pentru un scor hibrid


   #Căutăm în ChromaDB fragmentele ale căror embeddings sunt cele mai apropiate de embedding-ul întrebării.
    semantic_results = collection.query(
        query_embeddings=[question_vector],
        # Cerem cel mult HYBRID_SEARCH_RESULT_COUNT rezultate,
        # dar niciodată mai multe decât numărul documentelor existente.
        n_results=min (HYBRID_SEARCH_RESULT_COUNT,len(documents)),
        #documentele si metadatele rezultate
        include=["documents", "metadatas"],
    )


    # Mapam ID-urile la indexurile documentelor (cheia este indexul fragmentului/chunkulu)
    id_to_index = {
        doc_id: index
        for index, doc_id in enumerate(ids)
    }

    # Dicționarul hybrid_scores va reține scorul combinat
    hybrid_scores = {}


    #Scorul cautarii semantice
    # Rank 0 primeste 1 punct, rank 1 primeste 1/2 puncte etc

    semantic_ids = semantic_results["ids"][0]
    #Parcurgem rezultatele în ordinea relevanței.
    for rank, doc_id in enumerate(semantic_ids):
        #Transformăm ID-ul ChromaDB în poziția documentului
        chunk_index = id_to_index[doc_id]

        #Acordăm un scor în funcție de poziția rezultatului:
        #locul 1 primește 1 punct;
        # locul 2 primește 1/2 puncte;
        # locul 3 primește 1/3 puncte  etc
        hybrid_scores[chunk_index] = 1 / (rank + 1)

    # Ranking BM25
    #tokenizam intrebarea si cerem BM25 sa calculeze un scor pentru fiecare fragment indexat
    bm25_scores = bm25.get_scores(tokenize(question))

    # Vom salva perechi de forma (scor, index_fragment).
    bm25_results = []
    #Primul fragment are indexul 0.
    chunk_index = 0

    # Asociem fiecare scor BM25 cu indexul fragmentului corespunzător.
    for score in bm25_scores:
        bm25_results.append((score, chunk_index))
        chunk_index += 1
        # Sortăm descrescător după scor (de la cel mai mare scor)
    bm25_results.sort(reverse=True)

    #Luăm primele HYBRID_SEARCH_RESULT_COUNT rezultate BM25
    for rank, (score, chunk_index) in enumerate(bm25_results[:HYBRID_SEARCH_RESULT_COUNT]):
        hybrid_scores[chunk_index] = hybrid_scores.get(chunk_index, 0) + 1 / (rank + 1)

    #Transformăm dicționarul într-o listă de perechi:(scor_hibrid, index_fragment)

    combined_results = []
    for chunk_index, score in hybrid_scores.items():
        combined_results.append((score, chunk_index))
    combined_results.sort(reverse=True) # Sortăm descrescător: fragmentele cu scorul cel mai mare vor apare primele

#Construim contextul
    results = [] #lista fragmentelor selectate
    for _, chunk_index in combined_results[:HYBRID_SEARCH_RESULT_COUNT]:
        results.append(
            f"{documents[chunk_index]}\nSource: {metadatas[chunk_index]['source']}, page {metadatas[chunk_index]['page']}"
        )
     #unim fragmentele intr-un singur sir de caractere separate de doua linii
    context = "\n\n".join(results) if results else "NU s-au gasit informatii relevante"

    # validare context final cu pydantic
    return SearchResponse(context=context).context


#LLM
def complete(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict],
):
    #Trimitem conversația către model prin clientul OpenAI,comunica cu OpenRouter
    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),#citirea numelui modelului
        messages=messages,#Istoricul conversației: system, user, assistant și tool.
        tools=tools,#instrumentele pe care le poate apela
        tool_choice="auto",#modelul decide daca re nevoie sau nu sa apeleze un tool
    )
    #returnam mesajul generat de model
    return response.choices[0].message






def main():
    load_dotenv() #Încărcăm variabilele de mediu din fișierul .env.

    #initializam clientul OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    #load embedding model
    model = SentenceTransformer(MODEL_NAME)

    #  Creează clientul ChromaDB care salvează datele permanent pe disc.
    # Dacă aceasta nu există, get_or_create_collection o creează.
    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


    # Citim și indexăm fișierele din folderul DATA_SOURCES_DIR.
    index_data_sources(collection, model)

    # Descarcam fragmentele indexate
    documents, metadatas, ids = load_indexed_documents(collection)

    if not documents:
        print("Nu exista documente indexate.")
        return

    #Pregatirea indexului BM25
    tokenized_documents = []#versiunea tokenized a fiecarui document

    # Parcurgem toate fragmentele încărcate din ChromaDB.
    for doc in documents:
        # Împărțim fragmentul în cuvinte.
        tokens = tokenize(doc)
        tokenized_documents.append(tokens)

    #indexul BM25 va fi folosit pentru cautarea lexicala
    bm25 = BM25Okapi(tokenized_documents)

    #definim toolul pe care il poate apela LLM-ul
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_file",# # Numele funcției pe care modelul o poate solicita.
                #Se explica modelului cand trebuie utilizata functia
                "description": ("""Search the indexed PDF files. 
                    Use this tool for questions about 
                    PDF documents or their contents."""),
                #Structura argumentelor functiei
                "parameters":  {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Question about the contents of the indexed documents",
                                }
                             },
                            "required": ["question"],
                },
            },
        }
    ]
    #stabilește instrucțiunile generale pentru LLM.
    messages = [
        {
            "role": "system",
                    "content":  SYSTEM_PROMPT
        }
    ]
    print("Scrie 'quit' pentru a iesi.")
    while True:
        raw_question = input("you> ")
        if raw_question.strip().lower() == "quit":
            break

        # validam intrebarea utilizatorului cu pydantic
        try:
            request = SearchRequest(question=raw_question)
        except ValidationError as error:
            print("agent> Întrebare invalidă:", error.errors()[0]["msg"])
            # continua cu o noua intrebare in bucla pana cand utilizatorul iese cu "quit"
            continue

        #Adăugăm întrebarea validată în istoricul conversației
        messages.append({"role": "user", "content": request.question})


        # Apelăm modelul  cu istoricul conversației si cu descrierea toolurilor disponibile
        try:
            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))
            # exclude_none=True elimină câmpurile care au valoarea None.

            # folosirea Tool call ( in caz ca modelul cere mai multe tooluri)
            while response.tool_calls:
                # Parcurgem fiecare apel de tool solicitat de model.
                for tool_call in response.tool_calls:
                    # Verificăm dacă modelul a cerut tool-ul search_file.
                    if tool_call.function.name == "search_file":
                        try:
                            #Executăm căutarea hibridă în documentele indexate
                            #Argumentele sunt furnizate de LLM sub forma de json
                            output = rag_search(
                                collection, model, bm25,
                                documents, metadatas, ids,
                                tool_call.function.arguments,
                            )
                        except (json.JSONDecodeError, ValidationError) as error:
                            output = f"error: {error}"
                    else:
                        output = f"error: unknown tool {tool_call.function.name}"

                    #  trimitem rezultatul tool-ului inapoi la LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    })

                # După ce a primit rezultatul căutării, apelăm din nou LLM-ul.
                # modelul poate formula răspunsul folosind contextul găsit.
                response = complete(client, messages, tools)
                #Adăugăm și acest răspuns în istoricul conversației.
                messages.append(response.model_dump(exclude_none=True))
            #afisarea raspunsului final
            print("agent>", response.content or "")

        except Exception as error:
            print("agent> Eroare la apelul LLM:", error)


if __name__ == "__main__":
    main()