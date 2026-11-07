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
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import re

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
Do not use your own general knowledge. Always mention the source
(file name and page) from the context."""

#Validare pydantic input si output pentru tool
#Input - intrebarea utilizatorului
class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    question: str = Field(
        min_length=1,
        max_length=3000,
        description="Question about the contents of the indexed documents",
    )


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



# Tokenizarea

def tokenize(text: str) -> list[str]:
    #regex
    return re.findall(r"\b\w+\b", text.lower())


# Impartim textul in chunkuri
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

#Indexam un fisier

def index_file(collection, model, file_path: Path) -> int:

    records = []

    # Obtinem chunk-urile pentru fiecare pagina
    for page_number, page_text in read_file_pages(file_path):
        page_chunks = chunk_text(page_text)

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
    ids=[]
    documents = []
    metadatas = []

    for chunk, page_number, chunk_number in records:

        documents.append(chunk)

        metadatas.append({
            "source": file_path.name,
            "page": page_number,
            "chunk": chunk_number,

        })
        ids.append (
            f"{file_path.name}:p{page_number}:c{chunk_number}")


#Adaugam datele in ChromaDB
    collection.upsert(

        ids=ids,
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(), #vectorizarea
        documents=documents,
        metadatas=metadatas,
    )
    print(f" {file_path.name}: {len(documents)} chunk-uri indexate.")
    return len(documents)

#Indexam toate fisierele din folder
def index_data_sources(collection, model) -> None:

    if not DATA_SOURCES_DIR.exists():

        DATA_SOURCES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Folderul a fost creat"
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

#Adaugam documentele pentru Bm25
def load_indexed_documents(collection):

    result = collection.get(
        include=["documents", "metadatas"]
    )

    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    ids = result["ids"]

    return documents, metadatas, ids


#  Tool: RAG search (hybrid: semantic + BM25)
def rag_search(collection, model, bm25, documents, metadatas, ids, args:str,) -> str:
    """
        Primeste argumentele tool-ului (JSON string), valideaza input-ul,
        face hybrid search si returneaza un string simplu validat cu pydantic.
        """
    request = SearchRequest.model_validate(json.loads(args))
    question= request.question

    if not documents:
        return SearchResponse(context="Nu exista document indexat").context

    # top_k = min(RESULT_COUNT, len(documents))
    # id_to_index = {doc_id: index for index, doc_id in enumerate(ids)}
    # hybrid_scores: dict[int, float] = {}


    #Cautarea semantica Chroma DB
    question_vector = model.encode(question, normalize_embeddings=True).tolist()

    # folosim 2 rezultate si le combinam pentru un scor hibrid

    semantic_results = collection.query(
        query_embeddings=[question_vector],
        n_results=min (HYBRID_SEARCH_RESULT_COUNT,len(documents)),
        include=["documents", "metadatas"],
    )


    # Mapam ID-urile la indexurile documentelor
    id_to_index = {
        doc_id: index
        for index, doc_id in enumerate(ids)
    }

    hybrid_scores = {}



    # Rank 0 primeste 1 punct, rank 1 primeste 1/2 puncte etc
    semantic_ids = semantic_results["ids"][0]
    for rank, doc_id in enumerate(semantic_ids):
        chunk_index = id_to_index[doc_id]
        hybrid_scores[chunk_index] = 1 / (rank + 1)

    # Ranking BM25
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_results = []
    chunk_index = 0

    # scoatem scorul cu ajutorul bm25 in ordine inversa
    for score in bm25_scores:
        bm25_results.append((score, chunk_index))
        chunk_index += 1
    bm25_results.sort(reverse=True)


    for rank, (score, chunk_index) in enumerate(bm25_results[:HYBRID_SEARCH_RESULT_COUNT]):
        hybrid_scores[chunk_index] = hybrid_scores.get(chunk_index, 0) + 1 / (rank + 1)

    combined_results = []
    for chunk_index, score in hybrid_scores.items():
        combined_results.append((score, chunk_index))
    combined_results.sort(reverse=True)

#Construim contextul
    results = []
    for _, chunk_index in combined_results[:HYBRID_SEARCH_RESULT_COUNT]:
        results.append(
            f"{documents[chunk_index]}\nSource: {metadatas[chunk_index]['source']}, page {metadatas[chunk_index]['page']}"
        )
    context = "\n\n".join(results) if results else "NU s-au gasit informatii relevante"

    # validare output cu pydantic
    return SearchResponse(context=context).context


#LLM
def complete(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict],
):
    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    return response.choices[0].message




# def answer_with_context(client: OpenAI, collection, model, args: str) -> str:
#     try:
#         request = SearchRequest.model_validate(json.loads(args))
#         vector = model.encode(request.question, normalize_embeddings=True).tolist()
#         result = collection.query(
#             query_embeddings=[vector],
#             n_results=8,
#             include=["documents", "metadatas"],
#         )
#         context_pieces = []
#         documents = result["documents"][0]
#         metadatas = result["metadatas"][0]
#
#         # construim un string structurat pentru  pentru LLM
#         for i in range(len(documents)):
#             document = documents[i]
#             metadata = metadatas[i]
#             context_pieces.append(
#                 f"{document}\nSource: {metadata['source']}, page {metadata['page']}"
#             )
#
#         context = "\n\n".join(context_pieces)
#         return client.chat.completions.create(
#             model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
#             messages=[
#                 {
#                     "role": "system",
#                     "content":  """You are a helpful assistant that answers questions using a knowledge base
#             of indexed documents. ALWAYS call search_file before answering any
#             factual or informational question, even if you think you know the answer.
#             Answer only from the retrieved context. If the context does not contain
#             the answer, say that the information was not found in the documents.
#             Do not use your own general knowledge. Always mention the source
#             (file name and page) from the context."""
#                 },
#                 {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context}"},
#             ],
#         ).choices[0].message.content
#
#     except (json.JSONDecodeError, ValidationError, KeyError) as error:
#         return f"error: {error}"
#

def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    #load embedding model
    model = SentenceTransformer(MODEL_NAME)

    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


    # Index PDF files
    index_data_sources(collection, model)

    # Load all indexed chunks
    documents, metadatas, ids = load_indexed_documents(collection)

    if not documents:
        print("Nu exista documente indexate.")
        return

    tokenized_documents = []

    for doc in documents:
        tokens = tokenize(doc)
        tokenized_documents.append(tokens)

    bm25 = BM25Okapi(tokenized_documents)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_file",
                "description": ("""Search the indexed PDF files. 
                    Use this tool for questions about 
                    PDF documents or their contents."""),
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
    messages = [
        {
            "role": "system",
                    "content":  SYSTEM_PROMPT
        }
    ]
    print("Scrie 'quit' a iesi.")
    while True:
        raw_question = input("you> ")
        if raw_question.strip().lower() == "quit":
            break

        # validam intrebarea utilizatorului cu pydantic
        try:
            request = SearchRequest(question=raw_question)
        except ValidationError as error:
            print("agent> Întrebare invalidă:", error.errors()[0]["msg"])
            continue

        messages.append({"role": "user", "content": request.question})

        try:
            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

            # folosirea Tool call ( in caz ca modelul cere tool-ul de mai multe ori)
            while response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "search_file":
                        try:
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

                # primirea raspunsului de la LLM dupa ce a vazut contextul
                response = complete(client, messages, tools)
                messages.append(response.model_dump(exclude_none=True))

            print("agent>", response.content or "")

        except Exception as error:
            print("agent> Eroare la apelul LLM:", error)


if __name__ == "__main__":
    main()