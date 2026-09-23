# Tema 2 - Agent simplu pentru cautare in proiect
#
# Creati un agent LLM simplu, dupa modelul dela curs, care primeste doua unelte:
#
# 1. O unealta care listeaza toate fisierele din repository. Puteti folosi "dir /s" pentru a lista toate fisierele din proiect.
# 2. O unealta care cauta un cuvant cheie in continutul tuturor fisierelor. puteti folosi de exemplu, pe windows: "findstr /s /i "your_keyword" *.*"
#
# Cuvantul cheie cautat va fi: `pytest`.
#
# Cerințe:
#
# 1. Agentul va apela mai intai unealta de listare a fisierelor.
# 2. Agentul va cauta apoi cuvantul `pytest` in intregul proiet.
# 3. Afisati fisierele in care a fost gasit cuvantul cheie.
# 4. Scrieti un test cu `pytest` pentru acest agent, puteti doar sa rulati simplu unealta, si sa verificati ca agentul a returnat text care contine numele unui fisier la care ne asteptam sa contina acel string pytest. puteti verifica daca textul returnat de agent contine un substring, folosind : "if 'substring' in given_string", unde given_string poate fi intregul text returnat de agent.
#
# Puneti rezultatul acestei teme intr-un fisier tema2.py

from dotenv import load_dotenv
import os #legatura cu systemul de operare (schimbe folderul curent al terminalului, citeste fisiere)
from openai import OpenAI
from openai.types.conversations import message
import subprocess
from pathlib import Path
import json
from pydantic import BaseModel, EmailStr, Field, ValidationError





def find_keyword()-> str:
    try:
        result = subprocess.run(
            args= ["cmd", "/c", 'findstr /s /i "pytest" *.*'],
                   #cwd - current work directory
            cwd =Path.cwd(),
            capture_output=True,
            text=True,
            encoding="cp850",
            errors="replace",
            timeout=60




        )

        return (result.stdout or "No output.") + "" + (result.stdout or "")
    except Exception as e:
        print(e)

def list_files()-> str:
    try:
        result = subprocess.run(
            args= ["cmd", "/c", "dir", "/s"],
                   #cwd - current work directory
                   cwd =Path.cwd(),
            capture_output=True,
            text=True,
            encoding="cp850",
            errors="replace",
            timeout=60



        )
        return (result.stdout or "No output.") + "" + (result.stdout or "")
    except Exception as e:
        print(e)

def complete(client, messages, tools):
    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools, tool_choice="auto" )
    return response.choices[0].message

def main ():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv(
            "OPEN_ROUTER_API_KEY"),)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all the files from repository"
            }

        },
        {
            "type": "function",
            "function": {
                "name": "find_keyword",
                "description": "Find the given keyword in all files"
            }

        }
    ]

    messages = []
    # system prompt
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant. If the user asks to list all the files from repository, please only use list_files to list all the files from project"

    })

    messages.append({
        "role": "system",
        "content": "You are a helpful assistant. If the user asks to find the keyword from all files from repository, please only use find_keyword to find the keyword from the files from project"

    })

    while True:
        inp = input("you> ")
        if inp == "quit":
            break
        messages.append({
            "role":"user",
            #"content": input("you> ")
            "content":inp
        }
        )

        response = complete(client, messages, tools)

        if response.tool_calls:
            # adaugam raspunsul agentului in conversatie
            messages.append(response)
            for tool_call in response.tool_calls:
                list_files_response = list_files()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": list_files_response

                })
                find_keyword_response = find_keyword()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": find_keyword_response

                })
            response = complete(client, messages, tools)
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            print("agent>", response.content)


if __name__ == "__main__":
    main()