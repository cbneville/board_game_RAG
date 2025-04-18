## C. Neville, 04/16/2025, for python 3.13 ##
""" Script to ping an LLM with RAG to answer questions about board game rules.
Usage: python bg_assist.py v1/v2 Board-Game-Name
E.g.: python bg_assist.py v1 Everdell
If name is not in game_manuals/inventory.csv, will revert to a basic chat model.
v1 tends to provide more detailed answers but prints more manual context than necessary
v2 tends to quote more directly from the manual and only prints the relevant manual sections
To end, type Done or hit Ctrl+C."""

## Import libraries ##
import sys
from RAG_v1 import rag_v1
from RAG_v2 import rag_v2
from general_llm import general_llm
import pandas as pd

## Define print functions for different response types ##

def v1_print(response):
    print()
    print(response["answer"])
    print("\n\nReferenced manual sections:\n")
    for doc in response["context"]:
        print("Page %s:\n%s\n\n"%(doc.metadata['page_label'], doc.page_content))
    print()

def v2_print(response):
    print()
    print(response['answer'].answer)
    print("\nCitations:")
    for cite in response['answer'].citations:
        print(cite)
    print()

def general_print(response):
    print()
    print(response.content)
    print()

## Run interactively ##
if __name__ == "__main__":
    gamelist = pd.read_csv("game_manuals/inventory.csv")['game_name'].to_list()
    game_name = sys.argv[2]
    if game_name in gamelist:
        while True:
            query = input("What is your question?\n")
            if query == "Done":
                sys.exit()
            else:
                if sys.argv[1] == "v1":
                    response = rag_v1(query, game_name)
                    v1_print(response)
                elif sys.argv[1] == "v2":
                    response = rag_v2(query, game_name)
                    v2_print(response)
                else:
                    print("First argument must be either 'v1' or 'v2'")
            print("==========================================")
    else:
        print("That game name is not in inventory.csv. The chatbot will ping Gemini without using RAG.")
        while True:
            query = input("What is your question?\n")
            if query == "Done":
                sys.exit()
            else:
                response = general_llm(query, game_name)
                general_print(response)
            print("==========================================")