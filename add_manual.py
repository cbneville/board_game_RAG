#C. Neville, 04/18/2025, for python 3.13
"""Script to add a game manual to the vector store and inventory.csv
First upload pdf of manual to game_manuals directory. Then type:
python add_manual.py game-name manual-filename"""

import sys
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

google_embedding = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GEMINI_API_KEY"), model="models/embedding-001")
vector_store = InMemoryVectorStore(embedding=google_embedding).load("board_game_vector_store.json", google_embedding)

pdf_name = sys.argv[2]
game_name = sys.argv[1]
try:
    #Load text from pdf
    loader = PyPDFLoader("./game_manuals/"+pdf_name) #mode='single' to not separate by pages
    pages = []
    docs_lazy = loader.lazy_load()
    for doc in docs_lazy:
        doc.metadata['game'] = game_name
        pages.append(doc)

    #Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "."],
                                                chunk_size=600,
                                                chunk_overlap=100,
                                                add_start_index=True)
    split_text = text_splitter.split_documents(pages)

    #Encode chunks and add to vector store
    doc_ids = vector_store.add_documents(split_text)

except:
    print("Issue adding info from %s. File may be encrypted or incompatible with PyPDFLoader."%pdf_name)
    sys.exit()

#Save vector store to json file
vector_store.dump("board_game_vector_store.json")

#Update inventory.csv
inventory = open("game_manuals/inventory.csv",'a')
inventory.write("\n"+game_name+","+pdf_name)
inventory.close()
