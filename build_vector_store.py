## C. Neville, 04/16/2025, for python 3.13 ##
## Script to build and save a vector store made up of info from board game manuals ##

## Import libraries ##
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd

## List of manuals to import ##
path = "./game_manuals/" #path to directory containing game manuals
manual_list = pd.read_csv(path+"inventory.csv")
# manual_list = [("Everdell","everdell-rulebook.pdf"),
#                ("Mysterium","mysterium-rulebook.pdf"),
#                ("Puerto Rico", "Puerto-Rico-Deluxe-Rules.pdf")]

## Create vector store ##
google_embedding = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GEMINI_API_KEY"), model="models/embedding-001")
vector_store = InMemoryVectorStore(embedding=google_embedding)

## Load manuals into vector store ##
for i in range(len(manual_list)):
    print("Processing %s"%manual_list['game_name'].iloc[i])
    pdf_name = path+manual_list['file_name'].iloc[i]
    game_name = manual_list['game_name'].iloc[i]
    try:
        #Load text from pdf
        loader = PyPDFLoader(pdf_name) #mode='single' to not separate by pages
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
        print("Issue adding info from %s"%pdf_name)
        continue

## Save vector store to json file ##
vector_store.dump("board_game_vector_store.json")
print("Done!")