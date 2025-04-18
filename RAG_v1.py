import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

#Initialize llm
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

#Load vector store from file
google_embedding = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")
vector_store = InMemoryVectorStore(embedding=google_embedding).load("board_game_vector_store.json", google_embedding)

#Define application state
class State(TypedDict):
    game: str
    question: str
    context: List[Document]
    answer: str

#Retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], filter=lambda doc: doc.metadata.get("game")==state["game"])
    return {"context": retrieved_docs}

#Generation step
def generate(state: State):
    prompt_text = "You are an assistant for looking up board game rules. Use the following pieces of retrieved context from the board game manual, in addition to your own knowledge, to answer the question.\n" \
    "Game: {game}\n"\
    "Question: {question}\n" \
    "Context: {context}"
    prompt = PromptTemplate.from_template(prompt_text)
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"game": state["game"], "question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return({"answer":response.content})

#Build state graph
def rag_v1(query, game_name):
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question":query, "game":game_name})
    return(response)