#Import libraries
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from pydantic import BaseModel, Field

#Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

#Load vector store from file
google_embedding = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")
vector_store = InMemoryVectorStore(embedding=google_embedding).load("board_game_vector_store.json", google_embedding)

#Function to include pdf page labels with retrieved document contents
def format_docs_with_page(docs: List[Document]) -> str:
    formatted = [
        f"Page Number: {doc.metadata['page_label']}\nManual Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

#Define quoted answer and citation classes

class Citation(BaseModel):
    page_num: int = Field(
        ...,
        description="The page number of the SPECIFIC manual snippet which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified manual snippet that justifies the answer.",
    )

class QuotedAnswer(BaseModel):
    """Answer the player's question based on the retrieved context from the game manual as well as your knowledge, and cite the relevant retrieved manual snippet."""
    answer: str = Field(
        ...,
        description="The answer to the user question, based on the retrieved context and your knowledge.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given context that justify the answer."
    )

#Define application state
class State(TypedDict):
    game: str
    question: str
    context: List[Document]
    answer: QuotedAnswer

#Retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], filter=lambda doc: doc.metadata.get("game")==state["game"])
    return {"context": retrieved_docs}

#Generation step
def generate(state: State):
    formatted_docs = format_docs_with_page(state["context"])
    #Prompt engineering
    prompt_text = "You are an assistant for looking up board game rules. Use the following pieces of retrieved context from the board game manual, in addition to your own knowledge, to answer the question.\n" \
    "Game: {game}\n"\
    "Question: {question}\n" \
    "Context: {context}"
    prompt = PromptTemplate.from_template(prompt_text)
    messages = prompt.invoke({"game": state["game"], "question": state["question"], "context": formatted_docs})
    structured_llm = llm.with_structured_output(QuotedAnswer)
    response = structured_llm.invoke(messages)
    return({"answer":response})

#Build state graph
def rag_v2(query, game_name):
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question":query, "game":game_name})
    return(response)