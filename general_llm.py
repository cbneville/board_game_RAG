#C. Neville, 04/18/2025, for python 3.13
"""Function to run a vanilla LLM implementation for cases when
the board game in question is not in inventory.csv"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def general_llm(query, game_name):
    prompt_text = "You are an assistant for looking up board game rules. Answer the following question about the following game. Be succinct.\n"\
    "Game: {game}\n"\
    "Question: {question}"
    prompt = PromptTemplate.from_template(prompt_text)
    messages = prompt.invoke({"game":game_name, "question":query})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    response = llm.invoke(messages)
    return(response)