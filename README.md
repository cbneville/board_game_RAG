### RAG for Board Game Rule Lookup
This is a sample project for using retrieval-augmented generation (RAG) with Google's Gemini LLM to answer questions about board game rules. The goal is to provide fast, accurate, and trustworthy answers to questions during game play, so no one has to stop to dig through a manual. 

Included files:
* RAG.ipynb: Jupyter notebook demonstrating the utility of RAG for board game rule lookup and comparing the performance of the two different RAG implementations used here.
* bg_assist.py: The main script to use to access the application. 
    * Usage: python bg_assist.py [v1/v2] [Board-Game-Name]
    * Example usage: python bg_assist.py v2 "Ticket to Ride Europe"
    * Using RAG implementation v2 results in more succinct answers and more specific manual citations. Using RAG implementation v1 results in longer, more creative answers and prints all retrieved manual snippets, not only the most relevant ones.
    * Game names containing spaces need to be enclosed in quotes. If a game name is not in game_manuals/inventory.csv, the script will run a simple call to Gemini without RAG.
    * Accessing Gemini requires that the environment variable GOOGLE_API_KEY be set. If using venv, it may be easiest to add a line exporting the variable to venv/bin/activate. Instructions on obtaining an API key can be found here: https://ai.google.dev/gemini-api/docs/api-key
* RAG_v1.py, RAG_v2.py, and general_llm.py: The RAG/LLM implementation scripts called by bg_assist.py
* add_manual.py: Script to add a game manual to the vector store for RAG (also adds game to game_manuals/inventory.csv). First ensure the game manual pdf is saved in the game_manuals directory. Usage: python add_manual.py game_name manual_file_name.pdf
* board_game_vector_store.json: Vector store containing manual snippets and their embeddings for similarity search. Created by build_vector_store.py based off the list of games and manuals in game_manuals/inventory.csv
* game_manuals: directory for pdf files of game manuals and the inventory.csv file used by build_vector_store.py, add_manual.py, and bg_assist.py
* requirements.txt: list of python libraries required to run the scripts in this repository

NB: The game_manuals directory here only contains inventory.csv, because I don't want to research the legality of disseminating pdf manuals I didn't create. You can download the manuals noted in inventory.csv from these links, and put them in the game_manuals directory:
* Pandemic: https://cdn.svc.asmodee.net/production-zman/uploads/2024/09/Pandemic_Rulebook.pdf
* Puerto Rico: https://www.riograndegames.com/wp-content/uploads/2019/08/Puerto-Rico-Deluxe-Rules.pdf
* Ticket to Ride Europe: https://ncdn0.daysofwonder.com/tickettoride/de/img/te_rules_2015_en.pdf
* Mysterium: https://cdn.1j1ju.com/medias/ae/89/37-mysterium-rulebook.pdf
* Everdell: https://cdn.1j1ju.com/medias/c6/cd/89-everdell-rulebook.pdf

#### Initial Setup Guide
* Create a python environment for the project
    * python3 -m venv bg_env
* Add google API key to bg_env/bin/activate file
    * export GOOGLE_API_KEY="your-key-here"
* Activate environment
    * source bg_env/bin/activate
* Download all pdf manual files of interest and place in the game_manuals directory
* To build a new vector store from scratch:
    * Update game_manuals/inventory.csv with the game names and manual filenames you want included in the vector store
    * run: python build_vector_store.py
    * This will overwrite the existing board_game_vector_store.json file. 
* Alternately, to update the existing vector store:
    * Ensure the manual you want to add is in the game_manuals directory
    * run: python add_manual.py [game-name] [manual-filename.pdf]
* Run assistant script:
    * python bg_assist.py [v1/v2] [game-name]
    * The script will prompt you to input questions. When you are finished, enter 'Done' to end the program.
