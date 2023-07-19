import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from src.customGpt import ask_and_get_answer, ask_with_memory, insert_or_fetch_embeddings
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the vector store on app load
@app.before_first_request
def load_vector_store():
    # Initialize Pinecone
    app.vector_store = insert_or_fetch_embeddings(os.environ.get('PINECONE_INDEX'))

@app.route('/')
def home():
    return 'Hello World'


@app.route('/askwithmemory', methods=['POST'])
def ask_question_with_memory():
    data = request.get_json()

    # Convert JSON chat history to the required format
    chat_history = [(str(x[0]), str(x[1])) for x in data.get('chat_history', [])]
    question = data.get('question', '')

    result, updated_chat_history = ask_with_memory(app.vector_store, question, chat_history)

    # Convert updated chat history back to JSON format
    updated_chat_history_json = [[str(x[0]), str(x[1])] for x in updated_chat_history]

    response = {
        'answer': result['answer'],
        'chat_history': updated_chat_history_json
    }

    return jsonify(response)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()

    question = data.get('question', '')

    result = ask_and_get_answer(app.vector_store, question)

    response = {
        'answer': result,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)