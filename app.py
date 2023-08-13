import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from src.customGpt import ask_with_memory_and_prompt, insert_or_fetch_embeddings
from flask import Flask, request
import pinecone

app = Flask(__name__)

# Load the vector store on app load
@app.before_request
def load_vector_store():
    # Initialize Pinecone
    app.vector_store = insert_or_fetch_embeddings(os.environ.get('PINECONE_INDEX'))
    # Load the vector store
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    index = pinecone.Index('test-index')
    app.index = index

@app.route('/')
def home():
    return 'Hello World'

def document_to_dict(document):
    return {
        'page_content': document.page_content,
        'metadata': document.metadata
    }

@app.route('/ask', methods=['POST'])
def ask_question_with_memory():
    data = request.get_json()

    # Convert JSON chat history to the required format (from json dict to tuple)
    chat_history = [(str(x[0]), str(x[1])) for x in data.get('chat_history', [])]
    question = data.get('question', '')

    ai_response  = ask_with_memory_and_prompt(app.vector_store, question, chat_history)
    answer = ai_response['answer']
    chat_history.append((question, answer))

    response = {
            'question': ai_response['question'],
            'answer': ai_response['answer'],
            'chat_history': chat_history,
            'source_documents': [document_to_dict(doc) for doc in ai_response['source_documents']]
        }

    return response

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)