import os
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def insert_or_fetch_embeddings(index_name, chunks=None):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    # Load vector store
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ...')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
        print(f'Index {index_name} does not exist. Creating index and embeddings')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)

    return vector_store

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwars={'k': 3})

    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    print(chat_history)
    result = chain({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwars={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer=chain.run(q)
    return answer
