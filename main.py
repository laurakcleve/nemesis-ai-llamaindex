import os
import pickle
from flask import ( Flask, request, jsonify )
from flask_cors import CORS, cross_origin
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    ChatPromptTemplate,
    Settings
)
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from traceloop.sdk import Traceloop
from dotenv import load_dotenv

from prompts import (
    system_message, 
    user_message
)


load_dotenv()
TRACELOOP_API_KEY = os.environ['TRACELOOP_API_KEY']
Traceloop.init(disable_batch=True)

app = Flask(__name__)


@app.route("/query", methods=["POST"])
@cross_origin(origin='http://localhost:5173', methods=["POST"], headers=['Content-Type'])
def query_index():
    Settings.llm = OpenAI(model="gpt-4")

    index, retriever = load_index()

    query_engine = RetrieverQueryEngine.from_args(retriever)

    query_text = request.json

    messages = [
        ( 'system', system_message),
        ( 'user', user_message)
    ]
    qa_template = ChatPromptTemplate.from_messages(messages)
    query_engine = index.as_query_engine(text_qa_template=qa_template)

    response = query_engine.query(query_text)

    return jsonify(response.response), 200


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="./data/index")
    index = load_index_from_storage(storage_context)

    with open('./data/all_nodes_dict.pkl', 'rb') as f:
        all_nodes_dict = pickle.load(f)

    vector_retriever = index.as_retriever(similarity_top_k=4)

    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_nodes_dict
    )

    return index, retriever


if __name__ == "__main__":
    app.run()