import os
from flask import ( Flask, request, jsonify )
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    ChatPromptTemplate
)
from flask import Flask
from flask_cors import CORS, cross_origin
from traceloop.sdk import Traceloop
from dotenv import load_dotenv

load_dotenv()

TRACELOOP_API_KEY = os.environ['TRACELOOP_API_KEY']

Traceloop.init(disable_batch=True)

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World!"


@app.route("/query", methods=["POST"])
@cross_origin(origin='http://localhost:5173', methods=["POST"], headers=['Content-Type'])
def query_index():
    global index
    query_text = request.json

    messages = [
        ( 'system', 'Answer the user\'s question given the provided context. First look at the heading of the relevant section from the context and assess whether it applies to the situation of the question, then reason through the logic of the rules before giving an answer. Your answer should be as accurate as possible, and should not include the details of the headings and sections, nor your steps of reasoning. If the answer cannot be found in the context, respond that you could not find the answer, without mentioning the context.' ),
        ( 'user', 'Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information, answer the question: {query_str}\n' )
    ]
    qa_template = ChatPromptTemplate.from_messages(messages)
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    response = query_engine.query(query_text)
    return jsonify(response.response), 200


def load_index():
    global index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)


if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5601)