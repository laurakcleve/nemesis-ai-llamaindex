import os
from flask import ( Flask, request, jsonify )
from llama_index.core import (
    StorageContext,
    load_index_from_storage
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
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return jsonify(response.response), 200


def load_index():
    global index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)


if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5601)