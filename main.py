from flask import ( Flask, request, jsonify )
from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from flask import Flask
from flask_cors import CORS, cross_origin


app = Flask(__name__)

@app.route("/")

@app.route("/query", methods=["GET"])
@cross_origin(origin='http://localhost:5173', methods=["GET"], headers=['Content-Type'])


def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return jsonify(
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return jsonify(str(response)), 200


def home():
    return "Hello World!"

def load_index():
    global index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)


if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5601)