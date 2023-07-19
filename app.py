from flask import Flask, jsonify, make_response

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello World'
