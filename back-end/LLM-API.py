from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World! and welcome to this large language model API'


@app.route("/news-score")
def hello_world():
    return "<p>Hello, World! and welcome to the news scoring endpoint</p>"


@app.route("/j-d-entities")
def index_page():
    return "<p>Hello, World! and welcome to the job description entity and relation extraction endpoint</p>"
