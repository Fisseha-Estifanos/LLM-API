from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/index")
def index_page():
    return "<p>Hello, World! this is the index page.</p>"
