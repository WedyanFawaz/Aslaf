from flask import Flask, render_template, request, jsonify
from generator import Generator
from retriever import Retriever
from langchain.schema import Document
from prompt_template import (fewshot_prompt_template, fewshot_examples)

app = Flask(__name__)
generator = Generator()
retriever = Retriever()

@app.route("/")  # Route for the main page
def index():
    return render_template('main_page.html')  # Render the main page

@app.route("/chat_page")  # Route for the chat page
def chat_page():
    return render_template('chat_page.html')  # Render the chat page

@app.route("/get", methods=["GET", "POST"])  # Route to handle chat responses
def chat():
    msg = request.form["msg"]  # Get user input from the form
    response = get_chat_response(msg)
    return jsonify({'text': response})  # Return as JSON


@app.route("/simplify", methods=["POST"])
def simplify_response():
    original_response = request.form["response"]
    simplified_response = generator.get_explanation(original_response)
    return jsonify({'text': simplified_response})  # Return as JSON

def get_chat_response(msg:str) -> str:
    context, resources = retriever.get_context(msg)
    if resources:
        formatted_resources = "\n".join(resources.splitlines())
        return generator.get_response(msg, context) + f"\nالمصادر:\n{formatted_resources}"

    return generator.get_response(msg,context)

@app.route("/resources", methods=["POST"])
def get_from_user():
    msg = request.form["msg"]
    response = generator.get_recs(msg)  # Generate the response
    return jsonify({'text': response})  # Return as JSON

    





if __name__ == '__main__':
    app.run(debug=True)
