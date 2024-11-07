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
    return get_chat_response(msg)


@app.route("/simplify", methods=["POST"])
def simplify_response():
    original_response = request.form["response"]
    simplified_response = generator.get_explanation(original_response)
    return jsonify(simplified_response)

def get_chat_response(msg:str) -> str:
    context, resources = retriever.get_context(msg)
    prompt = fewshot_prompt_template.format(
        question = msg,
        context = context,
        few_shot_examples = fewshot_examples
    )
    formatted_resources = "\n".join(resources.splitlines())
    return generator.get_response(prompt) + f"\n\المصادر:\n{formatted_resources}"





if __name__ == '__main__':
    app.run(debug=True)
