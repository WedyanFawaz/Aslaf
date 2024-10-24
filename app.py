from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

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

def get_chat_response(text):
    # Chat for 1 response
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
