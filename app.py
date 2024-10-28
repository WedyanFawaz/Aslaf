from flask import Flask, render_template, request, jsonify


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

if __name__ == '__main__':
    app.run(debug=True)
