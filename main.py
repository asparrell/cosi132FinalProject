from flask import Flask, render_template, request, jsonify
import openai
from openai.error import RateLimitError
import json

with open("api.txt") as f:
    key = f.readline()

openai.api_key = key

# set up the cache
config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
app.config.from_mapping(config)


messages = [{"role": "system", "content": "You are an information retrieval ai assistant"}]

# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["GET", "POST"])
def results():
    user_input = request.args.get('user_input') if request.method == 'GET' else \
    request.form['user_input']
    messages = [{"role": "user", "content": user_input}]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=messages
        )
        content = response.choices[0].message["content"]
    except RateLimitError:
        content = "The server is experiencing a high volume of requests. Please try again later."

    return jsonify(content=content)


@app.route("/results/search", methods=["POST"])
def search_results(chat_output):
    return render_template("search_results.html", chat_output=chat_output)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

