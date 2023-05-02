from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import openai
from openai.error import RateLimitError
import json
from utils import search_sources

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
cache = Cache(app)

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

    cache.set("chat_output", content)

    return jsonify(content=content)


@app.route("/results/search", methods=["POST"])
def search_results():
    chat_output = cache.get("chat_output")
    google_results, scores = search_sources(chat_output)
    return render_template("search_results.html", chat_output=chat_output,
                           google_results=zip(google_results.items(), scores))


if __name__ == "__main__":
    # app.jinja_env.globals.update(zip=zip)
    app.run(debug=True, port=5000)

