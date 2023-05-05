from flask import Flask, render_template, request, jsonify
import openai
from openai.error import RateLimitError
from utils import search_sources
import json

with open("api.txt") as f:
    key = f.readline().strip()

openai.api_key = key


app = Flask(__name__)

# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["GET", "POST"])
def results():
    user_input = request.args.get('user_input') if request.method == 'GET' else \
    request.form['user_input']
    print(user_input)
    messages = [{"role": "user", "content": user_input}]
    # Second query to GPT in order to fetch sources for the original query
    message2 = [{"role": "user", "content": "Can you give me some sources and DOIs for " + user_input}]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=messages
        )
        content = response.choices[0].message["content"]
    except RateLimitError:
        content = "The server is experiencing a high volume of requests. Please try again later."
    # Second query to GPT in order to fetch sources for the original query
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=message2
        )
        sources = response.choices[0].message["content"]
    except RateLimitError:
        sources = "The server is experiencing a high volume of requests. Please try again later."

    print(content)
    print(sources)

    # uncomment the below if the google API is working:
    google_results = search_sources(content, sources, user_input)

    print(google_results)
    # sample output for dummy testing search_sources rendering
    sample = [
                      {"url": "https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571", "title": "Language Translation with RNNs. Build a recurrent neural ...", "description": "In this project, I build a deep neural network that functions as part of a machine translation pipeline. The pipeline accepts English text as input and returns ...", "score": "0.5912", "source_type": "Original Google query result"},
                      {"url": "https://www.ibm.com/topics/natural-language-processing", "title": "What is Natural Language Processing?", "description": "Natural language processing uses machine learning to analyze text or speech data ... NLP drives computer programs that translate text from one language to ...", "score": "0.5803",  "source_type": "Original Google query result"},
                      {"url": "https://medium.com/analytics-vidhya/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66", "title": "Natural Language Processing: From Basics to using RNN ...", "description": "This very arm of machine learning is called as Natural Language Processing. This post is an attempt at explaining the basics of Natural Language Processing ...", "score": "0.5447", "source_type": "ChatGPT generated source"},
                      {"url": "https://link.springer.com/article/10.1007/s11042-022-13428-4", "title": "Natural language processing: state of the art, current trends ...", "description": "by D Khurana · 2022 · Cited by 349 — NLU enables machines to understand natural language and analyze it by extracting concepts, entities, emotion, keywords etc. It is used in ...", "score": "0.5370", "source_type": "ChatGPT generated source"},
                      {"url": "https://www.qblocks.cloud/blog/natural-language-processing-machine-translation", "title": "Natural language processing (NLP) and its use in machine ...", "description": "Natural Language Processing combines computational linguistics, rule-based modeling of human language with some statistics, machine learning, and deep learning ...", "score": "0.5188", "source_type": "ChatGPT generated source"}
            ]
    s = str(json.dumps(sample))
    # print(s)
    # if google api is working, replace sources with google_results (uncomment above)
    return jsonify(content=content, sources=google_results)


# @app.route("/results/search", methods=["POST"])
# def search_results():
#     chat_output = cache.get("chat_output")
#     google_results = search_sources(chat_output)
#     return render_template("search_results.html", chat_output=chat_output, google_results=google_results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

