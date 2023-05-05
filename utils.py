import re
from googlesearch import search
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import tee
import json
from time import sleep

MODEL = SentenceTransformer('paraphrase-MiniLM-L12-v2')


# given output from ChatGPT in the form of a string, returns 3 lists of strings: sources, descriptions, and DOIs
# descriptions is empty if there are no descriptions
# dois contains empty strings for sources without a DOI
def process_gpt_output(output):
    sources = []
    descriptions = []
    dois = []
    # a boolean to keep track of whether the previous line was added to sources
    prev_line_source = False
    lines = re.split(r'\n', output)
    for line in lines:
        # the line is a source if it is a list item
        if re.match(r'\d+\.', line):
            source = re.sub(r'^\d+\. ', '', line)
            sources.append(source)
            dois.append(get_doi(source))
            prev_line_source = True
        # the line is a description if it is immediately after a source
        elif prev_line_source:
            descriptions.append(line)
            prev_line_source = False
    return sources, descriptions, dois


def get_doi(source):
    match = re.search(r'doi: [\w/\.]+', source)
    if match:
        string = match.group(0)
        return re.sub('doi: ', '', string)
    else:
        return ''


# Input: output from ChatGPT in the form of a string
# Output: the results from Googling the sources as a dictionary of generators
# The keys are the sources, and the values are the top three results for each source
# For each result, you can access .url, .title, and .description
def search_sources(original_gpt_output, sources_gpt_output, original_query):
    """
        json format:
    [
        {
        "url" = <url>
        "description" = <description>
        "title" = <title>
        "score" = <score>
        "source_type": <original query or gpt source>
        }
    ]
    :param sources_gpt_output: gpt output of sources request
    :param original_gpt_output: original gpt output
    :param original_query: user input query
    :return: json of google results
    """
    # processes gpt output
    sources, descriptions, dois = process_gpt_output(sources_gpt_output)

    # iterate through sources and get google results
    out = []
    for source in sources:
        # output of search() is a generator of dictionary-like objects
        google_results = search(source, advanced=True, num_results=3, sleep_interval=10)
        # gets json of google results
        out = results_to_json(google_results, original_gpt_output, "gpt source", out)

    # combine with google results of original query
    query_results = search(original_query, advanced=True, num_results=3, sleep_interval=10)
    out = results_to_json(query_results, original_gpt_output, "original query", out)

    # sort by score
    out = sorted(out, key=lambda d: d['score'], reverse=True)

    # list to json
    out = str(json.dumps(out))
    return out


def results_to_json(google_results, gpt_output, type, master_list):
    """
    returns list with new results added
    :param google_results: google results of query
    :param gpt_output: original gpt output
    :param type: type of source
    :param master_list: list to append to
    :return:
    """
    for result in google_results:
        sleep(5)
        hit = {"url": result.url, "title": result.title, "description": result.description}
        score = similarities(gpt_output, result.description)["bert_cosine"]
        hit["score"] = str(score)
        hit["source_type"] = type
        master_list.append(hit)

    return master_list


# def search_to_score(source, results):
#     total_similarity = 0
#     for i, result in enumerate(results):
#         similarity = similarities(source, result.description)
#         bert_cos = similarity["bert_cosine"]
#         total_similarity += bert_cos
#     total_similarity /= i
#
#     return total_similarity


def similarities(google_out, openai_out):
    """
    :param google_out: google output
    :param openai_out: openai output (order doesn't really matter)
    :return: a dict formatted:
    {tfidf cosine similarity: float, tfidf euclidean distance: float,
     sBERT cosine similarity: float, sBERT euclidean distance: float}
    """
    similarities = {}
    documents = [google_out, openai_out]

    # tf-idf scores
    tfidfvectoriser = TfidfVectorizer()
    tfidfvectoriser.fit(documents)
    tfidf_vectors = tfidfvectoriser.transform(documents)
    similarities["tfidf_cosine"] = cosine_similarity(tfidf_vectors)[0][1]
    similarities["tfidf_euclidean"] = euclidean_distances(tfidf_vectors)[0][1]

    # bert scores
    embeddings = MODEL.encode(documents, convert_to_tensor=True)
    similarities["bert_cosine"] = cosine_similarity(embeddings)[0][1]
    similarities["bert_euclidean"] = euclidean_distances(embeddings)[0][1]
    return similarities

# for debugging purposes
# if __name__ == '__main__':
#     gpt_output = "Sure, here's a list of sources that can help you learn how to use LaTeX:\n1. The LaTeX Project - " \
#                  "https://www.latex-project.org/\nThe LaTeX Project is the official website of LaTeX. It provides a " \
#                  "comprehensive user guide, tutorials, and documentation that cover various topics related to " \
#                  "LaTeX.\n2. Overleaf - https://www.overleaf.com/learn\nOverleaf is a cloud-based LaTeX editor that " \
#                  "provides various templates and tutorials for beginners. It also offers collaboration features and a " \
#                  "rich text editor to help you get started with LaTeX quickly. "
#     original_query = "how do I use LaTeX?"
#
#     print(search_sources(gpt_output, gpt_output, original_query))
