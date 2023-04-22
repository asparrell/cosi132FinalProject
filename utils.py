import re
from googlesearch import search


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


# given output from ChatGPT in the form of a string, prints the results from Googling the sources
# the print format is not pretty yet, but we can change it to fit our needs
# I assume we want to return the results eventually rather than print them
def search_sources(gpt_output):
    sources, descriptions, dois = process_gpt_output(gpt_output)
    for source in sources:
        results = search(source, advanced=True)  # output is a generator of dictionary-like objects
        for result in results:
            print(result)
