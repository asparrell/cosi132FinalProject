from serpapi import GoogleSearch
import json


class GoogleScholarSearch:
    def __init__(self, key: str):
        """
        :param key: SerpAPI key
        """
        self.key = key
        self.params = {
            'api_key': self.key,
            'engine': 'google_scholar',
            'hl': 'en',
            'gl': 'us',
            "start": "0",
            "num": "10",
            "output": "json"
        }

    def query(self, query: str) -> None:
        """
        outputs top 10 google scholar results for a query as a json. results can be found in "organic_results"
        :param query: query for google scholar
        """
        self.params['q'] = query
        search = GoogleSearch(self.params)
        with open('scholar_queries.json', 'w', encoding='utf-8') as f:
            json.dump(search.get_json(), f, ensure_ascii=False, indent=4)