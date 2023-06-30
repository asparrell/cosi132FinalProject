# COSI132 Final Project 
#### Annika Sparrell, Sonja Kleper, and Brynna Kilcline - Brandeis University, 2023

## About
This project is an attempt at making ChatGPT, the current technological sensation, a more reliable and verifiable source for users wanting answers to queries that are not necessarily well-answered by more traditional information retrieval tools such as Google. In the application, a user can submit a query and obtain a response from ChatGPT, version 3.5 turbo. A request is then made to the Google API to retrieve relevant sources relating to ChatGPT's answer. These sources come from two places: a direct Google query of the original input, and a request to ChatGPT for sources relating to its answer, which are then searched on Google. The sources are ranked on the righthand side of the page by their similarity score to the prose output of ChatGPT.

## To run this project:

### Initial setup instructions
Clone the repository into a directory of choice on your local drive. After cloning the project, use your terminal (Mac, Linux) or GitBash/Powershell (Windows) to navigate into the folder and run pip install -r requirements.txt to install the required packages. These packages, along with the versions we used, are
* googlesearch-python, version 1.2.3
* openai, gpt-3.5-turbo-0301
* sentence-transformers
* scikit-learn, version 1.2.2
* Flask, version 2.3.2
You will also need an OpenAI API key. Create a text file named api.txt in the top level directory of the project and paste your key into the newly created text file.

### Runtime instructions
From within the top level directory, run python main.py to begin the application and then navigate to http://127.0.0.1:5000/ in your browser. From there, you may enter a query of choice into the search bar on the webpage to test the functionality of the application. Note: due to limitations of the Google API, this process will take a while. Wait for approximately 8 minutes before resubmitting a query or refreshing the page. If you get a 429 error, congrats we did too! Unfortunately we have not found a good way around this. To see a sample output as a proxy, replace sources=google_results on line 67 of main.py with sources=s and rerun.
