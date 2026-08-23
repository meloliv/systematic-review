import os
import json
import time
import requests
from glob import glob
from dotenv import load_dotenv
from script import reconstruct_abstract

load_dotenv()

USER_EMAIL = os.getenv("EMAIL")
INPUT_FOLDER = "input"
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output_files")

def fetch_abstract(article):
    headers = {"User-Agent": f"mailto:{USER_EMAIL}"} if USER_EMAIL else {}
    doi = article.get("doi")
    if doi:
        doi_id = doi.replace("https://doi.org/", "") if "doi.org" in doi else doi
        url = f"https://api.openalex.org/works/https://doi.org/{doi_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return reconstruct_abstract(response.json().get("abstract_inverted_index"))
    title = article.get("title", "")
    if title:
        url = "https://api.openalex.org/works"
        params = {"filter": f"title.search:\"{title}\""}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                return reconstruct_abstract(data["results"][0].get("abstract_inverted_index"))
    return "abstract not found"

def process_all_files():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    input_files = glob(os.path.join(INPUT_FOLDER, "*.json"))
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_filename = filename.replace(".json", "_abstracts.json")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            articles = json.load(f)
        for article in articles:
            article['abstract'] = fetch_abstract(article)
            time.sleep(0.2)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    process_all_files()