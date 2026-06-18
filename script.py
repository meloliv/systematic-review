import os
import requests
import json
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

USER_EMAIL = os.getenv("EMAIL")
SEARCH_QUERY = os.getenv("SEARCH_QUERY")
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", 200))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output_files")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "Abstract not available"
    try:
        max_index = max([idx for indices in inverted_index.values() for idx in indices])
        words = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for idx in positions:
                words[idx] = word
        return " ".join(words)
    except Exception as e:
        return f"error {e}"

def fetch_openalex_articles(search_query, email, max_articles=100):
    headers = {"User-Agent": f"mailto:{email}"}
    base_url = "https://api.openalex.org/works"
    params = {
        "search": search_query,
        "per-page": 50,
        "cursor": "*"
    }
    collected_articles = []
    while len(collected_articles) < max_articles:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            break
        
        for item in results:
            if len(collected_articles) >= max_articles:
                break
            title = item.get("title", "No title")
            doi = item.get("doi", "No DOI")
            year = item.get("publication_year", "No year")
            authors_list = item.get("authorships", [])
            author_names = [author["author"]["display_name"] for author in authors_list if "author" in author]
            formatted_authors = "; ".join(author_names)
            abstract_index = item.get("abstract_inverted_index")
            abstract_text = reconstruct_abstract(abstract_index)
            collected_articles.append({
                "Title": title,
                "Authors": formatted_authors,
                "Year": year,
                "DOI": doi,
                "Abstract": abstract_text
            })
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break 
        params["cursor"] = next_cursor
        time.sleep(0.5)
    return collected_articles

def classify_by_keywords(abstract):
    abstract_lower = str(abstract).lower()
    mapping_scheme = {
        'mbt': [
            'model-based testing', 'mbt', 'model based', 'formal specification', 
            'state machine', 'uml', 'finite state'
        ],
        'obm': [
            'observation-based', 'obm', 'model inference', 'model learning', 
            'execution trace', 'log analysis', 'dynamic analysis', 'reverse engineering'
        ],
        'testes': [
            'test generation', 'test automation', 'test case', 'test suite', 
            'test oracle', 'coverage', 'test execution'
        ],
        'estudos de caso': [
            'case study', 'empirical evaluation', 'gui testing', 
            'web application', 'industrial application', 'experiment'
        ]
    }
    for category, keywords in mapping_scheme.items():
        if any(kw in abstract_lower for kw in keywords):
            return category
    return 'Geral/Outros'

def process_and_classify(data):
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df[df['Abstract'] != 'Abstract not available']
    df = df.dropna(subset=['Abstract', 'Year'])
    df['Year'] = df['Year'].astype(int)
    df['Category'] = df['Abstract'].apply(classify_by_keywords)
    df = df.sort_values(by=['Category', 'Year', 'Title'], ascending=[True, False, True])
    colunas_finais = ['Title', 'Year', 'Category', 'Authors', 'DOI', 'Abstract']
    return df[colunas_finais]

if __name__ == "__main__":
    json_raw_path = os.path.join(OUTPUT_FOLDER, '01_systematic_review_articles.json')
    json_classified_path = os.path.join(OUTPUT_FOLDER, '02_classified_articles.json')
    
    if os.path.exists(json_raw_path):
        with open(json_raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        raw_data = fetch_openalex_articles(SEARCH_QUERY, USER_EMAIL, MAX_ARTICLES) 
        if raw_data:
            with open(json_raw_path, 'w', encoding='utf-8') as json_file:
                json.dump(raw_data, json_file, indent=4, ensure_ascii=False)
    
    if raw_data:
        df_classified = process_and_classify(raw_data)
        if not df_classified.empty:
            df_classified.to_json(json_classified_path, orient='records', indent=4, force_ascii=False)