import os
import re
import requests
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

USER_EMAIL = os.getenv("EMAIL")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEARCH_QUERY = os.getenv("SEARCH_QUERY")
YEAR_FROM = os.getenv("YEAR_FROM") 
YEAR_TO = os.getenv("YEAR_TO")     
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", 10000))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output_files")
mapping_env = os.getenv("MAPPING_SCHEME")
MAPPING_SCHEME_DICT = json.loads(mapping_env) if mapping_env else {}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DASH_VARIANTS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
MULTI_SEP = re.compile(r"[-\s]+")

RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
retry_network = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)


@retry_network
def _get(url, headers=None, params=None, timeout=30):
    return requests.get(url, headers=headers, params=params, timeout=timeout)


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


def fetch_all_sources(search_query, email, max_articles=10000):
    articles = fetch_openalex_articles(search_query, email, max_articles, OPENALEX_API_KEY, YEAR_FROM, YEAR_TO)
    articles += fetch_crossref_articles(search_query, email, max_articles, YEAR_FROM, YEAR_TO)
    return dedup_by_doi(articles)


def _normalize(text):
    return MULTI_SEP.sub(" ", DASH_VARIANTS.sub("-", text).lower()).strip()


def classify_by_keywords(row):
    text_to_search = _normalize(f"{row['Title']} {row['Abstract']}")
    for category, keywords in MAPPING_SCHEME_DICT.items():
        if any(_normalize(kw) in text_to_search for kw in keywords):
            return category
    return 'Geral/Outros'


def process_and_classify(data):
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # "No year" e afins viram NaN
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df['Category'] = df.apply(classify_by_keywords, axis=1)
    df = df.sort_values(by=['Category', 'Year', 'Title'], ascending=[True, False, True])
    colunas_finais = ['Title', 'Year', 'Category', 'Authors', 'DOI', 'Abstract', 'Source']
    return df[colunas_finais]


def fetch_openalex_articles(search_query, email, max_articles=10000, api_key=None, year_from=None, year_to=None):
    headers = {"User-Agent": f"mailto:{email}"}
    base_url = "https://api.openalex.org/works"
    filter_parts = [f"title_and_abstract.search:{search_query}"]
    if year_from:
        filter_parts.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filter_parts.append(f"to_publication_date:{year_to}-12-31")
    params = {
        "filter": ",".join(filter_parts),
        "per-page": 50,
        "cursor": "*"
    }
    if api_key:
        params["api_key"] = api_key
    collected_articles = []
    while len(collected_articles) < max_articles:
        response = _get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"erro openalex {response.status_code}")
            print(f"detalhes: {response.text}")
            break
        data = response.json()
        results = data.get("results", [])
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
                "Abstract": abstract_text,
                "Source": "openalex"
            })
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        time.sleep(0.5)
    print(f"openalex: {len(collected_articles)} artigos")
    return collected_articles


def fetch_crossref_articles(search_query, email, max_articles=10000, year_from=None, year_to=None):
    base_url = "https://api.crossref.org/works"
    rows = 100
    offset = 0
    collected_articles = []
    query_terms = [t.lower() for t in search_query.replace('"', "").split() if len(t) > 2]

    filter_parts = []
    if year_from:
        filter_parts.append(f"from-pub-date:{year_from}-01-01")
    if year_to:
        filter_parts.append(f"until-pub-date:{year_to}-12-31")

    while len(collected_articles) < max_articles:
        params = {
            "query.bibliographic": search_query,
            "rows": rows,
            "offset": offset,
            "mailto": email,
        }
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        response = _get(base_url, params=params)
        if response.status_code != 200:
            print(f"erro crossref {response.status_code}")
            print(f"detalhes: {response.text}")
            break
        data = response.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            break
        for item in items:
            if len(collected_articles) >= max_articles:
                break
            title_list = item.get("title") or []
            title = title_list[0] if title_list else "No title"
            abstract = item.get("abstract", "Abstract not available")
            text_to_check = f"{title} {abstract}".lower()
            if query_terms and not any(term in text_to_check for term in query_terms):
                continue
            authors_list = item.get("author", [])
            author_names = [
                f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_list
            ]
            formatted_authors = "; ".join(author_names)
            year = (item.get("published") or {}).get("date-parts", [[None]])[0][0] or "No year"
            collected_articles.append({
                "Title": title,
                "Authors": formatted_authors,
                "Year": year,
                "DOI": item.get("DOI", "No DOI"),
                "Abstract": abstract,
                "Source": "crossref"
            })
        offset += rows
        if offset >= data.get("message", {}).get("total-results", 0):
            break
        time.sleep(0.5)
    print(f"crossref: {len(collected_articles)} artigos")
    return collected_articles


def dedup_by_doi(articles):
    seen = {}
    no_doi = []
    for art in articles:
        doi = art.get("DOI")
        if doi and doi != "No DOI":
            key = doi.lower().strip()
            if key not in seen:
                seen[key] = art
        else:
            no_doi.append(art)
    result = list(seen.values()) + no_doi
    print(f"dedup: {len(articles)} -> {len(result)} (removidos {len(articles) - len(result)} duplicados por doi)")
    return result


if __name__ == "__main__":
    json_raw_path = os.path.join(OUTPUT_FOLDER, '01_systematic_review_articles.json')
    json_classified_path = os.path.join(OUTPUT_FOLDER, '02_classified_articles.json')
    if os.path.exists(json_raw_path):
        with open(json_raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        raw_data = fetch_all_sources(SEARCH_QUERY, USER_EMAIL, MAX_ARTICLES)
        if raw_data:
            with open(json_raw_path, 'w', encoding='utf-8') as json_file:
                json.dump(raw_data, json_file, indent=4, ensure_ascii=False)
    if raw_data:
        df_classified = process_and_classify(raw_data)
        if not df_classified.empty:
            df_classified.to_json(json_classified_path, orient='records', indent=4, force_ascii=False)