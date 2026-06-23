import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output_files")

def load_data():
    file_path = os.path.join(OUTPUT_FOLDER, '02_classified_articles.json')
    if not os.path.exists(file_path):
        return None
    df = pd.read_json(file_path)
    df = df[df['Year'] > 1990] 
    return df

def get_custom_palette(categories):
    palette = {}
    for cat in categories:
        if cat == 'mbt':
            palette[cat] = '#e63946'         
        elif cat == 'obm':
            palette[cat] = '#457b9d'        
        elif cat == 'testes':
            palette[cat] = '#2a9d8f'       
        elif cat == 'estudos de caso':
            palette[cat] = '#e9c46a'         
        else:
            palette[cat] = '#d3d3d3'       
    return palette

def plot_articles_by_category(df):
    plt.figure(figsize=(10, 6))
    category_order = df['Category'].value_counts().index
    palette = get_custom_palette(category_order)
    sns.countplot(data=df, y='Category', order=category_order, palette=palette)
    plt.title('Artigos por Categoria (Revisão MBT)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Número de Artigos', fontsize=12)
    plt.ylabel('Categoria', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_01_categories.png'), dpi=300)
    plt.close()

def plot_articles_by_year(df):
    plt.figure(figsize=(12, 6))
    yearly_counts = df['Year'].value_counts().sort_index()
    plt.fill_between(yearly_counts.index, yearly_counts.values, color="#457b9d", alpha=0.4)
    plt.plot(yearly_counts.index, yearly_counts.values, color="#1d3557", linewidth=2, marker="o")
    plt.title('Evolução Global de Publicações por Ano', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Número de Artigos', fontsize=12)
    plt.xticks(yearly_counts.index, rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_02_years.png'), dpi=300)
    plt.close()

def plot_category_trends_over_time(df):
    plt.figure(figsize=(14, 7))
    trend_data = df.groupby(['Year', 'Category']).size().reset_index(name='Count')
    palette = get_custom_palette(trend_data['Category'].unique())
    
    sns.lineplot(data=trend_data, x='Year', y='Count', hue='Category', 
                 linewidth=3, marker='o', markersize=8, palette=palette)
    
    plt.title('Tendências das Áreas de Pesquisa (MBT e Análise Dinâmica)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Quantidade de Artigos', fontsize=12)
    plt.legend(title='Categorias', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_03_trends_linha.png'), dpi=300)
    plt.close()

def plot_category_distribution(df):
    plt.figure(figsize=(8, 8))
    category_counts = df['Category'].value_counts()
    palette = [get_custom_palette([cat])[cat] for cat in category_counts.index]
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=palette)
    plt.title('Distribuição Percentual de Artigos', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_04_category_distribution.png'), dpi=300)
    plt.close()

def plot_top_authors(df, top_n=10):
    plt.figure(figsize=(10, 6))
    all_authors = df['Authors'].str.split('; ').explode().dropna()
    all_authors = all_authors[all_authors.str.strip() != ""]
    top_authors = all_authors.value_counts().head(top_n)
    sns.barplot(x=top_authors.values, y=top_authors.index, color="#457b9d")
    plt.title(f'Top {top_n} Autores Mais Publicados na Área', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Número de Artigos', fontsize=12)
    plt.ylabel('Autor', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_05_top_authors.png'), dpi=300)
    plt.close()

def plot_market_share_area(df):
    plt.figure(figsize=(14, 7))
    crosstab = pd.crosstab(df['Year'], df['Category'])
    crosstab_perc = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    categories = crosstab.columns
    colors = [get_custom_palette([cat])[cat] for cat in categories]
    
    crosstab_perc.plot(kind='area', stacked=True, color=colors, alpha=0.85, figsize=(14, 7))

    plt.title('Distribuição Académica (Market Share)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Percentagem do Total de Publicações (%)', fontsize=12)
    plt.legend(title='Categorias', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.margins(x=0, y=0)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_07_market_share.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    df_articles = load_data()
    if df_articles is not None and not df_articles.empty:
        sns.set_theme(style="white")
        plot_articles_by_category(df_articles)
        plot_articles_by_year(df_articles)
        plot_category_trends_over_time(df_articles)
        plot_category_distribution(df_articles)
        plot_top_authors(df_articles)
        plot_market_share_area(df_articles)