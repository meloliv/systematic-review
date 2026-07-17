import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
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
        if cat == 'inferencia_e_modelagem':
            palette[cat] = '#e63946'         
        elif cat == 'testes_e_verificacao':
            palette[cat] = '#457b9d'        
        elif cat == 'sistemas_e_dominios':
            palette[cat] = '#2a9d8f'       
        else:
            palette[cat] = '#d3d3d3'       
    return palette

def plot_articles_by_category(df):
    plt.figure(figsize=(10, 6))
    category_order = df['Category'].value_counts().index
    palette = get_custom_palette(category_order)
    sns.countplot(data=df, y='Category', order=category_order, palette=palette, hue='Category', legend=False)
    plt.title('Artigos por Categoria (Testes e Sistemas Distribuídos)', fontsize=14, fontweight='bold', pad=15)
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
    plt.title('Tendências das áreas de pesquisa', fontsize=16, fontweight='bold', pad=20)
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
    plt.title('Distribuição de Artigos', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_04_category_distribution.png'), dpi=300)
    plt.close()

def plot_top_authors(df, top_n=10):
    plt.figure(figsize=(10, 6))
    all_authors = df['Authors'].str.split('; ').explode().dropna()
    all_authors = all_authors[all_authors.str.strip() != ""]
    top_authors = all_authors.value_counts().head(top_n)
    sns.barplot(x=top_authors.values, y=top_authors.index, color="#457b9d")
    plt.title(f'Top {top_n} Autores Mais Publicados', fontsize=14, fontweight='bold', pad=15)
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
    plt.title('Distribuição Acadêmica', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Total de Publicações (%)', fontsize=12)
    plt.legend(title='Categorias', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.margins(x=0, y=0)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_07_market_share.png'), dpi=300)
    plt.close()

def plot_wordcloud(df):
    plt.figure(figsize=(12, 6))
    text = " ".join(title for title in df['Title'].dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(["based", "model", "testing", "software", "system", "approach", "using", "method"])
    wordcloud = WordCloud(width=800, height=400, background_color="white", 
                          colormap="viridis", stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Termos Mais Frequentes nos Títulos', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_08_wordcloud.png'), dpi=300)
    plt.close()

def plot_author_activity_heatmap(df, top_n=10):
    plt.figure(figsize=(12, 8))
    df_authors = df.copy()
    df_authors['Author'] = df_authors['Authors'].str.split('; ')
    df_authors = df_authors.explode('Author').dropna(subset=['Author']).reset_index(drop=True)
    df_authors = df_authors[df_authors['Author'].str.strip() != ""]
    top_authors_list = df_authors['Author'].value_counts().head(top_n).index
    df_top = df_authors[df_authors['Author'].isin(top_authors_list)]
    activity_matrix = pd.crosstab(df_top['Author'], df_top['Year'])
    sns.heatmap(activity_matrix, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
    plt.title('Atividade dos Top Autores ao Longo do Tempo', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Autor', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_09_author_heatmap.png'), dpi=300)
    plt.close()

def plot_cumulative_growth_by_category(df):
    plt.figure(figsize=(14, 7))
    trend_data = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)
    cumulative_data = trend_data.cumsum()
    cumulative_long = cumulative_data.reset_index().melt(id_vars='Year', var_name='Category', value_name='Cumulative Count')
    palette = get_custom_palette(cumulative_long['Category'].unique())
    sns.lineplot(data=cumulative_long, x='Year', y='Cumulative Count', hue='Category', 
                 linewidth=3, palette=palette)
    plt.title('Crescimento Acumulado de Publicações por Categoria', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Total Acumulado de Artigos', fontsize=12)
    plt.legend(title='Categorias', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'chart_10_cumulative_category.png'), dpi=300)
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
        plot_wordcloud(df_articles)
        plot_author_activity_heatmap(df_articles)
        plot_cumulative_growth_by_category(df_articles)