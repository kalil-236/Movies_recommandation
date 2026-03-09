import streamlit as st
import pandas as pd
import os
import numpy as np
import requests  # <-- Ajout pour l'API TMDB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

st.set_page_config(page_title="Touvez votre prochain plaisir", page_icon="🎬", layout="wide")

# --- CONFIGURATION TMDB ---
# Remplacez par votre propre clé API si vous en avez une, sinon celle-ci est pour le test
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8" 

def get_movie_poster(movie_title):
    """Récupère l'URL de l'affiche via l'API TMDB"""
    base_url = "https://api.themoviedb.org/3/search/movie"
    # On nettoie le titre (on enlève l'année pour la recherche API)
    clean_title = movie_title.split('(')[0].strip()
    
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "language": "fr-FR"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    # Image par défaut si non trouvé
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- [VOS FONCTIONS UTILITAIRES RESTENT IDENTIQUES] ---
def extract_base_title(title):
    base = title.split('(')[0].strip()
    base = base.rstrip(', The').rstrip(', A').rstrip(', An').rstrip(',').strip()
    if ':' in base:
        base = base.split(':')[0].strip()
    return base

def normalize_title(title):
    title = title.strip()
    articles = [("The ", ", The"), ("A ", ", A"), ("An ", ", An")]
    for article_start, article_end in articles:
        if title.startswith(article_start):
            return title[len(article_start):] + article_end
    return title

def flexible_search(query, df):
    direct_match = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
    if not direct_match.empty: return direct_match
    normalized_query = normalize_title(query)
    normalized_match = df[df['title'].str.contains(normalized_query, case=False, na=False, regex=False)]
    if not normalized_match.empty: return normalized_match
    words = query.lower().split()
    if len(words) == 1: return pd.DataFrame()
    mask = df['title'].str.lower().str.contains(words[0], na=False, regex=False)
    for word in words[1:]:
        mask &= df['title'].str.lower().str.contains(word, na=False, regex=False)
    return df[mask]

# --- CHARGEMENT ET CACHING ---
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')
    ratings_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
    df = pd.merge(movies_df, movie_stats, on='movieId', how='left').fillna(0)
    df = df[df['genres'] != '(no genres listed)'].copy()
    return df, ratings_df

@st.cache_resource
def train_svd_model(_ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(_ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    svd.fit(trainset)
    return svd

@st.cache_data
def get_genres_matrix(df):
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(df['genres'].str.split('|'))

with st.spinner("⏳ Initialisation..."):
    df_final, ratings_raw = load_data()
    svd_model = train_svd_model(ratings_raw)
    genres_matrix = get_genres_matrix(df_final)

# --- LOGIQUE DE RECOMMANDATION ---
def get_collaborative_recommendations(movie_id, n):
    try:
        inner_id = svd_model.trainset.to_inner_iid(movie_id)
        movie_vector = svd_model.qi[inner_id].reshape(1, -1)
        sim_scores = cosine_similarity(movie_vector, svd_model.qi).flatten()
        related_inner_ids = sim_scores.argsort()[::-1][1:n+1]
        related_ids = [svd_model.trainset.to_raw_iid(i) for i in related_inner_ids]
        return df_final[df_final['movieId'].isin(related_ids)].copy()
    except:
        return get_content_based_recommendations(movie_id, n)

def get_content_based_recommendations(movie_id, n):
    idx = df_final[df_final['movieId'] == movie_id].index[0]
    sim_scores = cosine_similarity(genres_matrix[idx].reshape(1, -1), genres_matrix).flatten()
    related_indices = sim_scores.argsort()[::-1][1:n+1]
    return df_final.iloc[related_indices].copy()

def get_hybrid_recommendations(movie_id, movie_title, rating_count, n):
    base_title = extract_base_title(movie_title)
    franchise_films = df_final[df_final['title'].str.contains(base_title, case=False, na=False)].copy()
    franchise_films = franchise_films[franchise_films['movieId'] != movie_id]
    
    if len(franchise_films) >= 2 and rating_count >= 50:
        top_franchise = franchise_films.nlargest(2, 'rating_count')
        collab_recs = get_collaborative_recommendations(movie_id, n=n)
        collab_recs = collab_recs[~collab_recs['movieId'].isin(top_franchise['movieId'])]
        results = pd.concat([top_franchise, collab_recs]).head(n)
        method = "🎯 Hybride (Franchise + SVD)"
    elif rating_count >= 50:
        results = get_collaborative_recommendations(movie_id, n)
        method = "🤝 Collaborative Filtering (SVD)"
    else:
        results = get_content_based_recommendations(movie_id, n)
        method = "🎬 Content-Based (Genres)"
    return results, method

# --- INTERFACE STREAMLIT ---
st.title("🎬 Movie Reco Hybride")

st.sidebar.header("⚙️ Paramètres")
n_recs = st.sidebar.slider("Nombre de recommandations", 1, 10, 5)

query = st.text_input("🔍 Rechercher un film :", placeholder="Ex: Matrix, Inception...")

if query:
    matches = flexible_search(query, df_final)
    if not matches.empty:
        options = matches.set_index('movieId')['title'].to_dict()
        selected_id = st.selectbox("📽️ Choisissez le film exact :", list(options.keys()), 
                                    format_func=lambda x: options[x], key="movie_selector_stable")
        
        movie_info = df_final[df_final['movieId'] == selected_id].iloc[0]
        
        # Affichage du film sélectionné avec son affiche
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            st.image(get_movie_poster(movie_info['title']), width=200)
        with col_txt:
            st.subheader(movie_info['title'])
            st.write(f"**Genres** : {movie_info['genres']}")
            st.metric("⭐ Note", f"{movie_info['rating_mean']:.2f}/5")

        if st.button("🎬 Obtenir des recommandations", type="primary"):
            results, method_name = get_hybrid_recommendations(selected_id, movie_info['title'], 
                                                              movie_info['rating_count'], n_recs)
            
            st.info(f"**Méthode** : {method_name}")
            st.subheader(f"🎯 Top {len(results)} recommandations")
            
            # --- NOUVEL AFFICHAGE EN GRILLE ---
            cols = st.columns(n_recs)
            for i, (idx, row) in enumerate(results.iterrows()):
                with cols[i % n_recs]:
                    poster_url = get_movie_poster(row['title'])
                    
                    st.image(poster_url, width=200)
                    # On coupe le titre s'il est trop long pour la grille
                    short_title = (row['title'][:25] + '..') if len(row['title']) > 25 else row['title']
                    st.markdown(f"**{short_title}**")
                    st.caption(f"⭐ {row['rating_mean']:.1f}/5")
    else:
        st.warning("❌ Aucun film trouvé.")