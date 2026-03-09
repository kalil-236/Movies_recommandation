import streamlit as st
import pandas as pd
import os
import numpy as np
import requests
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# Configuration de la page
st.set_page_config(page_title="Movie Reco Hybride", page_icon="🎬", layout="wide")

# --- CONFIGURATION TMDB ---
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8" 

@st.cache_data(show_spinner=False)
def get_movie_poster(movie_title):
    """Récupère l'URL de l'affiche via l'API TMDB avec mise en cache"""
    base_url = "https://api.themoviedb.org/3/search/movie"
    clean_title = movie_title.split('(')[0].strip()
    params = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "fr-FR"}
    try:
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        if data['results'] and data['results'][0]['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['results'][0]['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=Pas+d'affiche"

# --- FONCTIONS UTILITAIRES ---
def extract_base_title(title):
    base = title.split('(')[0].strip()
    base = base.rstrip(', The').rstrip(', A').rstrip(', An').rstrip(',').strip()
    return base.split(':')[0].strip() if ':' in base else base

def normalize_title(title):
    title = title.strip()
    articles = [("The ", ", The"), ("A ", ", A"), ("An ", ", An")]
    for start, end in articles:
        if title.startswith(start): return title[len(start):] + end
    return title

def flexible_search(query, df):
    direct = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
    if not direct.empty: return direct
    norm = df[df['title'].str.contains(normalize_title(query), case=False, na=False, regex=False)]
    if not norm.empty: return norm
    words = query.lower().split()
    if len(words) == 1: return pd.DataFrame()
    mask = df['title'].str.lower().str.contains(words[0], na=False, regex=False)
    for word in words[1:]: mask &= df['title'].str.lower().str.contains(word, na=False, regex=False)
    return df[mask]

# --- CHARGEMENT ET CACHING ---
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    m_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')
    r_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')
    m_df, r_df = pd.read_csv(m_path), pd.read_csv(r_path)
    stats = r_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    stats.columns = ['movieId', 'rating_count', 'rating_mean']
    df = pd.merge(m_df, stats, on='movieId', how='left').fillna(0)
    return df[df['genres'] != '(no genres listed)'].copy(), r_df

@st.cache_resource
def train_svd_model(_ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(_ratings_df[['userId', 'movieId', 'rating']], reader)
    svd = SVD(n_factors=100, n_epochs=20).fit(data.build_full_trainset())
    return svd

@st.cache_data
def get_genres_matrix(df):
    return MultiLabelBinarizer().fit_transform(df['genres'].str.split('|'))

# Initialisation
with st.spinner("⏳ Préparation du moteur de recommandation..."):
    df_final, ratings_raw = load_data()
    svd_model = train_svd_model(ratings_raw)
    genres_matrix = get_genres_matrix(df_final)

# --- LOGIQUE RECO ---
def get_hybrid_recommendations(movie_id, movie_title, rating_count, n):
    base_title = extract_base_title(movie_title)
    franchise = df_final[df_final['title'].str.contains(base_title, case=False, na=False)]
    franchise = franchise[franchise['movieId'] != movie_id]
    
    if len(franchise) >= 1 and rating_count >= 50:
        collab_ids = [svd_model.trainset.to_raw_iid(i) for i in 
                      cosine_similarity(svd_model.qi[svd_model.trainset.to_inner_iid(movie_id)].reshape(1,-1), svd_model.qi).flatten().argsort()[::-1][1:n+1]]
        res = pd.concat([franchise.nlargest(2, 'rating_count'), df_final[df_final['movieId'].isin(collab_ids)]]).head(n)
        return res, "🎯 Hybride (Franchise + SVD)"
    return df_final.iloc[cosine_similarity(genres_matrix[df_final[df_final['movieId']==movie_id].index[0]].reshape(1,-1), genres_matrix).flatten().argsort()[::-1][1:n+1]].copy(), "🎬 Content-Based"

# --- INTERFACE ---
st.title("🎬 Trouvez votre prochain plaisir")

# Initialisation du Session State pour la stabilité
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
    st.session_state.method = ""

query = st.text_input("🔍 Rechercher un film :", placeholder="Matrix, Inception...", key="search_input")

if query:
    matches = flexible_search(query, df_final)
    if not matches.empty:
        options = matches.set_index('movieId')['title'].to_dict()
        selected_id = st.selectbox("📽️ Choisissez le film exact :", list(options.keys()), 
                                    format_func=lambda x: options[x], key="movie_selector_stable")
        
        movie_info = df_final[df_final['movieId'] == selected_id].iloc[0]
        
        # Détails du film sélectionné
        c1, c2 = st.columns([1, 3])
        with c1: st.image(get_movie_poster(movie_info['title']), width=180)
        with c2:
            st.subheader(movie_info['title'])
            st.caption(f"Genres : {movie_info['genres']}")
            st.metric("Note", f"{movie_info['rating_mean']:.2f}/5")

        if st.button("🎬 Obtenir des recommandations", type="primary"):
            with st.spinner("Analyse de vos goûts..."):
                recs, method = get_hybrid_recommendations(selected_id, movie_info['title'], movie_info['rating_count'], n_recs:=5)
                st.session_state.recommendations = recs
                st.session_state.method = method

        # Affichage Persistant (Session State)
        if st.session_state.recommendations is not None:
            st.divider()
            st.write(f"Méthode : {st.session_state.method}")
            st.subheader("🎯 Recommandations pour vous")
            
            # Grille de recommandations (5 colonnes)
            cols = st.columns(5)
            for i, (idx, row) in enumerate(st.session_state.recommendations.iterrows()):
                if i < 5:
                    with cols[i]:
                        st.image(get_movie_poster(row['title']), width=200)
                        st.markdown(f"**{row['title'][:25]}**")
                        st.caption(f"⭐ {row['rating_mean']:.1f}")
    else:
        st.warning("Aucun film trouvé.")