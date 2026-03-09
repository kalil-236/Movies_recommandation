import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

st.set_page_config(page_title="Movie Reco Hybride", page_icon="🎬")

# --- FONCTIONS UTILITAIRES ---

def extract_base_title(title):
    """
    Extrait le nom de base d'un film (sans année, sans articles)
    
    Exemples :
    - "Matrix, The (1999)" → "Matrix"
    - "Star Wars: Episode IV (1977)" → "Star Wars"
    - "Toy Story 2 (1999)" → "Toy Story 2"
    """
    # Retirer l'année entre parenthèses
    base = title.split('(')[0].strip()
    
    # Retirer les articles finaux
    base = base.rstrip(', The').rstrip(', A').rstrip(', An').rstrip(',').strip()
    
    # Retirer les sous-titres après ':' (optionnel)
    if ':' in base:
        base = base.split(':')[0].strip()
    
    return base
def normalize_title(title):
    """
    Normalise un titre en déplaçant les articles
    
    Exemples :
    - "The Matrix" → "Matrix, The"
    - "A Beautiful Mind" → "Beautiful Mind, A"
    - "Matrix" → "Matrix"
    - "The Matrix Reloaded" → "Matrix Reloaded, The"
    """
    title = title.strip()
    
    # Liste des articles à détecter (avec espace après pour éviter "Theater")
    articles = [
        ("The ", ", The"),
        ("A ", ", A"),
        ("An ", ", An")
    ]
    
    # Vérifier si le titre commence par un article
    for article_start, article_end in articles:
        if title.startswith(article_start):
            # Déplacer l'article à la fin
            return title[len(article_start):] + article_end
    
    return title

def flexible_search(query, df):
    """
    Recherche ultra-flexible qui gère :
    - L'ordre des mots
    - Les articles au début/fin
    - Les variantes de titres
    """
    # 1. Recherche directe (query exacte)
    direct_match = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
    if not direct_match.empty:
        return direct_match
    
    # 2. Normaliser la query (déplacer les articles)
    normalized_query = normalize_title(query)
    normalized_match = df[df['title'].str.contains(normalized_query, case=False, na=False, regex=False)]
    if not normalized_match.empty:
        return normalized_match
    
    # 3. Recherche par mots séparés (tous les mots présents)
    words = query.lower().split()
    if len(words) == 1:
        return pd.DataFrame()  # Un seul mot, déjà échoué
    
    # Tous les mots doivent être présents
    mask = df['title'].str.lower().str.contains(words[0], na=False, regex=False)
    for word in words[1:]:
        mask &= df['title'].str.lower().str.contains(word, na=False, regex=False)
    
    flexible_match = df[mask]
    return flexible_match


# --- CHARGEMENT ET CACHING ---

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')
    ratings_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')
    
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # Statistiques globales
    movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
    
    # Dataframe final pour l'affichage
    df = pd.merge(movies_df, movie_stats, on='movieId', how='left').fillna(0)
    df = df[df['genres'] != '(no genres listed)'].copy()
    
    return df, ratings_df


@st.cache_resource
def train_svd_model(_ratings_df):
    """Entraînement du modèle SVD (mis en cache)"""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(_ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    svd.fit(trainset)
    return svd


@st.cache_data
def get_genres_matrix(df):
    """Création de la matrice de genres (mise en cache)"""
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(df['genres'].str.split('|'))
    return matrix


# Initialisation des données et modèles
with st.spinner("⏳ Chargement des données et entraînement du modèle SVD..."):
    df_final, ratings_raw = load_data()
    svd_model = train_svd_model(ratings_raw)
    genres_matrix = get_genres_matrix(df_final)


# --- LOGIQUE DE RECOMMANDATION ---

def get_collaborative_recommendations(movie_id, n):
    """Recommandations collaborative filtering (SVD)"""
    try:
        inner_id = svd_model.trainset.to_inner_iid(movie_id)
        movie_vector = svd_model.qi[inner_id].reshape(1, -1)
        sim_scores = cosine_similarity(movie_vector, svd_model.qi).flatten()
        related_inner_ids = sim_scores.argsort()[::-1][1:n+1]
        
        related_ids = []
        for i in related_inner_ids:
            try:
                raw_id = svd_model.trainset.to_raw_iid(i)
                related_ids.append(raw_id)
            except:
                pass  # Ignorer si mapping impossible
        
        return df_final[df_final['movieId'].isin(related_ids)].copy()
    except:
        # Fallback si le film n'est pas dans le trainset
        return get_content_based_recommendations(movie_id, n)


def get_content_based_recommendations(movie_id, n):
    """Recommandations content-based (genres)"""
    idx = df_final[df_final['movieId'] == movie_id].index[0]
    sim_scores = cosine_similarity(genres_matrix[idx].reshape(1, -1), genres_matrix).flatten()
    related_indices = sim_scores.argsort()[::-1][1:n+1]
    return df_final.iloc[related_indices].copy()


def get_hybrid_recommendations(movie_id, movie_title, rating_count, n):
    """
    Système hybride intelligent :
    - Priorise les suites/prequels de la même franchise
    - Complète avec collaborative filtering ou content-based
    """
    
    # Extraire le nom de base du film
    base_title = extract_base_title(movie_title)
    
    # Chercher les films de la même franchise
    franchise_films = df_final[
        df_final['title'].str.contains(base_title, case=False, na=False)
    ].copy()
    
    # Exclure le film lui-même
    franchise_films = franchise_films[franchise_films['movieId'] != movie_id]
    
    # Décision de la stratégie
    if len(franchise_films) >= 2 and rating_count >= 50:
        # STRATÉGIE 1 : Hybride (Franchise + Collaborative)
        # Prendre les 2 films de franchise les plus populaires
        top_franchise = franchise_films.nlargest(2, 'rating_count')
        
        # Compléter avec collaborative (n-2 films)
        collab_recs = get_collaborative_recommendations(movie_id, n=n)
        
        # Exclure les films déjà dans franchise
        collab_recs = collab_recs[~collab_recs['movieId'].isin(top_franchise['movieId'])]
        
        # Combiner
        results = pd.concat([top_franchise, collab_recs]).head(n)
        method = "🎯 Hybride (Franchise + Collaborative SVD)"
        
    elif rating_count >= 50:
        # STRATÉGIE 2 : Pure Collaborative (film populaire sans franchise)
        results = get_collaborative_recommendations(movie_id, n)
        method = "🤝 Collaborative Filtering (SVD)"
        
    elif rating_count >= 10:
        # STRATÉGIE 3 : Content-Based pour films moyennement notés
        results = get_content_based_recommendations(movie_id, n)
        method = "🎬 Content-Based (Genres)"
        
    else:
        # STRATÉGIE 4 : Content-Based simple pour films obscurs
        results = get_content_based_recommendations(movie_id, n)
        method = "📽️ Content-Based Simple (Cold Start)"
    
    return results, method


# --- INTERFACE STREAMLIT ---

st.title("🎬 Système de Recommandation de Films Hybride")
st.markdown("*Powered by SVD Collaborative Filtering & Content-Based Filtering*")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
n_recs = st.sidebar.slider("Nombre de recommandations", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Méthodes utilisées")
st.sidebar.info(
    "**≥50 votes + franchise** : Hybride (Suites + SVD)\n\n"
    "**≥50 votes** : Collaborative SVD\n\n"
    "**≥10 votes** : Content-Based\n\n"
    "**<10 votes** : Content-Based (Cold Start)"
)

# Recherche
query = st.text_input(
    "🔍 Rechercher un film :", 
    placeholder="Ex: Matrix, Inception, Toy Story, Star Wars..."
)

if query:
    # Utilisation de la recherche flexible
    matches = flexible_search(query, df_final)
    
    if not matches.empty:
        # Afficher les options trouvées
        options = matches.set_index('movieId')['title'].to_dict()
        
        # --- CORRECTION ICI : Clé statique pour éviter le crash JS ---
        selected_id = st.selectbox(
            "📽️ Choisissez le film exact :", 
            list(options.keys()), 
            format_func=lambda x: options[x],
            key="movie_selector_stable" 
        )
        
        movie_info = df_final[df_final['movieId'] == selected_id].iloc[0]
        
        # Utilisation d'un container pour regrouper les résultats proprement
        result_container = st.container()

        with result_container:
            # Affichage des informations du film
            with st.expander("ℹ️ Informations sur ce film", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Nombre de votes", int(movie_info['rating_count']))
                col2.metric("⭐ Note moyenne", f"{movie_info['rating_mean']:.2f}/5")
                col3.metric("🎭 Genres", movie_info['genres'].count('|') + 1)
                
                st.markdown(f"**Genres** : {movie_info['genres']}")
            
            # Bouton de recommandation
            if st.button("🎬 Obtenir des recommandations", type="primary"):
                with st.spinner("Calcul des recommandations en cours..."):
                    # Génération des recommandations
                    results, method_name = get_hybrid_recommendations(
                        movie_id=selected_id,
                        movie_title=movie_info['title'],
                        rating_count=movie_info['rating_count'],
                        n=n_recs
                    )
                    
                    # Affichage des résultats
                    st.success(f"Films recommandés après avoir vu : **{movie_info['title']}**")
                    st.info(f"**Méthode utilisée** : {method_name}")
                    
                    # Tableau des recommandations
                    st.subheader(f"🎯 Top {len(results)} recommandations")
                    
                    display_df = results[['title', 'genres', 'rating_mean', 'rating_count']].copy()
                    display_df.columns = ['Titre', 'Genres', 'Note moyenne', 'Nombre de votes']
                    display_df['Note moyenne'] = display_df['Note moyenne'].apply(lambda x: f"{x:.2f}/5")
                    display_df['Nombre de votes'] = display_df['Nombre de votes'].astype(int)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Statistiques des recommandations
                    with st.expander("📈 Statistiques des recommandations"):
                        c1, c2 = st.columns(2)
                        c1.metric(
                            "Note moyenne", 
                            f"{results['rating_mean'].mean():.2f}/5"
                        )
                        c2.metric(
                            "Popularité moyenne", 
                            f"{int(results['rating_count'].mean())} votes"
                        )
    else:
        st.warning(f"❌ Aucun film trouvé pour '{query}'. Essayez un autre terme de recherche.")
else:
    st.info("👆 Entrez le nom d'un film dans la barre de recherche pour commencer !")