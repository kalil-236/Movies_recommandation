import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# --- 1. CHARGEMENT ET PRÉPARATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'movies.csv'))
ratings_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'ratings.csv'))

# Calcul des stats pour la logique hybride
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
df_final = pd.merge(movies_df, movie_stats, on='movieId', how='left').fillna(0)

# --- 2. PRÉPARATION DES MODÈLES ---

# A. Modèle Content-Based (Genres)
mlb = MultiLabelBinarizer()
genres_matrix = mlb.fit_transform(df_final['genres'].str.split('|'))
# On garde la matrice en mémoire pour calculer la similarité à la volée

# B. Modèle Collaborative (SVD)
reader = Reader(rating_scale=(0.5, 5.0))
# On entraîne sur les films ayant un minimum de vécu pour la SVD
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd_model.fit(trainset)

# --- 3. LES BRIQUES DE RECOMMANDATION ---

def get_movie_info(title_query):
    """Trouve le film et ses stats."""
    match = df_final[df_final['title'].str.contains(title_query, case=False, na=False)]
    return match.iloc[0] if not match.empty else None

def content_based_simple(movie_id, n=5):
    """Similitude basée uniquement sur les genres."""
    idx = df_final[df_final['movieId'] == movie_id].index[0]
    sim_scores = cosine_similarity(genres_matrix[idx].reshape(1, -1), genres_matrix).flatten()
    
    # On récupère les indices des n meilleurs (excluant lui-même)
    related_indices = sim_scores.argsort()[::-1][1:n+1]
    return df_final.iloc[related_indices].copy()

def collaborative_recommend_svd(movie_id, n=5):
    """Similitude basée sur les vecteurs latents SVD (Item-Item)."""
    try:
        inner_id = svd_model.trainset.to_inner_iid(movie_id)
        movie_vector = svd_model.qi[inner_id].reshape(1, -1)
        # Calcul de similarité entre le vecteur du film et TOUS les vecteurs de films SVD
        sim_scores = cosine_similarity(movie_vector, svd_model.qi).flatten()
        
        # On trie les indices internes
        related_inner_indices = sim_scores.argsort()[::-1][1:n+1]
        # Conversion index interne -> raw_id -> DataFrame
        related_ids = [svd_model.trainset.to_raw_iid(i) for i in related_inner_indices]
        return df_final[df_final['movieId'].isin(related_ids)].copy()
    except ValueError:
        # Si le film n'est pas dans le trainset SVD
        return content_based_simple(movie_id, n)

# --- 4. LOGIQUE HYBRIDE FINALE ---

def recommend_hybrid(movie_title, n=5):
    movie_info = get_movie_info(movie_title)
    
    if movie_info is None:
        return None, "Film introuvable"
    
    m_id = movie_info['movieId']
    m_title = movie_info['title']
    count = movie_info['rating_count']
    
    print(f"\n--- Analyse de '{m_title}' ({int(count)} votes) ---")
    
    if count >= 50:
        # Filtrage Collaboratif (SVD) : Très précis pour les films connus
        recs = collaborative_recommend_svd(m_id, n)
        method = "Collaborative Filtering (SVD Latent Space)"
    elif count >= 10:
        # On pourrait enrichir ici, mais utilisons le content-based pour l'instant
        recs = content_based_simple(m_id, n)
        method = "Content-Based (Genres + Popularité)"
    else:
        # Content-based pur : Seule option pour les films très peu connus
        recs = content_based_simple(m_id, n)
        method = "Content-Based (Simple Genres)"
    
    return recs, method

# --- 5. INTERFACE ---
if __name__ == "__main__":
    print("="*50)
    print("MOTEUR HYBRIDE (SVD + CONTENT-BASED)")
    print("="*50)
    
    while True:
        query = input("\nEntrez un film (ou 'q' pour quitter) : ")
        if query.lower() == 'q': break
        
        results, method = recommend_hybrid(query, n=5)
        
        if results is not None:
            print(f"Méthode utilisée : {method}")
            print(results[['title', 'genres', 'rating_count', 'rating_mean']].to_string(index=False))
        else:
            print(f"Erreur : {method}")




