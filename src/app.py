import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Reco", page_icon="🎬")


# --- CONCEPT CLÉ : LE CACHING ---
# Normalement, Streamlit relance tout le script dès que l'utilisateur clique sur un bouton.
# @st.cache_data permet de garder les données en mémoire : le script ne chargera les CSV qu'une seule fois !
@st.cache_data
def load_and_preprocess_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')
    ratings_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')
    
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # Prétraitement 
    movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
    movie_stats_filtered = movie_stats[movie_stats['rating_count'] >= 10]
    df = pd.merge(movies_df, movie_stats_filtered, on='movieId', how='inner')
    df = df[df['genres'] != '(no genres listed)'].copy()
    
    # Préparation de la matrice de similarité
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(df['genres'].str.split('|'))
    sim_values = cosine_similarity(genres_matrix)
    sim_df = pd.DataFrame(sim_values, index=df['title'], columns=df['title'])
    
    return df, sim_df

# Chargement des données
df_final, sim_df = load_and_preprocess_data()

# --- INTERFACE STREAMLIT ---
st.title("🎬 Mon Moteur de Recommandation")
st.markdown("Trouvez votre prochain film basé sur vos goûts !")

# --- SIDEBAR (Barre latérale) ---
st.sidebar.header("Paramètres")
n_recs = st.sidebar.slider("Nombre de recommandations", 1, 20, 5)

# --- RECHERCHE ---
# Remplacement de l'input() par un widget de texte
query = st.text_input("Entrez le nom d'un film que vous avez aimé :", placeholder="Ex: Matrix, Toy Story...")

if query:
    # On utilise ta logique de recherche approximative
    matches = df_final[df_final['title'].str.contains(query, case=False, na=False)]
    
    if matches.empty:
        st.error(f"Désolé, aucun film trouvé pour '{query}'")
    else:
        # On propose une liste déroulante si plusieurs films correspondent à la recherche
        # C'est plus ergonomique que de prendre le premier par défaut
        options = matches['title'].tolist()
        ref_movie = st.selectbox("Nous avons trouvé plusieurs films, lequel choisissez-vous ?", options)
        
        if st.button("Obtenir des recommandations"):
            # Calcul des scores 
            sim_scores = sim_df[ref_movie].copy()
            rec_df = sim_scores.to_frame(name='similarity').reset_index()
            
            rec_df = rec_df.merge(
                df_final[['title', 'rating_mean', 'rating_count', 'genres']], 
                on='title', how='inner'
            )
            
            # Exclusion et Tri
            rec_df = rec_df[rec_df['title'] != ref_movie]
            results = rec_df.sort_values(
                by=['similarity', 'rating_count', 'rating_mean'], 
                ascending=[False, False, False]
            ).head(n_recs)
            
            # --- AFFICHAGE WEB ---
            st.success(f"Films recommandés après avoir vu : **{ref_movie}**")
            
            # On utilise st.dataframe pour un tableau interactif
            st.dataframe(results[['title', 'genres', 'rating_mean', 'rating_count']], use_container_width=True)
