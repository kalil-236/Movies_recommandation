import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')
ratings_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')

if not os.path.exists(movies_path):
    raise FileNotFoundError(f"Fichier introuvable : {movies_path}")

movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)

# ---  NETTOYAGE ET FILTRAGE ---
# Calcul des stats : moyenne et nombre de votes
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']

# Filtrage (au moins 10 notes) et exclusion des films sans genres
movie_stats_filtered = movie_stats[movie_stats['rating_count'] >= 10]
df_final = pd.merge(movies_df, movie_stats_filtered, on='movieId', how='inner')
df_final = df_final[df_final['genres'] != '(no genres listed)'].copy()

# ---  VECTORISATION DES GENRES (One-Hot Encoding) ---
mlb = MultiLabelBinarizer()
genre_list = df_final['genres'].str.split('|')
genres_matrix_raw = mlb.fit_transform(genre_list)

# Matrice de similarité cosinus
# sim_df utilise les titres en index et colonnes pour faciliter la recherche
cosine_sim_values = cosine_similarity(genres_matrix_raw)
sim_df = pd.DataFrame(cosine_sim_values, index=df_final['title'], columns=df_final['title'])

# ---  FONCTION DE RECOMMANDATION ---
def recommend_movies(title_query, n=5):
    # Recherche approximative du titre
    matches = df_final[df_final['title'].str.contains(title_query, case=False, na=False)]
    
    if matches.empty:
        return f"Aucun film trouvé pour '{title_query}'"
    
    ref_movie_title = matches.iloc[0]['title']
    print(f"\n--- Recherche de recommandations pour : {ref_movie_title} ---")

    # Calcul des scores
    sim_scores = sim_df[ref_movie_title].copy()
    rec_df = sim_scores.to_frame(name='similarity').reset_index()

    # Fusion avec les stats (on utilise 'inner' pour garder uniquement le catalogue filtré)
    rec_df = rec_df.merge(
        df_final[['title', 'rating_mean', 'rating_count', 'genres']], 
        on='title', 
        how='inner'
    )

    # Exclure le film lui-même et trier
    rec_df = rec_df[rec_df['title'] != ref_movie_title]
    recommendations = rec_df.sort_values(
        by=['similarity', 'rating_count', 'rating_mean'], 
        ascending=[False, False, False]
    )

    return recommendations.head(n)

'''# --- 5. TEST ---
if __name__ == "__main__":
    resultats = recommend_movies("Matrix", n=5)
    
    if isinstance(resultats, pd.DataFrame):
        print(resultats[['title', 'similarity', 'rating_count', 'rating_mean']])
    else:
        print(resultats)'''


# --- 5. INTERFACE UTILISATEUR (MENU) ---
if __name__ == "__main__":
    print("="*50)
    print("BIENVENUE SUR VOTRE MOTEUR DE RECOMMANDATION")
    print("="*50)
    
    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1. Rechercher des recommandations")
        print("2. Quitter")
        
        choix = input("\nChoisissez une option (1-2) : ")
        
        if choix == '1':
            film_recherche = input("Entrez le nom d'un film (ex: Matrix) : ")
            nb_rec = input("Combien de recommandations souhaitez-vous ? (Défaut: 5) : ")
            
            # Gestion du nombre par défaut si l'entrée est vide
            try:
                n = int(nb_rec) if nb_rec.strip() != "" else 5
            except ValueError:
                print(" Entrée invalide pour le nombre. Utilisation de la valeur par défaut (5).")
                n = 5
            resultats = recommend_movies(film_recherche, n=n)          
            # Affichage des résultats
            if isinstance(resultats, pd.DataFrame):
                print("\n Voici ce que nous vous conseillons :")
                pd.options.display.max_colwidth = 50
                print(resultats[['title', 'similarity', 'rating_count', 'rating_mean']].to_string(index=False))
            else:
                print(f"\n {resultats}")          
        elif choix == '2':
            print("\nMerci d'avoir utilisé le système. À bientôt ! ")
            break
        else:
            print(" Option invalide, veuillez choisir 1 ou 2.")



