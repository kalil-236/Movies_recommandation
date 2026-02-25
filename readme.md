- **Movie Recommendation System**
Un système de recommandation de films intelligent basé sur l'analyse des genres et les préférences utilisateurs, développé dans le cadre d'un apprentissage pratique du Machine Learning.


- **Table des matières**

Aperçu
Fonctionnalités
Architecture du système
Technologies utilisées
Installation
Utilisation
Structure du projet
Méthodologie ML
Dataset
Roadmap


- **Aperçu**
Movie Recommendation System est une application web interactive qui recommande des films basés sur vos préférences. Le système utilise une approche content-based filtering en analysant les similarités entre les genres de films pour proposer des recommandations personnalisées et pertinentes.
Pourquoi ce projet ?
Ce projet a été développé comme deuxième projet pratique d'apprentissage du Machine Learning, après le célèbre projet Titanic de Kaggle. L'objectif était d'acquérir de l'expérience sur :

Les systèmes de recommandation
Le traitement de données réelles (MovieLens)
Le déploiement d'applications ML
Les bonnes pratiques du Machine Learning en production


- **Fonctionnalités**
Phase 1 (Actuelle) : Content-Based Filtering

 Recherche intelligente : Recherche approximative de films (insensible à la casse)
 Recommandations basées sur les genres : Algorithme de similarité cosinus sur les vecteurs de genres
 Filtrage de qualité : Exclusion des films avec moins de 10 évaluations
 Tri multi-critères : Classement par similarité, popularité et note moyenne
 Interface intuitive : Application web moderne avec Streamlit
 Gestion du cold start : Traitement des films sans genres listés
 Informations détaillées : Affichage des genres, notes moyennes et nombre d'évaluations

Phase 2 (En développement) :

 Collaborative Filtering (filtrage collaboratif)
 Système hybride (content-based + collaborative)
 Optimisation des recommandations pour nouveaux utilisateurs

Phase 3 (Planifiée) 

 API REST avec FastAPI
 Conteneurisation avec Docker
 Déploiement cloud
 Interface chatbot conversationnelle


- **Architecture du système**
Workflow général
┌─────────────────┐
│   Dataset CSV   │
│   (MovieLens)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Filtrage     │
│  - Nettoyage    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vectorisation  │
│  (Genres → One- │
│   Hot Encoding) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Similarité    │
│  (Cosine Sim.)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Recommandations │
│   (Top-N par    │
│   similarité)   │
└────────┬────────┘
         │
         ▼
┌───────────────────┐
│Interface Streamlit│
└───────────────────┘
Pipeline de recommandation

Chargement et preprocessing :

Fusion des fichiers movies.csv et ratings.csv
Calcul des statistiques (note moyenne, nombre de votes)
Filtrage des films avec au moins 10 évaluations
Exclusion des films sans genres


Vectorisation des genres :

Utilisation de MultiLabelBinarizer (sklearn)
Transformation des genres pipe-separated en matrice binaire
Exemple : "Action|Sci-Fi|Thriller" → [1, 0, 0, 1, 0, 1, ...]


Calcul de similarité :

Cosine Similarity entre tous les films
Génération d'une matrice de similarité (films × films)


Génération des recommandations :

Extraction du vecteur de similarité du film sélectionné
Tri par similarité décroissante
Départage par nombre d'évaluations puis note moyenne
Retour des top N films

- **Technologies utilisées**
Catégorie    Technologie     Usage
─────────────────────────────────────────────────────────────────────
Langage      Python 3.13.5    Développement principal
─────────────────────────────────────────────────────────────────────
ML/Data      Pandas            Manipulation de données
─────────────────────────────────────────────────────────────────────
             NumPy             Calculs numériques
─────────────────────────────────────────────────────────────────────
             Scikit-learn      MultiLabelBinarizer, Cosine Similarity
─────────────────────────────────────────────────────────────────────
Interface    Streamlit         Application web interactive
─────────────────────────────────────────────────────────────────────
Dev Tools   Jupyter Notebooks  Exploration de données
─────────────────────────────────────────────────────────────────────
            VS Code            Environnement de développement

- **Dépendances principales**
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.28.0

- **Installation**
Prérequis

Python 3.13.5 ou supérieur
pip (gestionnaire de paquets Python)

Étapes

Cloner le repository

bash   git clone https://github.com/kalil-236/Movies_recommandation.git
   cd movie_recommendation

Créer un environnement virtuel (recommandé)

bash   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate

Installer les dépendances

bash   pip install -r requirements.txt

Vérifier la structure des données
Assurez-vous que les fichiers CSV du dataset MovieLens sont présents dans le dossier data/ :

   data/
   ├── movies.csv
   ├── ratings.csv
   ├── tags.csv
   └── links.csv

Lancer l'application
Interface Streamlit (recommandé) :

bash   streamlit run src/app.py
Version console :
bash   python src/main.py

- **Utilisation**
Interface Streamlit

Lancer l'application : streamlit run src/app.py
Une page web s'ouvre automatiquement dans votre navigateur
Rechercher un film : Entrez le nom d'un film que vous avez aimé (ex: "Matrix", "Toy Story")
Sélectionner : Si plusieurs films correspondent, choisissez celui que vous voulez
Obtenir des recommandations : Cliquez sur le bouton pour voir les films similaires
Ajuster le nombre : Utilisez le slider dans la barre latérale (1-20 recommandations)

Interface Console

Lancer : python src/main.py
Suivez les instructions du menu interactif
Entrez le nom du film recherché
Choisissez le nombre de recommandations souhaitées

Exemples de recherche
Essayez avec ces films populaires pour tester le système :

"Matrix" → Films de science-fiction/action
"Toy Story" → Films d'animation/famille
"Inception" → Films complexes/thriller
"Godfather" → Films de crime/drame
"Star Wars" → Space opera/aventure


- **Structure du projet**
MOVIE_RECOMMENDATION/
│
├── data/                          # Données MovieLens
│   ├── links.csv                  # Liens vers IMDB et TMDB
│   ├── movies.csv                 # Informations sur les films
│   ├── ratings.csv                # Évaluations des utilisateurs
│   └── tags.csv                   # Tags appliqués aux films
│
├── models/                        # (Futur) Modèles entraînés sauvegardés
│
├── notebooks/                     # Jupyter notebooks d'exploration
│   ├── 01_exploration.ipynb       # Analyse exploratoire des données
│   └── 02_feature_engineering.ipynb  # Tests de feature engineering
│
├── src/                           # Code source
│   ├── app.py                     # Application Streamlit (interface web)
│   └── main.py                    # Version console (menu interactif)
│
├── .vscode/                       # Configuration VS Code
├── readme.md                      # Ce fichier
└── requirements.txt               # Dépendances Python

- **Méthodologie ML**
Approche Content-Based Filtering
Le système utilise une approche de filtrage basé sur le contenu qui analyse les caractéristiques intrinsèques des films (genres) pour trouver des similarités.
Pourquoi Content-Based pour Phase 1 ?
Avantages :

 Fonctionne sans historique utilisateur (pas besoin de compte)
 Pas de cold start pour les nouveaux films (tant qu'ils ont des genres)
 Transparent et explicable (on voit pourquoi un film est recommandé)
 Pas de problème de sparsité des données

Limitations :

 Recommandations parfois trop "safe" (films très similaires)
 Pas de découvertes surprenantes
 Limité aux features disponibles (genres uniquement ici)

Choix techniques clés
1. Multi-Label Binarization
Les genres sont encodés avec MultiLabelBinarizer car un film peut avoir plusieurs genres simultanément.
Exemple :
Input  : "Matrix" → "Action|Sci-Fi|Thriller"
Output : [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, ...]
           ↑           ↑           ↑
        Action      Sci-Fi     Thriller
2. Cosine Similarity
Mesure l'angle entre deux vecteurs de genres.
Formule :
similarity(A, B) = (A · B) / (||A|| × ||B||)
Interprétation :

1.0 = Films identiques (tous les genres en commun)
0.0 = Aucun genre en commun
0.5-0.8 = Similarité partielle (quelques genres communs)

Pourquoi Cosine et pas Euclidienne ?

Cosine mesure la direction (quels genres), pas la magnitude
Insensible au nombre de genres (1 genre vs 5 genres)
Plus adapté aux vecteurs binaires sparse

3. Tri multi-critères
Les recommandations sont classées selon cet ordre de priorité :

Similarité (priorité absolue)
Nombre d'évaluations (popularité/fiabilité)
Note moyenne (qualité)

Justification :

La similarité garantit la pertinence
Le nombre d'évaluations évite les films obscurs avec peu de données
La note moyenne assure un niveau de qualité minimal

4. Filtrage par qualité
Seuil : Films avec au moins 10 évaluations
Impact :

Catalogue initial : 9,742 films
Après filtrage : ~3,000 films
Élimination de 62% des films (cold start)

Justification :

Les notes moyennes avec <10 votes sont peu fiables
Évite de recommander des films obscurs ou de mauvaise qualité
Compromis entre couverture et qualité

- **Gestion des cas limites** : 
 Problème            Solution implémentée
─────────────────────────────────────────────────────────────────────
Film non trouvé      Recherche approximative (case-insensitive, contains())
─────────────────────────────────────────────────────────────────────
Plusieurs matches    Selectbox pour choisir le bon film
─────────────────────────────────────────────────────────────────────
Films sans genres    Exclusion lors du preprocessing
─────────────────────────────────────────────────────────────────────
Similarité ex-aequo  Départage par popularité puis note
─────────────────────────────────────────────────────────────────────
Film s'auto-recommandant  Exclusion explicite du film de référence
─────────────────────────────────────────────────────────────────────
Dev Tools   Jupyter Notebooks  Exploration de données

- **Dataset :**
MovieLens Small
Source : GroupLens Research
Statistiques :

9,742 films dans le catalogue complet
610 utilisateurs ayant évalué au moins 20 films
100,836 évaluations (ratings de 0.5 à 5.0 par pas de 0.5)
3,683 tags appliqués par les utilisateurs
Période : Mars 1996 - Septembre 2018

Genres disponibles (20)
Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, IMAX, (no genres listed)
Statistiques d'utilisation du système
Après preprocessing (filtrage ≥10 ratings) :

Films dans le système : ~3,000
Genres les plus fréquents : Drama (44.76%), Comedy, Thriller
Note moyenne globale : ~3.5/5.0
Sparsité de la matrice user-item : 98.3%

Citation du dataset
Si vous utilisez ce projet dans un contexte académique :

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872


-**Roadmap**
- **Phase 1 : MVP Content-Based **

 Exploration et analyse des données MovieLens
 Preprocessing et nettoyage des données
 Implémentation de la similarité cosinus sur les genres
 Interface console fonctionnelle
 Interface Streamlit avec UX soignée
 Gestion des cas limites et erreurs

- **Phase 2 : Collaborative Filtering**

 Analyse de la matrice user-item
 Implémentation du collaborative filtering (librairie Surprise)
 Système hybride (content-based + collaborative)
 Stratégie intelligente de cold start
 Optimisation des performances
 A/B testing des différentes approches

-  **Phase 3 : Production & Déploiement**

 API REST avec FastAPI
 Authentification utilisateur
 Base de données (PostgreSQL)
 Containerisation avec Docker
 Déploiement cloud (AWS/GCP/Heroku)
 CI/CD avec GitHub Actions
 Monitoring et logging
 Interface chatbot conversationnelle (optionnel)

- **Améliorations futures envisagées**

Utilisation des tags utilisateur pour enrichir les recommandations
Extraction de features depuis les posters (Computer Vision)
Analyse des synopsis avec NLP
Système de feedback utilisateur (like/dislike)
Recommandations temps réel
Filtres avancés (année, langue, durée)
Diversification des recommandations (serendipity)


- **Apprentissages clés**
Ce projet m'a permis d'acquérir des compétences sur :
Machine Learning
Systèmes de recommandation (content-based, collaborative filtering)
Feature engineering avec données réelles
Métriques de similarité et distance
Gestion du cold start problem
Preprocessing de données textuelles (genres)

Engineering

Architecture d'application ML
Gestion de l'état avec Streamlit
Optimisation des performances (caching)
Gestion des erreurs et cas limites
Structure de code propre et maintenable

- **Méthodologie**

Approche itérative (MVP → amélioration progressive)
Documentation technique
Exploration de données méthodique
Décisions basées sur l'analyse, pas l'intuition


- **Contribution**
Ce projet est un projet personnel d'apprentissage, mais les suggestions et retours sont les bienvenus !
Si vous souhaitez contribuer :

Fork le projet
Créez une branche feature (git checkout -b feature/AmazingFeature)
Commit vos changements (git commit -m 'Add some AmazingFeature')
Push vers la branche (git push origin feature/AmazingFeature)
Ouvrez une Pull Request


Licence
Ce projet utilise le dataset MovieLens sous licence GroupLens Research.
Conditions d'utilisation :

Le dataset peut être utilisé à des fins de recherche et d'éducation
Citation obligatoire dans les publications académiques
Redistribution autorisée sous les mêmes conditions
Usage commercial interdit sans autorisation préalable

Pour plus de détails, consultez : https://grouplens.org/datasets/movielens/



- **Ressources utiles**

Documentation Streamlit
Scikit-learn User Guide
MovieLens Datasets
Recommendation Systems - Coursera


Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !
