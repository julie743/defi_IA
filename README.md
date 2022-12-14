Lien vers le Ovearleaf : https://fr.overleaf.com/7884867438ppyjpxpjnfmb 

RECAP DES SCORES OBTENUS AVEC LES MODELES IMPLEMENTES : 
-------------------------------------------------------

1. Résultats : 
- Regression linéaire : 31.64 
- Regression Lasso : 35.86
- Arbre de regression : 27.92
    - Paramètre : max_depth = 36
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Boosting : 26.86
    - Paramètres : "n_estimators": 1000,
                   "max_depth": 20,
                   "min_samples_split": 5,
                   "learning_rate": 0.01,
                   "loss": "squared_error"
     - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Foret aléatoire : 28.21
- Foret aléatoire optimisée : 28.20

Questions : 
- Target encoder : overfit ? ==> https://maxhalford.github.io/blog/target-encoding/
- Tester target encoder codé à la main ou par Python 
- Utiliser Optuna au lieu de GridSearchCV

## To do 
- Gradio 
- Dockerfile 
- Feature engineering 
- Importance features 
- Data analysis 
