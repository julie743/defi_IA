Lien vers le Ovearleaf : https://fr.overleaf.com/7884867438ppyjpxpjnfmb 

RECAP DES SCORES OBTENUS AVEC LES MODELES IMPLEMENTES : 
-------------------------------------------------------

1. Résultats : 
- Regression linéaire : RMSE = 31.64 
- Regression Lasso : RMSE = 67.2
    - Paramètre : alpha = 0.01
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Arbre de regression : RMSE = 27.92
    - Paramètre : max_depth = 36
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Boosting : RMSE = 26.86
    - Paramètres : "n_estimators": 1000,
                   "max_depth": 20,
                   "min_samples_split": 5,
                   "learning_rate": 0.01,
                   "loss": "squared_error"
     - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Neural Network : RMSE = 30.21
    - Paramètres : 'alpha': 0.5, 'hidden_layer_sizes': (28,)
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Foret aléatoire : RMSE = 28.21 (pas à jour je crois ?)
- Foret aléatoire optimisée : RMSE = 28.20 (pas à jour je crois ?)


Questions : 
- Target encoder : overfit ? ==> https://maxhalford.github.io/blog/target-encoding/
- Tester target encoder codé à la main ou par Python 
- Utiliser Optuna au lieu de GridSearchCV

## To do 
- Gradio 
- Dockerfile 
- Feature engineering : ajout de la variable cost_life 
- Importance features 
- Data analysis 
