Lien vers l'Ovearleaf : https://fr.overleaf.com/7884867438ppyjpxpjnfmb 

Organisation du Github :
-------------------------------------------------------
**1. Dossier code : tous les codes développés pour le défi IA :**
- **utilities :** codes utilisés par tous les modèles développés :
    - data_loading.py : charger les données requêtées, ajouter les informations des hotels et définir les types de variables
    - data_preparation_for_models.py : mise en forme des données prêtes à être utilisées par les modèles (encodage, renormalisation, split train/validation...)
    - predict_validation_and_test.py : prédiction de l'échantillon de test et de validation (adapté à tous les modèles développés)
    - create_new_features.py : création de nouvelle variable, 
    - dowload_prediction.py : enregistrer la prédiction des données test au format accepté par kaggle
    - predictions_analysis.py : calculer les scores de prédiction + tracer les graphes pour l'analyse des résultats des modèles
    - remove_duplicates : enlever les doublons dans le jeu d'entrainement 

- **request_data_analysis :** 
    - data_analysis_request.ypnb : analyse de données des données requêtées 

- **models :** tous les modèles développés. 
    - modèles développés : linear regression, LASSO, regression tree, random forest, neural network, boosting, catboost, adaboost, XGboost
    - Pour chaque modèle il y a : 
        - un fichier .py qui contient les fonctions pour créer et le modèle, trouver les paramètres optimaux 
        - un fichier .ipynb qui contient la liste des paramètres optimaux trouvés + l'analyse des résultats du modèle sur l'échantillon de validation
    - stacking.py et naive_stacking : combiner les modèles construits pour obtenir un meilleur modèle
    
- **gradio :** 
    - gradio.py 
    - gradio.ipynb

- **Test_set_analysis :**
    - Test_set_analysis.ipynb : analyse de données des 17% du jeu de test
    - adversarial_validation.py : adversarial network pour sélectionner les données les plus similaires à l'échantillon de test
    - Adversarial_validation.ipynb

**2. Dossier data :**
    - adversarial_validation_data : données sélectionnées par l'adversarial validation network
    - all_data : toutes les données requêtées concaténées en un dataframe, set de test, features des hotels
    - results_requests : résultats des requêtes par semaines
    - stocks_requetes : liste des requêtes effectuées chaque semaine
    
**3. Dossier images :** contient les graphes d'analyse des résultats de chaque modèle sur l'échantillon de validation (chaque modèle a son propre sous dossier)

**4. Dossier predictions :** contient les fichiers .csv de la prédiction de chaque modèle sur l'échantillon de test (au format accepté par la plateforme kaggle)    

**5. Dossier weigths :** contient les poids des modèles retenus



Meilleurs paramètres et résultats obtenus avec les modèles implémentés: 
-------------------------------------------------------

1. Résultats sur le jeu de données avec la variable hotel_ID target encodée et la nouvelle variable cost_life
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
                   "learning_rate": 0.05,
                   "loss": "squared_error"
     - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Neural Network : RMSE = 30.21
    - Paramètres : 'alpha': 0.5, 'hidden_layer_sizes': (28,)
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Foret aléatoire : RMSE = 28.21 (pas à jour je crois ?)
- Foret aléatoire optimisée : RMSE = 28.20 (pas à jour je crois ?)

2. Résultats sur le jeu de données sélectionné par l'adversarial network : 
- Regression linéaire : RMSE = 28.39 
- Regression Lasso : RMSE = 48.17
    - Paramètre : alpha = 0.01
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Arbre de regression : RMSE = 28.11
    - Paramètre : max_depth = 36
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life

- Neural Network : RMSE = 23.58
    - Paramètres : 'n_estimators' : 500, 'alpha': 0.5, 'hidden_layer_sizes': (18,)
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life

## To do 
- Gradio (OK) : faire le ReadME
- Dockerfile 
- Faire tourner l'interprétabilité 
- ReadME Projet 
