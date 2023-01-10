Lien vers l'Ovearleaf : https://fr.overleaf.com/7884867438ppyjpxpjnfmb 

Utilisation du Github - fichiers requis pour l'évaluation du projet:
------------------------------------------------------
- `git pull https://github.com/julie743/defi_IA.git`
- La première chose à faire est de mettre à jour les paths du projet. Pour ce faire, se placer dans le dossier defi_IA et lancer le code set_path.py par la commande suivante : `python3 set_path.py`. Ce code permet de récupérer l'adresse de la racine du projet où vous l'avez enregistré et met à jour le path dans le fichiers  `set_path.txt` présent dans plusieurs sous dossiers du répertoire **code**. Par la suite, tous les codes font appel à ce ficier pour obtenir le path, il n'y a donc pas d'autre modification à faire pour run l'ensemble des codes du projets.  
- Le modèle final que nous avons sélectionné est le modèle de vote (nomé **average_models** dans le github) qui combine les modèles Catboost et Neural Network. Le fichier qui nous a permis de l'entrainer est dans le fichier `code/models/average_models.py` (-> correspond au fichier `train.py` requis pour l'évaluation). Les poids de ce modèle sont également enregistrés dans le dossier `weights/average_models_adversarial.sav`
- Le code pour lancer le gradio se trouve dans le fichier `code/gradio/gradio.py` (-> correspond au fichier `app.py` requis pour l'évaluation). Run le code `gradio.py` pour lancer le gradio en local. Deux url apparaissent, copier le lien public dans un navigateur. L'interface gradio s'ouvre, vous pouvez sélectionner les options et soumettre la proposition pour obtenir la prédiction du modèle final. 
- Le notebook d'analyse des résultats du modèle final se trouve dans le notebook `code/models/average_models.py`. Le notebbok d'interprétabilité se trouve dans `drive_interp/Interpretability.ypnb`, il a été run sur google collab. (Ces deux notebooks correspondent au fichier `analysis.ipynb` requis pour l'évaluation.)
- Le `Dockerfile` se trouve à la racine du projet, ci-dessous les instructions pour le construire. 

-------------------------------------------------------------------------------------------------
 
 Docker :
 -------------------------------------------------------
 **Etapes pour l'utilisation de docker avec gcloud :**

1. Créer une instance de VM sur Gcloud.

2. Aller sur gcloud au le lien suivant : (https://console.cloud.google.com/compute/instance) 

3. Activer l'instance de VM préalablement créée.

4. Sous le menu déroulant SSH, cliquer sur "Afficher la commande gcloud". Une fenêtre s'ouvre, copier la commande et la coller dans un terminal en local. Cela va ouvrir le terminal gcloud à distance, cela se voit car la ligne de départ du terminal de commande change et devient : "nom@nom_de_mon_instance" 

5. Cloner le répertoire github dans l'instance de VM en tapant la commande suivante : `git clone https://github.com/julie743/defi_IA.git`

6. Se placer dans le répertoire cloné : `cd defi_IA`

7. Construire l'image docker : `sudo docker build -t image1 .`

8. Créer le conteneur : `sudo docker run -it --name container1 -v "$(pwd)":/app image1`
Remarque : On voit que le container est bien ouvert et qu'on est dedans car la ligne de départ du terminal change à nouveau. 

9. Se placer dans le dossier app dans le conteneur : `cd /app`
Remarque : Si on fait un ls, on voit que l'on a bien tous nos fichiers du projet. 

10. Run le fichier set_path.py avec la comande suivante : `python3 set_path.py`
Cela permet de changer le path de référence dans tous les fichiers. 

**Pour run le gradio avec le modèle déjà entrainé, continuer en suivant les indications 11 à 14 :**

11. Se déplacer dans le dossier contenant le gradio avec la commande suivante : `cd code/gradio/`

12. Lancer le code gradio : `python3 gradio_main.py`

13. On voit des lignes apparaitre dans le terminal, et en particulier deux URL, cliquer sur l'URL public pour l'ouvrir. 

14. Une fenêtre s'ouvre dans le navigateur et on peut utiliser gradio : sélectionner les infos requises puis soumettre les informations, la prédiction s'affiche dans la fenêtre en haut à droite.  



**Pour faire l'entrainement du modèle, continuer en suivant les indications 15 à 17:**

15. Se placer dans le fichier du modèle à entrainer : `cd code/models/`

16. Lancer l'entrainement du modèle : `python3 average_models.py`

17. On voit des lignes apparaitre dans le terminal, à la fin de l'entrainement, les scores de RMSE, MAE et R2 sont affichés dans le terminal. Les plots pour l'analyse du modèle sur l'échantillon de validation sont automatiquement enregistrés dans le dossier images/average_models_adversarial. Le fichier .csv de la prédiction du set de test de kaggle est stocké automatiquement dans le dossier predictions/prediction_average_models_adversarial.csv

Fin : Pour sortir du container faire la commande ctrl-C ctrl-D. A la fin de l'utilisation du docker, ne pas oublier de désactiver l'instance de VM sur gcloud : cliquer sur "..." -> "Arrêter". 
 
 
------------------------------------------------------
 
Organisation du Github :
-------------------------------------------------------
**1. Dossier code : tous les codes développés pour le défi IA :**
- **utilities :** codes utilisés par tous les modèles développés :
    - data_loading.py : charger les données requêtées, ajouter les informations des hotels et définir les types de variables
    - data_preparation_for_models.py : mise en forme des données prêtes à être utilisées par les modèles (encodage, renormalisation, split train/validation...)
    - predict_validation_and_test.py : prédiction de l'échantillon de test et de validation (adapté à tous les modèles développés)
    - create_new_features.py : création de nouvelle variable, 
    - download_prediction.py : enregistrer la prédiction des données test au format accepté par kaggle
    - predictions_analysis.py : calculer les scores de prédiction + tracer les graphes pour l'analyse des résultats des modèles
    - remove_duplicates : enlever les doublons dans le jeu d'entrainement 

- **request_data_analysis :** 
    - data_analysis_request.ypnb : analyse de données des données requêtées 

- **models :** tous les modèles développés. 
    - modèles développés : linear regression, LASSO, regression tree, random forest, neural network, boosting, catboost, adaboost, XGboost, average model (modèle de vote), stacking model
    - Pour chaque modèle il y a : 
        - un fichier .py qui contient les fonctions pour créer et le modèle, trouver les paramètres optimaux 
        - un fichier .ipynb qui contient la liste des paramètres optimaux trouvés + l'analyse des résultats du modèle sur l'échantillon de validation
    - stacking.py et naive_stacking : combiner les modèles construits pour obtenir un meilleur modèle
    
- **gradio :** 
    - gradio_main.py : code permettant de lancer le gradio. Il utilise le meilleur modèle développé (le modèle de vote du fichier average_models.py)

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

**5. Dossier weigths :** contient les poids des modèles retenus sur les données de l'adversarial validation
- average_models_adversarial.sav : poids du modèle de vote entre catboost et neural network (le modèle final)
- catboost_adversarial.sav : poids du modèle catboost seul
- neural_network_adversarial.sav : poids du modèle neural network seul
- average_models_adversarial.sav : poids du modèle de stacking entre catboost et neural network 

**6. drive_interp :** sous dossier qui contient les éléments dont on a besoin pour faire l'interprétation du modèle final. Le code a été réalisé sur google collab car les temps de calcul étaient trop important pour le réaliser en local comme le reste du projet. 

--------------------------------------------------------------------------------------------------

Résultats : Meilleurs paramètres et résultats obtenus avec les modèles implémentés: 
---------------------------------------------------------------------------------

1. Résultats sur le jeu de données avec la variable hotel_ID target encodée et la nouvelle variable cost_life
- Regression linéaire : RMSE = 31.64 
- Regression Lasso : RMSE = 67.2
    - Paramètre : alpha = 0.01
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Arbre de regression : RMSE = 27.92
    - Paramètre : max_depth = 36
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Boosting : RMSE = 26.86
    - Paramètres : "n_estimators": 1500,
                   "max_depth": 28,
                   "min_samples_split": 5,
                   "learning_rate": 0.05,
                   "loss": "squared_error"
     - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Neural Network : RMSE = 30.21
    - Paramètres : 'alpha': 0.5, 'hidden_layer_sizes': (28,)
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Foret aléatoire : RMSE = 28.20

2. Résultats sur le jeu de données sélectionné par l'adversarial network : 
- Regression linéaire : RMSE = 28.39 
- Regression Lasso : RMSE = 48.17
    - Paramètre : alpha = 0.01
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Arbre de regression : RMSE = 28.11
    - Paramètre : max_depth = 36
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Boosting : RMSE = 28.6
    - Paramètres  (Optuna, 100 trials) : 'n_estimators' : 1500, 'learning_rate': 0.09090662548192852, 'max_depth': 12
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Neural Network : RMSE = 23.58
    - Paramètres : 'n_estimators' : 500, 'alpha': 0.5, 'hidden_layer_sizes': (18,)
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- XGBoost : RMSE = 27.34
    - Paramètres (Optuna, 5000 trials) : 'n_estimators': 2472, 'learning_rate': 0.2910994859019445, 'max_depth': 7
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Catboost : RMSE = 26.27
    - Paramètres (Optuna, 1000 trials) : 'n_estimators': 2478, 'learning_rate': 0.29014147234242005, 'max_depth': 10
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- LGBM : RMSE = 29.11 
    - Paramètres (Optuna, 5000 trials) : 'n_estimators': 1973, 'learning_rate': 0.2867571924057477, 'num_leaves': 720, 'max_depth': 12, 'min_data_in_leaf': 200, max_bin': 222
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
- Adaboost : RMSE = 56.51 
    - Paramètres (Optuna, 200 trials) :  Adaboost : {'n_estimators': 426, 'learning_rate': 0.0955534691912351}
    - Variables : target encoding de hotel_id, one-hot-encoding des autres variables qualitatives + ajout de la variable cost_life
