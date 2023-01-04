**Etapes pour l'utilisation de docker avec gcloud :**

1. Créer une instance de VM sur Gcloud 

2. Aller sur gcloud avec le lien suivant : (https://console.cloud.google.com/compute/instance) 

3. Activer l'instance de VM préalablement créée.

4. Sous le menu déroulant SSH, cliquer sur "Afficher la commande gcloud". Une fenêtre s'ouvre, copier la commande et la coller dans un terminal en local. Cela va ouvrir le terminal gcloud à distance, cela se voit car la ligne de départ du terminal de commande change et devient : "nom@nom_de_mon_instance" 

5. Cloner le répertoire github dans l'instance de VM en tapant la commande suivante : git clone https://github.com/julie743/defi_IA.git

6. Se placer dans le répertoire cloné : cd defi_IA

7. Construire l'image docker : sudo docker build -t image1 .

8. Créer le conteneur : sudo docker run -it --name container1 -v "$(pwd)":/app image1. Remarque : On voit que le container est bien ouvert et qu'on est dedans car la ligne de départ du terminal change à nouveau. 

9. Se placer dans le dossier app dans le conteneur : cd /app. Si on fait un ls, on voit que l'on a bien tous nos fichiers du projet. 

10. Run le fichier set_path.py avec la comande suivante : python3 set_path.py. Cela permet de changer le path de référence dans tous les fichiers. 


Pour run le gradio avec le modèle déjà entrainé, continuer en suivant les indications 11 à 14 :

11. Se déplacer dans le dossier contenant le gradio avec la commande suivante : cd code/gradio/

12. Lancer le code gradio : python3 gradio_main.py

13. On voit des lignes apparaitre dans le terminal, et en particulier deux URL, cliquer sur l'URL public pour l'ouvrir. 

14. Une fenêtre s'ouvre dans le navigateur et on peut utiliser gradio : sélectionner les infos requises puis soumettre les informations, la prédiction s'affiche dans la fenêtre en haut à droite.  



Pour faire l'entrainement du modèle, continuer en suivant les indications 15 à :

15. Se placer dans le fichier du modèle à entrainer : cd code/models/

16. Lancer l'entrainement du modèle : python3 average_models.py

17. On voit des lignes apparaitre dans le terminal, à la fin de l'entrainement, les scores de RMSE, MAE et R2 sont affichés dans le terminal. Les plots pour l'analyse du modèle sur l'échantillon de validation sont automatiquement enregistrés dans le dossier images/average_models_adversarial. Le fichier .csv de la prédiction du set de test de kaggle est stocké automatiquement dans le dossier predictions/prediction_average_models_adversarial.csv

---- 
Fin : Pour sortir du container faire la commande ctrl-C ctrl-D. A la fin de l'utilisation du docker, ne pas oublier de désactiver l'instance de VM sur gcloud : cliquer sur "..." -> "Arrêter". 

