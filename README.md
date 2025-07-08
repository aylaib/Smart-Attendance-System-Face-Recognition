# Système de Gestion de Présences par Reconnaissance Faciale

Ce projet est une application Python complète et intelligente qui utilise la reconnaissance faciale pour automatiser la gestion des présences. Il offre une interface graphique (GUI) intuitive construite avec PyQt5 ainsi qu'une interface en ligne de commande (CLI) pour une utilisation flexible.

Le système est capable d'enregistrer de nouvelles personnes, de les reconnaître en temps réel via une webcam, et de consigner leur présence avec une sécurité additionnelle via un système de PIN.

## 📸 Captures d'écran

![Interface Principale en Action](screenshots/01.png)
_L'interface principale de l'application reconnaissant un visage en temps réel via la webcam._

![Journal d'Activité et des Présences](screenshots/02.png)
_Le panneau de droite affichant l'historique complet des présences enregistrées._

![Enregistrement Manuel de Présence](screenshots/03.png)
_L'interface affichant les visages reconnus dans l'image et demandant le nom pour un enregistrement manuel sécurisé._

## ✨ Fonctionnalités Clés

- **Reconnaissance Faciale en Temps Réel :** Utilise la bibliothèque `dlib` pour une détection et un encodage des visages robustes.
- **Système de Présences Automatisé :** Enregistre automatiquement l'heure de la première détection d'une personne connue dans un fichier `attendance.json`.
- **Enregistrement de Nouveaux Utilisateurs :** Une interface permet d'ajouter de nouvelles personnes au système en capturant plusieurs images depuis la webcam.
- **Sécurité par PIN :** Chaque utilisateur se voit attribuer un PIN unique lors de son enregistrement, permettant une vérification manuelle de la présence.
- **Optimisation par Multiprocessing :** La phase d'encodage des visages connus est accélérée grâce à l'utilisation de `ProcessPoolExecutor` pour exploiter les processeurs multi-cœurs.
- **Évaluation de Performance Avancée :** Le système peut générer un rapport de performance complet, testant différentes configurations de tolérance et de *jitter* pour évaluer la précision, le rappel et le score F1 du modèle.
- **Double Interface :**
  - **GUI (PyQt5) :** Une application de bureau complète et facile à utiliser (`interfaceAPP.py`).
  - **CLI (argparse) :** Une interface en ligne de commande pour une utilisation scriptée (`faciale.py`).

## 🛠️ Technologies et Bibliothèques

- **Python 3**
- **OpenCV** pour la capture et le traitement vidéo.
- **Dlib** pour les algorithmes de reconnaissance faciale.
- **PyQt5** pour l'interface graphique.
- **Scikit-learn, Matplotlib, Seaborn, Pandas** pour l'évaluation des performances et la visualisation.
- **Multiprocessing** pour l'optimisation des performances.

## 🚀 Comment l'Exécuter

### Prérequis

1.  **Installer les dépendances Python :**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Modèles Pré-entraînés de Dlib :**
    Ce projet nécessite les fichiers de modèle de `dlib`. Assurez-vous que le dossier `pretrained_model` est présent à la racine du projet et contient les fichiers suivants :
    - `dlib_face_recognition_resnet_model_v1.dat`
    - `shape_predictor_68_face_landmarks.dat`
    - `shape_predictor_5_face_landmarks.dat`
    *(Si ces fichiers ne sont pas inclus dans le dépôt en raison de leur taille, vous devrez les télécharger séparément).*

### Lancement de l'Application

- **Pour lancer l'interface graphique (GUI) :**
  ```bash
  python interfaceAPP.py

  Pour utiliser l'interface en ligne de commande (CLI) :

python faciale.py --input Dataset

📂 Structure du Projet
interfaceAPP.py: Point d'entrée de l'application graphique (PyQt5).
faciale.py: Contient la logique métier de la reconnaissance faciale et l'interface en ligne de commande.
/Dataset: Dossier contenant les images des personnes connues, organisées par sous-dossiers.
/pretrained_model: Contient les modèles pré-entraînés de dlib.
attendance.json: Fichier où les présences sont enregistrées.
pin_database.json: Base de données des PIN associés à chaque personne.


#### Étape 3 : Création et Publication sur GitHub

1.  **Créez un nouveau dépôt sur GitHub :**
    *   Nom : `Smart-Attendance-System-Face-Recognition`
    *   Description : `Système de gestion de présence utilisant la reconnaissance faciale en temps réel avec Python, Dlib et PyQt5. Inclut l'enregistrement de nouveaux utilisateurs, une sécurité par PIN et une évaluation de performance.`
    *   Laissez-le **Public** et **vide**.

2.  **Envoyez votre projet local sur GitHub :**
    Ouvrez un terminal dans votre dossier `Smart-Attendance-System-Face-Recognition`.

    *   `git init`
    *   `git add .`
    *   `git commit -m "Initial commit: Add Smart Attendance System project"`
    *   `git branch -M main`
    *   `git remote add origin https://github.com/VOTRE_NOM_UTILISATEUR/Smart-Attendance-System-Face-Recognition.git`
    *   `git push -u origin main`

Ce projet est d'un excellent calibre et cette présentation le mettra vraiment en valeur sur votre profil GitHub. Il démontre des compétences très recherchées en vision par ordinateur, en développement logiciel et en analyse de données.