# Chest X-Ray Images (Pneumonia)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Adrian Ubeda Touati M1 IIA  
Numero etudiant: 24006547


# Chest X-Ray: Prédiction de la présence ou non d'une pneumonie virale ou bactérienne

Dans ce projet, nous explorons différentes approches pour construire un modèle capable de prédire la présence de pneumonie. Pour comparer deux modèles ayant le même objectif, nous utiliserons tout d'abord une validation croisée à 10 plis (10-fold cross-validation) afin d'évaluer les performances des réseaux. Après chaque entraînement, nous extrairons la valeur ROC, qui servira comme métrique principale pour évaluer le fonctionnement du réseau. Enfin, nous appliquerons le test statistique de Wilcoxon pour déterminer, avec une précision de 95 %, si un modèle est significativement meilleur qu’un autre, et si cette différence n’est pas due au hasard.

**Note** : Le jeu de données a été réduit pour la livraison, car il contient plus de 2 Go d'images (lien vers le jeu de données : [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia])

## Contenu du projet

### Dossier `notebooks`

Le dossier `notebooks` contient les étapes suivantes :

1. **Exploration des données** :
   - Exploration initiale des données.
   - Préparation du jeu de données et traitement des images.

2. **Réseaux explorés** :
   - **Premier réseau** : 
     - Objectif : prédire si un patient est sain ou atteint de pneumonie.
     - Architecture basée sur des couches Conv2D et MaxPooling2D pour extraire les caractéristiques des images.
     - Entraînement sur des images brutes redimensionnées à 16x16 pixels.

   **Deuxième réseau** :  
     - Similaire au premier, mais avec deux changements majeurs :  
       - Les images sont redimensionnées à 32x32 pixels.   
       - Validation croisée à 10 plis  
     - Classes: ['NORMAL', 'PNEUMONIA']

   **Troisième réseau** :  
     - Similaire au **Premier réseau**, mais cette fois, il vise à prédire trois classes au lieu de deux.
     - Objectif : prédire si un patient est sain ou atteint de pneumonie.  
     - Architecture basée sur des couches Conv2D et MaxPooling2D pour extraire les caractéristiques des images.  
     - Entraînement sur des images brutes redimensionnées à 16x16 pixels.  
     - Validation croisée à 10 plis  
     - Classes: ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']

   **Quatrième réseau** :  
     - Similaire au troisième, mais avec des images de 32x32 pixels  
    - Les images sont redimensionnées à 32x32 pixels.  
    - Validation croisée à 10 plis  
     - Classes: ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']

  **Cinquième réseau** :  
    - Les images sont redimensionnées à 32x32 pixels.  
    - Ajustements de contraste et de luminosité.  
    - Validation croisée à 10 plis.  
    - Classes : ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS'].

3. **Test de Wilcoxon** :
   - Un notebook dédié au test statistique de Wilcoxon.
   - Analyse des résultats d'entraînement sauvegardés dans un fichier texte pour déterminer s'il existe une différence significative entre les modèles.

### Résultats et observations

Bien que les réseaux utilisant le DataGen et les ajustements d'images (contraste et luminosité) n'améliorent pas significativement les performances du modèle de base, ces techniques pourraient être optimisées. En particulier, une recherche d'hyperparamètres pour le contraste et la luminosité pourrait potentiellement améliorer les résultats.

### Modèles entraînés

Les modèles entraînés sont sauvegardés dans le dossier `models`.
