# Exercice du jeudi 23 novembre


Pour ceux qui seraient intéréssés par la mise en production de leur modèle.

Je vous propose un projet visant à réaliser une pipeline de mise en production d'un modèle SKLEARN


## STEP1 IMPORT DES DONNEES

Choisir un jeu de donnée pour réaliser un modèle (idéalement le dataset iris qui est simple)

## STEP2 REALISATION DE VOS TRANSFORMATIONS

- Réaliser des transformations sur votre dataset, ajout de colonnes en transformant celles de bases
- Centrer réduire, les variables numériques (elles le sont toutes sauf la target)

## STEP3 CREATION DE LA PIPELINE

Essayer d'intégrer votre préparation des données dans une **pipeline sklearn**  (tuto ici)[https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html]

## STEP 4 CREER UN MODEL

Réaliser un modèle svm pour la classfication ou un autre modèle si vous souhaitez.

## STEP5 ENREGISTRER VOTRE MODELE AVEC JOBLIB

Cela permet de sauvegarder votre modèle dans un fichier que vous pourrez réutiliser par la suite

## STEP6 REALISER UNE PIPELINE POUR AUTOMATISER LE PROCESSUS DE PREDICTION

Importer votre modèle avec joblib + importer les transformations de sklearn pipeline que vous aurez également enregistré avec joblib