# Cours IAE Nantes

## TO DO

16/12/2022

- Trouver une base de données pour le projet noté. Minimun 5000 observations.<br>
  A valider avec moi en séance.
  
- Réaliser le TD N°1 disponible [ici](https://github.com/Roulitoo/cours_iae/tree/master/02_ANN/td) . C'est uniquement la reprise des éléments du cours des réseaux de neurones.


## ⚠️Problème pour importer jupyter notebook depuis Github⚠️

La solution est simple. Il suffit *d'enregistrer le lien sous...* le fichier **RAW** et non pas le IPYNB directement.
Cliquer sur [lien](https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/td/code/01_SVM_TD.ipynb) pour aller récupérer le TD1.

Puis faites clique droit sur **RAW** et *enregistrer le lien sous...*. Comme sur la photo ci-dessous
<br>
<p align='center'>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/import_jupyter_noteook.png" style="width:600px;"/>
</p>

Enregister le fichier dans un répertoire où vous pouvez accéder depuis jupyter notebook! Après ca le tour est joué :)

> Merci de me notifier si vous avez encore des problèmes pour importer les jupyter notebook provenant de Github

## Information cours
Ce reposiroty contient les cours *SVM et Réseaux de neurones* pour les étudiants du Master 2 ECAP, IAE NANTES.  
Le volume horaire de ce cours est de 18h et sera réparti en 5 séances de CM/TP.

Je vous invite à ramener votre ordinateur perso si vous en avez un. Que ce soit pour les CM/TP vous en aurez besoin.

Chaque repos est structuré de la manière suivante 

```
📦cours_iae
┣ 🗒️README.md       
┃
┣ 📁 chapitre_cours
┃  ┣  📁code┣ 🗒️code_cours
┃  ┃         
┃  ┣  📁 td ┣ 🗒️ td.ipynb
┃  ┃         ┣ 🗒️td_correction.ipynb
┃  ┃
┃  ┣  📁cours┣ 🗒️cours.md
┃  ┃          ┣ 🗒️cours.pdf
┃  ┣  📁 img

```

Si vous avez une remarque ou question, n'hésitez pas à me taguer directement avec un [issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). Vous pourrez voir les questions des autres et ma réponse.



L'objectif du cours est de vous fournir une compréhension théorique et pratique des SVM et Réseaux de Neurones (ANN).
Chaque modèle sera présenté en cours avant de passer à la partie pratique en python.
Vous aurez également des travaux à faire à la maison pour poursuivre votre apprentissage.



# Prérequis

## Python

Vous devez déja avoir une première expérience en programmation avec Python. 
Gérer des données avec python avec les packages usuels pour la data science doit être facile.

⚠️ Si vous ramenez votre ordinateur perso vous devez installer Python avant notre premier cours.

### Linux

```bash
$ sudo apt-get update
$ sudo apt-get install python3-virtualenv python3
$ sudo apt-get install gcc g++ python3-dev
```


### Windows, MacOS

Pour Windows le plus simple est de télécharger Anaconda qui est un produit commercial offrant une distribution de pyhton packagée.
Vous la trouverez [ici](https://www.anaconda.com/products/distribution), il suffit de suivre les instructions pour l'installer.

### Google Colab

Pour les personnes qui n'auraient pas de PC portable, vous pouvez créer un compte Doogle Drive.
Depuis Google Drive vous pourrez ouvrir un Google Colab qui n'est rien d'autre qu'un notebook en ligne utilisant des ressources gratuites de google.
Un tuto pour utiliser [Google Colab](https://machinelearningmastery.com/google-colab-for-machine-learning-projects/#:~:text=To%20create%20your%20Google%20Colab,on%20More%20%E2%96%B7%20Google%20Colaboratory.)

> Je ne connais pas les serveurs de l'université de Nantes, Google Colab permet de s'exonerer des contraintes du système d'administration.

### Github

Chaque étudiant devra créer un compte github qui lui permettra de récupérer les cours et td sur mon repository.
Ce sera également l'endroit où vous déposerez votre projet qui sera évalué à la fin du module.

Aucune formation ne sera faite pour github. Votre devrez être en mesure de vous former de votre coté à ce logiciel très utile pour votre vie 
professionnelle.


# Evaluation du cours

**Objectif :**
Réaliser un projet de machine learning sur un dataset de 5000 observations minimuns.
Vous devez obligatoirement réaliser une régression ou classification avec les modèles du cours ( SVM et réseaux de neurones).

**Modalité**
Pour l'évaluation vous devrez me soumettre un projet comportant un fichier .md(markdown) qui contient les commentaires et résultats de votre projet.
Présentation de vos résutlats, discussion du choix des méthodes, vos analyses, ...
Il faudra également me joindre un fichier en .py ou .ipynb avec votre code.

Tous ces éléments devront être déposé dans un repo Github à votre nom/vos noms!

Date de rendu **10 février 23h59h59s**. Si vous dépassez cette date ce sera 0 et pas de correction de ma part.

Elements de notation :

- Créer un repos Github pour 1 ou 2 personnes. (2 max par projet) (0 point)
- Choix d'un dataset que je valide (2 points)
- Me communiquer votre projet avant le 31 Janvier (2 point bonus)
- Un fichier .md explique ce que vous avez fait, ce que vous avez essayé, ce qui a marché, ce qui n'a pas marché, et quels sont vos résultats(10 points)
- Votre fichier avec le code en version .py ou .ipynb réutilisable, lisible avec des commentaires (8 points)
