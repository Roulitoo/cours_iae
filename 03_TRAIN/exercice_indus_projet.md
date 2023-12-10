# Mise en production d'un projet de machine learning

Merci de vous appuyer sur le projet de Théo disponible [ici](https://github.com/Bouedo/Reseaux_neurones/blob/master/exo2411.ipynb) 

## Structure d'un projet python à industrialiser avec repo github

Un projet que vous souhaitez industrialiser peut suivre la structure de ce repo dans 90% des cas.

```bash
├── main.py
├── data
│   └── mon_csv.csv
├
├── src
│   └── module.py
├── tests
│    └── test_module.py
├
├── README.md
├── requirements.txt
```
où :
- main est votre fichier principal qui va éxecuter vos modules dans src
- data l'endroit où vous stockez vos données
- src, les fichiers sources de votre projet qui seront éxecutés via le main.py
- tests est l'endroit où vous allez stocker vos fichiers qui vont tester votre scripts
- README la documentation essentielle du projet pour savoir comment le lancer
- requirements, les packages nécessaires à votre projet 

## Attendu

Vous devez pouvoir éxecuter vos fichiers sources depuis le fichier main.py.
Tout ce que vous faite devra être intégré dans une fonction python.

**Oubliez** le répertoire tests pour l'instant.

Une fonction correspond à une unique fonctionnalité, charger mes données, entrainer mon modèle, ...

Une fois que vos fichiers sources sont créés, vous pouvez les appeler dans le fichier main via from <nom_de_mon_module> import <ma_fonction>.

Exemple :
```python
#read csv est le nom de fonction disponible dans le fichier load_data
from src.load_data import read_csv
```

## Executer votre main

Pour éxecuter votre fichier principal vous pouvez le faire de la manière suivante

```bash
python3 main.py 
```
Pensez à ajouter des print() dans votre fichier main, cela vous permettra de voir dans le terminal de commande
ce qui se passe et où vous en êtes.

## Bonus, crontab

Je vous laisse chercher sur internet ce que c'est.
Si vous avez des questions, je répondrai en cours.