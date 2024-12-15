<center><h1> Chapitre d'introduction : Projet Data et concepts utiles</h1></center>
<p align="center">
<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/Logo_IAE_horizontal.png" alt="Logo IAE.png" style="width:200px;"/>
</p>

#### Table of Contents

[1. Mener un projet data](#1-etapes-dun-projet-data) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.1 Bien définir le problème](#11-bien-d%C3%A9finir-le-probl%C3%A8me) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.2 Trouver les données](#12-trouver-les-donn%C3%A9es) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.3 Explorer les données](#13-explorer-les-donn%C3%A9es) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.4 Préparer le dataset](#14-pr%C3%A9parer-le-dataset) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.5 Explorer vos modèles](#15-explorer-des-mod%C3%A8les-et-d%C3%A9terminer-une-short-list) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.6 Tuner les modèles](#16-tuner-les-mod%C3%A8les) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.7 Présenter votre solution](#17-pr%C3%A9senter-votre-solution) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[1.8 Automatiser,monitorer,maintenir](#18-automatiser-votre-mod%C3%A8le-monitorer-votre-mod%C3%A8le-et-le-maintenir) `<br>`

[2. Liste de concept utile](#-2-liste-de-concept-utile-) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Imbalanced dataset](#21-imbalanced-dataset) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.2 Feature scaling](#22-features-scaling) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.3 Gradient descent](#23-gradient-descent) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.4 Loss or metric function](#24-loss-function-or-metric-function) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.5 Hyperparamètres](#25-hyperpam%C3%A8tre) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.6 Grid search](#26-grid-search) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.7 Learning curve](#27-learning-curve) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.8 Computational complexity](#28-computational-complexity) `<br>`
&nbsp;&nbsp;&nbsp;&nbsp;[2.9 Python environnement virtuel](#29-Python-environnement-virtuel) `<br>`

## 1-Etapes d'un projet Data

## 1.1-Bien définir le problème

Bien que vous ayez un profil technique, la gestion de projets fera partie de votre métier.

Chaque projet doit d'avoir un problème bien cadré sinon vous allez dans le mur!
Si vous n'êtes pas en mesure de définir le problème, vous ne saurez pas à quoi répondre et donc vous ne pourrez rien développer ou alors vous répondrez à côté dans **90%** des cas!

Pour ce faire vous pouvez suivre les recommandations suivantes :

##### Explorez la problématique qui vous est posée.

Quel est le problème?
Pourquoi le problème existe, qu'est-ce que cela engendre?
Quelles sont les solutions pour y répondre aujourd'hui?
Mesure-t-on le problème aujourd'hui?
*S'il n'y a aucune données pour le mesurer, il sera compliqué pour vous de prouver a postériori que votre projet améliore quoi que ce soit.*
Qui est impacté par ce problème?

##### Echangez, parler, identifier les personnes qui pourront répondre à vos questions

Quand on débute on peut avoir envie d'aller directement à la solution mais définir le problème est généralement le fondement du projet.
Sauf si vous êtes déja expert dans le domaine d'application, pensez à interroger les personnes qui gravitent autour du problème et ne vous lancez pas directement dans le traitement des données!

Ce seront vos interpréteurs clés et ils vous suivront le long du projet. **Plus tôt vous intégrerez les utilisateurs finaux de votre projet plus vite vous verrez si vous êtes éloignés ou non de leurs attentes**

> A l'image d'un architecte, plus les plans de votre projet seront précis plus il sera facile de le développer après.
> Un problème clairement spécifié vous permettra de mieux découper votre travail et vous gagnerez du temps par la suite

##### Synthétiser votre travail souvent

A la fin de l'étape de définition du problème vous devriez être capable de :

- Définir le **PROBLEME** que vous réglerez et le **BESOIN** auquel il répond
- Mesurer avec quantitativement/qualitativement le problème
- Expliquer votre solution et ses impactes
- Découper votre solution en plusieurs étapes

`<u>`Synthétiser ces points dans un document et présenter-le `</u>`

##### Commencez petit

Parfois, il vaut mieux prototyper rapidement et présenter votre solution avant de vous lancer dans de grands développements.
Un prototype rapide à développer  qu'on peut tester rapidement restera mieux qu'un projet de 2ans où on développe dans son coin sans avoir de retour(parfois le pire est de travailler longtemps de son coté et se rendre compte que notre travail ne convient pas)

## 1.2-Trouver les données

Maintenant que vous avez un 'plan', vous savez comment répondre théoriquement au problème mais il va falloir se confronter à la réalité.

Vous allez devoir trouver les données dont vous avez besoin pour répondre à votre problématique:

**1** Lister intuitivement les données dont vous avez besoin `<br>`
**2** Trouver un interlocuteur ou un document vous expliquant où sont les données/ comment elles sont générées `<br>`
**3** Créer vous un nouvel espace de travail( **un espace par projet**)`<br>`
**4** Vérifier les **obligations légales relatives à vos données** (RGDP, Techniques, fuites de données, ...)`<br>`
**5** Demander des autorisations (si besoin)`<br>`
**6** Commencer à regarder le type des données dont vous avez besoin (Image, texte, tabulaire, temporelle, géographique,...)`<br>`
**7** Créer un **code automatisable** pour récupérer vos données `<br>`
**8** Structurer votre jeu de données pour que ce soit simple par la suite :

- Format des données
- Nom des colonnes
- Restriction sur votre périmètre

## 1.3-Explorer les données

Dans cette partie vous allez essayer de faire ressortir les *insights* de vos données
💡 **Pensez automatisation, si vous rajoutez des nouvelles données vous ne devez pas recoder l'analyse**

**1** Créer une copie de votre dataset pour travailler dessus (diminuer le taille s'il est trop volumineux)
`<br>`
`<br>`
**2** Pour de l'exploration jupyter notebook est très bien! (on l'oubliera pour le passage en production)`<br>`
**3** Analyser vos données de façon descriptive.

> Un conseil, regarder du coté de [html report pandas](https://github.com/ydataai/pandas-profiling)

**4** Modifier le type de vos données si nécessaire `<br>`
**5** Pour une analyse supervisée, identifier la variable cible (target)`<br>`
**6** Visualiser les données `<br>`
**7** Etudier les corrélations `<br>`
**8** Réfléchir à comment résoudre le problème en tant qu'humain sans coder&nbsp;&nbsp;&nbsp;Quelles informations utiliseriez-vous? Comment le feriez-vous?&nbsp;&nbsp;&nbsp;Après l'avoir fait, essayer de transposer votre approche en code**9** Commencer le *feature engineering* pour créer des nouvelles features `<br>`
**10** Retourner à l'étape 2 s'il manque des données

> Pensez à Documenter vos trouvailles, documenter, documenter, documenter!

## 1.4-Préparer le dataset

💡Travailler sur une copie du dataset
💡Ecrivez des functions et pas du code non réutilisable

### Plan pour préparer son dataset

- 1) **Data cleaning** (outliers, NA value, ...)
- 2) **Feature selection**(si besoin)

  - Etudes des corrélations
  - Variables d'importances
  - Régression pénalisée
  - Stats descriptives
- 3) **Feature engineering** adatpé à vos besoins

  - Discrétiser vos données continues
  - Recoder variables catégorielles
  - Ajouter des transformations de features
  - Agréger des features
- 4) **Feature scaling**

  - Standardiser ou normaliser vos features

> Ce [bouquin](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) est pas mal si ca vous intéresse d'en savoir plus `<br>`
> Un site pour la [Feature selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/])

## 1.5-Explorer des modèles et déterminer une short-list

**1)** Entrainer des modèles avec les hyperparamètres par défaut
&nbsp;&nbsp;&nbsp; *Des modèles avec des paradigmes différents (regressions, arbres, svm, neural net, xgboost,...*

**2** Mesurer les performances de chaque modèle
&nbsp;&nbsp;&nbsp;*Utiliser une cross-validation avec n-fold*

**3** Analyser les variables d'importances pour chaque modèle

**4** Analyser les erreurs du modèle

**5** Réaliser une liste des features pertinents

**6** Essayer de changer d'améliorer rapidement vos précédents modèles

**7** Garder une liste des 3 meilleurs modèles

## 1.6-Tuner les modèles

🏁 Vous allez maintenant utiliser l'ENSEMBLE de vos données pour obtenir le meilleur modèle possible

**1** Tunez vos modèles en utilisant une cross validation

- Par expérience, je vous conseille de traiter votre feature engineering comme un hyperparamètre.
  Surtout si vous n'êtes pas sûr de votre stratégie (ie, imputation NA, réunification Data, ...)
- Random grid, search, bayesian grid search

**2** Si vos modèles offres des performances faibles, testez les [modèles ensemblistes](https://scikit-learn.org/stable/modules/ensemble.html)

**3** Quand votre modèle est suffisament performant sur le **training set**, mesurer sa performance avec le **test set**

## 1.7-Présenter votre solution

**1** Documentez votre projet
&nbsp;&nbsp;&nbsp;*Pensez bien à expliquer les choix que vous avez faits*

**2** Créer une présentation sympa (pas de word SVP)
&nbsp;&nbsp;&nbsp;*Mettez en avant les informations importantes*

**3** Expliquer concrètement comment votre projet répond au besoin business (besoin de départ)

**4** Pensez à comment vous allez vendre votre projet!
&nbsp;&nbsp;&nbsp;*Si vous n'êtes pas dans une entreprise tech, il sera parfois compliqué de prouver que votre modèle est utile.*`<br>`
&nbsp;&nbsp;&nbsp;*Faites de la com, soyez imaginatif*

## 1.8-Automatiser votre modèle, monitorer votre modèle et le maintenir

**1** Préparer votre code pour passer en production

**2** Préparer un monitoring de votre code

- Suivre la performance de votre modèle (KPI)
- Suivre que votre modèle s'excute bien
- Vérifier que le modèle ne se dégrade pas
- Mesurer qu'il n'y a pas de dérive sur vos données

**3** Faites régulièrement des points avec le business pour prouver que votre solution améliore la situation

<center><h1> 2-Liste de concept utile </h1></center>

## 2.1-Imbalanced dataset

Le cas le plus commun de données déséquilibrées est une classification binaire.

Prenons l'exemple d'une fraude à la carte bancaire.
Nous avons un data set contenant 1 million d'opérations bancaires. La fraude étant un élément rare (heuresement) notre data set ne contient que 10 000 fraudes pour 990 000 non fraudes.

Si nous entrainons un modèle de machine learning pour une classification binaire sur ce projet, il sera incapable d'apprendre ce qu'est une fraude car nous ne lui présenterons pas suffisamment d'exemple pour qu'il arrive à définir une fraude.

Imaginons tout de même que nous entrainions une rehression logistique sur ce dataset.**Le modèle donne une accuracy de 89%**

> Est-ce une bonne nouvelle, le modèle est-il pertinent?

On aurait tendance à dire oui car 89% de bonne prédiction semble être une valeur élevée mais 89% est plus faible qu'une prédiction naïve...

Un algo qui dirait systématiquement qu'une opération n'est pas une fraude aurait raison à 99% du temps 99000/1000000.

Si vous êtes confronté à ce genre de problème, vous pouvez utiliser les méthodes suivantes :

- **Upsampling** : Augmenter l'événement rare avec un tirage aléatoire avec replacement
- **Downsampling** : Diminuer l'événement non rare en retirant des cas
- **Oversampling** : Algorithme ROSE ou SMOTE créant artificiellement de nouveaux cas rares

> Lien pour smote et rose https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

## 2.2-Features scaling

Le scaling feature (mettre vos données à la même échelle) permet d'exprimer différentes features avec différentes grandeurs numériques dans une même unités.

Il exite 2 grandes familles pour le feature scaling :

- La normalisation
- La standardisation

Les 2 permettent d'exprimer les colonnes numériques dans une même unités, améliorer le temps de calcul des modèles et pour certain modèle donner de meilleures performances.

### Normalisation

La normalisation est le fait de transformation vos features dans **une échelle [0,1]**. On l'appelle parfois *min-max scaling*.
Sa formule est la suivante

$X_{norm} = \frac{X-X_{min}}{X_{max}-X_{min}}$

```python
from sklearn.preprocessing import MinMaxScaler
#Exmple
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler() 
print(scaler.fit_transform(data))
```

### Standardisation

La standardisation est une technique qui permet quant à elle de transformer nos colonnes en variable avec une **moyenne de 0 et un écart type de 1.**
Les colonnes transformées auront donc les mêmes paramètres de distribution.
La standardisation présente des avantages quand il existe des outliers, comme on utilise pas la valeur Min et Max, la technique y est moins sensible!

$z = \frac{X-\mu}{\sigma}$

```python
from sklearn.preprocessing import StandardScaler
data = [[4, 8], [-5, 25], [4, 1], [9, 2.5]]
scaler = StandardScaler()
print(scaler.fit_transform(data))

```

> Article intéressant : https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf

## 2.3-Gradient Descent

Quand vous calculez les paramètres de votre modèle vous avez 2 possibilités :

- Utiliser une résolution mathématique pour obtenir la solution optimale (exemple, résolution MCO reg linéaire)
- Utiliser une résolution d'optimisation successive appellée **Descent de Gradient** qui va chercher itérativement les paramètres qui minisent la fonction de coût du modèle

> Plus d'informations ici https://developers.google.com/machine-learning/crash-course/reducing-loss/an-iterative-approach

Concrètement vous commencez avec un paramètr $\theta$ donné et vous allez le faire varier itérativement en fonction de la valeur de sa dérivée.
On peut l'observer graphiquement sur le graphique N°1

`<u>`Graphique N°1 :Descente de gradient `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/descente_gradient_1.png" alt="fig_1_descente_gradient.png" style="width:600px;"/>

Chaque point rouge représente une itération de descente de gradient et converge vers le minimum global de la fonction de perte.
Nous obtenons en ce point pour un paramètre $\theta$ dont la valeur minimise notre fonction de perte.

`<u>`Graphique N°2 :Descente de gradient, learning rate trop faible `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/descente_gradient_2.png" alt="fig_2_descente_gradient.png" style="width:600px;"/>

Il est important que la taille du 'saut' de mise à jour de la valeur de votre paramètre $\theta$ ne soit pas trop faible.
On appellera le paramètre qui contrôle le 'saut' **LEARNING RATE**.
Si celui-ci est trop faible vos 'sauts' seront petits, il faudra beaucoup d'itérations avant de trouver le paramètre optimal.

`<u>`Graphique N°3 :Descente de gradient, learning rate trop haut `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/descente_gradient_3.png" alt="fig_3_descente_gradient.png" style="width:600px;"/>

A l'inverse si le *LEARNING RATE* est trop élevé vous pourriez ne jamais trouver l'optimal de votre fonction.
Le calcul divergera et ne trouvera jamais de minimum local.

Un peu de math pour comprendre la descente de gradient😀.
C'est un concept fondamental pour les algorithmes de machine learning!!

Exemple de descente de gradient avec **fonction de coût MSE** pour un modèle linéaire:

On définit une fonction linéaire avec un vecteur de paramètre $\theta$
$\widehat{y} = \theta_0 + \theta_1x1+...+\theta_nxn$

où  :

- $\theta_0 : Biais\space du\space modele $
- $\theta_n : Paramètre \space du \space modèle$
- $\widehat{y} : Valeur\space prédite$
- $n : Nombre \space de \space features$

Au format vectoriel nous avons l'équation suivante :

$\widehat{y} = h_\theta(x) = \theta.X$

On définit la fonction de perte de ce modèle comme :

$MSE(X, h_\theta) = \frac{1}{N}\sum_{i=1}^N (y_i-\widehat{y_i})²$

Pour implémenter la descente de gradient, vous devez calculer le gradient de la fonction de coût MSE en fonction de ses paramètres $\theta$
On doit donc calculer toutes les dérivées partielles de la fonction MSE

$\frac{\partial}{\partial \theta_j} MSE(\theta) = \frac{2}{N}\sum_{i=1}^N (\theta^Tx^{(i)}-y^{(i)})x^{(i)}_j$

ou au format vectoriel

$$
\nabla_\theta MSE(\theta) = \begin{pmatrix}  \frac{\partial}{\partial \theta_0} \\ \frac{\partial}{\partial \theta_1} \\ . \\frac{\partial}{\partial \theta_n}  \end{pmatrix} =\frac{2}{N}X^T (X\theta-y)
$$

Une fois que vous avez le vecteur de descente de gradient, vous devez simplement mettre à jour vos paramètres $\theta$ jusqu'à atteindre le minimum de votre fonction.

$\theta^{(next)} = \theta - \eta\nabla_\theta MSE(\theta)$

#### Exemple descente de gradient

Un exemple en dimension 1 pour mieux comprendre 😀

Nous avons une fonction  $f(x) = 3x^2 -2x +5$ et nous souhaitons minimiser cette fonction

`<u>`Graphique N°4 :Exemple descente de gradient `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/exemple_grad_1D_4.png" alt="fonction_exemple_4.png" style="width:500px;"/>

**Etape 1 : On calcule son vecteur gradient **

En dimension le vecteur est de taille 1, donc on calcule uniquement une dérivée

$f'(x) = 6x -2$

**Etape 2 : On initialise une valeur de $x$ par défaut et une valeur pour le learning rate**

On pose $x_0 = 5$ et $\eta = 0.05$

La formule pour les étapes de descente de gradient en D1 est donc :
$x_{n+1} = x_n -\eta*f'(x_n)$

**Etape 3 : Itération sucessive descente de gradient**

`<u>`Graphique N°5 :Exemple descente de gradient `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/descente_grad_exemple_5.png" alt="fonction_exemple_descente_grad_5.png" style="width:500px;"/>

Successivement la valeur de $\theta$ se rapproche de la valeur de $x=\frac{1}{3}$ qui minise la fonction.
Quand vous utiliserez l'hyperparamètre **learning rate** pour un algo de machine learning c'est exactement ca qui se passera en back.

> Vous savez maintenant ce qu'est la descente de gradient, bravo !

## 2.4-LOSS function or Metric function?

Les 2 termes sont souvent confondus dans le domaine du machine learning mais il représente pourtant 2 concepts bien différents.

#### Loss function

La *loss function* ou *cost function* est utilisée pour entrainer notre modèle de ML et c'est la fonction que nous allons chercher à optimiser (minimiser ou maximiser) les paramètres du modèle.

Globalement elle donne l'écart entre la qualité de notre prédiction et la valeur de référence.

Exemple :

- Logistic sigmoid
- Mean squared error
- Cross-Entropy
- Hinge loss
- etc

#### Metric function

La *Metric function* est quant à lui un critère a postériori qui permet d'évaluer la qualité/performance du modèle. C'est un quantifieur permettant au créateur du modèle d'évaluer si son modèle est bon ou mauvais.

Exemple :

- Accuracy
- F1 Score
- Recall
- etc

#### Spoiler

Certaines $Loss function$ sont aussi des $Metric function$, c'est quasi tout le temps le cas pour les modèles de régression!

> Toutes les loss function de scikit : https://scikit-learn.org/stable/modules/model_evaluation.html

## 2.5-Hyperpamètre

Il ne faut pas confondre les paramètres d'un modèle qui dépendent directement des données et sont calculés analytiquement avec les hyperparamètres.

Tous les modèles de machine learning n'en possèdent pas. Par exemple la régression linéaire ne possède aucun hyperparamètre, l'ensemble de ses paramètres est calculé à partir des données.

En revanche, les modèles de machine learning complexes en possèdent énormement. Ils permettent de contrôler l'apprentissage du modèle et impactent donc directement les paramètres du modèle.
Leur valeur n'est pas connue à l'avance et la seule façon de trouver la combinaison optimale est de faire varier leur valeur tout en observant l'impact sur la fonction de perte.

Exemple :

- Le learning rate
- Regularization parameter (ridge, lasso,...)
- Max depth, Max features ( random forest, ...)

> Pour approndir
> https://towardsdatascience.com/parameters-and-hyperparameters-aa609601a9ac

## 2.6-Grid search

Le moyen le plus simple de trouver les valeurs de vos hyperparamètres qui maximise ou minimise ou votre *loss function* est de réaliser un *grid search*.

C’est une méthode d’optimisation (hyperparameter optimization) qui va nous permettre de tester une série de paramètres et de comparer les performances pour en déduire le meilleur paramétrage.
On définit une plage de valeur possible pour nos hyperparamètres et toutes les combinaisons seront testées pour voir lesquelles donnent le meilleur modèle.

Il existe 3 types de grid search :

- **Grid search** : On définit manuellement une grille de combinaison des hyperparamètres. Plus vous aurez de l'expérience plus il sera facile de définir les hyperparamètres et donc réduire l'espace de la grille.
- **Random Grid search** : On définit une grille dans lequel les hyperparamètres prennent leur valeur dans un espace que nous lui fournissons. Dans l'ensemble de cet espace, il va chercher les hyperparamètres qui donne le meilleu résultat.

On peut voir graphiquement le résultat des 2 approches.

`<u>`Graphique N°6 :Visualisation grid search avec 2 hyperparamètres `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/grid_search_6.png" alt="gris_search_6.png" style="width:1000px;"/>

Le random search donne généralement de meilleure performance mais il est aussi beaucoup plus couteux en temps de calcul... A vous d'arbitrer.

- **Bayesian Grid search** :  Cette méthode différe des autres car elle va sélectionner les hyperparamètres à chaque entrainement de modèle conditionnelement aux résultats du précédent. Théoriquement la combinaison de meilleur paramètre sera trouvé plus vite que pour les 2 précédents et elle retira automatiquement les espaces où la combinaison d'hyperparamètre est mauvaise. Comme pour les 2 autres, il faut fournir au départ l'espace où tester les hyperparamètres.

Pour implémenter ces méthodes en python, vous pouvez utiliser les codes suivants :

```python
#Import function
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Define grid
param_grid = {
    "max_samples": [0.2, 0.3, 0.4, 0.4, 0.1],
    "max_features": [1, 2],
    "max_depth": [ 4, 20,] 
            }
#Grid
reg_grid = GridSearchCV(RandomForestClassifier(),
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=4, 
                        scoring='accuracy'
                       )

#Fit model
model_grid = reg_grid.fit(X, y)
#get best estimator
model_grid.best_estimator_

```

#### Random search

```python
#Import function
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint

#Define grid
param_grid = {
    "max_samples": uniform(1e-2, 0.5),
    "max_features": randint(1,2),
    "max_depth": randint(4, 400) 
}

# Random
reg_rand = RandomizedSearchCV(RandomForestClassifier(),
                         param_distributions=param_grid,
                         cv=5,
                         n_jobs=4,
                         scoring='accuracy',
                         random_state=42)
#Fit model
model_rand = reg_rand.fit(X, y)

#Print best estimator
print(model_rand.best_estimator_)

```

#### Bayesian search

```python
#! pip install scikit-optimize
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer
#Real : Nombre réel
#Categorial : data catégorielle, exemple 'bleu', 'rouge'
#Integer : ...

#Define grid
param_grid = {
    "max_samples": Real(1e-2, 0.5),
    "max_features": Integer(1,2),
    "max_depth": Integer(4, 400) 
}
#grid
reg_bay = BayesSearchCV(estimator=RandomForestClassifier(),
                    search_spaces=param_grid,
                    cv=5,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state=42)
#Fit the data
model_bay = reg_bay.fit(X, y)
#Meilleur estimateur
print(model_bay.best_estimator_)

```

> Article intéressant sur le grid search et random search
> https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
> Comparaison 3 types de grid search, article
> https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-how-and-why-655b0ee0b399
> Doc implémentation bayesian grid search
> https://scikit-optimize.github.io/stable/index.html

## 2.7-Learning curve

En modélisation on dit souvent que "plus on a de data plus le modèle sera précis". Cette affirmation est vraie, augmenter le nombre de data améliore généralement les performance des modèles.`<br>`
`<br>`
Cependant, il existe une quantité de data à partir duquel le modèle arrête d'apprendre. Autrement dit rajouter des données ne sert à rien à part augmenter le temps de calcul!
Cela peut venir du fait qu'il existe un parttern simple dans vos données et le modèle apprend très vite ou malheuresement que vos données ne permettent de pas d'expliquer le phénomène étudié.

Une manière de visualiser cette relation quantité de data et performance du modèle est d'utiliser des learning curve.

On itère plusieurs modélisations du même modèle sans faire varier les hyperparamètres mais on entraine systématiquement avec un peu plus de data pour voir l'impact sur la qualité du modèle.

Généralement on représente cette technique avec un graphique qui ressemble à celui ci-dessous :

`<u>`Graphique N°7 :Visualisation learning curve `</u>`

<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/learning_curve_7.png" alt="learning_curve_7.png" style="width:600px;"/>

Pour implémenter ce type de graphique nous utilisons encore une fois [sklearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

```python
#Ploting learnin curve
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

#Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
#Model
lr = LogisticRegression(max_iter = 1000 , random_state= 42)

# Use learning curve to get training and test scores along with train sizes
#Learning curve function with train_sizes = découpage du dataset en 10 de 10% à 100%
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, 
                                                        X=X_train, 
                                                        y=y_train,
                                                        cv=10, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                       )

#
# Trop de fluctuation dans le modèle, on calcule la moyenne des métriques
#
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
#
# Plot the learning curve
#
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()

```

## 2.8-Computational complexity

En machine learning, le $computational complexity$ ou complexité de l'algorithme est le montant de ressources nécessaires pour utiliser un modèle.
On distingue le temps d'entrainement d'un modèle et le temps de prédiction d'un modèle déja entrainé.

A titre d'exemple.

La régression linéaire implémentée avec sklearn possède une complexité de $O(n_{samples} n^2_{features})$.
Si on double le nombre de lignes et de colonnes du dataset, on augmente alors de $2.2^2 = 8$  le temps de calcul. Un temps de calcul alors 8 fois plus long.

En revanche la prédiction ne dépend que du nombre de colonnes $O(n_{features})$

> Tableau comparaison model complexity :https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/

## 2.9-Les data pipelines

Une data pipeline ou "chaine de traitement data" est un processus visant à ingérer l'ensemble des données brutes afin de les déplacer dans un système permettant de les traiter et les transformer afin de les adapter à un enjeu ou des enjeux finaux.

Ce terme provient du domaine de l'engineering où les Data Ingénieur sont souvent amenés à traiter d'énormes volumes de données afin de les rendre exploitable par des Data Scientist, Data Analyst, Business Analyst, Chargé d'études, ...

Ce traitement suit plusieurs phases :

- L'identifiant des data sources : database SQL ou NOSQL, API, Fichiers, Server, ...)
- Business logic : Transformation des données brutes pour ressembler le plus possible au contexte où elles sont créées
- Data destination /target :  Lieu où nous allons les stocker (Data warehouse, Data Lake)
- Orchestration :  Quand  allons-nous exécuter toutes ces phases? Que faire s'il ya une erreur, ...

> Donner un exemple avec un ETL

### La transposition des data pipeline au machine learning

Créer un algorithme de machine learning demande la réalisation de nombreuses tâches. Ces tâches se décomposent souvent séquentiellement et vont de l'extraction de données pour constituer son dataset, le nettoyage de ses données, l'ajout de données avec votre expertise, l'entrainement du modèle, son déploiement et le suivi de ses performances.

Ces étapes individuelles jouent un rôle essentiel dans la mise en production industrielle des modèles de machine learning. Aujourd'hui, il est impensable d'exécuter manuellement ces tâches les unes après les autres.

Ces pourquoi les Data Scientist ont décidé d'appliquer les meilleures pratiques des Data Pipelines à nos problématiques de création de modèles.

![chaine ml](cours_iae\00_intro\img\machine_learning_pipeline.png)

Cette chaine de traitement s'associe à de bonne pratique :

![mlops_chaine](cours_iae\00_intro\img\mlops.png)

## 2.9-Python environnement virtuel

Python ne sait pas versionner les packages. Quand vous réaliser un *pip install*, la version que vous téléchargez écrase la précédente.

Cela peut poser de grave problème si vous mettez à jour vos packages, potentiellement vos anciens codes seront en erreur
car la nouvelle version du package à supprimer une fonction ou l'a tout simplement modifiée...

Prenons un exemple, je développe pour le projet ann & svm en python 3.7 et pandas 1.2 la fonction suivante :

```python
import pandas as pd

def calcul_moyenne_par_groupe(dataframe, colonne_groupe, colonne_valeur):
    resultats = dataframe.groupby(colonne_groupe).agg({colonne_valeur: 'mean'}).reset_index()
    return resultats

```

Maintenant pour un projet spécifique j'ai besoin de pandas 1.0 que je télécharger avec *pip install*.`<br>`
Je relance mon code python avec la nouvelle version de pandas et voilà ce que j'obtiens:

```python
Cette version de Pandas ne prend pas en charge la méthode 'agg' de cette manière. Mettez à jour vers une version plus récente de Pandas pour utiliser cette fonction.

```

**Heuresement** il existe une solution pour ce prémunir de ce genre de problème.`<br>`
Les environnements virtuels!!

![Alt text](https://raw.githubusercontent.com/Roulitoo/cours_iae/master/00_intro/img/pyenv.png)

Comment réaliser cette image en code

```powershell
#Comment créer un venv

##Windows powershell

python -m venv mon_env #On peut spécifier la version de python python3.7 -mv venv mon_env

.\mon_env\Scripts\Activate.ps1

#Désactiver 
deactivate

##Linux

python3 -m venv mon_env
source mon_env/bin/activate

#Désactiver
deactivate

```

Vous obtenez un environnement virtuel avec une version de python vide de tout package.

### Alternative avec Anaconda

Il existe une alternative plus "clique bouton" disponible avec Anaconda.
Vous pouvez suivre le tutoriel [ici](https://docs.anaconda.com/free/navigator/tutorials/manage-environments/)

### Installation de package avec version spécifique

Nous allons utiliser un fichier texte communement appelé requirements.txt qui contient le nom du package et sa version

![Alt text](https://raw.githubusercontent.com/Roulitoo/cours_iae/master/00_intro/img/requirements.png)

Après pour installer ces versions il suffit de réaliser la commande suivante :

```powershell
#Il faut avoir activé son environnement virtuel

pip install -r requirements.txt

```

> Nous venons d'illuster la célèbre phrase des data scientist. "Je ne comprends pas, ca marche sur mon pc"

## 2.10-Vscode

Visual Studio Code, souvent abrégé en **VSCode**, est un éditeur de code source gratuit et open-source développé par Microsoft. Il s'agit d'un outil de développement très populaire auprès des programmeurs du monde entier en raison de sa simplicité, de sa flexibilité et de sa variété d'extensions.

Avec VSCode, vous pouvez développer des applications dans divers langages de programmation. Il offre des fonctionnalités essentielles telles que la coloration syntaxique, la complétion automatique, le débogage intégré, la gestion de versions, et bien plus encore.

Ce logiciel a été conçu pour être léger, rapide et hautement personnalisable, ce qui en fait un choix idéal pour les développeurs de tous niveaux. Que vous travailliez sur des projets Web, des applications mobiles, des scripts ou des projets de données, Visual Studio Code est un outil polyvalent qui vous permet de coder de manière efficace et stylée.

C'est un excellent choix pour les étudiants qui découvrent le monde de la programmation, car il facilite l'apprentissage tout en offrant des fonctionnalités puissantes pour le développement de logiciels.

### Extensions à téléchager

A gauche cliquer sur le menu extensions(4 petits carrés) et télécharger les extensions suivantes :

- **Python**
- **Pylance**
- **Jupyter**
- **Markdown Extended**

## 2.11-Github

GitHub est une plateforme de développement collaboratif qui vous permet de gérer,
suivre et partager des projets de programmation, y compris ceux liés à l'économétrie.

Que vous soyez novice en programmation ou un développeur chevronné, GitHub peut être un outil précieux pour faciliter le travail d'équipe, le suivi des versions et la collaboration sur des projets.

### Qu'est-ce que GitHub ?

GitHub est un système de contrôle de version (VCS) qui permet de stocker et de suivre l'évolution des fichiers source de vos projets.

Il vous offre la possibilité de :

- Garder une trace des modifications apportées à vos fichiers au fil du temps.
- Travailler en collaboration avec d'autres personnes sur des projets.
- Gérer des branches pour travailler sur différentes fonctionnalités ou correctifs en parallèle.
- Faciliter la résolution des conflits lors de la fusion de modifications.
- Faciliter le partage de projets avec le reste du monde.

### Comment commencer avec GitHub ?

Pour commencer avec GitHub, suivez ces étapes simples :

1. Créez un compte GitHub : Rendez-vous sur GitHub.com et créez un compte gratuit.
2. Installez Git : Git est un logiciel de contrôle de version que vous devrez installer sur votre ordinateur.
   Vous pouvez le télécharger à [partir du site officiel de Git](https://git-scm.com/download/win).
