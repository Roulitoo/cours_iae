<center><h1> SVM sous Python</h1></center>
<p align="center">
<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/Logo_IAE_horizontal.png" alt="Logo IAE.png" style="width:200px;"/>
</p>

#### Table of Contents
[1. Préambule](#1-pr%C3%A9ambule)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1.1 Iris flower ](#11-iris-flower)<br>

[2. Support Vecteur Machine](#2-support-vecteur-machine)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Presentation intuitive d'un SVM](#21-presentation-intuitive-dun-svm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.2 Calcul de la marge ](#22-calcul-de-la-marge)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.3 Maximisation de la marge](#23-maximisation-de-la-marge)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.4 SVM LINEAIRE, HARDS MARGING VS SOFT MARGIN CLASSIFCATION](#24-svm-lineaire-hards-marging-vs-soft-margin-classifcation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.5 SVM non liénaire](#25-svm-non-li%C3%A9naire)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.5.1 Polynomial Kernel](#251-polynomial-kernel)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.5.2 Similarity Features](#252-similarity-features)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.6 Classification Multiclass pour les SVM](#26-classification-multiclass-pour-les-svm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.6.1 Application classication multiclass, dataset MNIST ](#261-application-classication-multiclass-dataset-mnist)<br>

[3. SVM pour la régression](#3-svm-pour-la-r%C3%A9gression)<br>
[4. Récapitulatif](#4-r%C3%A9capitulatif)<br>


## 1-Préambule

Connaissez-vous le fameux jeu de données IRIS produit par Ronald Fisher en 1936??  
Si non, nous allons y remédier!  

Nous pouvons accéder à ce dataset depuis sklearn avec le code suivant :

```python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data = load_iris()

# Array to Pandas DataFrame
iris = pd.DataFrame(np.c_[data['data'],
                          data['target']
                         ],
                    columns = data.feature_names + ['species']
                   )

iris['species'] = iris['species'].astype('int')
```

Le jeu de données Iris est un jeu de données regroupant 3 espèces de plantes :

- La Setosa
- La Versicolore
- La Virginica

Pour chaque plante nous avons mesuré en cm 4 caractéristiques.
La longueur de ses pétales et sépales (visible sous la photo ci-dessous) ainsi que leur largeur.

## 1.1-Iris flower

    
![iris_photo](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

On affiche le jeu de données Iris pour voir comment les informations sont représentées.  
Les 3 espèces ont été recodées de la manière suivante :

**0 = setos**

**1 = versicolor** 

**2 = virginica**

On compte 50 lignes par espèces pour un jeu de données au format (150,5)


<u>Tableau N°1 :  5 premieres lignes du jeu de données IRIS </u>

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | species |
|------------------:|-----------------:|------------------:|-----------------:|--------:|
|               5.1 |              3.5 |               1.4 |              0.2 |       0 |
|               4.9 |              3.0 |               1.4 |              0.2 |       0 |
|               4.7 |              3.2 |               1.3 |              0.2 |       0 |
|               4.6 |              3.1 |               1.5 |              0.2 |       0 |
|               5.0 |              3.6 |               1.4 |              0.2 |       0 |
    

Le tableau comporte 4 features(colonnes) ainsi que l'espèce de la fleur.  

Regardons également sur le graphique N°1 la dispersion de la longueur des pétales en fonction de leur largeur.

<u>Graphique N°1 :  Répartition des Iris en fonction de la longueur et largueur de ses pétales</u>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_1_iris_scatter_y3.png" style="width:600px;"/>

On remarque que les points rouges ont une distribution très différente des autres.  
Pour les points gris et oranges on voit également qu'ils appartiennent à 2 distributions distinctes mais la frontière entre les 2 est plus mince

<center><h1>2-Support Vecteur Machine</h1></center>

## 2.1-Presentation intuitive d'un SVM


Prenons un exemple de classification binaire avec le data set Iris.  
Pour faire simple nous utiliserons seulement 2 features (plus simple à représenter)

Nous décidons de conserver la longueur du petal et la largeur du petal( colonnes 3 & 4) 

🤔  
Imaginons maintenant qu'on nous demande de tracer une fonction permettant de séparer nos 2 nuages de points par une frontière.  
**Comment feriez-vous?**



Nous cherchons une fonction de la forme :<br>
$f(x) = X^T\beta = \beta_0 + x_1\beta_1 + x_2\beta_2 + ... + x_n\beta_n =0$

Cette fonction de permettre de déterminer :

$f(x)>0 : classe: 1$<br>
$f(x)<0 : classe:0$

<u>Graphique N°2 :Frontière de décision pour classification binaire</u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_2_intuition_svm.png" style="width:600px;"/>

Pour ce problème, il existe une **infinité de solutions**. Comment en déterminer une optimale??

Une première piste est qu'en machine learning, notre objectif est de réaliser une prédiction.  
Pour cela, il faut donc que notre modèle soit **généralisable**. Un point qui ne figure pas dans les données d'entrainement, devra être bien classifié.


Nous allons donc apporter une contrainte à notre frontière de décision $f(x)$.
Elle ne devra pas être trop proche des points en distance.  
Plus elle sera proche plus on aura de chance qu'un nouveau point issu de la distribution 0 ou 1 soit mal classifié.

**Autrement dit, nous devons trouver une contrainte à notre fonction afin d'apporter une solution unique.**




<u>Graphique N°3 : Calculer une distance </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/03_calcul_distance.png" alt="03_calcul_distance" style="width:400px;"/>

Pour rappel la distance entre un point et une droite se calcule de la manière suivante pour un espace à 2 dimensions

<u>Equation N°1 : Distance dans d'un point à une droite dans R²</u><br>

La distance entre une droite $D$ d'équation $ax+by+c = 0$ et un point $P$ de coordonnées $(x1,y1)$ est

$\normalsize d(D,P) = \frac{\vert{ax+by+c}\vert}{\sqrt{a²+b²}}$

On peut généraliser ce calcul de distance pour trouver les marges du SVM.

On va **chercher le ou les points** qui permettent de **maximiser l'écart entre la marge et la frontière de décision** (hyperplan) tout en **minimisant l'écart entre la marge à un des points d'entraînement**

Si on le formalise cela donne la formule suivante:


## 2.2-Calcul de la marge

<u>Equation N°2 : Formule de calcul SVM </u>

Si on transpose notre exemple de R² à notre problème de SVM on obtient :

L'équation de la frontière de décision noté $H$ est : $f(x) = \lt\beta.x \gt= \beta^Tx+b$ 

L'écart entre la frontière de décision et une marge est alors notée: $\large\frac{(\beta^Tx+b)}{\vert\vert\beta\vert\vert_2}$<br>
Comme il y a 2 marges on obtient la formule suivante :
<br>
$\normalsize Marge = 2d(x,H) =2\frac{(\omega^T\beta+b)}{\vert\vert\omega\vert\vert_2}$


où :<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\beta$ : paramètre du modèle <br>
&nbsp;&nbsp;&nbsp;&nbsp; $\vert\vert\beta\vert\vert_2$ désigne la norme euclidienne de $\beta$ : $\sqrt{\beta_1²+\beta_2²+\beta_3²+...+\beta_n²}$ 


💡Pour rappel, la marge est la distance minimale de l’hyperplan à un des points d’entraînement.


## 2.3-Maximisation de la marge

On veut trouver l’hyperplan de support qui permet de maximiser cette marge, c’est-à-dire qu’on veut trouver un hyperplan avec la plus grande marge possible.
<br>
Puisque l’on cherche l’hyperplan qui maximise la marge, on cherche l’unique hyperplan dont les paramètres $(\beta,b)$ sont donnés par la formule :
<br>

$\large arg max_{\beta,b} min_k \frac{l_k(\beta^Tx+b)}{\vert\vert\beta\vert\vert_2}$

où $l_k$ est le label de la données

<u>Graphique N°4 : Frontière de décision et marges, SVM linéaire</u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_4_visualisation_svm_lineaire.png" alt="fig_4_visualisation_svm_lineaire" style="width:600px;"/>
    
Sur le graphique N°4, on peut observer les points frontières qui maximisent l'écart entre la marge et la frontière de décision.
Ici on peut dire que la frontière de décision est bonne.  
Elle est suffisamment large pour que la probabilité qu'un point soit mal classifié est peu probable.

📝  **Pour info**
- On appelle les points prochent des marges ==> **Points edges** ou **points supports**
- La fonction qui sépare nos 2 ensembles de points est une **frontière de décision**
- Les droites proche des points edges sont les **marges**

❓**Questions**

Que se passe-t-il si je rajoute une observation? 
Ma frontière de décision change-t-telle?

⚠️**Attention**

Notez qu'il est important de toujours standardiser vos données lorsque vous utilisez des SVM.  
De manière générale quand vous utilisez des modèles avec calcul de distance, pensez à standardiser vos données.
   
   Pour cela vous avez plusieurs opérations mathématiques pour mettre vos données à la même échelle, voici les principaux à retenir :
   
   - **Min Max Scaling**
   - **Normalization** 
   

### Standardisation
Prenons le jeu de données suivant avec $x_1$ et $x_2$ des features associé à un label {0,1}.  
On a également rajouté la standardisation de nos 2 features $x_1scaled$ et $x_2scaled$

Nous lançons un SVM linéaire pour tenter de classifier les labels 0 et 1 en fonction de ces features.
Un premier avec les features brutes et l'autre avec les features standardisés

Regardons maintenant sur le graphique N°5 comment la standardisation impact la frontière de décision.

<u>Tableau N°2 : Feature scaling</u>
<br>
| x1 | x2 | x1_scaled | x2_scaled | label |
|----|----|-----------|-----------|-------|
| 1  | 50 | -1.507    | -0.115    | 0     |
| 5  | 20 | 0.904     | -1.5010   | 0     |
| 3 | 80  | -0.301    | 1.270     | 1     |
| 5  | 60 | 0.904     | 0.346     | 1     |

<u>Graphique N°5 : Influence de l'échelle des données sur le modèle </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_5_scaling_data.png" alt="scaling_features_5" style="width:600px;"/>

<style>
div.red { background-color:#ff000020; border-radius: 5px; padding: 20px;}
</style>
<div class = "red">
📝  <strong>Pour info</strong>
    
Les SVM sont sensibles à l'échelle des données. Il est important de standardiser ses données avant d'entrainer le modèle.

$\large Xscale = \frac{X-\mu}{\sigma}\$

</div>

## 2.4-SVM LINEAIRE, HARDS MARGING VS SOFT MARGIN CLASSIFCATION

A ce stade tout se passe bien et la classification d'un SVM semble être parfaite pour notre jeu de données.

Cependant il est très rare qu'un jeu de données soit linéairement séparable... pour ainsi dire jamais avec des données d'entreprise.
Pour l'instant nous avons vu ce qu'on appelle un SVM à *hard margin classification*. Chaque individu doit être d'un coté de la zone de décision.

Autrement dit, on ne peut pas trouver une versicolore du coté d'une setosa

**Mais celà pose 2 problèmes**



Prenons l'exemple du graphique N°6.

Il est impossible de réaliser une classification linéaire à l'aide d'un SVM.
L'outlier empêche de trouver une frontière de décision qui permettrait de classifier parfaitement nos 2 groupes.

Dans le 2nd cas, visible sur le graphique N°6, l'outlier est tellement éloigné du point moyen de son groupe qu'il réduit drastiquement l'écart des marges du SVM.
<br>
La frontière de décision ne sera pas optimale et il sera compliqué de généraliser ce modèle à d'autres individus.


<u>Graphique N°6 : Données non linéaire et SVM </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_6_svm_linear_problem.png" alt="fig_6_svm_linear_problem" style="width:800px;"/>

Pour éviter ce genre de problème, les statisticiens ont développé un modèle plus flexible.
Son objectif est de trouver un équilibre entre la maximisation des marges et le nombre de fois où l'on peut ignorer un point.
(nombre de points du mauvais coté, mal classifié)

Le modèle s'appelle le *soft margin classification* en opposition au *hard margin classification* vu plus haut.



### Soft Margin

La soft margin classification fait intervenir un nouveau paramètre dans le modèle. On l'appelle paramètre de régularisation $C$.
Il est à valeurs dans $]0,\infty[$.
On parle ici d'hyperparamètre car sa valeur optimale n'est pas fixée mais dépend du jeu de données. C'est à vous de le trouver pour optimiser votre modèle

Regardons comment faire avec le code python suivant:

```python
#Import Package
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype('int32') # Iris virginica

# Code

#Standardiser nos données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Entrainement modèle
# Loss='hinge' permet de dire qu'on utile un SVM classique
# Vous pouvez regarder la doc pour avoir + d'infos
svm_clf= LinearSVC(loss='hinge', C=1)
svm_clf.fit(X_scaled, y)

#Prediction pour un nouveau point
svm_clf.predict([[1,1]])
#==> array([1.])

```

**Plus on augmente la valeur de $C$ plus le modèle va avoir tendance à produire des marges proches de la frontière de décision et à l'inverse plus $C$ est petit plus la frontière sera grande.**
<br>
Pour mieux le comprendre, regardons le graphique N°7 . Cela représente 3 SVM entrainés avec 3 valeurs différentes de $C$ à savoir 1, 50 et 100

Code suivant
```python
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=50, loss="hinge", random_state=42)
svm_clf3 = LinearSVC(C=500, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf3 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf3),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)
scaled_svm_clf3.fit(X, y)
```

<u>Graphique N°7 :Influence du critère de regularisation </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_7_regularisation_critere.png" alt="fig_7_regularisation_critere" style="width:1400px;"/>

Pour les autres hyperparamtres fixés, une augmentation de C permet de diminuer la taille des marges. 
Plus la taille de la marge sera faible plus il sera compliqué de généraliser pour le modèle
<br>
<br>
ℹ️ Le SVM linéaire est également disponible à travers la fonction **SGDClassifier** de sklearn. La différence essentielle provient de l'optimiseur utilisé. Ici la fonction utilise une descente de gradient qui peut s'avérer utile dans le cas de dataset avec beaucoup de ligne (n grand)

On l'utilise de la même manière

```python

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, loss='hinge', alpha=0.001))
#alpha=1/(n*C)
clf.fit(X, Y)
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

```

## 2.5-SVM non liénaire

La classification linéaire est très utile et peut s'avérer précise pour de nombreux dataset. Malheureusement quand on se confronte à des données réelles et non pas un dataset kaggle, les données sont rarement linéairement séparables.

Les modèles linéaires fournissent des performances assez faibles et ne permettent pas de répondre à votre problème.
Heureusement, il existe des techniques pour faire évoluer les SVM et traiter les cas où les données ne sont pas linéairement séparables.

Prenons l'exemple intuitif suivant :

<u>Graphique N°8 : Données non linéairement séparables  </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_8_SVM_non_lineaire.png" alt="fig_8_SVM_non_lineaire" style="width:800px;"/>

Ce problème est un cas récurrent en machine learning. Nous cherchons à classifier des données mais les features disponibles ne permettent pas de le faire.

C'est un cas d'école de **feature engineering**.
Ici il faut transformer nos données brutes de telle sorte qu'on puisse classifier nos données linéairement après transformation.

Une transformation possible est d'ajouter un feature qui serait $X_2 = X_1²$

Regardons graphiquement le résultat

<u>Graphique N°9 : Astuce pour données non linéairement séparables </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_9_svm_separation_lineaire.png" alt="fig_9_svm_separation_lineaire" style="width:800px;"/>

En utilisant une transformation qui nous fait passer d'un problème à 1D à 2D, on trouve un espace où nos données sont linéairement séparables.
**Ce type de transformation est très utile pour les SVM mais s'applique à tous les modèles de machine learning.**


<style>
div.red { background-color:#ff000020; border-radius: 5px; padding: 20px;}
</style>
<div class = "blue">

Si vos données brutes offres des performances médiocres, pensez à faire du feature engineering sur vos données.
         
- Appliquer des fonctions sur vos données pour en créer des nouvelles (log, puissance, sigmoide, loi normale, ...)
- Combiner des données $X_{new} = X_1*X_2$
- Créer de nouveau feature avec des modèles (ACP, ACM, AUTOENCODER, KNN)
- Tester, soyez créatif ;)
</div>

Pour implémenter ce type d'approche sklearn offre des fonctions toutes faites.
Vous pouvez utiliser la fonction *PolynomiaFeatures* dans le module sklearn.preprocessing qui permet de faire des transformations polynomiales pour chaque feature numérique.

```Python
#Import package
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
#Function sklearn qui genere données en forme de lune
X, y = make_moons(n_samples=100, noise=0.15)
polynomial_svm_clf = Pipeline([
("poly_features", PolynomialFeatures(degree=3)),
("scaler", StandardScaler()),
("svm_clf", LinearSVC(C=10, loss="hinge"))
])
polynomial_svm_clf.fit(X, y)

```

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

#Transformation polynomiale de nos features
poly = PolynomialFeatures(degree=3)
X_degr3 = poly.fit_transform(X)

#Standardiser les données
scaler = StandardScaler()
X_degr3_scaled = scaler.fit_transform(X_degr3)

#SVM classification

polynomial_svm_clf_test= LinearSVC(C=1, loss='hinge', random_state=42)
polynomial_svm_clf_test.fit(X_degr3_scaled,y)
```

<u>Graphique N°10 : SVM transformation polynomiale </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_10_classification_non_lineaire.png" alt="fig_10_classification_non_lineaire" style="width:800px;"/>

### 2.5.1-Polynomial Kernel

L'approche par transformation des features convient parfaitement pour des datasets de petite taille et peu complexe.
En effet, une transformation polynomiale avec des degrés faibles ne pourra pas s'adapter à un dataset complexe et des degrés trop élevés créeront un dataset trop grand pour être entrainé dans un laps de temps intéressant.

Rappel :

Plus on augmente la taille d'un dataset (features ou lignes) plus le modèle devient complexe à entrainer et plus le temps de calcul sera long...


Heureusement, les mathématiciens ont pensé à tout et il existe une astuce avec les SVM.
On l'appelle le *kernel trick*. Cela permet d'obtenir les mêmes résultats qu'en ajoutant des *polynomial features* sans faire exploser la taille du dataset.

Sans passer par la démonstration mathématique, il faut retenir qu'utiliser le *kernel trick* permet d'avoir un modèle de complexité $O(n^d)$ vs $O(n)$.

Cette technique est implémentée directement dans sklearn avec la fonction suivante

```python
from sklearn.svm import SVC

svm = SVC(kernel="poly", degree=3, C=50,coef0=1)
svm.fit(X, y)

#Degree ==> Degre polinomiale
#Kernel ==> Le type de noyau
# C ==> Paramètre de tolérance (régularisation)
# coef0 ==> Contrôle l'infulence des polynomes
```

<u>Graphique N°11 : SVM non linéaire influence des hyperparamètres </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_11_svm_no_lineaire_hyperpara.png" alt="fig_11_svm_no_lineaire_hyperpara" style="width:1000px;"/>

La facon la plus simple de trouver les hyperparamètres adéquats et de réaliser un *grid search*.<br>
Nous verrons en TD comment l'implémenter avec sklearn

### 2.5.2-Similarity Features

Il existe de nombreux *kernel trick* utile pour trouver un espace à plus grande dimension où nos données sont linéairement séparables.
Vous pouvez aller voir celles qui sont implémentées avec [sklearn](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)

La dernière que nous allons voir est une des plus populaires pour les SVM est la fonction de similarité *Gaussian Radial Basis Function*. Elle se définit formellement de la façon suivante

<u>Equation N° 3  : kernel trick</u>

$\phi_\gamma(x,x') = exp(-\gamma\vert\vert x-x'\vert\vert²)$

où
$x'$ = Un point repère que nous choisisons 

Exemple :

Prenons le cas du graphique 1D N°12
Nous prenons $x'$ = {-2,1} comme repères et $x$ = -1 pour un $\gamma =0.3$

Nous obtenons donc les fonctions de similiratés suivantes pour 2 nouveaux features $x_2$ et $x_3$

$x_2 = exp(-0.3*1²) \simeq  0.74$

$x_3 = exp(-0.3*2²) \simeq  0.3$





<u>Graphique N°12 : Similarity feature construction </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_12_similarity_features.png" alt="fig_12_similarity_features" style="width:800px;"/>

Comme pour les *polynomials features*, vous pouvez créer à la main vous-mêmes les features que vous souhaitez rajouter dans votre dataset avec cette technique.
Choissier autant de 'repères' que vous avez de ligne dans votre dataset pour créer de nouveaux features.

**Problème**, cette technique peut rapidement **faire exploser la taille de votre Dataset si $n$ est grand**. Vous obtiendrez à la fin un Dataframe de taille $n*n$ (en supposant que vous supprimez les features de bases)

Heureusement pour nous sklearn propose une implémentation optimisée dans ses fonctions pour cette approche. Exactement comme pour les *polynomial features*.

On retrouve le *kernel trick* dans la fonction  SVC de sklearn avec comme noyau(kernel) 'rbf' pour Radial Basis Function kernel vu au-dessus.


```python

from sklearn.svm import SVC

svm = SVC(kernel="rbf",  gamma=5, C=0.001)
svm.fit(X, y)

#gamma rend la distribution plus étroite ce qui donne des frontières de décisions plus irrégulières
#chaque observations influences plus la frontière de décision

# C ici ne change pas. Il est toujours un criètre de tolérance


```

Notez également que gamma comme C est un hyperparamètre permettant de régulariser le modèle.<br>
Si votre modèle est en *overfitting* pensez à réduire gamma/C et inversement s'il est en *underfitting*

<u>Graphique N°13 : Frontière de décision et similarity features </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/13_rbf_kernel.png" alt="13_rbf_kernel" style="width:800px;"/>

<style>
div.blue { background-color: rgba(117, 190, 218, 0.5); border-radius: 5px; padding: 20px;}
</style>
<div class = "blue">

Tips :
Parfois vos données ne sont pas au format numérique et le *kernel trick* nécessite un noyau spécifique.
Il faut savoir qu'il existe des noyaux adaptés pour différentes structures de données :
    
- String kernel pour la classification de text par exemple ( cf *string subsequence kernel* ou *Levenshtein distance*)
    
    
Comment choisir son *kernel trick* parmis ceux disponibles?
Il n'y a pas de règle écrite qui permet de choisir directement. La meilleur réponse est ca dépend de vos données.
    
Mais le schéma suivant marche généralement bien:
    
- 1, commencer par un SVM linéaire dispo avec la fonction LinearSVC
- 2, si le training set n'est très grand vous pouvez utiliser le noyau *Gaussian RBF kernel*
- 3, si vous avez du temps, testez d'autres noyaux mais 1 et 2 est généralement suffisant pour voir si les SVM sont adaptés au problème

⚠️ Pensez bien à *tuner* votre modèle avec un *gridsearch* et une *cross-validation* avant de comparer vos modèles avec votre test set
    

</div>

## 2.6-Classification Multiclass pour les SVM


De nombreux modèles permettent nativement de réaliser des classifications multiclasse (Random Forest, SGBDclassifiers, Naives Bayes, ...).
Malheureusement les SVM n'en font pas partie.

Pour les ustiliser lors de classifications multiclasse, nous devons trouver une parade!

### Exemple, classification de chiffre manuscrit

#### Contexte
Imaginons qu'on nous donne un dataset contenant des images. Chaque image représente un chiffre manuscrit entre 0 et 9.
On nous demande de créer un modèle basé sur un SVM afin de classifier ces chiffres manuscrits (l'humain qui le fait habituellement en à marre de le faire).

Le data scientist en charge du projet à bien compris la problématique mais sait aussi qu'un SVM ne permet pas de faire de la classification multiclasse...
Il cherche alors une stratégie pour répondre parfaitement à la commande.

#### Solution
Sa **premiere intuition** est de **découper le problème en 10 problèmes distincts.**
10 classifications binaires où il va chercher à identifier les 1 VS les autres puis les 2 VS les autres etc.
Après les avoir entrainés, il obtiendra le score de décision grâce à sklearn et prendra celui qui le maximise.

Un collègue lui souffle également **une autre idée.**
**Réaliser des modèles par pair. Un modèle 1vs2, 1vs3, 1vs4,...2vs3,, ... etc**
Il essaye cette approche mais obtient **45 modèles** différents pour ce problème.


C'est 2 approches sont appelés **OVR(one versus rest)** pour la premiere et **OVO(one versus one)** pour la seconde.
Elles permettent d'approcher un problème multiclasse avec une classification binaire.
Leur principal défaut est de faire exploser le nombre de modèles à entrainer.

- **One versus rest** : On ramène un problème avec $N_{class}$ à $N$ classification binaire. On compare une classe à $N-1$ classe.<br>
<br>

- **One versus one** : On ramène le problème avec $N_{class}$ à $N\times\frac{N-1}{2}$ classification binaire. Cette fois on teste toutes les combinaisons de N possibles divisées par 2. (Classification $N_1 | N_2$ est la même que $N_2 | N_1$)



L'implémentation avec sklearn est encore une fois chose facile
#### OVR

```python
#Import OVO, OVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
#Import SVM
from sklearn.svm import SVC

OvR_SVC_clf = OneVsRestClassifier(SVC())

OvR_SVC_clf.fit(trainX, trainY)

```

### 2.6.1-Application classication multiclass, dataset MNIST

Pour exemple, nous pouvons utiliser ces stratégies pour le jeu de données MNIST.
C'est un jeu de données célèbre qui comprend 70 000 images de chiffre écrit à la main. Chaque image a été classifié et le jeu de données est parfait pour s'entrainer à ce type de données

Pour le charger utilisez la commande suivante sur python

```python
from sklearn.datasets import fetch_openml
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
#Import SVM
from sklearn.svm import SVC
##########
#GET DATA#
##########
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'details',
# 'categories', 'url'])

X , y = mnist["data"] , mnist['target']
X.shape
#(70000, 784)
# Chaque image contient 784 features qui correspodent à la distribution de ses pixels en nuance de gris.
# Sa valeur est entre 0 et 255

#############
#Train model#
#############
# Suivant votre quantité de RAM, attention à combien de ligne vous prenez pour entrainer votre modèle!!
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],y[60000:]


OvR_SVC_clf = OneVsRestClassifier(SVC())
OvO_SVC_clf = OneVsOneClassifier(SVC())

OvR_SVC_clf.fit(X_train, y_train) 
OvO_SVC_clf.fit(X_train , y_train)    

```

<u>Graphique N° 14: Visualisation chiffres manuscrits </u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_14_mnist.png" alt="fig_14_mnist" style="width:600px;"/>

<center>************************Demo avec le code 01_SVM_DEMO************************</center>

## 3-SVM pour la régression 

La dernière partie consacrée au SVM sera celle sur la régression.
Jusqu'à maintenant nous avons vu les SVM pour la classification et il n'est pas nécessairement intuitif de voir comment l'appliquer à la régression (du moins théoriquement, ca vous prendra une ligne avec sklearn)


Globalement les SVM pour la régression sont aussi flexibles. Ils permettent de faire de la régression *linéaire* et non *liénaire* avec les mêmes techniques vues précédemment.
La nuance est que nous devons inverser notre objectif! Ici on ne cherche plus à maximiser la marge entre 2 classes tout en limitant le nombre de violations.

Le modèle cherche à inclure le maximum d'observation à l'intérieur de ses marges tout en limitant le nombre d'observations à l'extérieur. La largeur de la frontière de décision sera contrôlée par un nouvel hyperparamètre $\epsilon$.

Regardons comme il agit à travers 2 exemples

<u>Graphique N°15 : Exemple SVM régression</u>
<br>
<img src="https://github.com/Roulitoo/cours_iae/blob/master/01_SVM/img/fig_15_svm_reg.png" alt="fig_15_svm_reg" style="width:700px;"/>

Vous pouvez utilisez le code suivant pour l'implémenter sous python.
La logique est la même que pour la classfication 😉

```python 
#SVM Regression liénaire
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

#SVM Regression non liénaire
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)



```

## 4-Récapitulatif

<u>Tableau N°3 : Résumé des fonction utilisées</u>

| Sklearn class | Scaling | Kernel trick | Hyperparameter                                          | Computational complexity | Multiclass                                | Sklearn import                                 |
|--------------:|---------|-------------:|---------------------------------------------------------|-------------------------:|-------------------------------------------|------------------------------------------------|
| SVC           | OUI     | OUI          | kernel,C                                            |          O(m^3n)         | OneVsRestClassifier<br>OneVsOneClassifier | from sklearn.svm import SVC                    |
| LinearSVC     | OUI     | NON          | loss ='hinge',<br>C=> Tolérance                         |          O(m*n)          | OneVsRestClassifier<br>OneVsOneClassifier | from sklearn.svm import LinearSVC              |
| SGDClassifier | OUI     | NON          | loss = 'hinge',<br>max_iter,<br>Alpha ==> Learning rate |          O(m*n)          | OneVsRestClassifier<br>OneVsOneClassifier | from sklearn.linear_model import SGDClassifier |
| LinearSVR     | OUI     | NON          |  Epsilon , C                                                      |          O(m*n)          |                                           | from sklearn.svm import LinearSVR              |
| SVR           | OUI     | OUI          |Epsilon,  C , kernel                                                        |          O(m^3n          |                                           | from sklearn.svm import SVR                    |

**Avantages des SVM**<br>
- Ils marchent relativement bien sur les dataset avec peu de features<br>
- Ils fournissent une séparation clair de vos données. On peut interpréter le modèle.<br>
- Ils s'adaptent très bien au dataset avec plus de features que de data points.<br>
- On peut spécifier différent $kernel$ pour trouver une meilleur frontière de décision<br>

**Désavantages des SVM** 
- Leur entrainement nécessite beaucoup de temps de calul et de ressource.<br> 
- Ce n'est pas recommandé de les utilisé si vous avez un dataset grand (millions de lignes)<br>
- Ils sont très sensibles aux outliers<br>
- Ne supportent pas nativement le multiclass<br>


> Si vous souhaitez un supplément de cours plus mathématique vous pouvez consulter le lien [suivant](https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-app-svm.pdf)

> La page wikipédia de [Vapnik](https://fr.wikipedia.org/wiki/Vladimir_Vapnik) l'inventeur du SVM

> Une video avec Vladmir Vapnik sur le [l'apprentissage automatique](https://www.youtube.com/watch?v=STFcvzoxVw4&ab_channel=LexFridman) 
