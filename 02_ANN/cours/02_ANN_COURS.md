<center><h1> Artificial Neural Network sous Python</h1></center>
<p align="center">
<img src="https://github.com/Roulitoo/cours_iae/blob/master/00_intro/img/Logo_IAE_horizontal.png" alt="Logo IAE.png" style="width:200px;"/>
</p>

#### Table of Contents
[1. Histoire des réseaux de neurones](#1-histoire-des-r%C3%A9seaux-de-neurones)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1.1 Neurone biologique & neurone artificiel ](#11-du-neurone-biologique-au-neurone-artificiel)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1.2 logique computationelle ](#12-logique-de-computation-dun-neurone)<br>

[2. Le perceptron](#2-le-perceptron)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Fonction d'activation](#21-fonction-dactivation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.2 Définition d'un perceptron](#22-d%C3%A9finition-perceptron)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.3 Entrainement perceptron](#23-entrainement-perceptrone)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.4 Classification non linéaire](#24-probl%C3%A8me-pour-la-classification-lin%C3%A9aire)<br>

[3. Perceptron multicouche & backpropagation](#3-perceptron-multicouche-et-backpropagation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1 Algorithm rétropropagation](#31-algorithme-de-r%C3%A9tropropagation-du-gradient)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2 Fonctions d'activations](#32-fonctions-dactivations-communes)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3 MLP pour la régression](#33-mlp-pour-la-r%C3%A9gression)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.4 MLP classification](#34-mlp-pour-la-classification
)<br>

[4. Tensorflow & Keras](#4-mpl-avec-keras-et-tensorflow)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1 Tensorflow](#41-tensorflow)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2 Keras](#42-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.3 Comment utiliser Keras](#43-comment-utiliser-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.4 Classification avec Keras](#44-classification-avec-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.4.1 Implémentation avec Keras](#441-impl%C3%A9mentation-avec-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.4.2 Compiler le modèle](#442-compiler-le-mod%C3%A8le)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.4.3 Entrainer et évaluer](#443-entrainer-et-%C3%A9valuer-le-mod%C3%A8le)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.4.4 Learning curve avec Keras](#444-leaning-curve-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.5 Régression avec Keras](#45-r%C3%A9gression-avec-keras)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.5.1 Wide and Deep Neural](#451-wide-and-deep-neural-model)<br>
[5.Hyperparamètre tuning](#5-hyperparameter-tuning)<br>


## 1-Histoire des réseaux de neurones

Historiquement les réseaux de neurones se sont inspirés du fonctionnement des neurones biologique humain.<br>
Leur conception provient d'une représentation schématique de comment marche notre traitement de l'information.<br>

Les réseaux de neurones sont donc un modèle de machine learning inspiré par le fonctionnement des neurones.<br>

> D'apres [Yann Lecun](https://fr.wikipedia.org/wiki/Yann_Le_Cun), ' Les réseaux de neurones ne prétendent pas plus d'écrire le cerveau qu'une aile d'avion copie celle d'un oiseau'

Aujourd'hui ces modèles se sont largement diffusés et sont utilisés pour des problématiques liées au traitement cognitif humain.<br> 
La majorité de leur domaine d'application a pour but de remplacer des actions humaines.

**Exemple :**

- Reconnaissance vocale avec Siri, Apple
- Système de recommandation, Netflix
- Voiture autonome, Tesla
- Jouer aux échecs, Google Deep Mind
- ...


Les premières recherches sur ce type de modèle **datent des années 1950** et ont été réalisées par 2 neurologues, **Warren McCulloch** et **Walter Pitts**.<br>
Leurs travaux consistaient à décrire comment fonctionnaient les neuronnes en utilisant les mathématiques afin de décrire des neurones dits formels.

Après ces premières recherches, le domaine a connu une longue période sans réelle avancée. De nombreux freins empéchaient leur utilisation et des modèles de machine learning traditionnels leur était préréféré (type SVM).

Cependant à la fin des années 90 de nombreuses avancées ont permis la démocratisation des réseaux de neurones (Artificial Neural Network)

- La quantité de Data disponible. Généralement les ANN offrent de meilleure performance sur les jeux de données très grand
- L'augmentation de la puissance de calcul et l'apparition des GPU facilitant le traitement distribué
- L'amélioration de l'algorithme d'apprentissage (descente de gradient)


### 1.1-Du neurone biologique au neurone artificiel

On peut observer sur l'image N°1 une cellule neurale obtenue dans le cerveau d'un animal.

Graphique N°1 : Cellule neurale d'un animal

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/neural_bio_01.png" alt="01_image_neurone.png" style="width:600px;"/>

La cellule est composée d'un noyau et un long axon permettant de transmettre l'information avec un faible courant électrique à une synapse qui ensuite, libère une substance chimique appelée neurotransmetteur et qui à son tour sera reçu par un neurone et ce nouveau neurone transmettra une information via une impulsion électrique.

Les neurones sont tous interconnectés entre eux et on dénombre en moyenne **86 milliards de neurones pour l'espèce humaine**.<br>
Sur l'image N°2 on peut observer un plan en coupe représentant l'interconnexion des neurones entre eux.

Graphique N°2 : Connexion entre neurones

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/multi_neural_bio_02.png" alt="02_image_connect_neural.png" style="width:600px;"/>

Garder à l'esprit la structure de cette image qui représente des neurones linéairement maillés.

### 1.2-Logique de computation d'un neurone

Logique de computation des neurones, travaux de 1950. Traduction d'un neurone biologique dans l'espace mathématique

Graphique N°3: Logique formelle des neurones

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/neural_formal_03.png" alt="03_logic_neural.png" style="width:800px;"/>

On dénombre **4 cas formulés par Warren McCulloch et Walter Pitts en 1943.**

- Le premier réseau s'active si le neurone A est activé puis C s'active car il reçoit l'input de A
- Le second cas s'active si le neurone A et le neurone B s'active puis envoie leur input à C. A seul ou B seul ne suffit pas pour activer C
- Le troisième cas s'active si A ou B est activé ou les deux.
- Le quatrième s'active si seulement A est activé et B est off.

Si on combine ces 4 formes de logique mathématique nous pouvons déja créer beaucoup de modèles de réseaux de neurones.<br> C'est d'ailleurs ce que nous allons voir par la suite.


## 2-Le perceptron

Le perceptron est le modèle de réseaux de neurones le plus simple. Il a été inventé en 1957 par Frank Rosenblatt.<br>
Il est basé sur un type de neurone légèrement différent de ceux du graphique N°3.<br>
On le nomme **threshold logic unit(TLU)**

Les intput et output sont ici des nombres et non pas des valeurs binaires (on, off)
Chaque input est associé à un poids (w)

Le TLU noté z est égal à 

Equation N°1 : Fonction liénaire<br>

$z=w_1x_1+w_2x_2+...+w_nx_n = X^TW$ <br>

Puis on applique à z la *step function* qui génère l'output.<br>

On note $h_w(x) = step(z)$ la *step function*

Graphique N°4 : Architecture d'un perceptron

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/percetron_04.png" alt="04_perceptron_archi.png" style="width:600px;"/>


### 2.1-Fonction d'activation

La fonction d'étape la plus utilisée pour le perceptron est la *Heaviside step function* de formule:<br>

Equation N°2 : Heaviside<br>

$$\normalsize heaviside(z)=\begin{cases}0&if(z\lt 0)\\\1&if(z\geq 0)\end{cases}$$

On peut également utiliser la fonction signe définie comme suit :<br>

Equation N°3 : Fonction sign <br>

$$\normalsize sgn(z)=\begin{cases}-1&if(z\lt 0)\\\0&if(z= 0)\\\\+1&if(z>0)\end{cases}$$

Ce type d'architecture peut être utlisée pour des classifications linéaires simples.<br>
Le modèle ressemble énormément à une régression logistique ou un SVM. La seule différence est qu'il génère systématiquement en output la classe à prédire et non pas une proba comme une régression logistique.

Note:

> On parle d'un réseau de neurones pleinement connecté si tous ses neurones sont reliés (fully connected layer)

> Le perceptron peut également servir pour du multiclass

Suivant le nombre de neurones dans la couche de sortie nous pouvons nous ramener à une problématique de multiclassification 

Graphique N°5 : Perceptron architecture pour multiclass 

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/perceptron_05.png" alt="05_perceptron_multi.png" style="width:600px;"/>

### 2.2-Définition perceptron

Mathématiquement, sa fonction est de la forme :

Equation N°4 : Fonction d'un perceptron *fully connected layer*<br>

$\normalsize h_{W,b} = \phi(XW+b)$

où

- $X$ : Matrice de nos inputs<br>
- $W$ :  Matrice des poids attribués à chaque lien entre neurones. Sauf pour le neurone biais.         
  Elle est de taille nombre de neurones dans l'input layer et nombre de colonnes par neurone artificiel<br>
- $b$ : Est le vecteur qui contient tous les poids entre le biais et les neurones artificiels<br>
- $\phi$ : Est la fonction d'activation. C'est un TLU quand la fonction d'activation est de type **step function**

### 2.3-Entrainement perceptron


Le premier modèle du perceptron développé en 1957 par **Frank Rosenblatt** s'inspire largement de la règle de *Hebb*(1949) pour l'apprentissage de son modèle.<br>

La règle de *Hebb* s'appuie sur une étude biologique de nos neurones et à déterminer que quand un neurone en déclenche un autre leur lien se renforce.<br>

C'est exactement sur ce principe que Frank Rosenblatt développe sa règle d'apprentissage pour son perceptron.
Le perceptron renforce **le lien entre ses neurones qui aides à réduire l'erreur de prédiction**, formellement cela se définit comme suit:

Equation N°6: Règle d'apprentissage du perceptron

$\normalsize w_{i,j} = w_{i,j}+ \eta(y_j-\hat{y_j})x_i $

où

- $w_{i,j}$ : Le poids de connexion entre le ième neurone et le jème output neuron 
- $x_i$ : La i_ème valeur de l'échantillon  passé par un neurone
- $\hat{y_j}$ : L'output de du jème neurone de sortie
- $y_i$ : La target du jème output neuron pour une ième valeur de l'échantillon
- $\eta$ : Learning rate

⚠️ <br>
Le **perceptron** comme la régression logistique est un **classifieur linéaire**.<br>
Il est incapable de produire une frontière de décision complexe!<br>
Cependant si la dataset est linéairement séparable il convergera vers une solution optimale.


L'implémentation de cet algorithme est disponible sous sklearn avec le code suivant:

```python
#Package
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
#Data
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) 
#Train model
per_clf = Perceptron()
per_clf.fit(X, y)
#Predict
y_pred = per_clf.predict([[2, 0.5]])
```

La fonction ressemble énormément à la fonction SGDClassifier (mêmes hyperparamètres).<br> 
Vous pouvez donc accéder au perceptron via la fonction SGDClassifier en spécifiant :
- la fonction de perteloss='perceptron'
- learning_rate="constant"
- eta0=1 (learning rate)
- penalty=None (no regularization).


```python
#Package
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
#Data
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) 
#Train
clf = SGDClassifier(max_iter = 1000, loss='perceptron', eta0=1, learning_reate='constan', penalty=None)
clf.fit(X, Y)
#Predict
print(clf.predict([[2, 0.5]]))
```

🛈 **En dehors des TD, préférez une régression logistique qui retourne la probabilité  d'affectation à une classe plutôt que le perceptron qui offre *hard threshold***

### 2.4-Problème pour la classification linéaire

#### XOR problème

En 1969, 2 auteurs,  Marvin Minsky and Seymour Papert,  ont révélé les nombreuses faiblesses du perceptron. Ce modèle est incapable de réaliser une classification non linéaire et les auteurs ont démontré qu'il ne pouvait résoudre un problème trivial comme **XOR**

Les chercheurs ont trouvé une parade astucieuse afin de continuer à utiliser ce type de modèle pour des problèmes de classification non linéaire.<br>
En stackant des perceptrons, on peut éliminer plusieurs limites du perceptron classique de Rosenblatt.

Sur le graphique N°, on peut observer l'architecture du perceptron stacké avec tous les poids égaux à 1 sauf les 4 poids mis en rouge.

Graphique N°6 : Perceptron pour problème XOR

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/xor_problem_06.png" alt="06_XOR.png" style="width:600px;"/>

🛈 On nomme ce type d'architecture MLP (multi layer perceptron).

## 3-Perceptron multicouche et backpropagation

Le multi layer perceptron est composé de : 
- Une couche d'entrée appelée *input layer* de taille $\large  n_{features}$ + 1 biais
- Une ou plusieurs couche de TLU appelé *hidden layer* de taille {1,n}
- Une couche finale de TLU appelée *output layer*, a adapté suivant la problématique de modélisation

La couche cachée proche de l'intput layer est appelée couche basse et la couche cachée proche de l'output layer est appellée couche haute.

Chaque couche doit contenir un biais **sauf la couche de sortie**

Graphique N°7 : Architecture d'un réseau de neurone multicouche

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/mlp_07.png" alt="07_MPL.png" style="width:600px;"/>

💡
> Graphiquement on observe que tous les flux vont dans une unique direction. On parle ici de **feedforward neural network**.

> Quand un ANN contient plusieurs couches cachées on parle de **deep neural network**


### 3.1-Algorithme de rétropropagation du gradient
Pour entrainer ce type de modèle, on utilise une variante de la descente de gradient vue au premier cours.
On l'appelle la rétropropagation du gradient (backprogation gradient).

Sans le définir mathématiquement, le concept de rétropogation suit les étapes suivantes :

- **1)** Sélectionner un mini-batch(sous ensemble de données) de taille = n qui passera par l'ensemble de notre train
<br>

- **2)** Chaque mini-batch passera par l'input layer jusqu'à l'output layer pour réaliser une prédiction. Chaque instance du mini-batch passera. 
On conserve également les résultats intermédiaires de chaque neurone.
<br>

- **3)** Avec notre fonction de perte on mesure l'écart entre notre prédiction et la valeur réelle
<br>

- **4)** L'algorithme calcule la contribution de la couche de sortie sur l'erreur calculée ( En appliquant la dérivée des fonctions composées)
<br>

- **5)** Ensuite on regarde la part contributive de quelle connexion avec la couche précédente produit l'erreur.(Toujours avec le principe de dérivée de fonctions composées).
On remonte jusqu'à la couche d'entrée avec ce procédé.
<br>

- **6)** Finalement, l'algorithme ajuste toutes les connexions en utilisant la descente de gradient pour ajuster les paramètres

**En résumé:**<br>
L'algorithme rétropropagation du gradient calcule pour chaque instance d'un mini-batch une prédiction avec un forward pass et mesure l'erreur produite.<br>
En faisant chemin inverse, on mesure la contribution de chaque connexion dans l'erreur produite et enfin, on met à jour les poids de connexion pour réduire l'erreur.<br>
[Si vouz voulez une vidéo explicative cliquez ici](https://www.youtube.com/watch?v=OgSA7liZMXI&ab_channel=Science4All)

##### TIPS ⚠️
Il est important **d'initialiser le poids de chaque connexion de manière aléatoire autrement l'entrainement du modèle va échouer**.<br>
Si tous les poids sont les mêmes la rétropropagation du gradient affectera les poids de la même manière et l'erreur produite ne changera pas!


**Noter :**

Un point important pour l'utilisation de l'algorithme de *backpropagation du gradient*. **Les step function sont remplacées par une sigmoide.**
A ce stade on ne parlera plus de **step function** mais de **fonction d'activation**.

Ce changement provient essentiellement que la step function contient seulement des segments plats. Une valeur constante ne possède pas de dérivée, il est donc impossible de calculer un gradient.

Les auteurs ont remplacé la fonction d'activation par une sigmoide mais de nombreuses fonctions d'activations existent. Examinons les et regardons comment les choisirs.

### 3.2-Fonctions d'activations communes

La fonction logistique ou sigmoide est en forme de S et possède une dérivée pour chaque point sur sa courbe.
C'est la même fonction utilisée pour la régression logistique.

$\normalsize sigmoide : \sigma(z) = \frac{1}{1+exp(-z)}$

La tangente hyperbolique est une variation de la fonction sigmoide. Elle possède également une forme en S mais prend ses valeurs dans l'intervalle [-1,1]. Cela tend à produire des output de neurones centrés autour de  0, ce qui améliore la vitesse de calcul lors de la descente de gradient. 

$\normalsize Hyperbolic tangent : tanh(z) = 2\sigma(2z)-1$

La fonction ReLU est continue mais non différentiable en 0 (la dérivée en 0 est 0). 
Cependant elle offre la possibilité de calculer très rapidement le gradient. On l'utilise souvent comme fonction par défaut avant de tuner le modèle.<br>

$\normalsize Rectified Linear Unit : ReLU(z) = max(0,z)$

Graphique N°8 : [Allure des fonctions d'activations communes](https://miro.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png)

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/activation_functions_plot_08.png" alt="08_active_function.png" style="width:800px;"/>

### 3.3-MLP pour la régression

Les réseaux de neurones peuvent être utilisés pour **modéliser une problématique quantitative**. 

Suivant le nombre de prédiction a réaliser, il faut choisir le bon nombre de neurones en sortie.
La prédiction d'une valeur, par exemple le prix d'un bien immobilier, il faudra un seul neurone en couche de sortie.

Pour la prédiction de coordonnées, par exemple rechercher le centre de coordonnée (x,y) dans une image. Il faudra prédire  2 output donc 2 neurones dans la couche de sortie seront nécessaires.

**De manière générale, il y aura autant de neurone en couche de sortie que de valeur à prédire**

**Tips**💡<br>
Généralement, lors de taches de régression il ne faut pas utiliser de fonction d'activation pour les neurones de sorties.
On préfère ne pas appliquer de transformation afin de laisser la plage de valeur libre.
  
Cependant, il peut être utile dans certains cas de contraindre la plage des valeurs avec une fonction d'activation.<br>
Par exemple, **la prédiction du revenu d'un client qui ne peut être négatif**.<br>
Dans ce cas, il est possible d'utiliser la fonction $softplus(z) = log(1+exp(z))$ qui permet de borner les valeurs entre ]0,infi[.<br>
<br>
Pour la fonction de perte vous pouvez utiliser ce que vous avez l'habitude d'utiliser **MAE,RMSE, Huber Loss**

Tableau N°1:  💡Tips pour la régression

| Hyperparamètre              | Tips sur les valeurs                                                      |
|-----------------------------|---------------------------------------------------------------------------|
| Input neurones              | Un par feature + le biais                                                 |
| Couche cachée               | Généralement entre 1 et 5                                                 |
| Neurones par couche cachées | Généralement entre 10 et 100                                              |
| Output neurones             | 1 par dimension à prédire                                                 |
| Fonction activation caché   | ReLU ou SELU                                                              |
| Fonction activation output  | Libre, ReLU, softplus(positive output),<br>logistic/tanh(variable bornée) |
| Fonction de perte           | MSE, MAE, Huber(moins sensible outliers)                                  |

### 3.4-MLP pour la classification

Bien évidemment, on peut utiliser les réseaux de neurones pour des problématiques de classification.<br>
Si on souhaite réaliser une classification binaire, il suffit d'utiliser une fonction d'activation logistique dans l'output neurone.
En sortie, le neurone produira un résultat entre 0 et 1 qui pourra être interprété comme une probabilité d'appartenance à classe positive de la classification.

Globalement, le modèle MLP permet de traiter toutes les taches de classification :

- **Multilabel binary classification**, il suffit de mettre une couche de sortie avec 2 neurones contenant chacun une fonction d'activation logistique.<br>
<br>
 
- **Multiclass classification** pour les problématiques de classification à plusieurs classes indépendantes à prédire. Type le jeu de données *MNIST* pour la prédiction de chiffre manuscrit.

> Quand vous prédisez des probabilités en sortie. Les fonctions de pertes log loss et cross-entropy sont une bonne idée.

Tableau N°2 : 💡 Tips pour la classification
    
| Hyperparamètre           	| Classification binaire       	| Multilabel binary classification    	| Multiclass classification    	|
|--------------------------	|------------------------------	|------------------------------	|------------------------------	|
| Input neurons            	| Nombre de features + biais   	| Nombre de features + biais   	| Nombre de features + biais   	|
| Hidden layer             	| Généralement entre 1 et 5    	| Généralement entre 1 et 5    	| Généralement entre 1 et 5    	|
| Neurons per hidden layer 	| Généralement entre 10 et 100 	| Généralement entre 10 et 100 	| Généralement entre 10 et 100 	|
| output neurons           	| 1                            	| 1 par label                  	| 1 par classe                 	|
| Output layer activation  	| Logistic                     	| Logistic                     	| Softmax                      	|
| Loss function            	| Cross entropy                	| Cross entropy                	| Cross entropy                	|

## 4-MPL avec Keras et Tensorflow
![logo keras tensor](https://lesdieuxducode.com/images/blog/titleimages/keras-tensorflow-logo.jpg)

### 4.1-Tensorflow

TensorFlow est un outil open source d'apprentissage automatique développé par Google. Le code source a été ouvert le 9 novembre 2015 par Google et publié sous licence Apache(open source).<br>

Il possède aujourd'hui des interfaces en **Python**, Julia et **R**.<br>

Tensorflow a été développé initialement pour être une librairie facilitant la manipulation de tenseur (tableau à N dimensions). Elle permet de réaliser des calculs numériques et l'apprentissage machine à grande échelle en utilisant pleinement les CPU et GPU.<br>
Tensorflow(TF) est utilisable avec python en utilisant son API qui en arrière-plan traite les opérations mathématiques en C++.

TF est un langage à très basse abstraction et permet de définir très finement l'architecture de ses réseaux de neurones mais il implique une très bonne maîtrise de son langage et peut être difficile à prendre en main.<br>

C'est pourquoi on l'utilise souvent avec keras qui est également une API permettant de faire du deep learning mais avec un langage plus simple à utiliser.<br>

### 4.2-Keras

Keras a été développé par un ingénieur francais *Francois Chollet*. Son projet est très rapidement devenu la librairie la plus utilisée des réseaux de neurones car elle offre une fléxibilité et une facilité non disponible dans les autres frameworks.

Keras utilise d'ailleurs tensorflow en arrière-plan pour réaliser les calcules mathématiques. 
Elle est généralement utilisée comme interface entre le code saisie par le data scientist et tensorflow en backend réalisant les calculs.

> Une alternative developée par Facebook *Pytorch* est également possible. Mais nous utiliserons Keras + Tensorflow ici

> Afin de vous entrainer et voir comment intéragir avec les hyperparamètres du réseau de neurones, vous pouvez utiliser **[tensorflow en mode sandbox](https://playground.tensorflow.org/)**
### 4.3-Comment utiliser Keras

**Mettre à jour sa distribution Anaconda et installer tensorflow**

```python
!pip install tensorflow==2.11.0
```

```python
#Import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
```

### 4.4-Classification avec Keras

Reprenons le jeu de données **MNIST** utilisé pour le chapitre 1 SVM.<br>
L'application sera cette fois faite avec un réseau de neurones pour la classification.<br>

Ce type de modèle est d'ailleurs recommandé pour les données non structurées.

Rappel :
C'est un jeu de données célèbre qui comprend 70 000 images de chiffre écrit à la main.

Le jeu de données dispo sous keras est représenté en 28*28 pixels, il faut le transformer en array 1D et 784 features pour le passer dans un modèle.<br>


#### 4.4.1-Implémentation avec keras

```python
#Get data from keras datasets
mnist = keras.datasets.mnist
#Train and test set
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

#Pas de validation set, déja fait.
#On divise pas 255 pour ramener dans l'interval [0,1] et faciliter la descente de gradient
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
```

**Dimension de la table X_train**
```python 
X_train_full.shape
>>(60000, 28, 28)
```

**Créer un modèle en utilisant l'API séquentiel**

```python
#Définir un model séquentiel
model = keras.models.Sequential()
#Ajouter les couches dans le modèle
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
#Couche de sortie, action softmax car multiclass
model.add(keras.layers.Dense(10, activation="softmax"))
```

**1-)** Def modèle <br>
<br>
La première ligne permet de créer un modèle de réseau de neurones.<br>
Le *Séquential* permet de créer un modèle que nous allons définir séquentiellement.<br>
<br>
**2-)** Ajouter des couches manuellement :
- layers.Flatten permet de changer la dimension des données pour l'intégrer comme features
- layers.dense permet de créer une couche de neurones avec un nombre de neurones et une fonction d'activation
- Il faut penser à toujours bien définir la couche de sortie en fonction de sa problématique

Il existe des manières alternatives pour coder avec keras. A la manière de sklearn qui propose une **fonction *pipeline***, on peut 
imbriquer le code<br>
```python
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

```

**Afficher la structure du modèle**

La module *summary* permet d'afficher le type de modèle utilisé ainsi que ses paramètres.<br>
```python 

model.summary() 

#Output
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_2 (Flatten)         (None, 784)               0         
                                                                 
 dense_5 (Dense)             (None, 300)               235500    
                                                                 
 dense_6 (Dense)             (None, 100)               30100     
                                                                 
 dense_7 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
```

**Remarque:** 💡

Un modèle basé sur les réseaux de neurones possèdent beaucoup **plus de** que des **modèles** de **machine learning** classique.<br>
Cela permet d'apprendre sur des schémas de données plus complexes mais cela augmente aussi le risque **d'overfitting**.<br>

Tous les paramètres du modèle sont disponibles dans 

```python
#Obtenir le poids pour les paramètres et biais
w, biais = model.layers[1].get_weights()
#Print weights
print(w)
>>array([[ 0.02804729,  0.06513095,  0.06868796, ...,  0.07075047,
>>        -0.0200577 , -0.00675891],
>>       [-0.06487748,  0.03767272,  0.01864241, ...,  0.05238518,
>>         0.0088407 , -0.07322135],
>>       [ 0.06850776, -0.05723136,  0.03121593, ...,  0.04858249,
>>        -0.03271103,  0.06832977],...
#Format de sortie
print(w.shape)
>>(300,784)
```


**Remarque :** 💡<br>
Les poids sont initialisés à défaut de façon aléatoire. Si tous les poids sont initialisés à 0 la mise à jour des poids serait la même pour chaque connexion<br>

Il est possible d'initialiser les poids avec des techniques plus spécifiques que nous ne présenterons pas ici mais vous pouvez creuser la question [ici](https://keras.io/api/layers/initializers/)

Pensez à spécifier la taille de votre input avec la fonction input_shape pour l'initialisation du modèle. Le nombre de poids dépend directement du nombre d'input! Keras le calculera automatiquement si vous ne le spécifiez pas mais il peut se tromper. 

#### 4.4.2-Compiler le modèle

Une fois que le modèle est défini, vous devez le compiler en définissant la fonction de perte la/les métriques que vous souhaitez utiliser pour estimer la qualité de votre modèle.

Implémentation en python:

```python
#Définir fonction de perte
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"]
             )
```

**Loss**

Cross entropy pour la classification est une mesure qui provient de la théorie de l'information.

Equation N°7 : Cross-entropy ou log loss <br>

$\normalsize H_p(q) = -\frac{1}{N}\sum_{i=0}^N y_i.log(p(y_i))+(1-y_i).log(1-p(y_i))$

où:

- $N:$ Taille de l'échantillon
- $y_i$ : [0,1] données binaire pour classification
- $p_i$ : probabilité du ième individu

> Une cross entropy proche de 0 signifie que le modèle classifie parfaitement nos données, à l'inverse plus le modèle aura une cross entropy élevée plus le modèle sera mauvais.

> Pour calculer la cross entropy vous devez obtenir des probabiltiés en couche de sortie! **ATTENTION** à vos d'activations de sortie 

**optimizer**

Ici nous utilisons le *sgd* qui signifie *stochastic gradient descent*.<br>
Il existe beaucoup d'optimiser mais nous utiliserons le *sgd*. Retenez que c'est l'optimiser qui permet de faire la rétropropagation de gradient!
Le learning rate associé est à 0.01, comme valeur par défaut.

#### 4.4.3-Entrainer et évaluer le modèle

Comme pour sklearn, il suffit d'utiliser la fonction .fit sur nos données d'entrainement pour que le modèle s'entraine.

```python 
train_model = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
#ou, si pas de data de validation, on peut spécifier un split sur le % de données
train_model = model.fit(X_train, y_train, epochs=30,  validation_split=0.1)
```

> L'hyperparamètre *validation_data* est optionnel

Pour keras la verbosité du modèle est très intéressante car vous pouvez examiner la qualité de votre modèle durant l'entrainement.

Graphique N°9 : Output keras

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/output_model_keras_09.png" alt="output_model_keras_09.png" style="width:600px;"/>


Définition dans le cadre des réseaux de neurones :

- **Sample** : C'est une observation de notre échantillon (une ligne)<br>
- **Batch size** : C'est un hyperparamètre du modèle qui contrôle le nombre d'observations utilisées par le modèle avant de mettre à jour les poids.
> Habituellement on prend des valeurs de 32,64,128,256
- **Epoch** : C'est un hyerparamètre à définir pour spécifier le nombre de fois où le modèle doit voir l'ensemble des données. 
> Habituellement, on prend des valeurs de 10,100,500,1000

La fonction .fit() retourne un historique de tout l'entrainement du modèle. Vous pouvez accèder à tous les poids qui ont évolué durant l'entrainement du modèle.

```python
#Paramètre du modèle
train_model.params
#Dictionnaire modèle history
train_model.history.keys()
>>dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```
`history.keys()`contient les mesures de la fonction de perte et des extras metrics à chaque fin des epochs.
Cela permet de regarder le qualité du modèle et sur/sous apprentissage possible.


#### 4.4.4-Leaning curve keras

Pour tracer les learning curve, on peut directement s'appuyer sur l'output disponible dans `.history()`.

```python
#Import package
import pandas as pd
import matplotlib.pyplot as plt

#Data to pandas dataframe
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) 
plt.show(    
```
On obtient le graphique suivant

Grapnique N°10 : Learning curve modèle ANN en fonction des Epochs

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/learning_curve_keras_11.png.png" alt="learning_curve_keras_11.png.png" style="width:600px;"/>


# Comment validation curve

> Remarque décalage des données plot loss https://twitter.com/aureliengeron/status/1110839223878184960

**Evaluation du modèle out sample**

Pour finir d'évaluer son modèle réaliser une évaluation du modèle out sample avec la commande 
```python 
train_model.evaluate(x_test, y_test)
```
**Prédire une nouvelle valeur**

```python
train_model.predict_classes(X_new)
```

Vous savez maintenant comment :

- Définir l'architecture d'un modèle de réseau de neurones pour la classification
- Entrainer le modèle
- L'évaluer
- Réaliser une prédiction

## 4.5-Régression avec Keras

L'architecture pour la régression suit le même principe que la classification.<br>
On peut toujours définir le modèle avec l'API séquentiel ou alors utiliser l'API fonctionnelle pour des modèles plus complexes que nous verrons ci-dessous.<br>

Application avec le dataset *california housing* contenant des informations relatives au prix de vente d'une maison en californie.
Ici, on cherche à prédire le prix de vente en centaine de millier de dollars.<br>

```python
#Get package
#On peut toujours faire son pré-processing avec sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Get data
housing = fetch_california_housing()
#Train and holdout
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
#Train data and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
#Scaling features with different size
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```

Pour la création séquentielle du modèle il faut procéder comme pour la classification.<br>

Les **différences majeures** à garder en tête est qu'il faut **seulement 1 neurone en sortie** (prédire une seule valeur).<br>
Ne pas mettre de fonction d'activation dans *l'output layer* sauf si vous cherchez à restreindre la valeur de sortie.<br>
Changer la fonction de perte du modèle (loss function)

**C'est parti!**

```python
#Initialisation du modèle
reg_model = keras.models.Sequential()
#Shape input, number of columns
reg_model.add(keras.layers.Input(shape=8))#input layer
reg_model.add(keras.layers.Dense(500, activation='relu'))#Hidden layer, 500 neurones
reg_model.add(keras.layers.Dense(1))#Output layer

#Compil model parameter
model.compile(loss="mean_squared_error", optimizer="sgd")
#Fit model
history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))
#Validation out sample
mse_test = model.evaluate(X_test, y_test)
# ATTENTION A BIEN FAIRE X[:k] et pas X[k]
X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)

```


Définir le modèle avec une approche séquentielle est très commune et répond à de nombreux usages.<br>
Cependant, il est parfois utile d'utiliser des modèles plus complexes pour répondre à un problème.<br>
C'est pourquoi *KERAS* propose une manière alternative pour définir l'archiecture de son réseau de neurones avec une **API fonctionnelle**.

### 4.5.1-Wide And Deep neural model

Ce type de modèle se définit de manière non séquentielle car il n'est pas une suite directe de couche de neurones. Il possède une architecture plus complexe.<br>
Il cherche à allier les modèles dit *Wide* et *Deep*

On dit qu'un modèle est Wide s'il possède beaucoup de neurones et peu de couche (neurones>500).
Les modèles wide s'adaptent très bien à des données avec un pattern simple à détecter dans un ensemble de données de taille faible ou moyenne.

On dit qu'un modèle est deep s'il possède beaucoup de couches cachées et peu de neurones. (hidden >3)
Les modèles deep se prettent mieux aux données avec des patterns complexes et des jeux de données relativement grand.

Graphique N°11 : Wide and Deep neural model schema

<img src="https://github.com/Roulitoo/cours_iae/blob/master/02_ANN/img/wide_deep_10.png" alt="wide_deep_11.png" style="width:600px;"/>


Pour implementer ce type de modèle avec keras il faut maintenant utiliser l'API fonctionnelle.

```python 
#set seed
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

#Define first layer
input_ = keras.layers.Input(shape=X_train.shape[1:])
#Hidden layer
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
#Concat first layer and output hidden
concat = keras.layers.Concatenate()([input_, hidden2])
#Output layer
output = keras.layers.Dense(1)(concat)
#define model
model = keras.Model(inputs=[input_], outputs=[output])
 
```

```python 
model.compile(loss="mse", 
              optimizer=keras.optimizers.SGD(learning_rate=1e-3)
             )
```

```python 
history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))
```

Description de ce qu'il se passe dans le modèle:

- **1)** Créer l'input layer en spécifiant le nombre de neurones avec le nombre de features<br>
<br>
 
- **2)** Créer un hidden layer avec 30 neurones et une fonction d'activation relu qui sera connecté à l'input
     Noter qu'on le définit comme une fonction avec un argument (input_) c'est pour ca qu'on parle d'API fonctionelle<br>
<br>
 
- **3)** Créer un second hidden layer avec la même architecture<br>
<br>
 
- **4)** On concate les résultats des couches cachées avec l'input neurone avant de passer dans la couche de sortie<br>
<br>
 
- **5)** Créer l'ouput layer avec 1 seul neurone sans fonction d'activation<br>
<br>
 
- **6)** Enregistrer l'architecture dans un modèle

> Si vous voulez un exemple plus complet des réseaux de neurones, wide, deep et wide and deep regarder ce [notebook kaggle](https://www.kaggle.com/code/hkapoor/wide-vs-deep-vs-wide-deep-neural-networks)

## 5-Hyperparameter tuning

Comme préciser plus haut dans le cours les réseaux de neurones offrent une souplesse et une précision peu égaler par les modèles de machine learning. Leur nombre de paramètres permet d'approcher au plus près ce qu'on souhaite modéliser.

Même pour des structures assez simples comme le MLP vous pouvez tuner les hyperparamètres suivants :

- Nb de neurones par couche
- Nb de couches
- Fonction d'activation
- Learning rate
- L'initiliasation des poids 
- ...

De manière générale, plus le **nombre d'hyperparamètres** à tuner **augmentent** plus le risque d'**overfitting** est élevé!

Une manière simple de tuner votre modèle est d'utiliser la même technique qu'en machine learning en définissant vos hyperparamètres avec une plage de valeur (avec un gridsearch par exemple).

**Attention**❗<br>
Les objets de **sklearn et keras ne communiquent pas nativement**. Ils ont des types différents.<br>
Pour utiliser un objet de keras avec sklearn, il faut utiliser un *wrapper* keras vers sklearn.<br>
La démarche est assez simple et keras possède une fonction wrap pour communiquer avec d'autres API.

### Step 1 imbriquer keras dans une fonction

Il faut passer nos fonctions keras dans une unique fonction. Ici, nous créons un modèle que nous devons imbriquer entièrement dans une fonction.<br>

Exemple avec une fonction pour créer un modèle permettant de faire une régression avec un MLP avec en paramètre à passer le nombre de couches, neurones par couche, learning_rate et input_shape.

```python 
def build_model(n_hidden, n_neurons, lr,input_shape=[8]):
    #Define sequential model
    model = keras.models.Sequential()
    #Input shape
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    #Add hidden layer with loop
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    #Ouput layer    
    model.add(keras.layers.Dense(1))
    #Define optimizer
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    #Compile model
    model.compile(loss="mse", optimizer=optimizer)
    return(model)

```


### Step 2 wrap function

Il faut maintenant intégrer notre fonction dans un wrapper pour utiliser la fonction de sklearn *gridsearchCV* avec un modèle provenant de keras.

```python
from scikeras.wrappers import KerasClassifier, KerasRegressor
keras_reg = KerasRegressor(build_model)
```

### Step 3 GridSearch et passage d'hyperparamètre

Pour passer les hyperparamètres de notre grille de paramètrage et notre *scikeras.wrappers*, il y a 2 solutions :

- Passage des paramètres avec suffixe **model__** qui fait référence au modèle définit par notre def fonction
- Passage des paramètres à travers la fonction KerasRegressor qui peut prendre en entrée des hyperparamètres

La solution avec suffixe étant plus lisible en matière de code, nous allons l'éxaminer.

```python
#Import fonction? 
from sklearn.model_selection import GridSearchCV #RandomizedSearchCV marche aussi
#Paramètre grille, model__ en suffixe devant nos arguments de fonction def
param_distribs = {
"model__n_hidden": [1, 2, 3],
"model__n_neurons":[50,100,200],
"model__lr":  [0.0001, 0.001, 0.1]
}
#On peut aussi ajouter le nombre epochs : [30] dans la grille

#grid search
rnd_search_cv = GridSearchCV(keras_reg, param_distribs,cv=2)
#Fit model with grid search
rnd_search_cv.fit(X_train, y_train)

print(rnd_search_cv.best_params_, rnd_search_cv.best_score_)

```

Vous savez maintenant créer un modèle avec keras et tensorflow en backend puis chercher les meilleurs hyperparamètres et après tuner votre modèle.


### Tips hyperparamètre ANN

##### Nombre de couches cachées

Généralement il suffit de 1 ou 2 hidden layer pour traiter votre problème. Il faut mieux utiliser peu de couche cachée et augmenter le nombre de neurones à l'intérieur.
Si vous continuez à avoir des performances faibles augmenter votre nombre de couches tant que vous n'êtes pas en overfitting.

##### Nombre de neurones par couches cachées

Le nombre de neurones en input et output est fixe et dépend du nombre de features et du type de modélisation.
Généralement on utilise 2 types d'approches:

- Technique en pyramide, plus on avance dans les couches cachées plus on diminue le nombre de neurones par couche. Exemple pour 3 couches cachées 300 puis 200 puis 100.

- Actuellement, on fixe souvent le même nombre de neurones sur chaque couche.

Globalement, Vous pouvez augmenter votre nombre de neurones tant que vous n'êtes pas en overfitting

##### Learning rate

C'est probablement l'hyperparamètre le plus influent les modèles.
Una manière de le choisir est de partir d'un niveau faible par exemple 1e-5 et monter jusqu'à 1 par pas de *exp(log(ecart_dizaine)/nb_iteration)*. Ici cela nous donnerait un pas de exp(log(10^5)/500) 

Une fois que vous avez fait ca tracer l'évolution de votre loss en fonction de l'évolution du learning rate et vous verez le point optimal assez facilement.

##### Batch size

Dans la littérature on retrouve un batch_size fixé entre 2 et 192. Il faut tester
Généralement la valeur par défaut est de 32.

##### Fonction activation

On utilise la fonction ReLU comme fonction d'activation par défaut car sa dérivée est simple à calculer pour la descente de gradient. Vous pouvez la changer en fonction des performances de votre modèle.

Pour l'activation de la fonction output cela dépend de ce que vous voulez obtenir à la fin

## Doc utile

- [Scikeras wrapper pour interaction keras et scikit-learn](https://www.adriangb.com/scikeras/refs/heads/master/index.html)
- [Kaggle wide and deep neural model](https://www.kaggle.com/code/hkapoor/wide-vs-deep-vs-wide-deep-neural-networks)
- [Surement le meilleur cours sur le deep learning](https://fr.coursera.org/specializations/deep-learning)
- [Cheat sheet architecture des modèles de réseaux de neurones](https://www.asimovinstitute.org/neural-network-zoo/)
- [Critère de early stopping pour ANN](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/)
- [Vidéo Keras tuner](https://www.youtube.com/watch?v=Un0JDL3i5Hg&t=24s)
- [Batch normalization pour accélèrer vos réseaux de neurones](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)
