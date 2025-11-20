## Installer UV

## Step 1

En ligne de commande puis Git bash vous pouvez lancer :

```bash

curl --ssl-no-revoke-LsSfhttps://astral.sh/uv/install.sh | sh

```

Si cela ne fonctionne ou que vous avez une erreur du type `schannel: next InitializeSecurityContext failed: Unknown error (0x80092012) - La fonction de révocation n'a pas pu vérifier la révocation du certificat.`

Vous pouvez utiliser directement windows powershell

```bash

powershell -ExecutionPolicyByPass-c"irm https://astral.sh/uv/install.ps1 | iex"

```

## Step 2

Vérifier en ligne de commande si **UV** est bien installé et si vous y avez accès en variable d'environnement.

```bash

uv --version

```

Vous devriez obtenir `uv 0.9.10 (44f5a14f4 2025-11-17)`

## Step 3

Installer une version de python 3.11

```bash

uv pythoninstall3.11

```

## Step 4

Se placer dans le répertoire contenant un fichier pyproject.toml et uv.lock

```bash

cd <racine_projet>/cours_iae/01_SVM

```

## Step 5

Créer un environnement virtuel et syncrhonisé avec ma version de package

```

uv sync

```

## Step 6 ajouter un autre package

```bash

uv add <package_name>

# Exemple uv add pandas

```

## Step 7 choisir la version de python disponible dans votre venv

```bash

#Mettre son repertoire courant dans le bon dossier

cd ./cours_iae/01_SVM/

# Fixer le repertoire courant pour vscode

code .

```

Alternative, executer depuis l'environnement virtuel jupyterlab

```bash

uv 

```
