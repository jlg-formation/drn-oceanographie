# Comprendre le fichier SHOM téléchargé

Ce document explique **en mots simples** ce que contient le fichier de données
océanographiques téléchargé depuis le SHOM (Service Hydrographique et
Océanographique de la Marine). Il est destiné à un débutant en océanographie.

---

## 1. D'où vient ce fichier ?

Le fichier est produit par un **modèle numérique de l'océan** appelé
**HYCOM 3D-SURF**, opéré par le SHOM. Un modèle océanographique, c'est un peu
comme un modèle météo : au lieu de prévoir le temps dans l'atmosphère, il
**simule et prévoit l'état de la mer** (température, salinité, courants...).

Le nom du fichier se décode ainsi :

```
HYDRODYN-SURF _ HYCOM3D-SURF _ R1000_MANGASC _ 20260911
   |                |               |              |
 hydrodynamique   modèle utilisé   zone géo.     date de la donnée
 de surface       (HYCOM 3D)       Manche-Gascogne (11 sept. 2026)
```

- **HYDRODYN-SURF** : données d'*hydrodynamique de surface* (le mouvement et
  l'état de l'eau tout en haut de l'océan).
- **HYCOM3D-SURF** : le modèle qui a calculé les données.
- **MANGASC** : la zone couverte, **Manche–Gascogne** (le Golfe de Gascogne,
  entre la France et l'Espagne, plus l'entrée de la Manche).
- **20260911** : la journée décrite, le **11 septembre 2026**.

---

## 2. Que contenait l'archive `.zip` ?

Le téléchargement est une archive contenant 3 fichiers :

| Fichier | Rôle |
|---------|------|
| `...nc.gz` | Les **données** elles-mêmes, dans un fichier **NetCDF** compressé (gzip). C'est le cœur du jeu de données (~68 Mo décompressé). |
| `...xml`   | Les **métadonnées** normalisées (norme ISO) : qui a produit la donnée, quand, comment, licence, etc. |
| `...gml`   | Une **description géographique** (norme INSPIRE) : liste les variables et l'emprise sous forme standardisée. |

Le fichier important est le **`.nc` (NetCDF)**.

---

## 3. Qu'est-ce qu'un fichier NetCDF ?

**NetCDF** (*Network Common Data Form*) est un format de fichier très utilisé en
sciences de la Terre (océan, atmosphère, climat). On peut le voir comme un
**tableau à plusieurs dimensions** rangé dans un seul fichier, accompagné
d'étiquettes qui expliquent chaque chose (unités, coordonnées, description).

Ici, les données sont organisées sur une **grille**, comme un immense damier
posé sur la mer. Chaque case du damier contient des valeurs (température,
salinité, courant), et on a une image du damier **pour chaque heure** de la
journée.

---

## 4. La grille : où et quand ?

Le fichier décrit une zone sous forme d'une grille régulière :

| Caractéristique | Valeur | En clair |
|-----------------|--------|----------|
| **Longitude** | −15,0° à +2,98° | d'ouest en est, de l'Atlantique au large jusqu'à la côte |
| **Latitude**  | 43,01° à 51,0°  | du sud (Espagne) au nord (Manche) |
| **Taille grille** | 720 × 471 points | ≈ 339 000 cases |
| **Résolution** | ≈ 0,02° soit **~1,9 km** | une case fait environ 2 km de côté |
| **Profondeur** | 0 m (1 seul niveau) | uniquement la **surface** de la mer |
| **Temps** | 24 pas de temps | une valeur **par heure**, sur 24 h du 11/09/2026 |

> En résumé : pour **chaque heure** de la journée et **chaque case de 2 km**, le
> fichier donne l'état de la surface de la mer.

---

## 5. Les variables : que mesure-t-on ?

Le fichier contient **4 grandeurs physiques** de la surface de l'océan :

| Variable | Nom | Unité | Ce que c'est | Valeurs observées |
|----------|-----|-------|--------------|-------------------|
| `temperature` | Température de l'eau | °C | Chaud ou froid en surface | 13,5 à 23,6 °C |
| `salinity` | Salinité | ‰ (pour mille) | Quantité de sel dissous dans l'eau | ~19 à 35,9 (typique océan ≈ 35) |
| `u` | Courant vers l'**Est** | m/s | Composante *horizontale* du courant (axe est-ouest) | −4,3 à +4,5 m/s |
| `v` | Courant vers le **Nord** | m/s | Composante *horizontale* du courant (axe nord-sud) | −3,3 à +4,0 m/s |

### À propos du courant (`u` et `v`)

Le courant marin est une **flèche** (un vecteur) qui indique dans quelle
direction et à quelle vitesse l'eau se déplace. Pour la ranger dans un tableau,
on la décompose en deux nombres :

- **`u`** = vitesse vers l'est (positif = vers l'est, négatif = vers l'ouest) ;
- **`v`** = vitesse vers le nord (positif = vers le nord, négatif = vers le sud).

En combinant `u` et `v`, on retrouve la **vitesse totale** et la **direction**
du courant à chaque endroit.

### Salinité en « ‰ »

La salinité se lit « pour mille » : une valeur de 35 signifie environ 35 grammes
de sel par kilogramme d'eau de mer. C'est la salinité typique de l'océan. Des
valeurs plus basses apparaissent près des côtes et des embouchures de fleuves,
où l'eau douce dilue le sel.

---

## 6. Les zones sans données (« masque terre »)

Environ **71 % des cases** contiennent une valeur ; les autres ~29 % sont vides.
C'est normal : la grille est un rectangle, mais une partie recouvre la **terre**
(côtes, Bretagne, Espagne...) où il n'y a pas d'eau à décrire. Ces cases sont
« masquées » (marquées comme non valides).

---

## 7. En résumé

Ce fichier est une **prévision océanographique de surface** du Golfe de
Gascogne pour le 11 septembre 2026. Il fournit, **heure par heure** et sur une
grille d'environ **2 km**, quatre informations sur la surface de la mer :

- sa **température**,
- sa **salinité**,
- et le **courant** (via ses composantes est `u` et nord `v`).

C'est le type de donnée utilisé pour la navigation, la prévision de dérive
(objets, pollutions), la pêche, ou l'étude du milieu marin.

---

## 8. Explorer le fichier soi-même

Le script [explore.py](explore.py) ouvre le fichier et affiche toutes ces
informations automatiquement :

```powershell
uv run python TP2/explore.py
```
