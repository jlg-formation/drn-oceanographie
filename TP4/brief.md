# TP4 — Segmentation automatique Terre / Mer

**Spécification fonctionnelle**

- **Projet :** Terre ou Mer ?
- **Contexte :** Formation Deep Learning — SHOM
- **Chapitre :** Réseaux de neurones convolutifs (CNN)
- **Version :** 1.0
- **Statut :** Spécification initiale

---

## 1. Présentation du projet

### 1.1. Contexte

Le Service hydrographique et océanographique de la Marine (SHOM) exploite des données géographiques, hydrographiques et océanographiques pour décrire et cartographier l'environnement maritime.

L'analyse d'images satellites constitue un cas d'usage pertinent pour l'étude du littoral et la distinction entre surfaces terrestres et maritimes.

Ce TP propose de développer un système capable de segmenter automatiquement une image satellite en utilisant un réseau de neurones convolutif (CNN).

### 1.2. Objectif

Le projet consiste à développer deux composants complémentaires :

1. Un environnement d'apprentissage permettant d'entraîner un CNN avec Python et PyTorch sur des images satellites annotées.
2. Une application web interactive permettant d'utiliser le modèle entraîné pour segmenter de nouvelles images satellites.

L'utilisateur final doit pouvoir déposer une image satellite dans l'application, lancer une analyse et visualiser les pixels identifiés comme appartenant à la terre ou à la mer.

### 1.3. Principe de fonctionnement

Le CNN réalise une **segmentation sémantique binaire**.

Pour chaque pixel d'une image, le modèle prédit son appartenance à l'une des deux classes :

- **Terre :** surfaces terrestres.
- **Mer :** surfaces maritimes.

Le résultat est un masque de segmentation ayant les mêmes dimensions que l'image analysée.

Le projet ne vise pas à reconnaître les objets individuels présents dans l'image, comme les navires, bâtiments ou véhicules.

### 1.4. Objectifs pédagogiques

À l'issue du TP, les participants doivent être capables de :

- Comprendre le principe de la segmentation sémantique.
- Préparer des images et des masques d'apprentissage.
- Construire et entraîner un CNN avec PyTorch.
- Comprendre le fonctionnement général d'une architecture U-Net.
- Évaluer un modèle de segmentation.
- Exporter un modèle entraîné.
- Réutiliser ce modèle dans une application web.
- Interpréter les résultats et identifier les limites de la segmentation automatique.

---

## 2. Périmètre fonctionnel

### 2.1. Fonctionnalités incluses

Le projet couvre :

- L'acquisition d'images satellites réelles.
- La génération automatique de masques terre/mer.
- La préparation d'un dataset supervisé.
- L'entraînement du CNN.
- L'évaluation du modèle.
- L'export du modèle entraîné.
- L'import d'images dans une application web.
- L'exécution de la segmentation dans le navigateur.
- La visualisation interactive des résultats.
- L'export des résultats.

### 2.2. Fonctionnalités exclues

La première version ne prévoit pas :

- La reconnaissance d'espèces marines.
- La détection de navires.
- La segmentation de plus de deux classes.
- La reconnaissance des fonds marins.
- L'entraînement du CNN directement dans le navigateur.
- L'analyse automatique de flux vidéo.
- Le traitement de scènes satellites multispectrales arbitraires.
- La mise à jour automatique du modèle à partir des images déposées.

---

## 3. Sources de données

### 3.1. Images satellites

Les images utilisées pour l'apprentissage proviennent de la mission européenne Sentinel-2.

**Source principale :**

https://browser.dataspace.copernicus.eu/

Les données doivent représenter des zones littorales contenant des surfaces terrestres et maritimes.

La zone géographique privilégiée est la Bretagne, notamment la rade de Brest.

Le jeu d'apprentissage doit comporter une diversité de configurations :

- Littoraux rectilignes.
- Côtes découpées.
- Îles et presqu'îles.
- Ports.
- Estuaires.
- Zones majoritairement terrestres.
- Zones majoritairement maritimes.

Les images devront être sélectionnées pour limiter la présence de nuages et les situations où la frontière terre/mer est difficile à identifier.

### 3.2. Données de référence

Les masques de référence sont produits à partir de données géographiques existantes.

**Source principale :**

https://osmdata.openstreetmap.de/data/land-polygons.html

Ces données permettent d'identifier les polygones terrestres.

La génération des masques doit être automatique : aucune annotation manuelle ne doit être nécessaire pour réaliser le TP.

Les masques devront être contrôlés visuellement avant leur utilisation pour l'entraînement.

### 3.3. Convention des masques

Chaque pixel valide possède une étiquette :

| Valeur | Signification |
|---|---|
| 0 | Mer |
| 1 | Terre |
| 255 | Pixel ignoré |

La valeur 255 permet d'exclure les pixels non exploitables ou dont l'annotation est incertaine.

Elle ne constitue pas une troisième classe.

### 3.4. Limites des annotations

Le trait de côte issu des données cartographiques peut différer du trait de côte visible sur une image satellite.

Ces différences peuvent notamment provenir :

- Des marées.
- De l'estran.
- Des variations du niveau de l'eau.
- Des estuaires.
- De la résolution des données.
- Des changements géographiques intervenus entre deux acquisitions.

Le projet doit permettre d'ignorer les pixels dont l'étiquette est incertaine.

---

## 4. Préparation du dataset

### F01 — Importer les images satellites

Le système doit permettre d'intégrer des images satellites sélectionnées pour le projet.

Chaque image doit être associée à des informations de provenance :

- Identifiant.
- Date d'acquisition.
- Zone géographique.
- Dimensions.
- Résolution spatiale.
- Source.

### F02 — Générer les masques de référence

Le système doit produire automatiquement un masque terre/mer pour chaque image utilisée dans l'apprentissage.

**Critères d'acceptation :**

- Le masque possède les mêmes dimensions que l'image.
- Les pixels terrestres et maritimes sont identifiables.
- Les pixels incertains peuvent être ignorés.
- Le masque est spatialement aligné avec l'image.
- Une visualisation de contrôle est disponible.

### F03 — Découper les images en patches

Les images satellites doivent être découpées en petites images exploitables par le CNN.

La dimension de référence est de **256 × 256 pixels**.

Chaque patch image doit posséder un patch masque correspondant.

Le système doit permettre d'inclure des patches contenant une frontière terre/mer, mais aussi des patches entièrement terrestres ou maritimes.

### F04 — Constituer les ensembles d'apprentissage

Le dataset doit être réparti en trois ensembles :

| Ensemble | Fonction |
|---|---|
| Entraînement | Apprentissage des paramètres du CNN |
| Validation | Suivi et sélection du modèle |
| Test | Évaluation finale sur des données inconnues |

La séparation doit éviter que des images géographiquement voisines ou quasi identiques apparaissent dans plusieurs ensembles.

Les ensembles doivent être reproductibles.

### F05 — Visualiser le dataset

Le système doit permettre de consulter des exemples du dataset.

Pour chaque exemple, l'utilisateur doit pouvoir observer :

- L'image satellite.
- Le masque de référence.
- Leur superposition.

L'objectif est de détecter les erreurs de préparation avant l'entraînement.

---

## 5. Entraînement du CNN

### F06 — Entraîner le modèle

Le système doit permettre d'entraîner un CNN de segmentation avec Python et PyTorch.

L'architecture de référence est un **U-Net compact**.

Le modèle reçoit une image RGB et produit une probabilité d'appartenance à la classe Terre pour chaque pixel.

L'entraînement doit pouvoir fonctionner sur CPU. L'utilisation d'un GPU constitue une accélération facultative.

### F07 — Paramétrer l'entraînement

L'utilisateur doit pouvoir définir les principaux paramètres :

- Nombre d'époques.
- Taille des batches.
- Taux d'apprentissage.
- Répertoire des données.
- Répertoire des résultats.
- Graine aléatoire.

Des valeurs par défaut adaptées au TP doivent être fournies.

### F08 — Suivre l'apprentissage

Pendant l'entraînement, le système doit afficher ou enregistrer :

- La perte d'entraînement.
- La perte de validation.
- Les métriques de validation.
- L'époque courante.
- La durée de l'entraînement.

À la fin, des courbes doivent permettre d'observer l'évolution de l'apprentissage.

### F09 — Sauvegarder le modèle

Le système doit conserver le meilleur modèle obtenu pendant l'entraînement, selon une métrique de validation définie.

Le modèle sauvegardé doit pouvoir être rechargé sans nouvel entraînement.

Les paramètres utilisés et les résultats associés doivent être conservés.

---

## 6. Évaluation du modèle

### F10 — Évaluer sur des images inconnues

Le système doit pouvoir appliquer le CNN sur les images de l'ensemble de test.

Ces images ne doivent pas avoir servi à l'entraînement ni à la sélection du modèle.

### F11 — Calculer les métriques

Les métriques attendues sont :

- Intersection over Union (IoU).
- Coefficient Dice.
- Précision.
- Rappel.
- Matrice de confusion.

Les pixels ignorés ne doivent pas intervenir dans les calculs.

Les métriques doivent être calculées pour les deux classes lorsque cela est pertinent.

### F12 — Visualiser les erreurs

Pour chaque image de test sélectionnée, le système doit pouvoir afficher :

- L'image originale.
- Le masque de référence.
- Le masque prédit.
- Les pixels mal classés.

Cette fonctionnalité doit permettre d'identifier les situations dans lesquelles le CNN rencontre des difficultés.

### F13 — Exporter le modèle pour le web

Le modèle entraîné doit être exportable au format ONNX.

Le modèle exporté doit produire des résultats cohérents avec ceux du modèle PyTorch sur les mêmes données d'entrée.

Le modèle ONNX est destiné à être utilisé par l'application web.

---

## 7. Application web — « Terre ou Mer ? »

### 7.1. Objectif

L'application web constitue la démonstration interactive du TP.

Elle doit permettre à un utilisateur ne connaissant pas PyTorch de tester le modèle entraîné.

L'expérience doit être simple, visuelle et ludique.

Le parcours principal doit tenir en trois actions :

1. Choisir une image satellite.
2. Cliquer sur **Analyser**.
3. Explorer le résultat.

### 7.2. Page principale

La page principale doit présenter :

- Le titre « Terre ou Mer ? ».
- Une courte explication du principe de segmentation.
- Une zone d'import d'image.
- Une galerie d'images satellites d'exemple.
- Un bouton d'analyse.
- Une zone de visualisation des résultats.

L'interface doit être utilisable sur ordinateur et s'adapter aux écrans plus petits.

### F14 — Choisir une image d'exemple

L'application doit proposer plusieurs images satellites prédéfinies.

Ces images doivent représenter différentes configurations littorales.

L'utilisateur peut sélectionner un exemple sans devoir télécharger de fichier.

Au moins une image d'exemple doit provenir d'une zone géographique non utilisée pour entraîner le CNN.

### F15 — Importer une image

L'utilisateur doit pouvoir importer une image depuis son ordinateur.

Formats acceptés :

- PNG.
- JPEG.

L'import doit fonctionner par sélection de fichier et par glisser-déposer.

L'application doit afficher un aperçu de l'image avant analyse.

Elle doit signaler les fichiers non pris en charge et les images dépassant les limites de taille définies.

**Restriction importante :** le modèle est conçu pour des images satellites RGB comparables aux données d'entraînement. Une photographie ordinaire ou une image fortement transformée peut produire des résultats non fiables.

### F16 — Lancer la segmentation

Un bouton **Analyser l'image** doit lancer la prédiction.

L'analyse doit être effectuée dans le navigateur à partir du modèle ONNX exporté.

Aucun serveur Python ne doit être nécessaire pour effectuer la prédiction.

Pendant le traitement, l'application doit afficher un indicateur de progression ou d'activité.

Une fois l'analyse terminée, les résultats doivent apparaître automatiquement.

### F17 — Segmenter une image entière

L'utilisateur ne doit pas être obligé de découper manuellement son image en patches.

L'application doit prendre en charge le traitement des images dont les dimensions dépassent celles attendues par le CNN.

Le résultat final doit conserver les dimensions de l'image originale.

La reconstruction ne doit pas laisser de zones non traitées.

### F18 — Afficher le masque de segmentation

L'application doit afficher un masque représentant les deux classes prédites.

Convention visuelle :

- **Terre :** vert.
- **Mer :** bleu.

Le masque doit être spatialement aligné avec l'image originale.

### F19 — Choisir le mode d'affichage

L'utilisateur doit pouvoir basculer entre trois modes :

**Mode Image originale**

Affiche uniquement l'image satellite.

**Mode Segmentation**

Affiche uniquement le masque terre/mer prédit.

**Mode Superposition**

Affiche le masque coloré par-dessus l'image satellite.

### F20 — Régler l'opacité

En mode Superposition, l'utilisateur doit pouvoir régler l'opacité du masque.

Le réglage doit être immédiat et ne doit pas nécessiter de nouvelle prédiction.

### F21 — Comparer avant et après

L'application doit proposer un comparateur interactif.

Un curseur vertical déplaçable horizontalement permet de révéler :

- À gauche : l'image originale.
- À droite : l'image segmentée.

Le déplacement du curseur doit être fluide.

Cette fonctionnalité constitue l'un des éléments principaux de l'expérience ludique.

### F22 — Afficher des statistiques

Après analyse, l'application doit afficher :

- Le pourcentage de pixels classés Terre.
- Le pourcentage de pixels classés Mer.
- La durée de l'inférence.

Les deux pourcentages doivent totaliser 100 % pour les pixels effectivement classés.

Ces valeurs décrivent la répartition des pixels dans l'image, et non nécessairement des surfaces géographiques réelles.

### F23 — Afficher les incertitudes

L'application doit pouvoir proposer un mode de visualisation des pixels pour lesquels la prédiction est incertaine.

Les zones où la probabilité de classification est proche de 50 % peuvent être mises en évidence.

Cette fonctionnalité est pédagogique : elle permet d'observer les limites du modèle.

Les probabilités produites ne doivent pas être présentées comme des garanties de justesse.

### F24 — Télécharger les résultats

L'utilisateur doit pouvoir télécharger :

- Le masque de segmentation en PNG.
- L'image originale avec la segmentation superposée en PNG.

Les fichiers exportés doivent conserver les dimensions de l'image analysée.

### F25 — Recommencer une analyse

L'utilisateur doit pouvoir sélectionner une autre image et relancer l'analyse sans recharger la page.

Les résultats précédents doivent être remplacés ou réinitialisés de manière explicite.

---

## 8. Visualisation cartographique

### F26 — Afficher les images géoréférencées sur une carte

Pour les exemples satellites disposant d'informations géographiques, l'application peut proposer une visualisation cartographique.

Cette fonctionnalité permet de :

- Localiser l'image sur une carte.
- Afficher l'image satellite.
- Superposer le masque prédit.
- Activer ou désactiver les couches.

La visualisation cartographique est facultative pour la première version.

Une image PNG ou JPEG importée sans géoréférencement doit pouvoir être analysée sans carte.

---

## 9. Exigences d'ergonomie

### 9.1. Simplicité

L'application doit pouvoir être utilisée sans connaissance préalable du deep learning.

Les actions principales doivent être immédiatement identifiables.

### 9.2. Retour utilisateur

L'application doit fournir un retour clair pour les événements suivants :

- Image chargée.
- Modèle en cours de chargement.
- Analyse en cours.
- Analyse terminée.
- Erreur de chargement.
- Erreur de traitement.

### 9.3. Visualisation

La comparaison entre image originale et segmentation doit être le point central de l'interface.

Les couleurs doivent être cohérentes dans toute l'application.

### 9.4. Valeur pédagogique

L'interface doit expliquer brièvement :

- Ce qu'est une segmentation sémantique.
- Ce que représentent les deux classes.
- Le rôle du CNN.
- Pourquoi certaines prédictions peuvent être incorrectes.

L'application doit distinguer clairement une prédiction d'une vérité terrain.

---

## 10. Exigences non fonctionnelles

### NF01 — Exécution locale

Le pipeline d'entraînement doit pouvoir fonctionner localement sur un ordinateur équipé de Python.

### NF02 — Compatibilité CPU

Une configuration d'entraînement réduite doit être prévue pour les machines ne disposant pas de GPU adapté.

### NF03 — Inférence dans le navigateur

L'application web doit exécuter le modèle localement dans le navigateur.

Les images importées ne doivent pas être envoyées à un serveur pour effectuer la segmentation.

### NF04 — Déploiement statique

L'application doit pouvoir être hébergée sur un service de pages web statiques.

### NF05 — Reproductibilité

La préparation des données et l'entraînement doivent être documentés et reproductibles.

Les versions des dépendances et les paramètres d'entraînement doivent être conservés.

### NF06 — Traçabilité

Chaque modèle entraîné doit pouvoir être associé :

- Au dataset utilisé.
- Aux paramètres d'entraînement.
- Aux métriques obtenues.
- À sa date de création.

### NF07 — Robustesse

Les erreurs de fichier, de chargement du modèle ou d'inférence doivent être signalées sans provoquer de blocage définitif de l'interface.

### NF08 — Performances

L'application doit rester réactive pour les images d'exemple fournies.

La durée de traitement dépendra du matériel, du navigateur et de la taille de l'image.

Des limites de dimensions et de consommation mémoire devront être définies et affichées.

---

## 11. Scénarios de recette fonctionnelle

### Scénario 1 — Entraînement

**Étant donné** un dataset préparé et valide,

**Quand** l'utilisateur lance l'entraînement,

**Alors** le système entraîne le CNN, enregistre les métriques et sauvegarde le meilleur modèle.

### Scénario 2 — Export ONNX

**Étant donné** un modèle PyTorch entraîné,

**Quand** l'utilisateur lance l'export,

**Alors** un modèle ONNX exploitable par l'application web est généré et validé.

### Scénario 3 — Analyse d'une image d'exemple

**Étant donné** que l'application est ouverte,

**Quand** l'utilisateur sélectionne une image d'exemple et clique sur « Analyser l'image »,

**Alors** l'application affiche un masque terre/mer aligné avec l'image.

### Scénario 4 — Import d'une image personnelle

**Étant donné** que l'utilisateur dispose d'une image satellite PNG,

**Quand** il dépose le fichier dans la zone d'import,

**Alors** l'application affiche un aperçu et permet de lancer l'analyse.

### Scénario 5 — Comparaison interactive

**Étant donné** qu'une image a été analysée,

**Quand** l'utilisateur déplace le curseur de comparaison,

**Alors** il peut révéler progressivement l'image originale et le résultat segmenté.

### Scénario 6 — Changement d'opacité

**Étant donné** qu'une segmentation est affichée,

**Quand** l'utilisateur modifie l'opacité,

**Alors** la superposition est actualisée immédiatement, sans nouvelle inférence.

### Scénario 7 — Téléchargement du résultat

**Étant donné** qu'une segmentation est disponible,

**Quand** l'utilisateur clique sur « Télécharger le masque »,

**Alors** un fichier PNG représentant le masque prédit est téléchargé.

### Scénario 8 — Fichier invalide

**Étant donné** que l'utilisateur tente d'importer un fichier non pris en charge,

**Quand** il sélectionne ce fichier,

**Alors** l'application affiche un message explicite et reste utilisable.

---

## 12. Critères de réussite du TP

Le TP est considéré comme réussi lorsque :

1. Un dataset d'images satellites et de masques terre/mer a été constitué sans annotation manuelle.
2. Un CNN U-Net a été entraîné avec PyTorch.
3. Le modèle a été évalué sur des images inconnues.
4. Les métriques et les résultats visuels sont disponibles.
5. Le modèle a été exporté au format ONNX.
6. Une application web permet de sélectionner ou importer une image satellite.
7. Le modèle effectue la segmentation directement dans le navigateur.
8. Le résultat est visualisable en masque, en superposition et dans un comparateur avant/après.
9. L'utilisateur peut télécharger le masque prédit.
10. Les limites et les erreurs du modèle peuvent être discutées à partir des résultats.

Aucun score de précision minimal n'est imposé comme condition de réussite pédagogique. L'objectif est de comprendre et de mettre en œuvre l'ensemble du processus.

---

## 13. Livrables attendus

### Livrable A — Apprentissage Python

- Pipeline de préparation des données.
- Dataset d'entraînement, validation et test.
- Modèle U-Net PyTorch.
- Script