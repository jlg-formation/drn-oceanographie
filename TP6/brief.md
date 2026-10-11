# TP6 — Génération de reliefs sous-marins avec VAE et GAN

**Spécification fonctionnelle — Version 1.0**

**Formation :** Deep Learning et réseaux de neurones — SHOM  
**Thème :** Modèles génératifs, autoencodeurs variationnels (VAE) et réseaux antagonistes génératifs (GAN)  
**Application :** Ocean Terrain Lab  
**Statut :** Proposition de spécification

---

## 1. Présentation du projet

### 1.1. Contexte

Les données bathymétriques décrivent la profondeur et la morphologie des fonds marins.

Le projet utilise des données bathymétriques réelles issues de GEBCO pour entraîner deux familles de modèles génératifs :

- Un **autoencodeur variationnel (VAE)**.
- Un **réseau antagoniste génératif (GAN)**.

Ces modèles doivent apprendre les caractéristiques statistiques de reliefs sous-marins afin de produire de nouvelles cartes bathymétriques synthétiques.

Le TP doit également permettre d'observer les différences fondamentales entre les deux approches.

### 1.2. Objectif général

Le projet comprend deux composants complémentaires.

**Composant A — Python / PyTorch**

Un environnement permettant de préparer les données, d'entraîner un VAE et un GAN, de suivre leur apprentissage, de générer des reliefs et de comparer leurs résultats.

**Composant B — Application web**

Une application interactive permettant de générer des reliefs synthétiques, de les visualiser en 2D et en 3D, de manipuler les espaces latents et de comparer les deux modèles.

Les modèles sont entraînés en Python, puis exploités dans le navigateur sans serveur d'inférence Python.

### 1.3. Objectifs pédagogiques

À l'issue du TP, les participants doivent être capables de :

- Comprendre le principe d'un modèle génératif.
- Préparer des données bathymétriques pour l'apprentissage.
- Comprendre le rôle de l'espace latent.
- Construire et entraîner un VAE avec PyTorch.
- Comprendre le principe de l'apprentissage antagoniste.
- Construire et entraîner un GAN avec PyTorch.
- Observer les différences de stabilité entre les entraînements.
- Générer de nouveaux exemples à partir de vecteurs latents.
- Reconstruire des données réelles avec un VAE.
- Comparer les résultats des deux architectures.
- Exporter des modèles entraînés et les exploiter dans une application web.

---

## 2. Périmètre fonctionnel

### 2.1. Cas d'usage

Le système doit générer des cartes synthétiques représentant des reliefs sous-marins.

Chaque carte est une grille régulière de valeurs bathymétriques.

La dimension de référence est de **64 × 64 pixels**.

Les reliefs peuvent représenter notamment :

- Des plaines abyssales.
- Des monts sous-marins.
- Des dorsales.
- Des vallées.
- Des canyons sous-marins.
- Des reliefs complexes.

Les données générées sont expérimentales et ne doivent pas être utilisées comme des mesures hydrographiques réelles.

### 2.2. Modèles étudiés

| Modèle | Fonction principale |
|---|---|
| VAE | Reconstruire et générer des reliefs |
| GAN | Générer des reliefs synthétiques |
| Comparaison | Étudier qualité, diversité et stabilité |

### 2.3. Fonctionnalités exclues

La première version ne prévoit pas :

- La génération de cartes bathymétriques destinées à la navigation.
- La production de données hydrographiques certifiées.
- La génération conditionnée par des coordonnées géographiques.
- La simulation physique de l'évolution des fonds marins.
- L'entraînement dans le navigateur.
- La génération de grandes cartes bathymétriques continues.
- L'utilisation de modèles de diffusion.

---

## 3. Données bathymétriques

### F01 — Acquisition des données

Le système doit utiliser des données bathymétriques réelles provenant de GEBCO.

**Source :** https://www.gebco.net/

Les données doivent permettre d'extraire des zones océaniques présentant des morphologies variées.

Les régions sélectionnées doivent être documentées et reproductibles.

### F02 — Sélection des zones géographiques

Le système doit permettre de sélectionner plusieurs régions d'intérêt.

Le jeu de données doit contenir une diversité suffisante de reliefs.

Il doit éviter d'être constitué principalement de surfaces presque plates.

Les zones contenant des terres émergées doivent être exclues ou traitées explicitement.

### F03 — Découpage en patches

Les données doivent être découpées en patches de 64 × 64 pixels.

Chaque patch doit contenir une grille bathymétrique cohérente.

Le système doit enregistrer les informations permettant de retrouver la zone d'origine de chaque patch.

### F04 — Normalisation

Les valeurs bathymétriques doivent être transformées en valeurs numériques adaptées à l'entraînement.

La transformation doit être cohérente entre :

- Les données d'entraînement.
- Les données de validation.
- Les données de test.
- Les données utilisées dans l'application web.

La méthode de normalisation et ses paramètres doivent être conservés.

L'application doit pouvoir afficher les valeurs générées dans une échelle physique interprétable, lorsque cette conversion est justifiée.

### F05 — Constitution du dataset

Le dataset de référence doit comporter environ **5 000 patches** pour la configuration pédagogique initiale.

Il doit être réparti en ensembles d'entraînement, de validation et de test.

Les patches provenant de zones géographiquement voisines ne doivent pas être répartis arbitrairement entre ces ensembles.

### F06 — Exploration du dataset

Le système doit produire une galerie d'exemples réels permettant d'observer :

- Les cartes bathymétriques en 2D.
- Les reliefs correspondants en 3D.
- La distribution des profondeurs.
- La diversité des formes.
- Les éventuelles anomalies des données.

---

## 4. Entraînement du VAE

### F07 — Construire le VAE

Le système doit permettre d'entraîner un autoencodeur variationnel convolutionnel avec PyTorch.

Le modèle comprend :

- Un encodeur.
- Un espace latent probabiliste.
- Un décodeur.

L'encodeur transforme une carte bathymétrique en paramètres d'une distribution latente.

Le décodeur transforme un vecteur latent en carte bathymétrique synthétique.

### F08 — Configurer le VAE

Les paramètres configurables doivent comprendre :

- La dimension de l'espace latent.
- Le nombre d'époques.
- La taille des batches.
- Le taux d'apprentissage.
- Le poids de régularisation de l'espace latent.
- La graine aléatoire.

Une configuration compacte doit être fournie pour un entraînement local.

La dimension latente initiale recommandée est de 8 ou 16.

### F09 — Entraîner le VAE

Le système doit permettre de lancer l'entraînement sur les patches GEBCO.

Il doit enregistrer :

- La perte totale.
- L'erreur de reconstruction.
- Le terme de régularisation KL.
- Les métriques de validation.
- La durée d'entraînement.

Le meilleur modèle doit pouvoir être sauvegardé et rechargé.

### F10 — Reconstruire des reliefs réels

Le système doit permettre de sélectionner un patch réel et de le faire passer dans le VAE.

Il doit afficher :

- Le relief original.
- Le relief reconstruit.
- Une carte des erreurs de reconstruction.

Les erreurs doivent pouvoir être quantifiées.

### F11 — Générer de nouveaux reliefs

Le système doit permettre d'échantillonner des vecteurs latents et de produire des reliefs synthétiques.

La génération doit être reproductible à partir d'une graine aléatoire.

### F12 — Explorer l'espace latent

Le système doit permettre de modifier certaines composantes d'un vecteur latent.

Chaque modification doit permettre d'observer son effet sur le relief généré.

Il doit également permettre d'interpoler entre deux vecteurs latents.

---

## 5. Entraînement du GAN

### F13 — Construire le GAN

Le système doit permettre d'entraîner un GAN convolutionnel compact avec PyTorch.

Le modèle comprend :

- Un générateur.
- Un discriminateur.

Le générateur produit des cartes bathymétriques à partir de vecteurs latents.

Le discriminateur apprend à distinguer les cartes réelles des cartes synthétiques.

### F14 — Configurer le GAN

Les paramètres configurables doivent comprendre :

- La dimension de l'espace latent.
- Le nombre d'époques.
- La taille des batches.
- Le taux d'apprentissage.
- Les paramètres des optimiseurs.
- La graine aléatoire.

Une configuration par défaut adaptée à un GPU modeste doit être proposée.

### F15 — Entraîner le GAN

Le système doit entraîner alternativement le générateur et le discriminateur.

Pendant l'entraînement, il doit enregistrer :

- La perte du générateur.
- La perte du discriminateur.
- La durée d'entraînement.
- Des exemples de reliefs générés à intervalles réguliers.

Le système doit permettre de visualiser l'évolution des résultats au fil des époques.

### F16 — Sauvegarder le GAN

Le système doit sauvegarder les paramètres du générateur et du discriminateur.

Le générateur doit pouvoir être rechargé indépendamment pour produire de nouveaux reliefs.

### F17 — Générer des reliefs avec le GAN

Le système doit permettre de produire un ou plusieurs reliefs à partir de vecteurs latents.

La génération doit être reproductible à partir d'une graine aléatoire.

### F18 — Explorer l'espace latent du GAN

Le système doit permettre de modifier les composantes d'un vecteur latent et d'observer les changements du relief.

Il doit également permettre une interpolation entre deux vecteurs latents.

---

## 6. Comparaison VAE / GAN

### F19 — Comparer les courbes d'entraînement

Le système doit afficher les courbes de pertes des deux modèles.

Les pertes du VAE et du GAN doivent être présentées séparément, car elles ne représentent pas directement la même grandeur.

L'objectif est d'étudier les dynamiques d'apprentissage et la stabilité des modèles.

### F20 — Comparer les reliefs générés

Le système doit permettre d'afficher côte à côte :

- Un relief réel issu de GEBCO.
- Un relief généré par le VAE.
- Un relief généré par le GAN.

Les trois cartes doivent utiliser des conventions visuelles comparables.

Les exemples VAE et GAN ne doivent pas être présentés comme des reconstructions du même relief, sauf lorsqu'une méthode explicite le permet.

### F21 — Comparer la diversité

Le système doit générer plusieurs exemples pour chaque modèle.

L'utilisateur doit pouvoir observer :

- La variété des reliefs.
- La répétition éventuelle de structures.
- Les reliefs atypiques.
- Les différences de formes et de textures.

Une attention particulière doit être portée au risque de *mode collapse* du GAN.

### F22 — Comparer les caractéristiques statistiques

Le système doit calculer des indicateurs simples sur les reliefs réels et synthétiques.

Les indicateurs peuvent inclure :

- Distribution des profondeurs normalisées.
- Amplitude du relief.
- Variabilité locale.
- Distribution des pentes.
- Mesures simples de rugosité.

Ces indicateurs doivent permettre une comparaison statistique des jeux de données.

Ils ne constituent pas, à eux seuls, une preuve de réalisme géologique.

### F23 — Synthétiser les résultats

Le système doit produire un tableau comparatif présentant :

- La durée d'entraînement.
- La stabilité observée.
- La diversité des générations.
- La qualité des reconstructions pour le VAE.
- Les limites rencontrées.
- Les observations pédagogiques.

---

## 7. Export des modèles

### F24 — Exporter le VAE

Le système doit exporter au format ONNX les composants nécessaires à l'application web :

- Le décodeur pour la génération.
- L'encodeur ou une version d'inférence adaptée pour la reconstruction.

### F25 — Exporter le GAN

Le système doit exporter le générateur GAN au format ONNX.

Le discriminateur n'est pas nécessaire pour la génération dans l'application web.

### F26 — Valider les exports

Le système doit vérifier que les modèles ONNX produisent des résultats cohérents avec leurs équivalents PyTorch sur les mêmes entrées.

Les informations nécessaires à l'inférence doivent être conservées.

---

## 8. Application web — Ocean Terrain Lab

### 8.1. Objectif

L'application web doit permettre de découvrir les modèles génératifs par l'expérimentation.

L'utilisateur doit pouvoir générer des reliefs, manipuler leurs représentations latentes et comparer les résultats du VAE et du GAN.

L'application doit fonctionner sans serveur Python pour l'inférence.

### F27 — Choisir le modèle

L'utilisateur doit pouvoir sélectionner :

- VAE.
- GAN.
- Mode comparaison.

Le modèle sélectionné doit être clairement identifié.

### F28 — Générer un relief

L'application doit proposer un bouton **Générer un relief**.

Cette action doit produire une nouvelle carte bathymétrique synthétique à partir d'un vecteur latent.

L'utilisateur doit pouvoir définir une graine aléatoire pour reproduire une génération.

### F29 — Visualiser en 2D

Le relief doit être affiché sous forme de carte colorée.

L'interface doit proposer :

- Une palette bathymétrique.
- Une légende.
- Une échelle des valeurs.
- Un affichage lisible des variations de relief.

### F30 — Visualiser en 3D

Le relief doit pouvoir être affiché sous forme de surface tridimensionnelle interactive.

L'utilisateur doit pouvoir :

- Faire tourner la vue.
- Zoomer.
- Modifier l'angle d'observation.
- Réinitialiser la caméra.
- Régler l'exagération verticale.

La visualisation doit être calculée à partir des valeurs générées, et non d'une simple illustration.

### F31 — Manipuler l'espace latent

L'application doit proposer des curseurs permettant de modifier les composantes du vecteur latent.

Une modification doit déclencher une nouvelle génération.

L'utilisateur doit pouvoir revenir au vecteur initial.

Pour conserver une interface simple, seules quelques dimensions peuvent être affichées simultanément.

### F32 — Interpoler entre deux reliefs

L'utilisateur doit pouvoir sélectionner deux vecteurs latents.

L'application doit générer une transition progressive entre les deux reliefs.

Cette transition doit pouvoir être explorée à l'aide d'un curseur ou d'une animation.

### F33 — Reconstruire un relief réel avec le VAE

L'application doit proposer une galerie de patches GEBCO réels.

L'utilisateur sélectionne un relief et demande sa reconstruction.

L'interface affiche côte à côte :

- Le relief réel.
- Le relief reconstruit.
- L'erreur de reconstruction.

Cette fonctionnalité est disponible uniquement pour le VAE.

### F34 — Comparer VAE et GAN

Le mode comparaison doit afficher deux panneaux :

**Panneau gauche : VAE**

**Panneau droit : GAN**

L'utilisateur doit pouvoir générer de nouveaux reliefs dans chaque panneau.

Les vues 2D et 3D doivent utiliser des échelles comparables.

Les générations doivent être identifiées comme indépendantes : une même graine ou un même vecteur numérique ne garantit pas une correspondance sémantique entre les deux modèles.

### F35 — Afficher des statistiques

Pour chaque relief généré, l'application doit afficher :

- Valeur minimale.
- Valeur maximale.
- Valeur moyenne.
- Amplitude du relief.
- Indicateur simple de rugosité.

Les valeurs doivent être clairement distinguées selon qu'elles sont normalisées ou exprimées en mètres.

### F36 — Télécharger les résultats

L'utilisateur doit pouvoir télécharger :

- La carte 2D en PNG.
- Une capture de la vue 3D en PNG.
- La grille numérique en CSV.
- Les paramètres de génération en JSON.

### F37 — Galerie de générations

L'application doit permettre de générer une galerie de plusieurs reliefs.

L'utilisateur doit pouvoir sélectionner une génération pour l'examiner en détail.

La galerie doit faciliter la comparaison de la diversité des modèles.

---

## 9. Exigences d'ergonomie

### 9.1. Expérience utilisateur

L'application doit être visuelle, interactive et adaptée à une démonstration en formation.

La génération doit être accessible en une action.

### 9.2. Organisation

L'interface doit distinguer trois espaces :

1. Génération.
2. Exploration de l'espace latent.
3. Comparaison VAE / GAN.

### 9.3. Cohérence visuelle

Les reliefs réels et synthétiques doivent être présentés avec des conventions de couleurs cohérentes.

Les axes, légendes et unités doivent être explicites.

### 9.4. Retours utilisateur

L'application doit signaler :

- Le chargement des modèles.
- La génération en cours.
- La génération terminée.
- Les erreurs éventuelles.

---

## 10. Exigences non fonctionnelles

### NF01 — Entraînement local

Les modèles doivent être entraînés localement avec Python et PyTorch.

### NF02 — Ressources modestes

Une configuration compacte doit permettre l'entraînement sur un GPU disposant d'environ 4 Go de VRAM.

Une exécution CPU doit également être possible, avec des durées potentiellement plus importantes.

### NF03 — Durée pédagogique

Le projet doit fournir des configurations d'entraînement courtes permettant d'obtenir de premiers résultats pendant une séance.

Des checkpoints préentraînés doivent être disponibles pour garantir le fonctionnement des démonstrations, même si l'entraînement n'est pas terminé.

### NF04 — Application statique

L'application web doit pouvoir être déployée sur un hébergement statique.

### NF05 — Inférence dans le navigateur

Les modèles ONNX doivent être exécutés directement dans le navigateur.

### NF06 — Reproductibilité

Les graines aléatoires, configurations, versions de modèles et paramètres de normalisation doivent être conservés.

### NF07 — Traçabilité

Chaque modèle doit être associé à son dataset, sa configuration et ses résultats d'entraînement.

### NF08 — Limites scientifiques

Les reliefs synthétiques doivent être explicitement identifiés comme tels.

Ils ne doivent pas être présentés comme des relevés réels ou comme des représentations géologiquement validées.

---

## 11. Scénarios de recette

### Scénario 1 — Préparation des données

**Étant donné** un ensemble de données GEBCO,

**Quand** le système prépare le dataset,

**Alors** il produit des patches bathymétriques normalisés et exploitables par les deux modèles.

### Scénario 2 — Entraînement VAE

**Étant donné** un dataset préparé,

**Quand** l'utilisateur lance l'entraînement du VAE,

**Alors** le modèle est entraîné, les pertes sont enregistrées et un checkpoint est sauvegardé.

### Scénario 3 — Entraînement GAN

**Étant donné** le même dataset,

**Quand** l'utilisateur lance l'entraînement du GAN,

**Alors** le générateur et le discriminateur sont entraînés et des exemples intermédiaires sont enregistrés.

### Scénario 4 — Reconstruction

**Étant donné** un patch bathymétrique réel,

**Quand** l'utilisateur lance une reconstruction VAE,

**Alors** le système affiche le relief original, le relief reconstruit et leurs différences.

### Scénario 5 — Génération web

**Étant donné** que l'application a chargé le modèle VAE,

**Quand** l'utilisateur clique sur « Générer un relief »,

**Alors** un nouveau relief synthétique apparaît.

### Scénario 6 — Exploration 3D

**Étant donné** un relief généré,

**Quand** l'utilisateur active la vue 3D,

**Alors** il peut tourner, zoomer et explorer la surface.

### Scénario 7 — Interpolation

**Étant donné** deux vecteurs latents,

**Quand** l'utilisateur déplace le curseur d'interpolation,

**Alors** le relief évolue progressivement entre les deux générations.

### Scénario 8 — Comparaison

**Étant donné** que les deux modèles sont disponibles,

**Quand** l'utilisateur active le mode comparaison,

**Alors** il peut générer et visualiser des reliefs issus du VAE et du GAN dans des panneaux distincts.

### Scénario 9 — Export

**Étant donné** un relief généré,

**Quand** l'utilisateur demande l'export CSV,

**Alors** la grille numérique du relief est téléchargée.

---

## 12. Priorisation

### Version 1 — Minimum fonctionnel

- Acquisition et préparation GEBCO.
- Entraînement VAE.
- Entraînement GAN.
- Sauvegarde des modèles.
- Export ONNX.
- Génération web VAE et GAN.
- Visualisation 2D et 3D.
- Comparaison côte à côte.

### Version 2 — Exploration enrichie

- Reconstruction VAE.
- Manipulation des dimensions latentes.
- Interpolation.
- Galerie de générations.
- Statistiques descriptives.
- Export des résultats.

### Version 3 — Analyse comparative

- Visualisation des courbes d'entraînement.
- Comparaison des distributions statistiques.
- Étude de la diversité.
- Mise en évidence des difficultés du GAN.
- Analyse des limites scientifiques.

---

## 13. Livrables attendus

### Livrable A — Python / PyTorch

- Pipeline d'acquisition et de préparation GEBCO.
- Dataset de patches.
- Modèle VAE.
- Modèle GAN.
- Scripts d'entraînement.
- Scripts de génération.
- Scripts d'évaluation et de comparaison.
- Checkpoints.
- Modèles ONNX.
- Courbes d'entraînement et exemples de génération.

### Livrable B — Application web

- Application Ocean Terrain Lab.
- Génération avec VAE.
- Génération avec GAN.
- Visualisation 2D.
- Visualisation 3D.
- Exploration latente.
- Reconstruction VAE.
- Comparaison VAE / GAN.
- Export PNG, CSV et JSON.

### Livrable C — Documentation

- Installation.
- Provenance des données.
- Préparation du dataset.
- Entraînement VAE.
- Entraînement GAN.
- Export ONNX.
- Lancement de l'application web.
- Analyse comparative.
- Limites des modèles.

---

## 14. Critères de réussite

Le TP est considéré comme réussi lorsque :

1. Un dataset bathymétrique réel a été constitué à partir de GEBCO.
2. Un VAE convolutionnel a été entraîné avec PyTorch.
3. Un GAN convolutionnel a été entraîné sur les mêmes données.
4. Le VAE peut reconstruire des reliefs réels.
5. Les deux modèles peuvent générer des reliefs synthétiques.
6. Les courbes d'entraînement et des exemples générés sont disponibles.
7. Les composants nécessaires ont été exportés au format ONNX.
8. L'application web exécute les générateurs dans le navigateur.
9. Les reliefs peuvent être visualisés en 2D et en 3D.
10. Une comparaison pédagogique VAE / GAN est possible.

Aucun score de qualité générative minimal n'est imposé. La réussite pédagogique repose sur la compréhension, la mise en œuvre et l'analyse des deux approches.

---

## 15. Résultat final attendu

Le projet doit démontrer deux chaînes d'apprentissage génératif :

**GEBCO → Préparation → Entraînement VAE → Export ONNX → Génération et reconstruction web.**

**GEBCO → Préparation → Entraînement GAN → Export ONNX → Génération web.**

L'application Ocean Terrain Lab doit réunir ces deux chaînes dans une interface interactive permettant de comparer les architectures et d'explorer les représentations latentes.

Les reliefs générés sont des données synthétiques destinées à l'expérimentation et à l'enseignement.

**Fin de la spécification fonctionnelle — Version 1.0**