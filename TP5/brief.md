# TP5 — Prévision des marées par réseaux récurrents

**Spécification fonctionnelle — Version 1.0**

**Formation :** Deep Learning et réseaux de neurones — SHOM  
**Thème :** Séries temporelles, RNN, GRU et LSTM  
**Application :** MaréeLab — Laboratoire interactif de prévision des marées  
**Statut :** Proposition de spécification

---

## 1. Présentation du projet

### 1.1. Contexte

Le SHOM collecte et diffuse des observations du niveau de la mer, notamment grâce au réseau de marégraphes RONIM et au portail REFMAR.

Ces observations constituent des séries temporelles particulièrement intéressantes pour l'apprentissage des réseaux de neurones récurrents.

Le TP consiste à exploiter des mesures historiques de hauteur d'eau afin d'entraîner plusieurs modèles capables de prévoir leur évolution.

### 1.2. Objectif général

Développer deux composants complémentaires :

1. **Un environnement Python/PyTorch** permettant de préparer les données, d'entraîner et d'évaluer trois architectures : RNN, GRU et LSTM.
2. **Une application web interactive** permettant de sélectionner une période historique, de lancer des prévisions et de comparer graphiquement les modèles.

Le projet doit être réalisable localement, sans infrastructure cloud obligatoire.

### 1.3. Objectifs pédagogiques

À l'issue du TP, les participants doivent savoir :

- Transformer une série temporelle en exemples supervisés.
- Comprendre le principe des fenêtres glissantes.
- Construire un RNN simple.
- Comprendre les différences entre RNN, GRU et LSTM.
- Entraîner un modèle de prévision avec PyTorch.
- Éviter les fuites de données temporelles.
- Évaluer des prévisions à plusieurs horizons.
- Comparer les performances de plusieurs architectures.
- Exporter et utiliser des modèles entraînés dans une application web.

---

## 2. Périmètre

### 2.1. Cas d'usage

Le système doit prévoir les **24 prochaines heures de hauteur d'eau**, à partir des **48 heures précédentes**.

La station de référence est **Brest**.

Les observations sont exprimées en mètres, dans le référentiel vertical associé aux données sources.

### 2.2. Modèles étudiés

Trois architectures doivent être disponibles :

| Modèle | Objectif pédagogique |
|---|---|
| RNN simple | Comprendre la récurrence et l'état caché |
| GRU | Étudier les mécanismes de portes |
| LSTM | Étudier la mémoire à court et long terme |

Les trois modèles doivent résoudre le même problème de prévision pour permettre une comparaison équitable.

### 2.3. Hors périmètre

La première version ne prévoit pas :

- La prévision opérationnelle des marées pour la navigation.
- Le remplacement des prédictions officielles du SHOM.
- La prévision de phénomènes extrêmes.
- L'assimilation de données météorologiques.
- L'entraînement directement dans le navigateur.
- La prévision simultanée de plusieurs stations.
- L'utilisation de modèles Transformer.

---

## 3. Données

### F01 — Acquisition des observations

Le système doit permettre de récupérer des observations historiques de hauteur d'eau issues de REFMAR.

**Source :**

https://data.shom.fr/donnees/refmar/download

**Documentation API :**

https://api.data.shom.fr/support/index.php/fr/services/refmar

Paramètres initiaux :

| Paramètre | Valeur |
|---|---|
| Station | Brest |
| Nature des données | Hauteurs d'eau observées |
| Qualité recherchée | Mesures horaires validées |
| Historique cible | 5 ans |
| Pas temporel | 1 heure |
| Unité | Mètre |
| Horodatage | UTC |

Le système doit conserver la provenance, la période, la référence verticale et les métadonnées de qualité disponibles.

### F02 — Contrôle qualité

Le système doit identifier :

- Les observations manquantes.
- Les doublons.
- Les intervalles irréguliers.
- Les valeurs invalides.
- Les ruptures de continuité.
- Les éventuels changements de référence verticale.

Les observations manquantes ne doivent pas être remplacées silencieusement.

Une fenêtre comportant des lacunes non traitées ne doit pas être utilisée pour l'entraînement.

### F03 — Visualisation exploratoire

Le système doit produire des visualisations des observations :

- Série temporelle complète.
- Extrait de plusieurs jours.
- Distribution des hauteurs.
- Statistiques descriptives.
- Identification des périodes manquantes.

### F04 — Préparation des fenêtres temporelles

Le système doit transformer la série en exemples supervisés.

**Configuration de référence :**

- Entrée : 48 mesures horaires consécutives.
- Sortie : 24 mesures horaires futures.
- Décalage entre fenêtres : 1 heure.

Exemple :

| Entrée | Sortie attendue |
|---|---|
| H−47 à H | H+1 à H+24 |

La sortie doit être une prévision directe de 24 valeurs, et non une succession obligatoire de 24 prédictions réinjectées dans le modèle.

Les paramètres de taille des fenêtres et d'horizon doivent pouvoir être modifiés.

### F05 — Séparation des données

Les données doivent être réparties chronologiquement en trois ensembles :

- Entraînement : environ 70 %.
- Validation : environ 15 %.
- Test : environ 15 %.

Aucun mélange aléatoire des observations entre les trois périodes n'est autorisé.

Les fenêtres utilisées pour l'entraînement ne doivent pas contenir de cibles appartenant à la période de validation ou de test.

Les transformations statistiques doivent être ajustées uniquement sur l'ensemble d'entraînement.

---

## 4. Entraînement des réseaux

### F06 — Entraîner un RNN simple

Le système doit permettre d'entraîner un réseau récurrent simple sur les fenêtres temporelles préparées.

Le modèle reçoit 48 hauteurs d'eau et prédit les 24 valeurs suivantes.

### F07 — Entraîner un GRU

Le système doit permettre d'entraîner un réseau GRU sur les mêmes données.

L'utilisateur doit pouvoir comparer ses performances à celles du RNN simple.

### F08 — Entraîner un LSTM

Le système doit permettre d'entraîner un réseau LSTM sur les mêmes données.

Le protocole d'évaluation doit rester identique pour les trois architectures.

### F09 — Paramétrer les entraînements

Les paramètres configurables doivent inclure :

- Nombre d'époques.
- Taille des batches.
- Taux d'apprentissage.
- Dimension de l'état caché.
- Nombre de couches récurrentes.
- Graine aléatoire.
- Répertoires des données et des modèles.

Une configuration légère compatible CPU doit être fournie.

### F10 — Suivre l'entraînement

Pour chaque modèle, le système doit enregistrer :

- La perte d'entraînement.
- La perte de validation.
- La durée d'entraînement.
- Le nombre d'époques.
- Les paramètres utilisés.
- Le meilleur checkpoint.

Les courbes doivent pouvoir être affichées et comparées.

### F11 — Sauvegarder les modèles

Chaque modèle doit pouvoir être sauvegardé et rechargé sans nouvel entraînement.

Les paramètres de normalisation nécessaires à son utilisation doivent être conservés.

### F12 — Exporter les modèles

Les trois modèles doivent pouvoir être exportés au format ONNX.

La cohérence des prédictions entre PyTorch et ONNX doit être vérifiée sur des exemples de référence.

---

## 5. Évaluation des prévisions

### F13 — Comparer les modèles

Les modèles doivent être évalués sur le même ensemble de test.

Les métriques principales sont :

- MAE : erreur absolue moyenne.
- RMSE : racine de l'erreur quadratique moyenne.

Les métriques doivent être exprimées en mètres après retour à l'échelle physique.

### F14 — Évaluer plusieurs horizons

Le système doit présenter les performances pour plusieurs échéances :

- H+1.
- H+6.
- H+12.
- H+24.

Une mesure globale sur les 24 heures doit également être fournie.

### F15 — Comparer à des références simples

Le projet doit intégrer au minimum deux prévisions de référence :

1. **Persistance :** répéter la dernière hauteur observée.
2. **Répétition à 24 heures :** reproduire la valeur observée 24 heures auparavant pour chaque heure prévue.

Ces références doivent être évaluées avec les mêmes métriques et sur les mêmes périodes que les réseaux de neurones.

Elles servent à vérifier si les modèles apprennent quelque chose d'utile.

### F16 — Visualiser les erreurs

Le système doit permettre de consulter :

- Les observations réelles.
- Les prévisions de chaque modèle.
- Les écarts entre prévisions et observations.
- Les métriques associées à la période étudiée.

---

## 6. Application web — MaréeLab

### 6.1. Objectif

L'application web doit permettre d'explorer visuellement le comportement des réseaux récurrents.

Le parcours utilisateur principal est :

1. Sélectionner une période historique.
2. Observer les 48 heures précédentes.
3. Choisir un ou plusieurs modèles.
4. Lancer la prévision.
5. Comparer les 24 heures prédites aux mesures réelles.

### F17 — Sélectionner une période

L'utilisateur doit pouvoir choisir une date et une heure de référence.

La période sélectionnée doit disposer :

- De 48 heures d'observations historiques valides.
- De 24 heures d'observations futures valides, pour permettre la comparaison.

Le système doit empêcher ou signaler la sélection de périodes incomplètes.

La sélection doit être limitée aux périodes du jeu de test pour les comparaisons présentées comme une évaluation sur données inconnues.

### F18 — Afficher les observations

L'application doit afficher un graphique temporel comprenant :

- Les 48 heures d'historique.
- L'instant de début de prévision.
- Les 24 heures futures.

Les axes doivent indiquer clairement les dates, heures et hauteurs d'eau en mètres.

Les heures doivent être affichées en UTC ou dans un fuseau explicitement indiqué.

### F19 — Choisir les modèles

L'utilisateur doit pouvoir activer indépendamment :

- RNN.
- GRU.
- LSTM.
- Références naïves.

Plusieurs courbes peuvent être affichées simultanément.

Chaque modèle doit disposer d'une couleur et d'une légende cohérentes.

### F20 — Lancer les prévisions

Un bouton **Prédire les prochaines 24 heures** doit lancer l'inférence.

Les modèles entraînés doivent être exécutés directement dans le navigateur à partir de leurs versions ONNX.

Aucun serveur Python ne doit être nécessaire pour effectuer les prédictions.

L'application doit signaler le chargement des modèles et les erreurs éventuelles.

### F21 — Afficher les prévisions

Le graphique doit afficher les courbes prédites à partir de l'instant de référence.

Les prévisions doivent être distinguées visuellement des observations historiques.

L'utilisateur doit pouvoir masquer ou afficher chaque courbe.

### F22 — Révéler les observations futures

L'application doit proposer un bouton **Révéler la réalité**.

Avant son activation, les 24 heures futures observées sont masquées.

Après activation, la courbe réelle apparaît sur le graphique.

L'objectif est de permettre aux stagiaires de comparer leurs attentes avec les résultats obtenus.

### F23 — Comparer les erreurs

Lorsque les observations futures sont révélées, l'application doit afficher :

- La MAE de chaque modèle.
- La RMSE de chaque modèle.
- Le modèle obtenant la plus faible erreur sur la période.

Le classement doit être présenté comme un résultat local à la période sélectionnée, et non comme une preuve de supériorité générale.

### F24 — Explorer plusieurs périodes

L'application doit proposer plusieurs exemples prédéfinis :

- Une période de marées de forte amplitude.
- Une période de marées de faible amplitude.
- Une période présentant une évolution intéressante.
- Une période pour laquelle les modèles rencontrent des difficultés, si elle existe dans le jeu de test.

Les exemples doivent être issus de mesures réelles.

### F25 — Animer les prévisions

L'application doit proposer un mode de lecture animé.

Ce mode révèle progressivement les 24 points de prévision, heure après heure.

L'utilisateur doit pouvoir :

- Démarrer.
- Mettre en pause.
- Recommencer.
- Régler la vitesse d'animation.

L'animation ne doit pas relancer l'inférence à chaque étape.

### F26 — Afficher les informations pédagogiques

L'application doit présenter des explications courtes sur :

- Le principe d'un RNN.
- Le rôle des portes d'un GRU.
- Le rôle de la mémoire d'un LSTM.
- La différence entre apprentissage et inférence.
- Le principe d'une prévision à horizon multiple.

### F27 — Exporter les résultats

L'utilisateur doit pouvoir télécharger :

- Une image PNG du graphique.
- Un fichier CSV contenant les observations et les prévisions.
- Un résumé JSON des métriques.

---

## 7. Exigences d'ergonomie

### 7.1. Interface

L'interface doit être claire, moderne et adaptée à une démonstration en formation.

Le graphique temporel doit occuper la place principale.

### 7.2. Interactivité

Les changements d'affichage, de courbes et de vitesse d'animation doivent être immédiats et ne pas nécessiter un nouvel entraînement.

### 7.3. Lisibilité

L'interface doit distinguer clairement :

- Historique observé.
- Prévisions.
- Observations futures.
- Modèles de référence.

Les unités et les horizons doivent rester visibles.

### 7.4. Accessibilité

Les courbes ne doivent pas être distinguées uniquement par leur couleur.

Des légendes, styles de traits ou marqueurs différents doivent être utilisés.

---

## 8. Exigences non fonctionnelles

### NF01 — Entraînement local

L'entraînement doit fonctionner localement avec Python et PyTorch.

### NF02 — Compatibilité CPU

Une configuration réduite doit permettre l'exécution du TP sans GPU.

### NF03 — Application statique

L'application web doit pouvoir être déployée sur un hébergement statique.

### NF04 — Inférence locale

Les modèles ONNX doivent être exécutés dans le navigateur.

### NF05 — Reproductibilité

Les datasets, configurations, modèles et métriques doivent être traçables.

### NF06 — Intégrité des données

Les observations ne doivent pas être modifiées silencieusement.

Les données manquantes doivent être identifiées et traitées explicitement.

### NF07 — Absence de fuite temporelle

Aucune observation future ne doit être utilisée comme entrée d'un modèle lors d'une prévision.

### NF08 — Performances

Les modèles et les périodes d'exemple doivent être suffisamment légers pour permettre une démonstration interactive sur un ordinateur courant.

### NF09 — Usage pédagogique

Les prévisions produites doivent être clairement identifiées comme expérimentales.

L'application ne doit pas être présentée comme un outil officiel de prédiction des marées du SHOM.

---

## 9. Scénarios de recette

### Scénario 1 — Acquisition

**Étant donné** une période de mesures disponible dans REFMAR,

**Quand** l'utilisateur lance l'import,

**Alors** le système récupère les observations et conserve leurs métadonnées.

### Scénario 2 — Préparation

**Étant donné** une série temporelle valide,

**Quand** le système prépare le dataset,

**Alors** il produit des couples de fenêtres de 48 heures et de cibles de 24 heures.

### Scénario 3 — Entraînement

**Étant donné** un dataset préparé,

**Quand** l'utilisateur lance l'entraînement des trois modèles,

**Alors** les modèles sont entraînés, évalués et sauvegardés.

### Scénario 4 — Export

**Étant donné** un modèle entraîné,

**Quand** son export ONNX est demandé,

**Alors** un modèle utilisable par l'application web est généré et validé.

### Scénario 5 — Prévision web

**Étant donné** une période de test valide,

**Quand** l'utilisateur sélectionne LSTM et lance la prévision,

**Alors** la courbe des 24 prochaines heures est affichée.

### Scénario 6 — Révélation

**Étant donné** une prévision affichée,

**Quand** l'utilisateur clique sur « Révéler la réalité »,

**Alors** les observations futures apparaissent et les erreurs sont calculées.

### Scénario 7 — Comparaison

**Étant donné** plusieurs modèles sélectionnés,

**Quand** les prévisions sont terminées,

**Alors** les courbes et les métriques permettent une comparaison sur la même période.

### Scénario 8 — Données manquantes

**Étant donné** une période contenant des observations manquantes,

**Quand** l'utilisateur tente de lancer une prévision,

**Alors** l'application signale que cette période n'est pas exploitable et propose une autre sélection.

---

## 10. Priorisation

### Version 1 — Cœur pédagogique

- Import des données REFMAR.
- Contrôle qualité.
- Préparation des fenêtres.
- Entraînement RNN, GRU et LSTM.
- Évaluation MAE et RMSE.
- Export ONNX.
- Application web avec sélection de période.
- Graphique observations/prévisions.
- Révélation des observations futures.

### Version 2 — Comparaison enrichie

- Comparaison avec les références naïves.
- Comparaison des métriques par horizon.
- Galerie de périodes prédéfinies.
- Animation des prévisions.
- Export CSV et PNG.

### Version 3 — Extensions

- Autres stations marégraphiques.
- Horizons configurables.
- Comparaison avec des prédictions harmoniques officielles, lorsque les données correspondantes sont disponibles.
- Analyse des situations où les modèles se trompent le plus.

---

## 11. Livrables

### Livrable A — Python/PyTorch

- Pipeline d'acquisition et de préparation des données.
- Dataset supervisé.
- Modèles RNN, GRU et LSTM.
- Scripts d'entraînement.
- Scripts d'évaluation.
- Métriques et courbes.
- Checkpoints des modèles.
- Exports ONNX.

### Livrable B — Application web

- Application MaréeLab.
- Sélection de périodes historiques.
- Chargement des modèles ONNX.
- Prévisions interactives.
- Comparaison graphique.
- Révélation des observations réelles.
- Affichage des erreurs.
- Export des résultats.

### Livrable C — Documentation

- Installation.
- Provenance et préparation des données.
- Entraînement.
- Évaluation.
- Export ONNX.
- Lancement de l'application web.
- Limites connues.

---

## 12. Critères de réussite

Le TP est considéré comme réussi lorsque :

1. Des observations réelles de Brest ont été récupérées.
2. Les données ont été contrôlées et transformées en fenêtres supervisées.
3. Un RNN, un GRU et un LSTM ont été entraînés avec PyTorch.
4. Les trois modèles ont été évalués sur une période chronologiquement distincte.
5. Les résultats ont été comparés à des références simples.
6. Les modèles ont été exportés en ONNX.
7. L'application web exécute les modèles sans serveur d'inférence.
8. L'utilisateur peut sélectionner une période et obtenir une prévision de 24 heures.
9. L'utilisateur peut révéler les observations réelles et comparer les erreurs.
10. Les limites des modèles sont clairement présentées.

Aucun seuil de précision minimal n'est imposé comme critère pédagogique. Les résultats doivent être mesurés, reproductibles et interprétables.

---

## 13. Résultat attendu

Le TP doit démontrer la chaîne complète :

**Observations REFMAR → Préparation temporelle → Entraînement RNN/GRU/LSTM → Évaluation → Export ONNX → Application web interactive → Comparaison des prévisions.**

L'application finale doit permettre aux stagiaires d'explorer intuitivement les différences entre architectures récurrentes, sur un problème océanographique réel.

**Fin de la spécification fonctionnelle — Version 1.0**