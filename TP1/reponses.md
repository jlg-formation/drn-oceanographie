# TP 1 — Corrigé

## Exercice 1 — Prévision du niveau de la mer

**Réponse : Prévision de série temporelle — apprentissage supervisé**

Les observations sont ordonnées dans le temps et les valeurs passées servent à prédire une valeur future.

Comme la grandeur prédite est une valeur numérique continue, on peut également parler de **régression sur série temporelle**.

---

## Exercice 2 — Identifier la nature du fond marin

**Réponse : Classification multiclasse — apprentissage supervisé**

Le modèle doit choisir une catégorie parmi plusieurs possibilités : sable, roche, vase, etc.

L'apprentissage est supervisé car on possède des exemples pour lesquels la nature du fond est déjà connue.

---

## Exercice 3 — Découvrir des groupes d'observations océaniques

**Réponse : Clustering — apprentissage non supervisé**

Aucune catégorie n'est fournie au modèle. Il doit découvrir lui-même des groupes d'observations présentant des caractéristiques similaires.

Des algorithmes comme **K-means** ou **DBSCAN** peuvent par exemple être utilisés.

---

## Exercice 4 — Estimer la hauteur des vagues

**Réponse : Régression — apprentissage supervisé**

Le modèle apprend à prédire une **valeur numérique continue**, ici la hauteur significative des vagues, à partir de plusieurs variables d'entrée.

Il s'agit d'apprentissage supervisé puisque les hauteurs réelles sont connues dans les données d'entraînement.

---

## Exercice 5 — Détecter une nappe d'hydrocarbures

**Réponse : Classification binaire supervisée avec classes fortement déséquilibrées**

Le modèle doit choisir entre deux possibilités : présence ou absence d'une nappe.

Le dataset est fortement déséquilibré :

- 99,5 % des images ne contiennent pas de nappe ;
- 0,5 % en contiennent une.

L'accuracy seule serait donc trompeuse : un modèle répondant toujours « aucune nappe » obtiendrait déjà 99,5 % d'accuracy.

Il faudra notamment examiner la **precision**, le **recall**, le **F1-score** et la **matrice de confusion**.

---

## Exercice 6 — Réduire le nombre de variables

**Réponse : Réduction de dimension — généralement apprentissage non supervisé**

Le but est de représenter une observation comportant 200 variables avec seulement quelques dimensions tout en conservant le maximum d'information utile.

On cherche donc à effectuer une transformation du type :

$$
\mathbb{R}^{200} \rightarrow \mathbb{R}^{3}
$$

La **PCA** est une méthode classique de réduction de dimension.

En Deep Learning, on peut notamment apprendre ce type de représentation avec un **auto-encodeur**.

---

## Exercice 7 — Identifier les situations courantes et rares

**Réponse : Estimation de densité — apprentissage non supervisé**

Le modèle cherche à apprendre la distribution des observations.

Avec la température $T$ et la salinité $S$, on cherche par exemple à estimer une densité :

$$
p(T,S)
$$

Une forte densité correspond à une région fréquemment observée, tandis qu'une faible densité correspond à une situation plus rare.

Attention : une **densité de probabilité** n'est pas elle-même nécessairement un nombre compris entre 0 et 1.

La probabilité qu'une observation appartienne à une région $A$ est obtenue en intégrant la densité sur cette région :

$$
P((T,S) \in A) = \int_A p(T,S)\,dT\,dS
$$

---

## Exercice 8 — Détecter automatiquement des anomalies

**Réponse : Détection d'anomalies — apprentissage non supervisé**

Les données ne comportent pas d'étiquettes `normal` ou `anormal`.

Le système doit apprendre ce qui caractérise les observations habituelles et détecter celles qui s'en éloignent fortement.

Le modèle peut par exemple produire un **score d'anomalie** :

```text
Observation A → score 0,04
Observation B → score 0,17
Observation C → score 0,93
```

On peut ensuite définir un seuil pour transformer ce score en décision :

```text
score < 0,8  → normal
score ≥ 0,8  → anomalie
```

Même si la décision finale est binaire, il ne s'agit pas nécessairement d'une classification binaire supervisée puisque le modèle n'a pas été entraîné à partir de deux classes étiquetées.

---

## Exercice 9 — Plusieurs phénomènes sur une même image

**Réponse : Classification multilabel — apprentissage supervisé**

Contrairement à une classification multiclasse, plusieurs catégories peuvent être vraies simultanément.

Par exemple :

```text
nuages         = 1
navires        = 1
algues         = 0
forte houle    = 1
hydrocarbures  = 0
```

Il ne faut donc pas confondre :

- **classification multiclasse** : une classe parmi plusieurs ;
- **classification multilabel** : plusieurs classes peuvent être présentes simultanément.

---

## Exercice 10 — Construire une représentation des situations océaniques

**Réponse : Apprentissage de représentations (Representation Learning)**

Le modèle apprend à transformer les données originales en **vecteurs de caractéristiques**, souvent appelés **embeddings**.

Par exemple, une observation $x$ peut être transformée par un encodeur en un vecteur latent $z$ :

$$
z = f(x)
$$

Ces vecteurs constituent une **représentation latente** des données dans un **espace latent**.

On cherche idéalement à obtenir une organisation dans laquelle des situations similaires possèdent des représentations proches.

Un **auto-encodeur** est l'une des architectures permettant d'apprendre ce type de représentation.