# TP 1 — Identifier les types de problèmes de Machine Learning

## Objectif

Pour chacune des situations suivantes, identifiez le **type de problème de Machine Learning** rencontré.

Lorsque cela est pertinent, précisez également s'il s'agit d'un **apprentissage supervisé ou non supervisé**, et justifiez brièvement votre réponse.

---

## Exercice 1 — Prévision du niveau de la mer

Un marégraphe mesure la hauteur de la mer toutes les 10 minutes :

```text
08:00 → 3,21 m
08:10 → 3,35 m
08:20 → 3,49 m
08:30 → 3,63 m
...
```

On dispose de plusieurs années de mesures historiques. On souhaite construire un modèle qui utilise les mesures des dernières heures pour **prédire la hauteur de la mer dans une heure**.

Par exemple :

```text
Mesures des dernières heures
        ↓
      Modèle
        ↓
Hauteur dans 1 heure : 4,17 m
```

**Quel type de problème de Machine Learning est-ce ?**

---

## Exercice 2 — Identifier la nature du fond marin

Un navire réalise des mesures à l'aide d'un sonar multifaisceaux. Les caractéristiques des signaux acoustiques retournés dépendent notamment de la nature du fond marin.

Pour de nombreuses zones, des experts ont déjà déterminé la véritable nature du fond :

```text
Zone A → sable
Zone B → roche
Zone C → vase
Zone D → sable
...
```

On souhaite entraîner un modèle capable de déterminer automatiquement la nature du fond pour de nouvelles mesures sonar.

**Quel type de problème est-ce ? S'agit-il d'un apprentissage supervisé ou non supervisé ?**

---

## Exercice 3 — Découvrir des groupes d'observations océaniques

On dispose de plusieurs dizaines de milliers de relevés océaniques. Chaque relevé contient différentes mesures :

```text
température
salinité
profondeur
oxygène dissous
vitesse du courant
...
```

Aucune catégorie n'a été attribuée aux observations.

On souhaite laisser l'algorithme analyser les données afin de **découvrir automatiquement des groupes d'observations qui se ressemblent**. Des océanographes pourront ensuite étudier ces groupes et essayer de leur donner une interprétation scientifique.

**Quel type de problème de Machine Learning est-ce ?**

---

## Exercice 4 — Estimer la hauteur des vagues

On dispose d'un grand nombre d'observations contenant notamment :

```text
vitesse du vent
direction du vent
pression atmosphérique
profondeur
distance à la côte
hauteur significative des vagues
```

On souhaite apprendre la relation entre ces différentes variables afin de pouvoir fournir au modèle de nouvelles conditions météorologiques et marines et obtenir, par exemple :

```text
Hauteur significative estimée : 2,73 m
```

Il ne s'agit pas nécessairement de prédire le futur, mais d'estimer une grandeur à partir d'autres grandeurs connues.

**Quel type de problème est-ce ?**

---

## Exercice 5 — Détecter une nappe d'hydrocarbures

On souhaite analyser automatiquement des images satellite afin de déterminer si une **nappe d'hydrocarbures** est visible.

Une base d'apprentissage de 100 000 images a été constituée :

```text
99 500 images → aucune nappe
   500 images → présence d'une nappe
```

Toutes les images ont été préalablement annotées par des experts. Le système devra ensuite analyser une nouvelle image et indiquer si une nappe est présente ou non.

**Quel type de problème est-ce ? Quel problème important remarquez-vous immédiatement dans le jeu de données ?**

---

## Exercice 6 — Réduire le nombre de variables

Une observation océanographique est décrite par environ **200 variables** : température à différentes profondeurs, salinité, courants, chlorophylle, oxygène, pression, paramètres bathymétriques, etc.

Ces 200 dimensions sont difficiles à visualiser et certaines informations sont probablement redondantes.

On souhaite transformer chaque observation en seulement **2 ou 3 nouvelles variables synthétiques**, tout en conservant autant que possible l'information contenue dans les données originales.

```text
200 variables → modèle → 2 ou 3 variables
```

**Quel type de tâche cherche-t-on à réaliser ?**

---

## Exercice 7 — Identifier les situations courantes et rares

On dispose de plusieurs millions de mesures de **température de surface et de salinité**.

On ne cherche ni à prédire une catégorie, ni à prédire une température ou une salinité. On souhaite plutôt apprendre **comment les observations sont distribuées**.

Le système devrait par exemple pouvoir déterminer qu'une combinaison :

```text
18 °C / 35,2 PSU
```

est très courante dans les données, alors qu'une autre combinaison est extrêmement rare.

**Quel type de problème de Machine Learning cherche-t-on à résoudre ?**

---

## Exercice 8 — Détecter automatiquement des anomalies

Une bouée océanographique mesure en permanence différentes grandeurs :

```text
température
salinité
pression
hauteur de houle
vitesse du vent
...
```

On possède plusieurs années de mesures, mais les données ne comportent **aucune étiquette indiquant si une observation est normale ou anormale**.

On souhaite construire un système capable de signaler automatiquement une nouvelle observation qui serait très différente de ce qui est habituellement observé.

**Quel type de problème est-ce ? Quel type d'apprentissage peut-on envisager ?**

---

## Exercice 9 — Plusieurs phénomènes sur une même image

On analyse des images satellite de zones maritimes. Plusieurs phénomènes peuvent être présents **simultanément sur une même image** :

```text
Image A → nuages + navires
Image B → algues + nuages + forte houle
Image C → navires + hydrocarbures
```

On possède des images déjà annotées et le système doit déterminer, parmi une liste de dix phénomènes possibles, **tous ceux qui sont présents sur une nouvelle image**.

Une image peut donc correspondre à plusieurs catégories en même temps.

**Quel type de problème de Machine Learning est-ce ?**

---

## Exercice 10 — Construire une représentation des situations océaniques

On possède vingt années d'observations contenant de nombreuses informations : température, salinité, courants, houle, pression, etc.

On souhaite transformer chaque situation océanique en un **petit vecteur de nombres** :

```text
Situation A → [0,21 ; -0,83 ; 1,17 ; ...]
Situation B → [0,24 ; -0,79 ; 1,20 ; ...]
Situation C → [2,91 ;  0,14 ; -1,32 ; ...]
```

Le système devrait apprendre une représentation dans laquelle les situations océaniques similaires possèdent des vecteurs proches, tandis que les situations très différentes sont éloignées.

On ne cherche directement ni à prédire une classe, ni à prédire une valeur particulière.

**Qu'apprend à construire le modèle ? Comment appelle-t-on ce type d'apprentissage ?**