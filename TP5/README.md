# TP5 — MaréeLab : prévision des marées par réseaux récurrents

Ce TP entraîne trois **réseaux récurrents** (PyTorch) à prévoir les **24 prochaines
heures** de hauteur d'eau au marégraphe de **Brest** à partir des **48 heures
précédentes** : un **RNN simple**, un **GRU** et un **LSTM**. Ils sont ensuite
exportés en **ONNX** pour l'application web **MaréeLab**, qui exécute les
prévisions **directement dans le navigateur** (onnxruntime-web, sans serveur
Python).

Les données sont des **observations réelles** issues de REFMAR (SHOM). La
spécification complète est dans [brief.md](brief.md).

> ⚠ Les prévisions sont **expérimentales** et pédagogiques. Ce n'est pas un outil
> officiel de prédiction des marées du SHOM : pour la navigation, utilisez
> <https://maree.shom.fr>.

---

## 1. Lancer le TP

```powershell
mise run tp5            # tout : download -> prepare -> train -> evaluate -> export -> test -> web
```

ou étape par étape :

| Tâche | Commande | Produit |
|---|---|---|
| 1. Acquérir | `mise run tp5_01_download` | `data/raw/` (réponses brutes de l'API), `data/brest_horaire.csv`, `data/provenance.json` |
| 2. Préparer | `mise run tp5_02_prepare` | `data/qc/rapport_qc.json`, `data/figures/*.png`, `data/dataset/` (`serie.npz`, `dataset.json`) |
| 3. Entraîner | `mise run tp5_03_train` | `runs/{rnn,gru,lstm}/` (`best.pt`, `history.json`, `config.json`, `courbes.png`), `runs/comparaison_pertes.png` |
| 3 bis. Sans GPU | `mise run tp5_03_train_cpu` | idem, configuration réduite |
| 4. Évaluer | `mise run tp5_04_evaluate` | `runs/evaluation.json`, `runs/evaluation/*.png` |
| 5. Exporter | `mise run tp5_05_export` | `model/{rnn,gru,lstm}.onnx`, `model/serie.json`, `model/metadata.json` |
| 6. Tester | `mise run tp5_06_test` | tests unitaires et de données |
| 7. Utiliser | `mise run tp5_07_web` | MaréeLab sur <http://localhost:5177> |

Le site ne dépend que du dossier `TP5/model/` (≈ 0,7 Mo), servi tel quel par
Vite (`publicDir`). `bun run build:tp5` produit un site 100 % statique (NF03).

Organisation du code, sur le modèle du TP4 :

| Fichier | Rôle |
|---|---|
| [maree.py](maree.py) | module commun : station, API, constantes, fenêtres, références naïves, métriques, styles des courbes |
| [download_data.py](download_data.py) | F01 : acquisition REFMAR |
| [prepare_dataset.py](prepare_dataset.py) | F02 à F05 : contrôle qualité, figures, fenêtres, séparation |
| [modeles.py](modeles.py) | F06 à F08 : RNN / GRU / LSTM et graphe d'export |
| [train.py](train.py) | F09 à F11 : entraînement, suivi, checkpoints |
| [evaluate.py](evaluate.py) | F13 à F16 : métriques, références, figures, périodes d'exemple |
| [export_onnx.py](export_onnx.py) | F12 : ONNX vérifié, données du site |
| [front-end/](front-end/) | F17 à F27 : application MaréeLab |

---

## 2. Les données (F01)

| | |
|---|---|
| Station | **Brest** (id SHOM 3, réseau RONIM), 48,383° N, 4,495° O |
| Source | API publique REFMAR `services.data.shom.fr/maregraphie/observation/json/3`, **sans compte ni clé** |
| Nature | **source 4 = hauteurs horaires validées** (1 : brutes haute fréquence, 2 : brutes 10 min, 3 : validées 10 min) |
| Période | 5 années civiles : **2021-01-01 → 2025-12-31** |
| Unité / référence | mètres au-dessus du **zéro hydrographique** (ZH = −3,635 m / IGN69) |
| Horodatage | UTC |
| Licence | © Shom, Licence Ouverte Etalab 2.0 |

L'API limite chaque réponse à environ un mois de mesures horaires. Le script
interroge donc l'API **mois par mois** et conserve chaque réponse brute dans
`data/raw/` (cache : un mois déjà présent n'est pas redemandé). `provenance.json`
garde la station, la source, la période, la référence verticale, les URL, la date
de téléchargement et le **journal de bord** du marégraphe (changements de capteur,
contrôles…), utile au contrôle qualité.

---

## 3. Contrôle qualité et préparation (`prepare_dataset.py`, F02 à F05)

### Contrôle qualité (F02, NF06)

Rien n'est corrigé en silence. Chaque catégorie est comptée et listée dans
`data/qc/rapport_qc.json`.

| Contrôle | Règle | Traitement | Résultat 2021-2025 |
|---|---|---|---|
| Doublons | même heure plusieurs fois | identiques : un seul gardé ; contradictoires : heure écartée | 0 |
| Hors grille | horodatage pas sur une heure ronde | écarté | 0 |
| Non numérique | valeur absente / NaN | écartée | 0 |
| Hors bornes | < −1 m ou > 10 m | écartée | 0 |
| Pic isolé | saut > 2,5 m en 1 h, aller **et** retour | écarté | 0 |
| Heures manquantes | grille horaire régulière | **non comblées** (NaN) | **617 h (1,41 %)** |
| Ruptures | suites d'heures manquantes | listées | **7**, la plus longue **488 h** (déc. 2024) |
| Référence verticale | moyennes annuelles (saut > 10 cm) + journal de bord | à vérifier si suspect | aucun changement (moyennes 4,22 → 4,28 m) |

Les données validées du SHOM sont propres : le seul vrai problème est celui des
**lacunes**. Elles ne sont jamais interpolées : **toute fenêtre qui touche une
lacune est écartée**.

### Visualisations (F03)

`data/figures/` : `01_serie_complete.png` (série, ensembles, lacunes),
`02_extrait_7_jours.png` (deux marées par jour, décalage d'environ 50 min par
jour), `03_distribution.png` (bimodale : pleines et basses mers),
`04_lacunes_par_mois.png`. Les statistiques descriptives sont dans le rapport :
moyenne 4,26 m, écart-type 1,56 m, min 0,42 m, max 8,05 m.

### Fenêtres et séparation (F04, F05)

```
          entrée : 48 h observées              cible : 24 h
  ├───────────────────────────────────────┤├──────────────────────┤
 H−47                                     H H+1                  H+24
```

- **Prévision directe** : le réseau sort les 24 valeurs en une passe. Il ne
  réinjecte pas ses propres prévisions.
- Décalage entre fenêtres : 1 h. `--entree`, `--horizon` et `--pas` sont modifiables.
- **Séparation chronologique** 70 / 15 / 15 %, coupures alignées sur minuit.
  Une fenêtre doit tenir **entièrement** dans un ensemble : celles qui sont à
  cheval sont écartées. Aucune cible d'entraînement n'appartient donc à la
  validation ou au test. Ce point est vérifié par une assertion et par les tests.
- **Normalisation** (moyenne 4,249 m, écart-type 1,564 m) ajustée **sur la seule
  période d'entraînement**.

| Ensemble | Période | Fenêtres | Écartées (lacunes) |
|---|---|---|---|
| train | 2021-01-01 → 2024-07-01 | 30 098 | 503 |
| val | 2024-07-02 → 2025-04-01 | 5 946 | 559 |
| test | 2025-04-02 → 2025-12-31 | 6 505 | 0 |

---

## 4. Les modèles ([modeles.py](modeles.py), F06 à F08)

Les trois modèles ont **exactement** la même structure, seule la cellule change :

```
48 hauteurs normalisées ─► [cellule récurrente] × 2 couches, état caché 64 ─► h₄₈ ─► Linéaire(64 → 24) ─► 24 hauteurs
                            RNN  |  GRU  |  LSTM
```

| Modèle | Cellule | Paramètres |
|---|---|---|
| RNN | h_t = tanh(W x_t + U h_{t−1} + b) | 14 168 |
| GRU | + portes de mise à jour et de réinitialisation | 39 384 |
| LSTM | + cellule mémoire, portes d'entrée, d'oubli, de sortie | 51 992 |

### Entraînement (`train.py`, F09 à F11)

Protocole identique pour les trois : mêmes fenêtres, **même graine**, même
ordre des mini-batches. Perte MSE sur les hauteurs normalisées, AdamW avec
décroissance cosinus, **écrêtage du gradient** (qui protège le RNN de l'explosion
du gradient). Le meilleur modèle est choisi sur la **perte de validation**.

| Paramètre | Défaut | Option |
|---|---|---|
| Modèle(s) | les trois | `--modele rnn\|gru\|lstm\|tous` |
| Époques | 60 | `--epochs` |
| Batch | 256 | `--batch-size` |
| Taux d'apprentissage | 2e-3 | `--lr` |
| État caché / couches | 64 / 2 | `--cachee` / `--couches` |
| Graine | 0 | `--seed` |
| Répertoires | `data/dataset` / `runs` | `--data-dir` / `--out-dir` |
| Configuration réduite CPU | état caché 32, 1 couche, 8 époques, 1 fenêtre sur 3 | `--rapide` (et `--cpu`) |

Durées mesurées : sur GPU (CUDA), 34 s (RNN), 78 s (GRU), 91 s (LSTM) pour
60 époques ; en `--rapide --cpu`, moins de 30 s pour les trois modèles (MAE de
validation autour de 21-23 cm).

`best.pt` contient poids, configuration, **normalisation**, époque et identifiant
du dataset : on le recharge sans réentraîner (`modeles.construire_modele`).

---

## 5. Évaluation (`evaluate.py`, F13 à F16)

Mêmes 6 505 fenêtres de test pour tous. Métriques en **mètres** (affichées en cm).
Les références naïves (F15) n'utilisent que les 48 h d'entrée :

- **persistance** : répète la dernière hauteur ;
- **répétition 24 h** : H+k ← H+k−24 ;
- **répétition 25 h** : H+k ← H+k−25, soit environ un jour lunaire (24 h 50) et donc
  deux marées semi-diurnes. C'est une référence ajoutée, bien plus difficile à battre.

Résultats obtenus (MAE en cm, jeu de test) :

| Modèle | H+1 | H+6 | H+12 | H+24 | **24 h** | RMSE 24 h |
|---|---|---|---|---|---|---|
| RNN | 2,5 | 5,9 | 6,3 | 7,6 | **6,3** | 8,4 |
| GRU | 3,1 | 5,8 | 6,2 | 7,5 | **6,3** | 8,4 |
| LSTM | 3,6 | 5,8 | 6,3 | 7,5 | **6,3** | 8,4 |
| Répétition 25 h | 27,1 | 27,1 | 27,0 | 27,0 | 27,0 | 32,4 |
| Répétition 24 h | 55,3 | 55,3 | 55,3 | 55,3 | 55,3 | 62,5 |
| Persistance | 66,6 | 259,8 | 29,8 | 55,3 | 172,7 | 217,8 |

À retenir :

- Les réseaux font **77 % d'erreur en moins** que la meilleure référence : ils
  ont appris quelque chose d'utile.
- **Les trois architectures font jeu égal.** Avec 48 h d'historique, la
  dépendance temporelle reste courte et la marée est un signal très régulier :
  les portes du GRU et du LSTM n'apportent rien ici. Le RNN, 3,7 fois plus petit,
  suffit. C'est un résultat à discuter : « plus complexe » ne veut pas dire
  « meilleur ».
- L'erreur **croît avec l'échéance** (2,5 → 7,5 cm). Le reste est surtout la
  partie **météorologique** du niveau (surcote, décote), que les modèles ne
  voient pas.
- La persistance est très mauvaise à H+6 (≈ une demi-marée : pleine mer ↔
  basse mer) et meilleure à H+12 (≈ une marée).

Figures : `runs/evaluation/erreur_par_echeance.png`,
`runs/evaluation/exemples_previsions.png` (observations, prévisions et écarts
sur les périodes d'exemple, F16), `runs/comparaison_pertes.png` (F10).

### Périodes d'exemple (F24)

Elles sont choisies automatiquement dans le test, à au moins une semaine
d'écart les unes des autres :

| Exemple | Instant H (UTC) | Critère |
|---|---|---|
| Période difficile | 22/10/2025 23:00 | plus grande erreur moyenne des réseaux (≈ 34 cm : surcote de tempête) |
| Forte amplitude | 07/10/2025 23:00 | marnage maximal (7,1 m, vives-eaux) |
| Faible amplitude | 02/09/2025 09:00 | marnage minimal (2,1 m, mortes-eaux) |
| Niveau moyen atypique | 07/12/2025 11:00 | niveau moyen sur 72 h le plus éloigné de la médiane |
| Période typique | 29/05/2025 00:00 | erreur des réseaux proche de la médiane |

---

## 6. Export ONNX (`export_onnx.py`, F12)

Chaque graphe contient **normalisation + réseau + dénormalisation** :

- entrée `hauteurs` : `(N, 48)` float32, en mètres ;
- sortie `prevision` : `(N, 24)`, en mètres.

L'export utilise l'exporteur TorchScript (`dynamo=False`) pour obtenir les
**opérateurs ONNX natifs `RNN`, `GRU` et `LSTM`**, bien lisibles dans Netron.
Chaque modèle est vérifié avec `onnx.checker` puis comparé à PyTorch sur
**toutes les fenêtres de test** : l'écart maximal est inférieur à 5 µm.
`metadata.json` contient aussi un **exemple de contrôle** que le site recalcule
au chargement (écart navigateur ↔ PyTorch affiché, ≈ 0,005 mm).

Taille des modèles : 61 ko (RNN), 162 ko (GRU), 214 ko (LSTM).

---

## 7. L'application web MaréeLab

Parcours : **choisir une période → observer les 48 h → choisir les modèles →
Prédire → Révéler la réalité → comparer**.

| Fonction | Réalisation |
|---|---|
| F17 Période | date et heure UTC, boutons ±1 h / ±1 j, « Au hasard », frise cliquable de la série. Par défaut, **jeu de test uniquement**. Sinon, un bandeau signale que la période a été vue à l'entraînement ou à la validation |
| Scénario 8 | période incomplète : message (nombre d'heures manquantes), bouton « Prédire » désactivé, lien vers **la période valide la plus proche**. Lacunes visibles sur le graphique et la frise (ex. : décocher « test », choisir le 05/12/2024) |
| F18 Graphique | 48 h d'historique, ligne verticale de l'instant H, zone de prévision teintée, axes en heures UTC et en mètres au-dessus du ZH, infobulle au survol |
| F19 / F21 Courbes | RNN, GRU, LSTM et trois références activables indépendamment, depuis la colonne de gauche ou la légende, sans nouvelle inférence |
| §7.4 Accessibilité | chaque courbe a sa couleur (palette pour daltoniens), **son style de trait et son marqueur** |
| F20 Inférence | onnxruntime-web (WebAssembly) : quelques millisecondes par modèle ; chargement et erreurs signalés |
| F22 Révéler | les 24 h futures restent masquées jusqu'au clic |
| F23 Erreurs | MAE, RMSE et erreur absolue à H+1/6/12/24 par courbe, plus faible erreur mise en avant, avec l'avertissement « résultat local à la période » |
| F16 Écarts | panneau « prévision − réalité » sous le graphique après révélation |
| F24 Exemples | les 5 périodes remarquables du test |
| F25 Animation | Lecture / Pause / Recommencer, vitesse de 1 à 12 h/s, sans relancer l'inférence |
| F10, F13, F14 | tableau des métriques sur tout le jeu de test, courbe MAE selon l'échéance, courbes d'apprentissage |
| F26 Pédagogie | RNN, portes du GRU, mémoire du LSTM, apprentissage vs inférence, horizon multiple, références, limites |
| F27 Exports | PNG du graphique, CSV (observations + prévisions ; la réalité future n'y figure que si elle a été révélée), JSON des métriques |

Fichiers : [main.ts](front-end/src/main.ts) (interface),
[donnees.ts](front-end/src/donnees.ts) (chargement, validation des périodes),
[modeles.ts](front-end/src/modeles.ts) (ONNX, références, métriques),
[graphique.ts](front-end/src/graphique.ts) (dessin canvas),
[exports.ts](front-end/src/exports.ts).

---

## 8. Limites connues

- **Pas de météo** : les modèles ne voient que 48 h de hauteurs. Une surcote de
  tempête n'est « prévue » que si elle est déjà visible dans l'historique (voir
  la période difficile du 22/10/2025).
- **Une seule station, une seule période** : Brest de 2021 à mi-2024. Rien ne
  garantit le même comportement dans un autre port ou sous un autre régime de marée.
- **Horizon court** : 24 h. La prévision de marée classique (harmonique) va des
  années à l'avance. Ces réseaux ne la remplacent pas.
- **Classement local** : sur une période de 24 h, l'ordre des modèles change
  d'une période à l'autre. Seule l'évaluation sur les 6 505 fenêtres de test a un
  sens statistique, et elle donne ici les trois réseaux à égalité.
- **Extrêmes** : les événements rares (fortes surcotes) sont peu représentés dans
  l'entraînement. Leur prévision est hors périmètre.
- **Pas de comparaison** avec les prédictions harmoniques officielles du SHOM
  (version 3 du brief).
