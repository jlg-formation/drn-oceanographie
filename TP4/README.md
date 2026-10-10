# TP4 — Terre ou Mer ? Segmentation d'images satellites par CNN

Ce TP entraîne un **réseau de neurones convolutif U-Net** (PyTorch) à classer
chaque pixel d'une image satellite Sentinel-2 en **terre** ou **mer**, puis
l'exporte en **ONNX** pour une application web qui segmente les images
**directement dans le navigateur** (onnxruntime-web, sans serveur Python).

C'est une **segmentation sémantique binaire** : le résultat est un masque de
mêmes dimensions que l'image. La spécification complète est dans
[brief.md](brief.md).

---

## 1. Lancer le TP

```powershell
mise run tp4            # tout : download -> dataset -> train -> evaluate -> export -> test -> web
```

ou étape par étape :

| Tâche | Commande | Produit |
|---|---|---|
| 1. Acquérir | `mise run tp4_01_download` | `data/land/` (polygones OSM par zone), `data/scenes/` (images + provenance) |
| 2. Préparer | `mise run tp4_02_dataset` | `data/masks/`, `data/controle/`, `data/dataset/` (patches + manifeste), `model/examples/` |
| 3. Entraîner | `mise run tp4_03_train` | `runs/unet/` (`best.pt`, `history.json`, `config.json`, `courbes.png`) |
| 3 bis. Sans GPU | `mise run tp4_03_train_cpu` | idem, configuration réduite |
| 4. Évaluer | `mise run tp4_04_evaluate` | `runs/unet/evaluation.json`, `runs/unet/erreurs/*.jpg` |
| 5. Exporter | `mise run tp4_05_export` | `model/unet_terre_mer.onnx`, `model/metadata.json` |
| 6. Tester | `mise run tp4_06_test` | tests unitaires et de données |
| 7. Utiliser | `mise run tp4_07_web` | site sur <http://localhost:5176> |

Le site ne dépend que du dossier `TP4/model/`, servi tel quel par Vite
(`publicDir`). `bun run build:tp4` produit un site 100 % statique (NF04).

---

## 2. Les données

### Images : Sentinel-2

| | |
|---|---|
| Produit | **Sentinel-2 cloudless 2024** (EOX, <https://s2maps.eu>) : mosaïque annuelle sans nuages construite à partir des scènes Sentinel-2 L2A de Copernicus |
| Accès | tuiles WMTS Web Mercator 256 × 256 JPEG, **sans compte ni clé** |
| Résolution | zoom 13, soit **≈ 12,8 m/pixel** en Bretagne (native Sentinel-2 RGB : 10 m) |
| Licence | CC BY-NC-SA 4.0 — *Sentinel-2 cloudless by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2024)* |

Le brief cite le Copernicus Browser comme source principale. Ses scènes
demandent un compte, une sélection manuelle de dates sans nuages et une
reprojection. La mosaïque cloudless donne les **mêmes données Sentinel-2**,
déjà débarrassées des nuages et alignées sur la grille Web Mercator. Le
téléchargement devient ainsi entièrement scripté et reproductible. En contrepartie,
la « date d'acquisition » est celle d'un composite annuel (2024), pas d'un passage
du satellite.

### Vérité d'apprentissage : polygones terrestres OpenStreetMap

| | |
|---|---|
| Produit | `land-polygons-split-4326.zip` (<https://osmdata.openstreetmap.de/data/land-polygons.html>) |
| Taille | ≈ 930 Mo, téléchargé **une seule fois** puis découpé par zone (quelques Mo) |
| Licence | ODbL — © OpenStreetMap contributors |

Aucune annotation manuelle n'est nécessaire.

### Zones

Toutes les zones mesurent 8 × 8 tuiles (2 048 px, ≈ 26 km), sauf les exemples
(4 × 4 tuiles). Elles sont définies dans [terre_mer.py](terre_mer.py).

| Ensemble | Zones | Rôle |
|---|---|---|
| **train** (12) | rade de Brest, Ouessant, Abers, baie de Morlaix, Granit rose, Saint-Malo/Rance, baie de Douarnenez, baie d'Audierne, golfe du Morbihan, Belle-Île, centre Bretagne (100 % terre), large de Sein (100 % mer) | apprentissage |
| **val** (2) | Lorient/Groix, baie de Saint-Brieuc | suivi, choix du meilleur modèle |
| **test** (2) | Concarneau/Glénan, Paimpol/Bréhat | évaluation finale |
| **exemple** (4) | bassin d'Arcachon, baie du Mont-Saint-Michel, calanques de Marseille, golfe d'Ajaccio | démonstration web, **hors Bretagne** |

La séparation se fait **par zone géographique** et non patch par patch. Si on
tirait les patches au hasard, deux patches voisins (quasi identiques)
tomberaient l'un en train, l'autre en test, et les scores seraient trop
optimistes. Un test vérifie que les zones ne se chevauchent pas et que deux
zones d'ensembles différents sont séparées d'au moins une tuile.

---

## 3. Préparation du dataset (`prepare_dataset.py`)

1. **Masque** (F02) : les polygones OSM sont projetés en Web Mercator puis
   rasterisés **sur la grille de pixels exacte de la scène** (mêmes tuiles),
   d'où un alignement au pixel près. Convention : `0` mer, `1` terre, `255` ignoré.
2. **Bande ignorée** : les pixels à moins de 2 px (≈ 25 m) du trait de côte
   valent 255. Le trait de côte de la carte ne coïncide jamais exactement avec
   l'image (marée, estran, généralisation, date). Ces pixels sont exclus de la
   perte et des métriques ; ce n'est **pas** une troisième classe.
3. **Contrôle visuel** (F02, F05) : `data/controle/<zone>.jpg` (image |
   masque | superposition) et `data/controle/dataset_exemples.png`.
4. **Patches** (F03) : découpage 256 × 256 non chevauchant. En entraînement,
   tous les patches côtiers sont gardés, ainsi qu'au plus autant de patches
   « purs » (100 % terre ou 100 % mer). En validation et en test, tout est
   gardé.
5. **Manifeste** (F04, NF06) : `data/dataset/dataset.json` (paramètres,
   répartition, provenance des scènes, liste des patches, identifiant du dataset).

Résultat obtenu (marge 2 px, graine 0) :

| Ensemble | Patches | côtiers | 100 % terre | 100 % mer |
|---|---|---|---|---|
| train | 434 | 196 | 116 | 122 |
| val | 128 | 38 | 47 | 43 |
| test | 128 | 30 | 18 | 80 |

---

## 4. Le modèle : un U-Net compact ([unet.py](unet.py))

```
entrée RGB 256×256 ─► [16] ─► [32] ─► [64] ─► [128] ─► [256] (16×16)
                       │       │       │        │          │
                       └──────►└──────►└───────►└──► décodeur ─► 1×1 conv ─► logit terre
                     skip connections (concaténation)
```

- **Encodeur** : (conv 3×3 → BatchNorm → ReLU) × 2 puis max-pooling, quatre
  fois. La résolution baisse, le contexte grandit : le réseau « voit » la
  baie, l'estuaire.
- **Décodeur** : convolutions transposées et concaténation avec les cartes de
  l'encodeur de même taille (*skip connections*). C'est ce qui permet de placer
  précisément le trait de côte.
- **1,94 M paramètres**, soit un ONNX d'environ 7,8 Mo.
- **Perte** : BCE + Dice calculées **sur les seuls pixels valides** (≠ 255).
  Le Dice compense le déséquilibre des classes (patch à 95 % de mer).
- **Augmentations** : rotations de 90°, symétries, gain par canal, contraste
  et luminosité. Elles apportent de la robustesse aux images importées dans le site.

### Entraînement (`train_unet.py`, F06 à F09)

| Paramètre | Défaut | Option |
|---|---|---|
| Époques | 40 | `--epochs` |
| Batch | 16 | `--batch-size` |
| Taux d'apprentissage | 1e-3 (AdamW, décroissance cosinus) | `--lr` |
| Données / résultats | `data/dataset` / `runs/unet` | `--data-dir` / `--out-dir` |
| Graine | 0 | `--seed` |
| Largeur du réseau | 16 canaux | `--base` |
| Configuration réduite CPU | 8 canaux, 12 époques, 160 patches | `--rapide` (et `--cpu`) |

Le meilleur modèle est choisi sur la **mIoU de validation** (moyenne des IoU
terre et mer) et sauvegardé dans `best.pt` avec sa configuration, sa
normalisation, son époque et l'identifiant du dataset. Il peut ainsi être
rechargé sans réentraînement.

Sur GPU (CUDA), 40 époques prennent environ 12 minutes (16 s par époque). Le meilleur
modèle (époque 39) atteint une **mIoU de validation de 0,992**. Les courbes sont
dans `runs/unet/courbes.png`.

---

## 5. Évaluation (`evaluate.py`, F10 à F12)

Les métriques ignorent les pixels 255 et sont données **pour chaque classe** :

- **IoU** = VP / (VP + FP + FN) : la recouvrement entre prédiction et référence ;
- **Dice** = 2 VP / (2 VP + FP + FN) ;
- **précision** = VP / (VP + FP) : « quand il dit terre, a-t-il raison ? » ;
- **rappel** = VP / (VP + FN) : « quelle part de la terre a-t-il trouvée ? » ;
- **matrice de confusion** `[référence mer/terre] × [prédiction mer/terre]`.

L'évaluation porte sur les patches de test, puis sur les **scènes entières**,
segmentées par fenêtres glissantes comme dans le site. Les zones d'exemple
hors Bretagne sont incluses.

Résultats obtenus :

| Données | IoU terre | IoU mer | Précision / rappel terre | Exactitude |
|---|---|---|---|---|
| Patches de test (128) | 0,984 | 0,994 | 0,986 / 0,998 | 99,57 % |
| Scène Glénan (test) | 0,990 | 0,996 | 0,993 / 0,997 | 99,71 % |
| Scène Bréhat (test) | 0,979 | 0,993 | 0,980 / 0,999 | 99,46 % |
| Arcachon (hors Bretagne) | 0,945 | 0,959 | 0,957 / 0,987 | 97,59 % |
| Calanques (hors Bretagne) | 0,999 | 1,000 | 0,999 / 1,000 | 99,98 % |
| Ajaccio (hors Bretagne) | 0,999 | 0,999 | 0,999 / 1,000 | 99,97 % |
| **Mont-Saint-Michel** | **0,632** | **0,531** | 0,633 / 1,000 | **74,03 %** |

Les erreurs sont presque toutes des « mer prédite terre », concentrées sur
les estrans et les prés salés (voir `runs/unet/erreurs/`). Au Mont-Saint-Michel,
OSM range toute la baie en mer, alors que l'image montre des herbus
végétalisés et des sables découverts. On peut se demander qui a « tort » :
c'est l'occasion de discuter de la définition même des classes.

---

## 6. Export ONNX (`export_onnx.py`, F13)

Le graphe exporté inclut **normalisation + U-Net + sigmoïde** :

- entrée `image` : `(N, 3, 256, 256)` float32, pixels RGB / 255 ;
- sortie `prob_terre` : `(N, 1, 256, 256)`, probabilité d'être de la terre.

Le site n'a donc aucune constante d'entraînement à recopier. L'export est
vérifié avec `onnx.checker`, puis comparé à PyTorch sur des patches de test
et sur une scène entière. `model/metadata.json` regroupe les paramètres
d'inférence, les couleurs, les limites d'import, les métriques et la
provenance du modèle (NF06).

---

## 7. L'application web « Terre ou Mer ? »

Parcours en trois actions : **choisir une image → Analyser → explorer**.

| Fonction | Réalisation |
|---|---|
| F14 Exemples | 6 images jamais vues à l'entraînement : 2 zones de test bretonnes, 4 zones hors Bretagne |
| F15 Import | sélection de fichier ou glisser-déposer, PNG/JPEG, ≤ 25 Mo, ≤ 4 096 px de côté et ≤ 16 Mpx ; messages explicites sinon |
| F16 Analyse | dans un **Web Worker** (interface fluide), **WebGPU** si disponible, sinon **WebAssembly** ; barre de progression |
| F17 Image entière | fenêtres 256 × 256 au pas de 192 px, dernière fenêtre collée au bord, fondu pondéré des recouvrements, complétion en miroir des petites images (même algorithme qu'en Python, testé) |
| F18 Masque | terre en vert, mer en bleu, aligné pixel à pixel |
| F19 Modes | Image originale, Segmentation, Superposition |
| F20 Opacité | curseur en CSS, immédiat, sans nouvelle inférence |
| F21 Comparateur | mode « Avant / après » : curseur vertical déplaçable à la souris, au doigt ou au clavier |
| F22 Statistiques | % terre + % mer = 100 %, durée d'inférence, nombre de fenêtres, moteur utilisé |
| F23 Incertitude | pixels dont P(terre) ∈ [0,3 ; 0,7], d'autant plus vifs que P ≈ 0,5 |
| F24 Téléchargements | masque PNG et superposition PNG, aux dimensions de l'image |
| F25 Recommencer | nouvelle image = résultats effacés explicitement, sans recharger |
| F26 Carte | pour les exemples géoréférencés : carte Leaflet avec couches image et masque prédit |

Fichiers : [front-end/src/main.ts](front-end/src/main.ts) (interface),
[worker.ts](front-end/src/worker.ts) (inférence ONNX),
[tiling.ts](front-end/src/tiling.ts) (fenêtres), [render.ts](front-end/src/render.ts)
(rendu, statistiques, exports), [map.ts](front-end/src/map.ts) (carte).

---

## 8. Limites à discuter

- **Vérité imparfaite** : le masque OSM est une carte. Il ignore la marée,
  l'estran, les chenaux temporaires et date d'une autre époque que l'image.
  Le modèle apprend « la terre selon OSM », pas la terre selon l'image.
- **Estran et marée** : la baie du Mont-Saint-Michel est le cas d'école. Selon
  la marée de l'image, de vastes surfaces sont « mer » sur la carte et
  « terre » sur l'image.
- **Domaine d'apprentissage** : uniquement la Bretagne, en couleurs naturelles
  et à environ 12,8 m/pixel. Calanques et Corse (roche calcaire claire, eaux
  turquoise) testent la généralisation. Une photo aérienne à 50 cm/pixel, une
  image en fausses couleurs ou une capture de carte sortent du domaine.
- **Pourcentages** : ce sont des parts de **pixels**. En Web Mercator, l'aire
  d'un pixel varie avec la latitude, et une image importée n'a pas d'échelle connue.
- **Probabilités** : une sortie à 0,99 n'est pas une garantie. Le réseau peut
  se tromper avec assurance hors de son domaine.
