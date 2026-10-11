# TP6 — Ocean Terrain Lab : générer des reliefs sous-marins avec un VAE et un GAN

Ce TP entraîne deux **modèles génératifs** (PyTorch) sur des cartes bathymétriques
réelles **GEBCO** de 64 × 64 pixels (≈ 118 km de côté) :

- un **autoencodeur variationnel (VAE)**, qui sait **reconstruire** un relief et en **générer** de nouveaux ;
- un **réseau antagoniste génératif (GAN)**, qui sait seulement **générer**.

Ils sont exportés en **ONNX** pour l'application web **Ocean Terrain Lab**, qui
génère, affiche en 2D et en 3D, explore les espaces latents et compare les deux
modèles **directement dans le navigateur** (onnxruntime-web + three.js, sans
serveur Python). La spécification complète est dans [brief.md](brief.md).

> ⚠ Les reliefs produits sont **synthétiques** et expérimentaux. Ce ne sont ni
> des relevés, ni des représentations géologiquement validées : ne pas les
> utiliser pour la navigation ou l'hydrographie.

---

## 1. Lancer le TP

```powershell
mise run tp6            # tout : download -> prepare -> VAE -> GAN -> generate -> evaluate -> export -> test -> web
```

ou étape par étape :

| Tâche | Commande | Produit |
|---|---|---|
| 1. Acquérir | `mise run tp6_01_download` | `data/raw/<region>.nc` (13 extraits GEBCO), `data/provenance.json` |
| 2. Préparer | `mise run tp6_02_prepare` | `data/dataset/` (`patches.npz`, `index.json`, `dataset.json`, `rapport.json`), `data/figures/*.png` |
| 3. VAE | `mise run tp6_03_train_vae` | `runs/vae/` (`best.pt`, `history.json`, `config.json`, `courbes.png`, `exemples.png`) |
| 3 bis. Sans GPU | `mise run tp6_03_train_vae_cpu` | idem, configuration compacte |
| 4. GAN | `mise run tp6_04_train_gan` | `runs/gan/` (`best.pt`, `last.pt`, `history.json`, `config.json`, `courbes.png`, `evolution.npz/.png`) |
| 4 bis. Sans GPU | `mise run tp6_04_train_gan_cpu` | idem, configuration compacte |
| 5. Générer | `mise run tp6_05_generate` | `runs/generation/` (galeries, interpolations, composantes latentes, reconstruction) |
| 6. Évaluer | `mise run tp6_06_evaluate` | `runs/evaluation.json`, `runs/synthese.md`, `runs/evaluation/*.png` |
| 7. Exporter | `mise run tp6_07_export` | `model/` (3 graphes ONNX, `reels.bin`, `evolution.bin`, `metadata.json`) |
| 8. Tester | `mise run tp6_08_test` | tests unitaires, de données et d'export |
| 9. Utiliser | `mise run tp6_09_web` | Ocean Terrain Lab sur <http://localhost:5178> |

Le site ne dépend que du dossier `TP6/model/` (≈ 9 Mo), servi tel quel par Vite
(`publicDir`). `bun run build:tp6` produit un site 100 % statique (NF04).

**Durées mesurées (NF02, NF03).** Sur une GTX 1050 (4 Go) : VAE 6,4 min
(60 époques), GAN 50 min (300 époques). Sans GPU, en configuration compacte
(`--rapide --cpu`, largeur 16) : VAE 5,7 min (15 époques, RMSE de validation
167 m, presque celle de la version complète) et GAN 11 min (30 époques, écart
statistique 0,54 contre 0,17 : reliefs encore grossiers, mais l'évolution est
déjà visible). Le GAN est l'étape longue de la séance.

**Checkpoints préentraînés (NF03).** Comme aux TP précédents, `runs/` et
`model/` sont recréés par les tâches `mise` et ne sont pas versionnés. Pour une
séance, distribuez le dossier `TP6/runs/` (≈ 28 Mo) déjà entraîné : les étapes 5
à 9 fonctionnent alors sans réentraîner, pendant que les participants lancent la
configuration compacte (`--rapide`).

Organisation du code, sur le modèle des TP4 et TP5 :

| Fichier | Rôle |
|---|---|
| [terrain.py](terrain.py) | module commun : régions, grille, patches, normalisation, indicateurs, graines, palette |
| [download_data.py](download_data.py) | F01 : acquisition GEBCO (ERDDAP Ifremer) |
| [prepare_dataset.py](prepare_dataset.py) | F02 à F06 : terres, patches, diversité, séparation spatiale, normalisation, figures |
| [modeles.py](modeles.py) | F07, F13 : VAE, générateur, discriminateur, graphes d'export |
| [train_vae.py](train_vae.py) | F08, F09 : entraînement du VAE |
| [train_gan.py](train_gan.py) | F14 à F16 : entraînement du GAN |
| [generate.py](generate.py) | F10 à F12, F17, F18 : génération, interpolation, composantes, reconstruction |
| [evaluate.py](evaluate.py) | F19 à F23 : comparaison VAE / GAN |
| [export_onnx.py](export_onnx.py) | F24 à F26 : ONNX vérifié, données du site |
| [front-end/](front-end/) | F27 à F37 : application Ocean Terrain Lab |

---

## 2. Les données (F01, F02)

| | |
|---|---|
| Produit | **GEBCO_2021 Grid**, 15″ d'arc ([doi:10.5285/c6612cbe-50b3-0cff-e053-6c86abc09f8f](https://doi.org/10.5285/c6612cbe-50b3-0cff-e053-6c86abc09f8f)) |
| Accès | ERDDAP de l'Ifremer, `erddap.ifremer.fr/erddap/griddap/gebco2021`, **sans compte ni clé**, une requête par emprise |
| Licence | domaine public, sous réserve des conditions d'utilisation GEBCO ; **pas pour la navigation** |
| Traçabilité | `data/provenance.json` : emprise, URL exacte, date, taille et SHA-256 de chaque extrait |

Le TP3 utilisait un extrait GEBCO_2026 du golfe de Gascogne déposé à la main.
Ici, il faut des **morphologies variées** dans plusieurs océans. L'ERDDAP permet
de demander chaque région par une simple URL, ce qui rend le dataset
reproductible (77 Mo au total).

Les 13 régions d'intérêt (`terrain.REGIONS`) :

| Région | Emprise | Morphologies | Terre |
|---|---|---|---|
| Dorsale médio-atlantique (Açores) | 35–41° N, 36–27° O | dorsale, vallée axiale, failles, volcans | 0,4 % |
| Dorsale de Reykjanes | 56–62° N, 36–24° O | dorsale lente, rides volcaniques obliques | 0 |
| Monts Great Meteor | 27–33° N, 32–25° O | guyots, collines abyssales | 0 |
| Plaine de Madère, ride Madère-Tore | 33,5–38° N, 26–17° O | plaine abyssale, ride volcanique | 0,3 % |
| Madère et Canaries | 27–33,5° N, 20–13° O | flancs d'îles volcaniques, glissements | 2,6 % |
| Marge ibérique, banc de Gorringe | 34–40° N, 16–8° O | canyons (Nazaré, Setúbal), banc, plaines | 6,5 % |
| Golfe de Gascogne | 43–48,5° N, 11–1° O | talus, canyons, plaine abyssale | 16,9 % |
| Fossé et banc de Rockall | 51–58° N, 20–10° O | bancs, fossé, pentes | 0,4 % |
| Monts de Nouvelle-Angleterre | 35–41° N, 68–58° O | chaîne de monts, canyons de marge | 0 |
| Méditerranée occidentale | 38–44,3° N, 3–10° E | canyons du golfe du Lion et de Ligurie | 20,5 % |
| Dorsale Est-Pacifique | 14–8° S, 108–100° O | dorsale rapide, collines abyssales alignées | 0 |
| Monts des Musiciens | 26–32° N, 165–157° O | monts isolés, rides volcaniques | 0 |
| Fosse de Porto Rico | 19–23,5° N, 70–61° O | fosse de subduction, pentes raides | 0,9 % |


---

## 3. Préparation du dataset (`prepare_dataset.py`, F02 à F06)

1. **Grille de travail.** Moyenne par blocs de la grille 15″ : 4 cellules en
   latitude (1′ = 1,855 km) et `round(4 / cos φ)` en longitude, pour des pixels à
   peu près **carrés** (1,73 à 2,03 km selon la région). La moyenne évite le
   repliement d'un simple sous-échantillonnage.
2. **Terres émergées (F02).** Un pixel est « terre » si **une seule** cellule 15″
   du bloc a une élévation ≥ 0. Tout patch qui touche la terre est **exclu**
   (2 476 candidats).
3. **Patches 64 × 64 (F03)** tous les 8 pixels. `index.json` garde pour chacun
   la région, la ligne, la colonne, l'emprise géographique et la profondeur
   moyenne.
4. **Séparation spatiale (F05).** Chaque région est découpée en blocs de
   128 × 128 pixels (≈ 240 km). Dans chaque région, un bloc complet et marin est
   tiré pour la validation, un autre pour le test (deux pour Porto Rico). Un
   patch n'est gardé que si **toute** son emprise est dans un même ensemble : les
   patches d'ensembles différents ne se recouvrent jamais (6 029 candidats à
   cheval écartés ; vérifié par les tests). Chaque région est ainsi présente
   dans les trois ensembles.
5. **Diversité (F02).** Les patches presque plats (écart-type < 30 m) sont
   limités à 10 % ; dans les faits ils sont très rares (10 en entraînement), car
   à 1,85 km par pixel même les plaines abyssales montrent des collines.
6. **Taille (F05).** Tirage uniforme dans chaque ensemble vers **5 000 patches** :
   3 500 / 750 / 750 (70 / 15 / 15 %).
7. **Normalisation (F04).** `z = clip((élévation − moyenne du patch) / 1 500 m, −1, 1)`.

   Les modèles apprennent des **formes** de relief, pas une profondeur absolue :
   une dorsale à 2 500 m et une à 3 500 m ont la même forme. L'échelle de 1 500 m
   est commune à tous les patches, donc l'amplitude reste **physique** : un
   relief généré se reconvertit en mètres (`relief = z × 1 500`), comme écart à
   la moyenne du patch. 2,6 % des pixels d'entraînement sont écrêtés (sommets de
   monts, flancs d'îles, fosse). Le paramètre est conservé dans
   `dataset.json`, dans les checkpoints et **dans les graphes ONNX**.

Le dataset est identifié par une empreinte des paramètres et des SHA-256 des
extraits (`gebco2021-87199524ac47`) que l'on retrouve dans chaque checkpoint
(NF07).

### Exploration (F06)

`data/figures/` : `01_regions.png` (régions, terres, blocs val/test, emprises
des patches), `02_galerie.png` (galerie 2D), `03_reliefs_3d.png` (8 reliefs en
3D, du plus doux au plus accidenté), `04_distributions.png` (profondeurs
absolues, valeurs normalisées, écarts-types, amplitudes par ensemble),
`05_par_region.png`, `06_anomalies.png`.

Anomalies repérées (`rapport.json`) : patches à **rugosité relative maximale**.
Ce sont souvent des traces de levés multifaisceaux (bandes nettes) au milieu de
zones déduites de l'altimétrie satellitaire, plus lisses : GEBCO est une
mosaïque de sources. Autre anomalie : les patches à **écrêtage maximal**
(fosse, flancs d'îles).


---

## 4. Les modèles ([modeles.py](modeles.py), F07, F13)

Le **décodeur du VAE** et le **générateur du GAN** ont **exactement la même
architecture**. Le discriminateur reprend la structure de l'encodeur. Seule la
façon d'apprendre diffère, ce qui rend la comparaison équitable.

```
Encodeur (VAE) / Discriminateur (GAN)
  64×64×1 ─► conv 4×4 /2 ─► 32×32×32 ─► 16×16×64 ─► 8×8×128 ─► 4×4×256 ─► linéaire ─► μ, log σ²  (VAE)
                                                                                     └► 1 logit  (GAN)
Décodeur (VAE) = Générateur (GAN)
  z ─► linéaire ─► 4×4×256 ─► [suréchantillonnage ×2 + conv 3×3 + BN + ReLU] × 4 ─► conv 3×3 ─► tanh ─► 64×64
```

Le suréchantillonnage « plus proche voisin + convolution » évite les damiers des
convolutions transposées.

| | VAE | GAN |
|---|---|---|
| Espace latent | 16 dimensions | 32 dimensions |
| Paramètres | encodeur 821 k, décodeur 475 k | générateur 541 k, discriminateur 693 k (non exporté) |
| Perte | Σ (x − x̂)² + β · KL(N(μ, σ²) ‖ N(0, I)) | entropie croisée binaire, G « non saturant » (option `--perte hinge`) |
| Stabilisation | — | **normalisation spectrale** du discriminateur (`--sans-spectrale` pour le DCGAN classique) |
| Optimiseur | Adam 1e-3, décroissance cosinus | Adam 2e-4 / 2e-4, β1 = 0,5 (réglage DCGAN) |
| Augmentation | 8 symétries du carré | idem, sur les cartes réelles |
| Sélection | plus faible perte de validation | plus faible **écart statistique** à la validation |

### Entraînement du VAE (`train_vae.py`, F08, F09)

| Paramètre | Défaut | Option |
|---|---|---|
| Dimension latente | 16 | `--latent` (8 ou 16 conseillé) |
| Époques | 60 | `--epochs` |
| Batch | 64 | `--batch-size` |
| Taux d'apprentissage | 1e-3 | `--lr` |
| Poids du KL | 0,1 | `--beta` |
| Graine | 0 | `--seed` |
| Configuration compacte | largeur 16, 15 époques | `--rapide` (et `--cpu`) |

`history.json` enregistre à chaque époque la perte totale, la reconstruction,
le KL (entraînement et validation), la RMSE de reconstruction en mètres et la
durée. `best.pt` contient poids, configuration, normalisation et identifiant du
dataset (`modeles.construire_vae`).

**Le choix de β est une expérience à faire en séance.** Mesures sur la GTX 1050 (4 Go) :

| β | Époques | Durée | KL (val) | RMSE de reconstruction (val) | Effet |
|---|---|---|---|---|---|
| 1 | 120 | 14 min | 17 nats | 165 m | reconstructions très floues : le KL « éteint » la plupart des dimensions |
| **0,1** | 60 | 6,4 min | 40 nats | **155 m** | plus de détails, générations avec monts et rides visibles |

Même avec β = 0,1, le VAE lisse fortement : 16 nombres ne suffisent pas à
décrire la texture fine d'un relief, et la perte quadratique favorise la moyenne
des possibles. C'est la limite classique du VAE.

### Entraînement du GAN (`train_gan.py`, F14 à F16)

| Paramètre | Défaut | Option |
|---|---|---|
| Dimension latente | 32 | `--latent` |
| Époques | 300 | `--epochs` |
| Batch | 64 | `--batch-size` |
| Taux d'apprentissage G / D | 2e-4 / 2e-4 | `--lr-g`, `--lr-d` |
| Paramètres d'Adam | β1 = 0,5, β2 = 0,999 | `--beta1`, `--beta2` |
| Perte | BCE | `--perte bce\|hinge` |
| Graine | 0 | `--seed` |
| Configuration compacte | largeur 16, 30 époques | `--rapide` (et `--cpu`) |

À chaque mini-batch : une mise à jour de **D** (réels contre générés, G figé),
puis une de **G** (tromper le D qui vient d'être mis à jour). On enregistre les
pertes de D et de G, D(x) et D(G(z)), la durée, et tous les 5 époques :

- **16 cartes à z fixé** (`evolution.npz`, `evolution.png`) : on voit les reliefs se former (F15) ;
- la **diversité** des générations, rapportée à celle des réels : une chute signale un *mode collapse* ;
- l'**écart statistique** entre 512 générations et 512 cartes de validation
  (distance de Wasserstein des distributions d'amplitude, d'écart-type, de
  variabilité locale, de pente et de rugosité).

Un GAN n'a pas de perte de validation interprétable : on retient donc le
générateur de **plus faible écart statistique** (`best.pt`). `last.pt` garde la
dernière époque. Les deux fichiers contiennent G **et** D (F16), mais
`modeles.construire_generateur` recharge G seul.

**Résultats (GTX 1050, 300 époques, 50 min ; 9 s par époque).** D(x) et D(G(z))
restent collés à 0,5 et les pertes de D (≈ 1,38 = 2 ln 2) et de G (≈ 0,71)
sont **presque constantes** de la 5e à la 300e époque : la normalisation
spectrale maintient D et G en équilibre, et ces pertes ne disent **rien** de la
qualité des cartes. Pendant ce temps, l'écart statistique passe de 1,8
(époque 1) à 0,17, mais il **remonte** par moments jusqu'à 0,5 d'un contrôle à
l'autre, et la diversité varie de 0,74 à 1,39. C'est l'instabilité du GAN, vue
autrement qu'à travers les pertes. Le générateur retenu est celui de
l'**époque 125** (écart 0,166) ; `evolution.png` montre les mêmes z passer de
taches floues et de bords artificiels (époque 1) à des textures de collines,
de rides et de monts.


---

## 5. Génération et exploration (`generate.py`, F10 à F12, F17, F18)

Les vecteurs latents sont tirés par **le même générateur pseudo-aléatoire en
Python et dans le site** (mulberry32 + Box-Muller, `terrain.vecteur_latent`) :
la graine 42 donne le même z, au bit près (vérifié par les tests), donc le même
relief. `generate.py` produit pour chaque modèle une galerie (graines
consécutives), une interpolation entre deux graines et le parcours de 6
composantes de −3 à +3, ainsi que la reconstruction d'un patch de test par le
VAE (original, reconstruit, carte d'erreur, RMSE et erreur maximale).

```powershell
uv run python TP6/generate.py --modele gan --graine 42 --n 32
uv run python TP6/generate.py --patch 120 --graine 3 --graine-b 7
```


---

## 6. Évaluation comparative (`evaluate.py`, F19 à F23)

Sur les 750 patches de **test** et 750 générations par modèle :

| | Réel (test) | VAE | GAN |
|---|---|---|---|
| Amplitude médiane | 1750 m | 1066 m | 1357 m |
| Écart-type médian | 295 m | 197 m | 239 m |
| Pente moyenne médiane | 1.47° | 0.73° | 1.30° |
| Pente des pixels, 90e centile | 4.7° | 1.6° | 3.5° |
| Rugosité médiane | 24 m | 7 m | 22 m |
| Pixels saturés (±1 500 m) | 1.9 % | 0.0 % | 0.9 % |
| **Écart aux réels** (W1 relatif moyen, 0 = identique) | — | **0.85** | **0.24** |
| Diversité (distance moyenne entre paires / réels) | 1,00 | 0.59 | 1.01 |
| Quasi-doublons (corrélation ≥ 0,95) | 0 % | 0 % | 0 % |
| Distance médiane au plus proche patch d'entraînement | 175 m | 158 m | 200 m |
| Reconstruction (RMSE sur le test) | — | 163 m (médiane par patch 117 m) | impossible |
| Entraînement | — | 6,4 min (60 ép.) | 50 min (300 ép.) |

À retenir :

- **Le VAE lisse.** Rugosité divisée par 3,5, pentes deux fois plus faibles,
  diversité de 0,59 : il produit des « moyennes » de reliefs. En contrepartie,
  son apprentissage est **stable** (validation qui décroît régulièrement), son
  espace latent est **régulier** (interpolations continues, chaque composante a
  un effet progressif), et lui seul **reconstruit** un relief donné.
- **Le GAN est plus réaliste statistiquement** (écart 0,24 contre 0,85) : pentes,
  rugosité et diversité proches des réels. Mais il **sature** plus souvent à
  ±1 500 m que les réels (plateaux artificiels) et présente encore quelques
  artefacts de grille. Son apprentissage ne donne aucun signal de convergence.
- **Pas de mode collapse mesuré** sur ce run (0 % de quasi-doublons, diversité
  1,01). Pour le provoquer en séance : `--sans-spectrale --lr-d 1e-3`, puis
  observer la courbe de diversité et la galerie.
- **Pas de copie évidente** : un patch de test réel est en médiane à 175 m
  (RMS) de son plus proche voisin d'entraînement, une génération du GAN à 200 m.
  Celles du VAE sont un peu plus proches (158 m), effet du lissage : en distance
  quadratique, une carte lisse est proche de nombreux patches doux.
- La RMSE de reconstruction est **dominée par les reliefs forts** (dorsales,
  fosse) : 117 m pour un patch médian, plus de 300 m sur les reliefs accidentés.

Fichiers : `runs/evaluation/` — `cote_a_cote.png` (F20 : réel, VAE, GAN avec la
même palette et la même échelle, **sans correspondance** entre colonnes),
`diversite.png` (F21), `statistiques.png` (F22), `reconstructions.png` (F10 :
meilleures, médianes et pires), `courbes.png` (F19, deux panneaux séparés) ;
`runs/synthese.md` (F23).



---

## 7. Export ONNX (`export_onnx.py`, F24 à F26)

| Graphe | Entrée | Sortie | Taille |
|---|---|---|---|
| `vae_encodeur.onnx` | `elevation` (N, 64, 64), m | `mu`, `logvar` (N, 16) | 3,3 Mo |
| `vae_decodeur.onnx` | `z` (N, 16) | `relief` (N, 64, 64), m | 1,9 Mo |
| `gan_generateur.onnx` | `z` (N, 32) | `relief` (N, 64, 64), m | 2,2 Mo |

Comme au TP5, la normalisation est **dans les graphes** : l'encodeur centre le
patch, divise par 1 500 m et écrête ; les générateurs multiplient par 1 500 m. Le
site n'a aucune constante à recopier. Le discriminateur n'est pas exporté.

Vérifications (F26) : `onnx.checker`, puis ONNX Runtime contre PyTorch sur les
mêmes entrées (750 patches de test pour l'encodeur, 256 vecteurs latents pour
les générateurs). Écarts maximaux : encodeur 9e-06, décodeur VAE 1.3 mm, générateur GAN 5.3 mm (seuil : 5 cm). `metadata.json` contient aussi un **exemple de
contrôle** (graine 0) que le site recalcule au chargement et dont il affiche
l'écart.

---

## 8. L'application web Ocean Terrain Lab

| Fonction | Réalisation |
|---|---|
| F27 Modèle | barre fixe **VAE / GAN / Comparaison VAE / GAN**, couleur du modèle actif (bleu VAE, orange GAN) |
| §9.2 Espaces | onglets **1. Génération**, **2. Espace latent**, **3. Reconstruction (VAE)** ; la comparaison est le troisième mode |
| F28 Générer | « Générer un relief » (nouvelle graine) et « Regénérer avec cette graine » ; une génération prend quelques ms |
| F29 2D | palette bathymétrique, légende graduée en mètres, échelle commune ±1 500 m ou adaptée, valeur et position au survol |
| F30 3D | surface three.js calculée sur les 64 × 64 valeurs : rotation, zoom, déplacement, vues « Dessus » et « Rasante », réinitialisation, exagération verticale ×1 à ×30, flèche du nord |
| F31 Latent | curseurs de −3 à +3, 6 composantes à la fois (pages z1–z6, z7–z12…), régénération à chaque mouvement, « Revenir au vecteur initial » |
| F32 Interpolation | graines A et B, curseur t, animation aller-retour, bande de 5 étapes cliquables |
| F33 Reconstruction | 64 patches réels de **test** filtrables par région ; réel, reconstruit et erreur côte à côte, RMSE et erreur maximale ; GAN désactivé (pas d'encodeur) |
| F34 Comparaison | trois panneaux **Réel / VAE / GAN** en 2D + 3D, même échelle et même exagération, graines indépendantes, avertissement d'indépendance |
| F35 Statistiques | min, max, moyenne, amplitude, rugosité, pente moyenne, en **mètres** et en **valeurs normalisées** (colonnes distinctes) |
| F36 Exports | PNG 2D (avec légende et mention « synthétique »), PNG de la vue 3D, CSV de la grille, JSON des paramètres (modèle, graine, z, interpolation, normalisation, statistiques) |
| F37 Galerie | 8, 12 ou 24 reliefs à graines consécutives ; clic pour examiner |
| F19, F21, F22, F23, F15 | en mode comparaison : galeries de diversité, tableau des statistiques de test, courbes d'apprentissage séparées, GAN au fil des époques, synthèse |
| §9.4 Retours | bandeau d'état : chargement, génération en cours, terminée (durée), erreurs, écart du contrôle ONNX |

Fichiers : [main.ts](front-end/src/main.ts) (interface),
[donnees.ts](front-end/src/donnees.ts) (chargement),
[modeles.ts](front-end/src/modeles.ts) (ONNX, graines),
[rendu2d.ts](front-end/src/rendu2d.ts), [vue3d.ts](front-end/src/vue3d.ts),
[stats.ts](front-end/src/stats.ts) (mêmes définitions qu'en Python),
[graphiques.ts](front-end/src/graphiques.ts), [exports.ts](front-end/src/exports.ts).

---

## 9. Limites

- **Formes relatives seulement.** Les modèles ne génèrent ni profondeur absolue,
  ni position, ni contexte géologique. Une « dorsale » générée n'a aucune
  cohérence tectonique.
- **Échelle unique de 120 km à 1,85 km par pixel.** Rien n'assure la continuité
  entre deux reliefs voisins : pas de grande carte continue (hors périmètre).
- **Écrêtage à ±1 500 m** autour de la moyenne : les sommets de monts, les fosses
  et les flancs d'îles sont tronqués, et le GAN reproduit (voire exagère) ces
  plateaux saturés.
- **Données hétérogènes.** GEBCO mélange levés multifaisceaux et altimétrie
  satellitaire : les modèles apprennent aussi ces artefacts.
- **Évaluation statistique.** Les indicateurs comparent des distributions. Une
  ressemblance statistique n'est **pas** une preuve de réalisme géologique.
- **GAN.** Aucun critère de convergence ; le résultat dépend de l'époque retenue
  et de la graine d'entraînement.
