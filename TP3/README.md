# TP3 — Cartographie bathymétrique interactive (GEBCO)

Ce TP transforme une **vraie grille bathymétrique GEBCO** (NetCDF) en fichiers
statiques qu'une carte **Leaflet** affiche selon deux modes :

- **Carte continue** : une image des profondeurs en nuances de bleu ;
- **Points** : environ 5 800 points colorés selon leur profondeur, avec popup.

Le cœur du TP est la manipulation de données scientifiques avec **xarray** et
**NumPy**. Le web ne fait que valoriser le résultat. La spécification complète
se trouve dans [brief.md](brief.md).

---

## 1. Lancer le TP

```powershell
mise run tp3            # tout : explore -> prepare -> test -> web
```

ou étape par étape :

| Tâche | Commande | Produit |
|---|---|---|
| 1. Inspecter | `mise run tp3_01_explore` | télécharge l'archive si besoin, affiche le diagnostic |
| 2. Préparer | `mise run tp3_02_prepare` | `TP3/output/` (PNG, GeoJSON, grille, metadata, rapport) |
| 3. Tester | `mise run tp3_03_test` | tests de données et points de contrôle |
| 4. Afficher | `mise run tp3_04_web` | site sur <http://localhost:5175> |

Le site ne lit **jamais** le NetCDF : il ne consomme que les exports de
`TP3/output/`, servis comme fichiers statiques par Vite (`publicDir`).

---

## 2. La donnée

| | |
|---|---|
| Produit | **GEBCO_2026 Grid** (publiée le 2026-04-17), 15″ d'arc |
| DOI | [10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa](https://doi.org/10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa) |
| Extrait | `gebco_2026_n60.0_s40.0_w-10.0_e10.0.nc` (46 Mo), préparé sur <https://download.gebco.net/> le 10/10/2026 |
| Archive | <https://jlg-consulting.com/orsys/DRN/shom/GEBCO_10_Oct_2026_55ebad3abb35.zip> (archive inchangée, avec la documentation et les conditions d'utilisation en PDF) |
| Licence | domaine public, utilisation gratuite sous réserve d'accepter les conditions GEBCO ; citation : *GEBCO Compilation Group (2026) GEBCO 2026 Grid* |
| Avertissement | **ne pas utiliser pour la navigation** |

L'archive est conservée telle quelle dans `TP3/data/raw/`, et
`TP3/data/raw/SOURCE.json` consigne l'URL, la date, la taille et le SHA-256.
L'extrait couvre 40–60° N / 10° O–10° E ; le pipeline en garde l'emprise du TP.

### Ce que révèle l'inspection (`explore.py`)

- une seule variable utile, **`elevation`** (float32, mètres), repérée par
  `standard_name = height_above_mean_sea_level` et non par son nom ;
- les coordonnées `lat`/`lon` (WGS84, EPSG:4326) sont **croissantes** et
  désignent les **centres** des cellules (`node_offset = 1`) ;
- le pas est de 0,0041667° = 15″ ≈ 460 m en latitude ;
- dans l'emprise : 1 440 × 2 160 cellules, aucune valeur manquante, élévation
  de −5 079 m à +2 527 m.

Toute hypothèse fausse (unité, CRS, axes décroissants, emprise non couverte)
arrête le script avec un message explicite (voir [gebco.py](gebco.py)).

---

## 3. Le pipeline Python (`prepare_bathymetry.py`)

1. **Sélection** des cellules dont le centre est dans `[-10, -1] × [43, 49]`.
2. **Masque** : `elevation >= 0` (terre et zéro) ou valeur non finie → exclue.
3. **Profondeur** : `depth_m = -elevation` sur les cellules marines, sans arrondi.
4. **Statistiques** : 2 543 039 cellules marines, 567 361 terrestres. La
   profondeur moyenne **des cellules** est de 2 222 m. Ce n'est *pas* une moyenne
   surfacique, car les cellules lat/lon rétrécissent vers le nord ; pondérée par
   `cos(lat)`, elle vaut 2 239 m. La distribution est bimodale : plateau
   continental (100–200 m) et plaine abyssale (4 000–5 000 m).
5. **Exports** dans `TP3/output/` :

| Fichier | Contenu | Taille |
|---|---|---|
| `bathymetry_3857.png` | raster RGBA reprojeté en Web Mercator, transparent sur terre | ≈ 1,5 Mo |
| `points.geojson` | `FeatureCollection` de `Point` `[lon, lat]` + `depth_m` | ≈ 0,65 Mo |
| `depth_grid.bin` | grille Float32 (720 × 1 080) pour la profondeur au clic | ≈ 3,1 Mo |
| `metadata.json` | bornes, palette, dimensions, source, traitements | ≈ 4 Ko |
| `report.json` | rapport complet : provenance, statistiques, échantillonnage, avertissements | |

### Le point critique : la projection

Leaflet affiche en **Web Mercator (EPSG:3857)** : les latitudes y sont
étirées vers le nord. Une image en lat/lon posée telle quelle avec
`L.imageOverlay` serait **décalée** : de 0,08° (≈ 9 km) au centre de
l'emprise.

Solution retenue : **reprojeter l'image en Python**. En Mercator, `x` est
linéaire en longitude (on garde les colonnes) et seul `y` est rééchantillonné.
Chaque ligne d'image correspond à une ordonnée Mercator régulière, convertie en
latitude, puis en ligne source la plus proche. Comme Leaflet étire l'image
linéairement *dans l'espace projeté*, l'alignement devient exact. La hauteur
(2 076 px pour 2 160 de large) est calculée pour obtenir des pixels carrés.

### Échantillonnage des points

On prend **une cellule sur 21** dans chaque direction, en partant de l'indice
10, et on ne garde que les cellules marines. On obtient 5 818 points espacés
d'environ 5′15″ (≈ 9,7 km) : c'est 1 cellule sur 441. La méthode est
déterministe et reproductible. Le pas est calculé pour viser environ 6 000
points (constante `CIBLE_POINTS`).

### Palette commune

| Profondeur | 0 m | 200 m | 1 000 m | 2 000 m | 3 500 m | 5 000 m |
|---|---|---|---|---|---|---|
| Couleur | `#C9F2F6` | `#79D2E7` | `#339AC9` | `#1761A4` | `#10346F` | `#06183C` |

Les couleurs sont interpolées linéairement en profondeur. Au-delà de 5 000 m
(91 cellules), la couleur est **saturée**. La palette est écrite dans
`metadata.json`, que le site relit. Python (raster) et TypeScript (points)
partagent donc **la même source** : une profondeur donnée a la même couleur
dans les deux modes.

---

## 4. Le site Leaflet (`front-end/`)

- carte cadrée sur l'emprise et fond OpenStreetMap ; l'attribution OSM et la
  citation GEBCO (DOI) sont affichées ;
- `L.control.layers` avec deux **boutons radio** : les modes sont exclusifs,
  sans rechargement ;
- les points sont des `circleMarker` (rayon fixe, rendu canvas) avec une popup
  latitude / longitude / profondeur ;
- **profondeur au clic en mode raster (F10)** : la valeur est lue dans
  `depth_grid.bin`, **jamais déduite de la couleur** du PNG ;
- la légende verticale « Profondeur (m) » apparaît dans les deux modes ;
- une ressource absente ou invalide affiche un message lisible qui indique la
  commande à relancer.

Fichiers : [main.ts](front-end/src/main.ts) (carte, couches, interactions),
[data.ts](front-end/src/data.ts) (chargements et erreurs),
[palette.ts](front-end/src/palette.ts) (couleurs et légende).

---

## 5. Vérifications (`tests/test_preparation.py`)

`uv run pytest TP3/tests -v` exécute 21 tests :

- **Palette** : couleurs exactes aux paliers, saturation, assombrissement
  monotone ; `metadata.json` identique à la palette Python.
- **Données** : emprise exportée = emprise cible ; GeoJSON valide, ordre
  `[lon, lat]`, 3 000 à 10 000 points ; profondeurs des points **égales** aux
  valeurs NetCDF et **aucun point sur une cellule terrestre** ; grille F10
  identique à la source ; statistiques du rapport recalculées à l'identique.
- **Points de contrôle cartographiques** : pour chaque point de contrôle, on
  calcule le pixel Mercator correspondant et on vérifie que sa couleur est celle
  de la profondeur source (±1 cellule). Une inversion nord/sud, un échange
  lat/lon ou une reprojection manquante feraient échouer ces tests.

| Point | Lat, lon | GEBCO | Attendu |
|---|---|---|---|
| Plaine abyssale | 45,5, −6,0 | −4 752 m | > 4 000 m |
| Plateau sud-Bretagne | 47,0, −4,0 | −119 m | 50–200 m |
| Mer d'Iroise | 48,3, −4,9 | −48 m | côtier |
| Gouf de Capbreton | 43,5, −1,6 | −42 m | côtier |
| Centre Bretagne | 48,2, −3,5 | +254 m | transparent / NaN |
| Landes | 44,0, −1,2 | +47 m | transparent / NaN |

Contrôle visuel : en zoomant sur Brest, l'estuaire de la Gironde ou la côte
basque, le trait de côte du raster doit coïncider avec celui d'OSM, sans dérive
au zoom ni au déplacement.

---

## 6. Limites scientifiques

- GEBCO est une **grille compilée** (sondages, altimétrie satellitaire,
  interpolation) : beaucoup de valeurs au large sont des **estimations**, pas
  des mesures directes.
- Le masque terre/mer repose sur le seul signe de l'élévation : à 15″ (≈ 460 m),
  les zones intertidales, les estuaires et les petites îles sont approximatifs.
- Le raster utilise le plus proche voisin, sans lissage. La grille F10 garde
  1 cellule sur 2 et les points 1 cellule sur 441 : un point ne représente pas
  la moyenne de sa zone.
- Les données ne doivent **pas servir à la navigation**.
