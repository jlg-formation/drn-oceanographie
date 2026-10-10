# TP3.1 — Exploration et cartographie bathymétrique interactive

**Formation :** Deep Learning et réseaux de neurones — les fondamentaux  
**Chapitre :** 3 — Outils usuels machine learning et deep learning  
**Version :** 1.0 — Spécification de réalisation  
**Durée cible :** 2 h 30 (avec environnement et squelette web préparés)

## 1. Contexte et objectif

Créer une application web statique permettant d'explorer la bathymétrie du golfe de Gascogne à partir d'un **véritable sous-ensemble NetCDF GEBCO**, préparé par un pipeline Python. L'utilisateur doit pouvoir basculer entre :

- **Carte continue** : représentation raster des profondeurs en nuances de bleu.
- **Carte de points** : points géographiques échantillonnés et colorés selon leur profondeur.

Le cœur pédagogique porte sur l'exploration et la transformation de données scientifiques avec **xarray et NumPy**. Leaflet sert à valoriser les résultats. Aucun entraînement de modèle ML n'est demandé.

## 2. Source de données

- **Source :** GEBCO, grille bathymétrique mondiale ; site officiel : <https://www.gebco.net/> ; téléchargement des sous-ensembles : <https://download.gebco.net/>.
- **Format requis :** NetCDF (`.nc`).
- **Emprise cible (WGS84) :** ouest `-10`, est `-1`, sud `43`, nord `49` (degrés décimaux).
- **Variable attendue :** `elevation`, en mètres, **à confirmer à l'inspection du fichier effectivement téléchargé**.
- **Convention de données :** altitude positive au-dessus du niveau de référence, valeurs négatives sous celui-ci ; profondeur d'affichage `depth_m = -elevation` pour les seules cellules `elevation < 0`.
- **Version GEBCO :** utiliser la version disponible sur le portail lors du téléchargement ; noter version, URL, date et licence dans le rapport. Ne pas supposer qu'une édition donnée est disponible avant vérification.
- **Provenance :** grille compilée pouvant inclure mesures et estimations ; ne pas présenter toutes les profondeurs comme des mesures directes.

### Acquisition

1. Télécharger depuis le portail officiel un sous-ensemble couvrant l'emprise cible au format NetCDF.
2. Conserver le fichier brut inchangé dans `data/raw/`.
3. Documenter nom, taille, origine, date de récupération et conditions de réutilisation.
4. Vérifier que le fichier s'ouvre et couvre la zone demandée ; en cas d'indisponibilité du portail, signaler le blocage plutôt que générer silencieusement de fausses données.

## 3. Architecture

```text
NetCDF GEBCO (data/raw/)
       |
       v
Python : xarray + NumPy
  - inspection / validation
  - sélection géographique
  - masque terre/mer
  - profondeur positive
  - préparation cartographique
       |
       +--> raster bathymétrique + géoréférencement
       |
       +--> GeoJSON de points échantillonnés
       |
       +--> métadonnées / rapport / grille facultative
       v
Application web statique TypeScript + Leaflet
  - fond cartographique
  - mode raster OU mode points
  - légende commune
  - zoom, déplacement, interrogation des points
```

**Pas de backend Python permanent** : le prétraitement s'exécute hors ligne ; le site consomme des fichiers statiques. Un serveur HTTP local est nécessaire pour servir les ressources.

## 4. Exigences fonctionnelles

| ID | Exigence | Priorité | Critère d'acceptation |
|---|---|---|---|
| F01 | Carte initialement centrée sur l'emprise du golfe de Gascogne | Obligatoire | L'emprise est visible à l'ouverture. |
| F02 | Fond de carte affichant les côtes | Obligatoire | Côtes et repères géographiques visibles ; attribution affichée. |
| F03 | Mode « Carte continue » | Obligatoire | Les profondeurs sont représentées par un raster correctement positionné. |
| F04 | Mode « Points » | Obligatoire | Les points GeoJSON apparaissent à leur position réelle. |
| F05 | Bascule exclusive des modes | Obligatoire | Un contrôle Leaflet active l'un des deux modes, sans rechargement. |
| F06 | Palette bathymétrique commune | Obligatoire | Une même profondeur correspond à la même couleur dans les deux modes. |
| F07 | Légende des profondeurs | Obligatoire | Unités en mètres, valeurs croissantes vers les grandes profondeurs. |
| F08 | Information au clic sur un point | Obligatoire | Popup avec latitude, longitude et profondeur en mètres. |
| F09 | Navigation cartographique | Obligatoire | Zoom et déplacement fonctionnels. |
| F10 | Profondeur au clic en mode raster | Optionnel | Valeur récupérée dans une grille numérique associée, pas déduite de la couleur PNG. |
| F11 | État de chargement et erreurs lisibles | Obligatoire | Une ressource absente ou invalide produit un message compréhensible. |

## 5. Spécification du pipeline Python

### 5.1 Inspection et validation

- Ouvrir le NetCDF avec `xarray` (moteur approprié à déterminer après inspection).
- Afficher dimensions, coordonnées, variables, attributs globaux, unités et valeurs manquantes.
- Identifier explicitement la variable d'élévation et les noms des coordonnées ; ne pas supposer leur nom sans contrôle.
- Vérifier que les coordonnées sont géographiques ou transformer explicitement le système de référence si nécessaire.
- Vérifier la couverture de l'emprise cible, l'orientation des axes et les valeurs extrêmes.
- Échouer avec un message explicite si une hypothèse de format est fausse.

### 5.2 Transformation scientifique

- Extraire l'emprise `[-10, -1] × [43, 49]`.
- Masquer les cellules `elevation >= 0` (terre et niveau zéro) ainsi que les valeurs invalides.
- Définir `depth_m = -elevation` sur les cellules marines valides.
- Conserver les profondeurs en mètres sans arrondi dans les traitements internes ; arrondir uniquement pour l'affichage.
- Calculer : dimensions de la grille, nombre de cellules marines/terrestres/invalides, profondeur minimale et maximale, profondeur moyenne **des cellules** et distribution des profondeurs. Ne pas qualifier la moyenne non pondérée de moyenne surfacique : les cellules en latitude/longitude n'ont pas toutes la même aire.

### 5.3 Export raster

- Produire une image bathymétrique colorée, avec transparence sur les cellules terrestres/invalides.
- Fixer une palette et des bornes de profondeur documentées, communes aux deux modes ; les valeurs hors plage sont saturées explicitement.
- Produire un fichier de métadonnées comprenant les bornes géographiques, les dimensions, la palette, la version des données et le traitement appliqué.
- **Point critique :** Leaflet affiche généralement en Web Mercator (EPSG:3857). Une image brute en latitude/longitude ne doit pas être simplement placée avec `L.imageOverlay` en supposant que les pixels s'alignent correctement. **Choix recommandé : reprojeter le raster vers EPSG:3857 et publier des tuiles XYZ**, ou utiliser une couche raster Leaflet capable de gérer la projection et les coordonnées du raster. Si `imageOverlay` est retenu, documenter et tester sa validité géométrique sur toute l'emprise ; ne pas déclarer l'alignement correct sans vérification.
- La solution retenue doit rester statique côté web.

### 5.4 Export GeoJSON

- Générer une `FeatureCollection` de points échantillonnés sur la grille marine.
- Chaque `Feature` est un `Point` avec coordonnées **`[longitude, latitude]`** et propriété numérique `depth_m`.
- Définir une stratégie d'échantillonnage reproductible (pas régulier ou réduction déterministe) et documenter la perte de résolution.
- Cible initiale : **3 000 à 10 000 points** ; ajuster selon la fluidité du navigateur.
- Ne pas exporter toutes les cellules de la grille sous forme de marqueurs Leaflet.
- Écarter les valeurs invalides et les cellules terrestres.

### 5.5 Rapport de préparation

Produire `data/processed/report.json` (ou Markdown) avec : provenance du fichier, dimensions, emprise, résolution native, nombre de cellules, statistiques, stratégie d'échantillonnage, nombre de points, palette, fichiers exportés et avertissements scientifiques.

## 6. Spécification de l'application Leaflet

- **Technologies :** TypeScript, Leaflet, HTML/CSS ; Vite est autorisé pour le développement et le build.
- **Carte :** projection compatible avec le fond choisi ; position initiale calculée sur l'emprise.
- **Couches :** une couche de fond, une couche bathymétrique raster et une couche GeoJSON ; les deux couches bathymétriques sont mutuellement exclusives via `L.control.layers` ou un contrôle équivalent.
- **Points :** préférer `circleMarker` (rayon constant en pixels) aux marqueurs HTML ; couleur déterminée par `depth_m` ; contour discret ; popup au clic.
- **Raster :** rendu transparent sur les terres, avec géoréférencement vérifié ; le fond de carte reste visible là où le raster est transparent.
- **Légende :** visible dans les deux modes, titre « Profondeur (m) », dégradé du bleu clair (peu profond) au bleu foncé (profond).
- **Interactions :** zoom, déplacement, bascule des couches, popup de point ; F10 facultatif.
- **Accessibilité minimale :** libellés explicites, contraste suffisant, messages d'erreur lisibles, contrôle utilisable au clavier dans la mesure du support Leaflet.
- **Attributions :** afficher l'attribution du fond cartographique et la mention de la source GEBCO selon sa licence et ses conditions de citation.

## 7. Palette et conventions visuelles

Palette indicative, à ajuster après inspection des valeurs réelles :

| Profondeur | Couleur indicative |
|---:|---|
| 0 m | `#C9F2F6` |
| 200 m | `#79D2E7` |
| 1 000 m | `#339AC9` |
| 2 000 m | `#1761A4` |
| 3 500 m | `#10346F` |
| 5 000 m | `#06183C` |

Interpolation continue entre les points de la palette. Les profondeurs positives représentent l'eau sous le niveau de référence ; la terre n'est pas coloriée comme de l'eau. Le point GeoJSON conserve une taille fixe ; **seule sa couleur** code la profondeur.

## 8. Arborescence cible

```text
tp3-1-bathymetrie/
├── README.md
├── docs/
│   └── SPECIFICATION.md
├── data/
│   ├── raw/                  # NetCDF téléchargé (non modifié)
│   └── processed/            # GeoJSON, raster/tuiles, metadata, rapport
├── python/
│   ├── inspect_dataset.py
│   └── prepare_bathymetry.py
├── web/
│   ├── index.html
│   ├── src/
│   │   ├── main.ts
│   │   └── style.css
│   └── public/data/          # exports publiables, éventuellement copiés au build
└── tests/
    ├── test_preparation.py
    └── ...
```

Les noms des scripts sont indicatifs. Les fichiers bruts volumineux peuvent être exclus de Git, mais le README doit fournir une procédure reproductible pour les récupérer. Les artefacts destinés au site doivent être présents dans le dossier servi ou copiés automatiquement au build.

## 9. Contraintes de qualité et tests

### Tests de données

- La grille exportée couvre l'emprise attendue et les coordonnées restent dans les plages valides.
- Les profondeurs exportées sont non négatives, finies et correspondent aux valeurs NetCDF d'origine à la précision prévue.
- Aucun point GeoJSON n'est situé sur une cellule classée terrestre dans la source (sous réserve de la résolution de la grille).
- Le GeoJSON est valide, avec coordonnées dans l'ordre longitude/latitude.
- Les statistiques annoncées dans le rapport sont reproductibles.

### Tests cartographiques

- Vérifier au moins **trois points de contrôle** (près des côtes et au large) entre grille source et carte pour détecter inversion nord/sud, longitude/latitude ou défaut de reprojection.
- Le raster ne dérive pas par rapport aux côtes lors du zoom et du déplacement.
- La même profondeur reçoit la même couleur dans les deux modes.
- Le changement de mode n'entraîne ni rechargement de page ni duplication des couches.
- Les popups affichent des valeurs plausibles et cohérentes avec la source.

### Performance

- L'application doit rester fluide sur un ordinateur de formation standard.
- Les volumes GeoJSON et raster doivent être mesurés dans le rapport ; ajuster la densité des points si nécessaire.
- Ne pas imposer le chargement du NetCDF complet dans le navigateur.

## 10. Déroulé pédagogique proposé (2 h 30)

| Étape | Durée | Production |
|---|---:|---|
| 1. Inspecter le NetCDF | 20 min | Diagnostic des dimensions, variables et attributs |
| 2. Transformer les données | 30 min | Grille de profondeurs marines et statistiques |
| 3. Exporter les deux représentations | 35 min | Raster géoréférencé et GeoJSON |
| 4. Intégrer dans Leaflet | 45 min | Carte interactive avec contrôle de couches |
| 5. Vérifier et discuter | 20 min | Tests de cohérence et limites scientifiques |

**Hypothèse de durée :** dépendances installées, fichier NetCDF déjà téléchargé, squelette Leaflet disponible. La construction complète du projet depuis zéro peut demander davantage de temps.

## 11. Livrables attendus

1. Code source du pipeline Python et instructions d'exécution.
2. Fichiers cartographiques générés (raster/tuiles, GeoJSON, métadonnées).
3. Application Leaflet statique opérationnelle.
4. Rapport de préparation et de statistiques.
5. README expliquant téléchargement, installation, préparation, lancement et limites.
6. Tests automatisés et vérifications cartographiques documentées.

## 12. Hors périmètre

- Entraînement d'un modèle de machine learning ou deep learning.
- Prévision de l'évolution des fonds marins.
- Backend d'API, base de données spatiale ou authentification.
- Affichage de millions de marqueurs individuels.
- Présentation des valeurs GEBCO comme des mesures directes garanties.
- Déploiement cloud obligatoire.

## 13. Décisions techniques à confirmer lors de l'implémentation

1. **Fichier réel :** version GEBCO et noms exacts des variables après téléchargement et inspection.
2. **Raster :** tuiles Web Mercator (solution recommandée) ou alternative raster compatible Leaflet ; le rendu doit être testé géographiquement.
3. **Taille :** résolution de l'image et nombre de points, à ajuster aux performances mesurées.
4. **Interrogation raster :** F10 reste optionnelle ; si réalisée, exporter une grille numérique correctement indexée géographiquement.
5. **Attributions :** reprendre les mentions imposées par les conditions de la source et du fond cartographique choisis.

## 14. Définition de terminé

Le TP est terminé lorsque le fichier GEBCO réel peut être téléchargé et préparé de façon reproductible, que les deux représentations affichent correctement la même bathymétrie dans Leaflet, que le contrôle de bascule fonctionne, que les profondeurs des points sont interrogeables et que les tests de données et d'alignement géographique passent.
