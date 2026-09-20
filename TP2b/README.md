# Idée 3 — Approximer une **distribution** (MDN)

Ce TP prolonge le TP2. Là où l'idée 1 apprend une **fonction**
`f(lon, lat) → température` (une valeur par point), l'idée 3 apprend une
**loi de probabilité** `p(T)` : la distribution des températures de surface de
tout le Golfe de Gascogne à un instant donné (le pas de temps `t = 0`).

On la compare à l'**histogramme réel** des ~240 000 valeurs valides du fichier
HYCOM 3D-SURF (SHOM). C'est la brique conceptuelle des modèles génératifs vus au
chapitre 6 (VAE/GAN).

> Le fichier NetCDF de données **reste dans `TP2/data/`** ; `TP2b/` ne contient
> que le code de ce TP. Le script le **télécharge automatiquement** s'il manque,
> donc ce TP est autonome (inutile d'avoir lancé le TP2 avant).

---

## 1. Le modèle : *Mixture Density Network* (design 1)

On modélise `p(T)` par un **mélange de K gaussiennes** :

```
p(T) = Σ_k  π_k · N(T ; μ_k, σ_k)      avec  Σ_k π_k = 1
```

Les paramètres `{π_k, μ_k, σ_k}` sont **appris par descente de gradient** en
minimisant la **log-vraisemblance négative (NLL)**. Contraintes garanties par
construction : `π` via *softmax* (somme = 1), `σ` via *softplus* (strictement
positif). La NLL est calculée de façon stable avec `logsumexp`.

### Entrée / sortie à l'inférence

Le réseau **ne prend pas** une température pour sortir une probabilité. Il
apprend les **paramètres** de la loi. Pour rendre l'inférence directe côté web,
l'export ONNX « cuit » l'évaluation analytique du mélange dans le graphe :

| | Entrée | Sortie |
|---|---|---|
| Modèle ONNX | une température `T` (°C) | la **densité** `p(T)` |

La normalisation `t = (T − mean) / std` et le changement de variable de densité
`p_T(T) = p_t(t) / std` sont inclus dans le graphe : le navigateur fournit des
températures physiques et lit une densité physique.

> Rappel : pour une variable continue, la sortie est une **densité** (peut
> dépasser 1, intègre à 1), pas une probabilité. La probabilité s'obtient sur un
> intervalle : `P(a < T < b) = ∫ p(T) dT`.

---

## 2. Entraînement

```powershell
uv run python TP2b/train_temperature_mdn.py
# options :
uv run python TP2b/train_temperature_mdn.py --num-components 8 --epochs 600
```

Trois artefacts sont écrits dans `TP2b/model/` :

| Fichier | Contenu |
|---------|---------|
| `temperature_mdn.onnx` | entrée = température (°C), sortie = densité `p(T)` |
| `metadata_mdn.json` | paramètres du mélange (espace physique), normalisation, métriques |
| `histogram.json` | histogramme réel de référence (densité + effectifs) |

### Métriques imprimées

- **NLL** (log-vraisemblance négative) sur la validation ;
- **KL(réel ‖ modèle)** : divergence entre l'histogramme et la densité apprise ;
- **Wasserstein-1** (Earth Mover's Distance) entre les vraies valeurs et un
  tirage du modèle — la métrique des Wasserstein-GAN (chapitre 6).

---

## 3. Visualisation web

```powershell
npm run dev:tp2b
```

Le site (port **5174**) affiche :

- l'**histogramme réel** (barres) des ~240 000 températures ;
- la **densité du modèle** `p(T)`, calculée en direct par le réseau (ONNX) sur
  une grille de températures ;
- un bouton **Rééchantillonner** qui tire des températures selon le mélange et
  superpose leur histogramme (première expérience « générative ») ;
- les métriques (NLL, KL, Wasserstein).

---

## 4. Pièges pédagogiques

- **collapse de variance** (`σ → 0`) : une gaussienne se contracte sur un point
  et fait diverger la NLL → on borne `σ` par le bas (`softplus + 1e-3`) ;
- **K trop grand** : composantes redondantes / sur-ajustement ;
- oublier la **normalisation** rend la NLL instable ;
- confondre **densité** (intègre à 1) et **histogramme en effectifs** lors de la
  superposition.

---

## 5. Ouverture (chapitre 6)

- conditionner sur l'**heure** → `p(T | heure)` et voir l'histogramme se décaler
  au fil de la journée ;
- remplacer le mélange gaussien par un **normalizing flow** (densité plus
  expressive) ;
- comparer « MDN par maximum de vraisemblance » vs « GAN par distance de
  Wasserstein » sur la même cible → transition vers VAE/GAN.
