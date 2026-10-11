"""Briques communes du TP6 : régions GEBCO, grille, patches, normalisation, statistiques, graines.

Ce module est partagé par ``download_data.py``, ``prepare_dataset.py``,
``train_vae.py``, ``train_gan.py``, ``generate.py``, ``evaluate.py``,
``export_onnx.py`` et les tests. Il centralise les hypothèses du TP (régions,
résolution, taille des patches, normalisation, définitions des indicateurs)
pour qu'elles soient définies à un seul endroit.

Conventions :

- **Élévation** GEBCO en mètres, négative sous le niveau moyen de la mer.
- Une carte (patch) est une grille 64 × 64. Ligne 0 = **nord**, colonne 0 =
  **ouest** (sens de lecture d'une image), quel que soit l'ordre du NetCDF.
- **Relief relatif** : élévation moins la moyenne du patch. Les modèles
  apprennent des **formes** de relief, pas une profondeur absolue (F04).
- **Normalisation** : ``z = clip(relief_relatif / ECHELLE_M, -1, 1)``. Le même
  facteur sert aux deux modèles, à l'évaluation et au site web ; il est
  conservé dans ``dataset.json`` et dans les graphes ONNX.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TP6_DIR = Path(__file__).parent
DATA_DIR = TP6_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # extraits NetCDF bruts, un par région (cache)
PROVENANCE_JSON = DATA_DIR / "provenance.json"
DATASET_DIR = DATA_DIR / "dataset"  # patches, métadonnées, normalisation (F03 à F05)
FIGURES_DIR = DATA_DIR / "figures"  # exploration du dataset (F06)
RUNS_DIR = TP6_DIR / "runs"  # un sous-dossier par modèle + génération + évaluation
MODEL_DIR = TP6_DIR / "model"  # servi tel quel par le site web (ONNX, métadonnées)

# --------------------------------------------------------------------------
# Source des données : GEBCO_2021 servi par l'ERDDAP de l'Ifremer (F01)
# --------------------------------------------------------------------------

# Grille GEBCO_2021 complète (15″ d'arc), interrogeable par sous-emprise, sans
# compte ni clé. La requête griddap rend un NetCDF de la seule zone demandée.
ERDDAP = "https://erddap.ifremer.fr/erddap/griddap/gebco2021"
ERDDAP_REQUETE = ERDDAP + ".nc?elevation%5B({sud}):1:({nord})%5D%5B({ouest}):1:({est})%5D"
GEBCO_PRODUIT = "GEBCO_2021 Grid (15 arc-secondes)"
GEBCO_DOI = "10.5285/c6612cbe-50b3-0cff-e053-6c86abc09f8f"
ATTRIBUTION = ("GEBCO Compilation Group (2021) GEBCO 2021 Grid "
               "(doi:10.5285/c6612cbe-50b3-0cff-e053-6c86abc09f8f), via ERDDAP Ifremer")
PAS_NATIF_DEG = 1 / 240  # 15″

# Régions d'intérêt (F02) : emprises WGS84 en degrés décimaux, choisies pour
# couvrir les morphologies du brief (§2.1). Elles ne se chevauchent pas.
REGIONS: list[dict] = [
    {"id": "acores", "nom": "Dorsale médio-atlantique (Açores)", "sud": 35.0, "nord": 41.0,
     "ouest": -36.0, "est": -27.0, "morphologie": "dorsale, vallée axiale, failles transformantes, volcans"},
    {"id": "reykjanes", "nom": "Dorsale de Reykjanes", "sud": 56.0, "nord": 62.0,
     "ouest": -36.0, "est": -24.0, "morphologie": "dorsale lente, rides volcaniques obliques"},
    {"id": "meteor", "nom": "Monts sous-marins Great Meteor", "sud": 27.0, "nord": 33.0,
     "ouest": -32.0, "est": -25.0, "morphologie": "monts sous-marins à sommet plat (guyots), collines abyssales"},
    {"id": "madere", "nom": "Plaine abyssale de Madère et ride Madère-Tore", "sud": 33.5, "nord": 38.0,
     "ouest": -26.0, "est": -17.0, "morphologie": "plaine abyssale, collines abyssales, ride volcanique"},
    {"id": "canaries", "nom": "Archipels de Madère et des Canaries", "sud": 27.0, "nord": 33.5,
     "ouest": -20.0, "est": -13.0, "morphologie": "flancs d'îles volcaniques, glissements, monts"},
    {"id": "iberie", "nom": "Marge ibérique et banc de Gorringe", "sud": 34.0, "nord": 40.0,
     "ouest": -16.0, "est": -8.0, "morphologie": "canyons (Nazaré, Setúbal), banc de Gorringe, plaines du Tage"},
    {"id": "gascogne", "nom": "Golfe de Gascogne", "sud": 43.0, "nord": 48.5,
     "ouest": -11.0, "est": -1.0, "morphologie": "talus continental, canyons, plaine abyssale"},
    {"id": "rockall", "nom": "Fossé et banc de Rockall", "sud": 51.0, "nord": 58.0,
     "ouest": -20.0, "est": -10.0, "morphologie": "bancs, fossé, pentes, monts"},
    {"id": "nouvelle_angleterre", "nom": "Monts sous-marins de Nouvelle-Angleterre", "sud": 35.0, "nord": 41.0,
     "ouest": -68.0, "est": -58.0, "morphologie": "chaîne de monts sous-marins, canyons de la marge américaine"},
    {"id": "mediterranee", "nom": "Méditerranée occidentale", "sud": 38.0, "nord": 44.3,
     "ouest": 3.0, "est": 10.0, "morphologie": "canyons du golfe du Lion et de Ligurie, plaine abyssale, marges"},
    {"id": "pacifique_est", "nom": "Dorsale Est-Pacifique", "sud": -14.0, "nord": -8.0,
     "ouest": -108.0, "est": -100.0, "morphologie": "dorsale rapide, collines abyssales alignées, failles transformantes"},
    {"id": "musiciens", "nom": "Monts sous-marins des Musiciens (Pacifique Nord)", "sud": 26.0, "nord": 32.0,
     "ouest": -165.0, "est": -157.0, "morphologie": "monts sous-marins isolés, rides volcaniques, plaine"},
    {"id": "porto_rico", "nom": "Fosse de Porto Rico", "sud": 19.0, "nord": 23.5,
     "ouest": -70.0, "est": -61.0, "morphologie": "fosse de subduction, pentes raides, plateaux"},
]

# --------------------------------------------------------------------------
# Grille de travail, patches et séparation (F03 à F05)
# --------------------------------------------------------------------------

# Moyenne par blocs de la grille 15″ : 4 cellules en latitude (1′ ≈ 1,85 km) et
# un nombre de cellules en longitude choisi pour des pixels à peu près carrés en
# kilomètres (les méridiens se resserrent vers le nord). Moyenner plutôt que
# sous-échantillonner évite le repliement (aliasing).
FACTEUR_LAT = 4
KM_PAR_DEG_LAT = 111.32

PATCH = 64  # taille de référence (64 × 64 pixels ≈ 119 km de côté)
PAS_PATCH = 8  # décalage entre patches candidats (recouvrement de 87,5 %), avant sous-échantillonnage
BLOC = 128  # côté (pixels) des blocs spatiaux qui servent à la séparation
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# Part des blocs tirés en val et en test. Elle est un peu inférieure à la cible :
# les patches d'entraînement qui touchent un bloc val/test sont perdus, si bien
# que la répartition finale des patches se rapproche de FRACTIONS.
PART_BLOCS_VAL_TEST = 0.12
CIBLE_PATCHES = 5000  # taille de référence du dataset (F05)

# Filtre de diversité (F02) : un patch est « presque plat » si l'écart-type de
# son relief est inférieur à SEUIL_PLAT_M. On n'en garde qu'une petite part.
SEUIL_PLAT_M = 30.0
PART_MAX_PLATS = 0.10

# Normalisation (F04) : relief relatif divisé par ECHELLE_M puis écrêté à ±1.
# 1 500 m couvre l'essentiel des reliefs à 120 km ; seuls quelques sommets de
# monts et de flancs d'îles dépassent (part écrêtée mesurée dans dataset.json).
ECHELLE_M = 1500.0


def facteur_lon(lat_centre: float) -> int:
    """Nombre de cellules 15″ moyennées en longitude pour des pixels ~carrés en km."""
    return max(1, round(FACTEUR_LAT / math.cos(math.radians(lat_centre))))


def taille_pixel_km(lat_centre: float) -> tuple[float, float]:
    """Taille (dy, dx) d'un pixel de la grille de travail, en km, au centre de la région."""
    dy = FACTEUR_LAT * PAS_NATIF_DEG * KM_PAR_DEG_LAT
    dx = facteur_lon(lat_centre) * PAS_NATIF_DEG * KM_PAR_DEG_LAT * math.cos(math.radians(lat_centre))
    return dy, dx


def moyenne_blocs(e: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Moyenne par blocs ``fy × fx`` (les bords incomplets sont ignorés). NaN si une cellule manque."""
    ny, nx = e.shape[0] // fy, e.shape[1] // fx
    b = e[:ny * fy, :nx * fx].reshape(ny, fy, nx, fx).astype(np.float64)
    return b.mean(axis=(1, 3)).astype(np.float32)


def normaliser(relief_m: np.ndarray, echelle: float = ECHELLE_M) -> np.ndarray:
    """Élévations (m) -> valeurs normalisées dans [-1, 1], patch par patch (dernières dimensions 64 × 64)."""
    r = np.asarray(relief_m, dtype=np.float32)
    r = r - r.mean(axis=(-2, -1), keepdims=True)
    return np.clip(r / echelle, -1.0, 1.0).astype(np.float32)


def denormaliser(z: np.ndarray, echelle: float = ECHELLE_M) -> np.ndarray:
    """Valeurs normalisées -> relief relatif en mètres (écart à la moyenne du patch)."""
    return (np.asarray(z, dtype=np.float32) * echelle).astype(np.float32)


# --------------------------------------------------------------------------
# Entrées / sorties
# --------------------------------------------------------------------------

def lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def ecrire_json(chemin: Path, contenu: dict | list) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")


def charger_dataset(data_dir: Path = DATASET_DIR) -> dict:
    """Patches préparés par ``prepare_dataset.py``.

    Retourne, pour chaque ensemble, ``z`` (N, 64, 64) normalisé et ``m`` (N, 64, 64)
    en mètres (élévation absolue), plus ``index`` (provenance de chaque patch) et
    le manifeste ``dataset.json``.
    """
    chemin = data_dir / "patches.npz"
    if not chemin.exists():
        raise SystemExit(f"Dataset absent : {chemin}. Lancez d'abord `mise run tp6_02_prepare`.")
    npz = np.load(chemin)
    manifeste = lire_json(data_dir / "dataset.json")
    echelle = manifeste["normalisation"]["echelle_m"]
    res: dict = {"manifeste": manifeste, "index": lire_json(data_dir / "index.json")}
    for s in FRACTIONS:
        m = npz[f"{s}_m"].astype(np.float32)
        res[s] = {"m": m, "z": normaliser(m, echelle)}
    return res


# --------------------------------------------------------------------------
# Indicateurs de relief (F22, F35) — mêmes définitions dans le site web
# --------------------------------------------------------------------------

def _moyenne_3x3(r: np.ndarray) -> np.ndarray:
    """Moyenne glissante 3 × 3 (bords répliqués) sur les deux dernières dimensions."""
    p = np.pad(r, [(0, 0)] * (r.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    h, w = r.shape[-2:]
    return sum(p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def pentes_deg(relief_m: np.ndarray, pixel_km: float) -> np.ndarray:
    """Pente locale (degrés) par différences centrées, pour chaque pixel intérieur."""
    r = np.asarray(relief_m, dtype=np.float64)
    gy = (r[..., 2:, 1:-1] - r[..., :-2, 1:-1]) / (2 * pixel_km * 1000)
    gx = (r[..., 1:-1, 2:] - r[..., 1:-1, :-2]) / (2 * pixel_km * 1000)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def indicateurs(relief_m: np.ndarray, pixel_km: float) -> dict[str, np.ndarray]:
    """Indicateurs simples par patch (tableaux de taille N) pour un lot (N, 64, 64) en mètres.

    - ``amplitude_m``  : max − min ;
    - ``ecart_type_m`` : écart-type du relief (variabilité globale) ;
    - ``variabilite_locale_m`` : écart-type moyen dans des fenêtres 3 × 3 ;
    - ``pente_moy_deg`` : pente moyenne ;
    - ``rugosite_m``   : écart-type du relief moins sa moyenne glissante 3 × 3
      (détail à l'échelle de quelques pixels, défini comme dans le site).
    """
    r = np.asarray(relief_m, dtype=np.float64)
    if r.ndim == 2:
        r = r[None]
    m3 = _moyenne_3x3(r)
    var_locale = np.clip(_moyenne_3x3(r**2) - m3**2, 0, None)
    return {
        "amplitude_m": r.max(axis=(-2, -1)) - r.min(axis=(-2, -1)),
        "ecart_type_m": r.std(axis=(-2, -1)),
        "variabilite_locale_m": np.sqrt(var_locale).mean(axis=(-2, -1)),
        "pente_moy_deg": pentes_deg(r, pixel_km).mean(axis=(-2, -1)),
        "rugosite_m": (r - m3).std(axis=(-2, -1)),
    }


# --------------------------------------------------------------------------
# Graines (F11, F17, F28) : même générateur en Python et dans le navigateur
# --------------------------------------------------------------------------

def mulberry32(graine: int):
    """Générateur pseudo-aléatoire 32 bits (mulberry32), identique à celui du site web."""
    etat = graine & 0xFFFFFFFF

    def suivant() -> float:
        nonlocal etat
        etat = (etat + 0x6D2B79F5) & 0xFFFFFFFF
        t = etat
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return suivant


def vecteur_latent(graine: int, dim: int) -> np.ndarray:
    """Vecteur latent N(0, I) reproductible : Box-Muller sur mulberry32 (même résultat dans le site)."""
    u = mulberry32(graine)
    z = np.empty(dim, dtype=np.float64)
    for i in range(0, dim, 2):
        u1, u2 = 1.0 - u(), u()  # u1 dans ]0, 1] : log défini
        rayon = math.sqrt(-2.0 * math.log(u1))
        z[i] = rayon * math.cos(2 * math.pi * u2)
        if i + 1 < dim:
            z[i + 1] = rayon * math.sin(2 * math.pi * u2)
    return z.astype(np.float32)


def lot_latent(graine: int, n: int, dim: int) -> np.ndarray:
    """``n`` vecteurs latents : le i-ème utilise la graine ``graine + i`` (galeries reproductibles)."""
    return np.stack([vecteur_latent(graine + i, dim) for i in range(n)])


# --------------------------------------------------------------------------
# Modèles : noms et couleurs communs (Python et site web)
# --------------------------------------------------------------------------

MODELES = ["vae", "gan"]
STYLES: dict[str, dict] = {
    "reel": {"nom": "Réel (GEBCO)", "couleur": "#475569"},
    "vae": {"nom": "VAE", "couleur": "#0072B2"},
    "gan": {"nom": "GAN", "couleur": "#D55E00"},
}

# Palette bathymétrique (du plus profond au moins profond), commune aux figures
# Python et au site : appliquée au relief relatif sur une échelle donnée.
PALETTE = ["#06183C", "#10346F", "#1761A4", "#339AC9", "#79D2E7", "#C9F2F6"]


def colormap():
    """Palette matplotlib correspondant à ``PALETTE``."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bathy_tp6", PALETTE)


def distance_w1(a: np.ndarray, b: np.ndarray, quantiles: int = 200) -> float:
    """Distance de Wasserstein-1 entre deux distributions 1D, estimée par leurs quantiles."""
    q = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def ecart_statistique(relief_gen_m: np.ndarray, relief_ref_m: np.ndarray, pixel_km: float = 1.85) -> dict:
    """Écart entre les indicateurs de deux jeux de reliefs : W1 de chaque indicateur / écart-type de référence.

    0 = mêmes distributions. Sert à suivre le GAN (pas de perte de validation
    interprétable) et à comparer les modèles (F22).
    """
    ig, ir = indicateurs(relief_gen_m, pixel_km), indicateurs(relief_ref_m, pixel_km)
    par = {k: distance_w1(ig[k], ir[k]) / max(float(ir[k].std()), 1e-9) for k in ir}
    return {"moyenne": float(np.mean(list(par.values()))), "par_indicateur": par}


def diversite(lot: np.ndarray) -> float:
    """Distance euclidienne moyenne entre paires de cartes d'un lot (N, 64, 64), par pixel (RMS)."""
    x = lot.reshape(len(lot), -1).astype(np.float64)
    carres = (x**2).sum(1)
    d2 = np.clip(carres[:, None] + carres[None, :] - 2 * x @ x.T, 0, None) / x.shape[1]
    iu = np.triu_indices(len(lot), 1)
    return float(np.sqrt(d2[iu]).mean())
