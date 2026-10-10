"""Briques communes du TP5 : station, API REFMAR, fenêtres temporelles, références, métriques.

Ce module est partagé par ``download_data.py``, ``prepare_dataset.py``,
``train.py``, ``evaluate.py``, ``export_onnx.py`` et les tests. Il centralise
les hypothèses du TP (station, période, taille des fenêtres, horizons évalués,
styles des courbes) pour qu'elles soient définies à un seul endroit.

Convention temporelle : toutes les séries sont sur une **grille horaire
régulière en UTC**. Une heure sans observation valide vaut ``NaN`` : elle n'est
jamais remplacée (NF06), et toute fenêtre qui la contient est écartée (F02).

Convention des fenêtres (F04) : une fenêtre est repérée par son **ancre** ``t``,
dernière heure observée. Entrée = ``h[t-47 .. t]`` (48 valeurs), cible =
``h[t+1 .. t+24]`` (24 valeurs). Aucune valeur postérieure à ``t`` n'entre dans
le modèle (NF07).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TP5_DIR = Path(__file__).parent
DATA_DIR = TP5_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # réponses brutes de l'API, une par mois (cache)
OBS_CSV = DATA_DIR / "brest_horaire.csv"  # observations assemblées, non modifiées
PROVENANCE_JSON = DATA_DIR / "provenance.json"
QC_DIR = DATA_DIR / "qc"  # rapport de contrôle qualité (F02)
FIGURES_DIR = DATA_DIR / "figures"  # visualisations exploratoires (F03)
DATASET_DIR = DATA_DIR / "dataset"  # série sur grille régulière + fenêtres (F04, F05)
RUNS_DIR = TP5_DIR / "runs"  # un sous-dossier par modèle + évaluation
MODEL_DIR = TP5_DIR / "model"  # servi tel quel par le site web (ONNX, métadonnées)

# --------------------------------------------------------------------------
# Source des données : REFMAR (SHOM), API publique sans clé
# --------------------------------------------------------------------------

STATION_ID = 3  # identifiant SHOM du marégraphe de Brest (réseau RONIM)
STATION_NOM = "Brest"
# Source 4 de l'API REFMAR = « hauteurs horaires validées » (1 : brutes haute
# fréquence, 2 : brutes temps différé 10 min, 3 : validées temps différé 10 min).
SOURCE_ID = 4
SOURCE_LIBELLE = "Hauteurs d'eau horaires validées"
API_OBSERVATIONS = "https://services.data.shom.fr/maregraphie/observation/json/{station}"
API_STATION = "https://services.data.shom.fr/maregraphie/service/completetidegauge/{station}"
PORTAIL = "https://data.shom.fr/donnees/refmar/download"
ATTRIBUTION = "Observations marégraphiques REFMAR — © Shom (Licence Ouverte Etalab 2.0)"

# Historique cible : 5 années civiles complètes (brief F01).
ANNEE_DEBUT = 2021
ANNEE_FIN = 2025

# --------------------------------------------------------------------------
# Contrôle qualité (F02)
# --------------------------------------------------------------------------

# Bornes physiques larges pour Brest (hauteurs au-dessus du zéro hydrographique :
# PBMA ≈ 0,25 m, PHMA ≈ 7,93 m) : surcotes et décotes comprises.
BORNE_MIN_M = -1.0
BORNE_MAX_M = 10.0
# Pic isolé : saut de plus de 2,5 m en une heure, aller ET retour. La marée de
# vive-eau à Brest ne dépasse pas ~1,8 m/h (3,5 m x 2π / 12,42 h).
PIC_ISOLE_M = 2.5
# Écart entre moyennes annuelles au-delà duquel on suspecte un changement de
# référence verticale (les variations naturelles sont de quelques centimètres).
SAUT_REFERENCE_M = 0.10

# --------------------------------------------------------------------------
# Fenêtres et séparation (F04, F05)
# --------------------------------------------------------------------------

ENTREE_H = 48  # heures d'historique données au modèle
HORIZON_H = 24  # heures prévues directement (sortie de 24 valeurs)
PAS_FENETRES_H = 1  # décalage entre deux fenêtres successives
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
HORIZONS_EVAL = [1, 6, 12, 24]  # échéances détaillées (F14)

# --------------------------------------------------------------------------
# Modèles et références : noms, couleurs et styles communs (Python et site web)
# --------------------------------------------------------------------------

MODELES = ["rnn", "gru", "lstm"]
REFERENCES = ["persistance", "repetition_24h", "repetition_25h"]

# Couleurs (palette lisible par les daltoniens), tirets et marqueurs différents
# pour que les courbes ne se distinguent pas uniquement par la couleur (§7.4).
STYLES: dict[str, dict] = {
    "observe": {"nom": "Observations (48 h)", "couleur": "#475569", "tirets": [], "marqueur": "aucun"},
    "realite": {"nom": "Réalité (24 h futures)", "couleur": "#0F172A", "tirets": [], "marqueur": "rond"},
    "rnn": {"nom": "RNN", "couleur": "#E69F00", "tirets": [10, 5], "marqueur": "triangle"},
    "gru": {"nom": "GRU", "couleur": "#009E73", "tirets": [12, 4, 3, 4], "marqueur": "carre"},
    "lstm": {"nom": "LSTM", "couleur": "#7C3AED", "tirets": [], "marqueur": "losange"},
    "persistance": {"nom": "Persistance", "couleur": "#9CA3AF", "tirets": [2, 4], "marqueur": "croix"},
    "repetition_24h": {"nom": "Répétition 24 h", "couleur": "#CC79A7", "tirets": [5, 4], "marqueur": "plus"},
    "repetition_25h": {"nom": "Répétition 25 h", "couleur": "#0072B2", "tirets": [7, 3, 1, 3], "marqueur": "cercle"},
}

DESCRIPTIONS_REFERENCES = {
    "persistance": "répète la dernière hauteur observée H pendant 24 h",
    "repetition_24h": "prévoit pour H+k la valeur observée à H+k−24 h",
    "repetition_25h": "prévoit pour H+k la valeur observée à H+k−25 h "
                      "(≈ un jour lunaire de 24 h 50 : deux marées semi-diurnes)",
}


# --------------------------------------------------------------------------
# Entrées / sorties
# --------------------------------------------------------------------------

def lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def ecrire_json(chemin: Path, contenu: dict | list) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")


def charger_serie(data_dir: Path = DATASET_DIR) -> dict:
    """Série horaire régulière préparée par ``prepare_dataset.py``.

    Retourne ``temps`` (datetime64[h] UTC), ``hauteur`` (float32, NaN = lacune)
    et le manifeste ``dataset.json`` (paramètres, ancres par ensemble, normalisation).
    """
    chemin = data_dir / "serie.npz"
    if not chemin.exists():
        raise SystemExit(f"Série absente : {chemin}. Lancez d'abord `mise run tp5_02_prepare`.")
    npz = np.load(chemin)
    return {
        "temps": npz["temps"].astype("datetime64[h]"),
        "hauteur": npz["hauteur"].astype(np.float32),
        "ancres": {s: npz[f"ancres_{s}"] for s in FRACTIONS},
        "manifeste": lire_json(data_dir / "dataset.json"),
    }


def iso(t: np.datetime64) -> str:
    """Horodatage ISO 8601 UTC (``2025-04-01T00:00:00Z``)."""
    return f"{np.datetime_as_string(t.astype('datetime64[s]'))}Z"


# --------------------------------------------------------------------------
# Fenêtres temporelles (F04) et séparation chronologique (F05)
# --------------------------------------------------------------------------

def bornes_split(n: int, fractions: dict[str, float] = FRACTIONS, multiple: int = 24) -> dict[str, tuple[int, int]]:
    """Intervalles d'indices ``[debut, fin)`` de chaque ensemble, dans l'ordre chronologique.

    Les coupures sont arrondies à un multiple de ``multiple`` heures (début de
    journée si la série commence à minuit).
    """
    i_val = int(n * fractions["train"]) // multiple * multiple
    i_test = int(n * (fractions["train"] + fractions["val"])) // multiple * multiple
    return {"train": (0, i_val), "val": (i_val, i_test), "test": (i_test, n)}


def ancres_valides(hauteur: np.ndarray, debut: int, fin: int, entree: int = ENTREE_H,
                   horizon: int = HORIZON_H, pas: int = PAS_FENETRES_H) -> np.ndarray:
    """Ancres ``t`` dont les ``entree + horizon`` heures sont dans ``[debut, fin)`` et sans lacune.

    Une fenêtre à cheval sur deux ensembles est écartée : l'entraînement ne
    voit ainsi aucune cible de validation ou de test (F05).
    """
    valide = np.isfinite(hauteur).astype(np.int64)
    cumul = np.concatenate([[0], np.cumsum(valide)])
    t = np.arange(debut + entree - 1, fin - horizon, pas)
    if t.size == 0:
        return t
    complet = (cumul[t + horizon + 1] - cumul[t - entree + 1]) == entree + horizon
    return t[complet]


def fenetres(hauteur: np.ndarray, ancres: np.ndarray, entree: int = ENTREE_H,
             horizon: int = HORIZON_H) -> tuple[np.ndarray, np.ndarray]:
    """Couples (entrée, cible) : ``X[i] = h[t-entree+1 .. t]``, ``Y[i] = h[t+1 .. t+horizon]``."""
    ancres = np.asarray(ancres, dtype=np.int64)
    ix = ancres[:, None] + np.arange(-entree + 1, 1)[None, :]
    iy = ancres[:, None] + np.arange(1, horizon + 1)[None, :]
    return hauteur[ix].astype(np.float32), hauteur[iy].astype(np.float32)


# --------------------------------------------------------------------------
# Références naïves (F15) : calculées uniquement à partir de l'entrée
# --------------------------------------------------------------------------

def persistance(x: np.ndarray, horizon: int = HORIZON_H) -> np.ndarray:
    """Répète la dernière valeur observée."""
    return np.repeat(x[:, -1:], horizon, axis=1)


def repetition(x: np.ndarray, decalage: int, horizon: int = HORIZON_H) -> np.ndarray:
    """Prévoit H+k par la valeur observée à H+k−``decalage`` (dans l'entrée).

    ``x[:, -1]`` est H ; H+k−d correspond à l'indice ``-1 + k - d``.
    """
    entree = x.shape[1]
    if not horizon <= decalage <= entree:
        raise ValueError(f"décalage {decalage} h incompatible avec entrée {entree} h / horizon {horizon} h")
    debut = entree - decalage
    return x[:, debut:debut + horizon].copy()


def predire_references(x: np.ndarray, horizon: int = HORIZON_H) -> dict[str, np.ndarray]:
    return {
        "persistance": persistance(x, horizon),
        "repetition_24h": repetition(x, 24, horizon),
        "repetition_25h": repetition(x, 25, horizon),
    }


# --------------------------------------------------------------------------
# Métriques (F13, F14) — en mètres
# --------------------------------------------------------------------------

def metriques(prevision: np.ndarray, verite: np.ndarray, horizons: list[int] = HORIZONS_EVAL) -> dict:
    """MAE et RMSE globales (24 h), par échéance détaillée et pour chaque pas."""
    erreur = np.asarray(prevision, dtype=np.float64) - np.asarray(verite, dtype=np.float64)
    mae_pas = np.abs(erreur).mean(axis=0)
    rmse_pas = np.sqrt((erreur**2).mean(axis=0))
    return {
        "global": {"mae": float(np.abs(erreur).mean()), "rmse": float(np.sqrt((erreur**2).mean()))},
        "horizons": {f"H+{h}": {"mae": float(mae_pas[h - 1]), "rmse": float(rmse_pas[h - 1])}
                     for h in horizons if h <= erreur.shape[1]},
        "par_pas": {"mae": [round(float(v), 5) for v in mae_pas],
                    "rmse": [round(float(v), 5) for v in rmse_pas]},
        "n_fenetres": int(erreur.shape[0]),
    }
