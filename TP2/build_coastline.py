"""Génère un trait de côte (polygones de terre) recadré à l'emprise du modèle.

Télécharge Natural Earth 1:10m « land » (domaine public), le découpe à l'emprise
du modèle lue dans ``metadata.json``, arrondit les coordonnées pour alléger, et
écrit ``TP2/model/land.geojson``. Ce fichier sert au site web à découper le champ
le long des côtes réelles (rendu net) et à tester « mer / terre » pour la sonde.

Usage :
    uv run python TP2/build_coastline.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = Path(__file__).parent / "model"
META_PATH = MODEL_DIR / "metadata.json"
OUT_PATH = MODEL_DIR / "land.geojson"

SOURCE_URL = (
    "https://raw.githubusercontent.com/martynafford/"
    "natural-earth-geojson/master/10m/physical/ne_10m_land.json"
)

# Marge autour de l'emprise pour garantir des côtes complètes jusqu'au bord.
MARGE_DEG = 0.5
# Précision conservée (4 décimales ≈ 11 m), suffisante et bien plus légère.
PRECISION = 4


def emprise() -> tuple[float, float, float, float]:
    """Lit l'emprise (lon/lat min/max) du modèle dans metadata.json."""
    if not META_PATH.exists():
        raise SystemExit(
            "metadata.json introuvable : lance d'abord `mise run tp2_train`."
        )
    norm = json.loads(META_PATH.read_text(encoding="utf-8"))["normalization"]
    return norm["lon_min"], norm["lat_min"], norm["lon_max"], norm["lat_max"]


def arrondir(obj: object) -> object:
    """Arrondit récursivement toutes les coordonnées d'une géométrie GeoJSON."""
    if isinstance(obj, float):
        return round(obj, PRECISION)
    if isinstance(obj, list):
        return [arrondir(x) for x in obj]
    if isinstance(obj, tuple):
        return [arrondir(x) for x in obj]
    return obj


def main() -> None:
    lon_min, lat_min, lon_max, lat_max = emprise()
    zone = box(
        lon_min - MARGE_DEG,
        lat_min - MARGE_DEG,
        lon_max + MARGE_DEG,
        lat_max + MARGE_DEG,
    )
    print(f"Emprise : lon [{lon_min:.2f}, {lon_max:.2f}], lat [{lat_min:.2f}, {lat_max:.2f}]")
    print("Téléchargement de Natural Earth 1:10m land…")
    with urllib.request.urlopen(SOURCE_URL) as resp:
        data = json.load(resp)

    morceaux = []
    for feat in data["features"]:
        decoupe = shape(feat["geometry"]).intersection(zone)
        if not decoupe.is_empty:
            morceaux.append(decoupe)

    terre = unary_union(morceaux)
    geojson = {
        "type": "Feature",
        "properties": {"source": "Natural Earth 1:10m land (domaine public)"},
        "geometry": arrondir(mapping(terre)),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Écrit : {OUT_PATH} ({OUT_PATH.stat().st_size / 1000:.0f} Ko)")


if __name__ == "__main__":
    main()
