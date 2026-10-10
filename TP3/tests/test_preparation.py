"""Tests de données et de cohérence cartographique du TP3.

Usage :
    uv run pytest TP3/tests -v

Les tests sur les exports supposent que ``prepare_bathymetry.py`` a été lancé.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from gebco import (
    BBOX,
    OUTPUT_DIR,
    PALETTE,
    RAW_DIR,
    couleurs,
    extraire_emprise,
    hex_rgb,
    mercator_lat,
    mercator_y,
    ouvrir,
)
from prepare_bathymetry import statistiques

# Points de contrôle (lat, lon, attendu) : près des côtes, au large, et à terre.
CONTROLES = [
    (45.5, -6.0, "abysses"),  # plaine abyssale de Gascogne
    (47.0, -4.0, "plateau"),  # plateau continental sud-Bretagne
    (48.3, -4.9, "cotier"),  # mer d'Iroise, près de la pointe bretonne
    (43.5, -1.6, "cotier"),  # gouf de Capbreton / côte basque
    (48.2, -3.5, "terre"),  # centre Bretagne (Montagnes Noires)
    (44.0, -1.2, "terre"),  # Landes
]


# --------------------------------------------------------------------------
# Tests unitaires (sans données)
# --------------------------------------------------------------------------

def test_palette_paliers_exacts():
    for depth, code in PALETTE:
        assert tuple(couleurs(np.array([depth]))[0]) == hex_rgb(code)


def test_palette_saturee_et_monotone():
    assert tuple(couleurs(np.array([9000.0]))[0]) == hex_rgb(PALETTE[-1][1])
    # Plus profond => plus sombre (somme RGB décroissante).
    d = np.linspace(0, 5000, 200)
    lum = couleurs(d).astype(int).sum(axis=1)
    assert np.all(np.diff(lum) <= 0)


def test_mercator_aller_retour():
    lats = np.linspace(43, 49, 50)
    assert np.allclose(mercator_lat(mercator_y(lats)), lats)


# --------------------------------------------------------------------------
# Fixtures sur les données réelles
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def source():
    ncs = list(RAW_DIR.glob("*.nc"))
    if not ncs:
        pytest.skip("NetCDF absent : lancer `mise run tp3_01_explore`.")
    g = ouvrir(ncs[0])
    sub = extraire_emprise(g)
    return g, sub


@pytest.fixture(scope="session")
def meta():
    if not (OUTPUT_DIR / "metadata.json").exists():
        pytest.skip("Exports absents : lancer `mise run tp3_02_prepare`.")
    return json.loads((OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def points(meta):
    return json.loads((OUTPUT_DIR / meta["points"]["fichier"]).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def grille(meta):
    gm = meta["grille"]
    data = np.fromfile(OUTPUT_DIR / gm["fichier"], dtype="<f4")
    return data.reshape(gm["lignes"], gm["colonnes"])


@pytest.fixture(scope="session")
def image(meta):
    return np.asarray(Image.open(OUTPUT_DIR / meta["raster"]["fichier"]).convert("RGBA"))


def xr_indexer(values):
    return xr.DataArray(values, dims="p")


def elevation_proche(source, lat, lon) -> float:
    g, sub = source
    return float(sub.sel({g.lat: lat, g.lon: lon}, method="nearest"))


# --------------------------------------------------------------------------
# Tests de données
# --------------------------------------------------------------------------

def test_emprise_exportee(meta):
    b = meta["bounds"]
    for cote in ("west", "east", "south", "north"):
        assert b[cote] == pytest.approx(BBOX[cote], abs=1e-6)
    assert -180 <= b["west"] < b["east"] <= 180
    assert -90 <= b["south"] < b["north"] <= 90


def test_palette_metadata_identique(meta):
    assert [(p["depth_m"], p["couleur"]) for p in meta["palette"]["paliers"]] == PALETTE


def test_geojson_valide(points, meta):
    assert points["type"] == "FeatureCollection"
    n = len(points["features"])
    assert n == meta["points"]["nombre_points"]
    assert 3000 <= n <= 10000
    for f in points["features"]:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Point"
        lon, lat = f["geometry"]["coordinates"]  # ordre [lon, lat]
        assert BBOX["west"] <= lon <= BBOX["east"]
        assert BBOX["south"] <= lat <= BBOX["north"]
        d = f["properties"]["depth_m"]
        assert math.isfinite(d) and d > 0


def test_points_conformes_a_la_source(points, source):
    g, sub = source
    lons = np.array([f["geometry"]["coordinates"][0] for f in points["features"]])
    lats = np.array([f["geometry"]["coordinates"][1] for f in points["features"]])
    depth = np.array([f["properties"]["depth_m"] for f in points["features"]])
    elev = sub.sel(
        {g.lat: xr_indexer(lats), g.lon: xr_indexer(lons)}, method="nearest"
    ).values
    assert np.all(elev < 0), "un point est situé sur une cellule terrestre"
    assert np.array_equal(depth, -elev.astype(np.float64))


def test_grille_f10_conforme(grille, meta, source):
    g, sub = source
    gm = meta["grille"]
    pas = gm["pas_sous_echantillonnage"]
    attendu = -sub.values[::pas, ::pas][::-1].astype(np.float32)
    attendu[attendu <= 0] = np.nan
    assert np.array_equal(grille, attendu, equal_nan=True)
    valides = grille[np.isfinite(grille)]
    assert np.all(valides > 0)


def test_rapport_reproductible(meta, source):
    g, sub = source
    report = json.loads((OUTPUT_DIR / "report.json").read_text(encoding="utf-8"))
    assert report["statistiques"] == statistiques(sub.values, sub[g.lat].values)


# --------------------------------------------------------------------------
# Tests cartographiques : points de contrôle grille source <-> exports
# --------------------------------------------------------------------------

def pixel_mercator(meta, lat, lon) -> tuple[int, int]:
    """Pixel (ligne, colonne) de l'image Mercator qui couvre (lat, lon)."""
    b, r = meta["bounds"], meta["raster"]
    col = (lon - b["west"]) / (b["east"] - b["west"]) * r["largeur_px"]
    yn, ys = mercator_y(b["north"]), mercator_y(b["south"])
    lig = (yn - mercator_y(lat)) / (yn - ys) * r["hauteur_px"]
    return int(lig), int(col)


@pytest.mark.parametrize("lat,lon,nature", CONTROLES)
def test_point_de_controle_raster(meta, image, source, lat, lon, nature):
    g, sub = source
    lig, col = pixel_mercator(meta, lat, lon)
    rgba = image[lig, col]
    i = int(np.abs(sub[g.lat].values - lat).argmin())
    j = int(np.abs(sub[g.lon].values - lon).argmin())
    voisinage = sub.values[i - 1 : i + 2, j - 1 : j + 2]  # tolérance ±1 cellule

    if nature == "terre":
        assert elevation_proche(source, lat, lon) >= 0
        assert rgba[3] == 0, "la terre doit être transparente"
        return
    assert rgba[3] == 255
    possibles = {tuple(c) for c in couleurs(-voisinage[voisinage < 0])}
    assert tuple(rgba[:3]) in possibles, "décalage ou inversion du raster"

    depth = -elevation_proche(source, lat, lon)
    if nature == "abysses":
        assert depth > 4000
    elif nature == "plateau":
        assert 50 < depth < 200
    else:
        assert 0 < depth < 200


@pytest.mark.parametrize("lat,lon,nature", CONTROLES)
def test_point_de_controle_grille_f10(meta, grille, source, lat, lon, nature):
    gm = meta["grille"]
    lig = round((gm["lat_premiere_ligne"] - lat) / gm["pas_lat_deg"])
    col = round((lon - gm["lon_premiere_colonne"]) / gm["pas_lon_deg"])
    v = float(grille[lig, col])
    if nature == "terre":
        assert math.isnan(v)
    else:
        # La cellule F10 est à moins d'une cellule GEBCO de la référence : même ordre de grandeur.
        ref = -elevation_proche(source, lat, lon)
        assert v == pytest.approx(ref, rel=0.25, abs=30)
