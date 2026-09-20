"""Exploration du fichier NetCDF HYCOM 3D-SURF téléchargé depuis le SHOM.

Ce script ouvre le fichier NetCDF issu du modèle océanographique HYCOM 3D-SURF
(zone Manche-Gascogne) et décrit son contenu : dimensions, coordonnées,
variables physiques, emprise géographique, période temporelle et statistiques.

Usage :
    uv run python TP2/explore.py
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATA_DIR = Path(__file__).parent / "data"
NC_FILE = DATA_DIR / "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.nc"

# Archive d'origine sur le portail de téléchargement du SHOM (data.shom.fr).
ZIP_URL = (
    "https://services.data.shom.fr/telechargement/oceano/"
    "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.dl/file/"
    "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.dl.zip"
)


def telecharger_donnees() -> None:
    """Télécharge l'archive SHOM et extrait le NetCDF si le fichier est absent."""
    if NC_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fichier NetCDF absent. Téléchargement depuis le SHOM :")
    print(f"  {ZIP_URL}")
    with urllib.request.urlopen(ZIP_URL) as resp:
        archive = io.BytesIO(resp.read())
    with zipfile.ZipFile(archive) as zf:
        membre = next(
            (n for n in zf.namelist() if n.endswith((".nc.gz", ".nc"))), None
        )
        if membre is None:
            raise SystemExit("Aucun fichier .nc(.gz) trouvé dans l'archive SHOM.")
        contenu = zf.read(membre)
    if membre.endswith(".gz"):
        contenu = gzip.decompress(contenu)
    NC_FILE.write_bytes(contenu)
    print(f"Extrait : {NC_FILE.name} ({NC_FILE.stat().st_size / 1e6:.1f} Mo)")


def separateur(titre: str) -> None:
    """Affiche un titre de section lisible."""
    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)


def decrire_global(ds: xr.Dataset) -> None:
    """Affiche les attributs globaux du fichier."""
    separateur("ATTRIBUTS GLOBAUX (métadonnées du fichier)")
    if not ds.attrs:
        print("Aucun attribut global.")
    for cle, valeur in ds.attrs.items():
        print(f"  {cle:20s} : {valeur}")


def decrire_dimensions(ds: xr.Dataset) -> None:
    """Affiche les dimensions de la grille."""
    separateur("DIMENSIONS (taille de la grille)")
    for nom, taille in ds.sizes.items():
        print(f"  {nom:15s} : {taille} points")


def decrire_coordonnees(ds: xr.Dataset) -> None:
    """Affiche les coordonnées (temps, profondeur, latitude, longitude)."""
    separateur("COORDONNÉES (axes de la grille)")
    for nom in ds.coords:
        coord = ds[nom]
        unite = coord.attrs.get("units", "")
        valeurs = coord.values
        try:
            vmin, vmax = valeurs.min(), valeurs.max()
            print(f"  {nom:12s} [{unite}] : de {vmin} à {vmax} ({coord.size} valeurs)")
        except (TypeError, ValueError):
            print(f"  {nom:12s} [{unite}] : {coord.size} valeurs")


def decrire_variables(ds: xr.Dataset) -> None:
    """Affiche chaque variable physique avec ses statistiques."""
    separateur("VARIABLES PHYSIQUES (grandeurs océanographiques)")
    for nom in ds.data_vars:
        var = ds[nom]
        nom_long = var.attrs.get("long_name", var.attrs.get("standard_name", nom))
        unite = var.attrs.get("units", "sans unité")
        print(f"\n  - {nom}  ({nom_long})")
        print(f"      dimensions : {var.dims} -> {var.shape}")
        print(f"      unité      : {unite}")
        données = var.values.astype("float64")
        valides = données[np.isfinite(données)]
        if valides.size:
            print(
                f"      min={valides.min():.3f}  max={valides.max():.3f}  "
                f"moyenne={valides.mean():.3f}"
            )
            taux = 100 * valides.size / données.size
            print(f"      points valides (hors terre/masque) : {taux:.1f} %")
        else:
            print("      aucune valeur numérique valide")


def decrire_emprise(ds: xr.Dataset) -> None:
    """Résume l'emprise géographique et temporelle."""
    separateur("EMPRISE GÉOGRAPHIQUE ET TEMPORELLE")
    lat = next((ds[c] for c in ds.coords if c.lower() in ("lat", "latitude")), None)
    lon = next((ds[c] for c in ds.coords if c.lower() in ("lon", "longitude")), None)
    if lat is not None and lon is not None:
        print(
            f"  Zone : latitude {float(lat.min()):.2f}° à {float(lat.max()):.2f}°, "
            f"longitude {float(lon.min()):.2f}° à {float(lon.max()):.2f}°"
        )
        lat_uniq = np.unique(np.round(lat.values, 5))
        lon_uniq = np.unique(np.round(lon.values, 5))
        res_lat = float(np.median(np.diff(lat_uniq))) if lat_uniq.size > 1 else 0
        res_lon = float(np.median(np.diff(lon_uniq))) if lon_uniq.size > 1 else 0
        print(
            f"  Résolution approximative : {res_lat:.3f}° x {res_lon:.3f}° "
            f"(~{res_lat * 111:.1f} km)"
        )
    temps = next((ds[c] for c in ds.coords if "time" in c.lower()), None)
    if temps is not None:
        print(f"  Période : de {temps.values.min()} à {temps.values.max()}")
        print(f"  Nombre de pas de temps : {temps.size}")


def main() -> None:
    telecharger_donnees()

    print(f"Ouverture de : {NC_FILE.name}")
    print(f"Taille : {NC_FILE.stat().st_size / 1e6:.1f} Mo")

    with xr.open_dataset(NC_FILE) as ds:
        decrire_global(ds)
        decrire_dimensions(ds)
        decrire_coordonnees(ds)
        decrire_variables(ds)
        decrire_emprise(ds)

    separateur("SYNTHÈSE")
    print(
        "Ce fichier contient une prévision océanographique de surface issue du\n"
        "modèle HYCOM 3D-SURF (SHOM) pour la zone Manche-Gascogne (Golfe de\n"
        "Gascogne). Il fournit, sur une grille régulière et heure par heure, les\n"
        "champs physiques de l'océan de surface : température, salinité et\n"
        "composantes du courant marin."
    )


if __name__ == "__main__":
    main()
