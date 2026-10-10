"""Briques communes du TP3 : acquisition, ouverture validée, palette, Mercator.

Ce module est partagé par ``explore.py``, ``prepare_bathymetry.py`` et les
tests. Il centralise les hypothèses sur le fichier GEBCO pour qu'elles soient
**vérifiées** (et non supposées) à un seul endroit.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TP3_DIR = Path(__file__).parent
RAW_DIR = TP3_DIR / "data" / "raw"
OUTPUT_DIR = TP3_DIR / "output"

# Extrait GEBCO_2026 (n60 s40 w-10 e10) préparé sur download.gebco.net puis
# republié tel quel (archive inchangée).
ZIP_URL = "https://jlg-consulting.com/orsys/DRN/shom/GEBCO_10_Oct_2026_55ebad3abb35.zip"
ZIP_FILE = RAW_DIR / "GEBCO_10_Oct_2026_55ebad3abb35.zip"
SOURCE_FILE = RAW_DIR / "SOURCE.json"

# Emprise cible du TP (WGS84, degrés décimaux) : golfe de Gascogne.
BBOX = {"west": -10.0, "east": -1.0, "south": 43.0, "north": 49.0}

# Palette bathymétrique commune au raster et aux points : (profondeur m, couleur).
# Interpolation linéaire en profondeur entre les paliers ; saturation au-delà.
PALETTE: list[tuple[float, str]] = [
    (0.0, "#C9F2F6"),
    (200.0, "#79D2E7"),
    (1000.0, "#339AC9"),
    (2000.0, "#1761A4"),
    (3500.0, "#10346F"),
    (5000.0, "#06183C"),
]


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def telecharger_donnees() -> Path:
    """Télécharge l'archive (si absente), l'extrait et renvoie le chemin du NetCDF.

    L'archive brute est conservée inchangée dans ``data/raw/`` ; sa provenance
    (URL, date, taille, empreinte) est consignée dans ``data/raw/SOURCE.json``.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_FILE.exists():
        print("Archive GEBCO absente. Téléchargement :")
        print(f"  {ZIP_URL}")
        try:
            with urllib.request.urlopen(ZIP_URL) as resp:
                ZIP_FILE.write_bytes(resp.read())
        except OSError as exc:
            raise SystemExit(
                f"Téléchargement impossible ({exc}).\n"
                "Récupérez un extrait NetCDF sur https://download.gebco.net/ "
                f"et déposez l'archive dans {RAW_DIR}. Aucune donnée n'est simulée."
            ) from exc
        recupere = datetime.now(timezone.utc)
    else:
        recupere = datetime.fromtimestamp(ZIP_FILE.stat().st_mtime, timezone.utc)

    if not SOURCE_FILE.exists():
        SOURCE_FILE.write_text(
            json.dumps(
                {
                    "url": ZIP_URL,
                    "portail_origine": "https://download.gebco.net/",
                    "fichier": ZIP_FILE.name,
                    "taille_octets": ZIP_FILE.stat().st_size,
                    "sha256": sha256(ZIP_FILE),
                    "date_recuperation": recupere.isoformat(timespec="seconds"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    with zipfile.ZipFile(ZIP_FILE) as zf:
        membres = [n for n in zf.namelist() if n.endswith(".nc")]
        if len(membres) != 1:
            raise SystemExit(
                f"L'archive doit contenir exactement un .nc, trouvé : {membres}"
            )
        for nom in zf.namelist():
            # Les noms d'archive ne doivent pas sortir de data/raw/.
            cible = RAW_DIR / Path(nom).name
            if not cible.exists():
                cible.write_bytes(zf.read(nom))
                print(f"Extrait : {cible.name}")
    return RAW_DIR / Path(membres[0]).name


def source_info() -> dict:
    return json.loads(SOURCE_FILE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Ouverture validée
# --------------------------------------------------------------------------

@dataclass
class Grille:
    """Dataset GEBCO ouvert, avec les noms réellement trouvés dans le fichier."""

    ds: xr.Dataset
    var: str
    lat: str
    lon: str

    @property
    def resolution(self) -> tuple[float, float]:
        """Pas (dlat, dlon) en degrés, vérifié régulier."""
        return pas_regulier(self.ds[self.lat].values), pas_regulier(self.ds[self.lon].values)


def pas_regulier(coord: np.ndarray) -> float:
    d = np.diff(coord)
    if not np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
        raise SystemExit("Grille non régulière : le pipeline suppose un pas constant.")
    return float(d[0])


def trouver_coord(ds: xr.Dataset, unite: str, noms: tuple[str, ...]) -> str:
    for nom in ds.coords:
        if ds[nom].attrs.get("units") == unite or nom.lower() in noms:
            return str(nom)
    raise SystemExit(f"Coordonnée en '{unite}' introuvable parmi {list(ds.coords)}.")


def trouver_elevation(ds: xr.Dataset) -> str:
    """Identifie la variable d'élévation par ses attributs CF, pas par son nom."""
    for nom, var in ds.data_vars.items():
        if var.attrs.get("standard_name") in ("height_above_mean_sea_level", "altitude"):
            return str(nom)
    if "elevation" in ds.data_vars:
        return "elevation"
    raise SystemExit(f"Variable d'élévation introuvable parmi {list(ds.data_vars)}.")


def ouvrir(nc: Path) -> Grille:
    """Ouvre le NetCDF et valide toutes les hypothèses du pipeline."""
    ds = xr.open_dataset(nc, engine="netcdf4")
    lat = trouver_coord(ds, "degrees_north", ("lat", "latitude"))
    lon = trouver_coord(ds, "degrees_east", ("lon", "longitude"))
    var = trouver_elevation(ds)
    g = Grille(ds, var, lat, lon)

    if ds[var].dims != (lat, lon):
        raise SystemExit(f"Dimensions inattendues pour {var} : {ds[var].dims}")
    unite = ds[var].attrs.get("units")
    if unite not in ("m", "metres", "meters"):
        raise SystemExit(f"Unité de {var} inattendue : {unite!r} (mètres attendus).")
    if "crs" in ds and ds["crs"].attrs.get("grid_mapping_name") != "latitude_longitude":
        raise SystemExit("Le CRS du fichier n'est pas géographique (lat/lon).")

    dlat, dlon = g.resolution
    if dlat <= 0 or dlon <= 0:
        raise SystemExit("Axes non croissants : inverser explicitement avant traitement.")

    # Couverture évaluée sur les bords de cellules (centre ± demi-pas).
    lats, lons = ds[lat].values, ds[lon].values
    couvre = (
        lons[0] - dlon / 2 <= BBOX["west"] + 1e-9
        and lons[-1] + dlon / 2 >= BBOX["east"] - 1e-9
        and lats[0] - dlat / 2 <= BBOX["south"] + 1e-9
        and lats[-1] + dlat / 2 >= BBOX["north"] - 1e-9
    )
    if not couvre:
        raise SystemExit(f"Le fichier ne couvre pas l'emprise cible {BBOX}.")
    return g


def extraire_emprise(g: Grille) -> xr.DataArray:
    """Sous-grille dont les cellules (centres) sont dans l'emprise cible.

    Les axes restent croissants : ligne 0 = sud, colonne 0 = ouest.
    """
    return g.ds[g.var].sel(
        {g.lat: slice(BBOX["south"], BBOX["north"]), g.lon: slice(BBOX["west"], BBOX["east"])}
    )


# --------------------------------------------------------------------------
# Palette et projection
# --------------------------------------------------------------------------

def hex_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


def couleurs(depth: np.ndarray) -> np.ndarray:
    """Couleurs RGB (uint8) pour des profondeurs positives, saturées hors plage.

    Arrondi « demi vers le haut » pour coïncider avec ``Math.round`` côté web.
    """
    xs = np.array([p for p, _ in PALETTE])
    rgb = np.array([hex_rgb(c) for _, c in PALETTE], dtype=np.float64)
    d = np.clip(np.asarray(depth, dtype=np.float64), xs[0], xs[-1])
    out = np.stack([np.interp(d, xs, rgb[:, k]) for k in range(3)], axis=-1)
    return np.floor(out + 0.5).astype(np.uint8)


def mercator_y(lat_deg: np.ndarray | float) -> np.ndarray | float:
    """Ordonnée Web Mercator (sphère unité, radians) d'une latitude."""
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def mercator_lat(y: np.ndarray | float) -> np.ndarray | float:
    """Inverse de :func:`mercator_y`."""
    return np.degrees(np.arctan(np.sinh(y)))


def bornes_cellules(sub: xr.DataArray, g: Grille) -> dict[str, float]:
    """Bornes géographiques (bords de cellules) d'une sous-grille."""
    dlat, dlon = g.resolution
    lats, lons = sub[g.lat].values, sub[g.lon].values
    return {
        "west": float(lons[0] - dlon / 2),
        "east": float(lons[-1] + dlon / 2),
        "south": float(lats[0] - dlat / 2),
        "north": float(lats[-1] + dlat / 2),
    }


def hauteur_mercator(largeur: int, b: dict[str, float]) -> int:
    """Hauteur d'image donnant des pixels carrés en Web Mercator."""
    dx = math.radians(b["east"] - b["west"])
    dy = float(mercator_y(b["north"]) - mercator_y(b["south"]))
    return round(largeur * dy / dx)
