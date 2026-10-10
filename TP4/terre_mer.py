"""Briques communes du TP4 : zones, tuiles Sentinel-2, polygones terrestres, masques.

Ce module est partagé par ``download_data.py``, ``prepare_dataset.py``,
``train_unet.py``, ``evaluate.py``, ``export_onnx.py`` et les tests. Il
centralise les hypothèses géométriques (grille de tuiles Web Mercator, zones,
convention des masques) pour qu'elles soient définies à un seul endroit.

Convention des masques (brief §3.3) :
    0   = Mer
    1   = Terre
    255 = Pixel ignoré (bande d'incertitude le long du trait de côte)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TP4_DIR = Path(__file__).parent
DATA_DIR = TP4_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TILES_DIR = RAW_DIR / "tiles"
LAND_DIR = DATA_DIR / "land"  # polygones terrestres découpés par zone (GeoJSON)
SCENES_DIR = DATA_DIR / "scenes"  # images satellites assemblées + provenance
MASKS_DIR = DATA_DIR / "masks"  # masques de référence 0/1/255
CONTROLE_DIR = DATA_DIR / "controle"  # visualisations de contrôle (F02, F05)
DATASET_DIR = DATA_DIR / "dataset"  # patches 256x256 répartis train/val/test
RUNS_DIR = TP4_DIR / "runs" / "unet"  # résultats d'entraînement et d'évaluation
MODEL_DIR = TP4_DIR / "model"  # servi tel quel par le site web (ONNX, exemples)

# --------------------------------------------------------------------------
# Sources de données
# --------------------------------------------------------------------------

# Polygones terrestres OpenStreetMap (osmdata.openstreetmap.de), découpés en
# morceaux (« split ») et en WGS84. ~930 Mo : téléchargés une seule fois.
LAND_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
LAND_ZIP = RAW_DIR / "land-polygons-split-4326.zip"

# Mosaïque Sentinel-2 « cloudless » 2024 d'EOX (s2maps.eu) : composite RGB sans
# nuages construit à partir des scènes Sentinel-2 de l'année. Servie en tuiles
# Web Mercator 256x256 sans compte ni clé, ce qui rend le TP reproductible.
S2_LAYER = "s2cloudless-2024_3857"
S2_URL = "https://tiles.maps.eox.at/wmts/1.0.0/{layer}/default/g/{z}/{y}/{x}.jpg"
S2_ATTRIBUTION = (
    "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH "
    "(Contains modified Copernicus Sentinel data 2024)"
)
S2_LICENCE = "CC BY-NC-SA 4.0"

# Niveau de zoom Web Mercator : ~12,8 m/pixel en Bretagne (48° N), proche de la
# résolution native de 10 m des bandes RGB de Sentinel-2.
ZOOM = 13
TUILE = 256  # taille d'une tuile WMTS (pixels)

# --------------------------------------------------------------------------
# Paramètres du dataset
# --------------------------------------------------------------------------

PATCH = 256  # dimension de référence des patches (F03)
MER, TERRE, IGNORE = 0, 1, 255
# Demi-largeur (pixels) de la bande ignorée de part et d'autre du trait de côte :
# absorbe marées, estran et décalage carte/image (brief §3.4). 2 px ~ 25 m.
MARGE_IGNOREE_PX = 2

# Couleurs communes aux figures Python et au site web (F18).
COULEURS = {"terre": "#2E9D57", "mer": "#1F6FD1", "ignore": "#9AA0A6", "erreur": "#E5484D"}


# --------------------------------------------------------------------------
# Projection Web Mercator (EPSG:3857), coordonnées pixel globales au zoom Z
# --------------------------------------------------------------------------

def taille_monde(z: int = ZOOM) -> int:
    return TUILE * 2**z


def lonlat_vers_pixel(lon, lat, z: int = ZOOM):
    """Coordonnées pixel globales (origine nord-ouest) d'un point WGS84."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    monde = taille_monde(z)
    x = (lon + 180.0) / 360.0 * monde
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * monde
    return x, y


def pixel_vers_lonlat(x, y, z: int = ZOOM):
    """Inverse de :func:`lonlat_vers_pixel`."""
    monde = taille_monde(z)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = x / monde * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / monde))))
    return lon, lat


def resolution_m(lat: float, z: int = ZOOM) -> float:
    """Taille au sol (m) d'un pixel Web Mercator à la latitude donnée."""
    return 2 * math.pi * 6378137.0 * math.cos(math.radians(lat)) / taille_monde(z)


# --------------------------------------------------------------------------
# Zones géographiques
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    """Zone littorale carrée de ``tuiles x tuiles`` tuiles WMTS.

    ``split`` vaut ``train``, ``val``, ``test`` (dataset) ou ``exemple``
    (images de démonstration du site, jamais vues à l'entraînement).
    """

    id: str
    nom: str
    lon: float
    lat: float
    split: str
    configuration: str
    tuiles: int = 8

    @property
    def tx0(self) -> int:
        x, _ = lonlat_vers_pixel(self.lon, self.lat)
        return int(round(float(x) / TUILE - self.tuiles / 2))

    @property
    def ty0(self) -> int:
        _, y = lonlat_vers_pixel(self.lon, self.lat)
        return int(round(float(y) / TUILE - self.tuiles / 2))

    @property
    def taille_px(self) -> int:
        return self.tuiles * TUILE

    @property
    def origine_px(self) -> tuple[int, int]:
        """Coordonnées pixel globales du coin nord-ouest."""
        return self.tx0 * TUILE, self.ty0 * TUILE

    @property
    def bornes(self) -> dict[str, float]:
        """Bornes WGS84 exactes (bords de pixels) de l'image de la zone."""
        x0, y0 = self.origine_px
        west, north = pixel_vers_lonlat(x0, y0)
        east, south = pixel_vers_lonlat(x0 + self.taille_px, y0 + self.taille_px)
        return {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}

    def tuiles_xy(self) -> list[tuple[int, int]]:
        return [
            (self.tx0 + i, self.ty0 + j) for j in range(self.tuiles) for i in range(self.tuiles)
        ]


# Zones du dataset : Bretagne, configurations littorales variées (brief §3.1).
# La séparation train / val / test est faite PAR ZONE, et les zones de splits
# différents sont séparées d'au moins une tuile (vérifié par les tests) : aucun
# patch de test n'est voisin immédiat d'un patch d'entraînement (F04).
ZONES: list[Zone] = [
    # --- Entraînement ---
    Zone("rade_brest", "Rade de Brest", -4.42, 48.30, "train", "rade, estuaire, port"),
    Zone("ouessant", "Ouessant et Molène", -5.02, 48.43, "train", "îles, majoritairement mer"),
    Zone("abers", "Abers et Portsall", -4.62, 48.60, "train", "côte découpée, abers"),
    Zone("morlaix", "Baie de Morlaix", -3.85, 48.68, "train", "baie, îlots, estuaire"),
    Zone("perros", "Côte de Granit rose", -3.40, 48.82, "train", "côte rocheuse, archipel"),
    Zone("saint_malo", "Saint-Malo et la Rance", -2.05, 48.62, "train", "port, estuaire, ville"),
    Zone("douarnenez", "Baie de Douarnenez", -4.45, 48.03, "train", "baie, presqu'île"),
    Zone("audierne", "Baie d'Audierne", -4.42, 47.78, "train", "littoral rectiligne"),
    Zone("morbihan", "Golfe du Morbihan", -2.85, 47.58, "train", "golfe, îles, côte très découpée"),
    Zone("belle_ile", "Belle-Île", -3.18, 47.33, "train", "île, majoritairement mer"),
    Zone("argoat", "Centre Bretagne", -3.20, 48.25, "train", "entièrement terrestre"),
    Zone("large_sein", "Large de l'île de Sein", -5.25, 47.95, "train", "entièrement maritime"),
    # --- Validation ---
    Zone("lorient", "Lorient et Groix", -3.40, 47.70, "val", "port, rade, île"),
    Zone("saint_brieuc", "Baie de Saint-Brieuc", -2.68, 48.55, "val", "baie, falaises"),
    # --- Test (zones inconnues du modèle) ---
    Zone("glenan", "Concarneau et les Glénan", -3.92, 47.78, "test", "archipel, port"),
    Zone("brehat", "Paimpol et Bréhat", -2.98, 48.83, "test", "archipel, estuaire"),
    # --- Exemples du site, hors Bretagne (zones géographiques non apprises, F14) ---
    Zone("arcachon", "Bassin d'Arcachon", -1.18, 44.66, "exemple", "lagune, chenaux", tuiles=4),
    Zone("mont_saint_michel", "Baie du Mont-Saint-Michel", -1.52, 48.64, "exemple",
         "vaste estran (cas difficile)", tuiles=4),
    Zone("calanques", "Calanques de Marseille", 5.42, 43.21, "exemple", "falaises calcaires", tuiles=4),
    Zone("ajaccio", "Golfe d'Ajaccio", 8.70, 41.90, "exemple", "golfe, relief", tuiles=4),
]


def zones(split: str | None = None) -> list[Zone]:
    return [z for z in ZONES if split is None or z.split == split]


def zone(zone_id: str) -> Zone:
    for z in ZONES:
        if z.id == zone_id:
            return z
    raise KeyError(zone_id)


# --------------------------------------------------------------------------
# Scènes (images assemblées) et masques sur disque
# --------------------------------------------------------------------------

def chemin_scene(zone_id: str) -> Path:
    return SCENES_DIR / f"{zone_id}.png"


def chemin_masque(zone_id: str) -> Path:
    return MASKS_DIR / f"{zone_id}.png"


def charger_scene(zone_id: str) -> np.ndarray:
    """Image RGB uint8 (H, W, 3) d'une zone."""
    chemin = chemin_scene(zone_id)
    if not chemin.exists():
        raise SystemExit(f"Scène absente : {chemin}. Lancez d'abord `mise run tp4_01_download`.")
    return np.asarray(Image.open(chemin).convert("RGB"))


def charger_masque(zone_id: str) -> np.ndarray:
    """Masque uint8 (H, W) 0/1/255 d'une zone."""
    chemin = chemin_masque(zone_id)
    if not chemin.exists():
        raise SystemExit(f"Masque absent : {chemin}. Lancez d'abord `mise run tp4_02_dataset`.")
    return np.asarray(Image.open(chemin))


def lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def ecrire_json(chemin: Path, contenu: dict | list) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Génération des masques (F02)
# --------------------------------------------------------------------------

def polygones(geometrie) -> list:
    """Liste des ``Polygon`` d'une géométrie shapely (Polygon ou MultiPolygon)."""
    if geometrie.is_empty:
        return []
    if geometrie.geom_type == "Polygon":
        return [geometrie]
    if geometrie.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [p for g in geometrie.geoms for p in polygones(g)]
    return []


def rasteriser_terre(geometrie, z: Zone) -> np.ndarray:
    """Masque binaire uint8 (1 = terre) des polygones WGS84 sur la grille de la zone.

    Chaque sommet est projeté en Web Mercator puis exprimé en pixels de l'image :
    le masque est donc aligné pixel à pixel avec la scène (mêmes tuiles).
    Les polygones sont dessinés du plus grand au plus petit pour qu'un îlot situé
    dans un trou (lagune) ne soit pas effacé par ce trou.
    """
    x0, y0 = z.origine_px
    img = Image.new("L", (z.taille_px, z.taille_px), 0)
    dessin = ImageDraw.Draw(img)

    def anneau(coords) -> list[tuple[float, float]]:
        c = np.asarray(coords, dtype=np.float64)
        x, y = lonlat_vers_pixel(c[:, 0], c[:, 1])
        return list(zip((x - x0).tolist(), (y - y0).tolist()))

    for poly in sorted(polygones(geometrie), key=lambda p: p.area, reverse=True):
        dessin.polygon(anneau(poly.exterior.coords), fill=TERRE)
        for trou in poly.interiors:
            dessin.polygon(anneau(trou.coords), fill=MER)
    return np.asarray(img, dtype=np.uint8)


def ajouter_bande_ignoree(binaire: np.ndarray, marge: int = MARGE_IGNOREE_PX) -> np.ndarray:
    """Remplace par 255 les pixels à moins de ``marge`` pixels du trait de côte.

    Un pixel est dans la bande si son voisinage (2*marge+1)^2 contient à la fois
    de la terre et de la mer (dilatation != érosion).
    """
    if marge <= 0:
        return binaire.copy()
    img = Image.fromarray(binaire)
    k = 2 * marge + 1
    dilate = np.asarray(img.filter(ImageFilter.MaxFilter(k)))
    erode = np.asarray(img.filter(ImageFilter.MinFilter(k)))
    masque = binaire.copy()
    masque[dilate != erode] = IGNORE
    return masque


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------

def hex_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


def colorer_masque(masque: np.ndarray) -> np.ndarray:
    """Masque 0/1/255 -> image RGB (mer bleu, terre vert, ignoré gris)."""
    rgb = np.zeros((*masque.shape, 3), dtype=np.uint8)
    rgb[masque == MER] = hex_rgb(COULEURS["mer"])
    rgb[masque == TERRE] = hex_rgb(COULEURS["terre"])
    rgb[masque == IGNORE] = hex_rgb(COULEURS["ignore"])
    return rgb


def superposer(image: np.ndarray, masque: np.ndarray, opacite: float = 0.45) -> np.ndarray:
    """Masque coloré par-dessus l'image (les pixels ignorés restent visibles)."""
    couleur = colorer_masque(masque).astype(np.float32)
    alpha = np.where(masque == IGNORE, 0.0, opacite)[..., None]
    out = image.astype(np.float32) * (1 - alpha) + couleur * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
