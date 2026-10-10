"""Acquisition des données du TP4 (étape 1, F01).

1. Polygones terrestres OpenStreetMap : téléchargement unique de l'archive
   (~930 Mo), puis extraction des seuls polygones utiles, découpés par zone,
   dans ``data/land/<zone>.geojson`` (quelques Mo au total).
2. Images Sentinel-2 : tuiles de la mosaïque « cloudless » d'EOX, mises en
   cache dans ``data/raw/tiles/``, puis assemblées en une scène PNG par zone
   (``data/scenes/<zone>.png``) accompagnée de sa provenance (``<zone>.json``).

Le script est idempotent : ce qui est déjà présent n'est pas retéléchargé.

Usage :
    uv run python TP4/download_data.py
    uv run python TP4/download_data.py --supprimer-shapefile   # libère ~1,3 Go
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapefile
import shapely
from PIL import Image
from shapely.geometry import box, shape

from terre_mer import (
    LAND_DIR,
    LAND_URL,
    LAND_ZIP,
    RAW_DIR,
    S2_ATTRIBUTION,
    S2_LAYER,
    S2_LICENCE,
    S2_URL,
    SCENES_DIR,
    TILES_DIR,
    TUILE,
    ZONES,
    ZOOM,
    Zone,
    chemin_scene,
    ecrire_json,
    resolution_m,
)

USER_AGENT = "drn-oceanographie-TP4/1.0 (formation SHOM, usage pedagogique)"
SHP_DIR = RAW_DIR / "land-polygons-split-4326"
# Marge (degrés) autour de chaque zone lors du découpage des polygones.
MARGE_DEG = 0.02


# --------------------------------------------------------------------------
# Téléchargements
# --------------------------------------------------------------------------

def telecharger(url: str, cible: Path, progression: bool = False) -> None:
    """Télécharge ``url`` vers ``cible`` via un fichier temporaire ``.part``."""
    cible.parent.mkdir(parents=True, exist_ok=True)
    tmp = cible.with_suffix(cible.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        lu, dernier = 0, 0.0
        while bloc := resp.read(1 << 20):
            f.write(bloc)
            lu += len(bloc)
            if progression and time.monotonic() - dernier > 2:
                dernier = time.monotonic()
                pct = f"{100 * lu / total:5.1f} %" if total else ""
                print(f"  {lu / 1e6:8.1f} Mo {pct}", flush=True)
    tmp.replace(cible)


def telecharger_land_polygons() -> None:
    if LAND_ZIP.exists():
        return
    print("Archive des polygones terrestres OSM absente. Téléchargement (~930 Mo) :")
    print(f"  {LAND_URL}")
    try:
        telecharger(LAND_URL, LAND_ZIP, progression=True)
    except OSError as exc:
        raise SystemExit(
            f"Téléchargement impossible ({exc}).\n"
            f"Récupérez l'archive sur https://osmdata.openstreetmap.de/data/land-polygons.html "
            f"et déposez-la dans {RAW_DIR}."
        ) from exc


def decouper_land_polygons(a_traiter: list[Zone]) -> None:
    """Extrait, pour chaque zone, les polygones terrestres qui l'intersectent.

    Une seule lecture séquentielle du shapefile : pyshp ne décode que les
    enregistrements dont la boîte englobante touche l'emprise globale, puis
    chaque polygone est rattaché aux zones qu'il intersecte.
    """
    if not (SHP_DIR / "land_polygons.shp").exists():
        print(f"Extraction du shapefile dans {SHP_DIR} ...")
        with zipfile.ZipFile(LAND_ZIP) as zf:
            for nom in zf.namelist():
                if nom.endswith("/"):
                    continue
                cible = SHP_DIR / Path(nom).name  # pas de sortie de SHP_DIR
                cible.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(nom) as src, cible.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)

    boites = {}
    for z in a_traiter:
        b = z.bornes
        boites[z.id] = box(b["west"] - MARGE_DEG, b["south"] - MARGE_DEG,
                           b["east"] + MARGE_DEG, b["north"] + MARGE_DEG)
    emprise = shapely.unary_union(list(boites.values())).bounds
    morceaux: dict[str, list] = {zid: [] for zid in boites}

    print(f"Lecture des polygones dans l'emprise {tuple(round(v, 2) for v in emprise)} ...")
    debut = time.monotonic()
    with shapefile.Reader(str(SHP_DIR / "land_polygons.shp")) as shp:
        for forme in shp.iterShapes(bbox=list(emprise)):
            if forme is None or not forme.points:
                continue
            geom = shape(forme.__geo_interface__)
            for zid, boite in boites.items():
                if geom.intersects(boite):
                    morceaux[zid].append(geom.intersection(boite))
    print(f"  lecture terminée en {time.monotonic() - debut:.0f} s")

    LAND_DIR.mkdir(parents=True, exist_ok=True)
    for zid, geoms in morceaux.items():
        geom = shapely.union_all(geoms) if geoms else shapely.MultiPolygon()
        (LAND_DIR / f"{zid}.geojson").write_text(shapely.to_geojson(geom), encoding="utf-8")
        print(f"  {zid:20s} {len(geoms):5d} morceaux, surface {geom.area:.4f} deg²")


# --------------------------------------------------------------------------
# Tuiles Sentinel-2 et scènes
# --------------------------------------------------------------------------

def chemin_tuile(x: int, y: int) -> Path:
    return TILES_DIR / S2_LAYER / str(ZOOM) / str(x) / f"{y}.jpg"


def telecharger_tuile(xy: tuple[int, int]) -> None:
    x, y = xy
    cible = chemin_tuile(x, y)
    if cible.exists():
        return
    url = S2_URL.format(layer=S2_LAYER, z=ZOOM, x=x, y=y)
    for essai in range(3):
        try:
            telecharger(url, cible)
            return
        except OSError as exc:
            if essai == 2:
                raise SystemExit(f"Tuile inaccessible {url} : {exc}") from exc
            time.sleep(2 * (essai + 1))


def assembler_scene(z: Zone) -> None:
    """Assemble les tuiles d'une zone en une image et écrit sa provenance."""
    scene = Image.new("RGB", (z.taille_px, z.taille_px))
    for x, y in z.tuiles_xy():
        tuile = Image.open(chemin_tuile(x, y)).convert("RGB")
        scene.paste(tuile, ((x - z.tx0) * TUILE, (y - z.ty0) * TUILE))
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scene.save(chemin_scene(z.id))

    b = z.bornes
    ecrire_json(
        SCENES_DIR / f"{z.id}.json",
        {
            "identifiant": z.id,
            "zone": z.nom,
            "configuration": z.configuration,
            "split": z.split,
            "date_acquisition": "composite annuel 2024 (mosaïque sans nuages)",
            "largeur_px": z.taille_px,
            "hauteur_px": z.taille_px,
            "resolution_m_par_px": round(resolution_m((b["north"] + b["south"]) / 2), 2),
            "projection": "EPSG:3857 (Web Mercator)",
            "zoom": ZOOM,
            "tuiles": {"x0": z.tx0, "y0": z.ty0, "n": z.tuiles},
            "bornes_wgs84": b,
            "source": S2_ATTRIBUTION,
            "couche": S2_LAYER,
            "licence": S2_LICENCE,
            "moyenne_rgb": [round(float(v), 1) for v in np.asarray(scene).reshape(-1, 3).mean(0)],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supprimer-shapefile", action="store_true",
                        help="supprimer le shapefile décompressé après découpage")
    args = parser.parse_args()

    # 1. Polygones terrestres découpés par zone.
    manquantes = [z for z in ZONES if not (LAND_DIR / f"{z.id}.geojson").exists()]
    if manquantes:
        telecharger_land_polygons()
        decouper_land_polygons(manquantes)
    else:
        print("Polygones terrestres déjà découpés pour toutes les zones.")
    if args.supprimer_shapefile and SHP_DIR.exists():
        shutil.rmtree(SHP_DIR)
        print(f"Shapefile décompressé supprimé : {SHP_DIR}")

    # 2. Tuiles Sentinel-2 puis scènes.
    tuiles = sorted({xy for z in ZONES for xy in z.tuiles_xy()})
    absentes = [xy for xy in tuiles if not chemin_tuile(*xy).exists()]
    print(f"Tuiles Sentinel-2 : {len(tuiles)} au total, {len(absentes)} à télécharger.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(telecharger_tuile, absentes), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(absentes)}", flush=True)

    for z in ZONES:
        assembler_scene(z)
        print(f"  scène {z.id:20s} {z.taille_px}x{z.taille_px} px  [{z.split}]")

    ecrire_json(
        RAW_DIR / "SOURCE.json",
        {
            "land_polygons": {
                "url": LAND_URL,
                "portail": "https://osmdata.openstreetmap.de/data/land-polygons.html",
                "licence": "ODbL 1.0 - © OpenStreetMap contributors",
                "taille_octets": LAND_ZIP.stat().st_size if LAND_ZIP.exists() else None,
                "date_archive": (
                    datetime.fromtimestamp(LAND_ZIP.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                    if LAND_ZIP.exists() else None
                ),
            },
            "sentinel2": {
                "url_tuiles": S2_URL.replace("{layer}", S2_LAYER),
                "portail": "https://s2maps.eu",
                "attribution": S2_ATTRIBUTION,
                "licence": S2_LICENCE,
                "zoom": ZOOM,
                "nombre_tuiles": len(tuiles),
            },
            "date_recuperation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print("Terminé. Étape suivante :\n  uv run python TP4/prepare_dataset.py")


if __name__ == "__main__":
    main()
