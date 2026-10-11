"""Acquisition des extraits GEBCO des régions d'intérêt (TP6, étape 1 : F01, F02).

Chaque région de ``terrain.REGIONS`` est demandée à l'ERDDAP de l'Ifremer, qui
sert la grille GEBCO_2021 complète à 15″ d'arc. La réponse NetCDF est
conservée **telle quelle** dans ``data/raw/<region>.nc`` : une région déjà
présente n'est pas redemandée (cache).

``data/provenance.json`` consigne, pour chaque région, l'emprise demandée,
l'URL exacte, la date, la taille, l'empreinte SHA-256 et les attributs du
fichier (produit, DOI, licence) : le dataset est reproductible (F01, NF06).

Usage :
    uv run python TP6/download_data.py
    uv run python TP6/download_data.py --region acores --force
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from terrain import (
    ATTRIBUTION,
    ERDDAP,
    ERDDAP_REQUETE,
    GEBCO_DOI,
    GEBCO_PRODUIT,
    PROVENANCE_JSON,
    RAW_DIR,
    REGIONS,
    ecrire_json,
    lire_json,
)


def sha256(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def telecharger(region: dict, force: bool) -> dict:
    chemin = RAW_DIR / f"{region['id']}.nc"
    url = ERDDAP_REQUETE.format(**region)
    if chemin.exists() and not force:
        print(f"  {region['id']:20s} déjà présent ({chemin.stat().st_size / 1e6:.1f} Mo)")
        date = datetime.fromtimestamp(chemin.stat().st_mtime, timezone.utc)
    else:
        print(f"  {region['id']:20s} téléchargement… ", end="", flush=True)
        t0 = time.monotonic()
        temporaire = chemin.with_suffix(".part")
        for essai in range(3):
            try:
                with urllib.request.urlopen(url, timeout=300) as rep:
                    temporaire.write_bytes(rep.read())
                break
            except OSError as exc:
                if essai == 2:
                    raise SystemExit(
                        f"\nTéléchargement impossible ({exc}) :\n  {url}\n"
                        "Vérifiez l'accès à erddap.ifremer.fr, ou déposez un extrait GEBCO NetCDF "
                        f"de la même emprise (https://download.gebco.net/) sous {chemin}. "
                        "Aucune donnée n'est simulée."
                    ) from exc
                time.sleep(5)
        temporaire.replace(chemin)
        date = datetime.now(timezone.utc)
        print(f"{chemin.stat().st_size / 1e6:.1f} Mo en {time.monotonic() - t0:.0f} s")

    with xr.open_dataset(chemin, engine="netcdf4") as ds:
        e = ds["elevation"]
        lat, lon = ds["latitude"].values, ds["longitude"].values
        attributs = {k: ds.attrs.get(k) for k in ("title", "date_created", "identifier_product_doi", "license")}
        forme = [int(v) for v in e.shape]
        bornes = {"lat_min": float(lat.min()), "lat_max": float(lat.max()),
                  "lon_min": float(lon.min()), "lon_max": float(lon.max())}
    return {
        **{k: region[k] for k in ("id", "nom", "morphologie", "sud", "nord", "ouest", "est")},
        "fichier": chemin.name,
        "url": url,
        "date_telechargement": date.isoformat(timespec="seconds"),
        "taille_octets": chemin.stat().st_size,
        "sha256": sha256(chemin),
        "grille": {"forme_lat_lon": forme, "centres": bornes},
        "attributs": attributs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", choices=[r["id"] for r in REGIONS], help="une seule région")
    parser.add_argument("--force", action="store_true", help="retélécharger même si le fichier existe")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Source : {GEBCO_PRODUIT} — {ERDDAP}")
    precedent = lire_json(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else {}
    anciennes = {r["id"]: r for r in precedent.get("regions", [])}
    for region in REGIONS:
        if args.region and region["id"] != args.region:
            continue
        anciennes[region["id"]] = telecharger(region, args.force)

    ecrire_json(PROVENANCE_JSON, {
        "source": {"produit": GEBCO_PRODUIT, "doi": GEBCO_DOI, "serveur": ERDDAP,
                   "portail_gebco": "https://www.gebco.net/", "attribution": ATTRIBUTION,
                   "licence": "domaine public, sous réserve des conditions d'utilisation GEBCO",
                   "avertissement": "GEBCO ne doit pas être utilisé pour la navigation."},
        "regions": [anciennes[r["id"]] for r in REGIONS if r["id"] in anciennes],
    })
    total = sum(r["taille_octets"] for r in anciennes.values())
    print(f"{len(anciennes)} région(s), {total / 1e6:.0f} Mo -> {RAW_DIR}")
    print(f"Provenance : {PROVENANCE_JSON}")
    print("Étape suivante :\n  uv run python TP6/prepare_dataset.py")


if __name__ == "__main__":
    main()
