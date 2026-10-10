"""Préparation des fichiers cartographiques du TP3 (étapes 2 et 3).

À partir de l'extrait GEBCO, produit dans ``TP3/output/`` (servi tel quel par
le site web) :

- ``bathymetry_3857.png`` : raster coloré **reprojeté en Web Mercator**,
  transparent sur la terre, à poser avec ``L.imageOverlay`` ;
- ``points.geojson``     : points marins échantillonnés à pas régulier ;
- ``depth_grid.bin``     : grille Float32 de profondeurs (F10, clic raster) ;
- ``metadata.json``      : bornes, palette, dimensions, source, traitements ;
- ``report.json``        : rapport de préparation et statistiques.

Usage :
    uv run python TP3/prepare_bathymetry.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from gebco import (
    BBOX,
    OUTPUT_DIR,
    PALETTE,
    Grille,
    bornes_cellules,
    couleurs,
    extraire_emprise,
    hauteur_mercator,
    mercator_lat,
    mercator_y,
    ouvrir,
    source_info,
    telecharger_donnees,
)

# Nombre de points visé pour le GeoJSON (3 000 à 10 000 selon le brief).
CIBLE_POINTS = 6000
# Sous-échantillonnage de la grille numérique F10 (1 cellule sur N par axe).
PAS_GRILLE = 2
# Bornes de distribution des profondeurs (m) pour le rapport.
CLASSES = [0, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, math.inf]

PNG_NAME = "bathymetry_3857.png"
POINTS_NAME = "points.geojson"
GRID_NAME = "depth_grid.bin"


def statistiques(elev: np.ndarray, lats: np.ndarray) -> dict:
    """Comptages et statistiques des profondeurs (sans arrondi)."""
    finies = np.isfinite(elev)
    mer = finies & (elev < 0)
    depth = -elev[mer].astype(np.float64)

    # Moyenne pondérée par cos(lat) ~ aire relative des cellules lat/lon.
    poids = np.broadcast_to(np.cos(np.radians(lats))[:, None], elev.shape)[mer]

    hist, _ = np.histogram(depth, bins=CLASSES)
    return {
        "cellules_total": int(elev.size),
        "cellules_mer": int(mer.sum()),
        "cellules_terre": int((finies & (elev >= 0)).sum()),
        "cellules_invalides": int((~finies).sum()),
        "profondeur_min_m": float(depth.min()),
        "profondeur_max_m": float(depth.max()),
        "profondeur_moyenne_des_cellules_m": float(depth.mean()),
        "profondeur_moyenne_ponderee_cos_lat_m": float(np.average(depth, weights=poids)),
        "profondeur_mediane_m": float(np.median(depth)),
        "percentiles_m": {
            f"p{p}": float(v) for p, v in zip((5, 25, 75, 95), np.percentile(depth, (5, 25, 75, 95)))
        },
        "distribution": [
            {"de_m": lo, "a_m": None if math.isinf(hi) else hi, "cellules": int(n)}
            for lo, hi, n in zip(CLASSES[:-1], CLASSES[1:], hist)
        ],
        "note": (
            "La moyenne « des cellules » n'est pas une moyenne surfacique : en lat/lon "
            "les cellules rétrécissent vers le nord. La moyenne pondérée par cos(lat) "
            "l'approche."
        ),
    }


def exporter_raster(depth: np.ndarray, lats: np.ndarray, g: Grille, bornes: dict) -> dict:
    """Reprojette la grille lat/lon en Web Mercator (plus proche voisin) et l'exporte.

    En Mercator, l'axe x est linéaire en longitude : les colonnes sont reprises
    telles quelles. Seul l'axe y est rééchantillonné : chaque ligne de l'image
    correspond à une ordonnée Mercator régulière, convertie en latitude, puis en
    ligne source la plus proche. L'image ainsi produite s'aligne exactement
    avec ``L.imageOverlay`` puisque Leaflet étire l'image linéairement dans
    l'espace projeté EPSG:3857.
    """
    dlat, _ = g.resolution
    largeur = depth.shape[1]
    hauteur = hauteur_mercator(largeur, bornes)

    y_nord, y_sud = mercator_y(bornes["north"]), mercator_y(bornes["south"])
    # Centres des lignes de l'image, du nord (ligne 0) vers le sud.
    y = y_nord - (np.arange(hauteur) + 0.5) * (y_nord - y_sud) / hauteur
    lat_lignes = mercator_lat(y)
    src = np.clip(np.rint((lat_lignes - lats[0]) / dlat).astype(int), 0, depth.shape[0] - 1)

    d = depth[src, :]
    rgba = np.zeros((hauteur, largeur, 4), dtype=np.uint8)
    mer = np.isfinite(d)
    rgba[mer, :3] = couleurs(d[mer])
    rgba[mer, 3] = 255
    chemin = OUTPUT_DIR / PNG_NAME
    Image.fromarray(rgba, "RGBA").save(chemin, optimize=True)
    return {
        "fichier": PNG_NAME,
        "largeur_px": largeur,
        "hauteur_px": hauteur,
        "projection": "EPSG:3857 (Web Mercator), pixels carrés en coordonnées projetées",
        "reechantillonnage": "plus proche voisin en latitude, colonnes source conservées",
        "bounds": bornes,
        "transparence": "alpha = 0 sur terre (élévation >= 0) et cellules invalides",
    }


def exporter_grille(depth: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Grille numérique F10 : Float32 little-endian, ligne 0 = nord, NaN = terre."""
    sel = depth[::PAS_GRILLE, ::PAS_GRILLE][::-1]
    lat_sel = lats[::PAS_GRILLE][::-1]
    lon_sel = lons[::PAS_GRILLE]
    (OUTPUT_DIR / GRID_NAME).write_bytes(sel.astype("<f4").tobytes())
    return {
        "fichier": GRID_NAME,
        "dtype": "float32 little-endian",
        "lignes": int(sel.shape[0]),
        "colonnes": int(sel.shape[1]),
        "ordre": "ligne 0 = nord, colonne 0 = ouest ; index = ligne * colonnes + colonne",
        "lat_premiere_ligne": float(lat_sel[0]),
        "lon_premiere_colonne": float(lon_sel[0]),
        "pas_lat_deg": float(lat_sel[0] - lat_sel[1]),
        "pas_lon_deg": float(lon_sel[1] - lon_sel[0]),
        "valeur_absente": "NaN (terre ou invalide)",
        "pas_sous_echantillonnage": PAS_GRILLE,
        "sous_echantillonnage": f"1 cellule source sur {PAS_GRILLE} par axe, valeurs non modifiées",
    }


def exporter_points(depth: np.ndarray, lats: np.ndarray, lons: np.ndarray, n_mer: int) -> dict:
    """Échantillonnage régulier et déterministe des cellules marines."""
    pas = max(1, math.ceil(math.sqrt(n_mer / CIBLE_POINTS)))
    debut = pas // 2
    features = []
    for i in range(debut, depth.shape[0], pas):
        for j in range(debut, depth.shape[1], pas):
            d = depth[i, j]
            if not np.isfinite(d):
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lons[j]), 5), round(float(lats[i]), 5)],
                    },
                    "properties": {"depth_m": float(d)},
                }
            )
    geojson = {"type": "FeatureCollection", "features": features}
    (OUTPUT_DIR / POINTS_NAME).write_text(json.dumps(geojson, separators=(",", ":")), encoding="utf-8")
    return {
        "fichier": POINTS_NAME,
        "nombre_points": len(features),
        "strategie": (
            f"pas régulier de {pas} cellules en latitude et en longitude, à partir de "
            f"l'indice {debut} ; seules les cellules marines valides sont gardées"
        ),
        "pas_cellules": pas,
        "perte_de_resolution": (
            f"1 cellule sur {pas * pas} : espacement ≈ {pas * 15}″ "
            f"(≈ {pas * 15 / 3600 * 111:.1f} km en latitude)"
        ),
    }


def taille(nom: str) -> int:
    return (OUTPUT_DIR / nom).stat().st_size


def main() -> None:
    nc = telecharger_donnees()
    g = ouvrir(nc)
    sub = extraire_emprise(g)
    lats = sub[g.lat].values
    lons = sub[g.lon].values
    elev = sub.values  # ligne 0 = sud
    bornes = bornes_cellules(sub, g)
    print(f"Sous-grille : {elev.shape[0]} x {elev.shape[1]}, bornes {bornes}")

    # Masque terre/mer et profondeur positive (aucun arrondi).
    mer = np.isfinite(elev) & (elev < 0)
    depth = np.where(mer, -elev.astype(np.float64), np.nan)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = statistiques(elev, lats)
    raster = exporter_raster(depth, lats, g, bornes)
    grille = exporter_grille(depth, lats, lons)
    points = exporter_points(depth, lats, lons, stats["cellules_mer"])

    dlat, dlon = g.resolution
    attrs = g.ds.attrs
    doi = str(attrs.get("identifier_product_doi", "")).replace("DOI:", "").strip()
    source = {
        **source_info(),
        "fichier_netcdf": nc.name,
        "titre": attrs.get("title"),
        "version": "GEBCO_2026",
        "date_creation_grille": attrs.get("date_created"),
        "doi": doi,
        "licence": attrs.get("license"),
        "avertissement": attrs.get("comment"),
        "attribution": f"GEBCO Compilation Group (2026) GEBCO 2026 Grid (doi:{doi})",
    }
    palette = {
        "paliers": [{"depth_m": p, "couleur": c} for p, c in PALETTE],
        "interpolation": "linéaire en profondeur, par canal RGB",
        "saturation": f"profondeurs > {PALETTE[-1][0]:.0f} m affichées avec la dernière couleur",
    }
    metadata = {
        "emprise_cible": BBOX,
        "bounds": bornes,
        "resolution_native_deg": {"lat": dlat, "lon": dlon},
        "dimensions_grille": {"lignes": int(elev.shape[0]), "colonnes": int(elev.shape[1])},
        "palette": palette,
        "raster": raster,
        "grille": grille,
        "points": points,
        "source": source,
        "traitements": [
            f"sélection des cellules dont le centre est dans {BBOX}",
            "masque : élévation >= 0 (terre, niveau zéro) et valeurs non finies",
            "depth_m = -élévation sur les cellules marines, sans arrondi",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = {
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": source,
        "dimensions_grille": metadata["dimensions_grille"],
        "emprise": {"cible": BBOX, "effective_bords_cellules": bornes},
        "resolution_native": f"{dlat * 3600:.0f}″ x {dlon * 3600:.0f}″ (≈ {dlat * 111_320:.0f} m en latitude)",
        "statistiques": stats,
        "echantillonnage": points,
        "palette": palette,
        "fichiers_exportes": {
            nom: {"octets": taille(nom)}
            for nom in (PNG_NAME, POINTS_NAME, GRID_NAME, "metadata.json")
        },
        "avertissements": [
            "GEBCO est une grille compilée (sondages, altimétrie satellitaire, "
            "interpolation) : toutes les profondeurs ne sont pas des mesures directes.",
            str(attrs.get("comment", "")),
            f"Profondeurs supérieures à {PALETTE[-1][0]:.0f} m saturées dans la palette "
            f"(max observé {stats['profondeur_max_m']:.0f} m).",
            "Masque terre/mer dérivé du seul signe de l'élévation : zones intertidales "
            "et plans d'eau intérieurs mal représentés à 15″.",
            "Le raster est rééchantillonné au plus proche voisin ; les points sont "
            f"un sous-ensemble ({points['perte_de_resolution']}).",
        ],
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Cellules mer/terre/invalides : {stats['cellules_mer']} / "
          f"{stats['cellules_terre']} / {stats['cellules_invalides']}")
    print(f"Profondeur min/max/moyenne des cellules : {stats['profondeur_min_m']:.0f} / "
          f"{stats['profondeur_max_m']:.0f} / {stats['profondeur_moyenne_des_cellules_m']:.0f} m")
    print(f"Raster : {raster['largeur_px']} x {raster['hauteur_px']} px")
    print(f"Points : {points['nombre_points']} ({points['strategie']})")
    for nom, info in report["fichiers_exportes"].items():
        print(f"  {nom:22s} {info['octets'] / 1e6:6.2f} Mo")
    print(f"Fichiers écrits dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
