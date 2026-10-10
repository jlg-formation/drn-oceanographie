"""Inspection du NetCDF GEBCO (TP3, étape 1).

Télécharge l'extrait si besoin, puis affiche dimensions, coordonnées,
variables, attributs, valeurs manquantes, orientation des axes et couverture
de l'emprise cible. Toute hypothèse fausse provoque un arrêt explicite.

Usage :
    uv run python TP3/explore.py
"""

from __future__ import annotations

import numpy as np

from gebco import BBOX, extraire_emprise, ouvrir, telecharger_donnees


def separateur(titre: str) -> None:
    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)


def main() -> None:
    nc = telecharger_donnees()
    print(f"Ouverture de : {nc.name} ({nc.stat().st_size / 1e6:.1f} Mo)")
    g = ouvrir(nc)
    ds = g.ds

    separateur("ATTRIBUTS GLOBAUX")
    for cle in ("title", "date_created", "institution", "identifier_product_doi",
                "license", "comment", "geospatial_bounds_vertical_crs"):
        if cle in ds.attrs:
            print(f"  {cle:32s} : {ds.attrs[cle]}")
    print(f"  ({len(ds.attrs)} attributs au total)")

    separateur("DIMENSIONS")
    for nom, taille in ds.sizes.items():
        print(f"  {nom:6s} : {taille}")

    separateur("COORDONNÉES")
    dlat, dlon = g.resolution
    for nom, pas in ((g.lat, dlat), (g.lon, dlon)):
        c = ds[nom]
        print(
            f"  {nom:6s} [{c.attrs.get('units')}] : {float(c[0]):.5f} -> {float(c[-1]):.5f}"
            f"  pas {pas:.6f}° ({pas * 3600:.0f}″)  {'croissant' if pas > 0 else 'décroissant'}"
        )
    print("  Les coordonnées sont les CENTRES des cellules (node_offset = "
          f"{ds.attrs.get('node_offset', '?')}).")

    separateur("VARIABLES")
    for nom, var in ds.data_vars.items():
        print(f"  - {nom} {var.dims} {var.dtype}")
        for k, v in var.attrs.items():
            if k not in ("crs_wkt", "spatial_ref"):
                print(f"      {k} : {v}")
    print(f"\n  Variable d'élévation retenue : '{g.var}' (repérée par standard_name)")

    separateur(f"EMPRISE CIBLE {BBOX}")
    sub = extraire_emprise(g)
    e = sub.values
    finies = np.isfinite(e)
    print(f"  Sous-grille : {sub.shape[0]} lignes (lat) x {sub.shape[1]} colonnes (lon)")
    print(f"  Valeurs manquantes/non finies : {(~finies).sum()}")
    print(f"  Élévation min / max : {np.nanmin(e):.0f} m / {np.nanmax(e):.0f} m")
    print(f"  Cellules mer (élévation < 0)   : {(finies & (e < 0)).sum()}")
    print(f"  Cellules terre (élévation >= 0): {(finies & (e >= 0)).sum()}")

    # Contrôle d'orientation : le nord de l'emprise (Bretagne) contient de la
    # terre à l'est, le sud-ouest est en pleine mer profonde.
    so = float(sub.isel({g.lat: 0, g.lon: 0}))
    print(f"  Coin sud-ouest ({float(sub[g.lat][0]):.3f}, {float(sub[g.lon][0]):.3f}) : {so:.0f} m")
    if so >= 0:
        raise SystemExit("Coin sud-ouest émergé : axes probablement inversés.")

    separateur("SYNTHÈSE")
    print(
        "Grille GEBCO régulière en latitude/longitude WGS84 (EPSG:4326), valeurs en\n"
        "mètres, positives au-dessus du niveau moyen de la mer. Profondeur affichée :\n"
        "depth_m = -elevation sur les seules cellules marines. Prochaine étape :\n"
        "  uv run python TP3/prepare_bathymetry.py"
    )


if __name__ == "__main__":
    main()
