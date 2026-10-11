"""Préparation du dataset de patches bathymétriques (TP6, étape 2 : F02 à F06).

Pour chaque région téléchargée :

1. **Ouverture validée** du NetCDF (variable, unité, axes) puis orientation
   « ligne 0 = nord, colonne 0 = ouest ».
2. **Grille de travail** : moyenne par blocs de la grille 15″ (4 cellules en
   latitude, ``facteur_lon`` en longitude) : pixels d'environ 1,85 km de côté.
3. **Terres émergées** (F02) : un pixel est « terre » si au moins une cellule
   15″ du bloc a une élévation ≥ 0. Tout patch qui touche la terre est **exclu**.
4. **Patches** 64 × 64 (F03) tous les ``PAS_PATCH`` pixels.
5. **Séparation spatiale** (F05) : la grille est découpée en blocs de
   ``BLOC`` × ``BLOC`` pixels, tirés au sort par région en train / val / test.
   Un patch n'est gardé que si **toute** son emprise est dans des blocs du même
   ensemble : deux patches d'ensembles différents ne se recouvrent jamais.
6. **Diversité** (F02) : les patches presque plats (écart-type < ``SEUIL_PLAT_M``)
   sont limités à ``PART_MAX_PLATS`` de chaque ensemble ; le total est ramené
   vers ``CIBLE_PATCHES`` par tirage uniforme.
7. **Normalisation** (F04) : relief relatif / ``ECHELLE_M``, écrêté à ±1. La
   part de pixels écrêtés est mesurée et enregistrée.

Sorties :

  - ``data/dataset/patches.npz`` : élévations (m) de chaque ensemble ;
  - ``data/dataset/index.json``  : provenance de chaque patch (région, ligne,
    colonne, emprise géographique, profondeur moyenne) ;
  - ``data/dataset/dataset.json``: paramètres, normalisation, répartition,
    statistiques, identifiant ;
  - ``data/figures/*.png``       : exploration du dataset (F06) ;
  - ``data/dataset/rapport.json``: filtres appliqués et anomalies repérées.

Usage :
    uv run python TP6/prepare_dataset.py
    uv run python TP6/prepare_dataset.py --cible 3000 --pas 24 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

from terrain import (
    BLOC,
    CIBLE_PATCHES,
    DATASET_DIR,
    ECHELLE_M,
    FACTEUR_LAT,
    FIGURES_DIR,
    FRACTIONS,
    PART_BLOCS_VAL_TEST,
    PART_MAX_PLATS,
    PAS_NATIF_DEG,
    PAS_PATCH,
    PATCH,
    PROVENANCE_JSON,
    RAW_DIR,
    REGIONS,
    SEUIL_PLAT_M,
    colormap,
    ecrire_json,
    facteur_lon,
    indicateurs,
    lire_json,
    moyenne_blocs,
    normaliser,
    taille_pixel_km,
)

ENSEMBLES = list(FRACTIONS)
COULEURS_ENSEMBLES = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}


# --------------------------------------------------------------------------
# Grille de travail d'une région
# --------------------------------------------------------------------------

def charger_region(region: dict) -> dict:
    """Grille de travail (ligne 0 = nord) : élévation moyenne, masque terre, coordonnées des pixels."""
    chemin = RAW_DIR / f"{region['id']}.nc"
    if not chemin.exists():
        raise SystemExit(f"{chemin} absent. Lancez d'abord `mise run tp6_01_download`.")
    with xr.open_dataset(chemin, engine="netcdf4") as ds:
        var = ds["elevation"]
        if var.attrs.get("units") not in ("m", "metres", "meters"):
            raise SystemExit(f"{chemin.name} : unité inattendue {var.attrs.get('units')!r}")
        if var.dims != ("latitude", "longitude"):
            raise SystemExit(f"{chemin.name} : dimensions inattendues {var.dims}")
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        e = var.values.astype(np.float32)  # _FillValue déjà converti en NaN par xarray
    if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lat) != 0)):
        raise SystemExit(f"{chemin.name} : axes non monotones")
    if lat[1] > lat[0]:  # NetCDF du sud au nord -> image du nord au sud
        e, lat = e[::-1], lat[::-1]

    fy, fx = FACTEUR_LAT, facteur_lon(float(lat.mean()))
    grille = moyenne_blocs(e, fy, fx)
    terre = moyenne_blocs((e >= 0).astype(np.float32), fy, fx) > 0
    manquant = moyenne_blocs(np.isnan(e).astype(np.float32), fy, fx) > 0
    # Bords des pixels de travail (centres natifs ± demi-pas).
    lat_bords = np.concatenate([lat[::fy][:grille.shape[0]] + PAS_NATIF_DEG / 2,
                                [lat[grille.shape[0] * fy - 1] - PAS_NATIF_DEG / 2]])
    lon_bords = np.concatenate([lon[::fx][:grille.shape[1]] - PAS_NATIF_DEG / 2,
                                [lon[grille.shape[1] * fx - 1] + PAS_NATIF_DEG / 2]])
    dy, dx = taille_pixel_km(float(lat.mean()))
    return {"grille": grille, "terre": terre, "manquant": manquant, "lat_bords": lat_bords,
            "lon_bords": lon_bords, "facteurs": (fy, fx), "pixel_km": (dy, dx)}


def blocs_ensembles(terre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ensemble (0 train, 1 val, 2 test) de chaque bloc ``BLOC`` × ``BLOC``, tiré au sort dans la région.

    Les blocs val/test sont tirés parmi les blocs **complets** (pas en bord de
    grille) et presque entièrement marins : un bloc isolé doit pouvoir contenir
    des patches. Les blocs incomplets ou côtiers restent en entraînement.
    """
    nby, nbx = -(-terre.shape[0] // BLOC), -(-terre.shape[1] // BLOC)
    eligibles = [k for k in range(nby * nbx)
                 if (k // nbx + 1) * BLOC <= terre.shape[0] and (k % nbx + 1) * BLOC <= terre.shape[1]
                 and terre[k // nbx * BLOC:(k // nbx + 1) * BLOC, k % nbx * BLOC:(k % nbx + 1) * BLOC].mean() < 0.05]
    n = nby * nbx
    n_val = min(len(eligibles) // 2, max(1, round(n * PART_BLOCS_VAL_TEST)))
    n_test = min(len(eligibles) - n_val, max(1, round(n * PART_BLOCS_VAL_TEST)))
    affectation = np.zeros(n, dtype=np.int8)
    ordre = rng.permutation(eligibles)
    affectation[ordre[:n_val]] = 1
    affectation[ordre[n_val:n_val + n_test]] = 2
    return affectation.reshape(nby, nbx)


def patches_region(region: dict, g: dict, blocs: np.ndarray, pas: int) -> list[dict]:
    """Patches candidats de la région, avec leur ensemble et la raison d'un éventuel rejet."""
    ny, nx = g["grille"].shape
    res = []
    for r in range(0, ny - PATCH + 1, pas):
        for c in range(0, nx - PATCH + 1, pas):
            ens = np.unique(blocs[r // BLOC:(r + PATCH - 1) // BLOC + 1, c // BLOC:(c + PATCH - 1) // BLOC + 1])
            p = {"region": region["id"], "ligne": r, "colonne": c}
            if g["manquant"][r:r + PATCH, c:c + PATCH].any():
                p["rejet"] = "valeurs manquantes"
            elif g["terre"][r:r + PATCH, c:c + PATCH].any():
                p["rejet"] = "terre émergée"
            elif len(ens) > 1:
                p["rejet"] = "à cheval sur deux ensembles"
            else:
                p["ensemble"] = ENSEMBLES[int(ens[0])]
                bloc = g["grille"][r:r + PATCH, c:c + PATCH]
                p["ecart_type_m"] = float(bloc.std())
                p["plat"] = p["ecart_type_m"] < SEUIL_PLAT_M
            res.append(p)
    return res


# --------------------------------------------------------------------------
# Figures d'exploration (F06)
# --------------------------------------------------------------------------

def figure_regions(grilles: dict, blocs: dict, retenus: list[dict], chemin: Path) -> None:
    """Grille de travail de chaque région, terres en gris, emprises des patches retenus par ensemble."""
    n = len(grilles)
    cols = 5
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(4 * cols, 4 * -(-n // cols)))
    for ax, (rid, g) in zip(axes.flat, grilles.items()):
        img = np.where(g["terre"], np.nan, g["grille"])
        ax.imshow(img, cmap=colormap(), vmin=-5500, vmax=0, interpolation="nearest")
        ax.imshow(np.where(g["terre"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2, interpolation="nearest")
        for p in retenus:
            if p["region"] == rid:
                ax.add_patch(Rectangle((p["colonne"], p["ligne"]), PATCH, PATCH, fill=False, lw=0.3,
                                       ec=COULEURS_ENSEMBLES[p["ensemble"]], alpha=0.5))
        nby, nbx = blocs[rid].shape
        for by in range(nby):
            for bx in range(nbx):
                e = ENSEMBLES[blocs[rid][by, bx]]
                if e != "train":
                    ax.add_patch(Rectangle((bx * BLOC, by * BLOC), BLOC, BLOC, fill=False, lw=1.5,
                                           ec=COULEURS_ENSEMBLES[e], ls="--"))
        ax.set_title(next(r["nom"] for r in REGIONS if r["id"] == rid), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Régions GEBCO, terres (gris) et patches retenus — bleu : train, orange : val, vert : test "
                 "(tirets : blocs val/test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def figure_galerie(m: np.ndarray, infos: list[dict], titre: str, chemin: Path, n: int = 32,
                   rng: np.random.Generator | None = None, indices: np.ndarray | None = None) -> None:
    """Galerie 2D de patches réels, relief relatif sur une échelle commune ±ECHELLE_M."""
    if indices is None:
        indices = (rng or np.random.default_rng(0)).choice(len(m), size=min(n, len(m)), replace=False)
    cols = 8
    fig, axes = plt.subplots(-(-len(indices) // cols), cols, figsize=(2.1 * cols, 2.3 * -(-len(indices) // cols)))
    for ax, i in zip(np.atleast_1d(axes).flat, indices):
        r = m[i] - m[i].mean()
        ax.imshow(r, cmap=colormap(), vmin=-ECHELLE_M, vmax=ECHELLE_M)
        ax.set_title(f"{infos[i]['region']}\nσ={r.std():.0f} m", fontsize=7)
        ax.axis("off")
    for ax in list(np.atleast_1d(axes).flat)[len(indices):]:
        ax.axis("off")
    fig.suptitle(titre, fontsize=11)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def figure_3d(m: np.ndarray, infos: list[dict], indices: list[int], chemin: Path) -> None:
    """Reliefs réels en 3D (exagération verticale ×10, axes en km)."""
    fig = plt.figure(figsize=(16, 6))
    x = np.arange(PATCH) * 1.85
    xx, yy = np.meshgrid(x, x)
    for k, i in enumerate(indices):
        ax = fig.add_subplot(2, len(indices) // 2 + len(indices) % 2, k + 1, projection="3d")
        r = m[i] - m[i].mean()
        ax.plot_surface(xx, yy[::-1], r / 1000, cmap=colormap(), vmin=-ECHELLE_M / 1000, vmax=ECHELLE_M / 1000,
                        linewidth=0, antialiased=False, rstride=1, cstride=1)
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_title(f"{infos[i]['region']} (σ={r.std():.0f} m)", fontsize=8)
        ax.set_zlabel("km", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Reliefs réels en 3D (relief relatif, km ; boîte aplatie : exagération verticale ≈ ×10)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def figure_distributions(ens: dict[str, np.ndarray], infos: dict[str, list[dict]], chemin: Path) -> None:
    """Profondeurs absolues, valeurs normalisées, écarts-types et amplitudes par ensemble."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for s, m in ens.items():
        c = COULEURS_ENSEMBLES[s]
        axes[0, 0].hist(-m.reshape(-1)[::17], bins=80, histtype="step", density=True, color=c, label=s)
        axes[0, 1].hist(normaliser(m).reshape(-1)[::17], bins=100, histtype="step", density=True, color=c, label=s)
        ind = indicateurs(m, 1.85)
        axes[1, 0].hist(ind["ecart_type_m"], bins=60, histtype="step", density=True, color=c, label=s)
        axes[1, 1].hist(ind["amplitude_m"], bins=60, histtype="step", density=True, color=c, label=s)
    axes[0, 0].set_title("Profondeur absolue des pixels (m)")
    axes[0, 1].set_title(f"Valeurs normalisées (relief relatif / {ECHELLE_M:.0f} m, écrêté à ±1)")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_title("Écart-type du relief par patch (m)")
    axes[1, 0].axvline(SEUIL_PLAT_M, color="grey", ls=":", label=f"seuil « plat » {SEUIL_PLAT_M:.0f} m")
    axes[1, 1].set_title("Amplitude (max − min) par patch (m)")
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def figure_par_region(infos: list[dict], chemin: Path) -> None:
    """Nombre de patches et écart-type médian par région : la diversité du dataset."""
    regions = [r["id"] for r in REGIONS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    bas = np.zeros(len(regions))
    for s in ENSEMBLES:
        n = np.array([sum(1 for p in infos if p["region"] == r and p["ensemble"] == s) for r in regions])
        ax1.bar(regions, n, bottom=bas, color=COULEURS_ENSEMBLES[s], label=s)
        bas += n
    ax1.set_title("Patches retenus par région et par ensemble")
    ax1.legend()
    ax2.boxplot([[p["ecart_type_m"] for p in infos if p["region"] == r] or [0] for r in regions],
                tick_labels=regions, showfliers=False)
    ax2.set_title("Écart-type du relief par patch (m)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pas", type=int, default=PAS_PATCH, help="décalage entre patches candidats (pixels)")
    parser.add_argument("--cible", type=int, default=CIBLE_PATCHES, help="nombre de patches visé (F05)")
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage des blocs et des patches")
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    provenance = lire_json(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else {"regions": []}
    empreintes = {r["id"]: r["sha256"] for r in provenance["regions"]}
    grilles, blocs, candidats, infos_regions = {}, {}, [], {}
    print("Grilles de travail :")
    for region in REGIONS:
        g = charger_region(region)
        b = blocs_ensembles(g["terre"], rng)
        cand = patches_region(region, g, b, args.pas)
        grilles[region["id"]], blocs[region["id"]] = g, b
        candidats += cand
        dy, dx = g["pixel_km"]
        infos_regions[region["id"]] = {
            "nom": region["nom"], "morphologie": region["morphologie"],
            "emprise": {k: region[k] for k in ("sud", "nord", "ouest", "est")},
            "grille": list(g["grille"].shape), "facteurs_15s": list(g["facteurs"]),
            "pixel_km": {"dy": round(dy, 3), "dx": round(dx, 3)},
            "part_terre": round(float(g["terre"].mean()), 4),
            "blocs": {s: int((b == k).sum()) for k, s in enumerate(ENSEMBLES)},
            "sha256": empreintes.get(region["id"]),
        }
        ok = [p for p in cand if "rejet" not in p]
        print(f"  {region['id']:20s} {g['grille'].shape[0]:4d} × {g['grille'].shape[1]:4d} px "
              f"({dy:.2f} × {dx:.2f} km)  terre {100 * g['terre'].mean():4.1f} %  "
              f"candidats {len(cand):4d}  valides {len(ok):4d}")

    rejets: dict[str, int] = {}
    for p in candidats:
        if "rejet" in p:
            rejets[p["rejet"]] = rejets.get(p["rejet"], 0) + 1
    valides = [p for p in candidats if "rejet" not in p]

    # Diversité (F02) : plats limités à PART_MAX_PLATS de chaque ensemble.
    retenus: list[dict] = []
    ecartes_plats = 0
    for s in ENSEMBLES:
        relief = [p for p in valides if p["ensemble"] == s and not p["plat"]]
        plats = [p for p in valides if p["ensemble"] == s and p["plat"]]
        n_plats = min(len(plats), int(PART_MAX_PLATS / (1 - PART_MAX_PLATS) * len(relief)))
        ecartes_plats += len(plats) - n_plats
        retenus += relief + [plats[i] for i in sorted(rng.choice(len(plats), n_plats, replace=False))]
    # Taille cible (F05) : tirage uniforme dans chaque ensemble, vers cible × FRACTIONS.
    sous_echantillonnage = {}
    selection: list[dict] = []
    for s in ENSEMBLES:
        liste = [p for p in retenus if p["ensemble"] == s]
        n = min(len(liste), round(args.cible * FRACTIONS[s]))
        sous_echantillonnage[s] = {"avant": len(liste), "apres": n}
        selection += [liste[i] for i in np.sort(rng.choice(len(liste), n, replace=False))]
    retenus = selection

    # Assemblage, ordre stable : ensemble, région, ligne, colonne.
    ordre_region = {r["id"]: k for k, r in enumerate(REGIONS)}
    retenus.sort(key=lambda p: (ENSEMBLES.index(p["ensemble"]), ordre_region[p["region"]], p["ligne"], p["colonne"]))
    ens_m: dict[str, np.ndarray] = {}
    index: dict[str, list[dict]] = {}
    for s in ENSEMBLES:
        liste = [p for p in retenus if p["ensemble"] == s]
        ens_m[s] = np.stack([grilles[p["region"]]["grille"][p["ligne"]:p["ligne"] + PATCH,
                                                            p["colonne"]:p["colonne"] + PATCH] for p in liste])
        index[s] = []
        for i, p in enumerate(liste):
            g = grilles[p["region"]]
            r, c = p["ligne"], p["colonne"]
            index[s].append({
                "i": i, "region": p["region"], "ligne": r, "colonne": c,
                "nord": round(float(g["lat_bords"][r]), 4), "sud": round(float(g["lat_bords"][r + PATCH]), 4),
                "ouest": round(float(g["lon_bords"][c]), 4), "est": round(float(g["lon_bords"][c + PATCH]), 4),
                "profondeur_moy_m": round(float(-ens_m[s][i].mean()), 1),
                "ecart_type_m": round(p["ecart_type_m"], 1), "plat": bool(p["plat"]),
            })

    # Normalisation (F04) : part écrêtée mesurée sur l'entraînement.
    relief_train = ens_m["train"] - ens_m["train"].mean(axis=(1, 2), keepdims=True)
    part_ecretee = float((np.abs(relief_train) > ECHELLE_M).mean())
    z_train = normaliser(ens_m["train"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "patches.npz", **{f"{s}_m": ens_m[s].astype(np.float32) for s in ENSEMBLES})
    ecrire_json(args.out_dir / "index.json", index)

    parametres = {"patch": PATCH, "pas_patch": args.pas, "bloc": BLOC, "facteur_lat_15s": FACTEUR_LAT,
                  "fractions": FRACTIONS, "seuil_plat_m": SEUIL_PLAT_M, "part_max_plats": PART_MAX_PLATS,
                  "cible": args.cible, "seed": args.seed}
    empreinte = hashlib.sha256(
        (repr(sorted(parametres.items())) + repr(sorted(empreintes.items())) + repr(ECHELLE_M)).encode()
    ).hexdigest()[:12]
    stats = {}
    for s in ENSEMBLES:
        ind = indicateurs(ens_m[s], 1.85)
        stats[s] = {
            "patches": len(index[s]),
            "profondeur_moy_m": round(float(-ens_m[s].mean()), 1),
            "profondeur_min_max_m": [round(float(-ens_m[s].max()), 1), round(float(-ens_m[s].min()), 1)],
            "ecart_type_relief_median_m": round(float(np.median(ind["ecart_type_m"])), 1),
            "amplitude_mediane_m": round(float(np.median(ind["amplitude_m"])), 1),
            "plats": sum(p["plat"] for p in index[s]),
            "par_region": {r["id"]: sum(1 for p in index[s] if p["region"] == r["id"]) for r in REGIONS},
        }
    ecrire_json(args.out_dir / "dataset.json", {
        "identifiant": f"gebco2021-{empreinte}",
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": provenance.get("source"),
        "parametres": parametres,
        "convention": "ligne 0 = nord, colonne 0 = ouest ; élévation en mètres (négative sous le niveau de la mer)",
        "normalisation": {
            "methode": "relief relatif (élévation − moyenne du patch) / echelle_m, écrêté à [-1, 1]",
            "echelle_m": ECHELLE_M,
            "part_ecretee_train": round(part_ecretee, 6),
            "z_train": {"moyenne": round(float(z_train.mean()), 6), "ecart_type": round(float(z_train.std()), 6)},
            "conversion": "relief_m = z × echelle_m : écart à la profondeur moyenne du patch, en mètres",
        },
        "pixel_km_nominal": 1.85,
        "regions": infos_regions,
        "repartition": {s: len(index[s]) for s in ENSEMBLES},
        "statistiques": stats,
    })

    # Figures d'exploration (F06)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_regions(grilles, blocs, retenus, FIGURES_DIR / "01_regions.png")
    figure_galerie(ens_m["train"], index["train"], "Galerie de patches réels (train, relief relatif ±1 500 m)",
                   FIGURES_DIR / "02_galerie.png", rng=rng)
    ind_train = indicateurs(ens_m["train"], 1.85)
    # 3D : un patch par quantile d'écart-type, du plus doux au plus accidenté.
    ordre = np.argsort(ind_train["ecart_type_m"])
    figure_3d(ens_m["train"], index["train"],
              [int(ordre[int(q * (len(ordre) - 1))]) for q in (0.05, 0.25, 0.45, 0.6, 0.75, 0.85, 0.93, 0.99)],
              FIGURES_DIR / "03_reliefs_3d.png")
    figure_distributions(ens_m, index, FIGURES_DIR / "04_distributions.png")
    figure_par_region([p for s in ENSEMBLES for p in index[s] for p in [{**p, "ensemble": s}]],
                      FIGURES_DIR / "05_par_region.png")
    # Anomalies (F06) : rugosité extrême (artefacts de levés, empreintes de sondeurs
    # multifaisceaux au milieu de données satellitaires lissées) et écrêtage maximal.
    rug = ind_train["rugosite_m"] / np.maximum(ind_train["ecart_type_m"], 1)
    anomalies_rug = np.argsort(rug)[-16:][::-1]
    ecrete = (np.abs(relief_train) > ECHELLE_M).mean(axis=(1, 2))
    anomalies_ecr = np.argsort(ecrete)[-8:][::-1]
    figure_galerie(ens_m["train"], index["train"],
                   "Anomalies possibles : rugosité relative maximale (8 premiers), écrêtage maximal (8 derniers)",
                   FIGURES_DIR / "06_anomalies.png", indices=np.concatenate([anomalies_rug[:8], anomalies_ecr]))

    ecrire_json(args.out_dir / "rapport.json", {
        "candidats": len(candidats),
        "rejets": rejets,
        "plats_ecartes": ecartes_plats,
        "sous_echantillonnage": sous_echantillonnage,
        "retenus": {s: len(index[s]) for s in ENSEMBLES},
        "part_ecretee_train": part_ecretee,
        "anomalies": {
            "rugosite_relative_max": [{**index["train"][int(i)], "rugosite_relative": round(float(rug[i]), 3)}
                                      for i in anomalies_rug[:8]],
            "ecretage_max": [{**index["train"][int(i)], "part_ecretee": round(float(ecrete[i]), 3)}
                             for i in anomalies_ecr],
        },
    })

    print(f"\nCandidats : {len(candidats)}  rejets : {rejets}  plats écartés : {ecartes_plats}")
    print("Sous-échantillonnage : " + ", ".join(f"{s} {v['avant']} -> {v['apres']}"
                                                for s, v in sous_echantillonnage.items()))
    print("Répartition : " + ", ".join(f"{s} {len(index[s])}" for s in ENSEMBLES))
    for s in ENSEMBLES:
        print(f"  {s:5s} {stats[s]}")
    print(f"Normalisation : relief / {ECHELLE_M:.0f} m, part écrêtée (train) {100 * part_ecretee:.2f} %")
    print(f"Dataset {args.out_dir} ; figures {FIGURES_DIR}")
    print("Étape suivante :\n  uv run python TP6/train_vae.py")


if __name__ == "__main__":
    main()
