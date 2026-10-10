"""Préparation du dataset du TP4 (étape 2 : F02 à F05, F14).

Pour chaque zone téléchargée :

- **F02** génère le masque terre/mer (0 = mer, 1 = terre, 255 = ignoré) en
  rasterisant les polygones OSM sur la grille exacte de la scène, puis écrit
  une figure de contrôle image | masque | superposition dans ``data/controle/`` ;
- **F03** découpe scène et masque en patches 256 x 256 alignés ;
- **F04** range les patches dans train / val / test selon la zone d'origine
  (aucune zone n'est partagée entre deux ensembles) et écrit un manifeste
  ``data/dataset/dataset.json`` reproductible ;
- **F05** produit une planche d'exemples ``data/controle/dataset_exemples.png``.

Les zones « exemple » (hors Bretagne) et un extrait des zones de test sont
copiés dans ``model/examples/`` pour la galerie du site (F14).

Usage :
    uv run python TP4/prepare_dataset.py
    uv run python TP4/prepare_dataset.py --ratio-purs 0.5 --marge 3
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image

from terre_mer import (
    CONTROLE_DIR,
    DATASET_DIR,
    IGNORE,
    LAND_DIR,
    MARGE_IGNOREE_PX,
    MASKS_DIR,
    MER,
    MODEL_DIR,
    PATCH,
    SCENES_DIR,
    TERRE,
    TUILE,
    ZONES,
    Zone,
    ajouter_bande_ignoree,
    charger_scene,
    chemin_masque,
    colorer_masque,
    ecrire_json,
    lire_json,
    pixel_vers_lonlat,
    rasteriser_terre,
    superposer,
    zones,
)

# Un patch est « côtier » si sa part de terre (pixels valides) est dans cet intervalle.
SEUIL_PUR = 0.02
# Taille (en tuiles) des extraits des zones de test proposés en exemples web.
TUILES_EXEMPLE = 4


def statistiques_masque(masque: np.ndarray) -> dict:
    valides = masque != IGNORE
    n_valides = int(valides.sum())
    return {
        "pixels": int(masque.size),
        "terre": int((masque == TERRE).sum()),
        "mer": int((masque == MER).sum()),
        "ignores": int((~valides).sum()),
        "part_terre": float((masque == TERRE).sum() / n_valides) if n_valides else 0.0,
        "part_ignores": float((~valides).sum() / masque.size),
    }


def generer_masque(z: Zone, marge: int) -> np.ndarray:
    """F02 : masque aligné sur la scène, bande d'incertitude ignorée."""
    geom = shapely.from_geojson((LAND_DIR / f"{z.id}.geojson").read_text(encoding="utf-8"))
    masque = ajouter_bande_ignoree(rasteriser_terre(geom, z), marge)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masque).save(chemin_masque(z.id))
    return masque


def figure_controle(z: Zone, image: np.ndarray, masque: np.ndarray) -> None:
    """Figure de contrôle visuel : image | masque | superposition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    panneaux = [(image, "Image Sentinel-2"), (colorer_masque(masque), "Masque de référence"),
                (superposer(image, masque), "Superposition")]
    for ax, (img, titre) in zip(axes, panneaux):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(titre)
        ax.axis("off")
    s = statistiques_masque(masque)
    fig.suptitle(
        f"{z.nom} [{z.split}] — terre {100 * s['part_terre']:.1f} % des pixels valides, "
        f"ignorés {100 * s['part_ignores']:.1f} %   (vert = terre, bleu = mer, gris = ignoré)"
    )
    fig.tight_layout()
    CONTROLE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTROLE_DIR / f"{z.id}.jpg", dpi=110)
    plt.close(fig)


def decouper(z: Zone, image: np.ndarray, masque: np.ndarray, ratio_purs: float,
             rng: np.random.Generator) -> list[dict]:
    """F03 : patches non chevauchants, avec sélection des patches « purs » en train.

    En entraînement, les patches entièrement terrestres ou maritimes sont
    conservés dans la limite de ``ratio_purs`` x (nombre de patches côtiers)
    pour que le modèle voie surtout des frontières, sans ignorer les cas purs.
    En validation et test, tous les patches sont gardés : l'évaluation porte
    sur la distribution réelle des zones.
    """
    n = image.shape[0] // PATCH
    candidats = []
    for r in range(n):
        for c in range(n):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            s = statistiques_masque(masque[sl])
            if s["part_ignores"] > 0.5:
                continue
            categorie = ("mer" if s["part_terre"] <= SEUIL_PUR
                         else "terre" if s["part_terre"] >= 1 - SEUIL_PUR else "cote")
            candidats.append({"ligne": r, "colonne": c, "categorie": categorie,
                              "part_terre": round(s["part_terre"], 4),
                              "part_ignores": round(s["part_ignores"], 4), "_sl": sl})

    if z.split == "train":
        cotes = [p for p in candidats if p["categorie"] == "cote"]
        purs = [p for p in candidats if p["categorie"] != "cote"]
        # Zone sans côte (pleine terre / pleine mer) : on garde une part fixe.
        quota = int(round(ratio_purs * len(cotes))) if cotes else len(purs) // 2
        if len(purs) > quota:
            garde = sorted(rng.choice(len(purs), size=quota, replace=False).tolist())
            purs = [purs[i] for i in garde]
        candidats = sorted(cotes + purs, key=lambda p: (p["ligne"], p["colonne"]))

    sortie = []
    for p in candidats:
        nom = f"{z.id}_{p['ligne']}_{p['colonne']}.png"
        sl = p.pop("_sl")
        for sous_dossier, donnees in (("images", image[sl]), ("masks", masque[sl])):
            dossier = DATASET_DIR / z.split / sous_dossier
            dossier.mkdir(parents=True, exist_ok=True)
            Image.fromarray(donnees).save(dossier / nom)
        sortie.append({"fichier": nom, "zone": z.id, "split": z.split, **p})
    return sortie


def planche_exemples(patches: list[dict], rng: np.random.Generator, n_par_split: int = 3) -> None:
    """F05 : quelques patches côtiers par ensemble, image | masque | superposition."""
    choix = []
    for split in ("train", "val", "test"):
        cotes = [p for p in patches if p["split"] == split and p["categorie"] == "cote"]
        if cotes:
            choix += [cotes[i] for i in rng.choice(len(cotes), size=min(n_par_split, len(cotes)), replace=False)]
    if not choix:
        return
    fig, axes = plt.subplots(len(choix), 3, figsize=(9, 3 * len(choix)))
    axes = np.atleast_2d(axes)
    for ligne, p in zip(axes, choix):
        img = np.asarray(Image.open(DATASET_DIR / p["split"] / "images" / p["fichier"]))
        msk = np.asarray(Image.open(DATASET_DIR / p["split"] / "masks" / p["fichier"]))
        for ax, data, titre in zip(ligne, (img, colorer_masque(msk), superposer(img, msk)),
                                   ("Image", "Masque", "Superposition")):
            ax.imshow(data, interpolation="nearest")
            ax.set_title(f"{titre} — {p['split']} / {p['zone']}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(CONTROLE_DIR / "dataset_exemples.png", dpi=90)
    plt.close(fig)


def exporter_exemples_web() -> None:
    """F14 : images de démonstration du site (zones jamais vues à l'entraînement).

    - zones « exemple » (hors Bretagne) : scène entière ;
    - zones de test : extrait de 4 x 4 tuiles le plus côtier (plus de bande ignorée).
    """
    dossier = MODEL_DIR / "examples"
    if dossier.exists():
        shutil.rmtree(dossier)
    dossier.mkdir(parents=True)
    exemples = []
    for z in zones("test") + zones("exemple"):
        image = charger_scene(z.id)
        r0 = c0 = 0
        if z.tuiles > TUILES_EXEMPLE:
            masque = np.asarray(Image.open(chemin_masque(z.id)))
            t = TUILES_EXEMPLE * TUILE
            meilleur = -1
            for r in range(0, z.taille_px - t + 1, TUILE):
                for c in range(0, z.taille_px - t + 1, TUILE):
                    score = int((masque[r:r + t, c:c + t] == IGNORE).sum())
                    if score > meilleur:
                        meilleur, r0, c0 = score, r, c
            image = image[r0:r0 + t, c0:c0 + t]
        x0, y0 = z.origine_px
        west, north = pixel_vers_lonlat(x0 + c0, y0 + r0)
        east, south = pixel_vers_lonlat(x0 + c0 + image.shape[1], y0 + r0 + image.shape[0])
        fichier = f"{z.id}.jpg"
        Image.fromarray(image).save(dossier / fichier, quality=92)
        exemples.append({
            "id": z.id,
            "titre": z.nom,
            "configuration": z.configuration,
            "fichier": f"examples/{fichier}",
            "largeur_px": int(image.shape[1]),
            "hauteur_px": int(image.shape[0]),
            "bornes": {"west": float(west), "south": float(south), "east": float(east), "north": float(north)},
            "origine": "zone de test (Bretagne)" if z.split == "test" else "hors zone d'apprentissage",
            "vue_a_l_entrainement": False,
        })
    ecrire_json(MODEL_DIR / "examples.json", exemples)
    print(f"  {len(exemples)} exemples web écrits dans {dossier}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marge", type=int, default=MARGE_IGNOREE_PX,
                        help="demi-largeur (px) de la bande ignorée le long de la côte")
    parser.add_argument("--ratio-purs", type=float, default=1.0,
                        help="patches purs gardés en train, en proportion des patches côtiers")
    parser.add_argument("--seed", type=int, default=0, help="graine de sélection (reproductibilité)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    patches: list[dict] = []
    stats_zones = {}
    print("Masques, contrôles et patches :")
    for z in ZONES:
        image = charger_scene(z.id)
        masque = generer_masque(z, args.marge)
        assert masque.shape == image.shape[:2], "masque et image doivent avoir les mêmes dimensions"
        figure_controle(z, image, masque)
        stats_zones[z.id] = {"split": z.split, **statistiques_masque(masque)}
        if z.split in ("train", "val", "test"):
            liste = decouper(z, image, masque, args.ratio_purs, rng)
            patches += liste
            cats = {k: sum(p["categorie"] == k for p in liste) for k in ("cote", "terre", "mer")}
        else:
            cats = {}
        s = stats_zones[z.id]
        print(f"  {z.id:20s} [{z.split:7s}] terre {100 * s['part_terre']:5.1f} %  "
              f"ignorés {100 * s['part_ignores']:4.1f} %  patches {cats}")

    par_split = {}
    for split in ("train", "val", "test"):
        sel = [p for p in patches if p["split"] == split]
        par_split[split] = {
            "patches": len(sel),
            "zones": sorted({p["zone"] for p in sel}),
            "cote": sum(p["categorie"] == "cote" for p in sel),
            "terre": sum(p["categorie"] == "terre" for p in sel),
            "mer": sum(p["categorie"] == "mer" for p in sel),
        }
        print(f"  {split:5s} : {par_split[split]['patches']:4d} patches "
              f"(côte {par_split[split]['cote']}, terre {par_split[split]['terre']}, mer {par_split[split]['mer']})")

    empreinte = hashlib.sha256(
        "\n".join(f"{p['split']}/{p['fichier']}" for p in patches).encode() + f"marge={args.marge}".encode()
    ).hexdigest()[:16]
    ecrire_json(DATASET_DIR / "dataset.json", {
        "identifiant": f"terre-mer-{empreinte}",
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parametres": {"patch_px": PATCH, "marge_ignoree_px": args.marge,
                       "ratio_purs": args.ratio_purs, "seed": args.seed, "seuil_pur": SEUIL_PUR},
        "convention_masque": {"0": "mer", "1": "terre", "255": "ignoré"},
        "repartition": par_split,
        "zones": stats_zones,
        "scenes": {z.id: lire_json(SCENES_DIR / f"{z.id}.json") for z in ZONES},
        "patches": patches,
    })

    planche_exemples(patches, rng)
    exporter_exemples_web()
    print(f"Contrôles visuels : {CONTROLE_DIR}")
    print("Terminé. Étape suivante :\n  uv run python TP4/train_unet.py")


if __name__ == "__main__":
    main()
