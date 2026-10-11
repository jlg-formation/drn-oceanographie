"""Génération, exploration latente et reconstruction (TP6, étape 5 : F10 à F12, F16 à F18).

Recharge les modèles entraînés **sans réentraînement** (le générateur du GAN
seul, sans discriminateur) et produit dans ``--out-dir/generation/`` :

  - ``galerie_<modele>.png`` / ``.npy`` : ``--n`` reliefs, graines ``--graine`` …
    ``--graine + n − 1`` (F11, F17). Les vecteurs latents sont tirés par le même
    générateur pseudo-aléatoire que le site web : une graine donne le même z ;
  - ``interpolation_<modele>.png`` : transition entre les graines ``--graine``
    et ``--graine-b`` (F12, F18) ;
  - ``dimensions_<modele>.png`` : effet de chaque composante de z, les autres
    étant fixées (F12, F18) ;
  - ``reconstruction.png`` : patch réel de test, reconstruction du VAE et carte
    des erreurs, avec RMSE et erreur maximale en mètres (F10).

Usage :
    uv run python TP6/generate.py
    uv run python TP6/generate.py --modele gan --graine 42 --n 32
    uv run python TP6/generate.py --patch 120 --graine 3 --graine-b 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modeles import construire_generateur, construire_vae, generer, reconstruire
from terrain import (
    DATASET_DIR,
    MODELES,
    RUNS_DIR,
    STYLES,
    charger_dataset,
    colormap,
    denormaliser,
    ecrire_json,
    lot_latent,
    vecteur_latent,
)


def charger_decodeurs(run_dir: Path) -> dict:
    """Décodeur du VAE et générateur du GAN disponibles, avec leur VAE complet pour la reconstruction."""
    res = {}
    if (run_dir / "vae" / "best.pt").exists():
        ck = torch.load(run_dir / "vae" / "best.pt", map_location="cpu", weights_only=False)
        vae = construire_vae(ck)
        res["vae"] = {"decodeur": vae.decodeur, "vae": vae, "dim": ck["config"]["dim_latente"], "epoque": ck["epoque"]}
    if (run_dir / "gan" / "best.pt").exists():
        ck = torch.load(run_dir / "gan" / "best.pt", map_location="cpu", weights_only=False)
        res["gan"] = {"decodeur": construire_generateur(ck), "dim": ck["config"]["dim_latente"], "epoque": ck["epoque"]}
    if not res:
        raise SystemExit("Aucun modèle entraîné. Lancez `mise run tp6_03_train_vae` et `mise run tp6_04_train_gan`.")
    return res


def grille(cartes: np.ndarray, titres: list[str], titre: str, chemin: Path, cols: int = 8,
           echelle: float = 1500.0) -> None:
    lignes = -(-len(cartes) // cols)
    fig, axes = plt.subplots(lignes, cols, figsize=(1.9 * cols, 2.0 * lignes + 0.6), squeeze=False)
    for ax, c, t in zip(axes.flat, cartes, titres):
        ax.imshow(c, cmap=colormap(), vmin=-echelle, vmax=echelle)
        ax.set_title(t, fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(cartes):]:
        ax.axis("off")
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modele", choices=[*MODELES, "tous"], default="tous")
    parser.add_argument("--graine", type=int, default=0, help="graine du premier relief (F11, F17)")
    parser.add_argument("--graine-b", type=int, default=1, help="graine d'arrivée de l'interpolation")
    parser.add_argument("--n", type=int, default=24, help="nombre de reliefs de la galerie")
    parser.add_argument("--etapes", type=int, default=9, help="étapes de l'interpolation")
    parser.add_argument("--dims", type=int, default=6, help="composantes latentes parcourues")
    parser.add_argument("--patch", type=int, default=None, help="indice du patch de test à reconstruire (VAE)")
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    modeles = charger_decodeurs(args.run_dir)
    ds = charger_dataset(args.data_dir)
    echelle = ds["manifeste"]["normalisation"]["echelle_m"]
    out = args.run_dir / "generation"
    out.mkdir(parents=True, exist_ok=True)
    resume: dict = {"graine": args.graine, "graine_b": args.graine_b, "modeles": {}}

    for m in (MODELES if args.modele == "tous" else [args.modele]):
        if m not in modeles:
            print(f"  {m} : modèle absent, ignoré")
            continue
        info = modeles[m]
        nom, dim = STYLES[m]["nom"], info["dim"]
        dec = info["decodeur"]

        # Galerie reproductible (F11, F17)
        z = lot_latent(args.graine, args.n, dim)
        cartes = denormaliser(generer(dec, z), echelle)
        np.save(out / f"galerie_{m}.npy", cartes)
        grille(cartes, [f"graine {args.graine + i}" for i in range(args.n)],
               f"{nom} — reliefs synthétiques (relief relatif, ±{echelle:.0f} m)", out / f"galerie_{m}.png")

        # Interpolation linéaire entre deux vecteurs latents (F12, F18)
        za, zb = vecteur_latent(args.graine, dim), vecteur_latent(args.graine_b, dim)
        t = np.linspace(0, 1, args.etapes)
        z_interp = (1 - t[:, None]) * za[None] + t[:, None] * zb[None]
        grille(denormaliser(generer(dec, z_interp.astype(np.float32)), echelle),
               [f"t = {v:.2f}" for v in t],
               f"{nom} — interpolation linéaire entre les graines {args.graine} (t = 0) et {args.graine_b} (t = 1)",
               out / f"interpolation_{m}.png", cols=args.etapes)

        # Parcours de quelques composantes (F12, F18)
        valeurs = np.linspace(-3, 3, 7)
        lot = []
        for k in range(min(args.dims, dim)):
            for v in valeurs:
                zk = za.copy()
                zk[k] = v
                lot.append(zk)
        grille(denormaliser(generer(dec, np.stack(lot)), echelle),
               [f"z[{k}] = {v:+.0f}" for k in range(min(args.dims, dim)) for v in valeurs],
               f"{nom} — effet de chaque composante (ligne k : z[k] de −3 à +3, autres composantes de la graine "
               f"{args.graine})", out / f"dimensions_{m}.png", cols=len(valeurs))
        resume["modeles"][m] = {"dim_latente": dim, "epoque": info["epoque"],
                                "galerie": f"galerie_{m}.png", "interpolation": f"interpolation_{m}.png",
                                "dimensions": f"dimensions_{m}.png"}
        print(f"  {nom} : galerie de {args.n}, interpolation, {min(args.dims, dim)} composantes -> {out}")

    # Reconstruction d'un relief réel (F10) : VAE seulement
    if "vae" in modeles:
        x = ds["test"]["z"]
        index = ds["index"]["test"]
        i = args.patch if args.patch is not None else int(np.argsort([p["ecart_type_m"] for p in index])[len(index) // 2])
        rec = reconstruire(modeles["vae"]["vae"], x[i:i + 1])[0]
        orig_m, rec_m = denormaliser(x[i], echelle), denormaliser(rec, echelle)
        erreur = rec_m - orig_m
        rmse, emax = float(np.sqrt((erreur**2).mean())), float(np.abs(erreur).max())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        p = index[i]
        for ax, img, titre, cmap, lim in (
            (axes[0], orig_m, f"Original — {p['region']} ({p['nord']:.2f}° N, {p['ouest']:.2f}° E)", colormap(), echelle),
            (axes[1], rec_m, "Reconstruction VAE (décodage de μ)", colormap(), echelle),
            (axes[2], erreur, f"Erreur (reconstruit − original) : RMSE {rmse:.0f} m, max {emax:.0f} m", "RdBu_r",
             max(emax, 1.0)),
        ):
            im = ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
            ax.set_title(titre, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        fig.tight_layout()
        fig.savefig(out / "reconstruction.png", dpi=90)
        plt.close(fig)
        resume["reconstruction"] = {"patch_test": i, **p, "rmse_m": rmse, "erreur_max_m": emax}
        print(f"  Reconstruction du patch de test {i} ({p['region']}) : RMSE {rmse:.1f} m, max {emax:.0f} m")

    ecrire_json(out / "generation.json", resume)
    print("Étape suivante :\n  uv run python TP6/evaluate.py")


if __name__ == "__main__":
    main()
