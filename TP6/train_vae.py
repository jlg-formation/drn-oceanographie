"""Entraînement du VAE convolutionnel (TP6, étape 3 : F07 à F12).

Le VAE apprend à **compresser** chaque carte 64 × 64 en une petite gaussienne
de dimension ``--latent`` (encodeur), puis à la **reconstruire** (décodeur). Le
terme KL, pondéré par ``--beta``, rapproche ces gaussiennes de N(0, I) : on peut
alors tirer un z au hasard et le décoder pour **générer** un nouveau relief.

``--out-dir/vae/`` contient :

  - ``best.pt``      : poids + configuration + normalisation (rechargeable) ;
  - ``history.json`` : perte totale, reconstruction, KL (train et validation),
    RMSE de reconstruction en mètres, durée de chaque époque (F09) ;
  - ``config.json``  : paramètres, dataset, versions, durée totale (NF06, NF07) ;
  - ``courbes.png``  : évolution de l'apprentissage ;
  - ``exemples.png`` : reconstructions de validation et générations.

Usage :
    uv run python TP6/train_vae.py
    uv run python TP6/train_vae.py --latent 8 --beta 1 --epochs 80
    uv run python TP6/train_vae.py --rapide --cpu       # configuration compacte CPU (NF02, NF03)
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modeles import VAE, augmenter, nombre_parametres, perte_vae
from terrain import DATASET_DIR, RUNS_DIR, charger_dataset, colormap, ecrire_json, lot_latent


def evaluer(vae: VAE, x: torch.Tensor, beta: float, echelle: float, batch_size: int) -> dict:
    """Pertes de validation (z tiré comme à l'entraînement) et RMSE de la reconstruction déterministe (μ)."""
    vae.eval()
    tot = rec = kl = se = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            x_rec, mu, logvar = vae(xb)
            p, r, k = perte_vae(xb, x_rec, mu, logvar, beta)
            tot, rec, kl = tot + float(p) * len(xb), rec + float(r) * len(xb), kl + float(k) * len(xb)
            se += float(((vae.decodeur(mu) - xb) ** 2).sum())
    n = x.shape[0]
    return {"perte": tot / n, "reconstruction": rec / n, "kl": kl / n,
            "rmse_m": (se / x.numel()) ** 0.5 * echelle}


def tracer_courbes(historique: list[dict], chemin: Path) -> None:
    e = [h["epoque"] for h in historique]
    meilleure = min(historique, key=lambda h: h["val_perte"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for cle, ax, titre in (("perte", axes[0], "Perte totale = reconstruction + β·KL"),
                           ("reconstruction", axes[1], "Erreur de reconstruction (Σ carrés / carte)"),
                           ("kl", axes[2], "Régularisation KL (nats / carte)")):
        ax.plot(e, [h[f"train_{cle}"] for h in historique], label="entraînement")
        ax.plot(e, [h[f"val_{cle}"] for h in historique], label="validation")
        ax.set_title(titre)
        ax.set_xlabel("époque")
        ax.axvline(meilleure["epoque"], color="grey", ls=":", label=f"meilleur (époque {meilleure['epoque']})")
        ax.grid(alpha=0.3)
        ax.legend()
    ax2 = axes[1].twinx()
    ax2.plot(e, [h["val_rmse_m"] for h in historique], color="#009E73", lw=1, ls="--")
    ax2.set_ylabel("RMSE validation (m, tirets verts)", color="#009E73")
    fig.suptitle("VAE — courbes d'apprentissage")
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def tracer_exemples(vae: VAE, x_val: torch.Tensor, chemin: Path, dim: int) -> None:
    """Ligne 1 : originaux ; ligne 2 : reconstructions ; ligne 3 : générations (graines 0 à 7)."""
    vae.eval()
    with torch.no_grad():
        idx = torch.linspace(0, len(x_val) - 1, 8).long()
        orig = x_val[idx]
        mu, _ = vae.encoder(orig)
        rec = vae.decodeur(mu)
        gen = vae.decodeur(torch.from_numpy(lot_latent(0, 8, dim)).to(x_val.device))
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.4))
    for ligne, (lot, nom) in enumerate(((orig, "original"), (rec, "reconstruit"), (gen, "généré"))):
        for k in range(8):
            ax = axes[ligne, k]
            ax.imshow(lot[k, 0].cpu().numpy(), cmap=colormap(), vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nom)
    fig.suptitle("VAE — validation : originaux, reconstructions (décodage de μ) ; générations z ~ N(0, I)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent", type=int, default=16, help="dimension de l'espace latent (8 ou 16 conseillé)")
    parser.add_argument("--epochs", type=int, default=60, help="nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=64, help="taille des mini-batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage initial (Adam)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="poids de la régularisation KL (1 : VAE classique, très lissé ici ; 0,1 : compromis retenu)")
    parser.add_argument("--largeur", type=int, default=32, help="canaux de la première convolution")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles et résultats")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU même si un GPU est disponible")
    parser.add_argument("--rapide", action="store_true",
                        help="configuration compacte : largeur 16, 15 époques (quelques minutes sur CPU)")
    args = parser.parse_args()
    if args.rapide:
        args.largeur, args.epochs = 16, min(args.epochs, 15)
    return args


def main() -> None:
    args = parser_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Périphérique : {device}")

    ds = charger_dataset(args.data_dir)
    manifeste = ds["manifeste"]
    norm = manifeste["normalisation"]
    xt = torch.from_numpy(ds["train"]["z"]).unsqueeze(1).to(device)
    xv = torch.from_numpy(ds["val"]["z"]).unsqueeze(1).to(device)
    gen = torch.Generator().manual_seed(args.seed)

    vae = VAE(args.latent, args.largeur).to(device)
    n_params = nombre_parametres(vae)
    print(f"VAE : latent {args.latent}, largeur {args.largeur}, β = {args.beta}, {n_params} paramètres "
          f"(encodeur {nombre_parametres(vae.extracteur) + nombre_parametres(vae.mu) + nombre_parametres(vae.logvar)}, "
          f"décodeur {nombre_parametres(vae.decodeur)})")
    print(f"Patches : train {len(xt)}, val {len(xv)} (dataset {manifeste['identifiant']})")

    optimiseur = torch.optim.Adam(vae.parameters(), lr=args.lr)
    planning = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=args.epochs)

    out = args.out_dir / "vae"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele": {"type": "vae", "dim_latente": args.latent, "largeur": args.largeur, "parametres": n_params,
                   "architecture": "encodeur 4 conv. pas 2 -> (μ, log σ²) ; décodeur linéaire -> 4×4×256 -> "
                                   "4 × (suréchantillonnage + conv. 3×3) -> tanh"},
        "parametres": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"]},
        "normalisation": norm,
        "perte": "Σ (x − x̂)² sur 4 096 pixels + β · KL(N(μ, σ²) ‖ N(0, I)), moyennes par carte",
        "optimiseur": "Adam, décroissance cosinus du taux",
        "augmentation": "8 symétries du carré (rotations de 90°, miroirs), entraînement seulement",
        "selection": "plus faible perte totale de validation",
        "environnement": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "device": str(device)},
    }
    ecrire_json(out / "config.json", config)

    historique: list[dict] = []
    meilleure = float("inf")
    debut = time.monotonic()
    n = len(xt)
    for epoque in range(1, args.epochs + 1):
        t0 = time.monotonic()
        vae.train()
        somme = {"perte": 0.0, "reconstruction": 0.0, "kl": 0.0}
        permutation = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, args.batch_size):
            xb = augmenter(xt[permutation[i:i + args.batch_size]], gen)
            optimiseur.zero_grad()
            x_rec, mu, logvar = vae(xb)
            perte, rec, kl = perte_vae(xb, x_rec, mu, logvar, args.beta)
            perte.backward()
            optimiseur.step()
            for cle, v in (("perte", perte), ("reconstruction", rec), ("kl", kl)):
                somme[cle] += float(v) * len(xb)
        planning.step()

        val = evaluer(vae, xv, args.beta, norm["echelle_m"], 256)
        entree = {"epoque": epoque, **{f"train_{k}": v / n for k, v in somme.items()},
                  **{f"val_{k}": v for k, v in val.items() if k != "rmse_m"}, "val_rmse_m": val["rmse_m"],
                  "lr": optimiseur.param_groups[0]["lr"], "duree_s": round(time.monotonic() - t0, 2)}
        historique.append(entree)
        marque = ""
        if val["perte"] < meilleure:
            meilleure = val["perte"]
            marque = "  * meilleur"
            torch.save({
                "state_dict": vae.state_dict(),
                "config": {"type": "vae", "dim_latente": args.latent, "largeur": args.largeur, "beta": args.beta},
                "normalisation": norm,
                "epoque": epoque,
                "val": val,
                "dataset": manifeste["identifiant"],
                "date_creation": config["date_creation"],
            }, out / "best.pt")
        print(f"  époque {epoque:3d}/{args.epochs}  perte {entree['train_perte']:7.2f} "
              f"(rec {entree['train_reconstruction']:6.2f}, KL {entree['train_kl']:5.1f})  "
              f"val {val['perte']:7.2f} (rec {val['reconstruction']:6.2f}, KL {val['kl']:5.1f})  "
              f"RMSE val {val['rmse_m']:5.1f} m  ({entree['duree_s']:.1f} s){marque}", flush=True)
        ecrire_json(out / "history.json", historique)

    duree = time.monotonic() - debut
    best = min(historique, key=lambda h: h["val_perte"])
    config["resultats"] = {"duree_totale_s": round(duree, 1), "epoques": args.epochs,
                           "meilleure_epoque": best["epoque"], "val_perte": best["val_perte"],
                           "val_reconstruction": best["val_reconstruction"], "val_kl": best["val_kl"],
                           "val_rmse_m": best["val_rmse_m"]}
    ecrire_json(out / "config.json", config)
    tracer_courbes(historique, out / "courbes.png")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["state_dict"])
    tracer_exemples(vae, xv, out / "exemples.png", args.latent)
    print(f"VAE : {duree:.0f} s, meilleure époque {best['epoque']}, RMSE de reconstruction val "
          f"{best['val_rmse_m']:.1f} m, KL {best['val_kl']:.1f} nats -> {out}")
    print("Étape suivante :\n  uv run python TP6/train_gan.py")


if __name__ == "__main__":
    main()
