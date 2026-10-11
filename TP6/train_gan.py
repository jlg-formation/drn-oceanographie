"""Entraînement du GAN convolutionnel compact (TP6, étape 4 : F13 à F18).

Deux réseaux s'affrontent sur les mêmes patches que le VAE :

- le **discriminateur** D apprend à dire si une carte est réelle (GEBCO) ou
  produite par le générateur ;
- le **générateur** G (même architecture que le décodeur du VAE) apprend à
  produire, à partir d'un vecteur latent z ~ N(0, I), des cartes que D prend
  pour des vraies.

Les deux sont mis à jour **alternativement** à chaque mini-batch : D d'abord,
puis G. Il n'y a pas de perte de validation interprétable : les pertes de D et
G montent et descendent l'une contre l'autre. On suit donc aussi :

  - D(x) et D(G(z)) : probabilités moyennes « réel » données par D ;
  - la **diversité** des générations (distance moyenne entre paires, rapportée
    à celle des cartes réelles) : une chute signale un *mode collapse* (F21) ;
  - l'**écart statistique** entre générations et validation (distributions
    d'amplitude, pente, rugosité…) : il sert à choisir le générateur conservé.

``--out-dir/gan/`` contient ``best.pt`` (G et D, plus faible écart
statistique), ``last.pt`` (dernière époque), ``history.json``, ``config.json``,
``courbes.png``, ``evolution.npz`` et ``evolution.png`` (générations à z fixé
au fil des époques, F15).

Usage :
    uv run python TP6/train_gan.py
    uv run python TP6/train_gan.py --latent 16 --epochs 300 --lr-g 1e-4
    uv run python TP6/train_gan.py --sans-spectrale     # DCGAN classique, plus instable
    uv run python TP6/train_gan.py --rapide --cpu       # configuration compacte CPU (NF02, NF03)
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
import torch.nn.functional as F

from modeles import Decodeur, Discriminateur, augmenter, generer, nombre_parametres
from terrain import (
    DATASET_DIR,
    RUNS_DIR,
    charger_dataset,
    colormap,
    denormaliser,
    diversite,
    ecart_statistique,
    ecrire_json,
    lot_latent,
)

N_EVOLUTION = 16  # cartes suivies à z fixé au fil des époques (F15)


def pertes(d_reel: torch.Tensor, d_faux: torch.Tensor, type_perte: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pertes (D, G) à partir des logits du discriminateur.

    - ``bce``   : entropie croisée binaire, version « non saturante » pour G ;
    - ``hinge`` : perte charnière (souvent plus stable).
    """
    if type_perte == "hinge":
        return F.relu(1 - d_reel).mean() + F.relu(1 + d_faux).mean(), -d_faux.mean()
    perte_d = (F.binary_cross_entropy_with_logits(d_reel, torch.ones_like(d_reel))
               + F.binary_cross_entropy_with_logits(d_faux, torch.zeros_like(d_faux)))
    return perte_d, F.binary_cross_entropy_with_logits(d_faux, torch.ones_like(d_faux))


def tracer_courbes(historique: list[dict], meilleure: int, chemin: Path) -> None:
    e = [h["epoque"] for h in historique]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(e, [h["perte_d"] for h in historique], label="discriminateur D", color="#0072B2")
    axes[0].plot(e, [h["perte_g"] for h in historique], label="générateur G", color="#D55E00")
    axes[0].set_title("Pertes (moyennes par époque)")
    axes[1].plot(e, [h["d_reel"] for h in historique], label="D(x) : cartes réelles", color="#0072B2")
    axes[1].plot(e, [h["d_faux"] for h in historique], label="D(G(z)) : cartes générées", color="#D55E00")
    axes[1].axhline(0.5, color="grey", ls=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Probabilité « réel » donnée par D")
    suivi = [h for h in historique if h.get("ecart_stat") is not None]
    axes[2].plot([h["epoque"] for h in suivi], [h["ecart_stat"] for h in suivi], "o-", ms=3,
                 label="écart statistique à la validation", color="#7C3AED")
    axes[2].plot([h["epoque"] for h in suivi], [h["diversite_relative"] for h in suivi], "s--", ms=3,
                 label="diversité relative (1 = celle des réels)", color="#009E73")
    axes[2].set_title("Suivi de la qualité et de la diversité")
    for ax in axes:
        ax.set_xlabel("époque")
        ax.axvline(meilleure, color="grey", ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"GAN — courbes d'apprentissage (trait pointillé : générateur retenu, époque {meilleure})")
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def tracer_evolution(epoques: list[int], cartes: np.ndarray, chemin: Path, max_lignes: int = 10) -> None:
    """Générations à z fixé (colonnes) au fil des époques (lignes)."""
    lignes = np.unique(np.linspace(0, len(epoques) - 1, min(max_lignes, len(epoques))).round().astype(int))
    fig, axes = plt.subplots(len(lignes), 8, figsize=(12, 1.6 * len(lignes)))
    for li, k in enumerate(lignes):
        for c in range(8):
            ax = axes[li, c]
            ax.imshow(cartes[k, c].astype(np.float32), cmap=colormap(), vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"ép. {epoques[k]}", fontsize=8)
    fig.suptitle("GAN — mêmes vecteurs z (colonnes) au fil des époques (lignes)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=80)
    plt.close(fig)


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent", type=int, default=32, help="dimension de l'espace latent")
    parser.add_argument("--epochs", type=int, default=300, help="nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=64, help="taille des mini-batches")
    parser.add_argument("--lr-g", type=float, default=2e-4, help="taux d'apprentissage du générateur (Adam)")
    parser.add_argument("--lr-d", type=float, default=2e-4, help="taux d'apprentissage du discriminateur (Adam)")
    parser.add_argument("--beta1", type=float, default=0.5, help="β1 d'Adam (0,5 : réglage DCGAN)")
    parser.add_argument("--beta2", type=float, default=0.999, help="β2 d'Adam")
    parser.add_argument("--perte", choices=["bce", "hinge"], default="bce", help="fonction de perte adverse")
    parser.add_argument("--sans-spectrale", action="store_true",
                        help="discriminateur avec BatchNorm au lieu de la normalisation spectrale (DCGAN classique)")
    parser.add_argument("--largeur", type=int, default=32, help="canaux de la première convolution")
    parser.add_argument("--suivi", type=int, default=5, help="période (époques) du suivi et des instantanés")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles et résultats")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU même si un GPU est disponible")
    parser.add_argument("--rapide", action="store_true",
                        help="configuration compacte : largeur 16, 30 époques (quelques minutes sur CPU)")
    args = parser.parse_args()
    if args.rapide:
        args.largeur, args.epochs, args.suivi = 16, min(args.epochs, 30), 3
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
    echelle = norm["echelle_m"]
    xt = torch.from_numpy(ds["train"]["z"]).unsqueeze(1).to(device)
    # Référence du suivi : 512 cartes de validation (statistiques) et leur diversité.
    rng = np.random.default_rng(args.seed)
    ref_val = ds["val"]["z"][rng.choice(len(ds["val"]["z"]), min(512, len(ds["val"]["z"])), replace=False)]
    diversite_reelle = diversite(ref_val[:256])
    z_suivi = lot_latent(10_000, len(ref_val), args.latent)  # graines fixes, distinctes des galeries
    z_evolution = torch.from_numpy(lot_latent(0, N_EVOLUTION, args.latent)).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    gen_z = torch.Generator(device=device).manual_seed(args.seed)

    g = Decodeur(args.latent, args.largeur).to(device)
    d = Discriminateur(args.largeur, spectrale=not args.sans_spectrale).to(device)
    print(f"GAN : latent {args.latent}, largeur {args.largeur}, perte {args.perte}, "
          f"D {'avec' if not args.sans_spectrale else 'sans'} normalisation spectrale")
    print(f"     générateur {nombre_parametres(g)} paramètres, discriminateur {nombre_parametres(d)}")
    print(f"Patches : train {len(xt)} (dataset {manifeste['identifiant']})")
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    out = args.out_dir / "gan"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele": {"type": "gan", "dim_latente": args.latent, "largeur": args.largeur,
                   "parametres_generateur": nombre_parametres(g), "parametres_discriminateur": nombre_parametres(d),
                   "generateur": "même architecture que le décodeur du VAE",
                   "discriminateur": "4 conv. pas 2 (structure de l'encodeur) -> logit"
                                     + (", normalisation spectrale" if not args.sans_spectrale else ", BatchNorm")},
        "parametres": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"]},
        "normalisation": norm,
        "perte": {"bce": "entropie croisée binaire, G non saturant", "hinge": "perte charnière"}[args.perte],
        "optimiseurs": f"Adam (β1 = {args.beta1}, β2 = {args.beta2}), lr G {args.lr_g}, lr D {args.lr_d}",
        "entrainement": "alterné à chaque mini-batch : une mise à jour de D puis une de G",
        "augmentation": "8 symétries du carré (rotations de 90°, miroirs) des cartes réelles",
        "selection": "plus faible écart statistique entre 512 générations et 512 cartes de validation",
        "environnement": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "device": str(device)},
    }
    ecrire_json(out / "config.json", config)

    def sauver(chemin: Path, epoque: int, suivi: dict | None) -> None:
        torch.save({
            "generateur": g.state_dict(), "discriminateur": d.state_dict(),
            "config": {"type": "gan", "dim_latente": args.latent, "largeur": args.largeur,
                       "spectrale": not args.sans_spectrale, "perte": args.perte},
            "normalisation": norm, "epoque": epoque, "suivi": suivi,
            "dataset": manifeste["identifiant"], "date_creation": config["date_creation"],
        }, chemin)

    historique: list[dict] = []
    evolution_epoques: list[int] = []
    evolution: list[np.ndarray] = []
    meilleur = {"ecart": float("inf"), "epoque": 0}
    debut = time.monotonic()
    n = len(xt)
    for epoque in range(1, args.epochs + 1):
        t0 = time.monotonic()
        g.train()
        d.train()
        cumul = {"perte_d": 0.0, "perte_g": 0.0, "d_reel": 0.0, "d_faux": 0.0}
        lots = 0
        permutation = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n - args.batch_size + 1, args.batch_size):  # lots complets (BatchNorm)
            reel = augmenter(xt[permutation[i:i + args.batch_size]], gen)
            z = torch.randn(len(reel), args.latent, device=device, generator=gen_z)
            # 1) Discriminateur : distinguer réel et généré (G figé : detach).
            faux = g(z)
            d_reel, d_faux = d(reel), d(faux.detach())
            perte_d, _ = pertes(d_reel, d_faux, args.perte)
            opt_d.zero_grad()
            perte_d.backward()
            opt_d.step()
            # 2) Générateur : tromper le discriminateur qui vient d'être mis à jour.
            _, perte_g = pertes(d_reel.detach(), d(faux), args.perte)
            opt_g.zero_grad()
            perte_g.backward()
            opt_g.step()
            cumul["perte_d"] += float(perte_d)
            cumul["perte_g"] += float(perte_g)
            cumul["d_reel"] += float(torch.sigmoid(d_reel).mean())
            cumul["d_faux"] += float(torch.sigmoid(d_faux).mean())
            lots += 1

        entree = {"epoque": epoque, **{k: v / lots for k, v in cumul.items()},
                  "duree_s": round(time.monotonic() - t0, 2), "ecart_stat": None, "diversite_relative": None}
        if epoque % args.suivi == 0 or epoque == args.epochs or epoque == 1:
            g_cpu = g.to("cpu")
            gen_val = generer(g_cpu, z_suivi)
            g.to(device)
            ecart = ecart_statistique(denormaliser(gen_val, echelle), denormaliser(ref_val, echelle))
            entree["ecart_stat"] = ecart["moyenne"]
            entree["diversite_relative"] = diversite(gen_val[:256]) / diversite_reelle
            g.eval()
            with torch.no_grad():
                evolution.append(g(z_evolution).squeeze(1).cpu().numpy().astype(np.float16))
            evolution_epoques.append(epoque)
            marque = ""
            if ecart["moyenne"] < meilleur["ecart"]:
                meilleur = {"ecart": ecart["moyenne"], "epoque": epoque,
                            "diversite_relative": entree["diversite_relative"]}
                sauver(out / "best.pt", epoque, {"ecart_stat": ecart, "diversite_relative": entree["diversite_relative"]})
                marque = "  * retenu"
            suivi_txt = (f"  écart stat {ecart['moyenne']:.3f}  diversité {entree['diversite_relative']:.2f}"
                         f"{marque}")
        else:
            suivi_txt = ""
        historique.append(entree)
        print(f"  époque {epoque:3d}/{args.epochs}  perte D {entree['perte_d']:.3f}  G {entree['perte_g']:.3f}  "
              f"D(x) {entree['d_reel']:.2f}  D(G(z)) {entree['d_faux']:.2f}  ({entree['duree_s']:.1f} s){suivi_txt}",
              flush=True)
        ecrire_json(out / "history.json", historique)

    duree = time.monotonic() - debut
    sauver(out / "last.pt", args.epochs, None)
    np.savez_compressed(out / "evolution.npz", epoques=np.array(evolution_epoques), cartes=np.stack(evolution))
    config["resultats"] = {"duree_totale_s": round(duree, 1), "epoques": args.epochs,
                           "epoque_retenue": meilleur["epoque"], "ecart_stat": meilleur["ecart"],
                           "diversite_relative": meilleur.get("diversite_relative"),
                           "diversite_reelle_val": diversite_reelle}
    ecrire_json(out / "config.json", config)
    tracer_courbes(historique, meilleur["epoque"], out / "courbes.png")
    tracer_evolution(evolution_epoques, np.stack(evolution), out / "evolution.png")
    print(f"GAN : {duree:.0f} s, générateur retenu à l'époque {meilleur['epoque']} "
          f"(écart statistique {meilleur['ecart']:.3f}) -> {out}")
    print("Étape suivante :\n  uv run python TP6/generate.py")


if __name__ == "__main__":
    main()
