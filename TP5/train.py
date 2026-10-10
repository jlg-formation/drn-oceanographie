"""Entraînement des réseaux récurrents RNN, GRU et LSTM (TP5, étape 3 : F06 à F11).

Les trois modèles sont entraînés sur **les mêmes fenêtres**, avec **la même
graine**, **le même ordre des mini-batches** et **les mêmes hyperparamètres** :
seule la cellule récurrente change. Pour chaque modèle, ``--out-dir/<modele>/``
contient :

  - ``best.pt``      : poids + configuration + normalisation (rechargeable, F11) ;
  - ``history.json`` : pertes d'entraînement et de validation, MAE/RMSE de
    validation en mètres, durée de chaque époque (F10) ;
  - ``config.json``  : paramètres, dataset, versions, durée totale (NF05) ;
  - ``courbes.png``  : évolution de l'apprentissage.

``--out-dir/comparaison_pertes.png`` superpose les courbes des modèles entraînés.

Usage :
    uv run python TP5/train.py                      # les trois modèles
    uv run python TP5/train.py --modele lstm --epochs 50 --cachee 128
    uv run python TP5/train.py --rapide --cpu       # configuration réduite CPU (NF02)
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

from maree import DATASET_DIR, MODELES, RUNS_DIR, STYLES, charger_serie, ecrire_json, fenetres, lire_json
from modeles import ARCHITECTURES, Prevision, nombre_parametres


def donnees(serie: dict, split: str, sous_echantillon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    m = serie["manifeste"]["parametres"]
    ancres = serie["ancres"][split][::sous_echantillon]
    return fenetres(serie["hauteur"], ancres, m["entree_h"], m["horizon_h"])


def evaluer(modele: Prevision, x: torch.Tensor, y: torch.Tensor, ecart_type: float, batch_size: int) -> dict:
    """Perte (MSE normalisée), MAE et RMSE en mètres sur un ensemble."""
    modele.eval()
    se, ae = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            erreur = modele(x[i:i + batch_size]) - y[i:i + batch_size]
            se += float((erreur**2).sum())
            ae += float(erreur.abs().sum())
    n = y.numel()
    return {"perte": se / n, "mae_m": ae / n * ecart_type, "rmse_m": (se / n) ** 0.5 * ecart_type}


def tracer_courbes(historique: list[dict], nom: str, chemin: Path) -> None:
    epoques = [h["epoque"] for h in historique]
    meilleure = min(historique, key=lambda h: h["perte_val"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epoques, [h["perte_train"] for h in historique], label="entraînement")
    ax1.plot(epoques, [h["perte_val"] for h in historique], label="validation")
    ax1.set_yscale("log")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("perte MSE (hauteurs normalisées)")
    ax1.set_title(f"{nom} — pertes")
    ax2.plot(epoques, [100 * h["val_mae_m"] for h in historique], label="MAE validation")
    ax2.plot(epoques, [100 * h["val_rmse_m"] for h in historique], label="RMSE validation")
    ax2.set_xlabel("époque")
    ax2.set_ylabel("erreur (cm)")
    ax2.set_title(f"{nom} — erreurs de validation sur 24 h")
    for ax in (ax1, ax2):
        ax.axvline(meilleure["epoque"], color="grey", ls=":", label=f"meilleur (époque {meilleure['epoque']})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def tracer_comparaison(out_dir: Path) -> Path | None:
    """Courbes de validation de tous les modèles disponibles sur un même graphique (F10)."""
    historiques = {m: lire_json(out_dir / m / "history.json") for m in MODELES
                   if (out_dir / m / "history.json").exists()}
    if not historiques:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, h in historiques.items():
        s = STYLES[m]
        e = [x["epoque"] for x in h]
        tirets = (0, s["tirets"]) if s["tirets"] else None
        ax1.plot(e, [x["perte_train"] for x in h], color=s["couleur"], alpha=0.35, lw=1)
        ax1.plot(e, [x["perte_val"] for x in h], color=s["couleur"], lw=2, linestyle=tirets or "-",
                 label=f"{s['nom']} (validation)")
        ax2.plot(e, [100 * x["val_mae_m"] for x in h], color=s["couleur"], lw=2, linestyle=tirets or "-",
                 label=s["nom"])
    ax1.set_yscale("log")
    ax1.set_title("Pertes (trait fin : entraînement, épais : validation)")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("MSE normalisée")
    ax2.set_title("MAE de validation sur 24 h")
    ax2.set_xlabel("époque")
    ax2.set_ylabel("cm")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    chemin = out_dir / "comparaison_pertes.png"
    fig.savefig(chemin, dpi=100)
    plt.close(fig)
    return chemin


def entrainer(type_cellule: str, args: argparse.Namespace, serie: dict, device: torch.device) -> dict:
    manifeste = serie["manifeste"]
    p = manifeste["parametres"]
    norm = manifeste["normalisation"]
    mu, sigma = norm["moyenne_m"], norm["ecart_type_m"]

    x_train, y_train = donnees(serie, "train", 3 if args.rapide else 1)
    x_val, y_val = donnees(serie, "val")
    vers = lambda a: torch.from_numpy((a - mu) / sigma).float().to(device)  # noqa: E731
    xt, yt, xv, yv = vers(x_train), vers(y_train), vers(x_val), vers(y_val)

    # Même graine pour chaque modèle : mêmes mini-batches, initialisation reproductible.
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    modele = Prevision(type_cellule, cachee=args.cachee, couches=args.couches, horizon=p["horizon_h"]).to(device)
    n_params = nombre_parametres(modele)
    nom = STYLES[type_cellule]["nom"]
    print(f"\n=== {nom} : {ARCHITECTURES[type_cellule]} — état caché {args.cachee}, "
          f"{args.couches} couche(s), {n_params} paramètres")
    print(f"    fenêtres : train={len(xt)}  val={len(xv)}  (entrée {p['entree_h']} h -> {p['horizon_h']} h)")

    optimiseur = torch.optim.AdamW(modele.parameters(), lr=args.lr, weight_decay=1e-5)
    planning = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=args.epochs)
    perte_fn = torch.nn.MSELoss()

    out = args.out_dir / type_cellule
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele": {"type": type_cellule, "architecture": ARCHITECTURES[type_cellule], "cachee": args.cachee,
                   "couches": args.couches, "entree_h": p["entree_h"], "horizon_h": p["horizon_h"],
                   "parametres": n_params, "sortie": "prévision directe des 24 valeurs (tête linéaire)"},
        "parametres": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"],
                    "fenetres_train_utilisees": len(xt)},
        "normalisation": norm,
        "perte": "MSE sur hauteurs normalisées",
        "optimiseur": "AdamW, décroissance cosinus du taux, écrêtage de la norme du gradient à 1",
        "selection": "plus faible perte de validation",
        "environnement": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "device": str(device)},
    }
    ecrire_json(out / "config.json", config)

    historique: list[dict] = []
    meilleure = float("inf")
    debut = time.monotonic()
    n = xt.shape[0]
    for epoque in range(1, args.epochs + 1):
        t0 = time.monotonic()
        modele.train()
        permutation = torch.randperm(n, generator=gen).to(device)
        somme = 0.0
        for i in range(0, n, args.batch_size):
            idx = permutation[i:i + args.batch_size]
            optimiseur.zero_grad()
            perte = perte_fn(modele(xt[idx]), yt[idx])
            perte.backward()
            # Écrêtage : protège surtout le RNN simple de l'explosion du gradient.
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optimiseur.step()
            somme += perte.item() * len(idx)
        planning.step()

        val = evaluer(modele, xv, yv, sigma, args.batch_size * 4)
        entree = {"epoque": epoque, "perte_train": somme / n, "perte_val": val["perte"],
                  "val_mae_m": val["mae_m"], "val_rmse_m": val["rmse_m"],
                  "lr": optimiseur.param_groups[0]["lr"], "duree_s": round(time.monotonic() - t0, 2)}
        historique.append(entree)
        marque = ""
        if val["perte"] < meilleure:
            meilleure = val["perte"]
            marque = "  * meilleur"
            torch.save({
                "state_dict": modele.state_dict(),
                "config": {"type": type_cellule, "cachee": args.cachee, "couches": args.couches,
                           "entree": p["entree_h"], "horizon": p["horizon_h"]},
                "normalisation": norm,
                "epoque": epoque,
                "val": {"perte": val["perte"], "mae_m": val["mae_m"], "rmse_m": val["rmse_m"]},
                "dataset": manifeste["identifiant"],
                "date_creation": config["date_creation"],
            }, out / "best.pt")
        print(f"  époque {epoque:3d}/{args.epochs}  perte train {entree['perte_train']:.5f}  "
              f"val {val['perte']:.5f}  MAE val {100 * val['mae_m']:5.1f} cm  "
              f"RMSE val {100 * val['rmse_m']:5.1f} cm  ({entree['duree_s']:.1f} s){marque}", flush=True)
        ecrire_json(out / "history.json", historique)

    duree = time.monotonic() - debut
    best = min(historique, key=lambda h: h["perte_val"])
    config["resultats"] = {"duree_totale_s": round(duree, 1), "epoques": args.epochs,
                           "meilleure_epoque": best["epoque"], "val_perte": best["perte_val"],
                           "val_mae_m": best["val_mae_m"], "val_rmse_m": best["val_rmse_m"]}
    ecrire_json(out / "config.json", config)
    tracer_courbes(historique, nom, out / "courbes.png")
    print(f"  {nom} : {duree:.0f} s, meilleure époque {best['epoque']}, "
          f"MAE val {100 * best['val_mae_m']:.1f} cm, RMSE val {100 * best['val_rmse_m']:.1f} cm -> {out}")
    return {"modele": type_cellule, **config["resultats"], "parametres": n_params}


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modele", choices=[*MODELES, "tous"], default="tous", help="modèle(s) à entraîner")
    parser.add_argument("--epochs", type=int, default=60, help="nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=256, help="taille des mini-batches")
    parser.add_argument("--lr", type=float, default=2e-3, help="taux d'apprentissage initial (AdamW)")
    parser.add_argument("--cachee", type=int, default=64, help="dimension de l'état caché")
    parser.add_argument("--couches", type=int, default=2, help="nombre de couches récurrentes empilées")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles et résultats")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU même si un GPU est disponible")
    parser.add_argument("--rapide", action="store_true",
                        help="configuration réduite pour CPU : état caché 32, 1 couche, 8 époques, 1 fenêtre sur 3")
    args = parser.parse_args()
    if args.rapide:
        args.cachee, args.couches, args.epochs = 32, 1, min(args.epochs, 8)
    return args


def main() -> None:
    args = parser_arguments()
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Périphérique : {device}")
    serie = charger_serie(args.data_dir)
    print(f"Dataset {serie['manifeste']['identifiant']}, normalisation {serie['manifeste']['normalisation']}")

    resultats = [entrainer(m, args, serie, device) for m in (MODELES if args.modele == "tous" else [args.modele])]
    comparaison = tracer_comparaison(args.out_dir)

    print("\nRésumé (validation, meilleure époque) :")
    for r in resultats:
        print(f"  {STYLES[r['modele']]['nom']:5s}  {r['parametres']:7d} param.  MAE {100 * r['val_mae_m']:5.1f} cm  "
              f"RMSE {100 * r['val_rmse_m']:5.1f} cm  époque {r['meilleure_epoque']:3d}  {r['duree_totale_s']:6.0f} s")
    if comparaison:
        print(f"Courbes comparées : {comparaison}")
    print("Étape suivante :\n  uv run python TP5/evaluate.py")


if __name__ == "__main__":
    main()
