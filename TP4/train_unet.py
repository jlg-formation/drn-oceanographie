"""Entraînement du U-Net de segmentation terre/mer (TP4, étape 3 : F06 à F09).

Lit les patches préparés par ``prepare_dataset.py``, entraîne un U-Net compact
(PyTorch, CPU ou GPU) et conserve le meilleur modèle selon l'IoU moyenne (mIoU)
de validation. Produits dans ``--out-dir`` (défaut ``TP4/runs/unet``) :

  - ``best.pt``      : poids + configuration + normalisation (rechargeable) ;
  - ``history.json`` : pertes, métriques et durée de chaque époque ;
  - ``config.json``  : paramètres, dataset, versions, date (traçabilité NF06) ;
  - ``courbes.png``  : évolution de l'apprentissage.

Usage :
    uv run python TP4/train_unet.py
    uv run python TP4/train_unet.py --epochs 60 --batch-size 8 --lr 5e-4
    uv run python TP4/train_unet.py --rapide        # configuration réduite CPU (NF02)
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
from PIL import Image

from terre_mer import COULEURS, DATASET_DIR, RUNS_DIR, ecrire_json, lire_json
from unet import UNet, matrice_confusion, metriques, perte_segmentation


def charger_split(data_dir: Path, split: str, limite: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge tous les patches d'un ensemble en mémoire (uint8)."""
    manifeste = lire_json(data_dir / "dataset.json")
    fichiers = [p["fichier"] for p in manifeste["patches"] if p["split"] == split]
    if limite:
        fichiers = fichiers[:: max(1, len(fichiers) // limite)][:limite]
    if not fichiers:
        raise SystemExit(f"Aucun patch « {split} » dans {data_dir}. Lancez `mise run tp4_02_dataset`.")
    images = np.stack([np.asarray(Image.open(data_dir / split / "images" / f)) for f in fichiers])
    masques = np.stack([np.asarray(Image.open(data_dir / split / "masks" / f)) for f in fichiers])
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(masques)


def statistiques_normalisation(images: torch.Tensor) -> dict[str, list[float]]:
    """Moyenne et écart-type par canal (images dans [0, 1]) sur l'ensemble d'entraînement."""
    x = images.float() / 255.0
    return {
        "moyenne": [round(float(v), 5) for v in x.mean(dim=(0, 2, 3))],
        "ecart_type": [round(float(v), 5) for v in x.std(dim=(0, 2, 3))],
    }


def augmenter(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmentations cohérentes image/masque, tirées par échantillon.

    - géométriques (rotations de 90°, symétrie) : une côte n'a pas d'orientation ;
    - radiométriques (gain par canal, contraste, luminosité) : robustesse aux
      saisons, aux millésimes et aux images importées dans le site.
    """
    n = x.shape[0]
    tirages = torch.rand(n, 6, generator=gen)
    xs, ys = [], []
    for i in range(n):
        xi, yi = x[i], y[i]
        k = int(tirages[i, 0] * 4)
        xi, yi = torch.rot90(xi, k, (1, 2)), torch.rot90(yi, k, (0, 1))
        if tirages[i, 1] < 0.5:
            xi, yi = xi.flip(2), yi.flip(1)
        xs.append(xi)
        ys.append(yi)
    x, y = torch.stack(xs), torch.stack(ys)

    gain = (1 + (torch.rand(n, 3, 1, 1, generator=gen) - 0.5) * 0.2).to(x.device)  # ±10 % par canal
    contraste = (0.8 + 0.4 * tirages[:, 2]).view(n, 1, 1, 1).to(x.device)
    luminosite = ((tirages[:, 3] - 0.5) * 0.16).view(n, 1, 1, 1).to(x.device)
    moyenne = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - moyenne) * contraste + moyenne) * gain + luminosite
    return x.clamp(0, 1), y


def evaluer(modele: UNet, images: torch.Tensor, masques: torch.Tensor, norm: tuple[torch.Tensor, torch.Tensor],
            device: torch.device, batch_size: int) -> tuple[float, dict]:
    """Perte moyenne et métriques sur un ensemble (pixels ignorés exclus)."""
    modele.eval()
    moyenne, ecart = norm
    cm = np.zeros((2, 2), dtype=np.int64)
    pertes = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i + batch_size].to(device).float() / 255.0
            y = masques[i:i + batch_size].to(device)
            logits = modele((x - moyenne) / ecart)
            pertes.append(perte_segmentation(logits, y).item() * x.shape[0])
            cm += matrice_confusion(logits[:, 0] > 0, y)
    return float(np.sum(pertes) / images.shape[0]), metriques(cm)


def tracer_courbes(historique: list[dict], chemin: Path) -> None:
    epoques = [h["epoque"] for h in historique]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epoques, [h["perte_train"] for h in historique], label="entraînement")
    ax1.plot(epoques, [h["perte_val"] for h in historique], label="validation")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("perte (BCE + Dice)")
    ax1.set_title("Pertes")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epoques, [h["val"]["terre"]["iou"] for h in historique], color=COULEURS["terre"], label="IoU terre")
    ax2.plot(epoques, [h["val"]["mer"]["iou"] for h in historique], color=COULEURS["mer"], label="IoU mer")
    ax2.plot(epoques, [h["val"]["miou"] for h in historique], color="black", ls="--", label="mIoU")
    meilleure = max(historique, key=lambda h: h["val"]["miou"])
    ax2.axvline(meilleure["epoque"], color="grey", ls=":", label=f"meilleur modèle (époque {meilleure['epoque']})")
    ax2.set_xlabel("époque")
    ax2.set_title("Métriques de validation")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40, help="nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="taille des mini-batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage initial (AdamW)")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR, help="répertoire des résultats")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--base", type=int, default=16, help="canaux du premier niveau du U-Net")
    parser.add_argument("--profondeur", type=int, default=4, help="nombre de sous-échantillonnages")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU même si un GPU est disponible")
    parser.add_argument("--rapide", action="store_true",
                        help="configuration réduite pour CPU : 8 canaux, 12 époques, 160 patches")
    args = parser.parse_args()
    if args.rapide:
        args.base, args.epochs = 8, min(args.epochs, 12)
    return args


def main() -> None:
    args = parser_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Périphérique : {device}")

    manifeste = lire_json(args.data_dir / "dataset.json")
    x_train, y_train = charger_split(args.data_dir, "train", limite=160 if args.rapide else None)
    x_val, y_val = charger_split(args.data_dir, "val")
    print(f"Dataset {manifeste['identifiant']} : train={len(x_train)}  val={len(x_val)} patches")

    norm = statistiques_normalisation(x_train)
    moyenne = torch.tensor(norm["moyenne"], device=device).view(1, 3, 1, 1)
    ecart = torch.tensor(norm["ecart_type"], device=device).view(1, 3, 1, 1)
    print(f"Normalisation : moyenne={norm['moyenne']}  écart-type={norm['ecart_type']}")

    modele = UNet(base=args.base, profondeur=args.profondeur).to(device)
    n_params = sum(p.numel() for p in modele.parameters())
    print(f"U-Net : base={args.base}, profondeur={args.profondeur}, {n_params / 1e6:.2f} M paramètres")

    optimiseur = torch.optim.AdamW(modele.parameters(), lr=args.lr, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parametres": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"]},
        "modele": {"architecture": "U-Net compact", "base": args.base, "profondeur": args.profondeur,
                   "parametres": n_params, "entree": "RGB [0,1] normalisée", "sortie": "logit terre"},
        "perte": "BCE + Dice, pixels 255 exclus",
        "selection": "meilleure mIoU de validation",
        "normalisation": norm,
        "environnement": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "device": str(device)},
    }
    ecrire_json(args.out_dir / "config.json", config)

    historique: list[dict] = []
    meilleure_miou = -1.0
    debut = time.monotonic()
    n = x_train.shape[0]
    for epoque in range(1, args.epochs + 1):
        t0 = time.monotonic()
        modele.train()
        permutation = torch.randperm(n, generator=gen)
        pertes = []
        for i in range(0, n, args.batch_size):
            idx = permutation[i:i + args.batch_size]
            x = x_train[idx].to(device).float() / 255.0
            y = y_train[idx].to(device)
            x, y = augmenter(x, y, gen)
            optimiseur.zero_grad()
            perte = perte_segmentation(modele((x - moyenne) / ecart), y)
            perte.backward()
            optimiseur.step()
            pertes.append(perte.item() * len(idx))
        planning.step()

        perte_val, val = evaluer(modele, x_val, y_val, (moyenne, ecart), device, args.batch_size)
        entree = {
            "epoque": epoque,
            "perte_train": float(np.sum(pertes) / n),
            "perte_val": perte_val,
            "val": val,
            "lr": optimiseur.param_groups[0]["lr"],
            "duree_s": round(time.monotonic() - t0, 2),
        }
        historique.append(entree)
        marque = ""
        if val["miou"] > meilleure_miou:
            meilleure_miou = val["miou"]
            marque = "  * meilleur"
            torch.save({
                "state_dict": modele.state_dict(),
                "config": {"base": args.base, "profondeur": args.profondeur},
                "normalisation": norm,
                "epoque": epoque,
                "val": val,
                "dataset": manifeste["identifiant"],
                "date_creation": config["date_creation"],
            }, args.out_dir / "best.pt")
        print(f"  époque {epoque:3d}/{args.epochs}  perte train {entree['perte_train']:.4f}  "
              f"val {perte_val:.4f}  IoU terre {val['terre']['iou']:.3f}  mer {val['mer']['iou']:.3f}  "
              f"mIoU {val['miou']:.3f}  ({entree['duree_s']:.1f} s){marque}", flush=True)
        ecrire_json(args.out_dir / "history.json", historique)

    duree = time.monotonic() - debut
    meilleure = max(historique, key=lambda h: h["val"]["miou"])
    config["resultats"] = {
        "duree_totale_s": round(duree, 1),
        "meilleure_epoque": meilleure["epoque"],
        "val_miou": meilleure["val"]["miou"],
        "val_iou_terre": meilleure["val"]["terre"]["iou"],
        "val_iou_mer": meilleure["val"]["mer"]["iou"],
    }
    ecrire_json(args.out_dir / "config.json", config)
    tracer_courbes(historique, args.out_dir / "courbes.png")
    print(f"Durée totale : {duree:.0f} s. Meilleur modèle : époque {meilleure['epoque']}, "
          f"mIoU val {meilleure['val']['miou']:.3f}")
    print(f"Résultats : {args.out_dir}")
    print("Étape suivante :\n  uv run python TP4/evaluate.py")


if __name__ == "__main__":
    main()
