"""Évaluation du U-Net sur des images inconnues (TP4, étape 4 : F10 à F12).

1. **Patches de test** (zones jamais utilisées pour l'entraînement ni pour la
   sélection du modèle) : IoU, Dice, précision, rappel et matrice de confusion.
2. **Scènes entières** de test et zones « exemple » hors Bretagne : inférence
   par fenêtres glissantes (même algorithme que le site), métriques par zone
   et figure image | référence | prédiction | erreurs dans ``runs/unet/erreurs/``.

Les pixels ignorés (255) n'interviennent dans aucune métrique.

Usage :
    uv run python TP4/evaluate.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from terre_mer import (
    COULEURS,
    DATASET_DIR,
    MER,
    RUNS_DIR,
    TERRE,
    charger_masque,
    charger_scene,
    colorer_masque,
    ecrire_json,
    hex_rgb,
    zones,
)
from train_unet import charger_split
from unet import construire_modele_export, matrice_confusion, metriques, predicteur_torch, predire_image

# Couleurs de la carte d'erreurs.
FAUSSE_TERRE = "#E5484D"  # mer prédite terre
FAUSSE_MER = "#F5A524"  # terre prédite mer


def carte_erreurs(image: np.ndarray, masque: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Image désaturée + pixels mal classés en couleur (F12)."""
    gris = image.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.6
    out = gris.astype(np.uint8)
    out[(masque == MER) & (pred == TERRE)] = hex_rgb(FAUSSE_TERRE)
    out[(masque == TERRE) & (pred == MER)] = hex_rgb(FAUSSE_MER)
    return out


def figure_erreurs(nom: str, image: np.ndarray, masque: np.ndarray, pred: np.ndarray,
                   m: dict, chemin: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    panneaux = [
        (image, "Image originale"),
        (colorer_masque(masque), "Masque de référence (OSM)"),
        (colorer_masque(pred), "Masque prédit (CNN)"),
        (carte_erreurs(image, masque, pred), "Pixels mal classés"),
    ]
    for ax, (img, titre) in zip(axes, panneaux):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(titre)
        ax.axis("off")
    axes[3].legend(
        handles=[Patch(color=FAUSSE_TERRE, label="mer prédite terre"),
                 Patch(color=FAUSSE_MER, label="terre prédite mer")],
        loc="lower right", fontsize=9,
    )
    fig.suptitle(f"{nom} — IoU terre {m['terre']['iou']:.3f}, IoU mer {m['mer']['iou']:.3f}, "
                 f"exactitude {100 * m['exactitude']:.2f} %")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def resume(m: dict) -> str:
    return (f"IoU terre {m['terre']['iou']:.3f}  IoU mer {m['mer']['iou']:.3f}  "
            f"Dice terre {m['terre']['dice']:.3f}  P/R terre {m['terre']['precision']:.3f}/"
            f"{m['terre']['rappel']:.3f}  exactitude {100 * m['exactitude']:.2f} %")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR, help="répertoire du modèle entraîné")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    parser.add_argument("--cpu", action="store_true", help="forcer le CPU")
    args = parser.parse_args()

    chemin = args.run_dir / "best.pt"
    if not chemin.exists():
        raise SystemExit(f"Modèle absent : {chemin}. Lancez `mise run tp4_03_train`.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(chemin, map_location="cpu", weights_only=False)
    modele = construire_modele_export(checkpoint).to(device)
    predire_lot = predicteur_torch(modele, device)
    print(f"Modèle : époque {checkpoint['epoque']}, mIoU val {checkpoint['val']['miou']:.3f} ({device})")

    # 1. Patches de test (F10, F11).
    x_test, y_test = charger_split(args.data_dir, "test")
    cm = np.zeros((2, 2), dtype=np.int64)
    for i in range(0, len(x_test), 16):
        prob = predire_lot(x_test[i:i + 16].numpy().astype(np.float32) / 255.0)[:, 0]
        cm += matrice_confusion(prob > 0.5, y_test[i:i + 16].numpy())
    m_patches = metriques(cm)
    print(f"\nPatches de test ({len(x_test)}) : {resume(m_patches)}")
    print(f"  matrice de confusion [réf mer, réf terre] x [préd mer, préd terre] : {cm.tolist()}")

    # 2. Scènes entières : zones de test + exemples hors Bretagne (F12).
    dossier = args.run_dir / "erreurs"
    dossier.mkdir(parents=True, exist_ok=True)
    par_zone = {}
    print("\nScènes entières (inférence par fenêtres glissantes) :")
    for z in zones("test") + zones("exemple"):
        image, masque = charger_scene(z.id), charger_masque(z.id)
        t0 = time.monotonic()
        prob = predire_image(image, predire_lot)
        duree = time.monotonic() - t0
        pred = (prob > 0.5).astype(np.uint8)
        m = metriques(matrice_confusion(pred, masque))
        incertains = float(((prob > 0.3) & (prob < 0.7)).mean())
        par_zone[z.id] = {"nom": z.nom, "split": z.split, "configuration": z.configuration,
                          "duree_inference_s": round(duree, 2), "part_incertaine_0.3_0.7": incertains, **m}
        figure_erreurs(f"{z.nom} [{z.split}]", image, masque, pred, m, dossier / f"{z.id}.jpg")
        print(f"  {z.id:18s} [{z.split:7s}] {resume(m)}")

    ecrire_json(args.run_dir / "evaluation.json", {
        "modele": {"epoque": checkpoint["epoque"], "dataset": checkpoint["dataset"],
                   "date_creation": checkpoint["date_creation"]},
        "seuil": 0.5,
        "pixels_ignores": "exclus de toutes les métriques (valeur 255)",
        "patches_test": {"nombre": len(x_test), **m_patches},
        "scenes": par_zone,
    })
    print(f"\nRésultats : {args.run_dir / 'evaluation.json'}")
    print(f"Figures d'erreurs : {dossier}")
    print("Étape suivante :\n  uv run python TP4/export_onnx.py")


if __name__ == "__main__":
    main()
