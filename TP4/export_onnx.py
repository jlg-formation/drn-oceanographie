"""Export du U-Net au format ONNX pour le site web (TP4, étape 5 : F13).

Le graphe exporté contient normalisation + U-Net + sigmoïde : le site n'a qu'à
fournir des pixels RGB divisés par 255. Deux artefacts sont écrits dans
``TP4/model/`` (servi tel quel par Vite) :

  - ``unet_terre_mer.onnx`` : entrée ``image`` (N, 3, 256, 256) float32 [0, 1],
    sortie ``prob_terre`` (N, 1, 256, 256) ;
  - ``metadata.json`` : paramètres d'inférence (fenêtres, seuil), couleurs,
    limites d'import, provenance et métriques du modèle (traçabilité NF06).

Le modèle ONNX est vérifié contre PyTorch sur les patches de test et sur une
scène entière.

Usage :
    uv run python TP4/export_onnx.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from terre_mer import (
    COULEURS,
    DATASET_DIR,
    MODEL_DIR,
    PATCH,
    RUNS_DIR,
    S2_ATTRIBUTION,
    S2_LICENCE,
    charger_scene,
    ecrire_json,
    lire_json,
    zones,
)
from train_unet import charger_split
from unet import PAS_INFERENCE, construire_modele_export, predicteur_torch, predire_image

ONNX_OPSET = 18
ONNX_NAME = "unet_terre_mer.onnx"
# Limites d'import affichées et appliquées par le site (NF08).
LIMITES = {"taille_fichier_mo": 25, "cote_max_px": 4096, "pixels_max": 16_000_000}


def exporter(modele: torch.nn.Module, chemin: Path) -> None:
    exemple = torch.zeros(1, 3, PATCH, PATCH)
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        modele,
        (exemple,),
        str(chemin),
        input_names=["image"],
        output_names=["prob_terre"],
        dynamic_shapes={"image": {0: batch}},
        opset_version=ONNX_OPSET,
        external_data=False,
    )
    print(f"Modèle ONNX écrit : {chemin} ({chemin.stat().st_size / 1e6:.1f} Mo)")


def verifier(modele: torch.nn.Module, chemin: Path, data_dir: Path) -> dict:
    """Compare ONNX Runtime et PyTorch sur les patches de test et une scène entière."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(chemin)))
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])

    def predire_onnx(lot: np.ndarray) -> np.ndarray:
        return session.run(None, {"image": lot.astype(np.float32)})[0]

    predire_torch = predicteur_torch(modele, torch.device("cpu"))

    x_test, _ = charger_split(data_dir, "test")
    lot = x_test[:32].numpy().astype(np.float32) / 255.0
    p_ref, p_onnx = predire_torch(lot), predire_onnx(lot)
    ecart = float(np.abs(p_ref - p_onnx).max())
    accord = float(((p_ref > 0.5) == (p_onnx > 0.5)).mean())
    print(f"Patches de test : écart max de probabilité {ecart:.2e}, accord des classes {100 * accord:.4f} %")

    z = zones("test")[0]
    image = charger_scene(z.id)
    s_ref = predire_image(image, predire_torch)
    s_onnx = predire_image(image, predire_onnx)
    accord_scene = float(((s_ref > 0.5) == (s_onnx > 0.5)).mean())
    print(f"Scène « {z.id} » : accord des classes {100 * accord_scene:.4f} %")
    if ecart > 1e-3 or accord < 0.999:
        print("  ATTENTION : écart ONNX / PyTorch supérieur au seuil attendu.")
    return {"ecart_max_probabilite": ecart, "accord_classes_patches": accord,
            f"accord_classes_scene_{z.id}": accord_scene}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR, help="répertoire du modèle entraîné")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    args = parser.parse_args()

    chemin_pt = args.run_dir / "best.pt"
    if not chemin_pt.exists():
        raise SystemExit(f"Modèle absent : {chemin_pt}. Lancez `mise run tp4_03_train`.")
    checkpoint = torch.load(chemin_pt, map_location="cpu", weights_only=False)
    modele = construire_modele_export(checkpoint)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    chemin_onnx = MODEL_DIR / ONNX_NAME
    exporter(modele, chemin_onnx)
    verification = verifier(modele, chemin_onnx, args.data_dir)

    evaluation = args.run_dir / "evaluation.json"
    test = lire_json(evaluation)["patches_test"] if evaluation.exists() else None
    config = lire_json(args.run_dir / "config.json")

    ecrire_json(MODEL_DIR / "metadata.json", {
        "modele": {
            "fichier": ONNX_NAME,
            "architecture": f"U-Net compact (base {checkpoint['config']['base']}, "
                            f"profondeur {checkpoint['config']['profondeur']})",
            "parametres": config["modele"]["parametres"],
            "onnx_opset": ONNX_OPSET,
            "entree": {"nom": "image", "forme": ["N", 3, PATCH, PATCH], "type": "float32",
                       "description": "RGB divisé par 255 (normalisation incluse dans le graphe)"},
            "sortie": {"nom": "prob_terre", "forme": ["N", 1, PATCH, PATCH],
                       "description": "probabilité qu'un pixel soit de la terre"},
        },
        "inference": {
            "patch_px": PATCH,
            "pas_px": PAS_INFERENCE,
            "rampe_px": (PATCH - PAS_INFERENCE) // 2,
            "seuil_terre": 0.5,
            "incertitude": {"min": 0.3, "max": 0.7},
            "completion": "miroir si l'image est plus petite que le patch",
        },
        "classes": {"0": "mer", "1": "terre"},
        "couleurs": COULEURS,
        "limites": LIMITES,
        "entrainement": {
            "date_creation": checkpoint["date_creation"],
            "date_export": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": checkpoint["dataset"],
            "epoque": checkpoint["epoque"],
            "parametres": config["parametres"],
            "validation": {"miou": checkpoint["val"]["miou"], "iou_terre": checkpoint["val"]["terre"]["iou"],
                           "iou_mer": checkpoint["val"]["mer"]["iou"]},
            "test": None if test is None else {
                "miou": test["miou"], "iou_terre": test["terre"]["iou"], "iou_mer": test["mer"]["iou"],
                "exactitude": test["exactitude"]},
        },
        "verification_onnx": verification,
        "sources": {
            "images": f"{S2_ATTRIBUTION} — {S2_LICENCE}",
            "masques": "Polygones terrestres © OpenStreetMap contributors (ODbL) — osmdata.openstreetmap.de",
        },
    })
    print(f"Métadonnées écrites : {MODEL_DIR / 'metadata.json'}")
    print("Étape suivante :\n  bun run dev:tp4")


if __name__ == "__main__":
    main()
