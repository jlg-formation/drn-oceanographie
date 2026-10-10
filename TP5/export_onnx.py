"""Export ONNX des trois réseaux et des données du site MaréeLab (TP5, étape 5 : F11, F12).

Chaque graphe exporté contient **normalisation + réseau récurrent +
dénormalisation** : entrée ``hauteurs`` (N, 48) en mètres, sortie ``prevision``
(N, 24) en mètres. L'export utilise les opérateurs ONNX natifs ``RNN``, ``GRU``
et ``LSTM`` (lisibles dans Netron).

Artefacts écrits dans ``TP5/model/`` (servi tel quel par Vite) :

  - ``rnn.onnx``, ``gru.onnx``, ``lstm.onnx`` ;
  - ``serie.json``    : série horaire complète (``null`` = lacune) et bornes des ensembles ;
  - ``metadata.json`` : paramètres d'inférence, styles des courbes, métriques de
    test, historiques d'entraînement, périodes d'exemple, provenance (NF05) et
    un exemple de contrôle que le site recalcule au chargement.

La cohérence PyTorch / ONNX Runtime est vérifiée sur toutes les fenêtres de test.

Usage :
    uv run python TP5/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from maree import (
    ATTRIBUTION,
    DATASET_DIR,
    DESCRIPTIONS_REFERENCES,
    HORIZONS_EVAL,
    MODEL_DIR,
    MODELES,
    PROVENANCE_JSON,
    REFERENCES,
    RUNS_DIR,
    STATION_NOM,
    STYLES,
    charger_serie,
    ecrire_json,
    fenetres,
    iso,
    lire_json,
)
from modeles import ARCHITECTURES, construire_modele_export, nombre_parametres, predire

ONNX_OPSET = 17
TOLERANCE_M = 1e-4  # 0,1 mm


def exporter(modele: torch.nn.Module, entree: int, chemin: Path) -> None:
    # Exporteur TorchScript : il produit les opérateurs natifs RNN / GRU / LSTM,
    # plus lisibles pour le TP que la boucle déroulée de l'exporteur dynamo.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            modele, (torch.zeros(1, entree),), str(chemin),
            input_names=["hauteurs"], output_names=["prevision"],
            dynamic_axes={"hauteurs": {0: "N"}, "prevision": {0: "N"}},
            opset_version=ONNX_OPSET, dynamo=False,
        )


def verifier(modele: torch.nn.Module, chemin: Path, x: np.ndarray) -> dict:
    """Compare ONNX Runtime et PyTorch sur toutes les fenêtres de test (F12)."""
    import onnx
    import onnxruntime as ort

    graphe = onnx.load(str(chemin))
    onnx.checker.check_model(graphe)
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    p_onnx = session.run(None, {"hauteurs": x})[0]
    p_torch = predire(modele, x)
    ecart = float(np.abs(p_onnx - p_torch).max())
    operateurs = sorted({n.op_type for n in graphe.graph.node} & {"RNN", "GRU", "LSTM"})
    return {"fenetres": int(len(x)), "ecart_max_m": ecart, "conforme": ecart < TOLERANCE_M,
            "operateurs_recurrents": operateurs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles entraînés")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    args = parser.parse_args()

    serie = charger_serie(args.data_dir)
    manifeste = serie["manifeste"]
    p = manifeste["parametres"]
    x_test, _ = fenetres(serie["hauteur"], serie["ancres"]["test"], p["entree_h"], p["horizon_h"])
    evaluation_json = args.run_dir / "evaluation.json"
    evaluation = lire_json(evaluation_json) if evaluation_json.exists() else None
    if evaluation is None:
        print("ATTENTION : evaluation.json absent, le site n'aura ni métriques de test ni exemples. "
              "Lancez `mise run tp5_04_evaluate`.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Exemple de contrôle : première fenêtre de test. Le site le recalcule au chargement.
    controle_x = x_test[:1]
    modeles_meta: dict[str, dict] = {}
    controle: dict = {"ancre": iso(serie["temps"][serie["ancres"]["test"][0]]),
                      "entree": [round(float(v), 4) for v in controle_x[0]], "attendu": {}}
    for m in MODELES:
        chemin_pt = args.run_dir / m / "best.pt"
        if not chemin_pt.exists():
            print(f"  {m} : checkpoint absent ({chemin_pt}), non exporté.")
            continue
        checkpoint = torch.load(chemin_pt, map_location="cpu", weights_only=False)
        modele = construire_modele_export(checkpoint)
        chemin_onnx = MODEL_DIR / f"{m}.onnx"
        exporter(modele, p["entree_h"], chemin_onnx)
        verification = verifier(modele, chemin_onnx, x_test)
        etat = "OK" if verification["conforme"] else "ÉCART SUPÉRIEUR AU SEUIL"
        print(f"  {STYLES[m]['nom']:5s} -> {chemin_onnx.name} ({chemin_onnx.stat().st_size / 1e3:.0f} ko), "
              f"opérateur {verification['operateurs_recurrents']}, écart max ONNX/PyTorch "
              f"{verification['ecart_max_m']:.2e} m sur {verification['fenetres']} fenêtres : {etat}")
        controle["attendu"][m] = [round(float(v), 5) for v in predire(modele, controle_x)[0]]

        config = lire_json(args.run_dir / m / "config.json")
        historique = lire_json(args.run_dir / m / "history.json")
        modeles_meta[m] = {
            "fichier": chemin_onnx.name,
            "architecture": ARCHITECTURES[m],
            "cachee": checkpoint["config"]["cachee"],
            "couches": checkpoint["config"]["couches"],
            "parametres": nombre_parametres(modele.modele),
            "epoque": checkpoint["epoque"],
            "epoques": config["resultats"]["epoques"],
            "duree_entrainement_s": config["resultats"]["duree_totale_s"],
            "validation": checkpoint["val"],
            "historique": {
                "perte_train": [round(h["perte_train"], 6) for h in historique],
                "perte_val": [round(h["perte_val"], 6) for h in historique],
                "val_mae_m": [round(h["val_mae_m"], 5) for h in historique],
            },
            "verification_onnx": verification,
            "dataset": checkpoint["dataset"],
            "date_creation": checkpoint["date_creation"],
        }
    if not modeles_meta:
        raise SystemExit("Aucun modèle entraîné. Lancez d'abord `mise run tp5_03_train`.")

    # --- Série complète pour le site (aucune valeur comblée : null = lacune) ---
    temps, h = serie["temps"], serie["hauteur"]
    bornes = {}
    for s, r in manifeste["repartition"].items():
        a = int((np.datetime64(r["debut"].rstrip("Z"), "h") - temps[0]).astype(int))
        bornes[s] = {"debut": a, "fin": a + r["heures"], "debut_iso": r["debut"], "fin_iso": r["fin"]}
    # JSON compact : ~44 000 valeurs, une par ligne doublerait la taille du fichier.
    (MODEL_DIR / "serie.json").write_text(json.dumps({
        "station": STATION_NOM,
        "unite": "m",
        "reference": "zéro hydrographique",
        "fuseau": "UTC",
        "debut": iso(temps[0]),
        "pas_h": 1,
        "ensembles": bornes,
        "valeurs": [None if not np.isfinite(v) else round(float(v), 3) for v in h],
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    provenance = lire_json(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else {}
    metriques = evaluation["metriques"] if evaluation else {}
    ecrire_json(MODEL_DIR / "metadata.json", {
        "application": "MaréeLab",
        "avertissement": "Prévisions expérimentales produites dans un cadre pédagogique. Ce n'est pas un "
                         "outil officiel de prédiction des marées du SHOM : pour la navigation, utilisez les "
                         "prédictions officielles (maree.shom.fr).",
        "station": {"nom": STATION_NOM, **{k: provenance.get("station", {}).get(k)
                                           for k in ("id_shom", "latitude", "longitude", "reseau")}},
        "unite": "m",
        "fuseau": "UTC",
        "reference_verticale": provenance.get("reference_verticale"),
        "inference": {
            "entree": "hauteurs", "sortie": "prevision",
            "entree_h": p["entree_h"], "horizon_h": p["horizon_h"],
            "description": "entrée : 48 hauteurs horaires en mètres (H−47 … H) ; sortie : 24 hauteurs "
                           "(H+1 … H+24). Normalisation incluse dans le graphe.",
            "onnx_opset": ONNX_OPSET,
        },
        "normalisation": manifeste["normalisation"],
        "modeles": modeles_meta,
        "references": {r: {"description": DESCRIPTIONS_REFERENCES[r]} for r in REFERENCES},
        "styles": STYLES,
        "horizons_eval": HORIZONS_EVAL,
        "test": None if evaluation is None else {
            **evaluation["test"],
            "metriques": {n: {"global": m["global"], "horizons": m["horizons"], "mae_par_pas": m["par_pas"]["mae"]}
                          for n, m in metriques.items()},
            "classement_mae": evaluation["classement_mae"],
            "meilleure_reference": evaluation["meilleure_reference"],
        },
        "exemples": [] if evaluation is None else [
            {k: ex[k] for k in ("id", "titre", "description", "ancre", "marnage_m", "niveau_moyen_m")}
            for ex in evaluation["exemples"]
        ],
        "controle": controle,
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"],
                    "statistiques": manifeste["statistiques"]},
        "sources": {"observations": ATTRIBUTION, "portail": provenance.get("source", {}).get("portail"),
                    "qualite": provenance.get("qualite"),
                    "date_telechargement": provenance.get("date_telechargement")},
        "date_export": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"Site : {MODEL_DIR / 'metadata.json'}, {MODEL_DIR / 'serie.json'}")
    print("Étape suivante :\n  bun run dev:tp5")


if __name__ == "__main__":
    main()
