"""Export ONNX des modèles et des données du site Ocean Terrain Lab (TP6, étape 7 : F24 à F26).

Graphes exportés dans ``TP6/model/`` (servi tel quel par Vite) :

  - ``vae_encodeur.onnx``   : ``elevation`` (N, 64, 64) en m -> ``mu``, ``logvar`` (N, d).
    Le centrage (moins la moyenne du patch), la division par l'échelle et
    l'écrêtage sont **dans le graphe** (F24) ;
  - ``vae_decodeur.onnx``   : ``z`` (N, d) -> ``relief`` (N, 64, 64) en m (relief relatif) ;
  - ``gan_generateur.onnx`` : ``z`` (N, d) -> ``relief`` (N, 64, 64) en m (F25).
    Le discriminateur n'est pas exporté : il ne sert pas à générer.

Chaque graphe est vérifié (``onnx.checker``) puis comparé à PyTorch sur les
mêmes entrées : patches de test pour l'encodeur, 256 vecteurs latents pour les
générateurs (F26). Données du site :

  - ``reels.bin``     : galerie de patches réels de test, élévations Int16 (m) ;
  - ``evolution.bin`` : générations du GAN à z fixé au fil des époques (Uint8) ;
  - ``metadata.json`` : paramètres d'inférence, normalisation, historiques
    d'entraînement, résultats de l'évaluation, provenance, et un **exemple de
    contrôle** que le site recalcule au chargement.

Usage :
    uv run python TP6/export_onnx.py
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from modeles import (
    EncodeurExport,
    GenerateurExport,
    construire_generateur,
    construire_vae,
    nombre_parametres,
)
from terrain import (
    ATTRIBUTION,
    DATASET_DIR,
    MODEL_DIR,
    PALETTE,
    PATCH,
    PROVENANCE_JSON,
    RUNS_DIR,
    STYLES,
    charger_dataset,
    ecrire_json,
    lire_json,
    lot_latent,
    vecteur_latent,
)

ONNX_OPSET = 17
TOLERANCE_M = 0.05  # 5 cm sur des reliefs de plusieurs centaines de mètres
N_GALERIE = 64  # patches réels proposés dans le site (F33)
N_EVOLUTION = 8  # cartes du GAN suivies au fil des époques
MAX_INSTANTANES = 40


def exporter(modele: torch.nn.Module, entree: torch.Tensor, nom_entree: str, noms_sorties: list[str],
             chemin: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            modele, (entree,), str(chemin), input_names=[nom_entree], output_names=noms_sorties,
            dynamic_axes={nom_entree: {0: "N"}, **{s: {0: "N"} for s in noms_sorties}},
            opset_version=ONNX_OPSET, dynamo=False,
        )


def verifier(modele: torch.nn.Module, chemin: Path, nom_entree: str, x: np.ndarray) -> dict:
    """ONNX Runtime contre PyTorch sur les mêmes entrées (F26)."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(chemin)))
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    sorties_onnx = session.run(None, {nom_entree: x})
    with torch.no_grad():
        sorties_torch = modele(torch.from_numpy(x))
    if isinstance(sorties_torch, torch.Tensor):
        sorties_torch = (sorties_torch,)
    ecart = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(sorties_onnx, sorties_torch))
    return {"entrees": int(len(x)), "ecart_max": ecart, "conforme": ecart < TOLERANCE_M,
            "taille_octets": chemin.stat().st_size}


def choisir_galerie(index: list[dict], n: int) -> list[int]:
    """Patches de test variés : répartis sur les régions et, dans chaque région, sur les écarts-types."""
    regions = sorted({p["region"] for p in index})
    par_region = max(1, n // len(regions))
    choix: list[int] = []
    for r in regions:
        ids = sorted((i for i, p in enumerate(index) if p["region"] == r), key=lambda i: index[i]["ecart_type_m"])
        if ids:
            choix += [ids[int(k)] for k in np.unique(np.linspace(0, len(ids) - 1, par_region).round())]
    reste = [i for i in np.argsort([-p["ecart_type_m"] for p in index]) if i not in choix]
    return (choix + [int(i) for i in reste])[:n]


def historique_vae(h: list[dict]) -> dict:
    return {k: [round(e[k], 4) for e in h] for k in
            ("train_perte", "val_perte", "train_reconstruction", "val_reconstruction", "train_kl", "val_kl",
             "val_rmse_m")}


def historique_gan(h: list[dict]) -> dict:
    res = {k: [round(e[k], 4) for e in h] for k in ("perte_d", "perte_g", "d_reel", "d_faux")}
    suivi = [e for e in h if e.get("ecart_stat") is not None]
    res["suivi_epoques"] = [e["epoque"] for e in suivi]
    res["ecart_stat"] = [round(e["ecart_stat"], 4) for e in suivi]
    res["diversite_relative"] = [round(e["diversite_relative"], 4) for e in suivi]
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles entraînés")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    args = parser.parse_args()

    ds = charger_dataset(args.data_dir)
    manifeste = ds["manifeste"]
    echelle = manifeste["normalisation"]["echelle_m"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test_m = ds["test"]["m"]
    index_test = ds["index"]["test"]
    evaluation = lire_json(args.run_dir / "evaluation.json") if (args.run_dir / "evaluation.json").exists() else None
    if evaluation is None:
        print("ATTENTION : evaluation.json absent : le site n'aura pas les résultats comparatifs. "
              "Lancez `mise run tp6_06_evaluate`.")

    galerie = choisir_galerie(index_test, N_GALERIE)
    reels = np.round(test_m[galerie]).astype("<i2")
    (MODEL_DIR / "reels.bin").write_bytes(reels.tobytes())
    controle: dict = {}
    modeles_meta: dict = {}

    if (args.run_dir / "vae" / "best.pt").exists():
        ck = torch.load(args.run_dir / "vae" / "best.pt", map_location="cpu", weights_only=False)
        vae = construire_vae(ck)
        dim = ck["config"]["dim_latente"]
        enc, dec = EncodeurExport(vae, echelle).eval(), GenerateurExport(vae.decodeur, echelle).eval()
        exporter(enc, torch.from_numpy(test_m[:2]), "elevation", ["mu", "logvar"], MODEL_DIR / "vae_encodeur.onnx")
        exporter(dec, torch.zeros(2, dim), "z", ["relief"], MODEL_DIR / "vae_decodeur.onnx")
        v_enc = verifier(enc, MODEL_DIR / "vae_encodeur.onnx", "elevation", test_m)
        v_dec = verifier(dec, MODEL_DIR / "vae_decodeur.onnx", "z", lot_latent(0, 256, dim))
        config = lire_json(args.run_dir / "vae" / "config.json")
        z0 = vecteur_latent(0, dim)
        with torch.no_grad():
            relief0 = dec(torch.from_numpy(z0[None]))[0].numpy()
            mu0, _ = enc(torch.from_numpy(test_m[galerie[:1]].astype(np.float32)))
        controle["vae"] = {"graine": 0, "z": [round(float(v), 6) for v in z0],
                           "relief_echantillon": [round(float(v), 3) for v in relief0.reshape(-1)[::97]],
                           "mu_reel_0": [round(float(v), 5) for v in mu0[0]]}
        modeles_meta["vae"] = {
            "fichiers": {"encodeur": "vae_encodeur.onnx", "decodeur": "vae_decodeur.onnx"},
            "dim_latente": dim, "beta": ck["config"]["beta"], "largeur": ck["config"]["largeur"],
            "parametres": {"encodeur": nombre_parametres(vae.extracteur) + nombre_parametres(vae.mu)
                           + nombre_parametres(vae.logvar), "decodeur": nombre_parametres(vae.decodeur)},
            "epoque": ck["epoque"], "epoques": config.get("resultats", {}).get("epoques", ck["epoque"]),
            "duree_entrainement_s": config.get("resultats", {}).get("duree_totale_s"),
            "validation": {k: ck["val"][k] for k in ("perte", "reconstruction", "kl", "rmse_m")},
            "historique": historique_vae(lire_json(args.run_dir / "vae" / "history.json")),
            "verification_onnx": {"encodeur": v_enc, "decodeur": v_dec},
            "dataset": ck["dataset"], "date_creation": ck["date_creation"],
        }
        for nom, v in (("encodeur", v_enc), ("décodeur", v_dec)):
            print(f"  VAE {nom:8s} : {v['taille_octets'] / 1e3:.0f} ko, écart max ONNX/PyTorch {v['ecart_max']:.2e} "
                  f"sur {v['entrees']} entrées : {'OK' if v['conforme'] else 'ÉCART SUPÉRIEUR AU SEUIL'}")

    if (args.run_dir / "gan" / "best.pt").exists():
        ck = torch.load(args.run_dir / "gan" / "best.pt", map_location="cpu", weights_only=False)
        gen = GenerateurExport(construire_generateur(ck), echelle).eval()
        dim = ck["config"]["dim_latente"]
        exporter(gen, torch.zeros(2, dim), "z", ["relief"], MODEL_DIR / "gan_generateur.onnx")
        v_gen = verifier(gen, MODEL_DIR / "gan_generateur.onnx", "z", lot_latent(0, 256, dim))
        config = lire_json(args.run_dir / "gan" / "config.json")
        z0 = vecteur_latent(0, dim)
        with torch.no_grad():
            relief0 = gen(torch.from_numpy(z0[None]))[0].numpy()
        controle["gan"] = {"graine": 0, "z": [round(float(v), 6) for v in z0],
                           "relief_echantillon": [round(float(v), 3) for v in relief0.reshape(-1)[::97]]}
        # Évolution au fil des époques (F15) : Uint8, 0..255 <-> -1..1.
        evolution = None
        if (args.run_dir / "gan" / "evolution.npz").exists():
            ev = np.load(args.run_dir / "gan" / "evolution.npz")
            garde = np.unique(np.linspace(0, len(ev["epoques"]) - 1,
                                          min(MAX_INSTANTANES, len(ev["epoques"]))).round().astype(int))
            cartes = ev["cartes"][garde, :N_EVOLUTION].astype(np.float32)
            (MODEL_DIR / "evolution.bin").write_bytes(
                np.clip(np.round((cartes + 1) * 127.5), 0, 255).astype(np.uint8).tobytes())
            evolution = {"fichier": "evolution.bin", "epoques": [int(e) for e in ev["epoques"][garde]],
                         "cartes": N_EVOLUTION, "codage": "uint8 : z = v / 127,5 − 1 ; relief = z × echelle_m",
                         "graines": list(range(N_EVOLUTION))}
        modeles_meta["gan"] = {
            "fichiers": {"generateur": "gan_generateur.onnx"},
            "dim_latente": dim, "largeur": ck["config"]["largeur"], "perte": ck["config"]["perte"],
            "spectrale": ck["config"]["spectrale"],
            "parametres": {"generateur": nombre_parametres(gen.decodeur),
                           "discriminateur": config["modele"]["parametres_discriminateur"]},
            "epoque": ck["epoque"], "epoques": config.get("resultats", {}).get("epoques", ck["epoque"]),
            "duree_entrainement_s": config.get("resultats", {}).get("duree_totale_s"),
            "suivi": ck["suivi"], "selection": config["selection"],
            "historique": historique_gan(lire_json(args.run_dir / "gan" / "history.json")),
            "evolution": evolution,
            "verification_onnx": {"generateur": v_gen},
            "dataset": ck["dataset"], "date_creation": ck["date_creation"],
        }
        print(f"  GAN générateur : {v_gen['taille_octets'] / 1e3:.0f} ko, écart max ONNX/PyTorch "
              f"{v_gen['ecart_max']:.2e} sur {v_gen['entrees']} entrées : "
              f"{'OK' if v_gen['conforme'] else 'ÉCART SUPÉRIEUR AU SEUIL'}")
    if not modeles_meta:
        raise SystemExit("Aucun modèle entraîné. Lancez d'abord `mise run tp6_03_train_vae` / `tp6_04_train_gan`.")

    provenance = lire_json(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else {}
    regions = manifeste["regions"]
    ecrire_json(MODEL_DIR / "metadata.json", {
        "application": "Ocean Terrain Lab",
        "avertissement": "Reliefs SYNTHÉTIQUES produits par des modèles génératifs dans un cadre pédagogique. "
                         "Ce ne sont ni des relevés réels, ni des représentations géologiquement validées : "
                         "ne pas utiliser pour la navigation ou l'hydrographie.",
        "patch": PATCH,
        "pixel_km": manifeste["pixel_km_nominal"],
        "normalisation": manifeste["normalisation"],
        "inference": {
            "encodeur": {"entree": "elevation", "forme": [None, PATCH, PATCH], "unite": "m (élévation, négative en mer)",
                         "sorties": ["mu", "logvar"]},
            "generateurs": {"entree": "z", "sortie": "relief", "unite": "m (écart à la moyenne du patch)"},
            "graines": "z ~ N(0, I) par Box-Muller sur mulberry32 (identique en Python : terrain.vecteur_latent)",
            "onnx_opset": ONNX_OPSET,
        },
        "modeles": modeles_meta,
        "styles": STYLES,
        "palette": PALETTE,
        "reels": {"fichier": "reels.bin", "codage": "int16 little-endian, élévation (m), N × 64 × 64, ligne 0 = nord",
                  "patches": [{**index_test[i], "nom_region": regions[index_test[i]["region"]]["nom"]}
                              for i in galerie]},
        "regions": {k: {"nom": v["nom"], "morphologie": v["morphologie"], "emprise": v["emprise"]}
                    for k, v in regions.items()},
        "evaluation": None if evaluation is None else {
            k: evaluation[k] for k in ("patches_test", "diversite", "statistiques", "synthese_markdown")
        } | {"modeles": evaluation["modeles"]},
        "controle": controle,
        "dataset": {"identifiant": manifeste["identifiant"], "repartition": manifeste["repartition"],
                    "parametres": manifeste["parametres"], "statistiques": manifeste["statistiques"]},
        "sources": {"donnees": ATTRIBUTION, "serveur": provenance.get("source", {}).get("serveur"),
                    "portail": "https://www.gebco.net/"},
        "date_export": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"Site : {MODEL_DIR} (metadata.json, reels.bin : {len(galerie)} patches réels de test)")
    print("Étape suivante :\n  bun run dev:tp6")


if __name__ == "__main__":
    main()
