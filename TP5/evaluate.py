"""Évaluation des modèles et des références sur le jeu de test (TP5, étape 4 : F13 à F16, F24).

Même protocole pour tous : mêmes fenêtres de test (période chronologiquement
postérieure à l'entraînement et à la validation), mêmes métriques en mètres.

  - réseaux : RNN, GRU, LSTM (checkpoints ``runs/<modele>/best.pt``) ;
  - références naïves (F15) : persistance, répétition à 24 h, répétition à 25 h ;
  - métriques (F13, F14) : MAE et RMSE globales sur 24 h et à H+1, H+6, H+12, H+24 ;
  - figures (F16) : erreur selon l'échéance, prévisions et écarts sur des périodes types ;
  - périodes d'exemple pour le site (F24) : forte et faible amplitude, niveau
    moyen atypique, période typique, période la plus difficile pour les réseaux.

Produits : ``runs/evaluation.json`` et ``runs/evaluation/*.png``.

Usage :
    uv run python TP5/evaluate.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

from maree import (
    DATASET_DIR,
    DESCRIPTIONS_REFERENCES,
    HORIZONS_EVAL,
    MODELES,
    REFERENCES,
    RUNS_DIR,
    STYLES,
    charger_serie,
    ecrire_json,
    fenetres,
    iso,
    metriques,
    predire_references,
)
from modeles import construire_modele_export, predire

MARQUEURS_MPL = {"triangle": "^", "carre": "s", "losange": "D", "croix": "x", "plus": "+", "cercle": "o",
                 "rond": "o", "aucun": None}


def style_mpl(nom: str) -> dict:
    s = STYLES[nom]
    return {"color": s["couleur"], "linestyle": (0, s["tirets"]) if s["tirets"] else "-",
            "marker": MARQUEURS_MPL[s["marqueur"]], "markersize": 4, "label": s["nom"]}


def choisir_exemples(temps: np.ndarray, ancres: np.ndarray, x: np.ndarray, y: np.ndarray,
                     erreur_reseaux: np.ndarray, ecart_min_h: int = 7 * 24) -> list[dict]:
    """Périodes réelles du test, distinctes d'au moins une semaine (F24)."""
    tout = np.concatenate([x, y], axis=1)
    amplitude = tout.max(axis=1) - tout.min(axis=1)
    niveau = tout.mean(axis=1)
    criteres = [
        ("difficile", "Période difficile",
         "fenêtre de test où les trois réseaux font, en moyenne, leur plus grande erreur",
         np.argmax(erreur_reseaux)),
        ("vives_eaux", "Forte amplitude (vives-eaux)",
         "marnage maximal du jeu de test", np.argmax(amplitude)),
        ("mortes_eaux", "Faible amplitude (mortes-eaux)",
         "marnage minimal du jeu de test", np.argmin(amplitude)),
        ("niveau_atypique", "Niveau moyen atypique",
         "niveau moyen sur 72 h le plus éloigné de la moyenne : effet probable de la météo "
         "(pression, vent : surcote ou décote)", np.argmax(np.abs(niveau - np.median(niveau)))),
        ("typique", "Période typique",
         "erreur des réseaux proche de leur erreur médiane",
         np.argmin(np.abs(erreur_reseaux - np.median(erreur_reseaux)))),
    ]
    choisis: list[dict] = []
    for ident, titre, description, i in criteres:
        i = int(i)
        if any(abs(int(ancres[i]) - c["indice"]) < ecart_min_h for c in choisis):
            # Trop proche d'un exemple déjà retenu : on prend le meilleur candidat suivant.
            ordre = {"difficile": -erreur_reseaux, "vives_eaux": -amplitude, "mortes_eaux": amplitude,
                     "niveau_atypique": -np.abs(niveau - np.median(niveau)),
                     "typique": np.abs(erreur_reseaux - np.median(erreur_reseaux))}[ident]
            i = next(int(j) for j in np.argsort(ordre)
                     if all(abs(int(ancres[j]) - c["indice"]) >= ecart_min_h for c in choisis))
        choisis.append({"id": ident, "titre": titre, "description": description, "indice": int(ancres[i]),
                        "fenetre": i, "ancre": iso(temps[ancres[i]]),
                        "marnage_m": round(float(amplitude[i]), 2), "niveau_moyen_m": round(float(niveau[i]), 2),
                        "mae_moyenne_reseaux_m": round(float(erreur_reseaux[i]), 4)})
    return choisis


def figure_horizons(resultats: dict, chemin: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for nom, m in resultats.items():
        pas = np.arange(1, len(m["par_pas"]["mae"]) + 1)
        ax1.plot(pas, 100 * np.array(m["par_pas"]["mae"]), **style_mpl(nom))
        ax2.plot(pas, 100 * np.array(m["par_pas"]["mae"]), **style_mpl(nom))
    ax1.set_title("MAE selon l'échéance — jeu de test")
    ax2.set_title("Zoom sur les réseaux de neurones")
    ax2.set_ylim(0, 1.6 * 100 * max(max(resultats[m]["par_pas"]["mae"]) for m in MODELES if m in resultats))
    for ax in (ax1, ax2):
        ax.set_xlabel("échéance (heures après l'instant de prévision)")
        ax.set_ylabel("MAE (cm)")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def figure_exemples(exemples: list[dict], temps: np.ndarray, x: np.ndarray, y: np.ndarray,
                    previsions: dict[str, np.ndarray], chemin: Path) -> None:
    """Observations, prévisions et écarts sur chaque période d'exemple (F16)."""
    fig, axes = plt.subplots(2, len(exemples), figsize=(5 * len(exemples), 7.5),
                             gridspec_kw={"height_ratios": [3, 1.4]}, squeeze=False)
    entree, horizon = x.shape[1], y.shape[1]
    # Écarts : réseaux et meilleure référence (la persistance, à ±5 m, écraserait l'échelle).
    ecarts_affiches = [*MODELES, "repetition_25h"]
    for k, ex in enumerate(exemples):
        i, t = ex["fenetre"], ex["indice"]
        dates_x = temps[t - entree + 1:t + 1].astype("datetime64[s]").astype(datetime)
        dates_y = temps[t + 1:t + horizon + 1].astype("datetime64[s]").astype(datetime)
        ax, axe = axes[0, k], axes[1, k]
        ax.plot(dates_x, x[i], color=STYLES["observe"]["couleur"], lw=1.8, label=STYLES["observe"]["nom"])
        ax.plot(dates_y, y[i], color=STYLES["realite"]["couleur"], lw=2, marker="o", ms=3,
                label=STYLES["realite"]["nom"])
        for nom, p in previsions.items():
            ax.plot(dates_y, p[i], lw=1.3, **{**style_mpl(nom), "markersize": 3})
            if nom in ecarts_affiches:
                axe.plot(dates_y, 100 * (p[i] - y[i]), lw=1.1, **{**style_mpl(nom), "markersize": 3})
        ax.axvline(dates_x[-1], color="black", ls=":", lw=1)
        ax.set_title(f"{ex['titre']}\nprévision du {dates_x[-1]:%d/%m/%Y %H:%M} UTC", fontsize=10)
        ax.set_ylabel("hauteur (m)")
        axe.axhline(0, color="black", lw=0.8)
        axe.set_ylabel("prévision − réalité (cm)")
        axe.set_title("écarts : réseaux et répétition 25 h", fontsize=8)
        for a in (ax, axe):
            a.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%Hh"))
            a.grid(alpha=0.3)
        axe.set_xlim(ax.get_xlim())
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR, help="répertoire des modèles entraînés")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="répertoire du dataset")
    args = parser.parse_args()

    serie = charger_serie(args.data_dir)
    p = serie["manifeste"]["parametres"]
    ancres = serie["ancres"]["test"]
    x, y = fenetres(serie["hauteur"], ancres, p["entree_h"], p["horizon_h"])
    rep = serie["manifeste"]["repartition"]["test"]
    print(f"Jeu de test : {rep['debut'][:10]} → {rep['fin'][:10]}, {len(x)} fenêtres")

    previsions: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    for m in MODELES:
        chemin = args.run_dir / m / "best.pt"
        if not chemin.exists():
            print(f"  {m} : checkpoint absent ({chemin}), ignoré. Lancez `mise run tp5_03_train`.")
            continue
        checkpoint = torch.load(chemin, map_location="cpu", weights_only=False)
        if checkpoint["dataset"] != serie["manifeste"]["identifiant"]:
            print(f"  ATTENTION : {m} a été entraîné sur le dataset {checkpoint['dataset']}, "
                  f"différent du dataset courant {serie['manifeste']['identifiant']}.")
        previsions[m] = predire(construire_modele_export(checkpoint), x)
        infos[m] = {"epoque": checkpoint["epoque"], "dataset": checkpoint["dataset"]}
    if not previsions:
        raise SystemExit("Aucun modèle entraîné. Lancez d'abord `mise run tp5_03_train`.")
    references = predire_references(x, p["horizon_h"])

    resultats = {nom: metriques(prev, y) for nom, prev in {**previsions, **references}.items()}

    entete = "".join(f"{f'H+{h}':>8s}" for h in HORIZONS_EVAL)
    print(f"\nMAE en cm (RMSE globale en cm) — {len(x)} fenêtres de test")
    print(f"  {'modèle':16s}{entete}{'24 h':>8s}{'RMSE':>8s}")
    for nom, r in resultats.items():
        ligne = "".join(f"{100 * r['horizons'][f'H+{h}']['mae']:8.1f}" for h in HORIZONS_EVAL)
        print(f"  {STYLES[nom]['nom']:16s}{ligne}{100 * r['global']['mae']:8.1f}{100 * r['global']['rmse']:8.1f}")

    classement = sorted(resultats, key=lambda n: resultats[n]["global"]["mae"])
    meilleure_ref = min(REFERENCES, key=lambda n: resultats[n]["global"]["mae"])
    gains = {m: round(1 - resultats[m]["global"]["mae"] / resultats[meilleure_ref]["global"]["mae"], 4)
             for m in previsions}
    print(f"\nClassement global (MAE 24 h) : {' < '.join(STYLES[n]['nom'] for n in classement)}")
    print(f"Meilleure référence : {STYLES[meilleure_ref]['nom']} ; réduction d'erreur des réseaux : "
          + ", ".join(f"{STYLES[m]['nom']} {100 * g:.0f} %" for m, g in gains.items()))

    erreur_reseaux = np.mean([np.abs(previsions[m] - y).mean(axis=1) for m in previsions], axis=0)
    exemples = choisir_exemples(serie["temps"], ancres, x, y, erreur_reseaux)
    for ex in exemples:
        maes = {nom: round(float(np.abs(prev[ex["fenetre"]] - y[ex["fenetre"]]).mean()), 4)
                for nom, prev in {**previsions, **references}.items()}
        ex["mae_m"] = maes
        print(f"  exemple « {ex['titre']} » : {ex['ancre']}, marnage {ex['marnage_m']} m, "
              f"MAE réseaux {100 * ex['mae_moyenne_reseaux_m']:.1f} cm")

    sortie = args.run_dir / "evaluation"
    sortie.mkdir(parents=True, exist_ok=True)
    figure_horizons(resultats, sortie / "erreur_par_echeance.png")
    figure_exemples(exemples, serie["temps"], x, y, {**previsions, **references}, sortie / "exemples_previsions.png")

    ecrire_json(args.run_dir / "evaluation.json", {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": serie["manifeste"]["identifiant"],
        "test": {"debut": rep["debut"], "fin": rep["fin"], "fenetres": int(len(x))},
        "protocole": "mêmes fenêtres de test pour tous ; métriques en mètres ; "
                     "références calculées à partir des seules 48 h d'entrée",
        "modeles": infos,
        "references": DESCRIPTIONS_REFERENCES,
        "metriques": resultats,
        "classement_mae": classement,
        "meilleure_reference": meilleure_ref,
        "reduction_erreur_vs_meilleure_reference": gains,
        "exemples": exemples,
    })
    print(f"\nRésultats : {args.run_dir / 'evaluation.json'}, figures dans {sortie}")
    print("Étape suivante :\n  uv run python TP5/export_onnx.py")


if __name__ == "__main__":
    main()
