"""Évaluation et comparaison VAE / GAN (TP6, étape 6 : F10, F19 à F23).

Sur le **jeu de test** (blocs géographiques jamais vus) :

  - **Reconstruction** (VAE, F10) : RMSE et erreur maximale en mètres, globales
    et par région ; meilleurs, médians et pires exemples ;
  - **Générations** : autant de reliefs que de patches de test pour chaque
    modèle, à partir de graines fixes ;
  - **Diversité** (F21) : distance moyenne entre paires (rapportée aux réels),
    part de quasi-doublons (signe de *mode collapse*), distance au plus proche
    voisin d'entraînement (mémorisation) ;
  - **Statistiques** (F22) : distributions des valeurs normalisées, de
    l'amplitude, de l'écart-type, de la variabilité locale, des pentes et de la
    rugosité ; distance de Wasserstein-1 aux réels ;
  - **Courbes** (F19) : pertes VAE et GAN sur deux panneaux séparés ;
  - **Synthèse** (F23) : tableau comparatif (``synthese.md``) et observations.

Sorties : ``runs/evaluation.json``, ``runs/synthese.md``, ``runs/evaluation/*.png``.

Usage :
    uv run python TP6/evaluate.py
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
    RUNS_DIR,
    STYLES,
    charger_dataset,
    colormap,
    denormaliser,
    distance_w1,
    diversite,
    ecrire_json,
    indicateurs,
    lire_json,
    lot_latent,
    pentes_deg,
)

GRAINE_EVAL = 100_000  # graines des générations d'évaluation (distinctes des galeries du site)
PIXEL_KM = 1.85
NOMS_INDICATEURS = {
    "amplitude_m": "Amplitude (m)",
    "ecart_type_m": "Écart-type du relief (m)",
    "variabilite_locale_m": "Variabilité locale 3×3 (m)",
    "pente_moy_deg": "Pente moyenne (°)",
    "rugosite_m": "Rugosité (m)",
}


def distances_ppv(a: np.ndarray, b: np.ndarray, bloc: int = 256) -> np.ndarray:
    """Distance RMS (par pixel) de chaque carte de ``a`` à sa plus proche voisine dans ``b``."""
    a2 = a.reshape(len(a), -1).astype(np.float32)
    b2 = b.reshape(len(b), -1).astype(np.float32)
    nb = (b2**2).sum(1)
    res = []
    for i in range(0, len(a2), bloc):
        x = a2[i:i + bloc]
        d2 = (x**2).sum(1)[:, None] + nb[None, :] - 2 * x @ b2.T
        res.append(np.sqrt(np.clip(d2.min(1), 0, None) / a2.shape[1]))
    return np.concatenate(res)


SEUIL_CORRELATION = 0.95


def part_quasi_doublons(lot: np.ndarray, seuil: float = SEUIL_CORRELATION) -> float:
    """Part des cartes dont la forme est presque identique à celle d'une autre carte du lot.

    On compare les **formes** (corrélation de Pearson entre cartes), pas les
    amplitudes : deux reliefs doux de quelques mètres ne sont pas des doublons.
    """
    x = lot.reshape(len(lot), -1).astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    r = x @ x.T
    np.fill_diagonal(r, -1)
    return float((r.max(1) >= seuil).mean())


def sans_recouvrement(index: list[dict]) -> np.ndarray:
    """Indices de patches de test deux à deux disjoints (les patches d'un même bloc se chevauchent)."""
    garde: list[int] = []
    for i, p in enumerate(index):
        if all(q["region"] != p["region"] or abs(q["ligne"] - p["ligne"]) >= 64 or abs(q["colonne"] - p["colonne"]) >= 64
               for q in (index[j] for j in garde)):
            garde.append(i)
    return np.array(garde)


def stabilite_vae(h: list[dict]) -> dict:
    v = np.array([e["val_perte"] for e in h])
    fin = v[int(0.8 * len(v)):]
    return {"variation_relative_fin": float(fin.std() / fin.mean()),
            "baisse_totale": float(1 - v.min() / v[0]), "monotone_lissee": bool(np.all(np.diff(np.convolve(v, np.ones(5) / 5, "valid")) < 0.02 * v.mean()))}


def stabilite_gan(h: list[dict]) -> dict:
    pd = np.array([e["perte_d"] for e in h])
    pg = np.array([e["perte_g"] for e in h])
    suivi = [e for e in h if e.get("ecart_stat") is not None]
    ecarts = np.array([e["ecart_stat"] for e in suivi])
    div = np.array([e["diversite_relative"] for e in suivi])
    moitie = len(ecarts) // 2
    return {"ecart_type_perte_d": float(pd.std()), "ecart_type_perte_g": float(pg.std()),
            "variation_ecart_stat_2e_moitie": float(ecarts[moitie:].std() / max(ecarts[moitie:].mean(), 1e-9)),
            "diversite_relative_min": float(div.min()), "diversite_relative_max": float(div.max()), "diversite_relative_fin": float(div[-1]),
            "chutes_diversite": int((div < 0.5).sum())}


def figure_cote_a_cote(reel: np.ndarray, vae: np.ndarray | None, gan: np.ndarray | None, chemin: Path,
                       echelle: float, n: int = 6) -> None:
    """F20 : réel / VAE / GAN, mêmes conventions (relief relatif, même palette et même échelle)."""
    lignes = [("Réel (GEBCO, test)", reel)] + [(STYLES[m]["nom"] + " (généré)", g) for m, g in
                                                (("vae", vae), ("gan", gan)) if g is not None]
    fig, axes = plt.subplots(len(lignes), n, figsize=(2.2 * n, 2.4 * len(lignes)), squeeze=False)
    for li, (nom, lot) in enumerate(lignes):
        for k in range(n):
            ax = axes[li, k]
            im = ax.imshow(lot[k], cmap=colormap(), vmin=-echelle, vmax=echelle)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nom, fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, label="relief relatif (m)")
    fig.suptitle("Réels et synthétiques côte à côte — les colonnes ne se correspondent PAS : "
                 "VAE et GAN sont des générations indépendantes", fontsize=10)
    fig.savefig(chemin, dpi=90, bbox_inches="tight")
    plt.close(fig)


def figure_reconstructions(orig: np.ndarray, rec: np.ndarray, rmse: np.ndarray, index: list[dict], chemin: Path,
                           echelle: float) -> list[dict]:
    """Meilleures, médianes et pires reconstructions (original, reconstruit, erreur)."""
    ordre = np.argsort(rmse)
    choix = [("meilleure", ordre[0]), ("meilleure", ordre[1]), ("médiane", ordre[len(ordre) // 2]),
             ("médiane", ordre[len(ordre) // 2 + 1]), ("pire", ordre[-2]), ("pire", ordre[-1])]
    fig, axes = plt.subplots(3, len(choix), figsize=(2.4 * len(choix), 7.4))
    for k, (nom, i) in enumerate(choix):
        for li, (img, cmap, lim) in enumerate(((orig[i], colormap(), echelle), (rec[i], colormap(), echelle),
                                              (rec[i] - orig[i], "RdBu_r", 500))):
            ax = axes[li, k]
            ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, k].set_title(f"{nom}\n{index[i]['region']}", fontsize=8)
        axes[2, k].set_xlabel(f"RMSE {rmse[i]:.0f} m", fontsize=8)
    for li, nom in enumerate(("original", "reconstruit", "erreur ±500 m")):
        axes[li, 0].set_ylabel(nom)
    fig.suptitle("VAE — reconstructions du jeu de test")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)
    return [{"type": nom, "patch_test": int(i), "region": index[i]["region"], "rmse_m": float(rmse[i])}
            for nom, i in choix]


def figure_diversite(lots: dict[str, np.ndarray], chemin: Path, echelle: float, n: int = 24) -> None:
    """F21 : galeries de chaque source pour juger la variété et les répétitions."""
    fig, axes = plt.subplots(len(lots), 1, figsize=(16, 2.4 * len(lots) * n / 12))
    for ax, (nom, lot) in zip(np.atleast_1d(axes), lots.items()):
        mosaique = np.full((2 * 64 + 2, 12 * 66 - 2), np.nan)
        for k in range(min(n, len(lot))):
            r, c = divmod(k, 12)
            mosaique[r * 66:r * 66 + 64, c * 66:c * 66 + 64] = lot[k]
        ax.imshow(mosaique, cmap=colormap(), vmin=-echelle, vmax=echelle)
        ax.set_title(nom)
        ax.axis("off")
    fig.suptitle("Diversité : 24 reliefs par source (graines consécutives pour VAE et GAN)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=80)
    plt.close(fig)


def figure_statistiques(ind: dict[str, dict], valeurs: dict[str, np.ndarray], pentes: dict[str, np.ndarray],
                        chemin: Path) -> None:
    """F22 : distributions des indicateurs, réels contre synthétiques."""
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    axes = list(axes.flat)
    for nom, v in valeurs.items():
        axes[0].hist(v, bins=100, range=(-1, 1), histtype="step", density=True, lw=1.5,
                     color=STYLES[nom]["couleur"], label=STYLES[nom]["nom"])
    axes[0].set_yscale("log")
    axes[0].set_title("Valeurs normalisées des pixels")
    for nom, p in pentes.items():
        axes[1].hist(p, bins=80, range=(0, 30), histtype="step", density=True, lw=1.5,
                     color=STYLES[nom]["couleur"], label=STYLES[nom]["nom"])
    axes[1].set_yscale("log")
    axes[1].set_title("Pentes des pixels (°)")
    for ax, (cle, titre) in zip(axes[2:], NOMS_INDICATEURS.items()):
        hi = max(np.quantile(i[cle], 0.99) for i in ind.values())
        for nom, i in ind.items():
            ax.hist(i[cle], bins=50, range=(0, hi), histtype="step", density=True, lw=1.5,
                    color=STYLES[nom]["couleur"], label=STYLES[nom]["nom"])
        ax.set_title(f"{titre} par patch")
    axes[-1].axis("off")
    for ax in axes[:-1]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Caractéristiques statistiques : réels (test) et synthétiques — une ressemblance statistique "
                 "n'est pas une preuve de réalisme géologique")
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def figure_courbes(run_dir: Path, chemin: Path) -> None:
    """F19 : courbes VAE et GAN sur des panneaux séparés (grandeurs différentes)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    if (run_dir / "vae" / "history.json").exists():
        h = lire_json(run_dir / "vae" / "history.json")
        e = [x["epoque"] for x in h]
        ax1.plot(e, [x["train_perte"] for x in h], label="perte totale (train)", color=STYLES["vae"]["couleur"])
        ax1.plot(e, [x["val_perte"] for x in h], label="perte totale (val)", color=STYLES["vae"]["couleur"], ls="--")
        ax1.plot(e, [x["val_reconstruction"] for x in h], label="reconstruction (val)", color="#009E73")
        ax1.plot(e, [x["val_kl"] for x in h], label="KL (val)", color="#CC79A7")
        ax1.set_title("VAE : une perte qui décroît, validation disponible")
    if (run_dir / "gan" / "history.json").exists():
        h = lire_json(run_dir / "gan" / "history.json")
        e = [x["epoque"] for x in h]
        ax2.plot(e, [x["perte_d"] for x in h], label="perte D", color="#0072B2")
        ax2.plot(e, [x["perte_g"] for x in h], label="perte G", color=STYLES["gan"]["couleur"])
        ax2.set_title("GAN : deux pertes en équilibre, sans convergence vers 0")
    for ax in (ax1, ax2):
        ax.set_xlabel("époque")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    ds = charger_dataset(args.data_dir)
    echelle = ds["manifeste"]["normalisation"]["echelle_m"]
    x_test, x_train = ds["test"]["z"], ds["train"]["z"]
    index = ds["index"]["test"]
    n = len(x_test)
    out = args.run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    resultats: dict = {"jeu": "test", "patches_test": n, "dataset": ds["manifeste"]["identifiant"],
                       "graine_generations": GRAINE_EVAL, "modeles": {}}

    generations: dict[str, np.ndarray] = {}
    configs: dict[str, dict] = {}
    vae_ck = args.run_dir / "vae" / "best.pt"
    gan_ck = args.run_dir / "gan" / "best.pt"
    if vae_ck.exists():
        ck = torch.load(vae_ck, map_location="cpu", weights_only=False)
        vae = construire_vae(ck)
        rec = reconstruire(vae, x_test)
        orig_m, rec_m = denormaliser(x_test, echelle), denormaliser(rec, echelle)
        err = rec_m - orig_m
        rmse = np.sqrt((err**2).mean(axis=(1, 2)))
        par_region = {}
        for r in sorted({p["region"] for p in index}):
            sel = np.array([p["region"] == r for p in index])
            par_region[r] = {"patches": int(sel.sum()), "rmse_m": float(np.sqrt((err[sel]**2).mean()))}
        exemples = figure_reconstructions(orig_m, rec_m, rmse, index, out / "reconstructions.png", echelle)
        generations["vae"] = generer(vae.decodeur, lot_latent(GRAINE_EVAL, n, ck["config"]["dim_latente"]))
        configs["vae"] = lire_json(args.run_dir / "vae" / "config.json")
        resultats["modeles"]["vae"] = {
            "epoque": ck["epoque"], "dim_latente": ck["config"]["dim_latente"], "beta": ck["config"]["beta"],
            "reconstruction": {"rmse_m": float(np.sqrt((err**2).mean())), "mae_m": float(np.abs(err).mean()),
                               "rmse_mediane_patch_m": float(np.median(rmse)),
                               "erreur_max_m": float(np.abs(err).max()), "par_region": par_region,
                               "exemples": exemples},
            "stabilite": stabilite_vae(lire_json(args.run_dir / "vae" / "history.json")),
        }
        print(f"VAE : reconstruction test RMSE {resultats['modeles']['vae']['reconstruction']['rmse_m']:.1f} m "
              f"(médiane par patch {np.median(rmse):.1f} m)")
    if gan_ck.exists():
        ck = torch.load(gan_ck, map_location="cpu", weights_only=False)
        generations["gan"] = generer(construire_generateur(ck), lot_latent(GRAINE_EVAL, n, ck["config"]["dim_latente"]))
        configs["gan"] = lire_json(args.run_dir / "gan" / "config.json")
        resultats["modeles"]["gan"] = {"epoque": ck["epoque"], "dim_latente": ck["config"]["dim_latente"],
                                       "stabilite": stabilite_gan(lire_json(args.run_dir / "gan" / "history.json"))}
    if not generations:
        raise SystemExit("Aucun modèle entraîné. Lancez d'abord l'entraînement du VAE et du GAN.")

    # Diversité (F21) : distances entre paires, quasi-doublons, plus proche voisin d'entraînement.
    rng = np.random.default_rng(0)
    sous = rng.choice(len(x_train), min(len(x_train), 3500), replace=False)
    sources = {"reel": x_test, **generations}
    div_reel = diversite(x_test)
    # Quasi-doublons : sur des réels disjoints et sur autant de générations par modèle.
    disjoints = sans_recouvrement(index)
    ppv_reel = distances_ppv(x_test, x_train[sous])
    diversites = {}
    for nom, lot in sources.items():
        ppv = ppv_reel if nom == "reel" else distances_ppv(lot, x_train[sous])
        diversites[nom] = {
            "distance_moyenne_paires_m": diversite(lot) * echelle,
            "diversite_relative": diversite(lot) / div_reel,
            "part_quasi_doublons": part_quasi_doublons(lot[disjoints] if nom == "reel" else lot[:len(disjoints)]),
            "distance_ppv_train_mediane_m": float(np.median(ppv)) * echelle,
            "part_tres_proches_train": float((ppv < 0.5 * np.median(ppv_reel)).mean()),
        }
    resultats["diversite"] = {"quasi_doublon": f"corrélation ≥ {SEUIL_CORRELATION} avec une autre carte, "
                                               f"sur {len(disjoints)} cartes (réels de test sans recouvrement)",
                              "par_source": diversites}
    figure_diversite({STYLES[k]["nom"]: denormaliser(v, echelle) for k, v in sources.items()},
                     out / "diversite.png", echelle)

    # Statistiques (F22)
    reliefs = {k: denormaliser(v, echelle) for k, v in sources.items()}
    ind = {k: indicateurs(v, PIXEL_KM) for k, v in reliefs.items()}
    pentes = {k: pentes_deg(v[:300], PIXEL_KM).reshape(-1) for k, v in reliefs.items()}
    valeurs = {k: v.reshape(-1)[::7] for k, v in sources.items()}
    statistiques: dict = {}
    for k in sources:
        statistiques[k] = {
            "medianes": {c: float(np.median(ind[k][c])) for c in NOMS_INDICATEURS},
            "pente_pixels_p90_deg": float(np.quantile(pentes[k], 0.9)),
            "part_valeurs_saturees": float((np.abs(sources[k]) > 0.98).mean()),
        }
        if k != "reel":
            w = {c: distance_w1(ind[k][c], ind["reel"][c]) for c in NOMS_INDICATEURS}
            statistiques[k]["w1"] = w
            statistiques[k]["w1_relatif_moyen"] = float(np.mean([w[c] / ind["reel"][c].std() for c in w]))
            statistiques[k]["w1_valeurs"] = distance_w1(valeurs[k], valeurs["reel"])
            statistiques[k]["w1_pentes_deg"] = distance_w1(pentes[k], pentes["reel"])
    resultats["statistiques"] = statistiques
    figure_statistiques(ind, valeurs, pentes, out / "statistiques.png")

    # Côte à côte (F20) et courbes (F19)
    figure_cote_a_cote(reliefs["reel"][::max(1, n // 6)][:6], reliefs.get("vae"), reliefs.get("gan"),
                       out / "cote_a_cote.png", echelle)
    figure_courbes(args.run_dir, out / "courbes.png")

    # Synthèse (F23)
    lignes = ["| Critère | VAE | GAN |", "|---|---|---|"]

    def cellule(m: str, f) -> str:
        return f(m) if m in resultats["modeles"] else "—"

    def duree(m: str) -> str:
        r = configs[m].get("resultats")
        return f"{r['duree_totale_s'] / 60:.1f} min ({r['epoques']} ép.)" if r else "entraînement inachevé"

    lignes.append(f"| Durée d'entraînement | {cellule('vae', duree)} | {cellule('gan', duree)} |")
    lignes.append("| Stabilité observée | " + cellule("vae", lambda m: (
        f"perte de validation décroissante, variation finale {100 * resultats['modeles'][m]['stabilite']['variation_relative_fin']:.1f} %"))
        + " | " + cellule("gan", lambda m: (
            f"pertes D et G presque constantes (σ perte D {resultats['modeles'][m]['stabilite']['ecart_type_perte_d']:.2f}) : "
            f"elles ne disent rien de la qualité ; écart stat. instable d'un contrôle à l'autre "
            f"(± {100 * resultats['modeles'][m]['stabilite']['variation_ecart_stat_2e_moitie']:.0f} %), "
            f"diversité de {resultats['modeles'][m]['stabilite']['diversite_relative_min']:.2f} à "
            f"{resultats['modeles'][m]['stabilite']['diversite_relative_max']:.2f}")) + " |")
    lignes.append("| Diversité (1 = réels) | " + " | ".join(
        cellule(m, lambda m: f"{diversites[m]['diversite_relative']:.2f}, quasi-doublons "
                             f"{100 * diversites[m]['part_quasi_doublons']:.1f} %") for m in ("vae", "gan")) + " |")
    lignes.append("| Écart statistique aux réels (W1 relatif) | " + " | ".join(
        cellule(m, lambda m: f"{statistiques[m]['w1_relatif_moyen']:.2f}") for m in ("vae", "gan")) + " |")
    lignes.append("| Rugosité médiane (réels : "
                  f"{statistiques['reel']['medianes']['rugosite_m']:.0f} m) | " + " | ".join(
        cellule(m, lambda m: f"{statistiques[m]['medianes']['rugosite_m']:.0f} m") for m in ("vae", "gan")) + " |")
    lignes.append("| Reconstruction (test) | " + cellule("vae", lambda m: (
        f"RMSE {resultats['modeles'][m]['reconstruction']['rmse_m']:.0f} m")) + " | impossible (pas d'encodeur) |")
    lignes.append("| Limites | reliefs lissés, détails fins perdus (perte quadratique + KL) | "
                  "pas de critère de convergence, risque de mode collapse, pas de reconstruction |")
    lignes.append("| Observations pédagogiques | espace latent régulier : interpolations continues | "
                  "textures plus nettes ; qualité très dépendante de l'époque retenue |")
    texte = "\n".join(["# Synthèse comparative VAE / GAN (jeu de test)", "",
                       f"Dataset `{ds['manifeste']['identifiant']}`, {n} patches de test, "
                       f"{n} générations par modèle (graines {GRAINE_EVAL}…).", "", *lignes, "",
                       "Les indicateurs comparent des **distributions** : ils ne prouvent pas le réalisme "
                       "géologique des reliefs synthétiques.", ""])
    (args.run_dir / "synthese.md").write_text(texte, encoding="utf-8")
    resultats["synthese_markdown"] = texte
    ecrire_json(args.run_dir / "evaluation.json", resultats)

    print("\nDiversité (1 = réels) et écart statistique :")
    for k in sources:
        w = statistiques[k].get("w1_relatif_moyen")
        print(f"  {STYLES[k]['nom']:14s} diversité {diversites[k]['diversite_relative']:.2f}  "
              f"quasi-doublons {100 * diversites[k]['part_quasi_doublons']:5.1f} %  "
              f"ppv train {diversites[k]['distance_ppv_train_mediane_m']:6.1f} m  "
              f"rugosité {statistiques[k]['medianes']['rugosite_m']:5.1f} m"
              + (f"  W1 relatif {w:.2f}" if w is not None else ""))
    print(f"\n{texte}")
    print(f"Figures : {out}\nÉtape suivante :\n  uv run python TP6/export_onnx.py")


if __name__ == "__main__":
    main()
