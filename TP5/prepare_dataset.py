"""Contrôle qualité, exploration et préparation des fenêtres (TP5, étape 2 : F02 à F05).

1. **Contrôle qualité** (F02) : doublons, horodatages hors grille horaire,
   intervalles irréguliers, valeurs invalides (non numériques, hors bornes
   physiques, pics isolés), heures manquantes, ruptures de continuité,
   changements possibles de référence verticale. Rien n'est corrigé en
   silence : chaque valeur écartée est comptée et listée dans
   ``data/qc/rapport_qc.json``.
2. **Grille horaire régulière** : une case par heure UTC ; ``NaN`` pour une
   heure sans observation valide. **Aucune interpolation.**
3. **Visualisations** (F03) : ``data/figures/*.png`` et statistiques descriptives.
4. **Séparation chronologique** (F05) : 70 % / 15 % / 15 % ; les fenêtres à
   cheval sur deux ensembles ou contenant une lacune sont écartées (F02, F04).
5. **Normalisation** ajustée sur la seule période d'entraînement.
6. **Manifeste** ``data/dataset/dataset.json`` (paramètres, coupures, nombre de
   fenêtres, normalisation, identifiant du dataset) et ``serie.npz``.

Usage :
    uv run python TP5/prepare_dataset.py
    uv run python TP5/prepare_dataset.py --entree 72 --horizon 12
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from maree import (
    BORNE_MAX_M,
    BORNE_MIN_M,
    DATASET_DIR,
    ENTREE_H,
    FIGURES_DIR,
    FRACTIONS,
    HORIZON_H,
    OBS_CSV,
    PAS_FENETRES_H,
    PIC_ISOLE_M,
    PROVENANCE_JSON,
    QC_DIR,
    SAUT_REFERENCE_M,
    STATION_NOM,
    ancres_valides,
    bornes_split,
    ecrire_json,
    iso,
    lire_json,
)

COULEURS_SPLIT = {"train": "#DBEAFE", "val": "#FEF3C7", "test": "#DCFCE7"}
NOMS_SPLIT = {"train": "entraînement", "val": "validation", "test": "test"}
# Mots du journal de bord du marégraphe qui peuvent signaler un changement de
# référence verticale (à relire par un humain, F02).
MOTS_REFERENCE = re.compile(r"z[ée]ro|r[ée]f[ée]rence|calage|nivellement|d[ée]calage|offset|coefficient",
                            re.IGNORECASE)


# --------------------------------------------------------------------------
# Lecture et contrôle qualité (F02)
# --------------------------------------------------------------------------

def lire_observations() -> list[tuple[str, str]]:
    if not OBS_CSV.exists():
        raise SystemExit(f"Observations absentes : {OBS_CSV}. Lancez d'abord `mise run tp5_01_download`.")
    with OBS_CSV.open(encoding="utf-8") as f:
        return [(r["horodatage_utc"], r["hauteur_m"]) for r in csv.DictReader(f)]


def intervalles(masque: np.ndarray, temps: np.ndarray) -> list[dict]:
    """Suites consécutives de ``True`` -> [{debut, fin, duree_h}]."""
    m = np.concatenate([[False], masque, [False]]).astype(np.int8)
    d = np.flatnonzero(np.diff(m))
    return [{"debut": iso(temps[a]), "fin": iso(temps[b - 1]), "duree_h": int(b - a)}
            for a, b in zip(d[::2], d[1::2])]


def controler(lignes: list[tuple[str, str]], debut: np.datetime64, fin: np.datetime64,
              provenance: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Contrôle qualité et mise sur grille horaire régulière.

    Retourne (temps, hauteur, rapport). ``hauteur`` vaut NaN pour toute heure
    sans observation valide ; le rapport détaille chaque catégorie d'écart.
    """
    temps = np.arange(debut, fin + np.timedelta64(1, "h"), dtype="datetime64[h]")
    n = temps.size
    hauteur = np.full(n, np.nan, dtype=np.float64)
    rapport: dict = {"lignes_brutes": len(lignes)}

    # --- Valeurs non numériques et horodatages hors grille -----------------
    non_numeriques, hors_grille, hors_periode = [], [], 0
    par_heure: dict[int, list[float]] = {}
    for brut_t, brut_v in lignes:
        t = np.datetime64(brut_t.rstrip("Z"), "s")
        try:
            v = float(brut_v)
        except ValueError:
            v = math.nan
        if not math.isfinite(v):
            non_numeriques.append(brut_t)
            continue
        if t.astype("datetime64[h]") != t:
            hors_grille.append(brut_t)
            continue
        i = int((t.astype("datetime64[h]") - debut).astype(int))
        if not 0 <= i < n:
            hors_periode += 1
            continue
        par_heure.setdefault(i, []).append(v)
    rapport["valeurs_non_numeriques"] = {"nombre": len(non_numeriques), "exemples": non_numeriques[:10]}
    rapport["horodatages_hors_grille_horaire"] = {"nombre": len(hors_grille), "exemples": hors_grille[:10]}
    rapport["hors_periode"] = hors_periode

    # --- Doublons : identiques (on en garde un) ou contradictoires (écartés) --
    doublons_identiques, doublons_contradictoires = [], []
    for i, valeurs in par_heure.items():
        if len(valeurs) > 1:
            if max(valeurs) - min(valeurs) < 1e-9:
                doublons_identiques.append(iso(temps[i]))
            else:
                doublons_contradictoires.append({"heure": iso(temps[i]), "valeurs": valeurs})
                continue
        hauteur[i] = valeurs[0]
    rapport["doublons"] = {
        "identiques": {"nombre": len(doublons_identiques), "traitement": "une seule valeur conservée",
                       "exemples": doublons_identiques[:10]},
        "contradictoires": {"nombre": len(doublons_contradictoires),
                            "traitement": "heure écartée (NaN)", "liste": doublons_contradictoires[:50]},
    }

    # --- Intervalles entre observations successives ---------------------------
    presentes = np.flatnonzero(np.isfinite(hauteur))
    pas = np.diff(presentes)
    valeurs_pas, effectifs = np.unique(pas, return_counts=True)
    rapport["intervalles_entre_observations_h"] = {str(int(k)): int(c) for k, c in zip(valeurs_pas, effectifs)}

    # --- Valeurs physiquement invalides ------------------------------------------
    hors_bornes = np.isfinite(hauteur) & ((hauteur < BORNE_MIN_M) | (hauteur > BORNE_MAX_M))
    avant = np.diff(hauteur, prepend=np.nan)
    apres = -np.diff(hauteur, append=np.nan)
    with np.errstate(invalid="ignore"):
        pic = (np.abs(avant) > PIC_ISOLE_M) & (np.abs(apres) > PIC_ISOLE_M) & (np.sign(avant) == np.sign(apres))
    rapport["valeurs_invalides"] = {
        "bornes_m": [BORNE_MIN_M, BORNE_MAX_M],
        "hors_bornes": {"nombre": int(hors_bornes.sum()),
                        "liste": [{"heure": iso(temps[i]), "valeur": float(hauteur[i])}
                                  for i in np.flatnonzero(hors_bornes)[:50]]},
        "pics_isoles": {"seuil_m_par_h": PIC_ISOLE_M, "nombre": int(pic.sum()),
                        "liste": [{"heure": iso(temps[i]), "valeur": float(hauteur[i])}
                                  for i in np.flatnonzero(pic)[:50]]},
        "traitement": "valeurs écartées (NaN), jamais corrigées",
    }
    hauteur[hors_bornes | pic] = np.nan

    # --- Heures manquantes et ruptures de continuité ----------------------------
    manquant = ~np.isfinite(hauteur)
    lacunes = intervalles(manquant, temps)
    rapport["grille"] = {"debut": iso(temps[0]), "fin": iso(temps[-1]), "heures_attendues": n,
                         "heures_valides": int((~manquant).sum()), "heures_manquantes": int(manquant.sum()),
                         "completude_pct": round(100 * float((~manquant).mean()), 3)}
    rapport["ruptures_de_continuite"] = {
        "nombre": len(lacunes),
        "plus_longue_h": max((l["duree_h"] for l in lacunes), default=0),
        "par_duree": {
            "1 h": sum(l["duree_h"] == 1 for l in lacunes),
            "2 à 6 h": sum(2 <= l["duree_h"] <= 6 for l in lacunes),
            "7 à 48 h": sum(7 <= l["duree_h"] <= 48 for l in lacunes),
            "> 48 h": sum(l["duree_h"] > 48 for l in lacunes),
        },
        "liste": lacunes,
        "traitement": "non comblées : toute fenêtre qui touche une lacune est écartée",
    }

    # --- Référence verticale ------------------------------------------------------
    annees = temps.astype("datetime64[Y]").astype(int) + 1970
    moyennes = {int(a): round(float(np.nanmean(hauteur[annees == a])), 4) for a in np.unique(annees)}
    valeurs = list(moyennes.values())
    sauts = [{"de": a, "a": b, "ecart_m": round(moyennes[b] - moyennes[a], 4)}
             for a, b in zip(list(moyennes)[:-1], list(moyennes)[1:])]
    suspects = [s for s in sauts if abs(s["ecart_m"]) > SAUT_REFERENCE_M]
    an0, an1 = str(temps[0].astype("datetime64[Y]")), str(temps[-1].astype("datetime64[Y]"))
    journal = [e for e in provenance.get("journal_de_bord", [])
               if e.get("date") and an0 <= e["date"][:4] <= an1 and MOTS_REFERENCE.search(e["texte"])]
    rapport["reference_verticale"] = {
        "declaree": provenance.get("reference_verticale"),
        "moyennes_annuelles_m": moyennes,
        "ecart_max_entre_annees_m": round(max(valeurs) - min(valeurs), 4),
        "sauts_suspects": suspects,
        "seuil_m": SAUT_REFERENCE_M,
        "evenements_journal_de_bord": journal,
        "conclusion": ("aucun changement de référence détecté" if not suspects and not journal
                       else "changement possible : à vérifier avant usage"),
    }
    return temps, hauteur.astype(np.float32), rapport


def statistiques(hauteur: np.ndarray) -> dict:
    h = hauteur[np.isfinite(hauteur)].astype(np.float64)
    q = np.percentile(h, [1, 25, 50, 75, 99])
    return {"n": int(h.size), "moyenne_m": round(float(h.mean()), 4), "ecart_type_m": round(float(h.std()), 4),
            "min_m": round(float(h.min()), 3), "p01_m": round(float(q[0]), 3), "p25_m": round(float(q[1]), 3),
            "mediane_m": round(float(q[2]), 3), "p75_m": round(float(q[3]), 3), "p99_m": round(float(q[4]), 3),
            "max_m": round(float(h.max()), 3)}


# --------------------------------------------------------------------------
# Visualisations exploratoires (F03)
# --------------------------------------------------------------------------

def figures(temps: np.ndarray, hauteur: np.ndarray, bornes: dict[str, tuple[int, int]], stats: dict) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dates = temps.astype("datetime64[s]").astype(datetime)
    manquant = ~np.isfinite(hauteur)
    produits = []

    # 1. Série complète, ensembles et lacunes
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for split, (a, b) in bornes.items():
        ax.axvspan(dates[a], dates[b - 1], color=COULEURS_SPLIT[split], label=NOMS_SPLIT[split], zorder=0)
    ax.plot(dates, hauteur, lw=0.25, color="#1F4E79")
    for i in np.flatnonzero(manquant):
        ax.axvline(dates[i], color="#E5484D", lw=0.6, alpha=0.5, zorder=1)
    ax.plot([], [], color="#E5484D", label="heure manquante")
    ax.set_title(f"{STATION_NOM} — hauteurs d'eau horaires ({dates[0]:%Y}–{dates[-1]:%Y}), m au-dessus du ZH")
    ax.set_ylabel("hauteur (m)")
    ax.legend(loc="upper right", ncols=4, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_serie_complete.png", dpi=110)
    plt.close(fig)
    produits.append("01_serie_complete.png")

    # 2. Extrait de 7 jours (premier jour du test entièrement valide) : on voit
    #    les deux marées par jour et le décalage d'environ 50 min d'un jour à l'autre.
    a = bornes["test"][0]
    while np.any(manquant[a:a + 7 * 24]):
        a += 24
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates[a:a + 7 * 24], hauteur[a:a + 7 * 24], "-o", ms=2.5, lw=1, color="#1F4E79")
    ax.set_title(f"Extrait de 7 jours à partir du {dates[a]:%d/%m/%Y} (UTC) — une mesure par heure")
    ax.set_ylabel("hauteur (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_extrait_7_jours.png", dpi=110)
    plt.close(fig)
    produits.append("02_extrait_7_jours.png")

    # 3. Distribution des hauteurs
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(hauteur[np.isfinite(hauteur)], bins=80, color="#1F4E79", alpha=0.85)
    for cle, style in (("moyenne_m", "-"), ("p01_m", ":"), ("p99_m", ":")):
        ax.axvline(stats[cle], color="#E69F00", ls=style, label=f"{cle[:-2]} = {stats[cle]:.2f} m")
    ax.set_title("Distribution des hauteurs horaires (bimodale : pleines et basses mers)")
    ax.set_xlabel("hauteur (m)")
    ax.set_ylabel("nombre d'heures")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_distribution.png", dpi=110)
    plt.close(fig)
    produits.append("03_distribution.png")

    # 4. Complétude mensuelle
    mois = temps.astype("datetime64[M]")
    uniques = np.unique(mois)
    manquantes = np.array([int(manquant[mois == m].sum()) for m in uniques])
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.bar(uniques.astype(datetime), manquantes, width=25, color=np.where(manquantes > 0, "#E5484D", "#9CA3AF"))
    ax.set_title("Heures manquantes par mois (aucune n'est comblée)")
    ax.set_ylabel("heures manquantes")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_lacunes_par_mois.png", dpi=110)
    plt.close(fig)
    produits.append("04_lacunes_par_mois.png")
    return produits


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entree", type=int, default=ENTREE_H, help="heures d'historique (entrée)")
    parser.add_argument("--horizon", type=int, default=HORIZON_H, help="heures prévues (sortie)")
    parser.add_argument("--pas", type=int, default=PAS_FENETRES_H, help="décalage entre fenêtres (h)")
    args = parser.parse_args()

    provenance = lire_json(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else {}
    periode = provenance.get("periode", {})
    debut = np.datetime64(periode.get("debut", "2021-01-01T00:00:00Z").rstrip("Z"), "h")
    fin = np.datetime64(periode.get("fin", "2025-12-31T23:00:00Z").rstrip("Z"), "h")

    lignes = lire_observations()
    temps, hauteur, rapport = controler(lignes, debut, fin, provenance)
    stats = statistiques(hauteur)
    rapport["statistiques"] = stats

    g, r = rapport["grille"], rapport["ruptures_de_continuite"]
    print(f"Lignes brutes : {rapport['lignes_brutes']}  ->  grille {g['debut']} … {g['fin']} ({g['heures_attendues']} h)")
    print(f"  doublons identiques : {rapport['doublons']['identiques']['nombre']}, "
          f"contradictoires : {rapport['doublons']['contradictoires']['nombre']}")
    print(f"  hors grille horaire : {rapport['horodatages_hors_grille_horaire']['nombre']}, "
          f"non numériques : {rapport['valeurs_non_numeriques']['nombre']}")
    print(f"  hors bornes : {rapport['valeurs_invalides']['hors_bornes']['nombre']}, "
          f"pics isolés : {rapport['valeurs_invalides']['pics_isoles']['nombre']}")
    print(f"  heures manquantes : {g['heures_manquantes']} ({100 - g['completude_pct']:.2f} %), "
          f"{r['nombre']} ruptures, la plus longue : {r['plus_longue_h']} h")
    print(f"  référence verticale : {rapport['reference_verticale']['conclusion']} "
          f"(moyennes annuelles {rapport['reference_verticale']['moyennes_annuelles_m']})")
    print(f"Statistiques : moyenne {stats['moyenne_m']} m, écart-type {stats['ecart_type_m']} m, "
          f"min {stats['min_m']} m, max {stats['max_m']} m")

    # --- Séparation chronologique et fenêtres (F04, F05) ----------------------
    bornes = bornes_split(len(hauteur))
    ancres = {s: ancres_valides(hauteur, a, b, args.entree, args.horizon, args.pas) for s, (a, b) in bornes.items()}
    possibles = {s: len(range(a + args.entree - 1, b - args.horizon, args.pas)) for s, (a, b) in bornes.items()}
    a_train, b_train = bornes["train"]
    h_train = hauteur[a_train:b_train]
    normalisation = {"moyenne_m": round(float(np.nanmean(h_train)), 5),
                     "ecart_type_m": round(float(np.nanstd(h_train)), 5),
                     "ajustee_sur": "période d'entraînement uniquement"}

    repartition = {}
    for s, (a, b) in bornes.items():
        repartition[s] = {
            "debut": iso(temps[a]), "fin": iso(temps[b - 1]), "heures": b - a,
            "part_pct": round(100 * (b - a) / len(hauteur), 2),
            "fenetres": int(len(ancres[s])),
            "fenetres_ecartees_lacunes": int(possibles[s] - len(ancres[s])),
        }
        print(f"  {s:5s} : {repartition[s]['debut'][:10]} → {repartition[s]['fin'][:10]}  "
              f"{repartition[s]['fenetres']:6d} fenêtres ({repartition[s]['fenetres_ecartees_lacunes']} écartées)")
    # Vérification d'absence de fuite : la dernière cible d'entraînement précède la 1re heure de validation.
    assert ancres["train"].max() + args.horizon < bornes["val"][0]
    assert ancres["val"].max() + args.horizon < bornes["test"][0]

    empreinte = hashlib.sha1()
    empreinte.update(np.nan_to_num(hauteur, nan=-999).tobytes())
    empreinte.update(f"{args.entree}-{args.horizon}-{args.pas}-{FRACTIONS}".encode())
    identifiant = empreinte.hexdigest()[:10]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET_DIR / "serie.npz", temps=temps.astype("datetime64[h]").astype(np.int64),
                        hauteur=hauteur, **{f"ancres_{s}": a for s, a in ancres.items()})
    ecrire_json(DATASET_DIR / "dataset.json", {
        "identifiant": identifiant,
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "station": STATION_NOM,
        "source": provenance.get("source"),
        "reference_verticale": provenance.get("reference_verticale"),
        "unite": "m", "horodatage": "UTC", "pas_temporel_h": 1,
        "parametres": {"entree_h": args.entree, "horizon_h": args.horizon, "pas_fenetres_h": args.pas,
                       "fractions": FRACTIONS},
        "convention": "ancre t : entrée h[t-entree+1..t], cible h[t+1..t+horizon] ; "
                      "fenêtre écartée si elle touche une lacune ou deux ensembles",
        "repartition": repartition,
        "normalisation": normalisation,
        "statistiques": stats,
        "fichiers": {"serie": "serie.npz (temps en heures depuis 1970 UTC, hauteur NaN = lacune, ancres_*)"},
    })
    rapport["date_creation"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ecrire_json(QC_DIR / "rapport_qc.json", rapport)
    produits = figures(temps, hauteur, bornes, stats)

    print(f"Dataset {identifiant} : {DATASET_DIR}")
    print(f"Rapport qualité : {QC_DIR / 'rapport_qc.json'}")
    print(f"Figures : {', '.join(produits)} dans {FIGURES_DIR}")
    print("Étape suivante :\n  uv run python TP5/train.py")


if __name__ == "__main__":
    main()
