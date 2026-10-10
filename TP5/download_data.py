"""Acquisition des observations REFMAR du marégraphe de Brest (TP5, étape 1 : F01).

Interroge l'API publique du SHOM (sans compte ni clé) mois par mois : l'API
limite chaque réponse à environ un mois de mesures horaires. Les réponses
brutes sont conservées telles quelles dans ``data/raw/`` (cache : un mois déjà
téléchargé n'est pas redemandé), puis assemblées sans aucune modification des
valeurs dans :

  - ``data/brest_horaire.csv`` : ``horodatage_utc,hauteur_m,source`` ;
  - ``data/provenance.json``   : station, source, période, référence verticale,
    URL interrogées, date de téléchargement, journal de bord du marégraphe.

Usage :
    uv run python TP5/download_data.py
    uv run python TP5/download_data.py --debut 2022 --fin 2025 --force
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from maree import (
    ANNEE_DEBUT,
    ANNEE_FIN,
    API_OBSERVATIONS,
    API_STATION,
    ATTRIBUTION,
    DATA_DIR,
    OBS_CSV,
    PORTAIL,
    PROVENANCE_JSON,
    RAW_DIR,
    SOURCE_ID,
    SOURCE_LIBELLE,
    STATION_ID,
    STATION_NOM,
    ecrire_json,
    lire_json,
)

ENTETES = {"User-Agent": "drn-oceanographie-TP5/1.0 (formation SHOM)"}


def telecharger_json(url: str, essais: int = 4) -> dict:
    """GET JSON avec quelques nouvelles tentatives (réseau capricieux)."""
    for essai in range(1, essais + 1):
        try:
            requete = urllib.request.Request(url, headers=ENTETES)
            with urllib.request.urlopen(requete, timeout=60) as reponse:
                return json.loads(reponse.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            if essai == essais:
                raise SystemExit(f"Échec du téléchargement après {essais} essais : {url}\n  {err}") from err
            attente = 2 * essai
            print(f"    erreur ({err}), nouvel essai dans {attente} s")
            time.sleep(attente)
    raise AssertionError("inaccessible")


def url_mois(annee: int, mois: int) -> str:
    dernier = calendar.monthrange(annee, mois)[1]
    return (API_OBSERVATIONS.format(station=STATION_ID)
            + f"?sources={SOURCE_ID}&dtStart={annee}-{mois:02d}-01T00:00:00Z"
            + f"&dtEnd={annee}-{mois:02d}-{dernier}T23:59:59Z")


def telecharger_mois(annee: int, mois: int, force: bool) -> list[dict]:
    """Observations brutes d'un mois (depuis le cache si disponible)."""
    chemin = RAW_DIR / f"brest_{annee}-{mois:02d}.json"
    if chemin.exists() and not force:
        return lire_json(chemin)["reponse"]["data"]
    url = url_mois(annee, mois)
    reponse = telecharger_json(url)
    if "data" not in reponse:
        raise SystemExit(f"Réponse inattendue de l'API pour {annee}-{mois:02d} : {str(reponse)[:200]}")
    ecrire_json(chemin, {
        "url": url,
        "date_telechargement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reponse": reponse,
    })
    time.sleep(0.3)  # politesse envers le serveur
    return reponse["data"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debut", type=int, default=ANNEE_DEBUT, help="première année (incluse)")
    parser.add_argument("--fin", type=int, default=ANNEE_FIN, help="dernière année (incluse)")
    parser.add_argument("--force", action="store_true", help="retélécharger même les mois en cache")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Station : {STATION_NOM} (id SHOM {STATION_ID}) — source {SOURCE_ID} : {SOURCE_LIBELLE}")

    station = telecharger_json(API_STATION.format(station=STATION_ID))
    ecrire_json(RAW_DIR / "station.json", station)
    ref = station.get("verticalRef", {})
    print(f"Référence verticale : zéro hydrographique (ZH = {ref.get('zh_ref')} m / {ref.get('nom_ref')}), "
          f"fuseau horaire des données : UTC+{station.get('fuseau_horaire')}")

    lignes: list[tuple[str, float | None, int]] = []
    par_mois: dict[str, int] = {}
    for annee in range(args.debut, args.fin + 1):
        for mois in range(1, 13):
            donnees = telecharger_mois(annee, mois, args.force)
            par_mois[f"{annee}-{mois:02d}"] = len(donnees)
            for d in donnees:
                # Horodatage de l'API : « 2024/01/01 00:00:00 » en UTC.
                t = datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S")
                lignes.append((t.strftime("%Y-%m-%dT%H:%M:%SZ"), d.get("value"), int(d.get("idsource", SOURCE_ID))))
        print(f"  {annee} : {sum(v for k, v in par_mois.items() if k.startswith(str(annee)))} mesures")

    # L'API renvoie la borne de fin incluse : un horodatage peut apparaître dans
    # deux mois consécutifs. On ne déduplique PAS ici : le contrôle qualité
    # (prepare_dataset.py) identifie et traite explicitement les doublons.
    lignes.sort(key=lambda r: r[0])
    with OBS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["horodatage_utc", "hauteur_m", "source"])
        for t, v, s in lignes:
            w.writerow([t, "" if v is None else v, s])
    print(f"Observations écrites : {OBS_CSV} ({len(lignes)} lignes)")

    ecrire_json(PROVENANCE_JSON, {
        "station": {
            "nom": STATION_NOM,
            "id_shom": STATION_ID,
            "reseau": station.get("reseau"),
            "longitude": station.get("longitude"),
            "latitude": station.get("latitude"),
            "gestionnaire": station.get("gestionnaire"),
            "capteur": station.get("descriptif_capteur"),
            "fiche": station.get("info_maregraphe"),
            "premiere_observation": station.get("date_prem_obs"),
            "niveau_moyen_m": station.get("niveau_moyen"),
        },
        "source": {"id": SOURCE_ID, "libelle": SOURCE_LIBELLE, "portail": PORTAIL,
                   "api": API_OBSERVATIONS.format(station=STATION_ID), "attribution": ATTRIBUTION},
        "periode": {"debut": f"{args.debut}-01-01T00:00:00Z", "fin": f"{args.fin}-12-31T23:00:00Z"},
        "unite": "m",
        "horodatage": "UTC",
        "reference_verticale": {
            "nom": "zéro hydrographique (ZH)",
            "zh_par_rapport_a": ref.get("nom_ref"),
            "zh_m": ref.get("zh_ref"),
            "remarque": "hauteurs comptées positivement au-dessus du zéro hydrographique local",
        },
        "qualite": "données horaires validées par le SHOM (source 4 de l'API REFMAR)",
        "date_telechargement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mesures_par_mois": par_mois,
        "journal_de_bord": [
            {"date": e.get("due_date"), "texte": " ".join(str(e.get("content", "")).split())}
            for e in sorted(station.get("events", []), key=lambda e: e.get("due_date") or "")
        ],
    })
    print(f"Provenance écrite : {PROVENANCE_JSON}")
    print(f"Données dans {DATA_DIR}. Étape suivante :\n  uv run python TP5/prepare_dataset.py")


if __name__ == "__main__":
    main()
