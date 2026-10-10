"""Tests unitaires et de données du TP5.

Usage :
    uv run pytest TP5/tests -v

Les tests sur les données supposent que les étapes précédentes ont été lancées
(téléchargement, préparation, entraînement, export) ; ils sont sautés sinon.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from maree import (
    DATASET_DIR,
    ENTREE_H,
    HORIZON_H,
    MODEL_DIR,
    MODELES,
    QC_DIR,
    STYLES,
    ancres_valides,
    bornes_split,
    fenetres,
    lire_json,
    metriques,
    persistance,
    predire_references,
    repetition,
)
from modeles import ModeleExport, Prevision, construire_modele, predire

# --------------------------------------------------------------------------
# Fenêtres temporelles (F04)
# --------------------------------------------------------------------------


def test_fenetres_alignement():
    """Entrée h[t-47..t], cible h[t+1..t+24] : aucune valeur future dans l'entrée (NF07)."""
    h = np.arange(200, dtype=np.float32)
    x, y = fenetres(h, np.array([47, 100]))
    assert x.shape == (2, ENTREE_H) and y.shape == (2, HORIZON_H)
    assert x[0, 0] == 0 and x[0, -1] == 47 and y[0, 0] == 48 and y[0, -1] == 71
    assert x[1, -1] == 100 and y[1, 0] == 101
    assert np.all(x.max(axis=1) < y.min(axis=1))


def test_fenetres_parametrables():
    h = np.arange(100, dtype=np.float32)
    x, y = fenetres(h, np.array([20]), entree=10, horizon=5)
    assert x[0].tolist() == list(range(11, 21)) and y[0].tolist() == list(range(21, 26))


def test_ancres_ecartent_les_lacunes():
    h = np.ones(300, dtype=np.float32)
    h[150] = np.nan
    ancres = ancres_valides(h, 0, 300)
    assert ancres.min() == ENTREE_H - 1 and ancres.max() == 300 - HORIZON_H - 1
    # Aucune fenêtre (entrée + cible) ne touche l'heure 150.
    assert not np.any((ancres - ENTREE_H + 1 <= 150) & (ancres + HORIZON_H >= 150))
    assert 150 - HORIZON_H - 1 in ancres and 150 + ENTREE_H in ancres
    x, y = fenetres(h, ancres)
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_ancres_pas():
    h = np.ones(200, dtype=np.float32)
    assert np.all(np.diff(ancres_valides(h, 0, 200, pas=3)) == 3)


# --------------------------------------------------------------------------
# Séparation chronologique (F05)
# --------------------------------------------------------------------------


def test_bornes_chronologiques_et_contigues():
    b = bornes_split(43824)
    assert b["train"][0] == 0 and b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0]
    assert b["test"][1] == 43824
    assert all(v % 24 == 0 for v in (b["val"][0], b["test"][0]))
    assert 0.69 < b["train"][1] / 43824 < 0.71


def test_pas_de_cible_hors_de_son_ensemble():
    h = np.random.default_rng(0).normal(size=5000).astype(np.float32)
    b = bornes_split(len(h))
    for split, (a, f) in b.items():
        ancres = ancres_valides(h, a, f)
        assert ancres.min() - ENTREE_H + 1 >= a, split
        assert ancres.max() + HORIZON_H < f, split


# --------------------------------------------------------------------------
# Références naïves (F15) et métriques (F13, F14)
# --------------------------------------------------------------------------


def test_persistance():
    x = np.arange(48, dtype=np.float32)[None, :]
    assert np.all(persistance(x) == 47)


def test_repetition_utilise_seulement_le_passe():
    x = np.arange(48, dtype=np.float32)[None, :]  # x[i] = heure H-47+i
    # H+k-24 : H+1 -> H-23 (indice 24), H+24 -> H (indice 47)
    assert repetition(x, 24)[0].tolist() == list(range(24, 48))
    # H+k-25 : H+1 -> H-24 (indice 23), H+24 -> H-1 (indice 46)
    assert repetition(x, 25)[0].tolist() == list(range(23, 47))
    with pytest.raises(ValueError):
        repetition(x, 12)


def test_repetition_25h_parfaite_sur_un_signal_de_periode_25h():
    t = np.arange(100)
    h = np.sin(2 * np.pi * t / 25).astype(np.float32)
    x, y = fenetres(h, np.array([60]))
    refs = predire_references(x)
    assert np.allclose(refs["repetition_25h"], y, atol=1e-5)
    assert not np.allclose(refs["repetition_24h"], y, atol=1e-2)


def test_metriques():
    vrai = np.zeros((2, 24))
    prev = np.ones((2, 24))
    prev[:, 0] = 3.0
    m = metriques(prev, vrai)
    assert m["horizons"]["H+1"] == {"mae": 3.0, "rmse": 3.0}
    assert m["horizons"]["H+24"]["mae"] == 1.0
    assert m["global"]["mae"] == pytest.approx((3 + 23) / 24)
    assert m["global"]["rmse"] == pytest.approx(np.sqrt((9 + 23) / 24))
    assert len(m["par_pas"]["mae"]) == 24 and m["n_fenetres"] == 2


# --------------------------------------------------------------------------
# Modèles (F06 à F08, F11)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cellule", MODELES)
def test_modele_dimensions(cellule):
    modele = Prevision(cellule, cachee=8, couches=2)
    assert modele(torch.zeros(5, ENTREE_H)).shape == (5, HORIZON_H)
    # Le même réseau accepte une autre longueur d'entrée (horizons configurables).
    assert modele(torch.zeros(2, 72)).shape == (2, HORIZON_H)


def test_modele_cellule_inconnue():
    with pytest.raises(ValueError):
        Prevision("transformer")


def test_modele_export_normalisation():
    """Le graphe exporté reçoit et rend des mètres (normalisation incluse)."""
    torch.manual_seed(0)
    modele = Prevision("gru", cachee=8, couches=1).eval()
    export = ModeleExport(modele, moyenne=4.0, ecart_type=2.0).eval()
    x = np.random.default_rng(0).uniform(1, 7, size=(3, ENTREE_H)).astype(np.float32)
    with torch.no_grad():
        attendu = modele(torch.from_numpy((x - 4.0) / 2.0)).numpy() * 2.0 + 4.0
    assert np.allclose(predire(export, x), attendu, atol=1e-5)


def test_sauvegarde_rechargement(tmp_path):
    modele = Prevision("lstm", cachee=8, couches=1).eval()
    chemin = tmp_path / "best.pt"
    torch.save({"state_dict": modele.state_dict(),
                "config": {"type": "lstm", "cachee": 8, "couches": 1, "entree": 48, "horizon": 24}}, chemin)
    recharge = construire_modele(torch.load(chemin, weights_only=False))
    x = torch.randn(2, 48)
    with torch.no_grad():
        assert torch.allclose(modele(x), recharge(x))


def test_styles_distincts_sans_la_couleur():
    """§7.4 : chaque courbe de prévision a un style de trait ou un marqueur qui lui est propre."""
    courbes = [k for k in STYLES if k not in ("observe",)]
    signatures = {(tuple(STYLES[k]["tirets"]), STYLES[k]["marqueur"]) for k in courbes}
    assert len(signatures) == len(courbes)


# --------------------------------------------------------------------------
# Données préparées (sautés si les étapes précédentes n'ont pas été lancées)
# --------------------------------------------------------------------------

serie_absente = not (DATASET_DIR / "serie.npz").exists()


@pytest.mark.skipif(serie_absente, reason="lancez `mise run tp5_02_prepare`")
def test_serie_reguliere_et_lacunes_non_comblees():
    npz = np.load(DATASET_DIR / "serie.npz")
    temps, h = npz["temps"], npz["hauteur"]
    assert np.all(np.diff(temps) == 1), "grille horaire régulière"
    rapport = lire_json(QC_DIR / "rapport_qc.json")
    assert int(np.isnan(h).sum()) == rapport["grille"]["heures_manquantes"], "lacunes non comblées (NF06)"
    assert np.nanmin(h) > -1 and np.nanmax(h) < 10


@pytest.mark.skipif(serie_absente, reason="lancez `mise run tp5_02_prepare`")
def test_dataset_sans_fuite_temporelle():
    npz = np.load(DATASET_DIR / "serie.npz")
    h = npz["hauteur"]
    manifeste = lire_json(DATASET_DIR / "dataset.json")
    p = manifeste["parametres"]
    debut_val = npz["ancres_val"].min() - p["entree_h"] + 1
    debut_test = npz["ancres_test"].min() - p["entree_h"] + 1
    assert npz["ancres_train"].max() + p["horizon_h"] < debut_val
    assert npz["ancres_val"].max() + p["horizon_h"] < debut_test
    for s in ("train", "val", "test"):
        x, y = fenetres(h, npz[f"ancres_{s}"], p["entree_h"], p["horizon_h"])
        assert np.isfinite(x).all() and np.isfinite(y).all(), s
        assert len(x) == manifeste["repartition"][s]["fenetres"]
    # Normalisation ajustée sur l'entraînement seulement.
    fin_train = int(npz["ancres_train"].max()) + p["horizon_h"] + 1
    assert manifeste["normalisation"]["moyenne_m"] == pytest.approx(float(np.nanmean(h[:fin_train])), abs=0.01)


export_absent = not (MODEL_DIR / "metadata.json").exists()


@pytest.mark.skipif(export_absent, reason="lancez `mise run tp5_05_export`")
def test_onnx_reproduit_l_exemple_de_controle():
    import onnxruntime as ort

    meta = lire_json(MODEL_DIR / "metadata.json")
    x = np.array([meta["controle"]["entree"]], dtype=np.float32)
    for m, info in meta["modeles"].items():
        assert info["verification_onnx"]["conforme"], m
        session = ort.InferenceSession(str(MODEL_DIR / info["fichier"]), providers=["CPUExecutionProvider"])
        sortie = session.run(None, {meta["inference"]["entree"]: x})[0]
        assert sortie.shape == (1, meta["inference"]["horizon_h"])
        assert np.allclose(sortie[0], meta["controle"]["attendu"][m], atol=1e-3), m


@pytest.mark.skipif(export_absent, reason="lancez `mise run tp5_05_export`")
def test_exemples_du_site_dans_le_test_et_complets():
    meta = lire_json(MODEL_DIR / "metadata.json")
    serie = json.loads((MODEL_DIR / "serie.json").read_text(encoding="utf-8"))
    debut = np.datetime64(serie["debut"].rstrip("Z"), "h")
    test = serie["ensembles"]["test"]
    entree, horizon = meta["inference"]["entree_h"], meta["inference"]["horizon_h"]
    assert len(meta["exemples"]) >= 4
    for ex in meta["exemples"]:
        t = int((np.datetime64(ex["ancre"].rstrip("Z"), "h") - debut).astype(int))
        assert test["debut"] <= t - entree + 1 and t + horizon < test["fin"], ex["id"]
        assert all(v is not None for v in serie["valeurs"][t - entree + 1:t + horizon + 1]), ex["id"]
