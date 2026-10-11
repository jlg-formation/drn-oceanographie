"""Tests unitaires et de données du TP6.

Usage :
    uv run pytest TP6/tests -v

Les tests sur les données et les modèles supposent que les étapes précédentes
ont été lancées (téléchargement, préparation, entraînement, export) ; ils sont
sautés sinon.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from modeles import (
    VAE,
    Decodeur,
    Discriminateur,
    EncodeurExport,
    GenerateurExport,
    augmenter,
    perte_vae,
)
from terrain import (
    DATASET_DIR,
    ECHELLE_M,
    FRACTIONS,
    MODEL_DIR,
    PATCH,
    REGIONS,
    denormaliser,
    distance_w1,
    diversite,
    facteur_lon,
    indicateurs,
    lire_json,
    lot_latent,
    moyenne_blocs,
    mulberry32,
    normaliser,
    pentes_deg,
    vecteur_latent,
)

# --------------------------------------------------------------------------
# Graines : mêmes vecteurs latents qu'en JavaScript (F11, F17, F28)
# --------------------------------------------------------------------------


def test_mulberry32_identique_au_site():
    """Valeurs obtenues avec la fonction ``mulberry32`` de front-end/src/modeles.ts (bun)."""
    u = mulberry32(42)
    assert [u(), u(), u()] == [0.6011037519201636, 0.44829055899754167, 0.8524657934904099]


def test_vecteur_latent_identique_au_site():
    attendu = [-1.2848381996154785, 0.43278026580810547, -0.9453527927398682, -1.7128043174743652,
               -0.6112802624702454]
    assert vecteur_latent(42, 5).tolist() == attendu
    assert vecteur_latent(123456, 3).tolist() == [0.28724291920661926, -0.9386709928512573, 1.803175926208496]


def test_vecteur_latent_reproductible_et_gaussien():
    assert np.array_equal(vecteur_latent(7, 16), vecteur_latent(7, 16))
    assert not np.array_equal(vecteur_latent(7, 16), vecteur_latent(8, 16))
    z = lot_latent(0, 2000, 16)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    assert np.array_equal(z[5], vecteur_latent(5, 16))


# --------------------------------------------------------------------------
# Grille, normalisation, indicateurs (F03, F04, F22)
# --------------------------------------------------------------------------


def test_facteur_lon_pixels_carres():
    assert facteur_lon(0) == 4
    assert facteur_lon(45) == 6  # 4 / cos 45° ≈ 5,66
    assert facteur_lon(60) == 8


def test_moyenne_blocs():
    e = np.arange(24, dtype=np.float32).reshape(4, 6)
    m = moyenne_blocs(e, 2, 3)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(e[:2, :3].mean())
    assert np.isnan(moyenne_blocs(np.array([[1, np.nan], [1, 1]], dtype=np.float32), 2, 2)[0, 0])


def test_normalisation_centree_ecretee_et_reversible():
    rng = np.random.default_rng(0)
    m = -4000 + rng.normal(0, 300, (3, PATCH, PATCH)).astype(np.float32)
    z = normaliser(m)
    assert np.allclose(z.mean(axis=(1, 2)), 0, atol=1e-5)
    assert np.allclose(denormaliser(z), m - m.mean(axis=(1, 2), keepdims=True), atol=0.01)
    # Indépendante de la profondeur absolue : même relief, autre profondeur moyenne.
    assert np.allclose(normaliser(m + 1234), z, atol=1e-6)
    pic = np.zeros((PATCH, PATCH), dtype=np.float32)
    pic[0, 0] = 10 * ECHELLE_M
    assert normaliser(pic).max() == 1.0


def test_indicateurs_plan_et_constante():
    plat = np.full((PATCH, PATCH), -3000.0)
    i = indicateurs(plat, 1.85)
    assert all(float(v[0]) == pytest.approx(0, abs=1e-9) for v in i.values())
    # Plan incliné : 18,5 m par pixel de 1,85 km = pente de 1 % (0,573°).
    y = np.arange(PATCH)[:, None] * 18.5 * np.ones((1, PATCH))
    assert pentes_deg(y, 1.85).mean() == pytest.approx(math.degrees(math.atan(0.01)), rel=1e-6)
    assert indicateurs(y, 1.85)["amplitude_m"][0] == pytest.approx(63 * 18.5)


def test_distance_w1_et_diversite():
    a = np.random.default_rng(0).normal(0, 1, 5000)
    assert distance_w1(a, a) == 0
    assert distance_w1(a, a + 2) == pytest.approx(2, rel=1e-6)
    lot = np.zeros((4, 8, 8))
    assert diversite(lot) == 0
    lot[1] += 1
    assert diversite(lot) == pytest.approx(0.5)  # 3 paires à distance 1 sur 6


# --------------------------------------------------------------------------
# Modèles (F07, F13)
# --------------------------------------------------------------------------


def test_vae_formes_et_perte():
    torch.manual_seed(0)
    vae = VAE(16, 8)
    x = torch.rand(4, 1, PATCH, PATCH) * 2 - 1
    rec, mu, logvar = vae(x)
    assert rec.shape == x.shape and mu.shape == (4, 16) and logvar.shape == (4, 16)
    assert rec.abs().max() <= 1
    _, r, kl = perte_vae(x, x, torch.zeros(4, 16), torch.zeros(4, 16), 1.0)
    assert float(r) == 0 and float(kl) == 0
    _, _, kl = perte_vae(x, x, torch.ones(4, 16), torch.zeros(4, 16), 1.0)
    assert float(kl) == pytest.approx(8.0)  # ½ Σ μ² sur 16 dimensions


def test_generateur_et_discriminateur():
    g = Decodeur(32, 8).eval()
    d = Discriminateur(8)
    cartes = g(torch.randn(3, 32))
    assert cartes.shape == (3, 1, PATCH, PATCH)
    assert d(cartes).shape == (3,)


def test_augmentation_conserve_les_valeurs():
    x = torch.arange(2 * PATCH * PATCH, dtype=torch.float32).reshape(2, 1, PATCH, PATCH)
    a = augmenter(x, torch.Generator().manual_seed(0))
    for k in range(2):
        assert torch.equal(a[k].flatten().sort().values, x[k].flatten().sort().values)


def test_graphes_export_en_metres():
    torch.manual_seed(0)
    vae = VAE(16, 8).eval()
    enc = EncodeurExport(vae, ECHELLE_M).eval()
    e = -4000 + 300 * torch.randn(2, PATCH, PATCH)
    with torch.no_grad():
        mu1, _ = enc(e)
        mu2, _ = enc(e - 1500)  # centrage dans le graphe : profondeur absolue sans effet
    assert torch.allclose(mu1, mu2, atol=1e-4)
    gen = GenerateurExport(vae.decodeur, ECHELLE_M).eval()
    with torch.no_grad():
        r = gen(torch.zeros(1, 16))
    assert r.shape == (1, PATCH, PATCH) and float(r.abs().max()) <= ECHELLE_M


# --------------------------------------------------------------------------
# Dataset (F02 à F05) — si préparé
# --------------------------------------------------------------------------

dataset_absent = not (DATASET_DIR / "patches.npz").exists()


@pytest.fixture(scope="module")
def ds():
    npz = np.load(DATASET_DIR / "patches.npz")
    return {"m": {s: npz[f"{s}_m"] for s in FRACTIONS}, "index": lire_json(DATASET_DIR / "index.json"),
            "manifeste": lire_json(DATASET_DIR / "dataset.json")}


@pytest.mark.skipif(dataset_absent, reason="dataset non préparé (mise run tp6_02_prepare)")
class TestDataset:

    def test_formes_et_index(self, ds):
        for s in FRACTIONS:
            assert ds["m"][s].shape[1:] == (PATCH, PATCH)
            assert len(ds["index"][s]) == len(ds["m"][s]) == ds["manifeste"]["repartition"][s]
            assert np.isfinite(ds["m"][s]).all()

    def test_taille_et_repartition(self, ds):
        total = sum(len(ds["m"][s]) for s in FRACTIONS)
        assert 4000 <= total <= 6000  # « environ 5 000 patches » (F05)
        for s, f in FRACTIONS.items():
            assert abs(len(ds["m"][s]) / total - f) < 0.05

    def test_aucune_terre(self, ds):
        """Moyenne par blocs : tout pixel marin a une élévation moyenne négative (F02)."""
        for s in FRACTIONS:
            assert (ds["m"][s] < 0).all()

    def test_diversite_pas_que_des_plats(self, ds):
        ecarts = [p["ecart_type_m"] for p in ds["index"]["train"]]
        assert np.mean(np.array(ecarts) < 30) <= 0.11
        regions = {p["region"] for p in ds["index"]["train"]}
        assert regions == {r["id"] for r in REGIONS}

    def test_separation_spatiale(self, ds):
        """Aucun patch de validation ou de test ne recouvre un patch d'un autre ensemble (F05)."""
        par_region: dict[str, dict[str, np.ndarray]] = {}
        for s in FRACTIONS:
            for p in ds["index"][s]:
                par_region.setdefault(p["region"], {}).setdefault(s, []).append((p["ligne"], p["colonne"]))
        for region, ens in par_region.items():
            for a in FRACTIONS:
                for b in FRACTIONS:
                    if a >= b or a not in ens or b not in ens:
                        continue
                    pa, pb = np.array(ens[a]), np.array(ens[b])
                    dy = np.abs(pa[:, None, 0] - pb[None, :, 0])
                    dx = np.abs(pa[:, None, 1] - pb[None, :, 1])
                    assert not ((dy < PATCH) & (dx < PATCH)).any(), f"{region} : {a} recouvre {b}"

    def test_normalisation_conservee(self, ds):
        n = ds["manifeste"]["normalisation"]
        assert n["echelle_m"] == ECHELLE_M
        assert n["part_ecretee_train"] < 0.05

    def test_provenance_des_patches(self, ds):
        p = ds["index"]["test"][0]
        assert p["nord"] > p["sud"] and p["est"] > p["ouest"]
        assert p["profondeur_moy_m"] == pytest.approx(-float(ds["m"]["test"][0].mean()), abs=0.1)


# --------------------------------------------------------------------------
# Export ONNX et données du site (F24 à F26) — si exportés
# --------------------------------------------------------------------------

export_absent = not (MODEL_DIR / "metadata.json").exists()


@pytest.fixture(scope="module")
def meta():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(export_absent, reason="modèles non exportés (mise run tp6_07_export)")
class TestExport:

    def test_verifications_conformes(self, meta):
        for info in meta["modeles"].values():
            for v in info["verification_onnx"].values():
                assert v["conforme"], v

    def test_controle_recalculable(self, meta):
        import onnxruntime as ort

        for m, c in meta["controle"].items():
            f = meta["modeles"][m]["fichiers"]
            fichier = f.get("decodeur") or f.get("generateur")
            session = ort.InferenceSession(str(MODEL_DIR / fichier), providers=["CPUExecutionProvider"])
            z = vecteur_latent(c["graine"], meta["modeles"][m]["dim_latente"])
            assert np.allclose(z, c["z"], atol=1e-6)
            relief = session.run(None, {"z": z[None]})[0][0].reshape(-1)[::97]
            assert np.allclose(relief, c["relief_echantillon"], atol=0.05)

    def test_reels_bin(self, meta):
        brut = np.frombuffer((MODEL_DIR / "reels.bin").read_bytes(), dtype="<i2")
        assert brut.size == len(meta["reels"]["patches"]) * PATCH * PATCH
        assert (brut < 0).all()

    def test_avertissement_synthetique(self, meta):
        assert "SYNTHÉTIQUES" in meta["avertissement"]
