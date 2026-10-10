"""Tests unitaires et de données du TP4.

Usage :
    uv run pytest TP4/tests -v

Les tests sur les données supposent que les étapes précédentes ont été lancées
(téléchargement, dataset, export) ; ils sont sautés sinon.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box

from terre_mer import (
    DATASET_DIR,
    IGNORE,
    MER,
    MODEL_DIR,
    PATCH,
    TERRE,
    ZONES,
    Zone,
    ajouter_bande_ignoree,
    chemin_masque,
    chemin_scene,
    lire_json,
    lonlat_vers_pixel,
    pixel_vers_lonlat,
    rasteriser_terre,
    zone,
)
from unet import (
    UNet,
    indices_miroir,
    matrice_confusion,
    metriques,
    perte_segmentation,
    poids_fenetre,
    positions,
    predire_image,
)

# Points de contrôle (lat, lon, attendu) dans la zone « rade_brest ».
CONTROLES = [
    (48.330, -4.420, MER),  # milieu de la rade de Brest
    (48.395, -4.490, TERRE),  # Brest, centre-ville
    (48.230, -4.320, TERRE),  # presqu'île de Crozon (intérieur)
]


# --------------------------------------------------------------------------
# Géométrie
# --------------------------------------------------------------------------

def test_projection_aller_retour():
    lon, lat = np.array([-4.5, 0.0, 8.7]), np.array([48.3, 0.0, 41.9])
    x, y = lonlat_vers_pixel(lon, lat)
    lon2, lat2 = pixel_vers_lonlat(x, y)
    assert np.allclose(lon, lon2) and np.allclose(lat, lat2)


def test_zones_identifiants_uniques_et_splits():
    ids = [z.id for z in ZONES]
    assert len(ids) == len(set(ids))
    assert {z.split for z in ZONES} == {"train", "val", "test", "exemple"}


def test_zones_disjointes_et_separees_entre_splits():
    """F04 : pas de recouvrement, et au moins une tuile d'écart entre splits différents."""

    def ecart(a: Zone, b: Zone) -> int:
        gx = max(b.tx0 - (a.tx0 + a.tuiles), a.tx0 - (b.tx0 + b.tuiles))
        gy = max(b.ty0 - (a.ty0 + a.tuiles), a.ty0 - (b.ty0 + b.tuiles))
        return max(gx, gy)

    for i, a in enumerate(ZONES):
        for b in ZONES[i + 1:]:
            assert ecart(a, b) >= 0, f"{a.id} chevauche {b.id}"
            if a.split != b.split:
                assert ecart(a, b) >= 1, f"{a.id} ({a.split}) touche {b.id} ({b.split})"


def _zone_essai() -> Zone:
    return Zone("essai", "essai", -4.0, 48.0, "train", "", tuiles=2)


def test_rasterisation_alignee():
    """Un carré de terre est rasterisé exactement aux pixels attendus."""
    z = _zone_essai()
    x0, y0 = z.origine_px
    # Carré de pixels [100, 300) x [150, 350) converti en WGS84.
    w, n = pixel_vers_lonlat(x0 + 100, y0 + 150)
    e, s = pixel_vers_lonlat(x0 + 300, y0 + 350)
    m = rasteriser_terre(box(float(w), float(s), float(e), float(n)), z)
    assert m.shape == (z.taille_px, z.taille_px)
    assert m[250, 200] == TERRE and m[10, 10] == MER
    lignes, colonnes = np.nonzero(m)
    assert abs(lignes.min() - 150) <= 1 and abs(lignes.max() - 349) <= 1
    assert abs(colonnes.min() - 100) <= 1 and abs(colonnes.max() - 299) <= 1


def test_rasterisation_trou_et_ilot():
    """Un trou (lagune) est de la mer ; un îlot dans ce trou redevient de la terre."""
    z = _zone_essai()
    x0, y0 = z.origine_px

    def carre(a: int, b: int) -> list[tuple[float, float]]:
        coins = [(a, a), (b, a), (b, b), (a, b)]
        return [tuple(float(v) for v in pixel_vers_lonlat(x0 + px, y0 + py)) for px, py in coins]

    terre = Polygon(carre(50, 450), holes=[carre(150, 350)])
    ilot = Polygon(carre(220, 280))
    m = rasteriser_terre(MultiPolygon([ilot, terre]), z)
    assert m[100, 100] == TERRE and m[170, 170] == MER and m[250, 250] == TERRE


def test_bande_ignoree():
    m = np.zeros((40, 40), dtype=np.uint8)
    m[:, 20:] = TERRE
    out = ajouter_bande_ignoree(m, marge=2)
    assert set(np.unique(out)) == {MER, TERRE, IGNORE}
    assert np.all(out[:, 18:22] == IGNORE)
    assert np.all(out[:, :18] == MER) and np.all(out[:, 22:] == TERRE)


# --------------------------------------------------------------------------
# Métriques, perte, modèle
# --------------------------------------------------------------------------

def test_metriques_connues():
    # Référence : 6 mer, 4 terre ; prédiction : 1 fausse terre, 1 fausse mer.
    m = metriques(np.array([[5, 1], [1, 3]]))
    assert m["terre"]["iou"] == pytest.approx(3 / 5)
    assert m["mer"]["iou"] == pytest.approx(5 / 7)
    assert m["terre"]["precision"] == pytest.approx(3 / 4)
    assert m["terre"]["rappel"] == pytest.approx(3 / 4)
    assert m["terre"]["dice"] == pytest.approx(6 / 8)
    assert m["exactitude"] == pytest.approx(0.8)


def test_matrice_confusion_ignore_255():
    cible = np.array([[0, 1, 255], [1, 0, 255]], dtype=np.uint8)
    pred = np.array([[0, 1, 1], [0, 0, 0]])
    assert matrice_confusion(pred, cible).tolist() == [[2, 0], [1, 1]]


def test_perte_insensible_aux_pixels_ignores():
    torch.manual_seed(0)
    cible = torch.randint(0, 2, (2, 32, 32), dtype=torch.uint8)
    cible[:, :8] = IGNORE
    logits = torch.randn(2, 1, 32, 32)
    autre = logits.clone()
    autre[:, :, :8] = 50.0  # valeurs extrêmes sur les seuls pixels ignorés
    assert perte_segmentation(logits, cible).item() == pytest.approx(perte_segmentation(autre, cible).item())


def test_unet_forme_sortie():
    modele = UNet(base=4, profondeur=4).eval()
    with torch.no_grad():
        assert modele(torch.zeros(2, 3, PATCH, PATCH)).shape == (2, 1, PATCH, PATCH)


# --------------------------------------------------------------------------
# Inférence par fenêtres (F17)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("longueur", [1, 100, 256, 257, 500, 1024, 1999])
def test_positions_couvrent_tout(longueur):
    debuts = positions(longueur)
    couvert = np.zeros(max(longueur, PATCH), dtype=bool)
    for d in debuts:
        couvert[d:d + PATCH] = True
    assert couvert[:longueur].all()
    assert debuts[-1] == max(0, longueur - PATCH)


def test_indices_miroir():
    assert indices_miroir(7, 3).tolist() == [0, 1, 2, 1, 0, 1, 2]
    assert indices_miroir(3, 1).tolist() == [0, 0, 0]


def test_poids_fenetre_positifs():
    w = poids_fenetre()
    assert w.shape == (PATCH, PATCH) and w.min() > 0 and w.max() == 1.0


@pytest.mark.parametrize("forme", [(100, 180), (256, 256), (300, 700), (777, 513)])
def test_predire_image_reconstruction_exacte(forme):
    """Un « modèle » qui renvoie le canal rouge doit redonner exactement l'image :
    aucune zone non traitée, aucune couture, dimensions conservées."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(*forme, 3), dtype=np.uint8)
    prob = predire_image(image, lambda lot: lot[:, :1])
    assert prob.shape == forme
    assert np.allclose(prob, image[..., 0] / 255.0, atol=1e-6)


# --------------------------------------------------------------------------
# Données préparées (sautés si absentes)
# --------------------------------------------------------------------------

donnees = pytest.mark.skipif(not (DATASET_DIR / "dataset.json").exists(),
                             reason="dataset absent : lancer `mise run tp4_02_dataset`")


@donnees
@pytest.mark.parametrize("z", ZONES, ids=lambda z: z.id)
def test_masque_aligne_et_valeurs(z):
    image = Image.open(chemin_scene(z.id))
    masque = np.asarray(Image.open(chemin_masque(z.id)))
    assert masque.shape == (image.height, image.width)
    assert set(np.unique(masque).tolist()) <= {MER, TERRE, IGNORE}


@donnees
@pytest.mark.parametrize("lat, lon, attendu", CONTROLES)
def test_points_de_controle(lat, lon, attendu):
    z = zone("rade_brest")
    x, y = lonlat_vers_pixel(lon, lat)
    x0, y0 = z.origine_px
    masque = np.asarray(Image.open(chemin_masque(z.id)))
    assert masque[int(y - y0), int(x - x0)] == attendu


@donnees
def test_manifeste_splits_par_zone():
    manifeste = lire_json(DATASET_DIR / "dataset.json")
    zones_par_split: dict[str, set] = {}
    for p in manifeste["patches"]:
        zones_par_split.setdefault(p["split"], set()).add(p["zone"])
        assert (DATASET_DIR / p["split"] / "images" / p["fichier"]).exists()
        assert (DATASET_DIR / p["split"] / "masks" / p["fichier"]).exists()
    splits = list(zones_par_split.values())
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not (a & b), "une zone apparaît dans deux ensembles"


onnx_present = pytest.mark.skipif(not (MODEL_DIR / "metadata.json").exists(),
                                  reason="modèle ONNX absent : lancer `mise run tp4_05_export`")


@onnx_present
def test_onnx_coherent_avec_metadata():
    import onnxruntime as ort

    meta = lire_json(MODEL_DIR / "metadata.json")
    session = ort.InferenceSession(str(MODEL_DIR / meta["modele"]["fichier"]), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((3, 3, PATCH, PATCH), dtype=np.float32)
    sortie = session.run(None, {meta["modele"]["entree"]["nom"]: x})[0]
    assert sortie.shape == (3, 1, PATCH, PATCH)
    assert sortie.min() >= 0 and sortie.max() <= 1
    assert meta["verification_onnx"]["ecart_max_probabilite"] < 1e-3


@donnees
def test_exemples_web():
    exemples = lire_json(MODEL_DIR / "examples.json")
    assert len(exemples) >= 3
    assert any(e["origine"] == "hors zone d'apprentissage" for e in exemples)
    for e in exemples:
        assert (MODEL_DIR / e["fichier"]).exists()
        assert not e["vue_a_l_entrainement"]
