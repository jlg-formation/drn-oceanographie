"""U-Net compact, perte, métriques et inférence par tuiles du TP4.

Partagé par ``train_unet.py``, ``evaluate.py``, ``export_onnx.py`` et les
tests. L'algorithme de découpage/recollement (:func:`predire_image`) est
reproduit à l'identique côté web (``front-end/src/tiling.ts``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from terre_mer import IGNORE, PATCH, TERRE

# Pas entre deux fenêtres lors de l'inférence d'une image entière : 64 px de
# recouvrement, fondus progressivement pour éviter les coutures (F17).
PAS_INFERENCE = 192


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2 : la brique de base du U-Net."""

    def __init__(self, entree: int, sortie: int) -> None:
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(entree, sortie, 3, padding=1, bias=False),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
            nn.Conv2d(sortie, sortie, 3, padding=1, bias=False),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloc(x)


class UNet(nn.Module):
    """U-Net compact : encodeur / décodeur symétriques reliés par des « skip connections ».

    - L'**encodeur** divise la résolution par 2 à chaque niveau (max-pooling)
      et double le nombre de canaux : il capte le contexte (« c'est une baie »).
    - Le **décodeur** remonte à la pleine résolution (convolution transposée)
      en concaténant les cartes de l'encodeur de même taille : il retrouve la
      précision spatiale (« le trait de côte passe exactement ici »).
    - Une convolution 1x1 finale donne un logit par pixel : P(terre) = sigmoid(logit).

    Avec ``profondeur`` niveaux, H et W doivent être multiples de 2**profondeur.
    """

    def __init__(self, base: int = 16, profondeur: int = 4) -> None:
        super().__init__()
        canaux = [base * 2**i for i in range(profondeur + 1)]
        self.encodeurs = nn.ModuleList()
        entree = 3
        for c in canaux[:-1]:
            self.encodeurs.append(DoubleConv(entree, c))
            entree = c
        self.pool = nn.MaxPool2d(2)
        self.fond = DoubleConv(canaux[-2], canaux[-1])
        self.montees = nn.ModuleList()
        self.decodeurs = nn.ModuleList()
        for c in reversed(canaux[:-1]):
            self.montees.append(nn.ConvTranspose2d(2 * c, c, 2, stride=2))
            self.decodeurs.append(DoubleConv(2 * c, c))
        self.tete = nn.Conv2d(canaux[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sauts = []
        for enc in self.encodeurs:
            x = enc(x)
            sauts.append(x)
            x = self.pool(x)
        x = self.fond(x)
        for montee, dec, saut in zip(self.montees, self.decodeurs, reversed(sauts)):
            x = dec(torch.cat([montee(x), saut], dim=1))
        return self.tete(x)  # logits (N, 1, H, W)


class ModeleExport(nn.Module):
    """Modèle « prêt à l'emploi » exporté en ONNX : normalisation + U-Net + sigmoïde.

    Entrée : image RGB float32 (N, 3, H, W) dans [0, 1].
    Sortie : probabilité d'être de la terre (N, 1, H, W) dans [0, 1].
    Toute la normalisation est dans le graphe : le site n'a qu'à diviser par 255.
    """

    def __init__(self, unet: UNet, moyenne: list[float], ecart_type: list[float]) -> None:
        super().__init__()
        self.unet = unet
        self.register_buffer("moyenne", torch.tensor(moyenne, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("ecart_type", torch.tensor(ecart_type, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.unet((image - self.moyenne) / self.ecart_type))


def construire_modele_export(checkpoint: dict) -> ModeleExport:
    """Recrée le modèle complet à partir d'un checkpoint ``best.pt``."""
    cfg = checkpoint["config"]
    unet = UNet(base=cfg["base"], profondeur=cfg["profondeur"])
    unet.load_state_dict(checkpoint["state_dict"])
    modele = ModeleExport(unet, checkpoint["normalisation"]["moyenne"], checkpoint["normalisation"]["ecart_type"])
    return modele.eval()


# --------------------------------------------------------------------------
# Perte : les pixels 255 n'interviennent jamais
# --------------------------------------------------------------------------

def perte_segmentation(logits: torch.Tensor, cible: torch.Tensor) -> torch.Tensor:
    """BCE + Dice, calculées uniquement sur les pixels valides.

    ``cible`` : (N, H, W) uint8 en 0/1/255. La BCE apprend pixel par pixel ; le
    Dice compense le déséquilibre entre classes (un patch à 95 % de mer).
    """
    valides = (cible != IGNORE).float()
    terre = (cible == TERRE).float()
    logits = logits[:, 0]
    n = valides.sum().clamp(min=1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, terre, reduction="none") * valides).sum() / n
    p = torch.sigmoid(logits) * valides
    dice = 1 - (2 * (p * terre).sum() + 1) / (p.sum() + (terre * valides).sum() + 1)
    return bce + dice


# --------------------------------------------------------------------------
# Métriques (F11)
# --------------------------------------------------------------------------

def matrice_confusion(pred_terre: np.ndarray | torch.Tensor, cible: np.ndarray | torch.Tensor) -> np.ndarray:
    """Matrice 2x2 ``cm[référence, prédiction]`` (0 = mer, 1 = terre), pixels 255 exclus."""
    if isinstance(pred_terre, torch.Tensor):
        pred_terre = pred_terre.detach().cpu().numpy()
    if isinstance(cible, torch.Tensor):
        cible = cible.detach().cpu().numpy()
    valides = cible != IGNORE
    ref = cible[valides].astype(np.int64)
    pred = np.asarray(pred_terre)[valides].astype(np.int64)
    return np.bincount(2 * ref + pred, minlength=4).reshape(2, 2)


def metriques(cm: np.ndarray) -> dict:
    """IoU, Dice, précision et rappel pour chaque classe, + moyennes et exactitude."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    resultat: dict = {}
    for k, nom in ((0, "mer"), (1, "terre")):
        vp = cm[k, k]
        fp = cm[:, k].sum() - vp  # prédit k à tort
        fn = cm[k, :].sum() - vp  # k manqué
        div = lambda a, b: float(a / b) if b > 0 else float("nan")  # noqa: E731
        resultat[nom] = {
            "iou": div(vp, vp + fp + fn),
            "dice": div(2 * vp, 2 * vp + fp + fn),
            "precision": div(vp, vp + fp),
            "rappel": div(vp, vp + fn),
            "pixels_reference": int(cm[k, :].sum()),
        }
    resultat["miou"] = float(np.nanmean([resultat["mer"]["iou"], resultat["terre"]["iou"]]))
    resultat["dice_moyen"] = float(np.nanmean([resultat["mer"]["dice"], resultat["terre"]["dice"]]))
    resultat["exactitude"] = float(np.trace(cm) / total) if total else float("nan")
    resultat["matrice_confusion"] = {
        "lignes": "référence (mer, terre)",
        "colonnes": "prédiction (mer, terre)",
        "valeurs": cm.astype(np.int64).tolist(),
    }
    return resultat


# --------------------------------------------------------------------------
# Inférence d'une image entière par fenêtres glissantes (F17)
# --------------------------------------------------------------------------

def positions(longueur: int, taille: int = PATCH, pas: int = PAS_INFERENCE) -> list[int]:
    """Débuts des fenêtres le long d'un axe ; la dernière est collée au bord.

    Garantit que chaque pixel est couvert au moins une fois. Si l'image est plus
    petite que la fenêtre, une seule fenêtre (complétée par effet miroir).
    """
    if longueur <= taille:
        return [0]
    debuts = list(range(0, longueur - taille, pas))
    debuts.append(longueur - taille)
    return debuts


def indices_miroir(n: int, longueur: int) -> np.ndarray:
    """Indices 0..n-1 repliés en miroir dans [0, longueur) (complétion des petites images)."""
    if longueur == 1:
        return np.zeros(n, dtype=np.int64)
    periode = 2 * (longueur - 1)
    i = np.arange(n) % periode
    return np.where(i < longueur, i, periode - i)


def poids_fenetre(taille: int = PATCH, rampe: int = (PATCH - PAS_INFERENCE) // 2) -> np.ndarray:
    """Pondération 2D en « tente aplatie » : 1 au centre, décroît vers les bords.

    Les bords d'une fenêtre, où le CNN voit moins de contexte, comptent moins
    dans la moyenne pondérée : les coutures entre fenêtres disparaissent.
    """
    d = np.minimum(np.arange(taille), np.arange(taille)[::-1]) + 1
    w = np.clip(d / max(rampe, 1), 1 / max(rampe, 1), 1.0)
    return np.outer(w, w).astype(np.float32)


def predire_image(
    image: np.ndarray,
    predire_lot: Callable[[np.ndarray], np.ndarray],
    taille: int = PATCH,
    pas: int = PAS_INFERENCE,
    lot: int = 8,
) -> np.ndarray:
    """Probabilité de terre (H, W) pour une image RGB uint8 (H, W, 3) de taille quelconque.

    ``predire_lot`` reçoit un tableau float32 (B, 3, taille, taille) dans [0, 1]
    et renvoie des probabilités (B, 1, taille, taille).
    """
    h, w = image.shape[:2]
    hp, wp = max(h, taille), max(w, taille)
    # Complétion miroir si l'image est plus petite qu'une fenêtre.
    if (hp, wp) != (h, w):
        image = image[indices_miroir(hp, h)][:, indices_miroir(wp, w)]
    x = image.astype(np.float32).transpose(2, 0, 1) / 255.0

    poids = poids_fenetre(taille)
    somme = np.zeros((hp, wp), dtype=np.float32)
    somme_poids = np.zeros((hp, wp), dtype=np.float32)
    fenetres = [(r, c) for r in positions(hp, taille, pas) for c in positions(wp, taille, pas)]
    for i in range(0, len(fenetres), lot):
        groupe = fenetres[i:i + lot]
        entree = np.stack([x[:, r:r + taille, c:c + taille] for r, c in groupe])
        prob = predire_lot(entree)[:, 0]
        for (r, c), p in zip(groupe, prob):
            somme[r:r + taille, c:c + taille] += p * poids
            somme_poids[r:r + taille, c:c + taille] += poids
    return (somme / somme_poids)[:h, :w]


def predicteur_torch(modele: ModeleExport, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Adapte un :class:`ModeleExport` PyTorch à la signature de ``predire_lot``."""

    @torch.no_grad()
    def predire_lot(lot: np.ndarray) -> np.ndarray:
        return modele(torch.from_numpy(lot).to(device)).cpu().numpy()

    return predire_lot
