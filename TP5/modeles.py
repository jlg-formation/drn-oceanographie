"""Réseaux récurrents de prévision : RNN simple, GRU et LSTM (TP5, F06 à F08).

Les trois modèles partagent **exactement** la même structure, seule la cellule
récurrente change. La comparaison est donc équitable :

    48 hauteurs normalisées ──► cellule récurrente (RNN | GRU | LSTM),
    ``couches`` couches, état caché de dimension ``cachee``, lue pas à pas
    ──► dernier état caché h_48 ──► couche linéaire ──► 24 hauteurs prévues

C'est une **prévision directe multi-horizon** : les 24 valeurs sortent en une
seule passe, sans réinjecter les prévisions dans le réseau (F04).

``ModeleExport`` ajoute la normalisation et la dénormalisation autour du réseau :
le graphe ONNX reçoit et rend des **mètres**, le site n'a aucune constante à
recopier (F11, F12).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

CELLULES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

ARCHITECTURES = {
    "rnn": "RNN simple (Elman, tanh)",
    "gru": "GRU (portes de mise à jour et de réinitialisation)",
    "lstm": "LSTM (cellule mémoire, portes d'entrée, d'oubli et de sortie)",
}


class Prevision(nn.Module):
    """Réseau récurrent + tête linéaire : (N, entree) -> (N, horizon), valeurs normalisées."""

    def __init__(self, type_cellule: str, cachee: int = 64, couches: int = 2, horizon: int = 24):
        super().__init__()
        if type_cellule not in CELLULES:
            raise ValueError(f"cellule inconnue : {type_cellule} (attendu : {', '.join(CELLULES)})")
        self.type_cellule = type_cellule
        self.recurrent = CELLULES[type_cellule](input_size=1, hidden_size=cachee, num_layers=couches,
                                                batch_first=True)
        self.tete = nn.Linear(cachee, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sorties, _ = self.recurrent(x.unsqueeze(-1))  # (N, entree, cachee) : un état caché par heure
        return self.tete(sorties[:, -1])  # dernier état caché -> 24 valeurs


class ModeleExport(nn.Module):
    """Normalisation + réseau + dénormalisation : mètres en entrée, mètres en sortie."""

    def __init__(self, modele: Prevision, moyenne: float, ecart_type: float):
        super().__init__()
        self.modele = modele
        self.register_buffer("moyenne", torch.tensor(moyenne, dtype=torch.float32))
        self.register_buffer("ecart_type", torch.tensor(ecart_type, dtype=torch.float32))

    def forward(self, hauteurs: torch.Tensor) -> torch.Tensor:
        z = (hauteurs - self.moyenne) / self.ecart_type
        return self.modele(z) * self.ecart_type + self.moyenne


def nombre_parametres(modele: nn.Module) -> int:
    return sum(p.numel() for p in modele.parameters())


def construire_modele(checkpoint: dict) -> Prevision:
    """Recrée un modèle entraîné à partir de son checkpoint, sans réentraînement (F11)."""
    c = checkpoint["config"]
    modele = Prevision(c["type"], cachee=c["cachee"], couches=c["couches"], horizon=c["horizon"])
    modele.load_state_dict(checkpoint["state_dict"])
    return modele.eval()


def construire_modele_export(checkpoint: dict) -> ModeleExport:
    n = checkpoint["normalisation"]
    return ModeleExport(construire_modele(checkpoint), n["moyenne_m"], n["ecart_type_m"]).eval()


def predire(modele: ModeleExport, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Prévisions en mètres d'un ``ModeleExport`` pour des entrées (N, entree) en mètres."""
    modele.eval()
    sorties = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            sorties.append(modele(torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size]))).numpy())
    return np.concatenate(sorties).astype(np.float32)
