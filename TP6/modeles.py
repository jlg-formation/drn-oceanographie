"""Modèles génératifs du TP6 : VAE convolutionnel et GAN convolutionnel compact (F07, F13).

Les deux modèles partagent des briques **identiques**, pour que la comparaison
porte sur l'apprentissage et non sur l'architecture :

    Encodeur (VAE) / Discriminateur (GAN) :
        64×64 ─► Conv 4×4 /2 ─► 32×32×32 ─► 16×16×64 ─► 8×8×128 ─► 4×4×256 ─► Linéaire

    Décodeur (VAE) = Générateur (GAN) :
        z ─► Linéaire ─► 4×4×256 ─► [suréchantillonnage ×2 + Conv 3×3] × 4 ─► 64×64 ─► tanh

- l'**encodeur** du VAE sort la moyenne μ et le log-variance log σ² d'une
  gaussienne dans l'espace latent ; on tire z = μ + σ·ε (reparamétrisation) ;
- le **décodeur** transforme z en carte ; le **générateur** du GAN est le même
  réseau, entraîné autrement (contre un discriminateur) ;
- le **discriminateur** a la structure de l'encodeur mais sort un seul nombre :
  « réel ou synthétique » (logit). Il n'est pas exporté pour le site.

Le suréchantillonnage « plus proche voisin + convolution » évite les damiers
des convolutions transposées.

Les classes ``*Export`` ajoutent la normalisation autour des réseaux : les
graphes ONNX reçoivent et rendent des **mètres** (F24 à F26).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _bloc_descendant(c_in: int, c_out: int, normalisation: str) -> nn.Sequential:
    conv = nn.Conv2d(c_in, c_out, 4, stride=2, padding=1)
    if normalisation == "spectrale":
        return nn.Sequential(spectral_norm(conv), nn.LeakyReLU(0.2))
    if normalisation == "batch":
        return nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))
    return nn.Sequential(conv, nn.LeakyReLU(0.2))


class Extracteur(nn.Module):
    """Quatre convolutions à pas 2 : (N, 1, 64, 64) -> (N, 4096). Base de l'encodeur et du discriminateur."""

    def __init__(self, largeur: int = 32, normalisation: str = "batch"):
        super().__init__()
        c = [1, largeur, 2 * largeur, 4 * largeur, 8 * largeur]
        self.blocs = nn.Sequential(*[_bloc_descendant(c[i], c[i + 1], "aucune" if i == 0 else normalisation)
                                     for i in range(4)])
        self.sortie = c[-1] * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocs(x).flatten(1)


class Decodeur(nn.Module):
    """Vecteur latent (N, dim) -> carte (N, 1, 64, 64) dans [-1, 1]. Décodeur du VAE = générateur du GAN."""

    def __init__(self, dim_latente: int, largeur: int = 32):
        super().__init__()
        self.c0 = 8 * largeur
        self.entree = nn.Sequential(nn.Linear(dim_latente, self.c0 * 4 * 4), nn.BatchNorm1d(self.c0 * 4 * 4),
                                    nn.ReLU())
        c = [8 * largeur, 4 * largeur, 2 * largeur, largeur, largeur]
        self.blocs = nn.Sequential(*[
            nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                          nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU())
            for i in range(4)
        ])
        self.sortie = nn.Conv2d(largeur, 1, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.entree(z).view(-1, self.c0, 4, 4)
        return torch.tanh(self.sortie(self.blocs(h)))


# --------------------------------------------------------------------------
# VAE
# --------------------------------------------------------------------------

class VAE(nn.Module):
    """Autoencodeur variationnel convolutionnel (F07)."""

    def __init__(self, dim_latente: int = 16, largeur: int = 32):
        super().__init__()
        self.dim_latente = dim_latente
        self.extracteur = Extracteur(largeur, "batch")
        self.mu = nn.Linear(self.extracteur.sortie, dim_latente)
        self.logvar = nn.Linear(self.extracteur.sortie, dim_latente)
        self.decodeur = Decodeur(dim_latente, largeur)

    def encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.extracteur(x)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # reparamétrisation
        return self.decodeur(z), mu, logvar


def perte_vae(x: torch.Tensor, x_rec: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
              beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perte du VAE par carte : reconstruction (somme des carrés sur les 4 096 pixels) + β · KL.

    KL(N(μ, σ²) ‖ N(0, I)) = ½ Σ (μ² + σ² − log σ² − 1), en nats, sommée sur les dimensions latentes.
    """
    reconstruction = ((x_rec - x) ** 2).flatten(1).sum(1).mean()
    kl = (0.5 * (mu**2 + logvar.exp() - logvar - 1)).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl


# --------------------------------------------------------------------------
# GAN
# --------------------------------------------------------------------------

class Discriminateur(nn.Module):
    """Carte (N, 1, 64, 64) -> logit (N,) : élevé si la carte semble réelle (F13)."""

    def __init__(self, largeur: int = 32, spectrale: bool = True):
        super().__init__()
        self.extracteur = Extracteur(largeur, "spectrale" if spectrale else "batch")
        lin = nn.Linear(self.extracteur.sortie, 1)
        self.sortie = spectral_norm(lin) if spectrale else lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sortie(self.extracteur(x)).squeeze(1)


# --------------------------------------------------------------------------
# Graphes d'export : mètres en entrée et en sortie
# --------------------------------------------------------------------------

class EncodeurExport(nn.Module):
    """Élévations (N, 64, 64) en m -> (μ, log σ²). Le relief est centré et normalisé dans le graphe."""

    def __init__(self, vae: VAE, echelle_m: float):
        super().__init__()
        self.vae = vae
        self.register_buffer("echelle", torch.tensor(echelle_m, dtype=torch.float32))

    def forward(self, elevation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relief = elevation - elevation.mean(dim=(1, 2), keepdim=True)
        z = torch.clamp(relief / self.echelle, -1.0, 1.0)
        return self.vae.encoder(z.unsqueeze(1))


class GenerateurExport(nn.Module):
    """Vecteur latent (N, dim) -> relief relatif (N, 64, 64) en m. Sert au décodeur VAE et au générateur GAN."""

    def __init__(self, decodeur: Decodeur, echelle_m: float):
        super().__init__()
        self.decodeur = decodeur
        self.register_buffer("echelle", torch.tensor(echelle_m, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decodeur(z).squeeze(1) * self.echelle


# --------------------------------------------------------------------------
# Chargement et inférence
# --------------------------------------------------------------------------

def nombre_parametres(modele: nn.Module) -> int:
    return sum(p.numel() for p in modele.parameters())


def construire_vae(checkpoint: dict) -> VAE:
    """Recrée un VAE entraîné à partir de son checkpoint, sans réentraînement (F09)."""
    c = checkpoint["config"]
    vae = VAE(c["dim_latente"], c["largeur"])
    vae.load_state_dict(checkpoint["state_dict"])
    return vae.eval()


def construire_generateur(checkpoint: dict) -> Decodeur:
    """Recrée le générateur du GAN seul (F16) : le discriminateur n'est pas nécessaire pour générer."""
    c = checkpoint["config"]
    g = Decodeur(c["dim_latente"], c["largeur"])
    g.load_state_dict(checkpoint["generateur"])
    return g.eval()


def construire_discriminateur(checkpoint: dict) -> Discriminateur:
    c = checkpoint["config"]
    d = Discriminateur(c["largeur"], c["spectrale"])
    d.load_state_dict(checkpoint["discriminateur"])
    return d.eval()


def generer(decodeur: Decodeur, z: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Cartes normalisées (N, 64, 64) à partir de vecteurs latents (N, dim)."""
    decodeur.eval()
    sorties = []
    with torch.no_grad():
        for i in range(0, len(z), batch_size):
            sorties.append(decodeur(torch.from_numpy(np.ascontiguousarray(z[i:i + batch_size], dtype=np.float32)))
                           .squeeze(1).numpy())
    return np.concatenate(sorties).astype(np.float32)


def encoder(vae: VAE, x: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(μ, log σ²) pour des cartes normalisées (N, 64, 64)."""
    vae.eval()
    mus, logvars = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            mu, lv = vae.encoder(torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size])).unsqueeze(1))
            mus.append(mu.numpy())
            logvars.append(lv.numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def reconstruire(vae: VAE, x: np.ndarray) -> np.ndarray:
    """Reconstruction déterministe : décodage de la moyenne μ (comme dans le site web, F10)."""
    mu, _ = encoder(vae, x)
    return generer(vae.decodeur, mu)


def augmenter(x: torch.Tensor, generateur: torch.Generator) -> torch.Tensor:
    """Une des 8 symétries du carré (rotations de 90° et miroirs) tirée par carte.

    Le relief n'a pas d'orientation privilégiée pour un modèle non conditionné
    par la géographie : on multiplie ainsi la variété des exemples d'entraînement.
    """
    k = torch.randint(0, 4, (x.shape[0],), generator=generateur).to(x.device)
    miroir = torch.randint(0, 2, (x.shape[0],), generator=generateur).to(x.device).bool()
    x = torch.where(miroir[:, None, None, None], x.flip(-1), x)
    sortie = x.clone()
    for r in range(1, 4):
        choix = k == r
        if choix.any():
            sortie[choix] = torch.rot90(x[choix], r, dims=(-2, -1))
    return sortie
