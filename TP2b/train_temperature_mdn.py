"""Approximation de la DISTRIBUTION des températures de surface (idée 3, design 1).

Contrairement à l'idée 1 (un MLP qui prédit UNE température par point), ce script
apprend la **loi de probabilité** p(T) des températures de surface de toute la zone
Manche-Gascogne à un instant donné. C'est un *Mixture Density Network* (MDN) dans sa
forme la plus simple : un mélange de K gaussiennes

    p(T) = somme_k  pi_k * N(T ; mu_k, sigma_k)

dont les paramètres {pi_k, mu_k, sigma_k} sont appris par descente de gradient en
maximisant la vraisemblance (donc en minimisant la log-vraisemblance négative, NLL).

Le modèle est ensuite exporté en ONNX de sorte que, côté navigateur, on lui donne une
température T (en °C) et il renvoie directement la densité p(T) : l'évaluation
analytique du mélange (et la dé-normalisation) est « cuite » dans le graphe ONNX.

Trois artefacts sont produits dans ``TP2b/model/`` :
  - ``temperature_mdn.onnx`` : entrée = température (°C), sortie = densité p(T) ;
  - ``metadata_mdn.json``     : paramètres du mélange + normalisation + métriques ;
  - ``histogram.json``        : histogramme réel de référence (pour la superposition).

Usage :
    uv run python TP2b/train_temperature_mdn.py
    uv run python TP2b/train_temperature_mdn.py --num-components 8 --epochs 600
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Le fichier NetCDF reste dans TP2/data ; seul TP2b contient le code de ce TP.
DATA_DIR = Path(__file__).parent.parent / "TP2" / "data"
NC_FILE = DATA_DIR / "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.nc"
MODEL_DIR = Path(__file__).parent / "model"

# Archive d'origine sur le portail de téléchargement du SHOM (data.shom.fr).
ZIP_URL = (
    "https://services.data.shom.fr/telechargement/oceano/"
    "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.dl/file/"
    "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.dl.zip"
)

VARIABLE = "temperature"
TIME_INDEX = 0
ONNX_OPSET = 18
HISTOGRAM_BINS = 60


def telecharger_donnees() -> None:
    """Télécharge l'archive SHOM et extrait le NetCDF (dans TP2/data) si absent.

    Rend ce TP autonome : inutile d'avoir lancé le TP2 au préalable.
    """
    if NC_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fichier NetCDF absent. Téléchargement depuis le SHOM :")
    print(f"  {ZIP_URL}")
    with urllib.request.urlopen(ZIP_URL) as resp:
        archive = io.BytesIO(resp.read())
    with zipfile.ZipFile(archive) as zf:
        membre = next(
            (n for n in zf.namelist() if n.endswith((".nc.gz", ".nc"))), None
        )
        if membre is None:
            raise SystemExit("Aucun fichier .nc(.gz) trouvé dans l'archive SHOM.")
        contenu = zf.read(membre)
    if membre.endswith(".gz"):
        contenu = gzip.decompress(contenu)
    NC_FILE.write_bytes(contenu)
    print(f"Extrait : {NC_FILE.name} ({NC_FILE.stat().st_size / 1e6:.1f} Mo)")


def charger_temperatures(nc_file: Path) -> np.ndarray:
    """Charge le vecteur 1D des températures valides (hors terre/masque) à t=0."""
    with xr.open_dataset(nc_file) as ds:
        champ = ds[VARIABLE].isel(time=TIME_INDEX, depth=0)
        temp = np.asarray(champ.values, dtype="float64").ravel()
    return temp[np.isfinite(temp)]


def calculer_normalisation(temperatures: np.ndarray) -> dict[str, float]:
    """Bornes et centre-réduction servant à normaliser la température."""
    return {
        "temp_mean": float(temperatures.mean()),
        "temp_std": float(temperatures.std()),
        "temp_min": float(temperatures.min()),
        "temp_max": float(temperatures.max()),
    }


class MixtureDensity(nn.Module):
    """Mélange de K gaussiennes 1D, appris en espace normalisé (centré-réduit).

    Les paramètres bruts sont contraints pour rester valides :
      - poids pi via softmax (somme = 1) ;
      - écarts-types sigma via softplus (strictement positifs, bornés en bas).
    """

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_components))
        # Moyennes initiales étalées sur l'intervalle normalisé typique [-2, 2].
        self.means = nn.Parameter(torch.linspace(-2.0, 2.0, num_components))
        self.raw_scales = nn.Parameter(torch.zeros(num_components))

    def parametres(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poids = torch.softmax(self.logits, dim=0)
        sigmas = nn.functional.softplus(self.raw_scales) + 1e-3
        return poids, self.means, sigmas

    def log_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Log-densité du mélange en chaque point ``t`` (forme (N, 1))."""
        poids, moyennes, sigmas = self.parametres()
        z = (t - moyennes) / sigmas  # diffusion -> (N, K)
        log_comp = -0.5 * math.log(2 * math.pi) - torch.log(sigmas) - 0.5 * z * z
        log_pond = log_comp + torch.log(poids)
        return torch.logsumexp(log_pond, dim=1, keepdim=True)


class DensityEvaluator(nn.Module):
    """Enveloppe exportée en ONNX : température (°C) -> densité p(T).

    Elle « cuit » dans le graphe la normalisation ``t = (T - mean) / std`` et le
    changement de variable de densité ``p_T(T) = p_t(t) / std``. Le navigateur n'a
    donc qu'à fournir des températures physiques et lire la densité physique.
    """

    def __init__(self, melange: MixtureDensity, mean: float, std: float) -> None:
        super().__init__()
        self.melange = melange
        self.register_buffer("mean", torch.tensor([mean], dtype=torch.float32))
        self.register_buffer("std", torch.tensor([std], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = (x - self.mean) / self.std
        return torch.exp(self.melange.log_prob(t)) / self.std


def entrainer(
    melange: MixtureDensity,
    t_train: torch.Tensor,
    t_val: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> MixtureDensity:
    """Minimise la NLL (log-vraisemblance négative) et garde les meilleurs poids."""
    melange.to(device)
    t_train, t_val = t_train.to(device), t_val.to(device)
    optimiseur = torch.optim.Adam(melange.parameters(), lr=lr)

    meilleure_val = float("inf")
    meilleurs_poids = {k: v.detach().clone() for k, v in melange.state_dict().items()}

    for epoch in range(1, epochs + 1):
        melange.train()
        optimiseur.zero_grad()
        nll = -melange.log_prob(t_train).mean()
        nll.backward()
        optimiseur.step()

        melange.eval()
        with torch.no_grad():
            val = (-melange.log_prob(t_val).mean()).item()
        if val < meilleure_val:
            meilleure_val = val
            meilleurs_poids = {
                k: v.detach().clone() for k, v in melange.state_dict().items()
            }
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  val_nll(norm)={val:.5f}")

    melange.load_state_dict(meilleurs_poids)
    print(f"  meilleure val_nll(norm) = {meilleure_val:.5f}")
    return melange


def exporter_onnx(modele: DensityEvaluator, onnx_path: Path, device: torch.device) -> None:
    """Exporte l'évaluateur de densité en ONNX (axe batch dynamique)."""
    modele.eval()
    exemple = torch.zeros(1, 1, device=device)
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        modele,
        (exemple,),
        str(onnx_path),
        input_names=["temperature"],
        output_names=["density"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=ONNX_OPSET,
        external_data=False,
    )
    print(f"Modèle ONNX écrit : {onnx_path}")


def verifier_onnx(
    modele: DensityEvaluator, onnx_path: Path, temps: np.ndarray, device: torch.device
) -> None:
    """Vérifie le graphe ONNX et compare son inférence à celle de PyTorch."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    echantillon = temps[:256].reshape(-1, 1).astype("float32")
    modele.eval()
    with torch.no_grad():
        ref = modele(torch.from_numpy(echantillon).to(device)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sortie = session.run(None, {"temperature": echantillon})[0]

    ecart = float(np.max(np.abs(ref - sortie)))
    print(f"Vérification ONNX vs PyTorch : écart max = {ecart:.2e}")
    if ecart > 1e-3:
        print("  ATTENTION : écart supérieur au seuil attendu.")


def parametres_physiques(
    melange: MixtureDensity, norm: dict[str, float]
) -> dict[str, list[float]]:
    """Convertit les paramètres normalisés du mélange vers l'espace physique (°C)."""
    with torch.no_grad():
        poids, moyennes, sigmas = melange.parametres()
    poids = poids.detach().cpu().numpy()
    moyennes = moyennes.detach().cpu().numpy()
    sigmas = sigmas.detach().cpu().numpy()
    mean, std = norm["temp_mean"], norm["temp_std"]
    return {
        "weights": poids.tolist(),
        "means": (moyennes * std + mean).tolist(),
        "scales": (sigmas * std).tolist(),
    }


def echantillonner_melange(
    rng: np.random.Generator, params: dict[str, list[float]], n: int
) -> np.ndarray:
    """Tire ``n`` températures selon le mélange (en espace physique)."""
    poids = np.asarray(params["weights"])
    moyennes = np.asarray(params["means"])
    sigmas = np.asarray(params["scales"])
    composantes = rng.choice(len(poids), size=n, p=poids / poids.sum())
    return rng.normal(moyennes[composantes], sigmas[composantes])


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """Distance de Wasserstein-1 (Earth Mover's Distance) entre deux échantillons 1D."""
    n = min(len(a), len(b))
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def densite_modele(params: dict[str, list[float]], x: np.ndarray) -> np.ndarray:
    """Densité analytique du mélange (espace physique) évaluée sur ``x``."""
    poids = np.asarray(params["weights"])[None, :]
    moyennes = np.asarray(params["means"])[None, :]
    sigmas = np.asarray(params["scales"])[None, :]
    z = (x[:, None] - moyennes) / sigmas
    comp = np.exp(-0.5 * z * z) / (sigmas * math.sqrt(2 * math.pi))
    return (poids * comp).sum(axis=1)


def calculer_metriques(
    temperatures: np.ndarray,
    params: dict[str, list[float]],
    hist_density: np.ndarray,
    bin_centers: np.ndarray,
    bin_width: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """NLL physique, divergence KL(hist||modèle) et distance de Wasserstein-1."""
    dens = densite_modele(params, temperatures)
    nll_phys = float(-np.mean(np.log(np.clip(dens, 1e-12, None))))

    q = np.clip(densite_modele(params, bin_centers), 1e-12, None)
    masque = hist_density > 0
    kl = float(
        np.sum(hist_density[masque] * np.log(hist_density[masque] / q[masque]) * bin_width)
    )

    ech_modele = echantillonner_melange(rng, params, len(temperatures))
    emd = wasserstein1(temperatures, ech_modele)
    return {"val_nll_phys": nll_phys, "kl_hist_modele": kl, "wasserstein1": emd}


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-components", type=int, default=6, help="nombre de gaussiennes")
    parser.add_argument("--epochs", type=int, default=500, help="nombre d'époques")
    parser.add_argument("--lr", type=float, default=5e-2, help="taux d'apprentissage")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="part de validation")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    return parser.parse_args()


def main() -> None:
    args = parser_arguments()
    telecharger_donnees()
    if not NC_FILE.exists():
        raise SystemExit(f"Fichier introuvable : {NC_FILE}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")

    print("Chargement des données...")
    temperatures = charger_temperatures(NC_FILE)
    print(f"  {temperatures.size} températures valides (hors terre/masque)")

    norm = calculer_normalisation(temperatures)
    t_norm = ((temperatures - norm["temp_mean"]) / norm["temp_std"]).astype("float32")

    indices = rng.permutation(t_norm.size)
    n_val = int(t_norm.size * args.val_fraction)
    idx_val, idx_train = indices[:n_val], indices[n_val:]
    t_train = torch.from_numpy(t_norm[idx_train]).reshape(-1, 1)
    t_val = torch.from_numpy(t_norm[idx_val]).reshape(-1, 1)
    print(f"  train={t_train.shape[0]}  val={t_val.shape[0]}")

    melange = MixtureDensity(num_components=args.num_components)
    print(f"Modèle : mélange de {args.num_components} gaussiennes")

    print("Entraînement (minimisation de la NLL)...")
    melange = entrainer(
        melange, t_train, t_val, epochs=args.epochs, lr=args.lr, device=device
    )
    melange.to("cpu")

    params = parametres_physiques(melange, norm)

    # Histogramme réel de référence (densité), pour la superposition côté web.
    hist_density, bin_edges = np.histogram(
        temperatures, bins=HISTOGRAM_BINS, density=True
    )
    hist_counts, _ = np.histogram(temperatures, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])

    metriques = calculer_metriques(
        temperatures, params, hist_density, bin_centers, bin_width, rng
    )
    print("Métriques :")
    print(f"  NLL (physique)          = {metriques['val_nll_phys']:.4f}")
    print(f"  KL(hist || modèle)      = {metriques['kl_hist_modele']:.5f}")
    print(f"  Wasserstein-1 (EMD)     = {metriques['wasserstein1']:.4f} °C")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    evaluateur = DensityEvaluator(melange, norm["temp_mean"], norm["temp_std"])
    onnx_path = MODEL_DIR / "temperature_mdn.onnx"
    exporter_onnx(evaluateur, onnx_path, torch.device("cpu"))
    verifier_onnx(evaluateur, onnx_path, temperatures, torch.device("cpu"))

    metadata = {
        "variable": VARIABLE,
        "time_index": TIME_INDEX,
        "input_name": "temperature",
        "input_unit": "degC",
        "output_name": "density",
        "num_components": args.num_components,
        "normalization": norm,
        "mixture_phys_space": params,
        "metrics": metriques,
        "onnx_opset": ONNX_OPSET,
        "description": (
            "Le modèle ONNX prend une température (°C) en entrée et renvoie la densité "
            "de probabilité p(T). Les paramètres du mélange (espace physique) sont aussi "
            "fournis pour échantillonner ou tracer la courbe côté client."
        ),
    }
    (MODEL_DIR / "metadata_mdn.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Métadonnées écrites : {MODEL_DIR / 'metadata_mdn.json'}")

    histogramme = {
        "variable": VARIABLE,
        "unit": "degC",
        "n": int(temperatures.size),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_width": bin_width,
        "density": hist_density.tolist(),
        "counts": hist_counts.tolist(),
    }
    (MODEL_DIR / "histogram.json").write_text(
        json.dumps(histogramme, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Histogramme écrit : {MODEL_DIR / 'histogram.json'}")
    print("Terminé.")


if __name__ == "__main__":
    main()
