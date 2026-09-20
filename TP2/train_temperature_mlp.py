"""Approximation du champ de température de surface par un MLP (idée 1 du TP2).

Ce script apprend la fonction ``f(lon, lat) -> temperature`` sur un pas de temps
du fichier HYCOM 3D-SURF (SHOM), puis exporte le réseau au format ONNX pour une
exécution ultérieure côté web avec ``onnxruntime-web``.

Deux artefacts sont produits dans ``TP2/model/`` :
  - ``temperature_mlp.onnx`` : le réseau entraîné ;
  - ``metadata.json``        : constantes de normalisation et emprise, pour que
    le site applique EXACTEMENT la même transformation en entrée/sortie.

Usage :
    uv run python TP2/train_temperature_mlp.py
    uv run python TP2/train_temperature_mlp.py --fourier --epochs 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATA_DIR = Path(__file__).parent / "data"
NC_FILE = DATA_DIR / "HYDRODYN-SURF_HYCOM3D-SURF_R1000_MANGASC_20260911.nc"
MODEL_DIR = Path(__file__).parent / "model"

VARIABLE = "temperature"
TIME_INDEX = 0
ONNX_OPSET = 18


def charger_points(nc_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge (lon, lat) et température à un pas de temps, hors cases masquées.

    Retourne :
        entrees  : tableau (N, 2) des couples (lon, lat) valides ;
        cibles   : tableau (N, 1) des températures correspondantes.
    """
    with xr.open_dataset(nc_file) as ds:
        champ = ds[VARIABLE].isel(time=TIME_INDEX, depth=0)
        lon = np.asarray(ds["lon"].values, dtype="float64")
        lat = np.asarray(ds["lat"].values, dtype="float64")
        temp = np.asarray(champ.values, dtype="float64")

    # lon/lat sont des grilles 2D (Y, X), comme le champ de température.
    lon = np.broadcast_to(lon, temp.shape).ravel()
    lat = np.broadcast_to(lat, temp.shape).ravel()
    temp = temp.ravel()

    valides = np.isfinite(temp) & np.isfinite(lon) & np.isfinite(lat)
    entrees = np.stack([lon[valides], lat[valides]], axis=1)
    cibles = temp[valides].reshape(-1, 1)
    return entrees, cibles


def calculer_normalisation(
    entrees: np.ndarray, cibles: np.ndarray
) -> dict[str, float]:
    """Calcule les bornes/moyennes servant à normaliser entrées et sortie."""
    return {
        "lon_min": float(entrees[:, 0].min()),
        "lon_max": float(entrees[:, 0].max()),
        "lat_min": float(entrees[:, 1].min()),
        "lat_max": float(entrees[:, 1].max()),
        "temp_mean": float(cibles.mean()),
        "temp_std": float(cibles.std()),
        "temp_min": float(cibles.min()),
        "temp_max": float(cibles.max()),
    }


def normaliser_entrees(entrees: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    """Ramène (lon, lat) dans l'intervalle [-1, 1]."""
    lon = 2 * (entrees[:, 0] - norm["lon_min"]) / (norm["lon_max"] - norm["lon_min"]) - 1
    lat = 2 * (entrees[:, 1] - norm["lat_min"]) / (norm["lat_max"] - norm["lat_min"]) - 1
    return np.stack([lon, lat], axis=1)


def normaliser_cibles(cibles: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    """Centre-réduit la température."""
    return (cibles - norm["temp_mean"]) / norm["temp_std"]


class FourierFeatures(nn.Module):
    """Encodage de Fourier aléatoire des coordonnées (optionnel).

    Projette l'entrée (N, 2) sur ``[sin(2π x B), cos(2π x B)]`` de dimension
    ``2 * num_frequencies``. La matrice B est un buffer fixe : elle est donc
    incluse dans le graphe ONNX, le site n'a rien à recalculer.
    """

    def __init__(self, num_frequencies: int, sigma: float, seed: int = 0) -> None:
        super().__init__()
        generateur = torch.Generator().manual_seed(seed)
        b = torch.randn(2, num_frequencies, generator=generateur) * sigma
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2 * torch.pi * (x @ self.b)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class MLP(nn.Module):
    """Perceptron multicouche approximant température = f(lon, lat)."""

    def __init__(
        self,
        hidden: int = 64,
        depth: int = 3,
        fourier: bool = False,
        num_frequencies: int = 64,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        couches: list[nn.Module] = []
        if fourier:
            self.encodage = FourierFeatures(num_frequencies, sigma)
            dim_entree = 2 * num_frequencies
        else:
            self.encodage = nn.Identity()
            dim_entree = 2

        taille = dim_entree
        for _ in range(depth):
            couches.append(nn.Linear(taille, hidden))
            couches.append(nn.Tanh())
            taille = hidden
        couches.append(nn.Linear(taille, 1))
        self.reseau = nn.Sequential(*couches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reseau(self.encodage(x))


def entrainer(
    modele: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    """Entraîne le modèle et conserve les meilleurs poids sur la validation."""
    modele.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr)
    perte = nn.MSELoss()
    n = x_train.shape[0]

    meilleure_val = float("inf")
    meilleurs_poids = {k: v.detach().clone() for k, v in modele.state_dict().items()}

    for epoch in range(1, epochs + 1):
        modele.train()
        permutation = torch.randperm(n, device=device)
        for debut in range(0, n, batch_size):
            indices = permutation[debut : debut + batch_size]
            optimiseur.zero_grad()
            sortie = modele(x_train[indices])
            cout = perte(sortie, y_train[indices])
            cout.backward()
            optimiseur.step()

        modele.eval()
        with torch.no_grad():
            val = perte(modele(x_val), y_val).item()
        if val < meilleure_val:
            meilleure_val = val
            meilleurs_poids = {
                k: v.detach().clone() for k, v in modele.state_dict().items()
            }
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  val_mse={val:.5f}")

    modele.load_state_dict(meilleurs_poids)
    print(f"  meilleure val_mse = {meilleure_val:.5f}")
    return modele


def exporter_onnx(modele: nn.Module, onnx_path: Path, device: torch.device) -> None:
    """Exporte le modèle en ONNX avec un axe batch dynamique."""
    modele.eval()
    exemple = torch.zeros(1, 2, device=device)
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        modele,
        (exemple,),
        str(onnx_path),
        input_names=["coords"],
        output_names=["temperature_norm"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=ONNX_OPSET,
        external_data=False,
    )
    print(f"Modèle ONNX écrit : {onnx_path}")


def verifier_onnx(
    modele: nn.Module, onnx_path: Path, x_val: torch.Tensor, device: torch.device
) -> None:
    """Vérifie le graphe ONNX et compare son inférence à celle de PyTorch."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    echantillon = x_val[:256]
    modele.eval()
    with torch.no_grad():
        ref = modele(echantillon.to(device)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sortie = session.run(None, {"coords": echantillon.numpy().astype(np.float32)})[0]

    ecart = float(np.max(np.abs(ref - sortie)))
    print(f"Vérification ONNX vs PyTorch : écart max = {ecart:.2e}")
    if ecart > 1e-3:
        print("  ATTENTION : écart supérieur au seuil attendu.")


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier", action="store_true", help="activer l'encodage de Fourier")
    parser.add_argument("--num-frequencies", type=int, default=64, help="fréquences de Fourier")
    parser.add_argument("--sigma", type=float, default=2.0, help="échelle des fréquences de Fourier")
    parser.add_argument("--hidden", type=int, default=64, help="neurones par couche cachée")
    parser.add_argument("--depth", type=int, default=3, help="nombre de couches cachées")
    parser.add_argument("--epochs", type=int, default=300, help="nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=4096, help="taille de mini-batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="part de validation")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    return parser.parse_args()


def main() -> None:
    args = parser_arguments()
    if not NC_FILE.exists():
        raise SystemExit(f"Fichier introuvable : {NC_FILE}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")

    print("Chargement des données...")
    entrees, cibles = charger_points(NC_FILE)
    print(f"  {entrees.shape[0]} points valides (hors terre/masque)")

    norm = calculer_normalisation(entrees, cibles)
    x = normaliser_entrees(entrees, norm).astype("float32")
    y = normaliser_cibles(cibles, norm).astype("float32")

    indices = rng.permutation(x.shape[0])
    n_val = int(x.shape[0] * args.val_fraction)
    idx_val, idx_train = indices[:n_val], indices[n_val:]
    x_train = torch.from_numpy(x[idx_train])
    y_train = torch.from_numpy(y[idx_train])
    x_val = torch.from_numpy(x[idx_val])
    y_val = torch.from_numpy(y[idx_val])
    print(f"  train={x_train.shape[0]}  val={x_val.shape[0]}")

    modele = MLP(
        hidden=args.hidden,
        depth=args.depth,
        fourier=args.fourier,
        num_frequencies=args.num_frequencies,
        sigma=args.sigma,
    )
    print(f"Modèle : {'MLP + Fourier' if args.fourier else 'MLP brut'}")

    print("Entraînement...")
    modele = entrainer(
        modele,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    suffixe = "_fourier" if args.fourier else ""
    onnx_path = MODEL_DIR / f"temperature_mlp{suffixe}.onnx"
    meta_path = MODEL_DIR / f"metadata{suffixe}.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    exporter_onnx(modele, onnx_path, device)
    verifier_onnx(modele, onnx_path, x_val, device)

    metadata = {
        "variable": VARIABLE,
        "time_index": TIME_INDEX,
        "input_names": ["lon", "lat"],
        "output_name": VARIABLE,
        "output_unit": "degC",
        "normalization": norm,
        "fourier": args.fourier,
        "onnx_opset": ONNX_OPSET,
        "description": (
            "Normaliser lon/lat dans [-1,1] avec (lon|lat)_min/max, exécuter le "
            "modèle ONNX, puis dé-normaliser : temp = sortie * temp_std + temp_mean."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Métadonnées écrites : {meta_path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
