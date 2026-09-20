import "./style.css";

import { DensityModel, type Histogram } from "./inference";
import { DistributionChart, sampleToDensity } from "./distribution-chart";

const ONNX_URL = "/temperature_mdn.onnx";
const META_URL = "/metadata_mdn.json";
const HIST_URL = "/histogram.json";
const GRID_POINTS = 400;

const statusEl = document.getElementById("status")!;
const metricsEl = document.getElementById("metrics")!;
const sampleBtn = document.getElementById("sample-btn") as HTMLButtonElement;
const clearBtn = document.getElementById("clear-btn") as HTMLButtonElement;
const sampleCountEl = document.getElementById("sample-count") as HTMLInputElement;
const canvas = document.getElementById("chart") as HTMLCanvasElement;

async function main(): Promise<void> {
  statusEl.textContent = "Chargement du modèle…";
  const [model, hist] = await Promise.all([
    DensityModel.load(ONNX_URL, META_URL),
    fetch(HIST_URL).then((r) => r.json() as Promise<Histogram>),
  ]);

  const { temp_min, temp_max } = model.meta.normalization;
  // On élargit un peu la fenêtre pour voir les queues de la distribution.
  const marge = 0.05 * (temp_max - temp_min);
  const xMin = temp_min - marge;
  const xMax = temp_max + marge;

  const grid = new Float32Array(GRID_POINTS);
  for (let i = 0; i < GRID_POINTS; i++) {
    grid[i] = xMin + ((xMax - xMin) * i) / (GRID_POINTS - 1);
  }
  const modelDensity = await model.density(grid);

  const chart = new DistributionChart(canvas, hist);
  let sampleDensity: number[] | undefined;

  const render = (): void => {
    chart.draw({ modelGrid: grid, modelDensity, sampleDensity });
  };

  const m = model.meta.metrics;
  metricsEl.innerHTML = `
    <div><span>Valeurs réelles</span><b>${hist.n.toLocaleString("fr-FR")}</b></div>
    <div><span>Composantes (K)</span><b>${model.meta.num_components}</b></div>
    <div><span>NLL</span><b>${m.val_nll_phys.toFixed(3)}</b></div>
    <div><span>KL(réel ‖ modèle)</span><b>${m.kl_hist_modele.toFixed(4)}</b></div>
    <div><span>Wasserstein-1</span><b>${m.wasserstein1.toFixed(3)} °C</b></div>`;

  render();
  window.addEventListener("resize", render);
  statusEl.textContent = "Densité générée par le réseau (ONNX).";

  sampleBtn.addEventListener("click", () => {
    const n = Math.max(100, Number(sampleCountEl.value) || 5000);
    const samples = model.sample(n);
    sampleDensity = sampleToDensity(samples, hist);
    statusEl.textContent = `${n.toLocaleString("fr-FR")} températures échantillonnées.`;
    render();
  });

  clearBtn.addEventListener("click", () => {
    sampleDensity = undefined;
    statusEl.textContent = "Échantillons effacés.";
    render();
  });
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = "Erreur au chargement (voir la console).";
});
