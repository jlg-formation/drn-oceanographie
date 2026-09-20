import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./style.css";

import { TemperatureModel } from "./inference";
import { NeuralFieldLayer } from "./field-layer";
import { SeaMask } from "./sea-mask";
import { cssGradient } from "./colormap";

const MODELS = {
  fourier: {
    onnx: "/temperature_mlp_fourier.onnx",
    meta: "/metadata_fourier.json",
  },
  brut: {
    onnx: "/temperature_mlp.onnx",
    meta: "/metadata.json",
  },
} as const;

type ModelKey = keyof typeof MODELS;

const statusEl = document.getElementById("status")!;
const probeEl = document.getElementById("probe")!;
const legendEl = document.getElementById("legend")!;
const opacityEl = document.getElementById("opacity") as HTMLInputElement;
const selectEl = document.getElementById("model-select") as HTMLSelectElement;

// --- Carte Leaflet centrée sur le Golfe de Gascogne ---
const map = L.map("map", {
  center: [47.0, -6.0],
  zoom: 6,
  minZoom: 5,
  maxZoom: 11,
  zoomControl: true,
});

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> — Modèle HYCOM 3D-SURF (SHOM)',
  subdomains: "abc",
  maxZoom: 19,
}).addTo(map);

async function loadModel(key: ModelKey): Promise<TemperatureModel> {
  return TemperatureModel.load(MODELS[key].onnx, MODELS[key].meta);
}

function buildLegend(vmin: number, vmax: number): void {
  const mid = (vmin + vmax) / 2;
  legendEl.innerHTML = `
    <div class="bar" style="background: ${cssGradient()}"></div>
    <div class="labels">
      <span>${vmin.toFixed(1)}°C</span>
      <span>${mid.toFixed(1)}°C</span>
      <span>${vmax.toFixed(1)}°C</span>
    </div>`;
}

async function main(): Promise<void> {
  let currentKey: ModelKey = "fourier";
  let model = await loadModel(currentKey);

  const vmin = model.meta.normalization.temp_min;
  const vmax = model.meta.normalization.temp_max;
  buildLegend(vmin, vmax);

  const mask = await SeaMask.load(`/${model.meta.sea_mask}`, model.meta.grid);

  const field = new NeuralFieldLayer({
    model,
    mask,
    vmin,
    vmax,
    step: 8,
    opacity: Number(opacityEl.value),
  });
  field.addTo(map);
  statusEl.textContent = `Modèle « ${currentKey} » chargé — champ généré par le réseau.`;

  // Bascule entre les deux modèles.
  selectEl.addEventListener("change", async () => {
    const key = selectEl.value as ModelKey;
    if (key === currentKey) return;
    statusEl.textContent = `Chargement du modèle « ${key} »…`;
    model = await loadModel(key);
    currentKey = key;
    field.setModel(model);
    statusEl.textContent = `Modèle « ${key} » chargé.`;
  });

  // Opacité du calque.
  opacityEl.addEventListener("input", () => {
    field.setOpacity(Number(opacityEl.value));
  });

  // Sonde : température prédite sous le curseur (une inférence sur 1 point).
  let probeScheduled = false;
  let lastLatLng: L.LatLng | null = null;
  map.on("mousemove", (e: L.LeafletMouseEvent) => {
    lastLatLng = e.latlng;
    if (probeScheduled) return;
    probeScheduled = true;
    requestAnimationFrame(async () => {
      probeScheduled = false;
      if (!lastLatLng) return;
      if (!mask.isSea(lastLatLng.lng, lastLatLng.lat)) {
        probeEl.textContent = "Hors mer (pas de donnée)";
        return;
      }
      const t = await model.predict(
        new Float32Array([lastLatLng.lng]),
        new Float32Array([lastLatLng.lat]),
      );
      probeEl.textContent = `${t[0].toFixed(2)} °C  (${lastLatLng.lat.toFixed(2)}°, ${lastLatLng.lng.toFixed(2)}°)`;
    });
  });
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = "Erreur au chargement du modèle (voir la console).";
});
