import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./style.css";

import { DataError, DepthGrid, loadMetadata, loadPoints, type Metadata } from "./data";
import { DepthPalette } from "./palette";

const statusEl = document.getElementById("status")!;
const legendBarEl = document.getElementById("legend-bar")!;
const legendLabelsEl = document.getElementById("legend-labels")!;
const legendNoteEl = document.getElementById("legend-note")!;

const MODE_RASTER = "Carte continue";
const MODE_POINTS = "Points";

function setStatus(message: string, isError = false): void {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

const fmtDepth = (d: number): string =>
  `${d.toLocaleString("fr-FR", { maximumFractionDigits: 0 })} m`;

const fmtCoord = (lat: number, lon: number): string =>
  `Latitude ${lat.toFixed(4)}°<br>Longitude ${lon.toFixed(4)}°`;

function buildLegend(palette: DepthPalette): void {
  legendBarEl.style.background = palette.cssGradient();
  // 200 m serait collé à 0 m sur une échelle linéaire : on n'étiquette pas ce palier.
  const ticks = palette.stops.filter((d) => d === 0 || d / palette.maxDepth >= 0.1);
  legendLabelsEl.innerHTML = ticks
    .map((d) => `<span style="top: ${(d / palette.maxDepth) * 100}%">${fmtDepth(d)}</span>`)
    .join("");
  legendNoteEl.textContent = `Au-delà de ${fmtDepth(palette.maxDepth)} : couleur saturée.`;
}

function createMap(meta: Metadata): L.Map {
  const b = meta.bounds;
  const bounds = L.latLngBounds([b.south, b.west], [b.north, b.east]);
  const map = L.map("map", {
    zoomControl: false,
    // Rendu canvas : des milliers de circleMarkers restent fluides.
    preferCanvas: true,
    minZoom: 5,
    maxZoom: 12,
    maxBounds: bounds.pad(0.5),
  });
  // Le panneau couvre le bord gauche : on cadre l'emprise à sa droite (écrans larges).
  const panelWidth = document.getElementById("panel")!.offsetWidth + 16;
  map.fitBounds(bounds, {
    paddingTopLeft: window.innerWidth > 2 * panelWidth ? [panelWidth, 0] : [0, 0],
  });
  L.control.zoom({ position: "topright" }).addTo(map);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    subdomains: "abc",
    maxZoom: 19,
  }).addTo(map);
  map.attributionControl.addAttribution(
    `Bathymétrie : <a href="https://doi.org/${meta.source.doi}">${meta.source.attribution}</a>`,
  );
  return map;
}

async function main(): Promise<void> {
  const meta = await loadMetadata();
  const palette = new DepthPalette(meta.palette.paliers);
  buildLegend(palette);

  const map = createMap(meta);
  const b = meta.bounds;

  // --- Mode raster : PNG déjà reprojeté en Web Mercator par Python. ---
  const raster = L.imageOverlay(
    `/${meta.raster.fichier}`,
    [
      [b.south, b.west],
      [b.north, b.east],
    ],
    { opacity: 0.9, alt: "Raster bathymétrique GEBCO", className: "bathy-raster" },
  );
  raster.on("error", () =>
    setStatus(new DataError(meta.raster.fichier, "image illisible").message, true),
  );

  // --- Mode points : GeoJSON échantillonné, couleur = profondeur. ---
  const points = L.layerGroup();
  const modes: Record<string, L.Layer> = { [MODE_RASTER]: raster, [MODE_POINTS]: points };
  L.control.layers(modes, undefined, { collapsed: false, position: "topright" }).addTo(map);
  raster.addTo(map);

  let mode = MODE_RASTER;
  map.on("baselayerchange", (e: L.LayersControlEvent) => {
    mode = e.name;
    map.closePopup();
  });

  const [pointsResult, gridResult] = await Promise.allSettled([
    loadPoints(meta),
    DepthGrid.load(meta),
  ]);
  const errors: string[] = [];

  if (pointsResult.status === "fulfilled") {
    L.geoJSON(pointsResult.value, {
      pointToLayer: (feature, latlng) =>
        L.circleMarker(latlng, {
          radius: 3,
          fillColor: palette.color(feature.properties.depth_m),
          fillOpacity: 1,
          color: "#0f172a",
          weight: 0.5,
          opacity: 0.6,
        }).bindPopup(
          `${fmtCoord(latlng.lat, latlng.lng)}<br><strong>Profondeur ${fmtDepth(feature.properties.depth_m)}</strong>`,
        ),
    }).addTo(points);
  } else {
    errors.push(String(pointsResult.reason?.message ?? pointsResult.reason));
  }

  // --- F10 : profondeur au clic en mode raster, lue dans la grille numérique. ---
  if (gridResult.status === "fulfilled") {
    const grid = gridResult.value;
    map.on("click", (e: L.LeafletMouseEvent) => {
      if (mode !== MODE_RASTER) return;
      const { lat, lng } = e.latlng;
      const d = grid.depthAt(lat, lng);
      const texte =
        d === null
          ? "Hors de l'emprise des données"
          : Number.isNaN(d)
            ? "Terre (pas de profondeur)"
            : `<strong>Profondeur ${fmtDepth(d)}</strong><br><small>grille ${meta.grille.lignes} × ${meta.grille.colonnes}, cellule la plus proche</small>`;
      L.popup().setLatLng(e.latlng).setContent(`${fmtCoord(lat, lng)}<br>${texte}`).openOn(map);
    });
  } else {
    errors.push(String(gridResult.reason?.message ?? gridResult.reason));
  }

  if (errors.length) {
    setStatus(errors.join(" "), true);
  } else {
    setStatus(
      `${meta.source.version} — ${meta.points.nombre_points.toLocaleString("fr-FR")} points. ` +
        "Cliquez sur la carte ou un point pour lire la profondeur. " +
        "Données compilées, non destinées à la navigation.",
    );
  }
}

main().catch((err: unknown) => {
  console.error(err);
  const msg = err instanceof Error ? err.message : String(err);
  setStatus(msg, true);
});
