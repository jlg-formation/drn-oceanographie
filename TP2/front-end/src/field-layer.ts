import L from "leaflet";
import type { TemperatureModel } from "./inference";
import type { Coastline, Emprise } from "./coastline";
import { thermalColor } from "./colormap";

interface FieldOptions {
  model: TemperatureModel;
  coast: Coastline;
  emprise: Emprise;
  vmin: number;
  vmax: number;
  /** Pas d'échantillonnage en pixels (plus petit = plus fin, plus coûteux). */
  step?: number;
  opacity?: number;
}

/**
 * Calque Leaflet qui échantillonne le viewport, interroge le réseau ONNX et
 * peint le champ de température sur un canvas superposé à la carte.
 *
 * Le rendu se fait à basse résolution (grille `step` px) puis est étiré avec
 * lissage : on obtient un champ continu, redessiné à chaque déplacement.
 */
export class NeuralFieldLayer extends L.Layer {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private low: HTMLCanvasElement = document.createElement("canvas");
  private options2: Required<FieldOptions>;
  private rendering = false;
  private pending = false;

  constructor(options: FieldOptions) {
    super();
    this.options2 = {
      step: 8,
      opacity: 0.8,
      ...options,
    } as Required<FieldOptions>;
  }

  onAdd(map: L.Map): this {
    this.canvas = L.DomUtil.create("canvas", "neural-field leaflet-zoom-animated");
    this.ctx = this.canvas.getContext("2d")!;
    this.canvas.style.opacity = String(this.options2.opacity);
    map.getPanes().overlayPane.appendChild(this.canvas);

    map.on("moveend zoomend resize", this.reset, this);
    if (map.options.zoomAnimation) {
      map.on("zoomanim", this.animateZoom, this);
    }
    this.reset();
    return this;
  }

  onRemove(map: L.Map): this {
    map.off("moveend zoomend resize", this.reset, this);
    map.off("zoomanim", this.animateZoom, this);
    this.canvas.remove();
    return this;
  }

  setModel(model: TemperatureModel): void {
    this.options2.model = model;
    void this.render();
  }

  setRange(vmin: number, vmax: number): void {
    this.options2.vmin = vmin;
    this.options2.vmax = vmax;
    void this.render();
  }

  setOpacity(opacity: number): void {
    this.options2.opacity = opacity;
    if (this.canvas) this.canvas.style.opacity = String(opacity);
  }

  private animateZoom(e: L.ZoomAnimEvent): void {
    const map = this._map;
    const scale = map.getZoomScale(e.zoom, map.getZoom());
    // Repositionne/étire le canvas pendant l'animation de zoom.
    const offset = map
      // @ts-expect-error méthode interne de Leaflet, stable en pratique
      ._latLngToNewLayerPoint(map.containerPointToLatLng([0, 0]), e.zoom, e.center);
    L.DomUtil.setTransform(this.canvas, offset, scale);
  }

  private reset(): void {
    const map = this._map;
    const size = map.getSize();
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this.canvas, topLeft);
    // Annule tout transform résiduel laissé par l'animation de zoom.
    this.canvas.style.transform = `translate(${topLeft.x}px, ${topLeft.y}px)`;
    if (this.canvas.width !== size.x || this.canvas.height !== size.y) {
      this.canvas.width = size.x;
      this.canvas.height = size.y;
    }
    void this.render();
  }

  /** Génère le champ ; coalescing pour éviter les rendus concurrents. */
  private async render(): Promise<void> {
    if (!this._map || !this.canvas) return;
    if (this.rendering) {
      this.pending = true;
      return;
    }
    this.rendering = true;

    const map = this._map;
    const size = map.getSize();
    const step = this.options2.step;
    const cols = Math.max(2, Math.ceil(size.x / step));
    const rows = Math.max(2, Math.ceil(size.y / step));

    // Coordonnées (lon, lat) au centre de chaque cellule basse résolution.
    const lons = new Float32Array(cols * rows);
    const lats = new Float32Array(cols * rows);
    for (let j = 0; j < rows; j++) {
      const y = (j + 0.5) * (size.y / rows);
      for (let i = 0; i < cols; i++) {
        const x = (i + 0.5) * (size.x / cols);
        const ll = map.containerPointToLatLng([x, y]);
        const idx = j * cols + i;
        lons[idx] = ll.lng;
        lats[idx] = ll.lat;
      }
    }

    let temps: Float32Array;
    try {
      temps = await this.options2.model.predict(lons, lats);
    } catch (err) {
      console.error("Inférence échouée", err);
      this.rendering = false;
      return;
    }

    const { vmin, vmax } = this.options2;
    const span = vmax - vmin || 1;
    const img = this.ctx.createImageData(cols, rows);
    for (let k = 0; k < temps.length; k++) {
      const t = (temps[k] - vmin) / span;
      const [r, g, b] = thermalColor(t);
      img.data[k * 4] = r;
      img.data[k * 4 + 1] = g;
      img.data[k * 4 + 2] = b;
      img.data[k * 4 + 3] = 255;
    }

    // Grille basse résolution du champ, étirée avec lissage.
    this.low.width = cols;
    this.low.height = rows;
    this.low.getContext("2d")!.putImageData(img, 0, 0);

    const emp = this.options2.emprise;
    const tl = map.latLngToContainerPoint([emp.latMax, emp.lonMin]);
    const br = map.latLngToContainerPoint([emp.latMin, emp.lonMax]);

    this.ctx.clearRect(0, 0, size.x, size.y);
    this.ctx.save();
    // Clip au rectangle d'emprise du modèle (pas de champ hors domaine).
    this.ctx.beginPath();
    this.ctx.rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    this.ctx.clip();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = "high";
    this.ctx.drawImage(this.low, 0, 0, cols, rows, 0, 0, size.x, size.y);
    // Efface la terre le long des côtes réelles (trait net).
    this.ctx.globalCompositeOperation = "destination-out";
    this.options2.coast.buildLandPath(this.ctx, (lon, lat) =>
      map.latLngToContainerPoint([lat, lon]),
    );
    this.ctx.fill();
    this.ctx.globalCompositeOperation = "source-over";
    this.ctx.restore();

    this.rendering = false;
    if (this.pending) {
      this.pending = false;
      void this.render();
    }
  }
}
