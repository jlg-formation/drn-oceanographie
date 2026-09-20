import type { Histogram } from "./inference";

export interface Series {
  /** Densité modèle p(T) évaluée sur ``grid`` (même longueur que grid). */
  modelGrid: Float32Array;
  modelDensity: Float32Array;
  /** Histogramme d'un tirage synthétique (optionnel). */
  sampleDensity?: number[];
}

interface Theme {
  bg: string;
  axis: string;
  histFill: string;
  histStroke: string;
  curve: string;
  sample: string;
  text: string;
}

const THEME: Theme = {
  bg: "#0b1220",
  axis: "#334155",
  histFill: "rgba(148, 163, 184, 0.35)",
  histStroke: "rgba(148, 163, 184, 0.6)",
  curve: "#38bdf8",
  sample: "rgba(250, 205, 60, 0.9)",
  text: "#cbd5e1",
};

const PADDING = { left: 52, right: 16, top: 16, bottom: 34 };

/** Trace l'histogramme réel, la densité du modèle et un histogramme échantillonné. */
export class DistributionChart {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly hist: Histogram,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D indisponible");
    this.ctx = ctx;
  }

  private xRange(): [number, number] {
    const edges = this.hist.bin_edges;
    return [edges[0], edges[edges.length - 1]];
  }

  private yMax(series: Series): number {
    let m = Math.max(...this.hist.density);
    for (const v of series.modelDensity) m = Math.max(m, v);
    if (series.sampleDensity) m = Math.max(m, ...series.sampleDensity);
    return m * 1.08;
  }

  draw(series: Series): void {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = THEME.bg;
    ctx.fillRect(0, 0, cssW, cssH);

    const plotW = cssW - PADDING.left - PADDING.right;
    const plotH = cssH - PADDING.top - PADDING.bottom;
    const [xMin, xMax] = this.xRange();
    const yMax = this.yMax(series);

    const sx = (t: number): number =>
      PADDING.left + ((t - xMin) / (xMax - xMin)) * plotW;
    const sy = (d: number): number =>
      PADDING.top + plotH - (d / yMax) * plotH;

    this.drawAxes(sx, sy, xMin, xMax, yMax, plotW, plotH);
    this.drawHistogram(sx, sy);
    if (series.sampleDensity) this.drawSampleHistogram(series.sampleDensity, sx, sy);
    this.drawCurve(series.modelGrid, series.modelDensity, sx, sy);
    this.drawLegend(cssW, series.sampleDensity !== undefined);
  }

  private drawAxes(
    sx: (t: number) => number,
    sy: (d: number) => number,
    xMin: number,
    xMax: number,
    yMax: number,
    plotW: number,
    plotH: number,
  ): void {
    const { ctx } = this;
    ctx.strokeStyle = THEME.axis;
    ctx.fillStyle = THEME.text;
    ctx.lineWidth = 1;
    ctx.font = "11px system-ui, sans-serif";

    // Axe X (températures).
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top + plotH);
    ctx.lineTo(PADDING.left + plotW, PADDING.top + plotH);
    ctx.stroke();
    const nTicks = 6;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= nTicks; i++) {
      const t = xMin + ((xMax - xMin) * i) / nTicks;
      const x = sx(t);
      ctx.strokeStyle = THEME.axis;
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top + plotH);
      ctx.lineTo(x, PADDING.top + plotH + 4);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(1)}`, x, PADDING.top + plotH + 6);
    }
    ctx.fillText("température (°C)", PADDING.left + plotW / 2, PADDING.top + plotH + 20);

    // Axe Y (densité).
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top);
    ctx.lineTo(PADDING.left, PADDING.top + plotH);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const d = (yMax * i) / yTicks;
      const y = sy(d);
      ctx.fillText(d.toFixed(2), PADDING.left - 6, y);
    }
  }

  private drawHistogram(sx: (t: number) => number, sy: (d: number) => number): void {
    const { ctx, hist } = this;
    ctx.fillStyle = THEME.histFill;
    ctx.strokeStyle = THEME.histStroke;
    ctx.lineWidth = 1;
    for (let i = 0; i < hist.density.length; i++) {
      const x0 = sx(hist.bin_edges[i]);
      const x1 = sx(hist.bin_edges[i + 1]);
      const y = sy(hist.density[i]);
      const yBase = sy(0);
      ctx.fillRect(x0, y, x1 - x0, yBase - y);
      ctx.strokeRect(x0, y, x1 - x0, yBase - y);
    }
  }

  private drawSampleHistogram(
    density: number[],
    sx: (t: number) => number,
    sy: (d: number) => number,
  ): void {
    const { ctx, hist } = this;
    ctx.strokeStyle = THEME.sample;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < density.length; i++) {
      const xc = sx(hist.bin_centers[i]);
      const y = sy(density[i]);
      if (i === 0) ctx.moveTo(xc, y);
      else ctx.lineTo(xc, y);
    }
    ctx.stroke();
  }

  private drawCurve(
    grid: Float32Array,
    density: Float32Array,
    sx: (t: number) => number,
    sy: (d: number) => number,
  ): void {
    const { ctx } = this;
    ctx.strokeStyle = THEME.curve;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < grid.length; i++) {
      const x = sx(grid[i]);
      const y = sy(density[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  private drawLegend(cssW: number, hasSample: boolean): void {
    const { ctx } = this;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = "12px system-ui, sans-serif";
    let y = PADDING.top + 8;
    const x = cssW - PADDING.right - 150;

    const entry = (color: string, label: string, line: boolean): void => {
      ctx.fillStyle = color;
      if (line) ctx.fillRect(x, y - 1, 18, 3);
      else ctx.fillRect(x, y - 6, 18, 12);
      ctx.fillStyle = THEME.text;
      ctx.fillText(label, x + 24, y);
      y += 18;
    };

    entry(THEME.histFill, "histogramme réel", false);
    entry(THEME.curve, "densité MDN", true);
    if (hasSample) entry(THEME.sample, "échantillons", true);
  }
}

/** Histogramme (densité) d'un tirage, aligné sur les intervalles de référence. */
export function sampleToDensity(samples: Float32Array, hist: Histogram): number[] {
  const edges = hist.bin_edges;
  const counts = new Array(hist.density.length).fill(0);
  for (const v of samples) {
    if (v < edges[0] || v > edges[edges.length - 1]) continue;
    let k = Math.floor((v - edges[0]) / hist.bin_width);
    if (k >= counts.length) k = counts.length - 1;
    counts[k]++;
  }
  return counts.map((c) => c / (samples.length * hist.bin_width));
}
