/**
 * Palette bathymétrique, lue dans metadata.json : la même source que celle
 * utilisée par Python pour colorer le raster, donc une même profondeur reçoit
 * la même couleur dans les deux modes.
 */

import type { Palier } from "./data";

type RGB = [number, number, number];

function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB;
}

export class DepthPalette {
  private readonly depths: number[];
  private readonly colors: RGB[];

  constructor(paliers: Palier[]) {
    this.depths = paliers.map((p) => p.depth_m);
    this.colors = paliers.map((p) => hexToRgb(p.couleur));
  }

  get maxDepth(): number {
    return this.depths[this.depths.length - 1];
  }

  get stops(): number[] {
    return [...this.depths];
  }

  /** Couleur CSS d'une profondeur (interpolation linéaire, saturation hors plage). */
  color(depth: number): string {
    const d = Math.min(Math.max(depth, this.depths[0]), this.maxDepth);
    let i = 1;
    while (i < this.depths.length - 1 && d > this.depths[i]) i++;
    const f = (d - this.depths[i - 1]) / (this.depths[i] - this.depths[i - 1]);
    const a = this.colors[i - 1];
    const b = this.colors[i];
    // Math.round = arrondi « demi vers le haut », comme floor(x + 0.5) côté Python.
    const c = a.map((v, k) => Math.round(v + f * (b[k] - v)));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }

  /** Dégradé CSS vertical (0 m en haut), paliers placés à l'échelle des profondeurs. */
  cssGradient(): string {
    const parts = this.depths.map(
      (d, i) =>
        `rgb(${this.colors[i].join(", ")}) ${((d / this.maxDepth) * 100).toFixed(2)}%`,
    );
    return `linear-gradient(to bottom, ${parts.join(", ")})`;
  }
}
