/**
 * Cartes 2D (F29) : palette bathymétrique, légende graduée en mètres, lecture au survol.
 *
 * Convention : ligne 0 = nord, colonne 0 = ouest (comme dans le pipeline Python).
 * Les reliefs réels et synthétiques sont affichés en **relief relatif** (écart à
 * la moyenne du patch) avec la même palette, pour être comparables (§9.3).
 */
import { TAILLE } from "./donnees";

export type Rgb = [number, number, number];

export function hexRgb(code: string): Rgb {
  const c = code.replace("#", "");
  return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
}

/** Palette divergente pour les erreurs (bleu : sous-estimé, rouge : surestimé). */
export const PALETTE_ERREUR = ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"];

let palette: Rgb[] = [];

export function definirPalette(couleurs: string[]): void {
  palette = couleurs.map(hexRgb);
}

/** Couleur d'une valeur t ∈ [0, 1] le long d'une palette (interpolation linéaire, saturée). */
export function couleur(t: number, p: Rgb[] = palette): Rgb {
  const x = Math.min(1, Math.max(0, t)) * (p.length - 1);
  const i = Math.min(p.length - 2, Math.floor(x));
  const f = x - i;
  return [0, 1, 2].map((k) => Math.round(p[i][k] + f * (p[i + 1][k] - p[i][k]))) as Rgb;
}

export interface Echelle {
  min: number;
  max: number;
}

/** Échelle symétrique adaptée à un ou plusieurs reliefs (arrondie à 50 m). */
export function echelleAuto(reliefs: Float32Array[]): Echelle {
  let m = 50;
  for (const r of reliefs) for (const v of r) m = Math.max(m, Math.abs(v));
  m = Math.ceil(m / 50) * 50;
  return { min: -m, max: m };
}

/** Dessine une grille 64 × 64 dans un canvas (agrandissement sans lissage). */
export function dessinerCarte(canvas: HTMLCanvasElement, r: Float32Array, e: Echelle, p?: string[]): void {
  const rgb = p ? p.map(hexRgb) : palette;
  const tmp = document.createElement("canvas");
  tmp.width = TAILLE;
  tmp.height = TAILLE;
  const ctx = tmp.getContext("2d")!;
  const img = ctx.createImageData(TAILLE, TAILLE);
  for (let i = 0; i < r.length; i++) {
    const c = couleur((r[i] - e.min) / (e.max - e.min), rgb);
    img.data.set([c[0], c[1], c[2], 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  const out = canvas.getContext("2d")!;
  out.imageSmoothingEnabled = false;
  out.clearRect(0, 0, canvas.width, canvas.height);
  out.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

/** Légende verticale graduée (F29). */
export function dessinerLegende(canvas: HTMLCanvasElement, e: Echelle, unite: string, p?: string[]): void {
  const rgb = p ? p.map(hexRgb) : palette;
  const dpr = window.devicePixelRatio || 1;
  const l = canvas.clientWidth || 70;
  const h = canvas.clientHeight || 300;
  canvas.width = l * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, l, h);
  const haut = 10;
  const bas = h - 10;
  for (let y = haut; y < bas; y++) {
    const c = couleur(1 - (y - haut) / (bas - haut), rgb);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(0, y, 16, 1);
  }
  ctx.strokeStyle = "#94a3b8";
  ctx.strokeRect(0.5, haut + 0.5, 15, bas - haut - 1);
  ctx.fillStyle = "#334155";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const n = 6;
  for (let k = 0; k <= n; k++) {
    const v = e.max - (k / n) * (e.max - e.min);
    const y = haut + (k / n) * (bas - haut);
    ctx.fillRect(16, y, 4, 1);
    ctx.fillText(`${v > 0 ? "+" : ""}${Math.round(v)}`, 23, y);
  }
  ctx.save();
  ctx.translate(l - 4, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(unite, 0, 0);
  ctx.restore();
}

/** Pixel (ligne, colonne) sous le pointeur, ou null hors carte. */
export function pixelSousPointeur(canvas: HTMLCanvasElement, ev: MouseEvent): [number, number] | null {
  const b = canvas.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - b.left) / b.width) * TAILLE);
  const y = Math.floor(((ev.clientY - b.top) / b.height) * TAILLE);
  return x >= 0 && x < TAILLE && y >= 0 && y < TAILLE ? [y, x] : null;
}

/** Image PNG composée : carte, légende, titre et mention « synthétique » (F36). */
export function composerPng(r: Float32Array, e: Echelle, titre: string, mention: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 640;
  c.height = 580;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#0f172a";
  ctx.font = "bold 16px system-ui, sans-serif";
  ctx.fillText(titre, 20, 28);
  const carte = document.createElement("canvas");
  carte.width = 512;
  carte.height = 512;
  dessinerCarte(carte, r, e);
  ctx.drawImage(carte, 20, 44);
  const leg = document.createElement("canvas");
  leg.style.width = "90px";
  leg.style.height = "512px";
  Object.defineProperty(leg, "clientWidth", { value: 90 });
  Object.defineProperty(leg, "clientHeight", { value: 512 });
  dessinerLegende(leg, e, "relief relatif (m)");
  ctx.drawImage(leg, 540, 44, 90, 512);
  ctx.fillStyle = "#92400e";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(mention, 20, 572);
  return c;
}
