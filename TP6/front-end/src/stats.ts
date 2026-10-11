/**
 * Statistiques d'un relief (F35). Mêmes définitions que ``terrain.indicateurs`` en Python.
 */
import { TAILLE } from "./donnees";

export interface Statistiques {
  min: number;
  max: number;
  moyenne: number;
  amplitude: number;
  ecartType: number;
  /** Écart-type du relief moins sa moyenne glissante 3 × 3 (bords répliqués). */
  rugosite: number;
  /** Pente moyenne (degrés), différences centrées sur les pixels intérieurs. */
  penteMoyDeg: number;
}

export function statistiques(r: Float32Array, pixelKm: number): Statistiques {
  let min = Infinity;
  let max = -Infinity;
  let s = 0;
  let s2 = 0;
  for (const v of r) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    s += v;
    s2 += v * v;
  }
  const n = r.length;
  const moyenne = s / n;
  const at = (y: number, x: number) =>
    r[Math.min(TAILLE - 1, Math.max(0, y)) * TAILLE + Math.min(TAILLE - 1, Math.max(0, x))];

  let sd = 0;
  let sd2 = 0;
  for (let y = 0; y < TAILLE; y++) {
    for (let x = 0; x < TAILLE; x++) {
      let m3 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) m3 += at(y + dy, x + dx);
      const d = r[y * TAILLE + x] - m3 / 9;
      sd += d;
      sd2 += d * d;
    }
  }
  let pente = 0;
  const pas = 2 * pixelKm * 1000;
  for (let y = 1; y < TAILLE - 1; y++) {
    for (let x = 1; x < TAILLE - 1; x++) {
      const gy = (r[(y + 1) * TAILLE + x] - r[(y - 1) * TAILLE + x]) / pas;
      const gx = (r[y * TAILLE + x + 1] - r[y * TAILLE + x - 1]) / pas;
      pente += (Math.atan(Math.hypot(gx, gy)) * 180) / Math.PI;
    }
  }
  return {
    min,
    max,
    moyenne,
    amplitude: max - min,
    ecartType: Math.sqrt(Math.max(0, s2 / n - moyenne * moyenne)),
    rugosite: Math.sqrt(Math.max(0, sd2 / n - (sd / n) ** 2)),
    penteMoyDeg: pente / ((TAILLE - 2) * (TAILLE - 2)),
  };
}

/** RMSE et erreur absolue maximale (m) entre deux grilles. */
export function erreurs(a: Float32Array, b: Float32Array): { rmse: number; max: number; carte: Float32Array } {
  const carte = Float32Array.from(a, (v, i) => v - b[i]);
  let s = 0;
  let m = 0;
  for (const v of carte) {
    s += v * v;
    m = Math.max(m, Math.abs(v));
  }
  return { rmse: Math.sqrt(s / carte.length), max: m, carte };
}
