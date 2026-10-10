/**
 * Découpage d'une image en fenêtres et recollement pondéré (F17).
 *
 * Reproduction exacte de `positions`, `indices_miroir` et `poids_fenetre`
 * de TP4/unet.py : le site et l'évaluation Python segmentent une image entière
 * de la même manière.
 */

/** Débuts des fenêtres le long d'un axe ; la dernière est collée au bord. */
export function positions(longueur: number, taille: number, pas: number): number[] {
  if (longueur <= taille) return [0];
  const debuts: number[] = [];
  for (let d = 0; d < longueur - taille; d += pas) debuts.push(d);
  debuts.push(longueur - taille);
  return debuts;
}

/** Indice i (éventuellement hors image) replié en miroir dans [0, longueur). */
export function miroir(i: number, longueur: number): number {
  if (longueur === 1) return 0;
  const periode = 2 * (longueur - 1);
  const j = i % periode;
  return j < longueur ? j : periode - j;
}

/** Pondération 2D « tente aplatie » (taille x taille), 1 au centre. */
export function poidsFenetre(taille: number, rampe: number): Float32Array {
  const r = Math.max(rampe, 1);
  const w1 = new Float32Array(taille);
  for (let i = 0; i < taille; i++) {
    const d = Math.min(i, taille - 1 - i) + 1;
    w1[i] = Math.min(Math.max(d / r, 1 / r), 1);
  }
  const w = new Float32Array(taille * taille);
  for (let y = 0; y < taille; y++) {
    for (let x = 0; x < taille; x++) w[y * taille + x] = w1[y] * w1[x];
  }
  return w;
}

export interface Fenetre {
  r: number;
  c: number;
}

/** Toutes les fenêtres couvrant une image (complétée en miroir si trop petite). */
export function fenetres(largeur: number, hauteur: number, taille: number, pas: number): Fenetre[] {
  const hp = Math.max(hauteur, taille);
  const wp = Math.max(largeur, taille);
  const out: Fenetre[] = [];
  for (const r of positions(hp, taille, pas)) {
    for (const c of positions(wp, taille, pas)) out.push({ r, c });
  }
  return out;
}

/**
 * Remplit `dest` (offset `offset`) avec la fenêtre (r, c) au format CHW float32 [0, 1],
 * à partir de pixels RGBA. Les pixels hors image sont pris en miroir.
 */
export function extraireFenetre(
  rgba: Uint8ClampedArray,
  largeur: number,
  hauteur: number,
  f: Fenetre,
  taille: number,
  dest: Float32Array,
  offset: number,
): void {
  const plan = taille * taille;
  for (let y = 0; y < taille; y++) {
    const sy = miroir(f.r + y, hauteur);
    for (let x = 0; x < taille; x++) {
      const sx = miroir(f.c + x, largeur);
      const s = (sy * largeur + sx) * 4;
      const d = offset + y * taille + x;
      dest[d] = rgba[s] / 255;
      dest[d + plan] = rgba[s + 1] / 255;
      dest[d + 2 * plan] = rgba[s + 2] / 255;
    }
  }
}
