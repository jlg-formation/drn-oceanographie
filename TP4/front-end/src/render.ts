/** Rendu des résultats : masque coloré, carte d'incertitude, statistiques, exports PNG. */

export type RGB = [number, number, number];

export function hexRgb(code: string): RGB {
  const c = code.replace("#", "");
  return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
}

// Couleur de mise en évidence des pixels incertains (distincte de la terre et de la mer).
const INCERTAIN: RGB = [255, 214, 10];

function preparer(canvas: HTMLCanvasElement, largeur: number, hauteur: number): CanvasRenderingContext2D {
  canvas.width = largeur;
  canvas.height = hauteur;
  return canvas.getContext("2d")!;
}

/** Masque opaque : vert si P(terre) >= seuil, bleu sinon (F18). */
export function dessinerMasque(
  canvas: HTMLCanvasElement,
  prob: Float32Array,
  largeur: number,
  hauteur: number,
  seuil: number,
  terre: RGB,
  mer: RGB,
): void {
  const ctx = preparer(canvas, largeur, hauteur);
  const img = ctx.createImageData(largeur, hauteur);
  const d = img.data;
  for (let i = 0; i < prob.length; i++) {
    const c = prob[i] >= seuil ? terre : mer;
    d[4 * i] = c[0];
    d[4 * i + 1] = c[1];
    d[4 * i + 2] = c[2];
    d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Carte d'incertitude (F23) : seuls les pixels dont la probabilité est dans
 * [min, max] sont colorés, d'autant plus opaques qu'elle est proche de 50 %.
 */
export function dessinerIncertitude(
  canvas: HTMLCanvasElement,
  prob: Float32Array,
  largeur: number,
  hauteur: number,
  min: number,
  max: number,
): void {
  const ctx = preparer(canvas, largeur, hauteur);
  const img = ctx.createImageData(largeur, hauteur);
  const d = img.data;
  const demi = Math.max(0.5 - min, max - 0.5);
  for (let i = 0; i < prob.length; i++) {
    const p = prob[i];
    if (p < min || p > max) continue;
    d[4 * i] = INCERTAIN[0];
    d[4 * i + 1] = INCERTAIN[1];
    d[4 * i + 2] = INCERTAIN[2];
    d[4 * i + 3] = Math.round(255 * (0.35 + 0.65 * (1 - Math.abs(p - 0.5) / demi)));
  }
  ctx.putImageData(img, 0, 0);
}

export interface Statistiques {
  terrePct: number;
  merPct: number;
  incertainPct: number;
}

/** Pourcentages de pixels : terre + mer = 100 % (F22). */
export function statistiques(prob: Float32Array, seuil: number, min: number, max: number): Statistiques {
  let terre = 0;
  let incertain = 0;
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] >= seuil) terre++;
    if (prob[i] >= min && prob[i] <= max) incertain++;
  }
  const terrePct = (100 * terre) / prob.length;
  return { terrePct, merPct: 100 - terrePct, incertainPct: (100 * incertain) / prob.length };
}

/** Image originale + masque à l'opacité choisie, à la taille d'origine (F24). */
export function composerSuperposition(
  image: HTMLCanvasElement,
  masque: HTMLCanvasElement,
  opacite: number,
): HTMLCanvasElement {
  const out = document.createElement("canvas");
  const ctx = preparer(out, image.width, image.height);
  ctx.drawImage(image, 0, 0);
  ctx.globalAlpha = opacite;
  ctx.drawImage(masque, 0, 0);
  return out;
}

export function telecharger(canvas: HTMLCanvasElement, nom: string): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Export PNG impossible."));
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = nom;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      resolve();
    }, "image/png");
  });
}
