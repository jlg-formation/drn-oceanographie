/** Petits graphiques de courbes sur canvas : apprentissage du VAE et du GAN (F19). */

export interface Serie {
  nom: string;
  couleur: string;
  x: number[];
  y: number[];
  tirets?: number[];
  epaisseur?: number;
}

export interface OptionsGraphique {
  titre: string;
  xLabel: string;
  yLabel: string;
  yMin?: number;
  yMax?: number;
  repereX?: { valeur: number; texte: string };
  repereY?: number;
}

function graduations(min: number, max: number, n = 5): number[] {
  const brut = (max - min) / n;
  const p = 10 ** Math.floor(Math.log10(brut));
  const pas = [1, 2, 5, 10].map((k) => k * p).find((k) => k >= brut) ?? brut;
  const res = [];
  for (let v = Math.ceil(min / pas) * pas; v <= max + 1e-9; v += pas) res.push(+v.toFixed(10));
  return res;
}

export function dessinerCourbes(canvas: HTMLCanvasElement, series: Serie[], o: OptionsGraphique): void {
  const dpr = window.devicePixelRatio || 1;
  const l = canvas.clientWidth || 500;
  const h = canvas.clientHeight || 260;
  canvas.width = l * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, l, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, l, h);

  const g = { gauche: 56, droite: 12, haut: 30, bas: 40 + 18 * Math.ceil(series.length / 3) };
  const xs = series.flatMap((s) => s.x);
  const ys = series.flatMap((s) => s.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) return;
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  let y0 = o.yMin ?? Math.min(...ys);
  let y1 = o.yMax ?? Math.max(...ys);
  if (o.yMin === undefined || o.yMax === undefined) {
    const marge = (y1 - y0) * 0.05 || 1;
    if (o.yMin === undefined) y0 -= marge;
    if (o.yMax === undefined) y1 += marge;
  }
  const px = (x: number) => g.gauche + ((x - x0) / (x1 - x0 || 1)) * (l - g.gauche - g.droite);
  const py = (y: number) => h - g.bas - ((y - y0) / (y1 - y0 || 1)) * (h - g.haut - g.bas);

  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#0f172a";
  ctx.fillText(o.titre, g.gauche, 18);
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px system-ui, sans-serif";
  for (const v of graduations(y0, y1)) {
    ctx.beginPath();
    ctx.moveTo(g.gauche, py(v));
    ctx.lineTo(l - g.droite, py(v));
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(String(v), g.gauche - 6, py(v) + 4);
  }
  for (const v of graduations(x0, x1, 6)) {
    ctx.textAlign = "center";
    ctx.fillText(String(v), px(v), h - g.bas + 16);
  }
  ctx.textAlign = "center";
  ctx.fillText(o.xLabel, (g.gauche + l - g.droite) / 2, h - g.bas + 30);
  ctx.save();
  ctx.translate(14, (g.haut + h - g.bas) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(o.yLabel, 0, 0);
  ctx.restore();

  if (o.repereY !== undefined) {
    ctx.strokeStyle = "#94a3b8";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(g.gauche, py(o.repereY));
    ctx.lineTo(l - g.droite, py(o.repereY));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (o.repereX) {
    ctx.strokeStyle = "#475569";
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(px(o.repereX.valeur), g.haut);
    ctx.lineTo(px(o.repereX.valeur), h - g.bas);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#475569";
    // Libellé à gauche du repère s'il est proche du bord droit.
    const aGauche = px(o.repereX.valeur) + ctx.measureText(o.repereX.texte).width + 8 > l - g.droite;
    ctx.textAlign = aGauche ? "right" : "left";
    ctx.fillText(o.repereX.texte, px(o.repereX.valeur) + (aGauche ? -4 : 4), g.haut + 10);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(g.gauche, g.haut, l - g.gauche - g.droite, h - g.haut - g.bas);
  ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.couleur;
    ctx.lineWidth = s.epaisseur ?? 1.6;
    ctx.setLineDash(s.tirets ?? []);
    ctx.beginPath();
    s.x.forEach((x, i) => (i ? ctx.lineTo(px(x), py(s.y[i])) : ctx.moveTo(px(x), py(s.y[i]))));
    ctx.stroke();
  }
  ctx.restore();
  ctx.setLineDash([]);

  // Légende sous le graphique, 3 entrées par ligne.
  ctx.textAlign = "left";
  const largeurEntree = (l - g.gauche) / 3;
  series.forEach((s, k) => {
    const x = g.gauche + (k % 3) * largeurEntree;
    const y = h - g.bas + 48 + Math.floor(k / 3) * 18;
    ctx.strokeStyle = s.couleur;
    ctx.lineWidth = 2;
    ctx.setLineDash(s.tirets ?? []);
    ctx.beginPath();
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x + 22, y - 4);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#334155";
    ctx.fillText(s.nom, x + 28, y);
  });
}
