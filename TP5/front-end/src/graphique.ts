/**
 * Dessin des graphiques sur canvas (F18, F21, F22, F25, §7.3, §7.4).
 *
 * - graphique principal : 48 h d'historique, instant de prévision H, 24 h prévues,
 *   réalité masquée ou révélée, panneau des écarts une fois la réalité révélée ;
 * - frise : la série complète (5 ans), les ensembles, les lacunes et la fenêtre choisie ;
 * - petits graphiques de courbes (erreur par échéance, apprentissage).
 *
 * Chaque courbe a une couleur, un style de trait ET un marqueur (accessibilité).
 */

import type { Chronologie, Courbe, Style } from "./donnees";

const FOND = "#ffffff";
const ENCRE = "#0f172a";
const GRIS = "#64748b";
const GRILLE = "#e2e8f0";
const ZONE_PREVISION = "#eff6ff";
const POLICE = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

// --------------------------------------------------------------------------
// Outils
// --------------------------------------------------------------------------

/** Ajuste le canvas à sa taille CSS et à la densité de l'écran ; renvoie le contexte prêt. */
export function preparer(canvas: HTMLCanvasElement, hauteurCss?: number): CanvasRenderingContext2D {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const l = canvas.clientWidth || 800;
  const h = hauteurCss ?? (canvas.clientHeight || 400);
  canvas.width = Math.round(l * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = FOND;
  ctx.fillRect(0, 0, l, h);
  return ctx;
}

/** Graduations « rondes » couvrant [min, max]. */
export function graduations(min: number, max: number, cible = 6): number[] {
  const brut = (max - min) / Math.max(1, cible);
  const p = 10 ** Math.floor(Math.log10(brut));
  const pas = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= brut) ?? 10 * p;
  const res: number[] = [];
  for (let v = Math.ceil(min / pas) * pas; v <= max + 1e-9; v += pas) res.push(Number(v.toFixed(10)));
  return res;
}

export function marqueur(ctx: CanvasRenderingContext2D, type: Style["marqueur"], x: number, y: number, r = 3.5): void {
  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  switch (type) {
    case "rond":
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
      break;
    case "cercle":
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = FOND;
      ctx.fill();
      ctx.stroke();
      break;
    case "triangle":
      ctx.moveTo(x, y - r * 1.2);
      ctx.lineTo(x + r * 1.1, y + r * 0.8);
      ctx.lineTo(x - r * 1.1, y + r * 0.8);
      ctx.closePath();
      ctx.fill();
      break;
    case "carre":
      ctx.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7);
      ctx.fill();
      break;
    case "losange":
      ctx.moveTo(x, y - r * 1.25);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r * 1.25);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.fill();
      break;
    case "croix":
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r);
      ctx.lineTo(x - r, y + r);
      ctx.lineWidth = 1.6;
      ctx.stroke();
      break;
    case "plus":
      ctx.moveTo(x - r * 1.2, y);
      ctx.lineTo(x + r * 1.2, y);
      ctx.moveTo(x, y - r * 1.2);
      ctx.lineTo(x, y + r * 1.2);
      ctx.lineWidth = 1.6;
      ctx.stroke();
      break;
    case "aucun":
      break;
  }
  ctx.restore();
}

/** Trace une courbe (les valeurs null coupent le trait) avec ses marqueurs. */
function tracer(
  ctx: CanvasRenderingContext2D,
  points: [number, number | null][],
  style: Style,
  largeur: number,
  marqueurs = true,
): void {
  ctx.save();
  ctx.strokeStyle = style.couleur;
  ctx.fillStyle = style.couleur;
  ctx.lineWidth = largeur;
  ctx.lineJoin = "round";
  ctx.setLineDash(style.tirets);
  ctx.beginPath();
  let leve = true;
  for (const [x, y] of points) {
    if (y === null) {
      leve = true;
      continue;
    }
    if (leve) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    leve = false;
  }
  ctx.stroke();
  if (marqueurs) for (const [x, y] of points) if (y !== null) marqueur(ctx, style.marqueur, x, y);
  ctx.restore();
}

/** Petit échantillon de style (légende HTML). */
export function echantillon(style: Style, largeur = 42, hauteur = 14): HTMLCanvasElement {
  const c = document.createElement("canvas");
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  c.width = largeur * dpr;
  c.height = hauteur * dpr;
  c.style.width = `${largeur}px`;
  c.style.height = `${hauteur}px`;
  const ctx = c.getContext("2d")!;
  ctx.scale(dpr, dpr);
  tracer(ctx, [[2, hauteur / 2], [largeur - 2, hauteur / 2]], { ...style, marqueur: "aucun" }, 2.2);
  marqueur(ctx, style.marqueur, largeur / 2, hauteur / 2);
  return c;
}

const pad2 = (n: number) => String(n).padStart(2, "0");
export const fmtDate = (d: Date) => `${pad2(d.getUTCDate())}/${pad2(d.getUTCMonth() + 1)}/${d.getUTCFullYear()}`;
export const fmtHeure = (d: Date) => `${pad2(d.getUTCHours())}:00`;
export const fmtM = (v: number) => `${v.toFixed(2).replace(".", ",")} m`;

// --------------------------------------------------------------------------
// Graphique principal
// --------------------------------------------------------------------------

export interface Scene {
  chrono: Chronologie;
  ancre: number;
  entree: number;
  horizon: number;
  historique: (number | null)[];
  futur: (number | null)[];
  realiteVisible: boolean;
  /** Courbes de prévision à afficher (déjà filtrées par l'utilisateur). */
  previsions: Partial<Record<Courbe, Float32Array>>;
  /** Nombre de pas de prévision affichés (animation F25) : 0 à horizon. */
  pas: number;
  styles: Record<string, Style>;
  titre: string;
  /** Décalage survolé par la souris (-entree+1 .. horizon), ou null. */
  survol: number | null;
}

export interface Repere {
  gauche: number;
  largeur: number;
  /** Décalage horaire (H = 0) le plus proche d'une abscisse CSS. */
  decalage(x: number): number;
}

export function dessinerPrincipal(canvas: HTMLCanvasElement, s: Scene): Repere {
  const ctx = preparer(canvas);
  const L = canvas.clientWidth || 800;
  const Ht = canvas.clientHeight || 460;
  const avecEcarts = s.realiteVisible && Object.keys(s.previsions).length > 0 && s.pas > 0;
  const m = { g: 56, d: 18, h: 52, b: 40 };
  const hautEcarts = avecEcarts ? Math.round((Ht - m.h - m.b) * 0.26) : 0;
  const espace = avecEcarts ? 34 : 0;
  const plot = { x: m.g, y: m.h, l: L - m.g - m.d, h: Ht - m.h - m.b - hautEcarts - espace };
  const n = s.entree - 1 + s.horizon;
  const X = (o: number) => plot.x + ((o + s.entree - 1) / n) * plot.l;

  // --- Domaine vertical : historique, prévisions affichées, réalité si révélée ---
  const vals: number[] = s.historique.filter((v): v is number => v !== null);
  for (const p of Object.values(s.previsions)) vals.push(...Array.from(p!.slice(0, s.pas)));
  if (s.realiteVisible) vals.push(...s.futur.slice(0, s.pas).filter((v): v is number => v !== null));
  let ymin = Math.min(...vals);
  let ymax = Math.max(...vals);
  const marge = Math.max(0.2, (ymax - ymin) * 0.08);
  ymin -= marge;
  ymax += marge;
  const Y = (v: number) => plot.y + plot.h - ((v - ymin) / (ymax - ymin)) * plot.h;

  // --- Titre et unités ---
  ctx.fillStyle = ENCRE;
  ctx.font = `600 14px ${POLICE}`;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillText(s.titre, m.g, 18);
  ctx.fillStyle = GRIS;
  ctx.font = `12px ${POLICE}`;
  ctx.fillText("Hauteur d'eau en mètres au-dessus du zéro hydrographique — heures UTC", m.g, 35);

  // --- Zone de prévision ---
  ctx.fillStyle = ZONE_PREVISION;
  ctx.fillRect(X(0), plot.y, X(s.horizon) - X(0), plot.h);
  ctx.fillStyle = GRIS;
  ctx.font = `600 11px ${POLICE}`;
  ctx.textAlign = "center";
  ctx.fillText(`HISTORIQUE OBSERVÉ — ${s.entree} H`, (X(-s.entree + 1) + X(0)) / 2, plot.y + 14);
  ctx.fillStyle = "#1d4ed8";
  ctx.fillText(`PRÉVISION — ${s.horizon} H`, (X(0) + X(s.horizon)) / 2, plot.y + 14);

  // --- Grilles et axes ---
  ctx.font = `11px ${POLICE}`;
  ctx.lineWidth = 1;
  for (const v of graduations(ymin, ymax, 6)) {
    ctx.strokeStyle = GRILLE;
    ctx.beginPath();
    ctx.moveTo(plot.x, Y(v));
    ctx.lineTo(plot.x + plot.l, Y(v));
    ctx.stroke();
    ctx.fillStyle = GRIS;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`${v.toFixed(1).replace(".", ",")} m`, plot.x - 6, Y(v));
  }
  for (let o = -s.entree + 1; o <= s.horizon; o++) {
    const d = s.chrono.date(s.ancre + o);
    if (d.getUTCHours() % 6 !== 0) continue;
    const x = X(o);
    ctx.strokeStyle = d.getUTCHours() === 0 ? "#cbd5e1" : GRILLE;
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.fillStyle = GRIS;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${pad2(d.getUTCHours())}h`, x, Ht - m.b + 6);
    if (d.getUTCHours() === 0) {
      ctx.fillStyle = ENCRE;
      ctx.fillText(`${pad2(d.getUTCDate())}/${pad2(d.getUTCMonth() + 1)}`, x, Ht - m.b + 20);
    }
  }
  ctx.strokeStyle = "#94a3b8";
  ctx.strokeRect(plot.x, plot.y, plot.l, plot.h);

  // --- Instant de prévision H ---
  const dH = s.chrono.date(s.ancre);
  ctx.save();
  ctx.strokeStyle = ENCRE;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(X(0), plot.y - 6);
  ctx.lineTo(X(0), plot.y + plot.h);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = ENCRE;
  ctx.font = `600 11px ${POLICE}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(`H = ${fmtDate(dH)} ${fmtHeure(dH)} UTC`, X(0), plot.y - 9);

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.l, plot.h);
  ctx.clip();

  // --- Historique observé ---
  tracer(
    ctx,
    s.historique.map((v, k) => [X(k - s.entree + 1), v === null ? null : Y(v)]),
    s.styles.observe,
    2.4,
    false,
  );

  // --- Réalité future : masquée tant qu'elle n'est pas révélée (F22) ---
  if (s.realiteVisible) {
    const pts: [number, number | null][] = [[X(0), s.historique[s.entree - 1] === null ? null : Y(s.historique[s.entree - 1]!)]];
    s.futur.slice(0, s.pas).forEach((v, k) => pts.push([X(k + 1), v === null ? null : Y(v)]));
    tracer(ctx, pts.slice(1), s.styles.realite, 2.6);
    tracer(ctx, pts.slice(0, 2), { ...s.styles.realite, marqueur: "aucun" }, 2.6);
  } else {
    ctx.fillStyle = "#94a3b8";
    ctx.font = `600 13px ${POLICE}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const xc = (X(0) + X(s.horizon)) / 2;
    ctx.fillText("Réalité masquée", xc, plot.y + plot.h - 34);
    ctx.font = `12px ${POLICE}`;
    ctx.fillText("« Révéler la réalité » pour comparer", xc, plot.y + plot.h - 16);
  }

  // --- Prévisions (F21) ---
  for (const [nom, p] of Object.entries(s.previsions)) {
    if (!p) continue;
    const pts: [number, number | null][] = Array.from(p.slice(0, s.pas), (v, k) => [X(k + 1), Y(v)]);
    tracer(ctx, pts, s.styles[nom], 2);
  }
  ctx.restore();

  // --- Panneau des écarts (F16) ---
  if (avecEcarts) {
    const panneau = { x: plot.x, y: Ht - m.b - hautEcarts, l: plot.l, h: hautEcarts };
    const ecarts: number[] = [];
    for (const p of Object.values(s.previsions)) {
      for (let k = 0; k < s.pas; k++) if (s.futur[k] !== null) ecarts.push(100 * (p![k] - s.futur[k]!));
    }
    const amp = Math.max(5, ...ecarts.map(Math.abs)) * 1.15;
    const Ye = (v: number) => panneau.y + panneau.h / 2 - (v / amp) * (panneau.h / 2);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(panneau.x, panneau.y, panneau.l, panneau.h);
    ctx.fillStyle = ZONE_PREVISION;
    ctx.fillRect(X(0), panneau.y, X(s.horizon) - X(0), panneau.h);
    ctx.strokeStyle = "#94a3b8";
    ctx.strokeRect(panneau.x, panneau.y, panneau.l, panneau.h);
    ctx.strokeStyle = ENCRE;
    ctx.beginPath();
    ctx.moveTo(X(0), Ye(0));
    ctx.lineTo(X(s.horizon), Ye(0));
    ctx.stroke();
    ctx.fillStyle = GRIS;
    ctx.font = `11px ${POLICE}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const positifs = graduations(0, amp, 2).filter((v) => v > 0);
    const tick = positifs.length ? positifs[positifs.length - 1] : amp / 2;
    for (const v of [tick, 0, -tick]) ctx.fillText(`${v > 0 ? "+" : ""}${Math.round(v)} cm`, plot.x - 6, Ye(v));
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.font = `600 11px ${POLICE}`;
    ctx.fillText("Écart prévision − réalité (cm)", panneau.x + 4, panneau.y - 4);
    ctx.save();
    ctx.beginPath();
    ctx.rect(panneau.x, panneau.y, panneau.l, panneau.h);
    ctx.clip();
    for (const [nom, p] of Object.entries(s.previsions)) {
      const pts: [number, number | null][] = [];
      for (let k = 0; k < s.pas; k++) pts.push([X(k + 1), s.futur[k] === null ? null : Ye(100 * (p![k] - s.futur[k]!))]);
      tracer(ctx, pts, s.styles[nom], 1.6);
    }
    ctx.restore();
  }

  // --- Survol : réticule et infobulle ---
  if (s.survol !== null) dessinerSurvol(ctx, s, X, plot, L);

  return {
    gauche: plot.x,
    largeur: plot.l,
    decalage: (x: number) =>
      Math.max(-s.entree + 1, Math.min(s.horizon, Math.round(((x - plot.x) / plot.l) * n - s.entree + 1))),
  };
}

function dessinerSurvol(
  ctx: CanvasRenderingContext2D,
  s: Scene,
  X: (o: number) => number,
  plot: { x: number; y: number; l: number; h: number },
  L: number,
): void {
  const o = s.survol!;
  const d = s.chrono.date(s.ancre + o);
  const lignes: [string, string, Style | null][] = [];
  if (o <= 0) {
    const v = s.historique[o + s.entree - 1];
    lignes.push(["Observé", v === null ? "lacune" : fmtM(v), s.styles.observe]);
  } else if (o <= s.pas) {
    if (s.realiteVisible) {
      const v = s.futur[o - 1];
      lignes.push(["Réalité", v === null ? "lacune" : fmtM(v), s.styles.realite]);
    } else lignes.push(["Réalité", "masquée", null]);
    for (const [nom, p] of Object.entries(s.previsions)) lignes.push([s.styles[nom].nom, fmtM(p![o - 1]), s.styles[nom]]);
  } else lignes.push(["Prévision", "pas encore affichée", null]);

  const x = X(o);
  ctx.save();
  ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
  ctx.beginPath();
  ctx.moveTo(x, plot.y);
  ctx.lineTo(x, plot.y + plot.h);
  ctx.stroke();

  ctx.font = `12px ${POLICE}`;
  const titre = `${fmtDate(d)} ${fmtHeure(d)} UTC — ${o === 0 ? "H" : o > 0 ? `H+${o}` : `H${o}`}`;
  const largeur = Math.max(ctx.measureText(titre).width, ...lignes.map((l) => ctx.measureText(`${l[0]} : ${l[1]}`).width + 26)) + 16;
  const hauteur = 24 + lignes.length * 17;
  let bx = x + 12;
  if (bx + largeur > L - 4) bx = x - 12 - largeur;
  const by = plot.y + 24;
  ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.roundRect(bx, by, largeur, hauteur, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = ENCRE;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.font = `600 12px ${POLICE}`;
  ctx.fillText(titre, bx + 8, by + 6);
  ctx.font = `12px ${POLICE}`;
  lignes.forEach(([nom, val, style], k) => {
    const ly = by + 24 + k * 17;
    if (style) {
      ctx.fillStyle = style.couleur;
      ctx.strokeStyle = style.couleur;
      marqueur(ctx, style.marqueur === "aucun" ? "rond" : style.marqueur, bx + 14, ly + 7, 3.5);
    }
    ctx.fillStyle = ENCRE;
    ctx.fillText(`${nom} : ${val}`, bx + 26, ly);
  });
  ctx.restore();
}

// --------------------------------------------------------------------------
// Frise : la série complète et la fenêtre sélectionnée (F17)
// --------------------------------------------------------------------------

const COULEURS_ENSEMBLES = { train: "#dbeafe", val: "#fef3c7", test: "#dcfce7" };

export function dessinerFrise(
  canvas: HTMLCanvasElement,
  chrono: Chronologie,
  ancre: number | null,
  entree: number,
  horizon: number,
  testSeulement: boolean,
): (x: number) => number {
  const ctx = preparer(canvas);
  const L = canvas.clientWidth || 800;
  const H = canvas.clientHeight || 70;
  const { debut, fin } = testSeulement ? chrono.serie.ensembles.test : { debut: 0, fin: chrono.longueur };
  const n = fin - debut;
  const X = (i: number) => ((i - debut) / n) * L;
  const haut = 4;
  const bas = H - 16;
  const v = chrono.serie.valeurs;

  for (const [nom, e] of Object.entries(chrono.serie.ensembles)) {
    const a = Math.max(e.debut, debut);
    const b = Math.min(e.fin, fin);
    if (b <= a) continue;
    ctx.fillStyle = COULEURS_ENSEMBLES[nom as keyof typeof COULEURS_ENSEMBLES];
    ctx.fillRect(X(a), haut, X(b) - X(a), bas - haut);
  }
  // Enveloppe min/max par colonne de pixels ; lacunes en rouge.
  const Y = (h: number) => bas - ((h - 0) / 8.5) * (bas - haut);
  for (let px = 0; px < L; px++) {
    const a = debut + Math.floor((px / L) * n);
    const b = Math.max(a + 1, debut + Math.floor(((px + 1) / L) * n));
    let lo = Infinity;
    let hi = -Infinity;
    let lacune = false;
    for (let i = a; i < b; i++) {
      const h = v[i];
      if (h === null || h === undefined) lacune = true;
      else {
        lo = Math.min(lo, h);
        hi = Math.max(hi, h);
      }
    }
    if (lacune) {
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(px, haut, 1, bas - haut);
    }
    if (hi > -Infinity) {
      ctx.fillStyle = "#1e3a5f";
      ctx.fillRect(px, Y(hi), 1, Math.max(1, Y(lo) - Y(hi)));
    }
  }
  // Étiquettes des années (ou des mois en mode test).
  ctx.fillStyle = GRIS;
  ctx.font = `11px ${POLICE}`;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  let precedent = "";
  for (let i = debut; i < fin; i += 24) {
    const d = chrono.date(i);
    const cle = testSeulement ? `${d.getUTCFullYear()}-${d.getUTCMonth()}` : `${d.getUTCFullYear()}`;
    if (cle !== precedent) {
      precedent = cle;
      const lib = testSeulement
        ? d.toLocaleDateString("fr-FR", { month: "short", timeZone: "UTC" })
        : String(d.getUTCFullYear());
      ctx.fillText(lib, X(i) + 2, bas + 2);
      ctx.fillRect(X(i), bas, 1, 4);
    }
  }
  if (ancre !== null) {
    const a = X(ancre - entree + 1);
    const b = X(ancre + horizon + 1);
    ctx.fillStyle = "rgba(29, 78, 216, 0.25)";
    ctx.fillRect(a, 0, Math.max(3, b - a), bas + 2);
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    ctx.strokeRect(a, 1, Math.max(3, b - a), bas);
  }
  return (x: number) => Math.round(debut + (x / L) * n);
}

// --------------------------------------------------------------------------
// Petits graphiques de courbes (erreur par échéance, apprentissage)
// --------------------------------------------------------------------------

export interface Serie2D {
  style: Style;
  y: number[];
  x?: number[];
  epais?: boolean;
}

export function dessinerCourbes(
  canvas: HTMLCanvasElement,
  series: Serie2D[],
  options: { titre: string; xlabel: string; ylabel: string; logY?: boolean; ymax?: number; xTicks?: number[] },
): void {
  const ctx = preparer(canvas);
  const L = canvas.clientWidth || 500;
  const Ht = canvas.clientHeight || 260;
  const m = { g: 52, d: 12, h: 28, b: 38 };
  const plot = { x: m.g, y: m.h, l: L - m.g - m.d, h: Ht - m.h - m.b };
  const xs = series.flatMap((s) => s.x ?? s.y.map((_, k) => k + 1));
  const ys = series.flatMap((s) => s.y).filter((v) => Number.isFinite(v) && (!options.logY || v > 0));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const tr = (v: number) => (options.logY ? Math.log10(v) : v);
  let ymin = options.logY ? tr(Math.min(...ys)) : 0;
  let ymax = tr(options.ymax ?? Math.max(...ys));
  if (options.logY) {
    ymin = Math.floor(ymin * 2) / 2;
    ymax = Math.ceil(ymax * 2) / 2;
  } else ymax *= 1.08;
  const X = (v: number) => plot.x + ((v - xmin) / Math.max(1e-9, xmax - xmin)) * plot.l;
  const Y = (v: number) => plot.y + plot.h - ((tr(v) - ymin) / (ymax - ymin)) * plot.h;

  ctx.fillStyle = ENCRE;
  ctx.font = `600 13px ${POLICE}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(options.titre, m.g, 16);
  ctx.font = `11px ${POLICE}`;
  const ticksY = options.logY
    ? Array.from({ length: Math.round((ymax - ymin) * 2) + 1 }, (_, k) => 10 ** (ymin + k / 2)).filter(
        (_, k) => k % 2 === 0,
      )
    : graduations(0, ymax, 5);
  for (const v of ticksY) {
    ctx.strokeStyle = GRILLE;
    ctx.beginPath();
    ctx.moveTo(plot.x, Y(v));
    ctx.lineTo(plot.x + plot.l, Y(v));
    ctx.stroke();
    ctx.fillStyle = GRIS;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(options.logY ? v.toExponential(0) : String(Number(v.toFixed(2))).replace(".", ","), plot.x - 5, Y(v));
  }
  for (const v of options.xTicks ?? graduations(xmin, xmax, 6)) {
    ctx.fillStyle = GRIS;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(String(v), X(v), plot.y + plot.h + 5);
  }
  ctx.fillText(options.xlabel, plot.x + plot.l / 2, plot.y + plot.h + 20);
  ctx.save();
  ctx.translate(12, plot.y + plot.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText(options.ylabel, 0, 0);
  ctx.restore();
  ctx.strokeStyle = "#94a3b8";
  ctx.strokeRect(plot.x, plot.y, plot.l, plot.h);

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.l, plot.h);
  ctx.clip();
  for (const s of series) {
    const xv = s.x ?? s.y.map((_, k) => k + 1);
    const pts: [number, number | null][] = s.y.map((v, k) => [X(xv[k]), Number.isFinite(v) ? Y(v) : null]);
    const peu = pts.length <= 30;
    tracer(ctx, pts, s.epais === false ? { ...s.style, tirets: [2, 3] } : s.style, s.epais === false ? 1 : 1.8, peu);
  }
  ctx.restore();
}
