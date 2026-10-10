import "./style.css";

import { CarteExemple } from "./map";
import { loadExemples, loadMetadata, Segmenteur, type Exemple, type Metadata } from "./model";
import {
  composerSuperposition,
  dessinerIncertitude,
  dessinerMasque,
  hexRgb,
  statistiques,
  telecharger,
} from "./render";

type Mode = "comparateur" | "original" | "segmentation" | "superposition" | "incertitude";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const statusEl = $("status");
const analyzeBtn = $<HTMLButtonElement>("analyze");
const progressEl = $("progress");
const progressBar = progressEl.querySelector<HTMLElement>(".progress-bar")!;
const galleryEl = $("gallery");
const dropzone = $("dropzone");
const fileInput = $<HTMLInputElement>("file-input");
const stage = $("stage");
const stageWrap = $("stage-wrap");
const placeholder = $("placeholder");
const handle = $("handle");
const cvImage = $<HTMLCanvasElement>("cv-image");
const cvMask = $<HTMLCanvasElement>("cv-mask");
const cvUncert = $<HTMLCanvasElement>("cv-uncert");
const opacityEl = $<HTMLInputElement>("opacity");
const modeHelp = $("mode-help");
const resultsEl = $("results");
const mapCard = $("map-card");

const AIDE: Record<Mode, string> = {
  comparateur: "Faites glisser le curseur : à gauche l'image originale, à droite la segmentation prédite.",
  original: "Image satellite telle qu'elle a été chargée.",
  segmentation: "Masque prédit seul : vert = terre, bleu = mer.",
  superposition: "Masque prédit par-dessus l'image ; réglez son opacité avec le curseur.",
  incertitude:
    "En jaune : pixels où le modèle hésite (probabilité de terre proche de 50 %). " +
    "Plus le jaune est vif, plus l'hésitation est forte.",
};

// --------------------------------------------------------------------------
// État de l'application
// --------------------------------------------------------------------------

let meta: Metadata | null = null;
let segmenteur: Segmenteur | null = null;
let carte: CarteExemple | null = null;
let image: { nom: string; data: ImageData; exemple?: Exemple } | null = null;
let resultat: { prob: Float32Array } | null = null;
let analyseEnCours = false;
let mode: Mode = "original";

function setStatus(message: string, type: "info" | "ok" | "error" | "busy" = "info"): void {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

function majBouton(): void {
  analyzeBtn.disabled = !segmenteur || !image || analyseEnCours;
}

function fmtPct(v: number): string {
  return `${v.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %`;
}

// --------------------------------------------------------------------------
// Modes d'affichage, opacité, comparateur (F19 à F21, F23)
// --------------------------------------------------------------------------

function setMode(m: Mode): void {
  if (!resultat && m !== "original") m = "original";
  mode = m;
  stage.className = `stage mode-${m}`;
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
    const actif = b.dataset.mode === m;
    b.setAttribute("aria-selected", String(actif));
    b.disabled = !resultat && b.dataset.mode !== "original";
  });
  opacityEl.disabled = !(m === "superposition" || m === "comparateur");
  modeHelp.textContent = resultat ? AIDE[m] : "";
}

function setOpacite(v: number): void {
  stage.style.setProperty("--opacity", String(v));
  carte?.setOpaciteMasque(v);
}

function setPosition(pct: number): void {
  const p = Math.min(100, Math.max(0, pct));
  stage.style.setProperty("--pos", `${p}%`);
  handle.setAttribute("aria-valuenow", String(Math.round(p)));
}

function initComparateur(): void {
  let glisse = false;
  let x = 0;
  let planifie = false;
  const appliquer = (): void => {
    planifie = false;
    const r = stage.getBoundingClientRect();
    setPosition(((x - r.left) / r.width) * 100);
  };
  stage.addEventListener("pointerdown", (e) => {
    if (mode !== "comparateur") return;
    glisse = true;
    stage.setPointerCapture(e.pointerId);
    x = e.clientX;
    appliquer();
  });
  stage.addEventListener("pointermove", (e) => {
    if (!glisse) return;
    x = e.clientX;
    if (!planifie) {
      planifie = true;
      requestAnimationFrame(appliquer);
    }
  });
  const fin = (): void => {
    glisse = false;
  };
  stage.addEventListener("pointerup", fin);
  stage.addEventListener("pointercancel", fin);
  handle.addEventListener("keydown", (e) => {
    const actuel = Number(handle.getAttribute("aria-valuenow"));
    if (e.key === "ArrowLeft") setPosition(actuel - 2);
    else if (e.key === "ArrowRight") setPosition(actuel + 2);
    else return;
    e.preventDefault();
  });
  setPosition(50);
}

// --------------------------------------------------------------------------
// Chargement d'une image (F14, F15, F25)
// --------------------------------------------------------------------------

function reinitialiserResultats(): boolean {
  const avait = resultat !== null;
  resultat = null;
  resultsEl.hidden = true;
  for (const c of [cvMask, cvUncert]) {
    c.width = 1;
    c.height = 1;
  }
  carte?.retirerMasque();
  setMode("original");
  return avait;
}

function validerFichier(f: File): string | null {
  const types = ["image/png", "image/jpeg"];
  const extOk = /\.(png|jpe?g)$/i.test(f.name);
  if (!types.includes(f.type) && !(f.type === "" && extOk)) {
    return `Format non pris en charge (${f.type || f.name}). Formats acceptés : PNG et JPEG.`;
  }
  const maxMo = meta?.limites.taille_fichier_mo ?? 25;
  if (f.size > maxMo * 1e6) {
    return `Fichier trop volumineux (${(f.size / 1e6).toFixed(1)} Mo > ${maxMo} Mo).`;
  }
  return null;
}

async function chargerImage(source: Blob, nom: string, exemple?: Exemple): Promise<void> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(source);
  } catch {
    setStatus(`Impossible de lire « ${nom} » : fichier endommagé ou non reconnu comme image.`, "error");
    return;
  }
  const { width: w, height: h } = bitmap;
  const lim = meta?.limites;
  if (lim && (w > lim.cote_max_px || h > lim.cote_max_px || w * h > lim.pixels_max)) {
    bitmap.close();
    setStatus(
      `Image trop grande (${w} × ${h} px). Limites : ${lim.cote_max_px} px de côté et ` +
        `${(lim.pixels_max / 1e6).toFixed(0)} millions de pixels.`,
      "error",
    );
    return;
  }

  const efface = reinitialiserResultats();
  cvImage.width = w;
  cvImage.height = h;
  const ctx = cvImage.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  image = { nom, data: ctx.getImageData(0, 0, w, h), exemple };

  stage.style.setProperty("--ratio", String(w / h));
  placeholder.hidden = true;
  stage.hidden = false;

  document.querySelectorAll<HTMLButtonElement>(".thumb").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.id === exemple?.id));
  });

  if (exemple && carte) {
    mapCard.hidden = false;
    carte.afficherImage(`/${exemple.fichier}`, exemple.bornes);
    $("map-note").textContent = `${exemple.titre} — ${exemple.origine}.`;
  } else {
    mapCard.hidden = true;
  }

  const prefixe = efface ? "Nouvelle image, résultats précédents effacés. " : "";
  setStatus(
    `${prefixe}Image chargée : ${nom} (${w} × ${h} px).` +
      (segmenteur ? " Cliquez sur « Analyser l'image »." : ""),
    "ok",
  );
  majBouton();
}

async function chargerExemple(ex: Exemple): Promise<void> {
  try {
    const res = await fetch(`/${ex.fichier}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await chargerImage(await res.blob(), ex.titre, ex);
  } catch (err) {
    setStatus(`Erreur de chargement de l'exemple « ${ex.titre} » : ${(err as Error).message}`, "error");
  }
}

async function chargerFichier(f: File | undefined): Promise<void> {
  if (!f) return;
  const erreur = validerFichier(f);
  if (erreur) {
    setStatus(erreur, "error");
    return;
  }
  await chargerImage(f, f.name);
}

function construireGalerie(exemples: Exemple[]): void {
  galleryEl.innerHTML = "";
  for (const ex of exemples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.dataset.id = ex.id;
    b.setAttribute("role", "listitem");
    b.setAttribute("aria-pressed", "false");
    b.title = `${ex.titre} — ${ex.configuration} (${ex.origine})`;
    b.innerHTML = `
      <img src="/${ex.fichier}" alt="" loading="lazy" />
      <span class="thumb-title">${ex.titre}</span>
      <span class="thumb-sub">${ex.configuration}</span>`;
    b.addEventListener("click", () => void chargerExemple(ex));
    galleryEl.appendChild(b);
  }
}

// --------------------------------------------------------------------------
// Analyse (F16, F17, F18, F22, F23)
// --------------------------------------------------------------------------

async function analyser(): Promise<void> {
  if (!segmenteur || !image || !meta) return;
  analyseEnCours = true;
  majBouton();
  const courant = image;
  progressEl.hidden = false;
  progressBar.style.width = "0%";
  setStatus("Analyse en cours…", "busy");

  try {
    const r = await segmenteur.segmenter(courant.data, (fait, total) => {
      progressBar.style.width = `${(100 * fait) / total}%`;
      setStatus(`Analyse en cours… fenêtre ${fait} / ${total}`, "busy");
    });
    if (image !== courant) return; // une autre image a été choisie entre-temps

    const { width: w, height: h } = courant.data;
    const inf = meta.inference;
    dessinerMasque(cvMask, r.prob, w, h, inf.seuil_terre, hexRgb(meta.couleurs.terre), hexRgb(meta.couleurs.mer));
    dessinerIncertitude(cvUncert, r.prob, w, h, inf.incertitude.min, inf.incertitude.max);
    resultat = { prob: r.prob };

    const s = statistiques(r.prob, inf.seuil_terre, inf.incertitude.min, inf.incertitude.max);
    $("stat-terre").textContent = fmtPct(s.terrePct);
    $("stat-mer").textContent = fmtPct(s.merPct);
    $("stat-incertain").textContent = fmtPct(s.incertainPct);
    $("stat-duree").textContent = `${(r.ms / 1000).toFixed(2)} s (${r.fenetres} fenêtres, ${r.backend})`;
    $("stat-taille").textContent = `${w} × ${h} px`;
    $("bar-terre").style.width = `${s.terrePct}%`;
    resultsEl.hidden = false;

    if (courant.exemple && carte) carte.afficherMasque(cvMask.toDataURL("image/png"), Number(opacityEl.value));
    setMode("comparateur");
    setStatus(`Analyse terminée en ${(r.ms / 1000).toFixed(2)} s. Explorez le résultat !`, "ok");
  } catch (err) {
    console.error(err);
    setStatus(`Erreur pendant l'analyse : ${(err as Error).message}. Vous pouvez réessayer.`, "error");
  } finally {
    analyseEnCours = false;
    progressEl.hidden = true;
    majBouton();
  }
}

// --------------------------------------------------------------------------
// Téléchargements (F24)
// --------------------------------------------------------------------------

function nomBase(): string {
  return (image?.nom ?? "image").replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N}_-]+/gu, "_");
}

async function exporter(type: "masque" | "superposition"): Promise<void> {
  if (!resultat) return;
  try {
    const canvas = type === "masque" ? cvMask : composerSuperposition(cvImage, cvMask, Number(opacityEl.value));
    await telecharger(canvas, `${nomBase()}_${type}.png`);
  } catch (err) {
    setStatus((err as Error).message, "error");
  }
}

// --------------------------------------------------------------------------
// Initialisation
// --------------------------------------------------------------------------

function brancherEvenements(): void {
  analyzeBtn.addEventListener("click", () => void analyser());
  fileInput.addEventListener("change", () => {
    void chargerFichier(fileInput.files?.[0]);
    fileInput.value = ""; // permet de re-sélectionner le même fichier
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  for (const cible of [dropzone, stageWrap]) {
    cible.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    });
    cible.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
    cible.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
      void chargerFichier(e.dataTransfer?.files[0]);
    });
  }
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
    b.addEventListener("click", () => setMode(b.dataset.mode as Mode));
  });
  opacityEl.addEventListener("input", () => setOpacite(Number(opacityEl.value)));
  $("dl-mask").addEventListener("click", () => void exporter("masque"));
  $("dl-overlay").addEventListener("click", () => void exporter("superposition"));
  initComparateur();
}

function afficherInfosModele(m: Metadata): void {
  const e = m.entrainement;
  const date = new Date(e.date_creation).toLocaleDateString("fr-FR");
  const test = e.test
    ? ` Sur les zones de test : IoU terre ${e.test.iou_terre.toFixed(3)}, IoU mer ${e.test.iou_mer.toFixed(3)}, ` +
      `exactitude ${fmtPct(100 * e.test.exactitude)}.`
    : "";
  $("model-info").textContent =
    `Modèle : ${m.modele.architecture}, ${(m.modele.parametres / 1e6).toFixed(2)} M paramètres, ` +
    `entraîné le ${date} (dataset ${e.dataset}, époque ${e.epoque}).${test}`;
  $("sources").textContent = `Images : ${m.sources.images}. Vérité d'apprentissage : ${m.sources.masques}.`;
  const l = m.limites;
  $("limits").textContent =
    `Limites : ${l.taille_fichier_mo} Mo, ${l.cote_max_px} px de côté, ${(l.pixels_max / 1e6).toFixed(0)} Mpx.`;
  document.documentElement.style.setProperty("--terre", m.couleurs.terre);
  document.documentElement.style.setProperty("--mer", m.couleurs.mer);
}

async function main(): Promise<void> {
  brancherEvenements();
  setMode("original");
  setOpacite(Number(opacityEl.value));
  stage.hidden = true;

  try {
    const [m, exemples] = await Promise.all([loadMetadata(), loadExemples()]);
    meta = m;
    afficherInfosModele(m);
    construireGalerie(exemples);
  } catch (err) {
    setStatus((err as Error).message, "error");
    return;
  }

  carte = new CarteExemple($("map"));
  new ResizeObserver(() => carte?.rafraichir()).observe(mapCard);

  setStatus("Chargement du modèle…", "busy");
  try {
    const { segmenteur: s, backend } = await Segmenteur.charger(meta);
    segmenteur = s;
    setStatus(
      image
        ? `Modèle prêt (${backend}). Cliquez sur « Analyser l'image ».`
        : `Modèle prêt (${backend}). Choisissez une image.`,
      "ok",
    );
  } catch (err) {
    console.error(err);
    setStatus(`${(err as Error).message}. Les images restent consultables.`, "error");
  }
  majBouton();
}

void main();
