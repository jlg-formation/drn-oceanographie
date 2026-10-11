import "./style.css";

import {
  centrer,
  chargerEvolution,
  chargerMetadata,
  chargerReels,
  MODELES,
  N_PIXELS,
  TAILLE,
  type Metadata,
  type Modele,
} from "./donnees";
import { telechargerBlob, telechargerCsv, telechargerJson, telechargerPng } from "./exports";
import { dessinerCourbes } from "./graphiques";
import {
  chargerModeles,
  encoder,
  generer,
  graineAuHasard,
  interpoler,
  sessionGeneration,
  vecteurLatent,
  type Sessions,
} from "./modeles";
import {
  composerPng,
  definirPalette,
  dessinerCarte,
  dessinerLegende,
  echelleAuto,
  PALETTE_ERREUR,
  pixelSousPointeur,
  type Echelle,
} from "./rendu2d";
import { erreurs, statistiques, type Statistiques } from "./stats";
import { Vue3D } from "./vue3d";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

type Mode = Modele | "comparaison";
type Onglet = "generation" | "latent" | "reconstruction";

/** Relief affiché dans l'espace VAE / GAN, avec tout ce qu'il faut pour le reproduire (F36). */
interface Courant {
  modele: Modele;
  nature: "generation" | "latent" | "interpolation";
  graine: number | null;
  zInitial: Float32Array;
  z: Float32Array;
  relief: Float32Array;
  interpolation?: { graineA: number; graineB: number; t: number };
}

interface Reconstruction {
  indice: number;
  reel: Float32Array; // relief relatif (m)
  rec: Float32Array;
  erreur: Float32Array;
  rmse: number;
  max: number;
  mu: Float32Array;
}

// --------------------------------------------------------------------------
// État
// --------------------------------------------------------------------------

let meta: Metadata;
let sessions: Sessions = {};
let reels: Float32Array[] = [];
let evolution: Float32Array[][] | null = null;
let mode: Mode = "vae";
let onglet: Onglet = "generation";
const courants: Partial<Record<Modele, Courant>> = {};
let reconstruction: Reconstruction | null = null;
let reelChoisi = 0;
let pageDims = 0;
const DIMS_PAR_PAGE = 6;
let animation: number | null = null;
let echelleM = 1500;
let pixelKm = 1.85;

let vue: Vue3D | null = null;
const vuesCmp: Partial<Record<"reel" | Modele, Vue3D>> = {};
const cmp: Partial<Record<"reel" | Modele, { relief: Float32Array; graine?: number; indice?: number }>> = {};

const carte = $<HTMLCanvasElement>("carte");
const statusEl = $("status");

function setStatus(message: string, type: "info" | "ok" | "error" | "busy" = "info"): void {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

const nomModele = (m: string) => meta.styles[m]?.nom ?? m;
const dim = (m: Modele) => (m === "vae" ? meta.modeles.vae?.dim_latente : meta.modeles.gan?.dim_latente) ?? 0;
const fmt = (v: number, d = 0) => v.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
const signe = (v: number, d = 0) => `${v > 0 ? "+" : ""}${fmt(v, d)}`;
const disponible = (m: Modele) => !!sessionGeneration(sessions, m);

// --------------------------------------------------------------------------
// Génération
// --------------------------------------------------------------------------

async function decoder(m: Modele, z: Float32Array[]): Promise<Float32Array[]> {
  const s = sessionGeneration(sessions, m);
  if (!s) throw new Error(`modèle ${nomModele(m)} indisponible`);
  return generer(s, z);
}

async function genererGraine(m: Modele, graine: number): Promise<void> {
  const t0 = performance.now();
  setStatus(`Génération en cours (${nomModele(m)}, graine ${graine})…`, "busy");
  try {
    const z = vecteurLatent(graine, dim(m));
    const [relief] = await decoder(m, [z]);
    courants[m] = { modele: m, nature: "generation", graine, zInitial: z, z: Float32Array.from(z), relief };
    $<HTMLInputElement>("graine").value = String(graine);
    $<HTMLInputElement>("graine-a").value = String(graine);
    $<HTMLInputElement>("graine-b").value = String(graine + 1);
    $<HTMLInputElement>("t").value = "0";
    afficherCourant();
    construireCurseurs();
    setStatus(`Génération terminée : ${nomModele(m)}, graine ${graine} (${fmt(performance.now() - t0, 0)} ms).`, "ok");
  } catch (err) {
    setStatus(`Erreur de génération : ${err instanceof Error ? err.message : err}`, "error");
  }
}

/** Nouvelle inférence à partir du vecteur z courant (curseurs, interpolation). */
let enCours = false;
let enAttente = false;
async function regenererZ(): Promise<void> {
  const c = mode !== "comparaison" ? courants[mode] : undefined;
  if (!c) return;
  if (enCours) {
    enAttente = true;
    return;
  }
  enCours = true;
  try {
    [c.relief] = await decoder(c.modele, [c.z]);
    afficherCourant();
  } catch (err) {
    setStatus(`Erreur de génération : ${err instanceof Error ? err.message : err}`, "error");
  } finally {
    enCours = false;
    if (enAttente) {
      enAttente = false;
      void regenererZ();
    }
  }
}

// --------------------------------------------------------------------------
// Affichage de l'espace VAE / GAN
// --------------------------------------------------------------------------

function echelleCourante(reliefs: Float32Array[]): Echelle {
  return $<HTMLSelectElement>("echelle").value === "auto" ? echelleAuto(reliefs) : { min: -echelleM, max: echelleM };
}

function titreCourant(c: Courant): string {
  const nom = nomModele(c.modele);
  if (c.nature === "interpolation" && c.interpolation) {
    const i = c.interpolation;
    return `${nom} — interpolation graines ${i.graineA} → ${i.graineB}, t = ${fmt(i.t, 2)}`;
  }
  if (c.nature === "latent") return `${nom} — graine ${c.graine}, vecteur modifié`;
  return `${nom} — graine ${c.graine}`;
}

function reliefAffiche(): { relief: Float32Array; titre: string } | null {
  if (mode === "comparaison") return null;
  if (onglet === "reconstruction") {
    if (!reconstruction) return null;
    const choix = $<HTMLSelectElement>("recon-3d").value;
    return {
      relief: choix === "reel" ? reconstruction.reel : reconstruction.rec,
      titre: `VAE — reconstruction du patch réel n° ${reconstruction.indice + 1}`,
    };
  }
  const c = courants[mode];
  return c ? { relief: c.relief, titre: titreCourant(c) } : null;
}

function afficherCourant(): void {
  const r = reliefAffiche();
  const recon = onglet === "reconstruction" && mode === "vae";
  $("vue-simple").hidden = recon;
  $("vue-recon").hidden = !recon;
  $("recon-3d").hidden = !recon;
  for (const id of ["dl-png", "dl-png3d", "dl-csv", "dl-json"]) $<HTMLButtonElement>(id).disabled = !r;
  if (!r) {
    $("titre-carte").textContent = recon ? "Choisissez un relief réel puis reconstruisez-le" : "Carte 2D";
    $("stats").innerHTML = "";
    carte.getContext("2d")!.clearRect(0, 0, carte.width, carte.height);
    return;
  }
  $("titre-carte").textContent = r.titre;
  if (recon && reconstruction) {
    const e = echelleCourante([reconstruction.reel, reconstruction.rec]);
    dessinerCarte($<HTMLCanvasElement>("c-reel"), reconstruction.reel, e);
    dessinerCarte($<HTMLCanvasElement>("c-rec"), reconstruction.rec, e);
    const lim = Math.max(50, Math.ceil(reconstruction.max / 50) * 50);
    dessinerCarte($<HTMLCanvasElement>("c-err"), reconstruction.erreur, { min: -lim, max: lim }, PALETTE_ERREUR);
    dessinerLegende($<HTMLCanvasElement>("legende-recon"), e, "relief (m)");
    dessinerLegende($<HTMLCanvasElement>("legende-err"), { min: -lim, max: lim }, "erreur (m)", PALETTE_ERREUR);
    vue?.afficher(r.relief, e);
  } else {
    const e = echelleCourante([r.relief]);
    dessinerCarte(carte, r.relief, e);
    dessinerLegende($<HTMLCanvasElement>("legende"), e, "relief relatif (m)");
    vue?.afficher(r.relief, e);
  }
  afficherStats();
}

function ligneStats(nom: string, s: Statistiques, extra = ""): string {
  const e = echelleM;
  const cel = (m: number, signeOk = true) =>
    `<td>${signeOk ? signe(m) : fmt(m)} m</td><td class="norm">${signeOk ? signe(m / e, 3) : fmt(m / e, 3)}</td>`;
  return `<tr><th scope="row">${nom}</th>${cel(s.min)}${cel(s.max)}${cel(s.moyenne)}${cel(s.amplitude, false)}${cel(
    s.rugosite,
    false,
  )}<td>${fmt(s.penteMoyDeg, 1)}°</td>${extra}</tr>`;
}

function afficherStats(): void {
  const entete = `<thead><tr><th></th><th>Min (m)</th><th class="norm">norm.</th><th>Max (m)</th><th class="norm">norm.</th>
    <th>Moyenne (m)</th><th class="norm">norm.</th><th>Amplitude (m)</th><th class="norm">norm.</th>
    <th>Rugosité (m)</th><th class="norm">norm.</th><th>Pente moy.</th></tr></thead>`;
  let lignes = "";
  let note = "";
  if (onglet === "reconstruction" && reconstruction) {
    const p = meta.reels.patches[reconstruction.indice];
    lignes += ligneStats("Réel (GEBCO)", statistiques(reconstruction.reel, pixelKm));
    lignes += ligneStats("Reconstruit (VAE)", statistiques(reconstruction.rec, pixelKm));
    note = `Profondeur moyenne réelle du patch : ${fmt(p.profondeur_moy_m)} m (${p.nom_region}). Erreur de
      reconstruction : RMSE ${fmt(reconstruction.rmse)} m, maximum ${fmt(reconstruction.max)} m.`;
  } else {
    const c = mode !== "comparaison" ? courants[mode] : undefined;
    if (!c) return;
    lignes += ligneStats(`${nomModele(c.modele)} (synthétique)`, statistiques(c.relief, pixelKm));
    note = `Relief synthétique : valeurs en mètres relatives à la moyenne du patch (aucune profondeur absolue n'est
      générée). « norm. » = valeur divisée par l'échelle de normalisation (${fmt(echelleM)} m).`;
  }
  $("stats").innerHTML = `<table class="table">${entete}<tbody>${lignes}</tbody></table>
    <p class="note">${note} Rugosité : écart-type du relief moins sa moyenne glissante 3 × 3.</p>`;
}

// --------------------------------------------------------------------------
// Espace latent (F31) et interpolation (F32)
// --------------------------------------------------------------------------

function construireCurseurs(): void {
  if (mode === "comparaison") return;
  const c = courants[mode];
  const d = dim(mode);
  const pages = Math.ceil(d / DIMS_PAR_PAGE);
  pageDims = Math.min(pageDims, pages - 1);
  $("latent-intro").textContent =
    `Le ${nomModele(mode)} utilise ${d} composantes. ${DIMS_PAR_PAGE} sont affichées à la fois ; chaque ` +
    "mouvement relance une génération.";
  $("pages-dims").innerHTML = Array.from({ length: pages }, (_, p) => {
    const a = p * DIMS_PAR_PAGE + 1;
    const b = Math.min(d, (p + 1) * DIMS_PAR_PAGE);
    return `<button type="button" data-page="${p}" aria-pressed="${p === pageDims}">z${a}–z${b}</button>`;
  }).join("");
  const conteneur = $("curseurs");
  conteneur.innerHTML = "";
  if (!c) {
    conteneur.innerHTML = `<p class="note">Générez d'abord un relief.</p>`;
    return;
  }
  for (let k = pageDims * DIMS_PAR_PAGE; k < Math.min(d, (pageDims + 1) * DIMS_PAR_PAGE); k++) {
    const ligne = document.createElement("label");
    ligne.className = "curseur";
    ligne.innerHTML = `<span>z<sub>${k + 1}</sub></span>
      <input type="range" min="-3" max="3" step="0.05" value="${c.z[k].toFixed(2)}" data-k="${k}" />
      <output>${signe(c.z[k], 2)}</output>`;
    conteneur.appendChild(ligne);
  }
}

function modifierComposante(k: number, v: number, sortie: HTMLOutputElement): void {
  if (mode === "comparaison") return;
  const c = courants[mode];
  if (!c) return;
  c.z[k] = v;
  c.nature = "latent";
  sortie.textContent = signe(v, 2);
  void regenererZ();
}

function revenirInitial(): void {
  if (mode === "comparaison") return;
  const c = courants[mode];
  if (!c) return;
  c.z = Float32Array.from(c.zInitial);
  c.nature = "generation";
  delete c.interpolation;
  construireCurseurs();
  void regenererZ();
}

function graineDe(id: string): number {
  return Math.max(0, Math.floor(Number($<HTMLInputElement>(id).value) || 0));
}

function appliquerInterpolation(): void {
  if (mode === "comparaison" || !disponible(mode)) return;
  const d = dim(mode);
  const a = graineDe("graine-a");
  const b = graineDe("graine-b");
  const t = Number($<HTMLInputElement>("t").value);
  $("t-val").textContent = `t = ${fmt(t, 2)}`;
  const za = vecteurLatent(a, d);
  const z = interpoler(za, vecteurLatent(b, d), t);
  const c = courants[mode];
  courants[mode] = {
    modele: mode,
    nature: "interpolation",
    graine: c?.graine ?? a,
    zInitial: c?.zInitial ?? za,
    z,
    relief: c?.relief ?? new Float32Array(N_PIXELS),
    interpolation: { graineA: a, graineB: b, t },
  };
  void regenererZ();
}

async function dessinerBande(): Promise<void> {
  if (mode === "comparaison" || !disponible(mode)) return;
  const d = dim(mode);
  const za = vecteurLatent(graineDe("graine-a"), d);
  const zb = vecteurLatent(graineDe("graine-b"), d);
  const etapes = [0, 0.25, 0.5, 0.75, 1];
  const reliefs = await decoder(mode, etapes.map((t) => interpoler(za, zb, t)));
  const bande = $("bande");
  bande.innerHTML = "";
  reliefs.forEach((r, i) => {
    const fig = document.createElement("button");
    fig.type = "button";
    fig.className = "vignette";
    fig.title = `t = ${etapes[i]}`;
    const cv = document.createElement("canvas");
    cv.width = TAILLE;
    cv.height = TAILLE;
    dessinerCarte(cv, r, { min: -echelleM, max: echelleM });
    fig.append(cv, Object.assign(document.createElement("span"), { textContent: `t=${etapes[i]}` }));
    fig.addEventListener("click", () => {
      $<HTMLInputElement>("t").value = String(etapes[i]);
      appliquerInterpolation();
    });
    bande.appendChild(fig);
  });
}

function lancerAnimation(): void {
  arreterAnimation();
  let t = Number($<HTMLInputElement>("t").value);
  if (t >= 1) t = 0;
  let sens = 1;
  let dernier = performance.now();
  const pas = (maintenant: number) => {
    t += (sens * (maintenant - dernier)) / 2500; // A -> B en 2,5 s, puis retour
    dernier = maintenant;
    if (t >= 1 || t <= 0) {
      t = Math.min(1, Math.max(0, t));
      sens = -sens;
    }
    $<HTMLInputElement>("t").value = t.toFixed(3);
    appliquerInterpolation();
    animation = requestAnimationFrame(pas);
  };
  animation = requestAnimationFrame(pas);
  $<HTMLButtonElement>("interp-lecture").disabled = true;
  $<HTMLButtonElement>("interp-pause").disabled = false;
}

function arreterAnimation(): void {
  if (animation !== null) cancelAnimationFrame(animation);
  animation = null;
  $<HTMLButtonElement>("interp-lecture").disabled = false;
  $<HTMLButtonElement>("interp-pause").disabled = true;
}

// --------------------------------------------------------------------------
// Galeries (F37) et reconstruction (F33)
// --------------------------------------------------------------------------

function vignette(r: Float32Array, legende: string, e: Echelle, surClic: () => void): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "vignette";
  b.setAttribute("role", "listitem");
  const cv = document.createElement("canvas");
  cv.width = TAILLE;
  cv.height = TAILLE;
  dessinerCarte(cv, r, e);
  b.append(cv, Object.assign(document.createElement("span"), { textContent: legende }));
  b.addEventListener("click", surClic);
  return b;
}

async function genererGalerie(): Promise<void> {
  if (mode === "comparaison") return;
  const m = mode;
  const n = Number($<HTMLSelectElement>("galerie-n").value);
  const g0 = graineDe("graine");
  setStatus(`Génération d'une galerie de ${n} reliefs (${nomModele(m)})…`, "busy");
  const t0 = performance.now();
  try {
    const graines = Array.from({ length: n }, (_, i) => g0 + i);
    const reliefs = await decoder(m, graines.map((g) => vecteurLatent(g, dim(m))));
    const galerie = $("galerie");
    galerie.innerHTML = "";
    reliefs.forEach((r, i) =>
      galerie.appendChild(
        vignette(r, `graine ${graines[i]}`, { min: -echelleM, max: echelleM }, () => void genererGraine(m, graines[i])),
      ),
    );
    setStatus(`Galerie de ${n} reliefs générée (${fmt(performance.now() - t0)} ms). Cliquez une vignette.`, "ok");
  } catch (err) {
    setStatus(`Erreur de génération : ${err instanceof Error ? err.message : err}`, "error");
  }
}

function construireGalerieReels(): void {
  const filtre = $<HTMLSelectElement>("filtre-region").value;
  const g = $("galerie-reels");
  g.innerHTML = "";
  meta.reels.patches.forEach((p, i) => {
    if (filtre && p.region !== filtre) return;
    const b = vignette(centrer(reels[i]), p.region, { min: -echelleM, max: echelleM }, () => {
      reelChoisi = i;
      g.querySelectorAll(".vignette").forEach((v) => v.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      $<HTMLButtonElement>("reconstruire").disabled = !sessions.vae_encodeur || !sessions.vae_decodeur;
      $("recon-resultat").innerHTML = `<strong>${p.nom_region}</strong><br />${fmt(p.nord, 2)}° N → ${fmt(
        p.sud,
        2,
      )}° N, ${fmt(p.ouest, 2)}° → ${fmt(p.est, 2)}° E<br />profondeur moyenne ${fmt(p.profondeur_moy_m)} m,
        écart-type du relief ${fmt(p.ecart_type_m)} m`;
    });
    b.setAttribute("aria-pressed", String(i === reelChoisi));
    g.appendChild(b);
  });
}

async function reconstruire(): Promise<void> {
  if (!sessions.vae_encodeur || !sessions.vae_decodeur) return;
  setStatus("Reconstruction en cours (encodeur puis décodeur du VAE)…", "busy");
  const t0 = performance.now();
  try {
    const { mu } = await encoder(sessions.vae_encodeur, reels[reelChoisi]);
    const [rec] = await generer(sessions.vae_decodeur, [mu]);
    const reel = centrer(reels[reelChoisi]);
    const e = erreurs(rec, reel);
    reconstruction = { indice: reelChoisi, reel, rec, erreur: e.carte, rmse: e.rmse, max: e.max, mu };
    afficherCourant();
    const p = meta.reels.patches[reelChoisi];
    $("recon-resultat").innerHTML = `<strong>${p.nom_region}</strong> — RMSE <strong>${fmt(e.rmse)} m</strong>,
      erreur maximale ${fmt(e.max)} m (écart-type du relief : ${fmt(p.ecart_type_m)} m).<br />
      Le relief est compressé en ${mu.length} nombres (μ), puis décodé.`;
    setStatus(`Reconstruction terminée (${fmt(performance.now() - t0)} ms).`, "ok");
  } catch (err) {
    setStatus(`Erreur de reconstruction : ${err instanceof Error ? err.message : err}`, "error");
  }
}

// --------------------------------------------------------------------------
// Exports (F36)
// --------------------------------------------------------------------------

function nomFichier(): string {
  if (onglet === "reconstruction" && reconstruction) return `reconstruction_vae_patch${reconstruction.indice + 1}`;
  const c = mode !== "comparaison" ? courants[mode] : undefined;
  return c ? `relief_${c.modele}_graine${c.graine ?? "x"}${c.nature !== "generation" ? `_${c.nature}` : ""}` : "relief";
}

function parametres(): Record<string, unknown> {
  const commun = {
    application: meta.application,
    avertissement: meta.avertissement,
    date: new Date().toISOString(),
    dataset: meta.dataset.identifiant,
    taille: [TAILLE, TAILLE],
    pixel_km: pixelKm,
    convention: "ligne 0 = nord ; valeurs en mètres, relatives à la moyenne du patch",
    normalisation: meta.normalisation,
    affichage: {
      echelle: $<HTMLSelectElement>("echelle").value,
      exageration_verticale: Number($<HTMLInputElement>("exag").value),
    },
  };
  if (onglet === "reconstruction" && reconstruction) {
    const p = meta.reels.patches[reconstruction.indice];
    return {
      ...commun,
      nature: "reconstruction VAE d'un patch réel GEBCO (jeu de test)",
      modele: { nom: "VAE", fichiers: meta.modeles.vae?.fichiers, epoque: meta.modeles.vae?.epoque },
      patch_reel: p,
      mu: Array.from(reconstruction.mu),
      erreur: { rmse_m: reconstruction.rmse, max_m: reconstruction.max },
      statistiques: { reel: statistiques(reconstruction.reel, pixelKm), reconstruit: statistiques(reconstruction.rec, pixelKm) },
    };
  }
  const c = courants[mode as Modele]!;
  const info = c.modele === "vae" ? meta.modeles.vae : meta.modeles.gan;
  return {
    ...commun,
    nature: "relief SYNTHÉTIQUE",
    modele: { nom: nomModele(c.modele), fichiers: info?.fichiers, epoque: info?.epoque, dim_latente: dim(c.modele) },
    generation: {
      type: c.nature,
      graine: c.graine,
      interpolation: c.interpolation ?? null,
      methode_graine: "z ~ N(0, I), Box-Muller sur mulberry32 (identique à terrain.vecteur_latent)",
    },
    z: Array.from(c.z, (v) => +v.toFixed(6)),
    z_initial: Array.from(c.zInitial, (v) => +v.toFixed(6)),
    statistiques: statistiques(c.relief, pixelKm),
  };
}

function exporterCsv(): void {
  const r = reliefAffiche();
  if (!r) return;
  const colonnes = ["ligne", ...Array.from({ length: TAILLE }, (_, x) => `c${x}`)];
  const lignes = Array.from({ length: TAILLE }, (_, y) => [y, ...Array.from(r.relief.subarray(y * TAILLE, (y + 1) * TAILLE))]);
  telechargerCsv(colonnes, lignes as (number | string)[][], `${nomFichier()}.csv`);
}

async function exporterPng(): Promise<void> {
  const r = reliefAffiche();
  if (!r) return;
  const e = echelleCourante([r.relief]);
  await telechargerPng(composerPng(r.relief, e, r.titre, "Relief synthétique (Ocean Terrain Lab) — ne pas utiliser pour la navigation"), `${nomFichier()}.png`);
}

async function exporterPng3d(): Promise<void> {
  const blob = await vue?.capture();
  if (blob) telechargerBlob(blob, `${nomFichier()}_3d.png`);
}

// --------------------------------------------------------------------------
// Comparaison (F34, F20, F21) et analyse (F19, F22, F23, F15)
// --------------------------------------------------------------------------

function panneau(source: "reel" | Modele): HTMLElement {
  return document.querySelector(`.panneau[data-source="${source}"]`) as HTMLElement;
}

function echelleComparaison(): Echelle {
  const reliefs = Object.values(cmp).map((c) => c!.relief);
  return $<HTMLSelectElement>("cmp-echelle").value === "auto" ? echelleAuto(reliefs) : { min: -echelleM, max: echelleM };
}

function afficherComparaison(): void {
  const e = echelleComparaison();
  for (const source of ["reel", ...MODELES] as ("reel" | Modele)[]) {
    const c = cmp[source];
    const p = panneau(source);
    if (!c) continue;
    dessinerCarte(p.querySelector<HTMLCanvasElement>(".cmp-carte")!, c.relief, e);
    if (!vuesCmp[source]) vuesCmp[source] = new Vue3D(p.querySelector<HTMLElement>(".vue3d")!, pixelKm);
    vuesCmp[source]!.definirExageration(Number($<HTMLInputElement>("cmp-exag").value));
    vuesCmp[source]!.afficher(c.relief, e);
    const s = statistiques(c.relief, pixelKm);
    p.querySelector(".cmp-stats")!.innerHTML = `amplitude <strong>${fmt(s.amplitude)} m</strong> · écart-type
      ${fmt(s.ecartType)} m · rugosité ${fmt(s.rugosite)} m · pente moy. ${fmt(s.penteMoyDeg, 1)}°`;
  }
}

async function cmpGenerer(m: Modele, graine?: number): Promise<void> {
  if (!disponible(m)) return;
  const input = panneau(m).querySelector<HTMLInputElement>(".cmp-graine")!;
  const g = graine ?? Math.max(0, Math.floor(Number(input.value) || 0));
  input.value = String(g);
  const [relief] = await decoder(m, [vecteurLatent(g, dim(m))]);
  cmp[m] = { relief, graine: g };
}

function cmpReel(indice: number): void {
  reelChoisi = indice;
  cmp.reel = { relief: centrer(reels[indice]), indice };
  const p = meta.reels.patches[indice];
  $("cmp-reel-info").textContent = `${p.nom_region} — profondeur moyenne ${fmt(p.profondeur_moy_m)} m`;
}

async function toutRegenerer(nouvelles: boolean): Promise<void> {
  setStatus("Génération des reliefs de comparaison…", "busy");
  try {
    if (nouvelles) cmpReel(Math.floor(Math.random() * reels.length));
    else if (!cmp.reel) cmpReel(0);
    for (const m of MODELES) await cmpGenerer(m, nouvelles ? graineAuHasard() : undefined);
    afficherComparaison();
    setStatus("Comparaison prête : trois reliefs indépendants, même échelle.", "ok");
  } catch (err) {
    setStatus(`Erreur : ${err instanceof Error ? err.message : err}`, "error");
  }
}

/** Distance RMS moyenne entre paires de reliefs (m), comme ``terrain.diversite`` en Python. */
function diversite(lot: Float32Array[]): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < lot.length; i++) {
    for (let j = i + 1; j < lot.length; j++) {
      let d = 0;
      for (let k = 0; k < N_PIXELS; k++) d += (lot[i][k] - lot[j][k]) ** 2;
      s += Math.sqrt(d / N_PIXELS);
      n++;
    }
  }
  return n ? s / n : 0;
}

async function genererDiversite(): Promise<void> {
  const g0 = graineDe("div-graine");
  const e = { min: -echelleM, max: echelleM };
  const conteneur = $("diversite");
  conteneur.innerHTML = "";
  const indices = Array.from({ length: 12 }, (_, i) => (g0 * 7 + i * 5) % reels.length);
  const lots: [string, string, Float32Array[], string[]][] = [
    ["reel", "Réels (GEBCO, test)", indices.map((i) => centrer(reels[i])), indices.map((i) => meta.reels.patches[i].region)],
  ];
  for (const m of MODELES) {
    if (!disponible(m)) continue;
    const graines = Array.from({ length: 12 }, (_, i) => g0 + i);
    lots.push([m, nomModele(m), await decoder(m, graines.map((g) => vecteurLatent(g, dim(m)))), graines.map((g) => `graine ${g}`)]);
  }
  for (const [source, nom, reliefs, legendes] of lots) {
    const ligne = document.createElement("div");
    ligne.className = `rangee ${source}`;
    const d = diversite(reliefs);
    const evalDiv = meta.evaluation?.diversite.par_source[source];
    ligne.innerHTML = `<div class="rangee-titre"><i class="pastille ${source}"></i><strong>${nom}</strong>
      <span>distance moyenne entre ces 12 cartes : ${fmt(d)} m</span>
      ${evalDiv ? `<span class="muted">sur tout le test : diversité ${fmt(evalDiv.diversite_relative, 2)}, quasi-doublons ${fmt(100 * evalDiv.part_quasi_doublons, 1)} %</span>` : ""}</div>`;
    const v = document.createElement("div");
    v.className = "vignettes";
    reliefs.forEach((r, i) => v.appendChild(vignette(r, legendes[i], e, () => {})));
    ligne.appendChild(v);
    conteneur.appendChild(ligne);
  }
}

function tableStatistiques(): void {
  const ev = meta.evaluation;
  if (!ev) {
    $("cmp-statistiques").innerHTML = `<p class="note">Évaluation absente : lancez <code>mise run tp6_06_evaluate</code> puis l'export.</p>`;
    return;
  }
  const noms: Record<string, string> = {
    amplitude_m: "Amplitude (m)",
    ecart_type_m: "Écart-type du relief (m)",
    variabilite_locale_m: "Variabilité locale 3×3 (m)",
    pente_moy_deg: "Pente moyenne (°)",
    rugosite_m: "Rugosité (m)",
  };
  const sources = ["reel", ...MODELES].filter((s) => ev.statistiques[s]);
  const tete = `<tr><th>Médiane par patch</th>${sources.map((s) => `<th><i class="pastille ${s}"></i>${nomModele(s)}</th>`).join("")}</tr>`;
  const lignes = Object.entries(noms).map(
    ([cle, nom]) =>
      `<tr><th scope="row">${nom}</th>${sources.map((s) => `<td>${fmt(ev.statistiques[s].medianes[cle], cle === "pente_moy_deg" ? 2 : 0)}</td>`).join("")}</tr>`,
  );
  lignes.push(
    `<tr><th scope="row">Pente des pixels, 90e centile (°)</th>${sources.map((s) => `<td>${fmt(ev.statistiques[s].pente_pixels_p90_deg, 1)}</td>`).join("")}</tr>`,
    `<tr><th scope="row">Pixels saturés (±1 500 m)</th>${sources.map((s) => `<td>${fmt(100 * ev.statistiques[s].part_valeurs_saturees, 1)} %</td>`).join("")}</tr>`,
    `<tr class="fort"><th scope="row">Écart aux réels (W1 relatif moyen)</th>${sources.map((s) => `<td>${s === "reel" ? "—" : fmt(ev.statistiques[s].w1_relatif_moyen ?? NaN, 2)}</td>`).join("")}</tr>`,
    `<tr><th scope="row">Diversité (1 = réels)</th>${sources.map((s) => `<td>${fmt(ev.diversite.par_source[s].diversite_relative, 2)}</td>`).join("")}</tr>`,
    `<tr><th scope="row">Quasi-doublons</th>${sources.map((s) => `<td>${fmt(100 * ev.diversite.par_source[s].part_quasi_doublons, 1)} %</td>`).join("")}</tr>`,
    `<tr><th scope="row">Distance médiane au plus proche patch d'entraînement (m)</th>${sources.map((s) => `<td>${fmt(ev.diversite.par_source[s].distance_ppv_train_mediane_m)}</td>`).join("")}</tr>`,
  );
  $("cmp-statistiques").innerHTML = `<table class="table"><thead>${tete}</thead><tbody>${lignes.join("")}</tbody></table>`;
}

function tableSynthese(): void {
  const md = meta.evaluation?.synthese_markdown;
  if (!md) return;
  const lignes = md.split("\n").filter((l) => l.startsWith("|") && !/^\|-/.test(l));
  const cellules = (l: string) => l.split("|").slice(1, -1).map((c) => c.trim().replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"));
  const [tete, ...corps] = lignes;
  $("synthese").innerHTML = `<table class="table synthese"><thead><tr>${cellules(tete).map((c) => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${corps.map((l) => `<tr>${cellules(l).map((c, i) => (i ? `<td>${c}</td>` : `<th scope="row">${c}</th>`)).join("")}</tr>`).join("")}</tbody></table>`;
}

function courbes(): void {
  const vae = meta.modeles.vae;
  const gan = meta.modeles.gan;
  const cv = meta.styles.vae.couleur;
  const cg = meta.styles.gan.couleur;
  if (vae) {
    const e = vae.historique.val_perte.map((_, i) => i + 1);
    dessinerCourbes($<HTMLCanvasElement>("ch-vae"), [
      { nom: "perte totale (train)", couleur: cv, x: e, y: vae.historique.train_perte },
      { nom: "perte totale (val)", couleur: cv, x: e, y: vae.historique.val_perte, tirets: [6, 3] },
      { nom: "reconstruction (val)", couleur: "#009E73", x: e, y: vae.historique.val_reconstruction },
      { nom: "KL (val)", couleur: "#CC79A7", x: e, y: vae.historique.val_kl },
    ], { titre: "VAE : reconstruction + β·KL (par carte)", xLabel: "époque", yLabel: "perte", yMin: 0,
      repereX: { valeur: vae.epoque, texte: `retenu (ép. ${vae.epoque})` } });
    dessinerCourbes($<HTMLCanvasElement>("ch-vae-rmse"), [
      { nom: "RMSE de reconstruction (val)", couleur: cv, x: e, y: vae.historique.val_rmse_m },
    ], { titre: "VAE : erreur de reconstruction en mètres", xLabel: "époque", yLabel: "RMSE (m)",
      repereX: { valeur: vae.epoque, texte: "retenu" } });
  }
  if (gan) {
    const e = gan.historique.perte_d.map((_, i) => i + 1);
    dessinerCourbes($<HTMLCanvasElement>("ch-gan"), [
      { nom: "perte D (discriminateur)", couleur: "#0072B2", x: e, y: gan.historique.perte_d },
      { nom: "perte G (générateur)", couleur: cg, x: e, y: gan.historique.perte_g },
      { nom: "D(G(z)) : proba « réel »", couleur: "#94a3b8", x: e, y: gan.historique.d_faux, tirets: [4, 3] },
    ], { titre: "GAN : deux pertes en équilibre (aucune ne mesure la qualité)", xLabel: "époque", yLabel: "perte / probabilité", yMin: 0,
      repereX: { valeur: gan.epoque, texte: `retenu (ép. ${gan.epoque})` } });
    dessinerCourbes($<HTMLCanvasElement>("ch-gan-suivi"), [
      { nom: "écart statistique à la validation", couleur: "#7C3AED", x: gan.historique.suivi_epoques, y: gan.historique.ecart_stat },
      { nom: "diversité relative (1 = réels)", couleur: "#009E73", x: gan.historique.suivi_epoques, y: gan.historique.diversite_relative, tirets: [6, 3] },
    ], { titre: "GAN : qualité statistique et diversité", xLabel: "époque", yLabel: "indicateur", yMin: 0,
      repereY: 1, repereX: { valeur: gan.epoque, texte: "retenu" } });
  }
}

function afficherEvolution(): void {
  const ev = meta.modeles.gan?.evolution;
  if (!ev || !evolution) {
    $("evolution").innerHTML = `<p class="note">Instantanés du GAN indisponibles.</p>`;
    return;
  }
  const k = Number($<HTMLInputElement>("evo").value);
  $("evo-val").textContent = `${ev.epoques[k]} / ${meta.modeles.gan!.epoques}`;
  const conteneur = $("evolution");
  conteneur.innerHTML = "";
  evolution[k].forEach((r, i) =>
    conteneur.appendChild(vignette(r, `z${i}`, { min: -echelleM, max: echelleM }, () => {})),
  );
}

// --------------------------------------------------------------------------
// Navigation : modèle (F27) et espaces (§9.2)
// --------------------------------------------------------------------------

function majNavigation(): void {
  document.querySelectorAll<HTMLButtonElement>("#choix-modele button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.mode === mode));
    if (b.dataset.mode !== "comparaison") b.disabled = !disponible(b.dataset.mode as Modele);
  });
  document.querySelectorAll<HTMLButtonElement>("#onglets button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.onglet === onglet));
    if (b.dataset.onglet === "reconstruction") {
      b.disabled = mode !== "vae";
      b.title = mode === "gan" ? "Un GAN n'a pas d'encodeur : il ne peut pas reconstruire un relief donné." : "";
    }
  });
  document.querySelectorAll<HTMLElement>("[data-panneau]").forEach((p) => (p.hidden = p.dataset.panneau !== onglet));
  history.replaceState(null, "", mode === "comparaison" ? "#comparaison" : `#${mode}/${onglet}`);
  $("espace-modele").hidden = mode === "comparaison";
  $("espace-comparaison").hidden = mode !== "comparaison";
  if (mode !== "comparaison") {
    const chip = $("modele-actif");
    chip.textContent = nomModele(mode);
    chip.className = `chip ${mode}`;
    document.body.dataset.modele = mode;
  } else {
    document.body.dataset.modele = "comparaison";
  }
}

async function changerMode(m: Mode): Promise<void> {
  arreterAnimation();
  mode = m;
  if (m === "gan" && onglet === "reconstruction") onglet = "generation";
  majNavigation();
  if (m === "comparaison") {
    if (!cmp.vae && !cmp.gan) {
      await toutRegenerer(false);
      await genererDiversite();
    } else afficherComparaison();
    courbes();
    return;
  }
  if (!courants[m]) await genererGraine(m, 0);
  else {
    $<HTMLInputElement>("graine").value = String(courants[m]!.graine ?? 0);
    afficherCourant();
    construireCurseurs();
  }
  if (onglet === "latent") void dessinerBande();
}

function changerOnglet(o: Onglet): void {
  arreterAnimation();
  onglet = o;
  majNavigation();
  if (o === "latent") {
    construireCurseurs();
    void dessinerBande();
  }
  afficherCourant();
  // Reconstruction : on part directement du patch réel sélectionné.
  if (o === "reconstruction" && !reconstruction) void reconstruire();
}

// --------------------------------------------------------------------------
// Événements
// --------------------------------------------------------------------------

function brancher(): void {
  document.querySelectorAll<HTMLButtonElement>("#choix-modele button").forEach((b) =>
    b.addEventListener("click", () => void changerMode(b.dataset.mode as Mode)),
  );
  document.querySelectorAll<HTMLButtonElement>("#onglets button").forEach((b) =>
    b.addEventListener("click", () => changerOnglet(b.dataset.onglet as Onglet)),
  );
  $("generer").addEventListener("click", () => mode !== "comparaison" && void genererGraine(mode, graineAuHasard()));
  $("regenerer").addEventListener("click", () => mode !== "comparaison" && void genererGraine(mode, graineDe("graine")));
  $("graine").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && mode !== "comparaison") void genererGraine(mode, graineDe("graine"));
  });
  $("galerie-btn").addEventListener("click", () => void genererGalerie());
  $("echelle").addEventListener("change", afficherCourant);

  $("pages-dims").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-page]");
    if (!b) return;
    pageDims = Number(b.dataset.page);
    construireCurseurs();
  });
  $("curseurs").addEventListener("input", (e) => {
    const input = e.target as HTMLInputElement;
    if (input.dataset.k === undefined) return;
    modifierComposante(Number(input.dataset.k), Number(input.value), input.nextElementSibling as HTMLOutputElement);
  });
  $("revenir").addEventListener("click", revenirInitial);
  $("t").addEventListener("input", appliquerInterpolation);
  for (const id of ["graine-a", "graine-b"]) {
    $(id).addEventListener("change", () => {
      void dessinerBande();
      appliquerInterpolation();
    });
  }
  $("interp-lecture").addEventListener("click", lancerAnimation);
  $("interp-pause").addEventListener("click", arreterAnimation);

  $("filtre-region").addEventListener("change", construireGalerieReels);
  $("reconstruire").addEventListener("click", () => void reconstruire());
  $("recon-3d").addEventListener("change", afficherCourant);

  $("exag").addEventListener("input", () => {
    const v = Number($<HTMLInputElement>("exag").value);
    $("exag-val").textContent = `×${v}`;
    vue?.definirExageration(v);
  });
  $("cam-reset").addEventListener("click", () => vue?.reinitialiserCamera());
  $("cam-dessus").addEventListener("click", () => vue?.vueDeDessus());
  $("cam-rasante").addEventListener("click", () => vue?.vueRasante());

  carte.addEventListener("mousemove", (ev) => {
    const r = reliefAffiche();
    const p = pixelSousPointeur(carte, ev);
    if (!r || !p) return;
    const v = r.relief[p[0] * TAILLE + p[1]];
    $("survol").textContent = `Ligne ${p[0]}, colonne ${p[1]} (≈ ${fmt(p[1] * pixelKm, 0)} km E, ${fmt(
      p[0] * pixelKm,
      0,
    )} km S du coin nord-ouest) : ${signe(v)} m (normalisé ${signe(v / echelleM, 3)})`;
  });

  $("dl-png").addEventListener("click", () => void exporterPng());
  $("dl-png3d").addEventListener("click", () => void exporterPng3d());
  $("dl-csv").addEventListener("click", exporterCsv);
  $("dl-json").addEventListener("click", () => telechargerJson(parametres(), `${nomFichier()}.json`));

  // Comparaison
  $("cmp-reel").addEventListener("click", () => {
    cmpReel((reelChoisi + 1 + Math.floor(Math.random() * (reels.length - 1))) % reels.length);
    afficherComparaison();
  });
  for (const m of MODELES) {
    const p = panneau(m);
    p.querySelector(".cmp-generer")!.addEventListener("click", async () => {
      const input = p.querySelector<HTMLInputElement>(".cmp-graine")!;
      // Bouton « Générer » : graine saisie si elle a changé, sinon nouvelle graine.
      const g = Number(input.value) === cmp[m]?.graine ? graineAuHasard() : Number(input.value);
      await cmpGenerer(m, g);
      afficherComparaison();
    });
  }
  $("cmp-tout").addEventListener("click", () => void toutRegenerer(true));
  $("cmp-echelle").addEventListener("change", afficherComparaison);
  $("cmp-exag").addEventListener("input", () => {
    const v = Number($<HTMLInputElement>("cmp-exag").value);
    $("cmp-exag-val").textContent = `×${v}`;
    for (const v3 of Object.values(vuesCmp)) v3!.definirExageration(v);
  });
  $("div-btn").addEventListener("click", () => void genererDiversite());
  $("evo").addEventListener("input", afficherEvolution);
  window.addEventListener("resize", () => mode === "comparaison" && courbes());
}

// --------------------------------------------------------------------------
// Démarrage
// --------------------------------------------------------------------------

async function demarrer(): Promise<void> {
  setStatus("Chargement des métadonnées et des modèles…", "busy");
  try {
    meta = await chargerMetadata();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err), "error");
    return;
  }
  echelleM = meta.normalisation.echelle_m;
  pixelKm = meta.pixel_km;
  definirPalette(meta.palette);
  document.querySelector<HTMLOptionElement>('#echelle option[value="commune"]')!.textContent =
    `commune ±${fmt(echelleM)} m`;

  const [chargement, r, ev] = await Promise.all([
    chargerModeles(meta),
    chargerReels(meta),
    chargerEvolution(meta).catch(() => null),
  ]);
  sessions = chargement.sessions;
  reels = r;
  evolution = ev;
  vue = new Vue3D($("vue3d"), pixelKm);

  const regions = [...new Set(meta.reels.patches.map((p) => p.region))];
  $("filtre-region").innerHTML = `<option value="">toutes (${meta.reels.patches.length})</option>` +
    regions.map((id) => `<option value="${id}">${meta.regions[id]?.nom ?? id}</option>`).join("");
  construireGalerieReels();
  $<HTMLButtonElement>("reconstruire").disabled = !sessions.vae_encodeur || !sessions.vae_decodeur;
  if (evolution && meta.modeles.gan?.evolution) {
    const evo = $<HTMLInputElement>("evo");
    evo.max = String(evolution.length - 1);
    evo.value = String(evolution.length - 1);
  }
  afficherEvolution();
  tableStatistiques();
  tableSynthese();
  infosModeles(chargement.ecartControle);
  brancher();

  const ok = MODELES.filter(disponible);
  for (const id of ["generer", "regenerer", "galerie-btn"]) $<HTMLButtonElement>(id).disabled = !ok.length;
  if (!ok.length) {
    setStatus(`Aucun modèle n'a pu être chargé : ${chargement.erreurs.join(" ; ")}`, "error");
    majNavigation();
    return;
  }
  // Ancre d'URL (#gan, #vae/latent, #comparaison…) : utile pour préparer une démonstration.
  const [ancreMode, ancreOnglet] = location.hash.slice(1).split("/");
  if (["generation", "latent", "reconstruction"].includes(ancreOnglet)) onglet = ancreOnglet as Onglet;
  await changerMode(ancreMode === "comparaison" || ok.includes(ancreMode as Modele) ? (ancreMode as Mode) : ok[0]);
  if (onglet !== "generation" && mode !== "comparaison") changerOnglet(onglet);
  const controle = chargement.ecartControle;
  setStatus(
    `Modèles chargés : ${ok.map(nomModele).join(" et ")}` +
      (controle !== null ? ` (contrôle navigateur ↔ PyTorch : écart ${fmt(controle, 3)} m)` : "") +
      (chargement.erreurs.length ? ` — ${chargement.erreurs.join(" ; ")}` : "") + ".",
    chargement.erreurs.length ? "error" : "ok",
  );
}

function duree(s: number | null): string {
  return s === null ? "entraînement inachevé" : `${fmt(s / 60, 1)} min d'entraînement`;
}

function infosModeles(ecart: number | null): void {
  const v = meta.modeles.vae;
  const g = meta.modeles.gan;
  const parts: string[] = [];
  if (v) {
    parts.push(`<strong>VAE</strong> : latent ${v.dim_latente}, β = ${fmt(v.beta, 2)}, ${fmt(v.parametres.encodeur + v.parametres.decodeur)}
      paramètres, époque ${v.epoque}/${v.epoques}, ${duree(v.duree_entrainement_s)},
      RMSE de reconstruction (validation) ${fmt(v.validation.rmse_m)} m.`);
  }
  if (g) {
    parts.push(`<strong>GAN</strong> : latent ${g.dim_latente}, générateur ${fmt(g.parametres.generateur)} paramètres
      (discriminateur ${fmt(g.parametres.discriminateur)}, non exporté), perte ${g.perte.toUpperCase()},
      ${g.spectrale ? "normalisation spectrale" : "BatchNorm"}, époque retenue ${g.epoque}/${g.epoques},
      ${duree(g.duree_entrainement_s)}.`);
  }
  parts.push(`Dataset <code>${meta.dataset.identifiant}</code> : ${Object.entries(meta.dataset.repartition)
    .map(([k, n]) => `${k} ${n}`).join(", ")} patches 64 × 64 ; normalisation ${meta.normalisation.methode}
    (${fmt(echelleM)} m).` + (ecart !== null ? ` Contrôle ONNX navigateur ↔ PyTorch : ${fmt(ecart, 4)} m.` : ""));
  $("model-info").innerHTML = parts.join("<br />");
  $("sources").innerHTML = `Données : ${meta.sources.donnees} — <a href="${meta.sources.portail}" target="_blank"
    rel="noopener">gebco.net</a>. Export du ${new Date(meta.date_export).toLocaleString("fr-FR")}.`;
}

void demarrer();
