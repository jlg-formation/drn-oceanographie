import "./style.css";

import {
  Chronologie,
  chargerMetadata,
  chargerSerie,
  extraire,
  MODELES,
  plusProcheValide,
  REFERENCES,
  verifier,
  type Courbe,
  type Metadata,
  type Modele,
  type Periode,
  type Verdict,
} from "./donnees";
import { telechargerCsv, telechargerJson, telechargerPng } from "./exports";
import {
  dessinerCourbes,
  dessinerFrise,
  dessinerPrincipal,
  echantillon,
  fmtDate,
  fmtHeure,
  type Repere,
} from "./graphique";
import { chargerModeles, erreurs, predire, references, type Erreurs } from "./modeles";
import type { InferenceSession } from "onnxruntime-web/wasm";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>("graphique");
const friseCanvas = $<HTMLCanvasElement>("frise");
const placeholder = $("placeholder");
const statusEl = $("status");
const periodeMsg = $("periode-msg");
const dateInput = $<HTMLInputElement>("date");
const testOnly = $<HTMLInputElement>("test-only");
const predireBtn = $<HTMLButtonElement>("predire");
const revelerBtn = $<HTMLButtonElement>("reveler");
const lectureBtn = $<HTMLButtonElement>("lecture");
const pauseBtn = $<HTMLButtonElement>("pause");
const recommencerBtn = $<HTMLButtonElement>("recommencer");
const vitesseEl = $<HTMLInputElement>("vitesse");
const avertPeriode = $("avert-periode");

const COURBES: Courbe[] = [...MODELES, ...REFERENCES];

// --------------------------------------------------------------------------
// État de l'application
// --------------------------------------------------------------------------

let meta: Metadata;
let chrono: Chronologie;
let sessions: Partial<Record<Modele, InferenceSession>> = {};
let entree = 48;
let horizon = 24;

let ancre: number | null = null;
let periode: Periode | null = null;
let verdict: Verdict | null = null;
let previsions: Partial<Record<Courbe, Float32Array>> | null = null;
let revele = false;
let pas = 0; // pas de prévision affichés (animation F25)
let animation: { actif: boolean; dernier: number; cumul: number } = { actif: false, dernier: 0, cumul: 0 };
let survol: number | null = null;
let repere: Repere | null = null;
let versIndiceFrise: ((x: number) => number) | null = null;
const affichees = new Set<Courbe>(["rnn", "gru", "lstm", "repetition_25h"]);

function setStatus(message: string, type: "info" | "ok" | "error" | "busy" = "info"): void {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

const cm = (m: number) => `${(100 * m).toFixed(1).replace(".", ",")} cm`;
const nomCourbe = (c: string) => meta.styles[c]?.nom ?? c;
const testSeulement = () => testOnly.checked;

// --------------------------------------------------------------------------
// Rendu
// --------------------------------------------------------------------------

function titreGraphique(): string {
  if (ancre === null) return "";
  const d = chrono.date(ancre);
  return `${meta.station.nom} — prévision depuis le ${fmtDate(d)} à ${fmtHeure(d)} UTC`;
}

function previsionsAffichees(): Partial<Record<Courbe, Float32Array>> {
  const res: Partial<Record<Courbe, Float32Array>> = {};
  if (!previsions) return res;
  for (const c of COURBES) if (affichees.has(c) && previsions[c]) res[c] = previsions[c];
  return res;
}

function dessiner(): void {
  if (!periode || ancre === null || periode.historique.every((v) => v === null)) {
    canvas.hidden = true;
    placeholder.hidden = false;
    placeholder.textContent = verdict?.raison ?? "Choisissez une période.";
    repere = null;
    return;
  }
  canvas.hidden = false;
  placeholder.hidden = true;
  repere = dessinerPrincipal(canvas, {
    chrono,
    ancre,
    entree,
    horizon,
    historique: periode.historique,
    futur: periode.futur,
    realiteVisible: revele,
    previsions: previsionsAffichees(),
    pas: previsions ? pas : 0,
    styles: meta.styles,
    titre: titreGraphique(),
    survol,
  });
  $("anim-pas").textContent = previsions ? `H+${pas} / H+${horizon}` : "";
}

function dessinerLaFrise(): void {
  versIndiceFrise = dessinerFrise(friseCanvas, chrono, ancre, entree, horizon, testSeulement());
  $("frise-label").textContent = testSeulement()
    ? "Jeu de test — cliquez pour choisir une période"
    : "Série complète 2021–2025 — cliquez pour choisir une période";
}

function majBoutons(): void {
  const pret = Object.keys(sessions).length > 0;
  predireBtn.disabled = !pret || !verdict?.ok;
  revelerBtn.disabled = !previsions || revele;
  revelerBtn.textContent = revele ? "Réalité révélée" : "Révéler la réalité";
  lectureBtn.disabled = !previsions;
  pauseBtn.disabled = !animation.actif;
  recommencerBtn.disabled = !previsions;
  $<HTMLButtonElement>("dl-png").disabled = canvas.hidden;
  $<HTMLButtonElement>("dl-csv").disabled = !previsions;
  $<HTMLButtonElement>("dl-json").disabled = !previsions;
}

// --------------------------------------------------------------------------
// Légende et choix des courbes (F19, F21, §7.4)
// --------------------------------------------------------------------------

function construireChoix(): void {
  const legende = $("legende");
  legende.innerHTML = "";
  for (const fixe of ["observe", "realite"]) {
    const s = document.createElement("span");
    s.className = "item fixe";
    s.append(echantillon(meta.styles[fixe]), meta.styles[fixe].nom);
    legende.append(s);
  }
  for (const c of COURBES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "item";
    b.dataset.courbe = c;
    b.title = "Afficher / masquer cette courbe";
    b.append(echantillon(meta.styles[c]), nomCourbe(c));
    b.addEventListener("click", () => basculer(c, !affichees.has(c)));
    legende.append(b);
  }

  for (const [conteneur, liste] of [
    [$("choix-modeles"), MODELES],
    [$("choix-references"), REFERENCES],
  ] as const) {
    conteneur.innerHTML = "";
    for (const c of liste) {
      const label = document.createElement("label");
      label.className = "check";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.courbe = c;
      input.addEventListener("change", () => basculer(c, input.checked));
      const texte = document.createElement("span");
      const info = (MODELES as readonly string[]).includes(c)
        ? meta.modeles[c as Modele]?.architecture
        : meta.references[c]?.description;
      texte.innerHTML = `<strong>${nomCourbe(c)}</strong><small>${info ?? ""}</small>`;
      label.append(input, echantillon(meta.styles[c]), texte);
      conteneur.append(label);
    }
  }
  synchroniserChoix();
}

function synchroniserChoix(): void {
  document.querySelectorAll<HTMLInputElement>("input[data-courbe]").forEach((i) => {
    const c = i.dataset.courbe as Courbe;
    const indisponible = (MODELES as readonly string[]).includes(c) && !sessions[c as Modele];
    i.checked = affichees.has(c) && !indisponible;
    i.disabled = indisponible && Object.keys(sessions).length > 0;
  });
  document.querySelectorAll<HTMLButtonElement>("#legende button[data-courbe]").forEach((b) => {
    b.setAttribute("aria-pressed", String(affichees.has(b.dataset.courbe as Courbe)));
  });
}

/** Afficher / masquer une courbe : immédiat, sans nouvelle inférence (§7.2). */
function basculer(c: Courbe, visible: boolean): void {
  if (visible) affichees.add(c);
  else affichees.delete(c);
  synchroniserChoix();
  dessiner();
  afficherResultats();
}

// --------------------------------------------------------------------------
// Sélection de la période (F17, F24, scénario 8)
// --------------------------------------------------------------------------

function versInputDate(i: number): string {
  return chrono.date(i).toISOString().slice(0, 16);
}

function depuisInputDate(v: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(v);
  if (!m) return null;
  return chrono.indice(new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5])));
}

function majBornesDate(): void {
  const { debut, fin } = testSeulement() ? chrono.serie.ensembles.test : { debut: 0, fin: chrono.longueur };
  dateInput.min = versInputDate(debut + entree - 1);
  dateInput.max = versInputDate(fin - horizon - 1);
}

function arreterAnimation(): void {
  animation.actif = false;
}

function choisirAncre(t: number): void {
  arreterAnimation();
  ancre = t;
  previsions = null;
  revele = false;
  pas = 0;
  survol = null;
  periode = extraire(chrono, t, entree, horizon);
  verdict = verifier(chrono, t, entree, horizon, testSeulement());
  dateInput.value = versInputDate(t);

  periodeMsg.innerHTML = "";
  if (!verdict.ok) {
    periodeMsg.className = "message error";
    periodeMsg.append(`⚠ ${verdict.raison} `);
    const alt = plusProcheValide(chrono, t, entree, horizon, testSeulement());
    if (alt !== null) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "lien";
      const d = chrono.date(alt);
      b.textContent = `Aller à la période valide la plus proche (${fmtDate(d)} ${fmtHeure(d)})`;
      b.addEventListener("click", () => choisirAncre(alt));
      periodeMsg.append(b);
    }
  } else {
    const d0 = chrono.date(t - entree + 1);
    const d1 = chrono.date(t + horizon);
    periodeMsg.className = "message ok";
    periodeMsg.textContent =
      `✓ Période complète : historique du ${fmtDate(d0)} ${fmtHeure(d0)} au ${fmtDate(chrono.date(t))} ` +
      `${fmtHeure(chrono.date(t))}, réalité jusqu'au ${fmtDate(d1)} ${fmtHeure(d1)} UTC.`;
  }

  // Avertissement si la période a été vue pendant l'apprentissage (F17).
  const ens = new Set([chrono.ensemble(t - entree + 1), chrono.ensemble(t + horizon)]);
  if (!ens.has("test") || ens.size > 1) {
    avertPeriode.hidden = false;
    avertPeriode.textContent = ens.has("train")
      ? "⚠ Cette période appartient à l'ensemble d'ENTRAÎNEMENT : les modèles l'ont déjà vue. " +
        "Les erreurs affichées ne mesurent pas leur capacité à prévoir des données inconnues."
      : "⚠ Cette période n'est pas entièrement dans le jeu de test (validation : elle a servi à choisir " +
        "le meilleur modèle). Ce n'est pas une évaluation sur données inconnues.";
  } else avertPeriode.hidden = true;

  document.querySelectorAll<HTMLButtonElement>(".exemple").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.dataset.ancre) === t));
  });
  dessinerLaFrise();
  dessiner();
  afficherResultats();
  majBoutons();
  if (verdict.ok && Object.keys(sessions).length) {
    setStatus("Observez les 48 h d'historique, choisissez les modèles, puis lancez la prévision.", "info");
  }
}

function deplacer(delta: number): void {
  if (ancre === null) return;
  const { debut, fin } = testSeulement() ? chrono.serie.ensembles.test : { debut: 0, fin: chrono.longueur };
  choisirAncre(Math.max(debut + entree - 1, Math.min(fin - horizon - 1, ancre + delta)));
}

function auHasard(): void {
  const { debut, fin } = testSeulement() ? chrono.serie.ensembles.test : { debut: 0, fin: chrono.longueur };
  for (let essai = 0; essai < 200; essai++) {
    const t = debut + entree - 1 + Math.floor(Math.random() * (fin - debut - entree - horizon));
    if (verifier(chrono, t, entree, horizon, testSeulement()).ok) {
      choisirAncre(t);
      return;
    }
  }
}

function construireExemples(): void {
  const conteneur = $("exemples");
  conteneur.innerHTML = "";
  if (!meta.exemples.length) {
    conteneur.innerHTML = '<p class="note">Aucun exemple : lancez `mise run tp5_04_evaluate` puis tp5_05_export.</p>';
    return;
  }
  for (const ex of meta.exemples) {
    const t = chrono.indice(new Date(ex.ancre));
    const d = chrono.date(t);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "exemple";
    b.dataset.ancre = String(t);
    b.title = ex.description;
    b.innerHTML =
      `<span class="ex-titre">${ex.titre}</span>` +
      `<span class="ex-sous">${fmtDate(d)} ${fmtHeure(d)} UTC · marnage ${ex.marnage_m.toFixed(1).replace(".", ",")} m</span>`;
    b.addEventListener("click", () => {
      if (!testSeulement()) {
        testOnly.checked = true;
        majBornesDate();
      }
      choisirAncre(t);
    });
    conteneur.append(b);
  }
}

// --------------------------------------------------------------------------
// Prévision, révélation, animation (F20, F22, F25)
// --------------------------------------------------------------------------

async function lancerPrevision(): Promise<void> {
  if (!periode || !verdict?.ok || ancre === null) return;
  const t = ancre;
  const historique = Float32Array.from(periode.historique as number[]);
  predireBtn.disabled = true;
  setStatus("Inférence en cours…", "busy");
  try {
    const res: Partial<Record<Courbe, Float32Array>> = { ...references(historique, horizon) };
    const durees: string[] = [];
    for (const m of MODELES) {
      const s = sessions[m];
      if (!s) continue;
      const t0 = performance.now();
      res[m] = await predire(s, meta, historique);
      durees.push(`${nomCourbe(m)} ${(performance.now() - t0).toFixed(1).replace(".", ",")} ms`);
    }
    if (ancre !== t) return; // une autre période a été choisie entre-temps
    previsions = res;
    pas = horizon;
    revele = false;
    if (![...affichees].some((c) => res[c])) affichees.add("lstm");
    synchroniserChoix();
    dessiner();
    afficherResultats();
    setStatus(
      `Prévision calculée dans le navigateur (${durees.join(", ")}). ` +
        "Quelle courbe vous semble la plus crédible ? Puis « Révéler la réalité ».",
      "ok",
    );
  } catch (err) {
    console.error(err);
    setStatus(`Erreur pendant l'inférence : ${(err as Error).message}`, "error");
  } finally {
    majBoutons();
  }
}

function reveler(): void {
  if (!previsions) return;
  revele = true;
  dessiner();
  afficherResultats();
  majBoutons();
  setStatus("Réalité révélée : comparez les courbes et le tableau des erreurs.", "ok");
}

function boucleAnimation(maintenant: number): void {
  if (!animation.actif) return;
  const dt = (maintenant - animation.dernier) / 1000;
  animation.dernier = maintenant;
  animation.cumul += dt * Number(vitesseEl.value);
  const nouveau = Math.min(horizon, Math.floor(animation.cumul));
  if (nouveau !== pas) {
    pas = nouveau;
    dessiner();
  }
  if (pas >= horizon) {
    animation.actif = false;
    majBoutons();
    return;
  }
  requestAnimationFrame(boucleAnimation);
}

/** Lecture : révèle les 24 points heure par heure, sans relancer l'inférence. */
function lecture(depuisZero = false): void {
  if (!previsions) return;
  if (depuisZero || pas >= horizon) pas = 0;
  animation = { actif: true, dernier: performance.now(), cumul: pas };
  dessiner();
  majBoutons();
  requestAnimationFrame(boucleAnimation);
}

function pause(): void {
  arreterAnimation();
  majBoutons();
}

// --------------------------------------------------------------------------
// Erreurs sur la période (F23)
// --------------------------------------------------------------------------

function erreursPeriode(): Partial<Record<Courbe, Erreurs>> | null {
  if (!previsions || !revele || !periode || periode.futur.some((v) => v === null)) return null;
  const res: Partial<Record<Courbe, Erreurs>> = {};
  for (const c of COURBES) {
    const p = previsions[c];
    if (p && affichees.has(c)) res[c] = erreurs(p, periode.futur as number[], meta.horizons_eval);
  }
  return res;
}

function afficherResultats(): void {
  const zone = $("resultats-contenu");
  const e = erreursPeriode();
  if (!e || !Object.keys(e).length) {
    zone.className = "note";
    zone.textContent = !previsions
      ? "Lancez une prévision, puis révélez la réalité pour calculer les erreurs."
      : !revele
        ? "Les erreurs seront calculées quand vous révélerez la réalité."
        : "Sélectionnez au moins une courbe pour voir ses erreurs.";
    return;
  }
  const lignes = (Object.entries(e) as [Courbe, Erreurs][]).sort((a, b) => a[1].mae - b[1].mae);
  const meilleur = lignes[0][0];
  const table = document.createElement("table");
  table.className = "table";
  const hs = meta.horizons_eval.map((h) => `<th title="Erreur absolue à l'échéance H+${h}">|e| H+${h}</th>`).join("");
  table.innerHTML = `<thead><tr><th>Modèle</th><th>MAE 24 h</th><th>RMSE 24 h</th>${hs}</tr></thead>`;
  const corps = document.createElement("tbody");
  for (const [c, r] of lignes) {
    const tr = document.createElement("tr");
    if (c === meilleur) tr.className = "meilleur";
    const td = document.createElement("td");
    td.className = "nom";
    td.append(echantillon(meta.styles[c], 30, 12), nomCourbe(c));
    if (c === meilleur) td.append(Object.assign(document.createElement("span"), { className: "tag", textContent: "plus faible erreur" }));
    tr.append(td);
    tr.insertAdjacentHTML(
      "beforeend",
      `<td>${cm(r.mae)}</td><td>${cm(r.rmse)}</td>` +
        meta.horizons_eval.map((h) => `<td>${cm(r.echeances[`H+${h}`])}</td>`).join(""),
    );
    corps.append(tr);
  }
  table.append(corps);
  zone.className = "";
  zone.innerHTML = "";
  const resume = document.createElement("p");
  resume.className = "verdict";
  resume.innerHTML =
    `Sur <strong>cette période de 24 h</strong>, la plus faible erreur moyenne est celle de ` +
    `<strong>${nomCourbe(meilleur)}</strong> (MAE ${cm(e[meilleur]!.mae)}).`;
  const prudence = document.createElement("p");
  prudence.className = "note";
  prudence.textContent =
    "Ce classement est local à la période choisie : il ne prouve pas qu'un modèle est meilleur en général. " +
    "Pour cela, voyez la comparaison sur tout le jeu de test ci-dessous.";
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.append(table);
  zone.append(resume, scroll, prudence);
}

// --------------------------------------------------------------------------
// Comparaison sur tout le jeu de test et courbes d'apprentissage (F10, F13, F14)
// --------------------------------------------------------------------------

function afficherGlobal(): void {
  const t = meta.test;
  const intro = $("global-intro");
  const table = $<HTMLTableElement>("table-globale");
  if (!t) {
    intro.textContent = "Évaluation absente : lancez `mise run tp5_04_evaluate` puis tp5_05_export.";
    return;
  }
  const ref = t.meilleure_reference;
  const gains = MODELES.filter((m) => t.metriques[m])
    .map((m) => `${nomCourbe(m)} −${Math.round(100 * (1 - t.metriques[m].global.mae / t.metriques[ref].global.mae))} %`)
    .join(", ");
  const d0 = new Date(t.debut);
  const d1 = new Date(t.fin);
  intro.innerHTML =
    `Même protocole pour tous : <strong>${t.fenetres.toLocaleString("fr-FR")} fenêtres</strong> du ` +
    `${fmtDate(d0)} au ${fmtDate(d1)}, période jamais vue pendant l'apprentissage. ` +
    `Erreur des réseaux par rapport à la meilleure référence (${nomCourbe(ref)}) : ${gains}.`;
  const hs = meta.horizons_eval.map((h) => `<th>MAE H+${h}</th>`).join("");
  table.innerHTML = `<thead><tr><th>Modèle</th>${hs}<th>MAE 24 h</th><th>RMSE 24 h</th></tr></thead>`;
  const corps = document.createElement("tbody");
  t.classement_mae.forEach((c, rang) => {
    const r = t.metriques[c];
    const tr = document.createElement("tr");
    if (rang === 0) tr.className = "meilleur";
    const td = document.createElement("td");
    td.className = "nom";
    td.append(echantillon(meta.styles[c], 30, 12), nomCourbe(c));
    tr.append(td);
    tr.insertAdjacentHTML(
      "beforeend",
      meta.horizons_eval.map((h) => `<td>${cm(r.horizons[`H+${h}`].mae)}</td>`).join("") +
        `<td><strong>${cm(r.global.mae)}</strong></td><td>${cm(r.global.rmse)}</td>`,
    );
    corps.append(tr);
  });
  table.append(corps);
}

function dessinerMiniGraphiques(): void {
  const t = meta.test;
  if (t) {
    const courbes = COURBES.filter((c) => t.metriques[c]);
    const sansPersistance = courbes.filter((c) => c !== "persistance");
    const ymax = 100 * Math.max(...sansPersistance.flatMap((c) => t.metriques[c].mae_par_pas));
    dessinerCourbes(
      $<HTMLCanvasElement>("chart-horizon"),
      courbes.map((c) => ({ style: meta.styles[c], y: t.metriques[c].mae_par_pas.map((v) => 100 * v) })),
      {
        titre: "MAE selon l'échéance (persistance hors échelle)",
        xlabel: "échéance H+k (heures)",
        ylabel: "MAE (cm)",
        ymax,
        xTicks: [1, 6, 12, 18, 24],
      },
    );
  }
  const series = MODELES.flatMap((m) => {
    const h = meta.modeles[m]?.historique;
    if (!h) return [];
    return [
      { style: meta.styles[m], y: h.perte_val },
      { style: meta.styles[m], y: h.perte_train, epais: false },
    ];
  });
  if (series.length) {
    dessinerCourbes($<HTMLCanvasElement>("chart-apprentissage"), series, {
      titre: "Apprentissage : perte de validation (épais) et d'entraînement (fin)",
      xlabel: "époque",
      ylabel: "MSE normalisée (log)",
      logY: true,
    });
  }
}

// --------------------------------------------------------------------------
// Exports (F27)
// --------------------------------------------------------------------------

function nomFichier(ext: string): string {
  const d = ancre === null ? new Date() : chrono.date(ancre);
  const iso = d.toISOString().slice(0, 13).replace(/[-T:]/g, "");
  return `mareelab_${meta.station.nom.toLowerCase()}_${iso}h.${ext}`;
}

function exporterCsv(): void {
  if (!periode || !previsions || ancre === null) return;
  const courbes = COURBES.filter((c) => previsions![c]);
  const colonnes = ["horodatage_utc", "echeance", "phase", "observation_m", ...courbes.map((c) => `${c}_m`)];
  const lignes: (string | number | null)[][] = [];
  for (let o = -entree + 1; o <= horizon; o++) {
    const futur = o > 0;
    const obs = futur ? (revele ? periode.futur[o - 1] : null) : periode.historique[o + entree - 1];
    lignes.push([
      chrono.date(ancre + o).toISOString().replace(".000", ""),
      o === 0 ? "H" : o > 0 ? `H+${o}` : `H${o}`,
      futur ? (revele ? "futur (observé)" : "futur (réalité masquée)") : "historique",
      obs,
      ...courbes.map((c) => (futur ? previsions![c]![o - 1] : null)),
    ]);
  }
  telechargerCsv(colonnes, lignes, nomFichier("csv"));
}

function exporterJson(): void {
  if (!previsions || ancre === null) return;
  const e = erreursPeriode();
  const classement = e ? (Object.entries(e) as [Courbe, Erreurs][]).sort((a, b) => a[1].mae - b[1].mae) : [];
  telechargerJson(
    {
      application: meta.application,
      avertissement: meta.avertissement,
      station: meta.station,
      unite: "m",
      reference_verticale: meta.reference_verticale?.nom,
      fuseau: "UTC",
      instant_prevision: chrono.date(ancre).toISOString(),
      entree_h: entree,
      horizon_h: horizon,
      ensemble: chrono.ensemble(ancre),
      jeu_de_test_uniquement: testSeulement(),
      realite_revelee: revele,
      metriques_periode: e
        ? Object.fromEntries(
            classement.map(([c, r]) => [
              c,
              {
                mae_m: r.mae,
                rmse_m: r.rmse,
                erreur_absolue_m: r.echeances,
              },
            ]),
          )
        : null,
      plus_faible_erreur_periode: classement[0]?.[0] ?? null,
      remarque: "Classement local à cette période : il ne prouve pas la supériorité générale d'un modèle.",
      metriques_jeu_de_test: meta.test
        ? Object.fromEntries(Object.entries(meta.test.metriques).map(([c, m]) => [c, { global: m.global, horizons: m.horizons }]))
        : null,
      modeles: Object.fromEntries(
        Object.entries(meta.modeles).map(([m, i]) => [m, { architecture: i!.architecture, parametres: i!.parametres, epoque: i!.epoque }]),
      ),
      dataset: meta.dataset.identifiant,
      date_export: new Date().toISOString(),
    },
    nomFichier("json"),
  );
}

// --------------------------------------------------------------------------
// Initialisation
// --------------------------------------------------------------------------

function brancherEvenements(): void {
  predireBtn.addEventListener("click", () => void lancerPrevision());
  revelerBtn.addEventListener("click", reveler);
  lectureBtn.addEventListener("click", () => lecture());
  pauseBtn.addEventListener("click", pause);
  recommencerBtn.addEventListener("click", () => lecture(true));
  vitesseEl.addEventListener("input", () => {
    $("vitesse-val").textContent = `${vitesseEl.value} h/s`;
  });
  dateInput.addEventListener("change", () => {
    const t = depuisInputDate(dateInput.value);
    if (t !== null) choisirAncre(t);
  });
  testOnly.addEventListener("change", () => {
    majBornesDate();
    if (ancre !== null) choisirAncre(ancre);
  });
  document.querySelectorAll<HTMLButtonElement>(".nav button[data-pas]").forEach((b) => {
    b.addEventListener("click", () => deplacer(Number(b.dataset.pas)));
  });
  $("hasard").addEventListener("click", auHasard);

  friseCanvas.addEventListener("click", (e) => {
    if (!versIndiceFrise) return;
    const r = friseCanvas.getBoundingClientRect();
    const centre = versIndiceFrise(e.clientX - r.left);
    // On place la fenêtre de 72 h centrée sur le clic.
    choisirAncre(centre + Math.round((entree - horizon) / 2));
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!repere) return;
    const r = canvas.getBoundingClientRect();
    const o = repere.decalage(e.clientX - r.left);
    if (o !== survol) {
      survol = o;
      dessiner();
    }
  });
  canvas.addEventListener("pointerleave", () => {
    survol = null;
    dessiner();
  });

  $("dl-png").addEventListener("click", () => {
    const s = survol;
    survol = null; // pas d'infobulle dans l'image exportée
    dessiner();
    telechargerPng(canvas, nomFichier("png"))
      .catch((err) => setStatus((err as Error).message, "error"))
      .finally(() => {
        survol = s;
      });
  });
  $("dl-csv").addEventListener("click", exporterCsv);
  $("dl-json").addEventListener("click", exporterJson);

  let planifie = false;
  new ResizeObserver(() => {
    if (planifie) return;
    planifie = true;
    requestAnimationFrame(() => {
      planifie = false;
      if (!chrono) return;
      dessiner();
      dessinerLaFrise();
      dessinerMiniGraphiques();
    });
  }).observe(document.body);
}

function afficherInfos(): void {
  const infos = MODELES.filter((m) => meta.modeles[m]).map((m) => {
    const i = meta.modeles[m]!;
    return (
      `${nomCourbe(m)} : ${i.parametres.toLocaleString("fr-FR")} paramètres (état caché ${i.cachee}, ` +
      `${i.couches} couche${i.couches > 1 ? "s" : ""}), meilleure époque ${i.epoque}/${i.epoques}, ` +
      `MAE de validation ${cm(i.validation.mae_m)}, entraîné en ${Math.round(i.duree_entrainement_s)} s`
    );
  });
  $("model-info").textContent =
    `Modèles (dataset ${meta.dataset.identifiant}) — ${infos.join(" · ")}. ` +
    `Entrée : ${entree} hauteurs horaires, sortie : ${horizon} hauteurs (prévision directe).`;
  const ref = meta.reference_verticale;
  $("sources").textContent =
    `Données : ${meta.sources.observations} — ${meta.sources.qualite}, téléchargées le ` +
    `${new Date(meta.sources.date_telechargement).toLocaleDateString("fr-FR")} depuis ${meta.sources.portail}. ` +
    (ref ? `Hauteurs en mètres au-dessus du ${ref.nom} (ZH = ${ref.zh_m} m / ${ref.zh_par_rapport_a}). ` : "") +
    "Heures en UTC. " +
    meta.avertissement;
}

async function main(): Promise<void> {
  brancherEvenements();
  try {
    const [m, s] = await Promise.all([chargerMetadata(), chargerSerie()]);
    meta = m;
    chrono = new Chronologie(s);
    entree = m.inference.entree_h;
    horizon = m.inference.horizon_h;
  } catch (err) {
    placeholder.textContent = (err as Error).message;
    setStatus((err as Error).message, "error");
    return;
  }

  construireChoix();
  construireExemples();
  afficherGlobal();
  afficherInfos();
  majBornesDate();
  const premier = meta.exemples.find((e) => e.id === "typique") ?? meta.exemples[0];
  if (premier) choisirAncre(chrono.indice(new Date(premier.ancre)));
  else auHasard();
  dessinerMiniGraphiques();

  setStatus("Chargement des modèles ONNX…", "busy");
  try {
    const c = await chargerModeles(meta);
    sessions = c.sessions;
    const ok = Object.keys(sessions).map((k) => nomCourbe(k));
    const ko = Object.entries(c.erreurs).map(([k, v]) => `${nomCourbe(k)} : ${v}`);
    synchroniserChoix();
    if (!ok.length) {
      setStatus(`Aucun modèle n'a pu être chargé. ${ko.join(" ; ")}`, "error");
    } else {
      const controle =
        c.ecartControle === null
          ? ""
          : ` Contrôle navigateur ↔ PyTorch : écart max ${(1000 * c.ecartControle).toFixed(3).replace(".", ",")} mm` +
            (c.ecartControle < 1e-3 ? " ✓." : " ⚠ (supérieur à 1 mm).");
      setStatus(
        `${ok.length} modèle(s) chargé(s) : ${ok.join(", ")} (ONNX Runtime Web, WebAssembly).${controle}` +
          (ko.length ? ` Indisponible : ${ko.join(" ; ")}.` : "") +
          (verdict?.ok ? " Cliquez sur « Prédire les prochaines 24 heures »." : ""),
        ko.length ? "error" : "ok",
      );
    }
  } catch (err) {
    setStatus(`Chargement des modèles impossible : ${(err as Error).message}`, "error");
  }
  majBoutons();
}

void main();
