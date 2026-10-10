/** Chargement des fichiers produits par le pipeline Python et sélection des périodes (F17). */

export type Courbe = "rnn" | "gru" | "lstm" | "persistance" | "repetition_24h" | "repetition_25h";
export const MODELES = ["rnn", "gru", "lstm"] as const;
export const REFERENCES = ["persistance", "repetition_24h", "repetition_25h"] as const;
export type Modele = (typeof MODELES)[number];

export interface Style {
  nom: string;
  couleur: string;
  tirets: number[];
  marqueur: "aucun" | "rond" | "triangle" | "carre" | "losange" | "croix" | "plus" | "cercle";
}

export interface Metrique {
  mae: number;
  rmse: number;
}

export interface Exemple {
  id: string;
  titre: string;
  description: string;
  ancre: string;
  marnage_m: number;
  niveau_moyen_m: number;
}

export interface InfoModele {
  fichier: string;
  architecture: string;
  cachee: number;
  couches: number;
  parametres: number;
  epoque: number;
  epoques: number;
  duree_entrainement_s: number;
  validation: { mae_m: number; rmse_m: number };
  historique: { perte_train: number[]; perte_val: number[]; val_mae_m: number[] };
  verification_onnx: { ecart_max_m: number; conforme: boolean; operateurs_recurrents: string[] };
  dataset: string;
  date_creation: string;
}

export interface Metadata {
  application: string;
  avertissement: string;
  station: { nom: string; id_shom: number; latitude: number; longitude: number; reseau: string };
  unite: string;
  fuseau: string;
  reference_verticale: { nom: string; zh_m: string; zh_par_rapport_a: string } | null;
  inference: { entree: string; sortie: string; entree_h: number; horizon_h: number; onnx_opset: number };
  normalisation: { moyenne_m: number; ecart_type_m: number };
  modeles: Partial<Record<Modele, InfoModele>>;
  references: Record<string, { description: string }>;
  styles: Record<string, Style>;
  horizons_eval: number[];
  test: {
    debut: string;
    fin: string;
    fenetres: number;
    metriques: Record<string, { global: Metrique; horizons: Record<string, Metrique>; mae_par_pas: number[] }>;
    classement_mae: string[];
    meilleure_reference: string;
  } | null;
  exemples: Exemple[];
  controle: { ancre: string; entree: number[]; attendu: Partial<Record<Modele, number[]>> };
  dataset: { identifiant: string };
  sources: { observations: string; portail: string; qualite: string; date_telechargement: string };
}

export interface Ensemble {
  debut: number;
  fin: number;
  debut_iso: string;
  fin_iso: string;
}

export interface Serie {
  station: string;
  debut: string;
  pas_h: number;
  ensembles: { train: Ensemble; val: Ensemble; test: Ensemble };
  valeurs: (number | null)[];
}

/** Erreur lisible : quel fichier, pourquoi, et comment le régénérer. */
export class DataError extends Error {
  constructor(fichier: string, raison: string) {
    super(
      `Impossible de charger « ${fichier} » (${raison}). ` +
        "Lancez `mise run tp5` (ou au moins tp5_05_export) puis rechargez la page.",
    );
  }
}

async function fetchJson<T>(fichier: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`./${fichier}`);
  } catch {
    throw new DataError(fichier, "serveur injoignable");
  }
  if (!res.ok) throw new DataError(fichier, `HTTP ${res.status}`);
  // Le serveur de dev Vite répond index.html (200) pour un fichier inexistant.
  if (res.headers.get("content-type")?.includes("text/html")) throw new DataError(fichier, "fichier absent");
  return (await res.json()) as T;
}

export const chargerMetadata = (): Promise<Metadata> => fetchJson<Metadata>("metadata.json");
export const chargerSerie = (): Promise<Serie> => fetchJson<Serie>("serie.json");

// --------------------------------------------------------------------------
// Temps : indices horaires <-> dates UTC
// --------------------------------------------------------------------------

const HEURE_MS = 3_600_000;

export class Chronologie {
  readonly t0: number;

  constructor(readonly serie: Serie) {
    this.t0 = Date.parse(serie.debut);
  }

  get longueur(): number {
    return this.serie.valeurs.length;
  }

  date(i: number): Date {
    return new Date(this.t0 + i * HEURE_MS);
  }

  /** Indice de l'heure UTC la plus proche (arrondi à l'heure). */
  indice(d: Date): number {
    return Math.round((d.getTime() - this.t0) / HEURE_MS);
  }

  ensemble(i: number): "train" | "val" | "test" | null {
    for (const [nom, e] of Object.entries(this.serie.ensembles)) {
      if (i >= e.debut && i < e.fin) return nom as "train" | "val" | "test";
    }
    return null;
  }
}

// --------------------------------------------------------------------------
// Validation d'une période (F17, scénario 8)
// --------------------------------------------------------------------------

export interface Periode {
  ancre: number; // indice de H, dernière heure observée
  historique: (number | null)[]; // H-47 .. H
  futur: (number | null)[]; // H+1 .. H+24
}

export interface Verdict {
  ok: boolean;
  raison?: string;
  manquantesHistorique: number;
  manquantesFutur: number;
}

export function extraire(chrono: Chronologie, ancre: number, entree: number, horizon: number): Periode {
  const v = chrono.serie.valeurs;
  const tranche = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => v[a + k] ?? null);
  return { ancre, historique: tranche(ancre - entree + 1, ancre + 1), futur: tranche(ancre + 1, ancre + horizon + 1) };
}

/** Bornes [debut, fin) autorisées pour l'ensemble de la fenêtre (72 h). */
export function domaine(chrono: Chronologie, testSeulement: boolean): { debut: number; fin: number } {
  return testSeulement ? chrono.serie.ensembles.test : { debut: 0, fin: chrono.longueur };
}

export function verifier(
  chrono: Chronologie,
  ancre: number,
  entree: number,
  horizon: number,
  testSeulement: boolean,
): Verdict {
  const { debut, fin } = domaine(chrono, testSeulement);
  const vide = { manquantesHistorique: 0, manquantesFutur: 0 };
  if (ancre - entree + 1 < debut || ancre + horizon >= fin) {
    const ou = testSeulement ? "du jeu de test" : "des données disponibles";
    return {
      ok: false,
      raison: `La fenêtre (48 h d'historique + 24 h futures) sort ${ou}.`,
      ...vide,
    };
  }
  const p = extraire(chrono, ancre, entree, horizon);
  const mh = p.historique.filter((x) => x === null).length;
  const mf = p.futur.filter((x) => x === null).length;
  if (mh || mf) {
    return {
      ok: false,
      raison:
        `Période incomplète : ${mh} heure(s) manquante(s) dans l'historique et ${mf} dans les 24 h futures. ` +
        "Les lacunes ne sont jamais comblées : cette période n'est pas exploitable.",
      manquantesHistorique: mh,
      manquantesFutur: mf,
    };
  }
  return { ok: true, manquantesHistorique: 0, manquantesFutur: 0 };
}

/** Ancre valide la plus proche (scénario 8 : proposer une autre sélection). */
export function plusProcheValide(
  chrono: Chronologie,
  ancre: number,
  entree: number,
  horizon: number,
  testSeulement: boolean,
): number | null {
  const { debut, fin } = domaine(chrono, testSeulement);
  const min = debut + entree - 1;
  const max = fin - horizon - 1;
  for (let d = 0; d <= fin - debut; d++) {
    for (const c of [ancre - d, ancre + d]) {
      if (c >= min && c <= max && verifier(chrono, c, entree, horizon, testSeulement).ok) return c;
    }
  }
  return null;
}
