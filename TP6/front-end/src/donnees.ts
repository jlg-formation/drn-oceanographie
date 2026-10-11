/**
 * Chargement des artefacts du pipeline Python (TP6/model) : métadonnées,
 * patches réels de test et instantanés du GAN au fil des époques.
 */

export type Modele = "vae" | "gan";
export const MODELES: Modele[] = ["vae", "gan"];
export const TAILLE = 64;
export const N_PIXELS = TAILLE * TAILLE;

export interface Verification {
  entrees: number;
  ecart_max: number;
  conforme: boolean;
  taille_octets: number;
}

export interface InfoVae {
  fichiers: { encodeur: string; decodeur: string };
  dim_latente: number;
  beta: number;
  largeur: number;
  parametres: { encodeur: number; decodeur: number };
  epoque: number;
  epoques: number;
  duree_entrainement_s: number | null;
  validation: { perte: number; reconstruction: number; kl: number; rmse_m: number };
  historique: Record<
    "train_perte" | "val_perte" | "train_reconstruction" | "val_reconstruction" | "train_kl" | "val_kl" | "val_rmse_m",
    number[]
  >;
  verification_onnx: { encodeur: Verification; decodeur: Verification };
  dataset: string;
}

export interface InfoGan {
  fichiers: { generateur: string };
  dim_latente: number;
  largeur: number;
  perte: string;
  spectrale: boolean;
  parametres: { generateur: number; discriminateur: number };
  epoque: number;
  epoques: number;
  duree_entrainement_s: number | null;
  selection: string;
  historique: {
    perte_d: number[];
    perte_g: number[];
    d_reel: number[];
    d_faux: number[];
    suivi_epoques: number[];
    ecart_stat: number[];
    diversite_relative: number[];
  };
  evolution: { fichier: string; epoques: number[]; cartes: number; graines: number[] } | null;
  verification_onnx: { generateur: Verification };
  dataset: string;
}

export interface PatchReel {
  i: number;
  region: string;
  nom_region: string;
  nord: number;
  sud: number;
  ouest: number;
  est: number;
  profondeur_moy_m: number;
  ecart_type_m: number;
}

export interface Metadata {
  application: string;
  avertissement: string;
  patch: number;
  pixel_km: number;
  normalisation: { echelle_m: number; methode: string; part_ecretee_train: number; conversion: string };
  modeles: { vae?: InfoVae; gan?: InfoGan };
  styles: Record<string, { nom: string; couleur: string }>;
  palette: string[];
  reels: { fichier: string; patches: PatchReel[] };
  regions: Record<string, { nom: string; morphologie: string }>;
  evaluation: {
    patches_test: number;
    diversite: {
      quasi_doublon: string;
      par_source: Record<string, {
        diversite_relative: number;
        part_quasi_doublons: number;
        distance_ppv_train_mediane_m: number;
        distance_moyenne_paires_m: number;
      }>;
    };
    statistiques: Record<string, {
      medianes: Record<string, number>;
      w1_relatif_moyen?: number;
      pente_pixels_p90_deg: number;
      part_valeurs_saturees: number;
    }>;
    synthese_markdown: string;
    modeles: {
      vae?: { reconstruction: { rmse_m: number; rmse_mediane_patch_m: number } };
      gan?: { stabilite: Record<string, number> };
    };
  } | null;
  controle: Partial<Record<Modele, { graine: number; z: number[]; relief_echantillon: number[]; mu_reel_0?: number[] }>>;
  dataset: { identifiant: string; repartition: Record<string, number> };
  sources: { donnees: string; serveur: string; portail: string };
  date_export: string;
}

async function lireBinaire(url: string): Promise<ArrayBuffer> {
  const rep = await fetch(url);
  if (!rep.ok) throw new Error(`${url} : HTTP ${rep.status}`);
  return rep.arrayBuffer();
}

export async function chargerMetadata(): Promise<Metadata> {
  const rep = await fetch("./metadata.json");
  if (!rep.ok) {
    throw new Error("metadata.json introuvable : lancez `mise run tp6_07_export` (export ONNX et données du site).");
  }
  return rep.json();
}

/** Patches réels (élévations en mètres, ligne 0 = nord). */
export async function chargerReels(meta: Metadata): Promise<Float32Array[]> {
  const brut = new Int16Array(await lireBinaire(`./${meta.reels.fichier}`));
  const n = meta.reels.patches.length;
  if (brut.length !== n * N_PIXELS) throw new Error("reels.bin : taille inattendue");
  return Array.from({ length: n }, (_, k) => Float32Array.from(brut.subarray(k * N_PIXELS, (k + 1) * N_PIXELS)));
}

/** Instantanés du GAN : [époque][carte] -> relief relatif (m). */
export async function chargerEvolution(meta: Metadata): Promise<Float32Array[][] | null> {
  const ev = meta.modeles.gan?.evolution;
  if (!ev) return null;
  const brut = new Uint8Array(await lireBinaire(`./${ev.fichier}`));
  const e = meta.normalisation.echelle_m;
  return ev.epoques.map((_, k) =>
    Array.from({ length: ev.cartes }, (_, c) => {
      const debut = (k * ev.cartes + c) * N_PIXELS;
      return Float32Array.from(brut.subarray(debut, debut + N_PIXELS), (v) => (v / 127.5 - 1) * e);
    }),
  );
}

/** Relief relatif : élévation moins sa moyenne (convention des modèles). */
export function centrer(elevation: Float32Array): Float32Array {
  let s = 0;
  for (const v of elevation) s += v;
  const m = s / elevation.length;
  return Float32Array.from(elevation, (v) => v - m);
}
