/**
 * Inférence des réseaux récurrents dans le navigateur (F20, NF04) et références naïves (F15).
 *
 * Les modèles sont minuscules (48 valeurs en entrée, < 60 000 paramètres) :
 * une inférence prend quelques millisecondes en WebAssembly, sur le fil principal.
 */
import * as ort from "onnxruntime-web/wasm";

import { MODELES, type Courbe, type Metadata, type Modele } from "./donnees";

// Les binaires WASM d'onnxruntime-web sont servis depuis le CDN jsDelivr,
// à la version installée dans node_modules (voir package.json).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

export interface Chargement {
  sessions: Partial<Record<Modele, ort.InferenceSession>>;
  erreurs: Partial<Record<Modele, string>>;
  /** Écart maximal (m) entre ONNX dans le navigateur et PyTorch sur l'exemple de contrôle. */
  ecartControle: number | null;
}

export async function chargerModeles(meta: Metadata): Promise<Chargement> {
  const res: Chargement = { sessions: {}, erreurs: {}, ecartControle: null };
  await Promise.all(
    MODELES.map(async (m) => {
      const info = meta.modeles[m];
      if (!info) {
        res.erreurs[m] = "absent des métadonnées (modèle non entraîné ou non exporté)";
        return;
      }
      try {
        res.sessions[m] = await ort.InferenceSession.create(`./${info.fichier}`, { executionProviders: ["wasm"] });
      } catch (err) {
        res.erreurs[m] = err instanceof Error ? err.message : String(err);
      }
    }),
  );

  // Contrôle de cohérence (F12) : on recalcule l'exemple de référence exporté par Python.
  const entree = Float32Array.from(meta.controle.entree);
  let ecart = 0;
  for (const m of MODELES) {
    const attendu = meta.controle.attendu[m];
    if (!res.sessions[m] || !attendu) continue;
    const sortie = await predire(res.sessions[m]!, meta, entree);
    attendu.forEach((v, k) => (ecart = Math.max(ecart, Math.abs(v - sortie[k]))));
  }
  res.ecartControle = Object.keys(res.sessions).length ? ecart : null;
  return res;
}

/** Prévision de 24 h (m) à partir de 48 hauteurs (m). Normalisation incluse dans le graphe. */
export async function predire(
  session: ort.InferenceSession,
  meta: Metadata,
  historique: Float32Array,
): Promise<Float32Array> {
  const tenseur = new ort.Tensor("float32", historique, [1, meta.inference.entree_h]);
  const sorties = await session.run({ [meta.inference.entree]: tenseur });
  const prevision = (await sorties[meta.inference.sortie].getData()) as Float32Array;
  tenseur.dispose();
  return Float32Array.from(prevision);
}

/** Références naïves calculées à partir des seules 48 h d'historique (aucune fuite, NF07). */
export function references(historique: Float32Array, horizon: number): Record<Courbe, Float32Array> {
  const n = historique.length;
  const repetition = (decalage: number) => historique.slice(n - decalage, n - decalage + horizon);
  return {
    persistance: new Float32Array(horizon).fill(historique[n - 1]),
    repetition_24h: repetition(24),
    repetition_25h: repetition(25),
  } as Record<Courbe, Float32Array>;
}

// --------------------------------------------------------------------------
// Métriques locales à la période (F23)
// --------------------------------------------------------------------------

export interface Erreurs {
  mae: number;
  rmse: number;
  /** |erreur| aux échéances détaillées (H+1, H+6, H+12, H+24). */
  echeances: Record<string, number>;
}

export function erreurs(prevision: Float32Array, realite: number[], horizons: number[]): Erreurs {
  let sa = 0;
  let sq = 0;
  for (let k = 0; k < realite.length; k++) {
    const e = prevision[k] - realite[k];
    sa += Math.abs(e);
    sq += e * e;
  }
  const echeances: Record<string, number> = {};
  for (const h of horizons) echeances[`H+${h}`] = Math.abs(prevision[h - 1] - realite[h - 1]);
  return { mae: sa / realite.length, rmse: Math.sqrt(sq / realite.length), echeances };
}
