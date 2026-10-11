/**
 * Inférence des modèles génératifs dans le navigateur (F28, NF05) et graines reproductibles.
 *
 * Les graphes ONNX contiennent la normalisation : le décodeur du VAE et le
 * générateur du GAN rendent directement un relief en mètres, l'encodeur reçoit
 * des élévations en mètres. Une génération 64 × 64 prend quelques millisecondes
 * en WebAssembly : on peut régénérer à chaque mouvement de curseur.
 */
import * as ort from "onnxruntime-web/wasm";

import { N_PIXELS, TAILLE, type Metadata, type Modele } from "./donnees";

// Les binaires WASM d'onnxruntime-web sont servis depuis le CDN jsDelivr,
// à la version installée dans node_modules (voir package.json).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

// --------------------------------------------------------------------------
// Graines : mêmes vecteurs latents qu'en Python (terrain.vecteur_latent)
// --------------------------------------------------------------------------

/** Générateur pseudo-aléatoire 32 bits mulberry32. */
export function mulberry32(graine: number): () => number {
  let a = graine >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Vecteur latent N(0, I) reproductible (Box-Muller sur mulberry32). */
export function vecteurLatent(graine: number, dim: number): Float32Array {
  const u = mulberry32(graine);
  const z = new Float32Array(dim);
  for (let i = 0; i < dim; i += 2) {
    const u1 = 1 - u();
    const u2 = u();
    const rayon = Math.sqrt(-2 * Math.log(u1));
    z[i] = rayon * Math.cos(2 * Math.PI * u2);
    if (i + 1 < dim) z[i + 1] = rayon * Math.sin(2 * Math.PI * u2);
  }
  return z;
}

export function graineAuHasard(): number {
  return Math.floor(Math.random() * 1_000_000);
}

// --------------------------------------------------------------------------
// Sessions ONNX
// --------------------------------------------------------------------------

export interface Sessions {
  vae_encodeur?: ort.InferenceSession;
  vae_decodeur?: ort.InferenceSession;
  gan_generateur?: ort.InferenceSession;
}

export interface Chargement {
  sessions: Sessions;
  erreurs: string[];
  /** Écart maximal (m) entre le navigateur et PyTorch sur les exemples de contrôle (F26). */
  ecartControle: number | null;
}

const OPTIONS: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"] };

export async function chargerModeles(meta: Metadata): Promise<Chargement> {
  const res: Chargement = { sessions: {}, erreurs: [], ecartControle: null };
  const fichiers: [keyof Sessions, string | undefined][] = [
    ["vae_encodeur", meta.modeles.vae?.fichiers.encodeur],
    ["vae_decodeur", meta.modeles.vae?.fichiers.decodeur],
    ["gan_generateur", meta.modeles.gan?.fichiers.generateur],
  ];
  await Promise.all(
    fichiers.map(async ([cle, fichier]) => {
      if (!fichier) {
        res.erreurs.push(`${cle} : absent des métadonnées (modèle non entraîné ou non exporté)`);
        return;
      }
      try {
        res.sessions[cle] = await ort.InferenceSession.create(`./${fichier}`, OPTIONS);
      } catch (err) {
        res.erreurs.push(`${fichier} : ${err instanceof Error ? err.message : String(err)}`);
      }
    }),
  );

  // Contrôle de cohérence : on recalcule les exemples de référence exportés par Python.
  let ecart = 0;
  let controles = 0;
  for (const m of ["vae", "gan"] as Modele[]) {
    const c = meta.controle[m];
    const session = sessionGeneration(res.sessions, m);
    if (!c || !session) continue;
    const relief = (await generer(session, [Float32Array.from(c.z)]))[0];
    c.relief_echantillon.forEach((v, k) => (ecart = Math.max(ecart, Math.abs(v - relief[k * 97]))));
    controles++;
  }
  res.ecartControle = controles ? ecart : null;
  return res;
}

export function sessionGeneration(s: Sessions, m: Modele): ort.InferenceSession | undefined {
  return m === "vae" ? s.vae_decodeur : s.gan_generateur;
}

/** Reliefs relatifs (m, 64 × 64) pour un lot de vecteurs latents. */
export async function generer(session: ort.InferenceSession, z: Float32Array[]): Promise<Float32Array[]> {
  const dim = z[0].length;
  const entree = new Float32Array(z.length * dim);
  z.forEach((v, i) => entree.set(v, i * dim));
  const tenseur = new ort.Tensor("float32", entree, [z.length, dim]);
  const sorties = await session.run({ z: tenseur });
  const relief = (await sorties.relief.getData()) as Float32Array;
  tenseur.dispose();
  return z.map((_, i) => Float32Array.from(relief.subarray(i * N_PIXELS, (i + 1) * N_PIXELS)));
}

/** Encodeur du VAE : élévation (m) -> moyenne μ et log-variance de la gaussienne latente. */
export async function encoder(
  session: ort.InferenceSession,
  elevation: Float32Array,
): Promise<{ mu: Float32Array; logvar: Float32Array }> {
  const tenseur = new ort.Tensor("float32", elevation, [1, TAILLE, TAILLE]);
  const sorties = await session.run({ elevation: tenseur });
  tenseur.dispose();
  return {
    mu: Float32Array.from((await sorties.mu.getData()) as Float32Array),
    logvar: Float32Array.from((await sorties.logvar.getData()) as Float32Array),
  };
}

/** Interpolation linéaire entre deux vecteurs latents. */
export function interpoler(a: Float32Array, b: Float32Array, t: number): Float32Array {
  return Float32Array.from(a, (v, i) => (1 - t) * v + t * b[i]);
}
