/** Chargement des fichiers produits par TP3/prepare_bathymetry.py. */

import type { FeatureCollection, Point } from "geojson";

export interface Bounds {
  west: number;
  east: number;
  south: number;
  north: number;
}

export interface Palier {
  depth_m: number;
  couleur: string;
}

export interface Metadata {
  bounds: Bounds;
  palette: { paliers: Palier[]; saturation: string };
  raster: { fichier: string; largeur_px: number; hauteur_px: number };
  grille: {
    fichier: string;
    lignes: number;
    colonnes: number;
    lat_premiere_ligne: number;
    lon_premiere_colonne: number;
    pas_lat_deg: number;
    pas_lon_deg: number;
  };
  points: { fichier: string; nombre_points: number };
  source: { attribution: string; doi: string; version: string; avertissement: string };
}

export type DepthPoints = FeatureCollection<Point, { depth_m: number }>;

/** Erreur lisible : quel fichier, pourquoi, et comment le régénérer. */
export class DataError extends Error {
  constructor(fichier: string, raison: string) {
    super(
      `Impossible de charger « ${fichier} » (${raison}). ` +
        "Lancez `mise run tp3_02_prepare` puis rechargez la page.",
    );
  }
}

async function fetchOk(fichier: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`/${fichier}`);
  } catch {
    throw new DataError(fichier, "serveur injoignable");
  }
  if (!res.ok) throw new DataError(fichier, `HTTP ${res.status}`);
  // Le serveur de dev Vite répond index.html (200) pour un fichier inexistant.
  if (res.headers.get("content-type")?.includes("text/html")) {
    throw new DataError(fichier, "fichier absent");
  }
  return res;
}

async function fetchJson<T>(fichier: string): Promise<T> {
  const res = await fetchOk(fichier);
  try {
    return (await res.json()) as T;
  } catch {
    throw new DataError(fichier, "JSON invalide");
  }
}

export async function loadMetadata(): Promise<Metadata> {
  const meta = await fetchJson<Metadata>("metadata.json");
  if (!meta.bounds || !meta.palette?.paliers?.length) {
    throw new DataError("metadata.json", "champs bounds/palette manquants");
  }
  return meta;
}

export async function loadPoints(meta: Metadata): Promise<DepthPoints> {
  const fc = await fetchJson<DepthPoints>(meta.points.fichier);
  if (fc.type !== "FeatureCollection" || !Array.isArray(fc.features)) {
    throw new DataError(meta.points.fichier, "pas une FeatureCollection GeoJSON");
  }
  return fc;
}

/** Grille de profondeurs (F10) : interrogation au clic en mode raster. */
export class DepthGrid {
  private constructor(
    private readonly g: Metadata["grille"],
    private readonly values: Float32Array,
  ) {}

  static async load(meta: Metadata): Promise<DepthGrid> {
    const g = meta.grille;
    const buf = await (await fetchOk(g.fichier)).arrayBuffer();
    if (buf.byteLength !== g.lignes * g.colonnes * 4) {
      throw new DataError(g.fichier, "taille incohérente avec metadata.json");
    }
    return new DepthGrid(g, new Float32Array(buf));
  }

  /** Profondeur (m) de la cellule la plus proche, NaN sur terre, null hors grille. */
  depthAt(lat: number, lon: number): number | null {
    const row = Math.round((this.g.lat_premiere_ligne - lat) / this.g.pas_lat_deg);
    const col = Math.round((lon - this.g.lon_premiere_colonne) / this.g.pas_lon_deg);
    if (row < 0 || row >= this.g.lignes || col < 0 || col >= this.g.colonnes) return null;
    return this.values[row * this.g.colonnes + col];
  }
}
