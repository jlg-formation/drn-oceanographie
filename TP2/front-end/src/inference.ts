import * as ort from "onnxruntime-web";
import type { Grid } from "./sea-mask";

// Les binaires WASM d'onnxruntime-web sont servis depuis le CDN jsDelivr,
// à la version installée dans node_modules (voir package.json).
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

export interface Normalization {
  lon_min: number;
  lon_max: number;
  lat_min: number;
  lat_max: number;
  temp_mean: number;
  temp_std: number;
  temp_min: number;
  temp_max: number;
}

export interface Metadata {
  variable: string;
  output_unit: string;
  fourier: boolean;
  normalization: Normalization;
  grid: Grid;
  sea_mask: string;
}

/** Réseau de température chargé en ONNX + ses constantes de normalisation. */
export class TemperatureModel {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly meta: Metadata,
  ) {}

  static async load(onnxUrl: string, metaUrl: string): Promise<TemperatureModel> {
    const [session, meta] = await Promise.all([
      ort.InferenceSession.create(onnxUrl, {
        executionProviders: ["wasm"],
      }),
      fetch(metaUrl).then((r) => r.json() as Promise<Metadata>),
    ]);
    return new TemperatureModel(session, meta);
  }

  /**
   * Prédit la température (°C) pour un lot de points (lon, lat).
   * Applique la même normalisation que l'entraînement, puis dé-normalise.
   */
  async predict(lons: Float32Array, lats: Float32Array): Promise<Float32Array> {
    const n = lons.length;
    const { lon_min, lon_max, lat_min, lat_max, temp_mean, temp_std } =
      this.meta.normalization;
    const lonSpan = lon_max - lon_min;
    const latSpan = lat_max - lat_min;

    const input = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      input[i * 2] = (2 * (lons[i] - lon_min)) / lonSpan - 1;
      input[i * 2 + 1] = (2 * (lats[i] - lat_min)) / latSpan - 1;
    }

    const tensor = new ort.Tensor("float32", input, [n, 2]);
    const outputs = await this.session.run({ coords: tensor });
    const normed = outputs.temperature_norm.data as Float32Array;

    const temps = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      temps[i] = normed[i] * temp_std + temp_mean;
    }
    return temps;
  }
}
