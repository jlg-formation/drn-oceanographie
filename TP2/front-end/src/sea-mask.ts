/** Masque mer/terre chargé depuis le PNG exporté avec le modèle. */

export interface Grid {
  grid_lon_min: number;
  grid_lon_max: number;
  grid_lat_min: number;
  grid_lat_max: number;
  grid_nx: number;
  grid_ny: number;
}

export class SeaMask {
  private constructor(
    private readonly data: Uint8ClampedArray,
    private readonly grid: Grid,
  ) {}

  static async load(url: string, grid: Grid): Promise<SeaMask> {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = grid.grid_nx;
    canvas.height = grid.grid_ny;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0);
    const data = ctx.getImageData(0, 0, grid.grid_nx, grid.grid_ny).data;
    return new SeaMask(data, grid);
  }

  /** Vrai si le point (lon, lat) tombe sur une case de mer de la grille. */
  isSea(lon: number, lat: number): boolean {
    const g = this.grid;
    const fx = (lon - g.grid_lon_min) / (g.grid_lon_max - g.grid_lon_min);
    const fy = (g.grid_lat_max - lat) / (g.grid_lat_max - g.grid_lat_min);
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return false;
    const px = Math.round(fx * (g.grid_nx - 1));
    const py = Math.round(fy * (g.grid_ny - 1));
    return this.data[(py * g.grid_nx + px) * 4] > 127;
  }
}
