/** Trait de côte (polygones de terre) chargé depuis le GeoJSON Natural Earth. */

import type { Feature, Geometry } from "geojson";

type Point = [number, number]; // [lon, lat]
type Ring = Point[];
type Polygon = Ring[]; // anneau 0 = contour, suivants = trous

export interface Emprise {
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
}

function collectPolygons(geometry: Geometry, out: Polygon[]): void {
  if (geometry.type === "Polygon") {
    out.push(geometry.coordinates as Polygon);
  } else if (geometry.type === "MultiPolygon") {
    for (const poly of geometry.coordinates) out.push(poly as Polygon);
  } else if (geometry.type === "GeometryCollection") {
    for (const g of geometry.geometries) collectPolygons(g, out);
  }
}

function pointInRing(lon: number, lat: number, ring: Ring): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersect =
      yi > lat !== yj > lat &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export class Coastline {
  private constructor(private readonly polygons: Polygon[]) {}

  static async load(url: string): Promise<Coastline> {
    const feature = (await fetch(url).then((r) => r.json())) as Feature;
    const polygons: Polygon[] = [];
    collectPolygons(feature.geometry, polygons);
    return new Coastline(polygons);
  }

  /** Trace tous les polygones de terre projetés (pour effacer via destination-out). */
  buildLandPath(
    ctx: CanvasRenderingContext2D,
    project: (lon: number, lat: number) => { x: number; y: number },
  ): void {
    ctx.beginPath();
    for (const poly of this.polygons) {
      for (const ring of poly) {
        for (let i = 0; i < ring.length; i++) {
          const p = project(ring[i][0], ring[i][1]);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
      }
    }
  }

  /** Vrai si le point (lon, lat) tombe sur la terre. */
  isLand(lon: number, lat: number): boolean {
    for (const poly of this.polygons) {
      if (!pointInRing(lon, lat, poly[0])) continue;
      let inHole = false;
      for (let h = 1; h < poly.length; h++) {
        if (pointInRing(lon, lat, poly[h])) {
          inHole = true;
          break;
        }
      }
      if (!inHole) return true;
    }
    return false;
  }
}
