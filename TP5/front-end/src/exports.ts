/** Téléchargement des résultats : PNG du graphique, CSV des valeurs, JSON des métriques (F27). */

function telechargerBlob(blob: Blob, nom: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = nom;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function telechargerPng(canvas: HTMLCanvasElement, nom: string): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Export PNG impossible (canvas vide)."));
        return;
      }
      telechargerBlob(blob, nom);
      resolve();
    }, "image/png");
  });
}

/** CSV séparé par des virgules, point décimal, horodatages ISO UTC. */
export function telechargerCsv(colonnes: string[], lignes: (string | number | null)[][], nom: string): void {
  const cellule = (v: string | number | null) => {
    if (v === null) return "";
    if (typeof v === "number") return v.toFixed(4);
    return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  };
  const texte = [colonnes, ...lignes].map((l) => l.map(cellule).join(",")).join("\n") + "\n";
  telechargerBlob(new Blob([texte], { type: "text/csv;charset=utf-8" }), nom);
}

export function telechargerJson(contenu: unknown, nom: string): void {
  telechargerBlob(new Blob([JSON.stringify(contenu, null, 2)], { type: "application/json" }), nom);
}
