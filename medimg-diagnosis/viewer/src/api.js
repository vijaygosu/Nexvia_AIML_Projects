const BASE = "/api";

export async function api(path, { user = "", method = "GET", body } = {}) {
  const res = await fetch(BASE + path, {
    method,
    headers: { ...(body ? { "Content-Type": "application/json" } : {}), ...(user ? { "X-User": user } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const imageUrl = (id) => `${BASE}/studies/${id}/image.png`;
export const heatmapUrl = (id, label) =>
  `${BASE}/studies/${id}/heatmap.png${label ? `?label=${encodeURIComponent(label)}` : ""}`;
