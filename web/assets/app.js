/* GPS Strain web app: UI, station data, map. The strain maths and the map
   sheets run in Python (worker.js -> gps_strain.web). United States only. */

const QFAULTS = "https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer/21/query";
const BUILD = "__BUILD__";
const DATA_URL = `data/midas_us.NA.json?v=${BUILD}`;
const FONTS = ["../fonts/Gelasio.ttf", "../fonts/ArchivoBlack.ttf"];
const SITE_COLORS = ["#008000", "#0000ff", "#ff0000", "#ffa500"];   // as on the printed sheets
const TRI_COLORS = ["#0000ff", "#c000c0", "#d00000", "#008080", "#000000", "#806000", "#ff6600", "#6000c0"];
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v) => { const x = parseFloat(v); return Number.isFinite(x) ? x : null; };
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));

/* ------------------------------------------------------------------ worker */
const worker = new Worker(`assets/worker.js?v=${BUILD}`);
let seq = 0;
const pending = new Map();
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") { busyText(m.data); return; }
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.ok ? p.resolve(m.result) : p.reject(Object.assign(new Error(m.error), { detail: m.detail }));
};
function rpc(op, args = []) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
}
const pycall = (fn, ...args) => rpc("call", [fn, ...args]);
const pyReady = (async () => {
  const man = await fetch("py/manifest.json", { cache: "no-store" }).then((r) => r.json());
  return rpc("init", ["../py/" + man.wheel, FONTS]);
})();
pyReady.catch((e) => setStatus("The calculator could not start: " + e.message, "err"));
window.addEventListener("unhandledrejection", (e) => { console.error(e.reason); setStatus(String(e.reason?.message || e.reason), "err"); busy(false); });

/* ------------------------------------------------------------------- state */
const S = {
  stations: new Map(),   // site -> record (mm/yr)
  dataInfo: null,
  current: [],           // stations of the triangle being chosen (up to 3)
  triangles: [],         // kept triangles (arrays of 3 records)
  results: null,         // last analyze() output
  manual: false,
};

/* ---------------------------------------------------------------------- UI */
function setStatus(msg, kind = "") { const el = $("#status"); el.textContent = msg || ""; el.className = "status " + kind; }
function busy(on, text = "") { $("#busy").hidden = !on; busyText(text); }
function busyText(t) { if (t) $("#busy-text").textContent = t; }
const frame = () => $('input[name="frame"]:checked').value;
const glyph = () => $('input[name="glyph"]:checked').value;

$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => { x.classList.toggle("is-on", x === t); x.setAttribute("aria-selected", x === t); });
  $$(".tabpane").forEach((p) => { p.hidden = p.dataset.pane !== t.dataset.tab; });
}));
$$("[data-open]").forEach((b) => b.addEventListener("click", () => $("#dlg-" + b.dataset.open).showModal()));
$("#flow").addEventListener("submit", (e) => { e.preventDefault(); runAll(); });
$$('input[name="frame"], input[name="glyph"]').forEach((r) => r.addEventListener("change", () => { if (S.results) runAll(); }));

/* US coverage (USGS 3DEP and Quaternary faults): CONUS, Alaska, Hawaii, Puerto Rico */
function inUS(lon, lat) {
  return (lon >= -125.5 && lon <= -66 && lat >= 24 && lat <= 50) || (lon >= -170 && lon <= -129 && lat >= 51 && lat <= 72)
    || (lon >= -161 && lon <= -154 && lat >= 18 && lat <= 23) || (lon >= -68 && lon <= -65 && lat >= 17.5 && lat <= 18.7);
}

/* -------------------------------------------------------------------- data */
const dataReady = (async () => {
  const d = await fetch(DATA_URL).then((r) => { if (!r.ok) throw new Error(`station list (HTTP ${r.status})`); return r.json(); });
  const f = d.fields;
  for (const row of d.rows) {
    const rec = Object.fromEntries(f.map((k, i) => [k, row[i]]));
    S.stations.set(rec.site, rec);
  }
  S.dataInfo = d;
  $("#data-note").textContent = `Velocities: NGL MIDAS, North America fixed, ${d.count.toLocaleString()} US stations, updated ${d.generated}.`;
  // the map may still be loading (it waits while the tab is in the background): don't hold the data up
  setData("all", { type: "FeatureCollection", features: d.rows.map((r) => ({ type: "Feature", properties: { site: r[0] }, geometry: { type: "Point", coordinates: [r[2], r[1]] } })) });
  return d;
})();
dataReady.catch((e) => setStatus("Could not load the station list: " + e.message, "err"));

/* -------------------------------------------------------------------- map */
const BASEMAPS = {
  topo: { tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"], max: 16, attr: "USGS The National Map" },
  relief: { tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}"], max: 13, attr: "USGS The National Map" },
  imagery: { tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"], max: 16, attr: "USGS The National Map" },
  osm: { tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], max: 19, attr: "© OpenStreetMap contributors" },
};
const style = { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#c0c0c0" } }] };
for (const [k, b] of Object.entries(BASEMAPS)) {
  style.sources["bm-" + k] = { type: "raster", tiles: b.tiles, tileSize: 256, maxzoom: b.max, attribution: b.attr };
  style.layers.push({ id: "bm-" + k, type: "raster", source: "bm-" + k, layout: { visibility: k === "topo" ? "visible" : "none" } });
}
const map = new maplibregl.Map({
  container: "map", style, center: [-98.5, 39], zoom: 3.4, preserveDrawingBuffer: true,
  attributionControl: { compact: true }, maxBounds: [[-180, 10], [-40, 75]],
});
window.toolkitMap = map;
map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
$("#basemap").addEventListener("change", (e) => {
  for (const k of Object.keys(BASEMAPS)) map.setLayoutProperty("bm-" + k, "visibility", k === e.target.value ? "visible" : "none");
});

const mapReady = new Promise((res) => map.on("load", res));
const EMPTY = { type: "FeatureCollection", features: [] };
const triColor = ["to-color", ["at", ["%", ["get", "tri"], TRI_COLORS.length], ["literal", TRI_COLORS]]];
const L = (layer) => ["==", ["get", "layer"], layer];

mapReady.then(() => {
  for (const id of ["faults", "all", "overlay", "sel"]) map.addSource(id, { type: "geojson", data: EMPTY });
  map.addLayer({ id: "faults", type: "line", source: "faults", paint: { "line-color": "#e6d43a", "line-width": 1.8 } });
  map.addLayer({ id: "all", type: "circle", source: "all", minzoom: 5, paint: {
    "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 2, 9, 4.5], "circle-color": "#ffffff",
    "circle-stroke-color": "#404040", "circle-stroke-width": 1 } });
  map.addLayer({ id: "circle", type: "fill", source: "overlay", filter: L("circle"), paint: { "fill-color": "#ffffff", "fill-opacity": 0.45 } });
  map.addLayer({ id: "circle-line", type: "line", source: "overlay", filter: L("circle_line"), paint: { "line-color": "#708090", "line-width": 1.5, "line-dasharray": [3, 2] } });
  map.addLayer({ id: "triangle-edge", type: "line", source: "overlay", filter: L("triangle"), paint: { "line-color": "#ffffff", "line-width": 5, "line-opacity": 0.8 } });
  map.addLayer({ id: "triangle", type: "line", source: "overlay", filter: L("triangle"), paint: { "line-color": triColor, "line-width": 2.5, "line-dasharray": [3, 2] } });
  map.addLayer({ id: "axis-s1", type: "line", source: "overlay", filter: L("axis_s1"), layout: { "line-cap": "butt" }, paint: { "line-color": "#ff6347", "line-width": 5 } });
  map.addLayer({ id: "axis-s2", type: "line", source: "overlay", filter: L("axis_s2"), layout: { "line-cap": "butt" }, paint: { "line-color": "#6495ed", "line-width": 5 } });
  map.addLayer({ id: "ellipse", type: "line", source: "overlay", filter: L("ellipse"), paint: { "line-color": "#000000", "line-width": 2 } });
  map.addLayer({ id: "elongation", type: "fill", source: "overlay", filter: L("elongation"), paint: { "fill-color": "#ffffff" } });
  map.addLayer({ id: "elongation-line", type: "line", source: "overlay", filter: L("elongation"), paint: { "line-color": "#000000", "line-width": 2 } });
  map.addLayer({ id: "contraction", type: "fill", source: "overlay", filter: L("contraction"), paint: { "fill-color": "#000000" } });
  map.addLayer({ id: "velocity-edge", type: "line", source: "overlay", filter: L("velocity"), paint: { "line-color": "#ffffff", "line-width": 6 } });
  map.addLayer({ id: "velocity", type: "line", source: "overlay", filter: L("velocity"), paint: { "line-color": "#000000", "line-width": 3 } });
  map.addLayer({ id: "velocity-head", type: "fill", source: "overlay", filter: L("velocity_head"), paint: { "fill-color": "#000000", "fill-outline-color": "#ffffff" } });
  map.addLayer({ id: "sel", type: "circle", source: "sel", paint: {
    "circle-radius": 7, "circle-color": ["to-color", ["at", ["min", ["get", "order"], 3], ["literal", SITE_COLORS]]],
    "circle-stroke-color": "#000000", "circle-stroke-width": 1.5 } });
  map.on("click", (e) => {   // one handler, so a chosen station (two layers) toggles once
    const f = map.queryRenderedFeatures(e.point, { layers: ["sel", "all"] })[0];
    if (f) pickStation(f.properties.site);
  });
  for (const id of ["all", "sel"]) {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
  map.on("click", "faults", (e) => {
    if (map.queryRenderedFeatures(e.point, { layers: ["all", "sel"] }).length) return;
    const p = e.features[0].properties;
    new maplibregl.Popup().setLngLat(e.lngLat).setHTML(`<strong>${esc(p.fault_name || "Quaternary fault")}</strong>${p.section_name ? "<br>" + esc(p.section_name) : ""}<br>${esc(p.age || "")}${p.slip_sense ? ", " + esc(p.slip_sense) : ""}`).addTo(map);
  });
  $("#show-faults").addEventListener("change", (e) => map.setLayoutProperty("faults", "visibility", e.target.checked ? "visible" : "none"));
  $("#show-all").addEventListener("change", (e) => map.setLayoutProperty("all", "visibility", e.target.checked ? "visible" : "none"));
});
dataReady.then(readHash).catch(() => {});
/* map layers are added once the map has loaded; data sent earlier is applied then */
function setData(id, data) { mapReady.then(() => map.getSource(id)?.setData(data)); }

/* labels for the chosen stations (DOM markers: no glyph server needed) */
let labelMarkers = [];
function drawSelection() {
  const all = [...S.triangles.flat(), ...S.current];
  const seen = new Map();
  for (const s of all) if (!seen.has(s.site)) seen.set(s.site, s);
  const cur = new Set(S.current.map((s) => s.site));
  const feats = [...seen.values()].map((s) => {
    const i = S.current.findIndex((c) => c.site === s.site);
    return { type: "Feature", properties: { site: s.site, order: i >= 0 ? i : 3 }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } };
  });
  setData("sel", { type: "FeatureCollection", features: feats });
  labelMarkers.forEach((m) => m.remove());
  labelMarkers = [...seen.values()].map((s) => {
    const el = document.createElement("div");
    el.className = "sta-label";
    el.textContent = s.site;
    el.style.cssText = `font:700 11px var(--ui);background:var(--tip);border:1px solid #000;padding:0 3px;pointer-events:none;${cur.has(s.site) ? "" : "opacity:.8"}`;
    return new maplibregl.Marker({ element: el, anchor: "bottom-left", offset: [6, -6] }).setLngLat([s.lon, s.lat]).addTo(map);
  });
}

/* ---------------------------------------------------------------- stations */
function haversineKm(a, b) {
  const R = 6371.0088, r = Math.PI / 180;
  const dLat = (b.lat - a.lat) * r, dLon = (b.lon - a.lon) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
function triangleCheck(t) {
  if (t.length < 3) return [];
  const [a, b, c] = t;
  const A = haversineKm(b, c), B = haversineKm(a, c), C = haversineKm(a, b);
  const ang = (x, y, z) => Math.acos(Math.max(-1, Math.min(1, (y * y + z * z - x * x) / (2 * y * z)))) * 180 / Math.PI;
  const angles = [ang(A, B, C), ang(B, A, C), ang(C, A, B)];
  const out = [];
  if (Math.min(...angles) < 30) out.push(`Smallest angle ${Math.min(...angles).toFixed(0)}° (under 30°): a skinny triangle gives unreliable strain.`);
  if (Math.max(A, B, C) > 100) out.push(`Longest side ${Math.max(A, B, C).toFixed(0)} km (over 100 km, the GETSI Unit 6 limit).`);
  return out;
}

function renderStations() {
  const list = $("#sta-list");
  list.innerHTML = S.current.map((s, i) => {
    const short = s.years != null && s.years < 2.5;
    return `<li><i class="dot c${i}"></i><span class="site">${esc(s.site)}</span>
      <button type="button" class="linkish" data-rm="${esc(s.site)}">remove</button>
      <span class="meta">${fmt(s.lat, 4)}°, ${fmt(s.lon, 4)}° · E ${fmt(s.ve)} ± ${fmt(s.sig_e)}, N ${fmt(s.vn)} ± ${fmt(s.sig_n)} mm/yr${s.years != null ? ` · ${fmt(s.years, 1)} yr record` : ""}</span>
      ${short ? `<span class="warn">Short record: velocity less reliable.</span>` : ""}</li>`;
  }).join("");
  const warn = triangleCheck(S.current);
  if (warn.length) list.insertAdjacentHTML("beforeend", warn.map((w) => `<li><span></span><span class="warn" style="grid-column:2/4">${esc(w)}</span></li>`).join(""));
  $$("[data-rm]", list).forEach((b) => b.addEventListener("click", () => { S.current = S.current.filter((s) => s.site !== b.dataset.rm); afterSelection(); }));
  $("#step-sta").classList.toggle("is-done", S.current.length === 3);
  $("#run").disabled = !(S.current.length === 3 || (S.current.length === 0 && S.triangles.length));
  ["id1", "id2", "id3"].forEach((id, i) => { if (!S.manual) $("#" + id).value = S.current[i]?.site || ""; });
}
function afterSelection() { renderStations(); drawSelection(); }

function lookup(site) {
  const s = S.stations.get(site.trim().toUpperCase());
  if (!s) throw new Error(`Station ${site.trim().toUpperCase()} is not in the US station list (NGL MIDAS, this week).`);
  return { ...s };
}
function pickStation(site) {
  S.manual = false;
  const i = S.current.findIndex((s) => s.site === site);
  if (i >= 0) S.current.splice(i, 1);
  else {
    if (S.current.length >= 3) S.current = [];
    S.current.push(lookup(site));
  }
  setStatus(S.current.length < 3 ? `${3 - S.current.length} more station${S.current.length === 2 ? "" : "s"} to pick.` : "Three stations chosen. Press Compute the strain.");
  afterSelection();
}
async function useIds(ids) {
  await dataReady;
  try {
    S.manual = false;
    S.current = ids.map(lookup);
    if (new Set(S.current.map((s) => s.site)).size < 3) throw new Error("Choose three different stations.");
    afterSelection();
    fitTo(S.current);
    setStatus("");
    return true;
  } catch (e) { setStatus(e.message, "err"); return false; }
}
$("#use-ids").addEventListener("click", () => useIds(["#id1", "#id2", "#id3"].map((s) => $(s).value)).then((ok) => ok && runAll()));
["#id1", "#id2", "#id3"].forEach((s) => $(s).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("#use-ids").click(); } }));
$$(".example").forEach((b) => b.addEventListener("click", () => useIds(b.dataset.ids.split(",")).then((ok) => ok && runAll())));
$("#pick-clear").addEventListener("click", () => { S.current = []; afterSelection(); });

function fitTo(stations) {
  if (!stations.length) return;
  const lons = stations.map((s) => s.lon), lats = stations.map((s) => s.lat);
  mapReady.then(() => map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: 60, maxZoom: 10, duration: 600 }));
}

/* manual entry */
const MF = ["site", "lat", "lon", "ve", "vn", "sig_e", "sig_n"];
(function buildManual() {
  const g = $("#manual-grid");
  for (let r = 0; r < 3; r++) for (const k of MF) {
    const inp = document.createElement("input");
    inp.id = `m-${k}-${r}`;
    if (k === "site") { inp.maxLength = 8; inp.placeholder = ["P146", "P149", "P150"][r]; } else { inp.type = "number"; inp.step = "any"; }
    inp.setAttribute("aria-label", `${k} ${r + 1}`);
    g.appendChild(inp);
  }
})();
function manualRecords() {
  const recs = [];
  for (let r = 0; r < 3; r++) {
    const rec = Object.fromEntries(MF.map((k) => [k, k === "site" ? $(`#m-${k}-${r}`).value.trim().toUpperCase() : num($(`#m-${k}-${r}`).value)]));
    if (!rec.site) rec.site = `S${r + 1}`;
    if (MF.slice(1).some((k) => rec[k] == null)) throw new Error(`Fill in every value for station ${r + 1}.`);
    if (!(rec.sig_e > 0 && rec.sig_n > 0)) throw new Error("Uncertainties must be greater than zero.");
    if (!inUS(rec.lon, rec.lat)) throw new Error(`${rec.site} is outside the United States; the site covers US stations only for now.`);
    rec.years = null; rec.manual = true; recs.push(rec);
  }
  return recs;
}
$("#use-manual").addEventListener("click", () => {
  try { S.manual = true; S.current = manualRecords(); afterSelection(); fitTo(S.current); setStatus(""); runAll(); }
  catch (e) { setStatus(e.message, "err"); }
});
$("#csv-file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const lines = (await f.text()).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const head = lines[0].toLowerCase().split(/[,;\t]/).map((h) => h.trim());
  const alias = { station: "site", latitude: "lat", longitude: "lon", long: "lon", lng: "lon", vel_e: "ve", vel_n: "vn", sigma_e: "sig_e", sigma_n: "sig_n", se: "sig_e", sn: "sig_n" };
  const idx = Object.fromEntries(head.map((h, i) => [alias[h] || h, i]));
  const missing = MF.filter((k) => idx[k] == null);
  if (missing.length) { setStatus(`The CSV needs columns ${MF.join(", ")} (missing ${missing.join(", ")}).`, "err"); return; }
  lines.slice(1, 4).forEach((l, r) => { const c = l.split(/[,;\t]/); for (const k of MF) $(`#m-${k}-${r}`).value = c[idx[k]].trim(); });
  setStatus(`Loaded ${Math.min(3, lines.length - 1)} stations from ${f.name}. Check them, then press Use these values.`, "ok");
});

/* --------------------------------------------------------------- analysis */
const payload = (t) => t.map(({ site, lat, lon, ve, vn, sig_e, sig_n }) => ({ site, lat, lon, ve, vn, sig_e, sig_n }));
function allTriangles() { return S.current.length === 3 ? [...S.triangles, S.current] : [...S.triangles]; }

async function runAll() {
  const tris = allTriangles();
  if (!tris.length) { setStatus("Choose three stations first.", "err"); return; }
  const t0 = performance.now();
  busy(true, "Starting the calculator");
  setStatus("");
  try {
    await pyReady;
    busyText("Computing the strain");
    const out = JSON.parse(await pycall("analyze", JSON.stringify({ triangles: tris.map(payload), frame: frame(), glyph: glyph() })));
    if (!out.ok) throw new Error(out.error);
    S.results = out; S.resultTris = tris;
    setData("overlay", out.geojson);
    renderResults();
    $("#results").hidden = false;
    $$(".step").forEach((s) => s.classList.add("is-done"));
    const [w, s, e, n] = out.geojson.bbox;
    mapReady.then(() => map.fitBounds([[w, s], [e, n]], { padding: { top: 50, bottom: 40, left: 40, right: 40 }, maxZoom: 11, duration: 600 }));
    writeHash();
    setStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`, "ok");
    loadFaults([w, s, e, n]);
  } catch (err) {
    console.error(err.detail || err);
    setStatus(err.message, "err");
  } finally { busy(false); }
}

async function loadFaults([w, s, e, n]) {
  const pad = Math.max(0.4, (e - w) * 0.6, (n - s) * 0.6);
  const p = new URLSearchParams({ geometry: `${w - pad},${s - pad},${e + pad},${n + pad}`, geometryType: "esriGeometryEnvelope", inSR: "4326",
    spatialRel: "esriSpatialRelIntersects", outFields: "fault_name,section_name,age,slip_sense", outSR: "4326", f: "geojson", resultRecordCount: "2000" });
  try { setData("faults", await fetch(`${QFAULTS}?${p}`).then((r) => r.json())); } catch { /* map works without faults */ }
}

const rows = (t) => {
  const sg = t.sigma, pm = (v, e, d = 2) => `${fmt(v, d)}${e != null ? ` ± ${fmt(e, d)}` : ""}`;
  const az = (a) => `${fmt(a, 1)}° (or ${fmt((a + 180) % 360, 1)}°)`;
  return [
    ["Greatest horizontal lengthening, e1H", pm(t.e1, sg.e1), "nanostrain/yr"],
    ["Least horizontal lengthening, e2H", pm(t.e2, sg.e2), "nanostrain/yr"],
    ["Azimuth of e1H (S1H)", `${az(t.azimuth_s1)}${sg.azimuth != null ? ` ± ${fmt(sg.azimuth, 1)}°` : ""}`, ""],
    ["Azimuth of e2H (S2H)", az(t.azimuth_s2), ""],
    ["Maximum shear strain rate", pm(t.max_shear, sg.max_shear), "nanostrain/yr"],
    ["Area strain rate", pm(t.area_strain, sg.area_strain), "nanostrain/yr"],
    ["Rotation", `${pm(t.rotation, sg.rotation)} (${t.rotation_sense})`, "nanoradian/yr"],
    ["Translation", `${fmt(t.speed_mm)} toward ${fmt(t.translation_azimuth, 1)}°`, "mm/yr"],
  ];
};
function renderResults() {
  const tris = S.results.triangles;
  $("#tri-results").innerHTML = tris.map((t, i) => `
    <section class="plane-res">
      <h3><span style="display:none"></span><i class="tri-swatch" style="background:${TRI_COLORS[i % TRI_COLORS.length]}"></i>Triangle ${LETTERS[i]}: ${t.sites.map(esc).join("–")}</h3>
      <table class="res-table"><tbody>${rows(t).map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</tbody></table>
      ${t.warnings.length ? `<ul class="res-warn">${t.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      <details><summary>Full report</summary><pre class="res-report">${esc(t.report)}</pre></details>
    </section>`).join("");
  renderTriList();
}
function renderTriList() {
  const list = $("#tri-list");
  list.innerHTML = S.triangles.map((t, i) => `<li><span class="letter"><i class="tri-swatch" style="background:${TRI_COLORS[i % TRI_COLORS.length]}"></i>${LETTERS[i]}</span>
    <span class="title">${t.map((s) => esc(s.site)).join("–")}</span><button type="button" class="rm" data-tri="${i}" aria-label="Remove triangle ${LETTERS[i]}">×</button></li>`).join("");
  $$(".rm", list).forEach((b) => b.addEventListener("click", () => { S.triangles.splice(+b.dataset.tri, 1); drawSelection(); renderTriList(); if (allTriangles().length) runAll(); }));
}
$("#tri-add").addEventListener("click", () => {
  if (S.current.length !== 3) { setStatus("Compute a triangle first, then add another.", "err"); return; }
  S.triangles.push(S.current); S.current = []; S.manual = false;
  afterSelection(); renderTriList();
  $$(".tab").find((t) => t.dataset.tab === "pick").click();
  $("#panel").scrollTo({ top: 0, behavior: "smooth" });
  setStatus(`Triangle ${LETTERS[S.triangles.length - 1]} kept. Choose three more stations (tap them on the map or type their IDs).`, "ok");
});

/* ----------------------------------------------------------------- exports */
function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const baseName = () => "gps_strain_" + (S.resultTris || []).map((t) => t.map((s) => s.site).join("-")).join("_");
const request = (extra = {}) => JSON.stringify({ triangles: (S.resultTris || []).map(payload), frame: frame(), glyph: glyph(), ...extra });
$("#dl-csv").addEventListener("click", async () => download(baseName() + ".csv", new Blob([await pycall("results_csv", request())], { type: "text/csv" })));
$("#dl-geojson").addEventListener("click", () => download(baseName() + ".geojson", new Blob([JSON.stringify(S.results.geojson)], { type: "application/geo+json" })));
$("#dl-png").addEventListener("click", () => map.getCanvas().toBlob((b) => b && download(baseName() + "_map.png", b)));
$("#dl-txt").addEventListener("click", () => {
  const head = `GPS Strain (gps-strain.org). Method: Cronin & Resor (2021) triangle strain.\nVelocities: ${S.manual ? "entered by the user" : `NGL MIDAS, North America fixed, updated ${S.dataInfo?.generated}`}.\n${S.dataInfo?.citation || ""}\n\n`;
  download(baseName() + "_summary.txt", new Blob([head + S.results.triangles.map((t, i) => `Triangle ${LETTERS[i]}\n${t.report}`).join("\n\n")], { type: "text/plain" }));
});
$("#share").addEventListener("click", async () => {
  writeHash();
  try { await navigator.clipboard.writeText(location.href); setStatus("Share link copied. Anyone who opens it recomputes these triangles.", "ok"); }
  catch { setStatus("Copy the link from the address bar to share these triangles.", "ok"); }
});

/* ------------------------------------------------------------------ report */
let previewTimer = null, previewUrl = null;
function setRepMsg(t) { $("#rep-msg").textContent = t || ""; }
function reportRequest(fmtv, dpi, preview) {
  const sheet = $("#r-sheet").value;
  const tris = S.resultTris.map(payload);
  const pick = +($("#r-tri").value || tris.length - 1);
  const light = $("#r-light").value;
  return request({
    triangles: sheet.startsWith("network") ? tris : [tris[pick]], sheet, fmt: fmtv, dpi,
    title: $("#r-title").value.trim(), basemap: $("#r-basemap").value, greyscale: $("#r-grey").checked,
    visibility: $("#r-vis").checked, faults: $("#r-faults").checked,
    light_azimuth: light === "custom" ? num($("#r-az").value) ?? 315 : +light,
    sun_altitude: +$("#r-alt").value, exaggeration: +$("#r-ex").value,
    terrain_px: preview ? 900 : +$("#r-detail").value,
    data_label: S.resultTris.flat().some((s) => s.manual) ? "Station" : "NGL MIDAS",
  });
}
async function renderSheet(fmtv, dpi, preview) {
  await pyReady;
  setRepMsg(preview ? "Drawing the preview (terrain and faults download the first time)…" : "Drawing the sheet…");
  return rpc("bytes", ["report", reportRequest(fmtv, dpi, preview)]);
}
function schedulePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(updatePreview, 400); }
async function updatePreview() {
  const box = $(".rep-preview");
  box.classList.add("loading");
  try {
    const bytes = await renderSheet("png", matchMedia("(min-width: 861px)").matches ? 60 : 40, true);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
    $("#rep-img").src = previewUrl;
    setRepMsg("");
  } catch (e) { console.error(e.detail || e); setRepMsg(e.message); } finally { box.classList.remove("loading"); }
}
function syncReportForm() {
  const n = S.resultTris.length;
  $$("#r-sheet option").forEach((o) => { if (o.value.startsWith("network")) o.disabled = n < 2; });
  if (n < 2 && $("#r-sheet").value.startsWith("network")) $("#r-sheet").value = "strain";
  $("#r-tri").innerHTML = S.resultTris.map((t, i) => `<option value="${i}">${LETTERS[i]}: ${t.map((s) => esc(s.site)).join("–")}</option>`).join("");
  $("#r-tri").value = String(n - 1);
  $("#r-tri-row").hidden = n < 2 || $("#r-sheet").value.startsWith("network");
  const tiles = $("#r-basemap").value.startsWith("usgs");
  $("#r-grey").disabled = !tiles;
  ["#r-light", "#r-alt", "#r-ex"].forEach((s) => { $(s).disabled = tiles || $("#r-basemap").value === "none"; });
  $("#r-az").disabled = $("#r-light").disabled || $("#r-light").value !== "custom";
}
$("#rep-open").addEventListener("click", () => {
  if (!S.results) return;
  $("#r-sheet").value = $("#r-sheet").value || "strain";
  syncReportForm();
  $("#dlg-report").showModal();
  updatePreview();
});
$("#rep-form").addEventListener("input", (e) => {
  if (e.target.id === "r-alt") $("#r-alt-o").textContent = e.target.value + "°";
  if (e.target.id === "r-ex") $("#r-ex-o").textContent = e.target.value + "×";
  const keepTri = $("#r-tri").value;
  syncReportForm();
  if (keepTri) $("#r-tri").value = keepTri;
  if (["r-fmt", "r-detail"].includes(e.target.id)) return;
  schedulePreview();
});
$("#r-dl").addEventListener("click", async (ev) => {
  const btn = ev.target;
  btn.disabled = true;
  try {
    const [f, dpi] = $("#r-fmt").value.split(":");
    const bytes = await renderSheet(f, +(dpi || 150), false);
    const type = { png: "image/png", pdf: "application/pdf", svg: "image/svg+xml" }[f];
    download(`${baseName()}_${$("#r-sheet").value}.${f}`, new Blob([bytes], { type }));
    setRepMsg("Sheet downloaded.");
  } catch (e) { console.error(e.detail || e); setRepMsg(e.message); } finally { btn.disabled = false; }
});

/* ------------------------------------------------------------- share links */
function writeHash() {
  const tris = S.resultTris || allTriangles();
  if (!tris.length) return;
  const p = new URLSearchParams();
  const enc = (t) => (t.some((s) => s.manual)
    ? "m:" + t.map((s) => [s.site, s.lat, s.lon, s.ve, s.vn, s.sig_e, s.sig_n].join(":")).join("|")
    : t.map((s) => s.site).join(","));
  p.set("t", tris.map(enc).join(";"));
  if (frame() !== "enu") p.set("f", frame());
  if (glyph() !== "ellipse") p.set("g", glyph());
  history.replaceState(null, "", "#" + p.toString());
}
async function readHash() {
  if (location.hash.length < 3) return;
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("f")) $(`input[name="frame"][value="${p.get("f")}"]`).checked = true;
  if (p.get("g")) $(`input[name="glyph"][value="${p.get("g")}"]`).checked = true;
  try {
    const tris = (p.get("t") || "").split(";").filter(Boolean).map((tok) => {
      if (!tok.startsWith("m:")) return tok.split(",").map(lookup);
      S.manual = true;
      return tok.slice(2).split("|").map((x) => {
        const [site, ...v] = x.split(":");
        const [lat, lon, ve, vn, sig_e, sig_n] = v.map(Number);
        if (!inUS(lon, lat)) throw new Error(`${site} is outside the United States.`);
        return { site, lat, lon, ve, vn, sig_e, sig_n, years: null, manual: true };
      });
    });
    if (!tris.length) return;
    S.current = tris.pop();
    S.triangles = tris;
    afterSelection();
    setStatus("Loading a shared map…");
    await runAll();
  } catch (e) { setStatus(e.message, "err"); }
}
