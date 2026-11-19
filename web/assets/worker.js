/* Runs the gps_strain Python package in a Web Worker via Pyodide.
   Python downloads (USGS 3DEP terrain, Quaternary faults, USGS tiles for the
   map sheets) go through synchronous XMLHttpRequest, which workers allow. */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;

async function init(wheelUrl, fontUrls) {
  py = await loadPyodide({ indexURL: PYODIDE });
  post("progress", "Loading numerical libraries");
  await py.loadPackage(["numpy", "pandas", "matplotlib", "pillow", "micropip"]);
  post("progress", "Installing GPS Strain");
  const micropip = py.pyimport("micropip");
  await micropip.install.callKwargs(new URL(wheelUrl, self.location.href).href, { deps: false });
  py.runPython("from gps_strain import web; web.use_browser_http()");
  for (const u of fontUrls || []) {
    try { py.globals.get("web").add_font(new URL(u, self.location.href).href); } catch (e) { /* fall back to DejaVu */ }
  }
  return py.runPython("import gps_strain; gps_strain.__version__");
}

function post(type, data) { self.postMessage({ type, data }); }

const handlers = {
  init: ([wheelUrl, fontUrls]) => init(wheelUrl, fontUrls),
  // gps_strain.web.<fn>(*args) returning a str (JSON)
  call: ([fn, ...args]) => py.pyimport("gps_strain.web")[fn](...args),
  // gps_strain.web.<fn>(*args) returning bytes -> Uint8Array
  bytes: ([fn, ...args]) => {
    const r = py.pyimport("gps_strain.web")[fn](...args);
    const out = r.toJs();
    r.destroy();
    return out;
  },
};

self.onmessage = async (ev) => {
  const { id, op, args } = ev.data;
  try {
    const result = await handlers[op](args || []);
    const transfer = ArrayBuffer.isView(result) ? [result.buffer] : [];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    const lines = msg.trim().split("\n");
    self.postMessage({ id, ok: false, error: lines[lines.length - 1], detail: msg });
  }
};
