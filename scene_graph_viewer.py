#!/usr/bin/env python3
"""
Scene Graph Live Viewer
=======================
Avvia un server HTTP locale che serve una singola pagina interattiva.
La pagina esegue polling su /api/latest ogni secondo e aggiorna il grafo
in-place senza ricaricare, mostrando sempre l'ultimo JSON generato.

Uso:
  python3 scene_graph_viewer.py [--dir <json_dir>] [--port 8765]
"""

import argparse
import glob
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DEFAULT_JSON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "OutputData", "Scene_Graph_only_entities"
)

# ─────────────────────────────────────────────────────────────────────────────
# Stato globale condiviso tra thread
# ─────────────────────────────────────────────────────────────────────────────
_state = {"json_dir": DEFAULT_JSON_DIR}


def find_latest_json(json_dir: str):
    """Ritorna il path del JSON scritto più di recente (ordinato per mtime)."""
    files = glob.glob(os.path.join(json_dir, "scene_graph_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_latest() -> dict | None:
    path = find_latest_json(_state["json_dir"])
    if not path:
        return None
    # Riprova una volta se il file è ancora in scrittura (json.load fallisce)
    for _ in range(2):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = Path(path).name
            data["_mtime"] = os.path.getmtime(path)  # usato dal client per rilevare cambiamenti
            return data
        except (json.JSONDecodeError, OSError):
            import time; time.sleep(0.1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML della pagina (servito una volta sola, poi polling via fetch)
# ─────────────────────────────────────────────────────────────────────────────

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Scene Graph — Live Viewer</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {
      --bg:       #0f0f1a;
      --surface:  #1a1a2e;
      --surface2: #22223a;
      --accent:   #00c9b1;
      --red:      #e74c3c;
      --text:     #e0e0f0;
      --subtext:  #8888aa;
      --border:   #2e2e4e;
      --radius:   12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text);
      font-family: 'Inter', sans-serif;
      height: 100vh; display: flex; flex-direction: column; overflow: hidden;
    }

    /* ── Header ── */
    header {
      display: flex; align-items: center; gap: 14px;
      padding: 14px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .header-icon { font-size: 26px; }
    header h1   { font-size: 20px; font-weight: 700; color: var(--accent); letter-spacing: -0.4px; }
    .live-badge {
      display: flex; align-items: center; gap: 6px;
      background: rgba(0,201,177,0.12); border: 1px solid var(--accent);
      color: var(--accent); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
    }
    .live-dot {
      width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
      animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%       { opacity: 0.4; transform: scale(0.7); }
    }
    .header-meta { margin-left: auto; text-align: right; font-size: 12px; color: var(--subtext); line-height: 1.8; }
    .header-meta b { color: var(--text); }

    /* ── Layout ── */
    .layout { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ── */
    .sidebar {
      width: 250px; min-width: 200px;
      background: var(--surface); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
    }
    .sidebar-header {
      padding: 14px 16px 8px;
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 1.2px; color: var(--subtext);
      border-bottom: 1px solid var(--border);
    }
    .entity-list { flex: 1; overflow-y: auto; padding: 8px; }
    .entity-list::-webkit-scrollbar { width: 3px; }
    .entity-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    .entity-item {
      display: flex; align-items: center; gap: 9px;
      padding: 8px 10px; border-radius: 8px; margin-bottom: 3px;
      cursor: pointer; transition: background 0.15s;
    }
    .entity-item:hover { background: var(--surface2); }
    .entity-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .entity-name { font-size: 13px; font-weight: 500; flex: 1; white-space: nowrap;
                   overflow: hidden; text-overflow: ellipsis; }
    .entity-type { font-size: 10px; color: var(--subtext); white-space: nowrap; }

    .rel-section { padding: 12px 16px; border-top: 1px solid var(--border); }
    .rel-section-title { font-size: 10px; font-weight: 700; text-transform: uppercase;
                         letter-spacing: 1.2px; color: var(--subtext); margin-bottom: 8px; }
    .rel-item { font-size: 11px; color: var(--subtext); padding: 3px 0;
                display: flex; gap: 6px; align-items: center; }
    .rel-item span { color: var(--text); font-weight: 500; }

    /* ── Graph ── */
    .graph-wrapper {
      flex: 1; position: relative; overflow: hidden;
      background: radial-gradient(ellipse at center, #1c1c32 0%, #0f0f1a 80%);
    }
    #graph { width: 100%; height: 100%; }

    /* ── Overlay di update ── */
    #update-flash {
      position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
      background: rgba(0,201,177,0.18); border: 1px solid var(--accent);
      color: var(--accent); padding: 6px 18px; border-radius: 20px;
      font-size: 12px; font-weight: 600; pointer-events: none;
      opacity: 0; transition: opacity 0.3s;
    }

    /* ── Controls ── */
    .controls {
      position: absolute; top: 12px; right: 12px;
      display: flex; gap: 8px;
    }
    .btn {
      background: rgba(26,26,46,0.9); border: 1px solid var(--border);
      color: var(--text); padding: 7px 13px; border-radius: 8px;
      font-size: 12px; cursor: pointer; transition: border-color 0.15s, color 0.15s;
      font-family: 'Inter', sans-serif;
    }
    .btn:hover { border-color: var(--accent); color: var(--accent); }

    /* ── Legend ── */
    .legend {
      position: absolute; bottom: 14px; right: 14px;
      background: rgba(26,26,46,0.9); backdrop-filter: blur(8px);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 12px 16px; font-size: 11px;
    }
    .legend-section { margin-bottom: 9px; }
    .legend-section:last-child { margin-bottom: 0; }
    .legend-title { font-size: 9px; text-transform: uppercase; letter-spacing: 1px;
                    color: var(--subtext); margin-bottom: 5px; font-weight: 700; }
    .leg-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
    .leg-circle { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .leg-bar { width: 22px; height: 3px; border-radius: 2px; flex-shrink: 0; }

    /* ── Footer ── */
    footer {
      background: var(--surface); border-top: 1px solid var(--border);
      padding: 9px 24px; font-size: 11px; color: var(--subtext);
      display: flex; align-items: center; gap: 20px; flex-shrink: 0;
    }
    .badge {
      background: var(--surface2); border: 1px solid var(--border);
      padding: 3px 10px; border-radius: 20px; font-size: 11px;
    }
    .badge b { color: var(--accent); }
    #status-text { margin-left: auto; color: var(--subtext); }
    #status-text.error { color: var(--red); }

    /* ── No-data overlay ── */
    #no-data {
      position: absolute; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 16px;
      color: var(--subtext); text-align: center;
    }
    #no-data .nd-icon { font-size: 52px; opacity: 0.4; }
    #no-data p { font-size: 14px; opacity: 0.7; }

    .vis-tooltip {
      background: var(--surface2) !important; border: 1px solid var(--border) !important;
      color: var(--text) !important; border-radius: 8px !important;
      font-family: 'Inter', sans-serif !important; font-size: 12px !important;
      padding: 10px 14px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
    }
  </style>
</head>
<body>

<header>
  <span class="header-icon">🤖</span>
  <h1>Scene Graph Viewer</h1>
  <div class="live-badge"><div class="live-dot"></div>LIVE</div>
  <div class="header-meta">
    <div>File: <b id="hdr-file">—</b></div>
    <div>Frame: <b id="hdr-frame">—</b> &nbsp;|&nbsp; <b id="hdr-res">—</b></div>
  </div>
</header>

<div class="layout">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">Entities (<span id="entity-count">0</span>)</div>
    <div class="entity-list" id="entityList"></div>
    <div class="rel-section">
      <div class="rel-section-title">Relationships (<span id="rel-count">0</span>)</div>
      <div id="relList"></div>
    </div>
  </div>

  <!-- Grafo -->
  <div class="graph-wrapper">
    <div id="no-data">
      <div class="nd-icon">⏳</div>
      <p>In attesa del primo scene graph...</p>
      <p style="font-size:12px">Avvia il nodo ROS2 per iniziare</p>
    </div>

    <div id="graph"></div>
    <div id="update-flash">✨ Nuovo grafo</div>

    <div class="controls">
      <button class="btn" onclick="fitGraph()">⊡ Fit</button>
      <button class="btn" onclick="relayout()">⟳ Relayout</button>
    </div>

    <div class="legend">
      <div class="legend-section">
        <div class="legend-title">Tipo nodo</div>
        <div class="leg-row"><div class="leg-circle" style="background:#00c9b1"></div>structural</div>
        <div class="leg-row"><div class="leg-circle" style="background:#9b59b6"></div>human</div>
        <div class="leg-row"><div class="leg-circle" style="background:#f39c12"></div>object</div>
      </div>
      <div class="legend-section">
        <div class="legend-title">Relazione</div>
        <div class="leg-row"><div class="leg-bar" style="background:#e74c3c"></div>contact</div>
        <div class="leg-row"><div class="leg-bar" style="background:#3498db"></div>spatial</div>
        <div class="leg-row"><div class="leg-bar" style="background:#2ecc71"></div>interaction</div>
      </div>
    </div>
  </div>

</div>

<footer>
  <span>🖼 Frame <b id="ft-frame" style="color:var(--accent)">—</b></span>
  <span class="badge">Entities: <b id="ft-ent">—</b></span>
  <span class="badge">Relationships: <b id="ft-rel">—</b></span>
  <span id="status-text">Connecting...</span>
</footer>

<script>
// ── Palette ───────────────────────────────────────────────────────────────
const NODE_STYLE = {
  structural: { bg: "#00c9b1", border: "#00857a", font: "#ffffff" },
  human:      { bg: "#9b59b6", border: "#6c3483", font: "#ffffff" },
  object:     { bg: "#f39c12", border: "#b7770d", font: "#1a1a2e" },
  _default:   { bg: "#5dade2", border: "#2980b9", font: "#ffffff" },
};
const EDGE_COLOR = {
  on_top_of:"#e74c3c", inside:"#e74c3c", part_of:"#e74c3c",
  touching:"#e74c3c", not_touching:"#95a5a6", embedded_in:"#e74c3c",
  next_to:"#3498db", near:"#3498db", above:"#3498db", below:"#3498db",
  in_front_of:"#3498db", behind:"#3498db",
  on_the_left_of:"#3498db", on_the_right_of:"#3498db",
  facing:"#3498db", occluding:"#3498db",
  holding:"#2ecc71", held_by:"#2ecc71",
  pointed_by:"#2ecc71", looking_at:"#2ecc71", operating:"#2ecc71",
};

// ── vis.js setup ──────────────────────────────────────────────────────────
const nodesDS = new vis.DataSet([]);
const edgesDS = new vis.DataSet([]);
let network = null;
let currentFrameId = null;

function initNetwork() {
  const container = document.getElementById("graph");
  network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, {
    interaction: { hover: true, tooltipDelay: 150, keyboard: true, zoomView: true },
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -65,
        centralGravity: 0.01,
        springLength: 150,
        springConstant: 0.08,
      },
      stabilization: { iterations: 250, updateInterval: 25 },
    },
    nodes: { borderWidth: 2, shadow: { enabled: true, color: "rgba(0,0,0,0.5)", size: 10, x:3, y:3 } },
    edges: { selectionWidth: 3 },
  });
  network.on("stabilized", () => {
    network.setOptions({ physics: { enabled: false } });
  });
}

function fitGraph() { network && network.fit({ animation: { duration: 500 } }); }
function relayout() {
  if (!network) return;
  network.setOptions({ physics: { enabled: true } });
  setTimeout(() => network.setOptions({ physics: { enabled: false } }), 2500);
}

// ── Aggiornamento grafo ───────────────────────────────────────────────────
function buildVisNodes(entities) {
  return entities.map(e => {
    const s = NODE_STYLE[e.type] || NODE_STYLE._default;
    const states = (e.states || []).join(", ");
    const box = (e.spatial_info?.box_2d || []).join(", ");
    const action = e.action_description || "";
    const tooltip =
      `<b>${e.label}</b><br>` +
      `Type: ${e.type}<br>` +
      `States: ${states || "—"}<br>` +
      `BBox: [${box || "—"}]` +
      (action ? `<br>Action: ${action}` : "");
    return {
      id: e.id,
      label: e.label,
      title: tooltip,
      color: { background: s.bg, border: s.border,
               highlight: { background: s.bg, border: "#ffffff" },
               hover:      { background: s.bg, border: "#ffffff" } },
      font:  { color: s.font, size: 14, face: "Inter, sans-serif" },
      shape: "ellipse", size: 28, borderWidth: 2,
    };
  });
}

function buildVisEdges(relationships) {
  return relationships.map((r, i) => {
    const color = EDGE_COLOR[r.predicate] || "#aaaaaa";
    return {
      id: i,
      from: r.subject_id,
      to:   r.object_id,
      label: r.predicate,
      color: { color, highlight: "#ffffff", hover: "#ffffff" },
      font:  { color, size: 11, face: "Inter, sans-serif",
               strokeWidth: 2, strokeColor: "#1a1a2e" },
      arrows: { to: { enabled: true, scaleFactor: 0.8 } },
      width: 1.5,
      smooth: { type: "curvedCW", roundness: 0.2 },
    };
  });
}

function flashUpdate() {
  const el = document.getElementById("update-flash");
  el.style.opacity = "1";
  setTimeout(() => el.style.opacity = "0", 1200);
}

function updateGraph(data) {
  const sg = data.scene_graph || data;
  const entities = sg.entities || [];
  const rels = sg.relationships || [];
  const isFirst = currentFrameId === null;
  // Usa _mtime come discriminante primario (evita falsi negativi su frame_id riutilizzati)
  const changeKey = String(data._mtime ?? "") + "|" + String(data.frame_id ?? "");
  const isNew = changeKey !== currentFrameId;

  if (!isNew) return;  // nessun cambiamento
  currentFrameId = changeKey;

  // Nascondi no-data
  document.getElementById("no-data").style.display = "none";
  document.getElementById("graph").style.display = "block";

  // Aggiorna dataset vis in-place (evita re-render completo se i nodi coincidono)
  const newNodes = buildVisNodes(entities);
  const newEdges = buildVisEdges(rels);

  nodesDS.clear();
  edgesDS.clear();
  nodesDS.add(newNodes);
  edgesDS.add(newEdges);

  if (isFirst) {
    initNetwork();
  } else {
    // Forza breve stabilizzazione per il nuovo layout
    if (network) {
      network.setOptions({ physics: { enabled: true } });
      setTimeout(() => network.setOptions({ physics: { enabled: false } }), 2000);
      flashUpdate();
    }
  }

  // Header / footer
  document.getElementById("hdr-file").textContent   = data._source_file || "—";
  document.getElementById("hdr-frame").textContent  = data.frame_id ?? "—";
  document.getElementById("hdr-res").textContent    = `${data.width ?? "?"}×${data.height ?? "?"}`;
  document.getElementById("ft-frame").textContent   = data.frame_id ?? "—";
  document.getElementById("ft-ent").textContent     = entities.length;
  document.getElementById("ft-rel").textContent     = rels.length;
  document.getElementById("entity-count").textContent = entities.length;
  document.getElementById("rel-count").textContent    = rels.length;

  // Sidebar lista entità
  const el = document.getElementById("entityList");
  el.innerHTML = entities.map(e => {
    const s = NODE_STYLE[e.type] || NODE_STYLE._default;
    return `<div class="entity-item" onclick="focusNode(${e.id})">
      <span class="entity-dot" style="background:${s.bg}"></span>
      <span class="entity-name">${e.label}</span>
      <span class="entity-type">${e.type}</span>
    </div>`;
  }).join("");

  // Sidebar lista relazioni (max 8 per non overflow)
  const rl = document.getElementById("relList");
  const labelOf = id => (entities.find(e => e.id === id) || {}).label || id;
  rl.innerHTML = rels.slice(0, 8).map(r => {
    const c = EDGE_COLOR[r.predicate] || "#aaaaaa";
    return `<div class="rel-item">
      <span>${labelOf(r.subject_id)}</span>
      <span style="color:${c};font-size:10px">→ ${r.predicate} →</span>
      <span>${labelOf(r.object_id)}</span>
    </div>`;
  }).join("") + (rels.length > 8
    ? `<div style="font-size:10px;color:var(--subtext);padding-top:4px">+ ${rels.length - 8} altre...</div>`
    : "");
}

function focusNode(id) {
  if (!network) return;
  network.focus(id, { scale: 1.5, animation: { duration: 500, easingFunction: "easeInOutQuad" } });
  network.selectNodes([id]);
}

// ── Polling ───────────────────────────────────────────────────────────────
const STATUS = document.getElementById("status-text");
let pollErrors = 0;

async function poll() {
  try {
    const res = await fetch("/api/latest");
    if (res.status === 204) {
      STATUS.textContent = "In attesa di dati...";
      STATUS.className = "";
      pollErrors = 0;
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateGraph(data);
    const now = new Date().toLocaleTimeString();
    STATUS.textContent = `Ultimo aggiornamento: ${now}`;
    STATUS.className = "";
    pollErrors = 0;
  } catch (e) {
    pollErrors++;
    STATUS.textContent = pollErrors > 3 ? `Errore connessione (${e.message})` : STATUS.textContent;
    STATUS.className = "error";
  }
}

// Avvio polling ogni 1 secondo
poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silenzia log HTTP

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/latest":
            self._serve_latest()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_latest(self):
        data = load_latest()
        if data is None:
            # Nessun file trovato → 204 No Content
            self.send_response(204)
            self.end_headers()
            return
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scene Graph Live Viewer — pagina unica con aggiornamento automatico."
    )
    parser.add_argument("--dir", "-d", default=DEFAULT_JSON_DIR, metavar="DIR",
                        help=f"Cartella JSON da monitorare (default: {DEFAULT_JSON_DIR})")
    parser.add_argument("--port", "-p", type=int, default=8765,
                        help="Porta del server HTTP (default: 8765)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Non aprire il browser automaticamente")
    args = parser.parse_args()

    _state["json_dir"] = os.path.abspath(args.dir)

    server = HTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"

    print(f"🤖 Scene Graph Live Viewer")
    print(f"   Cartella JSON : {_state['json_dir']}")
    print(f"   Server        : {url}")
    print(f"   Polling       : ogni 1 secondo")
    print(f"   Premi Ctrl+C per uscire.\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server fermato.")
        server.shutdown()


if __name__ == "__main__":
    main()
