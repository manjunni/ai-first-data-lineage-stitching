#!/usr/bin/env python3
"""
AI-First Business Service Architecture — Service Architecture Renderer

Generates Draw.io diagrams from the canonical service domain model.
Each diagram shows ONE service level: services, AI Personas, actors,
external interactions, and flows at that level.

Usage:
    python service_architecture_renderer.py <model.json> <parent_id> <level> [output.drawio]
    python service_architecture_renderer.py <model.json> --l0 [output.drawio]

Examples:
    python service_architecture_renderer.py model.json mfg 1 manufacturing-l1.drawio
    python service_architecture_renderer.py model.json --l0 enterprise-l0.drawio
"""

import json
import math
import sys
import os
import shutil
import subprocess
from collections import deque


# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------

def load_style_config(config_path=None):
    """Load style config from JSON, searching the standard path if not given."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check ../design/ first (toolkit deployment: .bsa/diagram-tools/ → .bsa/design/)
    # then ../../design/ (source repo: distribution/diagram-tools/ → design/)
    for rel in (('..', 'design'), ('..', '..', 'design')):
        candidate = os.path.join(script_dir, *rel, 'style-config.json')
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError("style-config.json not found. Expected at design/style-config.json.")


# ---------------------------------------------------------------------------
# Layout Constants  — adjust these to shift diagram geometry
# ---------------------------------------------------------------------------

FONT_FAMILY = "Inter, 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif"

# Entity dimensions (pixels)
AI_W,  AI_H  = 225, 130   # AI Persona box width / height
SVC_W, SVC_H = 195,  95   # Business service box width / height
EXT_W, EXT_H = 160,  56   # External catalog box width / height
ACTOR_D       = 80         # Actor collision diameter for Graphviz

# Grid layout — used when services have visualMetadata.layout row/col hints
GRID_MARGIN_LEFT = 130     # Left margin before column 0
GRID_NUM_COLS    = 5       # Number of grid columns
GRID_COL_W       = 300     # Column pitch (box width + gap)
GRID_ROW_CY      = {0: 100, 1: 300, 2: 700}  # Vertical centre per row index
GRID_HUMAN_CY    = 550     # Y-centre for the human actor band
GRID_SYS_CY      = 880     # Y-centre for the system actor band
GRID_EXT_CY      = 1000    # Y-centre for external catalog boxes

# Manual fallback layout constants
MANUAL_MARGIN     = 60
MANUAL_TITLE_H    = 65
MANUAL_GRID_GAP_X = 100
MANUAL_GRID_GAP_Y = 80
MANUAL_SAT_DIST   = 70     # Gap from service edge to satellite actor centre


# ---------------------------------------------------------------------------
# Model Filtering  (port of canonical-model-renderer-filtering.js)
# ---------------------------------------------------------------------------

def filter_service_architecture_view(model, parent_service_id, target_level):
    """Return everything needed for a single drill-down service architecture view."""
    parent = next((s for s in model['services'] if s['id'] == parent_service_id), None)
    if not parent:
        raise ValueError(f"Parent service '{parent_service_id}' not found.")

    children = [s for s in model['services']
                if s.get('parentId') == parent_service_id and s.get('level') == target_level]
    child_ids = {s['id'] for s in children}

    ai_personas = [ap for ap in model.get('aiPersonas', []) if ap.get('serviceId') in child_ids]

    visible = set(child_ids)
    humans  = [a for a in model.get('humanActors', [])
               if any(x.get('serviceId') in child_ids for x in a.get('serviceAssociations', []))]
    systems = [a for a in model.get('systemActors', [])
               if any(x.get('serviceId') in child_ids for x in a.get('serviceAssociations', []))]
    for a in humans + systems:
        visible.add(a['id'])

    ext_interactions = [e for e in model.get('externalInteractions', [])
                        if e.get('internalServiceId') in child_ids]
    interactions = [i for i in model.get('interactions', [])
                    if i.get('sourceId') in visible and i.get('targetId') in visible]

    return {
        'viewType':    'service_architecture',
        'title':       f"{parent['name']} \u2014 Level {target_level} Services",
        'breadcrumb':  _breadcrumb(model, parent) + f' > Level {target_level}',
        'parentService': parent,
        'services':    children,
        'aiPersonas':  ai_personas,
        'humanActors': humans,
        'systemActors': systems,
        'externalInteractions': ext_interactions,
        'interactions': interactions,
    }


def filter_level0_view(model):
    """Return the enterprise L0 view with all associated entities."""
    l0 = [s for s in model['services'] if s.get('level') == 0]
    l0_ids = {s['id'] for s in l0}

    ai_personas = [ap for ap in model.get('aiPersonas', []) if ap.get('serviceId') in l0_ids]
    humans  = [a for a in model.get('humanActors', [])
               if any(x.get('serviceId') in l0_ids for x in a.get('serviceAssociations', []))]
    systems = [a for a in model.get('systemActors', [])
               if any(x.get('serviceId') in l0_ids for x in a.get('serviceAssociations', []))]
    ext_interactions = [e for e in model.get('externalInteractions', [])
                        if e.get('internalServiceId') in l0_ids]

    visible = l0_ids | {a['id'] for a in humans} | {a['id'] for a in systems}
    interactions = [i for i in model.get('interactions', [])
                    if i.get('sourceId') in visible and i.get('targetId') in visible]

    org = model.get('modelMetadata', {}).get('organisation', 'Enterprise')
    return {
        'viewType':   'service_architecture_l0',
        'title':      f"{org} \u2014 Enterprise Architecture (Level 0)",
        'breadcrumb': 'Enterprise',
        'services':   l0,
        'aiPersonas': ai_personas,
        'humanActors': humans,
        'systemActors': systems,
        'externalInteractions': ext_interactions,
        'interactions': interactions,
    }


def _breadcrumb(model, service):
    crumbs = [service['name']]
    cur = service
    while cur.get('parentId'):
        cur = next((s for s in model['services'] if s['id'] == cur['parentId']), None)
        if cur:
            crumbs.insert(0, cur['name'])
    crumbs.insert(0, 'Enterprise')
    return ' > '.join(crumbs)


# ---------------------------------------------------------------------------
# Text Helpers
# ---------------------------------------------------------------------------

def esc(text):
    """Escape text for XML attribute values."""
    return (str(text)
            .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;').replace("'", '&apos;'))


def text_width(text, font_size=12, bold=False):
    return len(text) * font_size * (0.62 if bold else 0.57)


def wrap_text(text, max_width, font_size=11):
    if not text:
        return []
    words = text.split()
    lines, current = [], ''
    for word in words:
        test = f'{current} {word}'.strip()
        if text_width(test, font_size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


# ---------------------------------------------------------------------------
# Graph Analysis
# ---------------------------------------------------------------------------

def build_service_graph(view):
    ids = {s['id'] for s in view['services']}
    fwd = {i: [] for i in ids}
    rev = {i: [] for i in ids}
    for inter in view.get('interactions', []):
        s, t = inter['sourceId'], inter['targetId']
        if s in ids and t in ids and s != t:
            fwd[s].append(t)
            rev[t].append(s)
    return fwd, rev


def assign_columns(view):
    """BFS column assignment — first visit wins to handle cycles."""
    fwd, rev = build_service_graph(view)
    ids = {s['id'] for s in view['services']}
    roots = sorted([i for i in ids if not rev[i]])
    if not roots:
        name = {s['id']: s['name'] for s in view['services']}
        roots = [max(ids, key=lambda i: (len(fwd.get(i, [])) - len(rev.get(i, [])), name.get(i, '')))]

    cols  = {}
    queue = deque((r, 0) for r in roots)
    while queue:
        node, col = queue.popleft()
        if node in cols:
            continue
        cols[node] = col
        for tgt in fwd.get(node, []):
            if tgt not in cols:
                queue.append((tgt, col + 1))
    for i in ids:
        if i not in cols:
            cols[i] = 0
    return cols


# ---------------------------------------------------------------------------
# Layout Engine — Grid  (when visualMetadata.layout hints are present)
# ---------------------------------------------------------------------------

def has_layout_hints(view):
    return any(
        s.get('visualMetadata', {}).get('layout', {}).get('row') is not None
        for s in view.get('services', [])
    )


def compute_layout_grid(view, style):
    """Grid layout driven by visualMetadata.layout.row/col on each service.

    Rows:
      0 — Orchestrator (centred at top)
      1 — Main workflow agents (left → right)
      2 — Infrastructure / data services

    Bands below the service grid (in Y order):
      Human actors  — GRID_HUMAN_CY, sorted and spread evenly across diagram width
      System actors — GRID_SYS_CY,   anchored to their service column
      External boxes — GRID_EXT_CY,  spread evenly across diagram width
    """
    positions = {}

    def col_cx(col):
        return GRID_MARGIN_LEFT + col * GRID_COL_W + GRID_COL_W / 2

    ai_ids = {s['id'] for s in view['services'] if s.get('isAIPersona')}

    # ── 1. Services ───────────────────────────────────────────────────────────
    for svc in view['services']:
        layout = svc.get('visualMetadata', {}).get('layout', {})
        if 'row' not in layout or 'col' not in layout:
            continue
        row, col = layout['row'], layout['col']
        is_ai = svc['id'] in ai_ids
        w, h  = (AI_W, AI_H) if is_ai else (SVC_W, SVC_H)
        cx    = col_cx(col)
        cy    = GRID_ROW_CY.get(row, GRID_ROW_CY[2] + (row - 2) * 180)
        positions[svc['id']] = {'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h, 'cx': cx, 'cy': cy}

    # Services without hints overflow below the last defined row
    for i, svc in enumerate(s for s in view['services'] if s['id'] not in positions):
        is_ai = svc['id'] in ai_ids
        w, h  = (AI_W, AI_H) if is_ai else (SVC_W, SVC_H)
        cx    = col_cx(i % GRID_NUM_COLS)
        cy    = GRID_ROW_CY[2] + 200
        positions[svc['id']] = {'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h, 'cx': cx, 'cy': cy}

    # ── 2. System actors — anchored to their primary service column ───────────
    svc_col = {
        s['id']: s['visualMetadata']['layout']['col']
        for s in view['services']
        if 'col' in s.get('visualMetadata', {}).get('layout', {})
    }

    def system_primary_col(actor):
        weights = {}
        for assoc in actor.get('serviceAssociations', []):
            sid = assoc['serviceId']
            if sid in svc_col:
                w = 3 if assoc.get('associationType') in ('integration_target', 'data_source') else 1
                weights[svc_col[sid]] = weights.get(svc_col[sid], 0) + w
        return max(weights, key=weights.get) if weights else GRID_NUM_COLS // 2

    col_sys = {}
    for actor in view.get('systemActors', []):
        col_sys.setdefault(system_primary_col(actor), []).append(actor)

    for col, actors in col_sys.items():
        n = len(actors)
        for i, actor in enumerate(actors):
            offset = (i - (n - 1) / 2) * 90
            positions[actor['id']] = {'cx': col_cx(col) + offset, 'cy': GRID_SYS_CY, 'r': 30}

    # ── 3. External boxes — spread evenly across the bottom ───────────────────
    exts = view.get('externalInteractions', [])
    if exts:
        n      = len(exts)
        total  = n * EXT_W + (n - 1) * 20
        canvas = GRID_NUM_COLS * GRID_COL_W
        start  = GRID_MARGIN_LEFT + (canvas - total) / 2
        for i, ext in enumerate(exts):
            ex = start + i * (EXT_W + 20)
            ey = GRID_EXT_CY - EXT_H / 2
            positions[ext['id']] = {'x': ex, 'y': ey, 'w': EXT_W, 'h': EXT_H,
                                    'cx': ex + EXT_W / 2, 'cy': GRID_EXT_CY}

    # ── 4. Human actors — evenly spaced across diagram width ─────────────────
    # Actors are sorted by their weighted column centre so placement tracks
    # which part of the workflow each actor belongs to.
    humans = view.get('humanActors', [])
    if humans:
        all_ints = view.get('interactions', [])

        def human_col_centre(actor):
            weights = {}
            for assoc in actor.get('serviceAssociations', []):
                sid = assoc['serviceId']
                if sid in svc_col:
                    w = 3 if assoc.get('associationType') in ('owner', 'consumer') else 1
                    weights[svc_col[sid]] = weights.get(svc_col[sid], 0) + w
            for inter in all_ints:
                other = None
                if inter['sourceId'] == actor['id'] and inter['targetId'] in svc_col:
                    other = inter['targetId']
                elif inter['targetId'] == actor['id'] and inter['sourceId'] in svc_col:
                    other = inter['sourceId']
                if other:
                    weights[svc_col[other]] = weights.get(svc_col[other], 0) + 2
            if not weights:
                return GRID_NUM_COLS / 2
            total_w = sum(weights.values())
            return sum(c * w for c, w in weights.items()) / total_w

        sorted_humans = sorted(humans, key=human_col_centre)

        left_x  = GRID_MARGIN_LEFT - 55
        right_x = GRID_MARGIN_LEFT + GRID_NUM_COLS * GRID_COL_W + 55

        n = len(sorted_humans)
        if n == 1:
            xs = [(left_x + right_x) / 2]
        elif n == 2:
            xs = [left_x, right_x]
        else:
            step = (right_x - left_x) / (n - 1)
            xs = [left_x + i * step for i in range(n)]

        for actor, cx in zip(sorted_humans, xs):
            positions[actor['id']] = {'cx': cx, 'cy': GRID_HUMAN_CY, 'r': 30}

    return positions


# ---------------------------------------------------------------------------
# Layout Engine — Manual (fallback when no hints and no Graphviz)
# ---------------------------------------------------------------------------

def compute_layout_manual(view, style):
    """Fallback: services in a centred grid, actors orbit their primary service."""
    positions = {}
    services  = view['services']
    n = len(services)
    if n == 0:
        return positions

    ai_ids  = {s['id'] for s in services if s.get('isAIPersona')}
    svc_ids = {s['id'] for s in services}

    has_ai   = bool(ai_ids)
    max_ew   = AI_W  if has_ai else SVC_W
    max_eh   = AI_H  if has_ai else SVC_H
    cell_w   = max_ew + MANUAL_GRID_GAP_X
    cell_h   = max_eh + MANUAL_GRID_GAP_Y
    grid_cols = max(1, min(n, math.ceil(math.sqrt(n * 1.5))))
    col_map   = assign_columns(view)
    ordered   = sorted(services, key=lambda s: (
        1 if s.get('serviceType') == 'support' else 0,
        col_map.get(s['id'], 0),
        s['name'],
    ))
    actor_zone = MANUAL_SAT_DIST + 70
    grid_left  = MANUAL_MARGIN + actor_zone
    grid_top   = MANUAL_MARGIN + MANUAL_TITLE_H + actor_zone

    for idx, svc in enumerate(ordered):
        row   = idx // grid_cols
        col   = idx % grid_cols
        n_row = min(grid_cols, n - row * grid_cols)
        offset = (grid_cols - n_row) * cell_w / 2
        is_ai  = svc['id'] in ai_ids
        w, h   = (AI_W, AI_H) if is_ai else (SVC_W, SVC_H)
        cx = grid_left + offset + col * cell_w + max_ew / 2
        cy = grid_top  + row * cell_h + max_eh / 2
        positions[svc['id']] = {'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h, 'cx': cx, 'cy': cy}

    svc_pos = {k: v for k, v in positions.items() if k in svc_ids}
    gb_cx   = sum(p['cx'] for p in svc_pos.values()) / len(svc_pos)
    gb_cy   = sum(p['cy'] for p in svc_pos.values()) / len(svc_pos)

    all_ints   = view.get('interactions', [])
    all_actors = list(view.get('humanActors', [])) + list(view.get('systemActors', []))

    def primary_service(actor):
        weights = {}
        for assoc in actor.get('serviceAssociations', []):
            sid = assoc['serviceId']
            if sid not in svc_ids:
                continue
            weights[sid] = weights.get(sid, 0) + (10 if assoc.get('associationType') == 'owner' else 1)
        for inter in all_ints:
            if inter['sourceId'] == actor['id'] and inter['targetId'] in svc_ids:
                weights[inter['targetId']] = weights.get(inter['targetId'], 0) + 2
            elif inter['targetId'] == actor['id'] and inter['sourceId'] in svc_ids:
                weights[inter['sourceId']] = weights.get(inter['sourceId'], 0) + 2
        return max(weights, key=weights.get) if weights else None

    svc_actors = {}
    unassigned = []
    for actor in all_actors:
        ps = primary_service(actor)
        if ps: svc_actors.setdefault(ps, []).append(actor)
        else:  unassigned.append(actor)

    placed = []
    for sid, actors in svc_actors.items():
        sp  = positions[sid]
        scx, scy = sp['cx'], sp['cy']
        sw, sh   = sp['w'],  sp['h']
        dx, dy   = scx - gb_cx, scy - gb_cy
        base_ang = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1 else -math.pi / 2
        na  = len(actors)
        fan = min(math.pi * 0.4, max(math.pi / 6, (na - 1) * math.pi / 5))
        for i, actor in enumerate(actors):
            t   = (i / (na - 1) - 0.5) if na > 1 else 0
            ang = base_ang + t * fan
            ca, sa = abs(math.cos(ang)), abs(math.sin(ang))
            if ca < 0.001:   edge_d = sh / 2
            elif sa < 0.001: edge_d = sw / 2
            else:            edge_d = min(sw / (2 * ca), sh / (2 * sa))
            dist   = edge_d + MANUAL_SAT_DIST
            ax, ay = scx + dist * math.cos(ang), scy + dist * math.sin(ang)
            for px, py in placed:
                if math.sqrt((ax - px) ** 2 + (ay - py) ** 2) < 80:
                    ax += 45 * math.cos(ang); ay += 45 * math.sin(ang)
            placed.append((ax, ay))
            positions[actor['id']] = {'cx': ax, 'cy': ay, 'r': 30}

    if unassigned:
        g_top = min(p['y'] for p in svc_pos.values())
        for i, actor in enumerate(unassigned):
            ax = gb_cx + (i - (len(unassigned) - 1) / 2) * 120
            positions[actor['id']] = {'cx': ax, 'cy': g_top - MANUAL_SAT_DIST, 'r': 30}

    exts = view.get('externalInteractions', [])
    if exts:
        g_right = max(p['x'] + p['w'] for p in svc_pos.values())
        g_top   = min(p['y'] for p in svc_pos.values())
        ext_x   = g_right + 100
        for i, ext in enumerate(exts):
            ey = g_top + 20 + i * (EXT_H + 30)
            positions[ext['id']] = {'x': ext_x, 'y': ey, 'w': EXT_W, 'h': EXT_H,
                                    'cx': ext_x + EXT_W / 2, 'cy': ey + EXT_H / 2}
    return positions


# ---------------------------------------------------------------------------
# Layout Engine — Graphviz  (primary when no hints)
# ---------------------------------------------------------------------------

def graphviz_available():
    return shutil.which('neato') is not None


def build_dot_graph(view, style):
    PT = 72
    ai_ids  = {s['id'] for s in view['services'] if s.get('isAIPersona')}
    svc_ids = {s['id'] for s in view['services']}
    lines = [
        'digraph G {',
        '  graph [layout=neato, overlap=prism, overlap_scaling=1.5, splines=curved, sep="+25", esep="+10"];',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]
    for svc in view['services']:
        is_ai = svc['id'] in ai_ids
        w = (AI_W if is_ai else SVC_W) / PT
        h = (AI_H if is_ai else SVC_H) / PT
        lines.append(f'  "{svc["id"]}" [width={w:.4f}, height={h:.4f}, fixedsize=true, shape=box];')

    d = ACTOR_D / PT
    for actor in list(view.get('humanActors', [])) + list(view.get('systemActors', [])):
        lines.append(f'  "{actor["id"]}" [width={d:.4f}, height={d:.4f}, fixedsize=true, shape=circle];')

    for ext in view.get('externalInteractions', []):
        lines.append(f'  "{ext["id"]}" [width={EXT_W/PT:.4f}, height={EXT_H/PT:.4f}, fixedsize=true, shape=box];')

    for inter in view.get('interactions', []):
        src, tgt = inter['sourceId'], inter['targetId']
        lbl = inter.get('label', '').replace('"', '\\"')
        s2s = src in svc_ids and tgt in svc_ids
        lines.append(f'  "{src}" -> "{tgt}" [label="{lbl}", len={"3.5" if s2s else "2.0"}, weight={1 if s2s else 2}];')

    all_actors = list(view.get('humanActors', [])) + list(view.get('systemActors', []))
    for actor in all_actors:
        connected = set()
        for inter in view.get('interactions', []):
            if inter['sourceId'] == actor['id'] and inter['targetId'] in svc_ids:
                connected.add(inter['targetId'])
            elif inter['targetId'] == actor['id'] and inter['sourceId'] in svc_ids:
                connected.add(inter['sourceId'])
        for assoc in actor.get('serviceAssociations', []):
            sid = assoc['serviceId']
            if sid in svc_ids and sid not in connected:
                w = 4 if assoc.get('associationType') == 'owner' else 2
                lines.append(f'  "{actor["id"]}" -> "{sid}" [style=invis, len=1.8, weight={w}];')
        if not connected and not any(a['serviceId'] in svc_ids for a in actor.get('serviceAssociations', [])):
            if view['services']:
                lines.append(f'  "{actor["id"]}" -> "{view["services"][0]["id"]}" [style=invis, len=4.0, weight=1];')

    lines.append('}')
    return '\n'.join(lines)


def run_graphviz(dot_string, engine='neato'):
    result = subprocess.run([engine, '-Tjson'], input=dot_string,
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Graphviz error: {result.stderr.strip()}")
    return json.loads(result.stdout)


def parse_edge_spline(pos_str, tx_fn, ty_fn):
    tokens = pos_str.strip().split()
    end_arrow = start_arrow = None
    pts = []
    for tok in tokens:
        if tok.startswith('e,'):
            c = tok[2:].split(','); end_arrow = (float(c[0]), float(c[1]))
        elif tok.startswith('s,'):
            c = tok[2:].split(','); start_arrow = (float(c[0]), float(c[1]))
        else:
            c = tok.split(',')
            if len(c) >= 2:
                try: pts.append((float(c[0]), float(c[1])))
                except ValueError: pass
    if not pts:
        return None
    sx, sy = start_arrow or pts[0]
    path = f'M {tx_fn(sx):.1f},{ty_fn(sy):.1f}'
    if len(pts) < 4:
        ex, ey = end_arrow or pts[-1]
        return path + f' L {tx_fn(ex):.1f},{ty_fn(ey):.1f}'
    i = 1
    while i + 2 <= len(pts):
        cp1, cp2 = pts[i], pts[i + 1]
        ep = end_arrow if (i + 3 > len(pts)) else (pts[i + 2] if i + 2 < len(pts) else end_arrow or pts[-1])
        path += (f' C {tx_fn(cp1[0]):.1f},{ty_fn(cp1[1]):.1f}'
                 f' {tx_fn(cp2[0]):.1f},{ty_fn(cp2[1]):.1f}'
                 f' {tx_fn(ep[0]):.1f},{ty_fn(ep[1]):.1f}')
        i += 3
    return path


def compute_layout_graphviz(view, style):
    margin, title_h = 60, 65
    dot = build_dot_graph(view, style)
    gv  = run_graphviz(dot)
    bb  = [float(v) for v in gv.get('bb', '0,0,800,600').split(',')]
    bb_llx, bb_lly, bb_urx, bb_ury = bb

    def tx(x): return x - bb_llx + margin
    def ty(y): return (bb_ury - y) + margin + title_h

    ai_ids  = {s['id'] for s in view['services'] if s.get('isAIPersona')}
    svc_ids = {s['id'] for s in view['services']}
    obj_by_gvid = {o.get('_gvid'): o for o in gv.get('objects', []) if o.get('_gvid') is not None}
    positions = {}

    for obj in gv.get('objects', []):
        name = obj.get('name', '')
        if not name or 'pos' not in obj: continue
        p  = obj['pos'].split(',')
        cx, cy = tx(float(p[0])), ty(float(p[1]))
        if name in svc_ids:
            is_ai = name in ai_ids
            w, h = (AI_W, AI_H) if is_ai else (SVC_W, SVC_H)
            positions[name] = {'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h, 'cx': cx, 'cy': cy}
        elif any(a['id'] == name for a in view.get('humanActors', []) + view.get('systemActors', [])):
            positions[name] = {'cx': cx, 'cy': cy, 'r': 30}
        elif any(e['id'] == name for e in view.get('externalInteractions', [])):
            positions[name] = {'x': cx - EXT_W/2, 'y': cy - EXT_H/2, 'w': EXT_W, 'h': EXT_H, 'cx': cx, 'cy': cy}

    return positions, {}


def compute_layout(view, style):
    """Choose the appropriate layout engine."""
    if has_layout_hints(view):
        return compute_layout_grid(view, style), {}
    if graphviz_available():
        try:
            return compute_layout_graphviz(view, style)
        except Exception as e:
            print(f"Warning: Graphviz failed ({e}). Using manual layout.", file=sys.stderr)
    else:
        print("Note: Install graphviz for better diagram layout.", file=sys.stderr)
    return compute_layout_manual(view, style), {}


def compute_canvas_size(positions, margin=60):
    if not positions:
        return 800, 600
    max_x = max_y = 0
    for p in positions.values():
        if 'x' in p:
            max_x = max(max_x, p['x'] + p['w'])
            max_y = max(max_y, p['y'] + p['h'])
        elif 'cx' in p:
            r = p.get('r', 30)
            max_x = max(max_x, p['cx'] + r + 60)
            max_y = max(max_y, p['cy'] + r + 40)
    return max(800, int(max_x + margin + 60)), max(500, int(max_y + margin + 60))


# ---------------------------------------------------------------------------
# Draw.io Cell Styles
# ---------------------------------------------------------------------------

# Edge styles per interaction type
_EDGE_STYLES = {
    'capability_invocation': 'edgeStyle=curved;html=1;strokeColor=#2C5282;strokeWidth=1.5;endArrow=block;endFill=1;',
    'data_flow':             'edgeStyle=curved;html=1;strokeColor=#059669;strokeWidth=1.5;endArrow=block;endFill=1;',
    'escalation':            'edgeStyle=curved;html=1;strokeColor=#DC2626;strokeWidth=1.5;dashed=1;endArrow=block;endFill=1;',
    'support':               'edgeStyle=curved;html=1;strokeColor=#15803D;strokeWidth=1.2;dashed=1;endArrow=open;endFill=0;',
    'external':              'edgeStyle=curved;html=1;strokeColor=#94A3B8;strokeWidth=1.2;dashed=1;endArrow=open;endFill=0;',
    'approval':              'edgeStyle=curved;html=1;strokeColor=#2C5282;strokeWidth=1.5;endArrow=block;endFill=1;',
}


# ---------------------------------------------------------------------------
# Draw.io Cell Renderers
# ---------------------------------------------------------------------------

def _cell(cell_id, value, style, x, y, w, h, parent='1', vertex='1'):
    """Helper: produce a single <mxCell> element with geometry."""
    return (f'<mxCell id="{esc(cell_id)}" value="{esc(value)}" style="{style}" '
            f'vertex="{vertex}" parent="{esc(parent)}">'
            f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" as="geometry"/>'
            f'</mxCell>')


def _edge_cell(cell_id, value, style, source, target):
    """Helper: produce a single <mxCell> edge element."""
    return (f'<mxCell id="{esc(cell_id)}" value="{esc(value)}" style="{style}" '
            f'edge="1" source="{esc(source)}" target="{esc(target)}" parent="1">'
            f'<mxGeometry relative="1" as="geometry"/>'
            f'</mxCell>')


def service_cells(sid, x, y, w, h, service):
    """Three cells: outer box (edge anchor) + blue header + mission text."""
    name    = service['name']
    mission = service.get('mission', '')
    out = []
    # Outer box — this is what edges attach to
    out.append(_cell(sid, '', 'rounded=1;arcSize=5;fillColor=#EBF3FB;strokeColor=#2C5282;strokeWidth=1.5;', x, y, w, h))
    # Blue header strip (child of outer box, positioned inside it)
    out.append(_cell(f'{sid}--hdr', name,
                     'rounded=1;arcSize=5;fillColor=#2C5282;strokeColor=none;'
                     'fontColor=#FFFFFF;fontStyle=1;fontSize=13;align=center;verticalAlign=middle;',
                     0, 0, w, 30, parent=sid))
    # Mission text
    if mission:
        out.append(_cell(f'{sid}--txt', mission,
                         'text;strokeColor=none;fillColor=none;align=left;verticalAlign=top;'
                         'fontSize=10;fontColor=#475569;whiteSpace=wrap;',
                         4, 32, w - 8, h - 36, parent=sid))
    return '\n'.join(out)


def ai_persona_cells(sid, x, y, w, h, service, persona):
    """Four cells: outer box + purple header + AI PERSONA label + authority + mission."""
    name    = service['name']
    mission = service.get('mission', '')
    auth    = ''
    if persona and persona.get('authorityModel'):
        auth = f'Authority: {persona["authorityModel"].upper()}'
    out = []
    # Outer box
    out.append(_cell(sid, '', 'rounded=1;arcSize=5;fillColor=#F3E8FF;strokeColor=#6B21A8;strokeWidth=2;', x, y, w, h))
    # Purple header
    out.append(_cell(f'{sid}--hdr', f'\U0001f916 {name}',
                     'rounded=1;arcSize=5;fillColor=#6B21A8;strokeColor=none;'
                     'fontColor=#FFFFFF;fontStyle=1;fontSize=13;align=center;verticalAlign=middle;',
                     0, 0, w, 34, parent=sid))
    # AI PERSONA label
    out.append(_cell(f'{sid}--lbl', 'AI PERSONA',
                     'text;strokeColor=none;fillColor=none;align=center;verticalAlign=top;'
                     'fontSize=9;fontColor=#7C3AED;fontStyle=1;',
                     0, 36, w, 14, parent=sid))
    body_y = 50
    # Authority badge
    if auth:
        out.append(_cell(f'{sid}--auth', auth,
                         'text;strokeColor=none;fillColor=none;align=center;verticalAlign=top;'
                         'fontSize=8;fontColor=#64748B;',
                         0, body_y, w, 14, parent=sid))
        body_y += 14
    # Mission text
    if mission:
        out.append(_cell(f'{sid}--txt', mission,
                         'text;strokeColor=none;fillColor=none;align=left;verticalAlign=top;'
                         'fontSize=9;fontColor=#475569;whiteSpace=wrap;',
                         4, body_y, w - 8, h - body_y - 4, parent=sid))
    return '\n'.join(out)


def human_actor_cell(aid, cx, cy, actor):
    """UML actor (stick figure) shape for a human actor."""
    w, h  = 30, 60
    label = actor['name'] + (' (EXT)' if actor.get('isExternal') else '')
    return _cell(aid, label,
                 'shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;'
                 'html=1;outlineConnect=0;fillColor=#FEF3C7;strokeColor=#92400E;'
                 'fontStyle=1;fontSize=11;fontColor=#1A202C;',
                 cx - w/2, cy - h/2, w, h)


def system_actor_cell(aid, cx, cy, actor):
    """Cylinder shape for a data store / system actor."""
    w, h = 60, 120
    return _cell(aid, actor['name'],
                 'shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;'
                 'fillColor=#E0F2FE;strokeColor=#0C4A6E;strokeWidth=1.5;'
                 'fontSize=10;fontColor=#1A202C;verticalAlign=middle;',
                 cx - w/2, cy - h/2, w, h)


def external_service_cell(eid, x, y, w, h, ext):
    """Dashed-border cell for an external catalog target."""
    name   = ext['externalServiceName']
    domain = ext.get('externalDomain', '')
    label  = f'{name}\n{domain}' if domain else name
    return _cell(eid, label,
                 'rounded=1;arcSize=5;whiteSpace=wrap;dashed=1;'
                 'fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.5;'
                 'fontSize=11;fontColor=#475569;fontStyle=1;verticalAlign=middle;',
                 x, y, w, h)


def interaction_edge(inter_id, source, target, label, itype):
    """Edge cell for one interaction."""
    style = _EDGE_STYLES.get(itype, _EDGE_STYLES['capability_invocation'])
    return _edge_cell(inter_id, label, style, source, target)


def external_edge(ext_id, source, target):
    """Edge cell from an internal service to an external catalog box."""
    return _edge_cell(f'{ext_id}--edge', '', _EDGE_STYLES['external'], source, target)


# ---------------------------------------------------------------------------
# Main Draw.io Rendering Pipeline
# ---------------------------------------------------------------------------

def render_drawio(view, style):
    """Render a complete Draw.io diagram and return the XML string."""
    positions, _ = compute_layout(view, style)
    canvas_w, canvas_h = compute_canvas_size(positions)
    margin = 60

    ai_ids = {s['id'] for s in view['services'] if s.get('isAIPersona')}

    cells = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>',
    ]

    # Title and breadcrumb
    cells.append(_cell('--title', view['title'],
                       'text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;'
                       'fontSize=20;fontStyle=1;fontColor=#1A202C;',
                       margin, margin, 900, 28))
    cells.append(_cell('--crumb', view.get('breadcrumb', ''),
                       'text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;'
                       'fontSize=11;fontColor=#64748B;',
                       margin, margin + 30, 600, 18))

    # Services
    for svc in view['services']:
        sid = svc['id']
        if sid not in positions or 'x' not in positions[sid]:
            continue
        p = positions[sid]
        if svc.get('isAIPersona'):
            persona = next((ap for ap in view.get('aiPersonas', []) if ap.get('serviceId') == sid), None)
            cells.append(ai_persona_cells(sid, p['x'], p['y'], p['w'], p['h'], svc, persona))
        else:
            cells.append(service_cells(sid, p['x'], p['y'], p['w'], p['h'], svc))

    # Human actors
    for actor in view.get('humanActors', []):
        aid = actor['id']
        if aid in positions:
            p = positions[aid]
            cells.append(human_actor_cell(aid, p['cx'], p['cy'], actor))

    # System actors
    for actor in view.get('systemActors', []):
        aid = actor['id']
        if aid in positions:
            p = positions[aid]
            cells.append(system_actor_cell(aid, p['cx'], p['cy'], actor))

    # External service boxes
    for ext in view.get('externalInteractions', []):
        eid = ext['id']
        if eid in positions:
            p = positions[eid]
            cells.append(external_service_cell(eid, p['x'], p['y'], p['w'], p['h'], ext))

    # Interaction edges
    for inter in view.get('interactions', []):
        src, tgt = inter['sourceId'], inter['targetId']
        if src in positions and tgt in positions:
            cells.append(interaction_edge(
                inter['id'], src, tgt,
                inter.get('label', ''),
                inter.get('interactionType', 'capability_invocation'),
            ))

    # External catalog edges (internal service → catalog box)
    for ext in view.get('externalInteractions', []):
        src = ext.get('internalServiceId')
        tgt = ext['id']
        if src in positions and tgt in positions:
            cells.append(external_edge(tgt, src, tgt))

    xml_body = '\n    '.join(cells)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<mxGraphModel dx="1422" dy="762" grid="0" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" '
            f'pageWidth="{canvas_w}" pageHeight="{canvas_h}" math="0" shadow="0">\n'
            f'  <root>\n'
            f'    {xml_body}\n'
            f'  </root>\n'
            f'</mxGraphModel>')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        model = json.load(f)

    style = load_style_config()

    if sys.argv[2] == '--l0':
        view        = filter_level0_view(model)
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'enterprise-l0.drawio'
    else:
        parent_id   = sys.argv[2]
        level       = int(sys.argv[3])
        output_path = sys.argv[4] if len(sys.argv) > 4 else f'{parent_id}-l{level}.drawio'
        view        = filter_service_architecture_view(model, parent_id, level)

    xml = render_drawio(view, style)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml)

    print(f"Generated: {output_path}")
    print(f"  View: {view['title']}")
    print(f"  Services: {len(view['services'])}")
    print(f"  AI Personas: {len(view.get('aiPersonas', []))}")
    print(f"  Human Actors: {len(view.get('humanActors', []))}")
    print(f"  System Actors: {len(view.get('systemActors', []))}")
    print(f"  External Interactions: {len(view.get('externalInteractions', []))}")
    print(f"  Interactions: {len(view.get('interactions', []))}")


if __name__ == '__main__':
    main()
