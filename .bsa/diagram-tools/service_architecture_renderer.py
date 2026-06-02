#!/usr/bin/env python3
"""
AI-First Business Service Architecture — Service Architecture Renderer

Generates professional SVG diagrams from the canonical service domain model.
Each diagram shows ONE service level: services, AI Personas, actors,
external interactions, and flows at that level.

Usage:
    python service_architecture_renderer.py <model.json> <parent_service_id> <level> [output.svg]
    python service_architecture_renderer.py <model.json> --l0 [output.svg]

Examples:
    python service_architecture_renderer.py model.json mfg 1 manufacturing-l1.svg
    python service_architecture_renderer.py model.json --l0 enterprise-l0.svg
"""

import json
import math
import sys
import os
from collections import deque
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Style Configuration Loader
# ---------------------------------------------------------------------------

def load_style_config(config_path=None):
    """Load style config from JSON or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check ../design/ first (toolkit deployment: .bsa/diagram-tools/ → .bsa/design/)
    # then ../../design/ (source repo: distribution/diagram-tools/ → design/)
    for rel in (('..', 'design'), ('..', '..', 'design')):
        candidate = os.path.join(script_dir, *rel, 'style-config.json')
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("style-config.json not found. Provide path via --style argument.")


# ---------------------------------------------------------------------------
# Model Filtering (Python port of canonical-model-renderer-filtering.js)
# ---------------------------------------------------------------------------

def filter_service_architecture_view(model, parent_service_id, target_level):
    """Extract everything needed for a single-level service architecture view."""
    parent_service = next((s for s in model['services'] if s['id'] == parent_service_id), None)
    if not parent_service:
        raise ValueError(f"Parent service '{parent_service_id}' not found in model.")

    child_services = [s for s in model['services']
                      if s.get('parentId') == parent_service_id and s.get('level') == target_level]
    child_ids = {s['id'] for s in child_services}

    ai_personas = [ap for ap in model.get('aiPersonas', []) if ap.get('serviceId') in child_ids]

    visible_ids = set(child_ids)

    human_actors = [a for a in model.get('humanActors', [])
                    if any(assoc.get('serviceId') in child_ids
                           for assoc in a.get('serviceAssociations', []))]
    for a in human_actors:
        visible_ids.add(a['id'])

    system_actors = [a for a in model.get('systemActors', [])
                     if any(assoc.get('serviceId') in child_ids
                            for assoc in a.get('serviceAssociations', []))]
    for a in system_actors:
        visible_ids.add(a['id'])

    external_interactions = [e for e in model.get('externalInteractions', [])
                             if e.get('internalServiceId') in child_ids]

    interactions = [i for i in model.get('interactions', [])
                    if i.get('sourceId') in visible_ids and i.get('targetId') in visible_ids]

    breadcrumb = build_breadcrumb(model, parent_service) + f' > Level {target_level}'

    return {
        'viewType': 'service_architecture',
        'title': f"{parent_service['name']} \u2014 Level {target_level} Services",
        'breadcrumb': breadcrumb,
        'parentService': parent_service,
        'services': child_services,
        'aiPersonas': ai_personas,
        'humanActors': human_actors,
        'systemActors': system_actors,
        'externalInteractions': external_interactions,
        'interactions': interactions,
    }


def filter_level0_view(model):
    """Extract enterprise-level L0 view with all associated entities.

    Shows all entities associated with L0 services: AI Personas, human actors
    (internal and external), system actors, and external interactions.
    This is consistent with how drill-down views handle other levels.
    """
    l0_services = [s for s in model['services'] if s.get('level') == 0]
    l0_ids = {s['id'] for s in l0_services}

    ai_personas = [ap for ap in model.get('aiPersonas', [])
                   if ap.get('serviceId') in l0_ids]
    persona_ids = {ap['id'] for ap in ai_personas}

    human_actors = [a for a in model.get('humanActors', [])
                    if any(assoc.get('serviceId') in l0_ids
                           for assoc in a.get('serviceAssociations', []))]
    human_ids = {a['id'] for a in human_actors}

    system_actors = [a for a in model.get('systemActors', [])
                     if any(assoc.get('serviceId') in l0_ids
                            for assoc in a.get('serviceAssociations', []))]
    system_ids = {a['id'] for a in system_actors}

    external_interactions = [e for e in model.get('externalInteractions', [])
                             if e.get('internalServiceId') in l0_ids]

    visible_ids = l0_ids | persona_ids | human_ids | system_ids
    interactions = [i for i in model.get('interactions', [])
                    if i.get('sourceId') in visible_ids and i.get('targetId') in visible_ids]

    org_name = model.get('modelMetadata', {}).get('organisation', 'Enterprise')
    return {
        'viewType': 'service_architecture_l0',
        'title': f"{org_name} \u2014 Enterprise Architecture (Level 0)",
        'breadcrumb': 'Enterprise',
        'services': l0_services,
        'aiPersonas': ai_personas,
        'humanActors': human_actors,
        'systemActors': system_actors,
        'externalInteractions': external_interactions,
        'interactions': interactions,
    }


def build_breadcrumb(model, service):
    crumbs = [service['name']]
    current = service
    while current.get('parentId'):
        current = next((s for s in model['services'] if s['id'] == current['parentId']), None)
        if current:
            crumbs.insert(0, current['name'])
    crumbs.insert(0, 'Enterprise')
    return ' > '.join(crumbs)


# ---------------------------------------------------------------------------
# SVG Primitives
# ---------------------------------------------------------------------------

FONT_FAMILY = "Inter, 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif"

def esc(text):
    """Escape XML special characters."""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def text_width(text, font_size=12, bold=False):
    """Estimate text width in pixels."""
    return len(text) * font_size * (0.62 if bold else 0.57)


def wrap_text(text, max_width, font_size=11):
    """Wrap text to fit within max_width, returning list of lines."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ''
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
    return lines[:3]  # max 3 lines


def svg_defs(style):
    """Generate SVG <defs> block with gradients, filters, and markers."""
    ai = style['entities']['aiPersona']
    interactions = style['interactions']

    defs = ['<defs>']

    # AI Persona header gradient
    defs.append(f'''  <linearGradient id="aiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:{ai['headerColorStart']};stop-opacity:1" />
    <stop offset="100%" style="stop-color:{ai['headerColorEnd']};stop-opacity:1" />
  </linearGradient>''')

    # AI Persona glow filter
    defs.append(f'''  <filter id="aiGlow" x="-15%" y="-15%" width="130%" height="140%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="5"/>
    <feOffset dx="0" dy="3" result="offsetblur"/>
    <feComponentTransfer>
      <feFuncA type="linear" slope="0.2"/>
    </feComponentTransfer>
    <feMerge>
      <feMergeNode/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>''')

    # Drop shadow for service boxes
    defs.append('''  <filter id="shadow" x="-5%" y="-5%" width="115%" height="120%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
    <feOffset dx="1" dy="2" result="offsetblur"/>
    <feComponentTransfer>
      <feFuncA type="linear" slope="0.12"/>
    </feComponentTransfer>
    <feMerge>
      <feMergeNode/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>''')

    # Arrow markers for each interaction type
    for int_type, int_style in interactions.items():
        s = int_style.get('arrowSize', 8)
        if int_style.get('arrowType') == 'filled':
            defs.append(f'''  <marker id="arr-{int_type}" viewBox="0 0 10 10" refX="10" refY="5"
      markerWidth="{s}" markerHeight="{s}" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 Z" fill="{int_style['color']}"/>
  </marker>''')
        elif int_style.get('arrowType') == 'open':
            defs.append(f'''  <marker id="arr-{int_type}" viewBox="0 0 10 10" refX="10" refY="5"
      markerWidth="{s}" markerHeight="{s}" orient="auto-start-reverse">
    <path d="M 0 1 L 10 5 L 0 9" fill="none" stroke="{int_style['color']}" stroke-width="1.5"/>
  </marker>''')

    defs.append('</defs>')
    return '\n'.join(defs)


# ---------------------------------------------------------------------------
# Graph Analysis for Layout
# ---------------------------------------------------------------------------

def build_service_graph(view):
    """Build directed graph from service-to-service interactions."""
    service_ids = {s['id'] for s in view['services']}
    forward = {sid: [] for sid in service_ids}
    reverse = {sid: [] for sid in service_ids}

    for inter in view.get('interactions', []):
        src, tgt = inter['sourceId'], inter['targetId']
        if src in service_ids and tgt in service_ids and src != tgt:
            forward[src].append(tgt)
            reverse[tgt].append(src)

    return forward, reverse


def assign_columns(view):
    """Assign each service to a column using BFS layering.

    Returns dict {service_id: column_index}.
    Column 0 is leftmost (source services), higher columns are downstream.
    Uses first-visit-wins to prevent cycles from inflating column numbers.
    """
    forward, reverse = build_service_graph(view)
    service_ids = {s['id'] for s in view['services']}

    # Find root services (no incoming from other services)
    roots = sorted(
        [sid for sid in service_ids if not reverse[sid]],
        key=lambda s: s  # deterministic order
    )
    if not roots:
        # Cyclic graph — pick the most "source-like" node
        # (highest net outflow, most outgoing edges, then name for tiebreak)
        svc_name = {s['id']: s['name'] for s in view['services']}
        roots = [max(service_ids, key=lambda s: (
            len(forward.get(s, [])) - len(reverse.get(s, [])),
            len(forward.get(s, [])),
            svc_name.get(s, s),
        ))]

    # BFS: first visit wins (prevents cycles from inflating columns)
    columns = {}
    queue = deque([(r, 0) for r in roots])

    while queue:
        node, col = queue.popleft()
        if node in columns:
            continue  # already assigned — skip
        columns[node] = col
        for tgt in forward.get(node, []):
            if tgt not in columns:
                queue.append((tgt, col + 1))

    # Unvisited (disconnected) services default to column 0
    for sid in service_ids:
        if sid not in columns:
            columns[sid] = 0

    return columns


# ---------------------------------------------------------------------------
# Icon Generators
# ---------------------------------------------------------------------------

def person_icon(cx, cy, size=40, color='#92400E'):
    """Stick figure person icon, centred at (cx, cy)."""
    s = size
    head_r = s * 0.16
    head_y = cy - s * 0.38
    neck_y = head_y + head_r
    shoulder_y = cy - s * 0.12
    hip_y = cy + s * 0.12
    foot_y = cy + s * 0.42
    arm_w = s * 0.28
    leg_w = s * 0.22
    sw = max(1.5, s * 0.06)

    return f'''<circle cx="{cx}" cy="{head_y:.1f}" r="{head_r:.1f}" fill="none" stroke="{color}" stroke-width="{sw:.1f}"/>
<line x1="{cx}" y1="{neck_y:.1f}" x2="{cx}" y2="{hip_y:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"/>
<line x1="{cx - arm_w:.1f}" y1="{shoulder_y + 4:.1f}" x2="{cx + arm_w:.1f}" y2="{shoulder_y + 4:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"/>
<line x1="{cx}" y1="{hip_y:.1f}" x2="{cx - leg_w:.1f}" y2="{foot_y:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"/>
<line x1="{cx}" y1="{hip_y:.1f}" x2="{cx + leg_w:.1f}" y2="{foot_y:.1f}" stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"/>'''


def robot_icon(cx, cy, size=18, color='#F3E8FF'):
    """Robot face icon for AI Persona headers."""
    w = size
    h = size * 0.85
    x = cx - w / 2
    y = cy - h / 2
    eye_r = w * 0.12
    eye_y = y + h * 0.38
    mouth_y = y + h * 0.7
    ant_h = h * 0.3
    return f'''<line x1="{cx}" y1="{y - ant_h:.1f}" x2="{cx}" y2="{y:.1f}" stroke="{color}" stroke-width="1.5"/>
<circle cx="{cx}" cy="{y - ant_h:.1f}" r="2.5" fill="{color}"/>
<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{w * 0.2:.1f}" fill="none" stroke="{color}" stroke-width="1.5"/>
<circle cx="{cx - w * 0.22:.1f}" cy="{eye_y:.1f}" r="{eye_r:.1f}" fill="{color}"/>
<circle cx="{cx + w * 0.22:.1f}" cy="{eye_y:.1f}" r="{eye_r:.1f}" fill="{color}"/>
<line x1="{cx - w * 0.22:.1f}" y1="{mouth_y:.1f}" x2="{cx + w * 0.22:.1f}" y2="{mouth_y:.1f}" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>'''


def server_icon(cx, cy, size=36, color='#0C4A6E'):
    """Server/system icon, centred at (cx, cy)."""
    w = size * 0.7
    h = size * 0.85
    x = cx - w / 2
    y = cy - h / 2
    tier_h = h / 3
    sw = max(1.2, size * 0.04)
    led_r = max(1.5, size * 0.05)
    return f'''<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="none" stroke="{color}" stroke-width="{sw:.1f}"/>
<line x1="{x:.1f}" y1="{y + tier_h:.1f}" x2="{x + w:.1f}" y2="{y + tier_h:.1f}" stroke="{color}" stroke-width="{sw * 0.7:.1f}"/>
<line x1="{x:.1f}" y1="{y + 2 * tier_h:.1f}" x2="{x + w:.1f}" y2="{y + 2 * tier_h:.1f}" stroke="{color}" stroke-width="{sw * 0.7:.1f}"/>
<circle cx="{x + w * 0.25:.1f}" cy="{y + tier_h * 0.5:.1f}" r="{led_r}" fill="{color}"/>
<circle cx="{x + w * 0.45:.1f}" cy="{y + tier_h * 0.5:.1f}" r="{led_r}" fill="{color}"/>
<circle cx="{x + w * 0.25:.1f}" cy="{y + tier_h * 1.5:.1f}" r="{led_r}" fill="{color}"/>
<circle cx="{x + w * 0.45:.1f}" cy="{y + tier_h * 1.5:.1f}" r="{led_r}" fill="{color}"/>'''


def sop_icon(cx, cy, size=20, color='#854D0E'):
    """Document with checkmark icon."""
    w = size * 0.6
    h = size * 0.7
    x = cx - w / 2
    y = cy - h / 2
    fold = w * 0.3
    return f'''<path d="M {x:.1f} {y:.1f} L {x + w - fold:.1f} {y:.1f} L {x + w:.1f} {y + fold:.1f}
              L {x + w:.1f} {y + h:.1f} L {x:.1f} {y + h:.1f} Z" fill="#FEFCE8" stroke="{color}" stroke-width="1"/>
<path d="M {x + w - fold:.1f} {y:.1f} L {x + w - fold:.1f} {y + fold:.1f} L {x + w:.1f} {y + fold:.1f}" fill="none" stroke="{color}" stroke-width="0.8"/>
<path d="M {cx - 1:.1f} {cy + 1:.1f} l 2 2 l 4 -4" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>'''


# ---------------------------------------------------------------------------
# Entity Renderers
# ---------------------------------------------------------------------------

def render_service_box(x, y, w, h, service, style):
    """Render a business service box with mission text."""
    s = style['entities']['service']
    hh = 30  # header height
    cr = s['cornerRadius']
    name = esc(service['name'])
    mission = service.get('mission', '')

    elements = [f'<g class="service" data-id="{service["id"]}" filter="url(#shadow)">']

    # Body
    elements.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{cr}" '
                    f'fill="{s["fillColor"]}" stroke="{s["strokeColor"]}" stroke-width="{s["borderWidth"]}"/>')

    # Header
    elements.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="{cr}" fill="{s["headerColor"]}"/>')
    elements.append(f'  <rect x="{x}" y="{y + hh - cr}" width="{w}" height="{cr}" fill="{s["headerColor"]}"/>')

    # Header text
    elements.append(f'  <text x="{x + w / 2}" y="{y + hh / 2 + 1}" text-anchor="middle" '
                    f'dominant-baseline="central" font-family="{FONT_FAMILY}" '
                    f'font-size="13" font-weight="600" fill="#FFFFFF">{name}</text>')

    # Mission text (wrapped)
    if mission:
        lines = wrap_text(mission, w - 16, 10)
        for i, line in enumerate(lines):
            ty = y + hh + 14 + i * 14
            elements.append(f'  <text x="{x + 8}" y="{ty}" font-family="{FONT_FAMILY}" '
                            f'font-size="10" fill="#475569">{esc(line)}</text>')

    elements.append('</g>')
    return '\n'.join(elements)


def render_ai_persona_box(x, y, w, h, service, persona, style):
    """Render an AI Persona service box with robot icon and branded styling."""
    s = style['entities']['aiPersona']
    hh = 34  # header height
    cr = s['cornerRadius']
    name = esc(service['name'])
    mission = service.get('mission', '')

    elements = [f'<g class="ai-persona" data-id="{service["id"]}" filter="url(#aiGlow)">']

    # Body
    elements.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{cr}" '
                    f'fill="{s["fillColor"]}" stroke="{s["strokeColor"]}" stroke-width="{s["borderWidth"]}"/>')

    # Header with gradient
    elements.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="{cr}" fill="url(#aiGrad)"/>')
    elements.append(f'  <rect x="{x}" y="{y + hh - cr}" width="{w}" height="{cr}" fill="url(#aiGrad)"/>')

    # Robot icon in header
    elements.append(f'  {robot_icon(x + 22, y + hh / 2, 18, "#F3E8FF")}')

    # Header text (offset for icon)
    elements.append(f'  <text x="{x + 42 + (w - 42) / 2}" y="{y + hh / 2 + 1}" text-anchor="middle" '
                    f'dominant-baseline="central" font-family="{FONT_FAMILY}" '
                    f'font-size="14" font-weight="700" fill="#FFFFFF">{name}</text>')

    # "AI PERSONA" label
    elements.append(f'  <text x="{x + w / 2}" y="{y + hh + 16}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="9" font-weight="600" '
                    f'fill="{s["aiLabelColor"]}" letter-spacing="1.5">AI PERSONA</text>')

    # Authority model badge
    if persona:
        auth = persona.get('authorityModel', '')
        if auth:
            badge_text = auth.upper()
            elements.append(f'  <text x="{x + w / 2}" y="{y + hh + 30}" text-anchor="middle" '
                            f'font-family="{FONT_FAMILY}" font-size="8" font-weight="500" '
                            f'fill="#64748B">Authority: {esc(badge_text)}</text>')

    # Mission text (wrapped)
    if mission:
        lines = wrap_text(mission, w - 16, 9)
        start_y = y + hh + 42
        for i, line in enumerate(lines):
            ty = start_y + i * 12
            if ty < y + h - 6:
                elements.append(f'  <text x="{x + 8}" y="{ty}" font-family="{FONT_FAMILY}" '
                                f'font-size="9" fill="#475569">{esc(line)}</text>')

    # SOP marker if applicable
    if persona and persona.get('sops'):
        elements.append(f'  <g transform="translate({x + w - 14},{y - 8})">')
        elements.append(f'    {sop_icon(10, 10, 20)}')
        elements.append('  </g>')

    elements.append('</g>')
    return '\n'.join(elements)


def render_human_actor(cx, cy, actor, style):
    """Render a human actor as a stick figure with name below."""
    s = style['entities']['humanActor']
    name = esc(actor['name'])
    is_ext = actor.get('isExternal', False)
    color = '#92400E'
    icon_size = 44

    elements = [f'<g class="human-actor" data-id="{actor["id"]}">']

    # Stick figure person
    elements.append(f'  {person_icon(cx, cy - 8, icon_size, color)}')

    # Name text below
    elements.append(f'  <text x="{cx}" y="{cy + 28}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="11" font-weight="500" '
                    f'fill="#1A202C">{name}</text>')

    # EXT badge for external actors
    if is_ext:
        elements.append(f'  <rect x="{cx + 16}" y="{cy - 28}" width="26" height="12" rx="3" '
                        f'fill="#E2E8F0" stroke="#94A3B8" stroke-width="0.8"/>')
        elements.append(f'  <text x="{cx + 29}" y="{cy - 20}" text-anchor="middle" '
                        f'font-family="{FONT_FAMILY}" font-size="7" font-weight="700" '
                        f'fill="#64748B">EXT</text>')

    elements.append('</g>')
    return '\n'.join(elements)


def render_system_actor(cx, cy, actor, style):
    """Render a system actor as a server icon with name below."""
    name = esc(actor['name'])
    color = '#0C4A6E'
    icon_size = 38

    elements = [f'<g class="system-actor" data-id="{actor["id"]}">']

    # Server icon
    elements.append(f'  {server_icon(cx, cy - 6, icon_size, color)}')

    # Name text below
    elements.append(f'  <text x="{cx}" y="{cy + 24}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="10" font-weight="500" '
                    f'fill="#1A202C">{name}</text>')

    elements.append('</g>')
    return '\n'.join(elements)


def render_external_service(x, y, w, h, ext, style):
    """Render an external service box with dashed border."""
    s = style['entities']['externalService']
    cr = s['cornerRadius']
    name = esc(ext['externalServiceName'])
    domain = esc(ext.get('externalDomain', ''))

    return f'''<g class="external-service" data-id="{ext['id']}" filter="url(#shadow)">
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{cr}" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="6,3"/>
  <rect x="{x}" y="{y}" width="{w}" height="22" rx="{cr}" fill="#94A3B8" opacity="0.12"/>
  <rect x="{x}" y="{y + 14}" width="{w}" height="8" fill="#94A3B8" opacity="0.12"/>
  <text x="{x + w / 2}" y="{y + 13}" text-anchor="middle" dominant-baseline="central" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" fill="#475569">{name}</text>
  <text x="{x + w / 2}" y="{y + h - 12}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="9" fill="#94A3B8">{domain}</text>
  <rect x="{x + w - 30}" y="{y + 3}" width="26" height="13" rx="3" fill="#E2E8F0" stroke="#94A3B8" stroke-width="0.8"/>
  <text x="{x + w - 17}" y="{y + 11.5}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="8" font-weight="700" fill="#64748B">EXT</text>
</g>'''


# ---------------------------------------------------------------------------
# Interaction Renderer
# ---------------------------------------------------------------------------

def render_interaction(interaction, positions, style, used_labels, pair_counts,
                       edge_data=None):
    """Render an interaction flow line between two entities.

    If edge_data is provided (from Graphviz), uses its pre-computed path
    and label position. Otherwise falls back to manual bezier calculation.
    """
    src_id = interaction['sourceId']
    tgt_id = interaction['targetId']

    if src_id not in positions or tgt_id not in positions:
        return ''

    sp = positions[src_id]
    tp = positions[tgt_id]
    int_type = interaction.get('interactionType', 'capability_invocation')
    ist = style['interactions'].get(int_type, style['interactions']['capability_invocation'])
    label = esc(interaction.get('label', ''))

    # Line style attributes
    dash = ''
    if ist.get('style') == 'dashed':
        dash = f' stroke-dasharray="{ist.get("dashArray", "6,4")}"'
    elif ist.get('style') == 'dotted':
        dash = f' stroke-dasharray="{ist.get("dashArray", "2,3")}"'

    marker = ''
    if ist.get('arrowType', 'none') != 'none':
        marker = f' marker-end="url(#arr-{int_type})"'

    # --- Build path and label position ---
    if edge_data and edge_data.get('path'):
        # Graphviz-computed path with proper edge routing
        path = edge_data['path']
        if edge_data.get('label_pos'):
            lx, ly = edge_data['label_pos']
        else:
            lx = (sp['cx'] + tp['cx']) / 2
            ly = (sp['cy'] + tp['cy']) / 2
    else:
        # Manual bezier fallback
        sx, sy = sp['cx'], sp['cy']
        tx_pt, ty_pt = tp['cx'], tp['cy']
        x1, y1 = clip_to_boundary(sx, sy, tx_pt, ty_pt, sp)
        x2, y2 = clip_to_boundary(tx_pt, ty_pt, sx, sy, tp)

        pair_key = tuple(sorted([src_id, tgt_id]))
        pair_idx = pair_counts.get(pair_key, 0)
        pair_counts[pair_key] = pair_idx + 1

        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx * dx + dy * dy) or 1
        curve_amt = min(40, max(15, dist * 0.08))
        direction = 1 if pair_idx % 2 == 0 else -1
        if pair_idx > 0:
            curve_amt += 18 * pair_idx

        px, py = -dy / dist, dx / dist
        ctrl_x = (x1 + x2) / 2 + px * curve_amt * direction
        ctrl_y = (y1 + y2) / 2 + py * curve_amt * direction

        path = f'M {x1:.1f},{y1:.1f} Q {ctrl_x:.1f},{ctrl_y:.1f} {x2:.1f},{y2:.1f}'

        lx = (x1 + 2 * ctrl_x + x2) / 4
        ly = (y1 + 2 * ctrl_y + y2) / 4
        lx += px * 10 * direction
        ly += py * 10 * direction

    # --- Render ---
    elements = [f'<g class="interaction" data-id="{interaction["id"]}">']
    elements.append(f'  <path d="{path}" fill="none" stroke="{ist["color"]}" '
                    f'stroke-width="{ist["width"]}"{dash}{marker}/>')

    if label:
        lw = text_width(label, 9, True) + 14
        for olx, oly, olw in used_labels:
            if abs(lx - olx) < (lw + olw) / 2 and abs(ly - oly) < 16:
                ly -= 20
        used_labels.append((lx, ly, lw))

        label_bg = ist.get('labelBackground', '#FFFFFF')
        elements.append(f'  <rect x="{lx - lw / 2:.1f}" y="{ly - 8:.1f}" width="{lw:.1f}" '
                        f'height="16" rx="3" fill="{label_bg}" opacity="0.92" '
                        f'stroke="{ist["color"]}" stroke-width="0.3" stroke-opacity="0.3"/>')
        elements.append(f'  <text x="{lx:.1f}" y="{ly + 2:.1f}" text-anchor="middle" '
                        f'dominant-baseline="central" font-family="{FONT_FAMILY}" '
                        f'font-size="9" font-weight="500" fill="{ist["color"]}">{label}</text>')

    elements.append('</g>')
    return '\n'.join(elements)


def clip_to_boundary(from_x, from_y, to_x, to_y, pos):
    """Clip a line from (from_x, from_y) heading toward (to_x, to_y) to the entity boundary."""
    if 'r' in pos:
        # Circle (actor icon)
        dx, dy = to_x - from_x, to_y - from_y
        d = math.sqrt(dx * dx + dy * dy) or 1
        return from_x + dx / d * pos['r'], from_y + dy / d * pos['r']

    # Rectangle
    x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
    cx, cy = x + w / 2, y + h / 2
    dx, dy = to_x - cx, to_y - cy

    if dx == 0 and dy == 0:
        return cx, cy

    # Find intersection with rectangle edges
    # Scale factor to reach edge
    t_candidates = []
    if dx != 0:
        t_candidates.append(abs((w / 2) / dx))
    if dy != 0:
        t_candidates.append(abs((h / 2) / dy))

    if not t_candidates:
        return cx, cy

    t = min(t_candidates)
    return cx + dx * t, cy + dy * t


# ---------------------------------------------------------------------------
# Layout Engine — Manual Network Layout (Fallback)
# ---------------------------------------------------------------------------

def compute_layout_manual(view, style):
    """Manual fallback layout: services in a centred grid, actors orbit their
    primary service on the outward-facing side.

    Layout strategy:
    1. Services in a centred grid, ordered by interaction topology.
    2. Each actor (human / system) is assigned to a PRIMARY service
       via weighted scoring (owner=10, other assoc=1, interaction=2).
    3. The actor is placed as a satellite of that service, projected
       outward from the grid centre so it sits on the periphery.
    4. Multiple actors sharing a service fan out in an arc.
    5. External services sit on the far right.
    """
    positions = {}
    margin = 60
    title_h = 65

    # Entity dimensions
    svc_w, svc_h = 195, 95
    ai_w, ai_h = 225, 130
    ext_w, ext_h = 160, 56

    services = view['services']
    n = len(services)
    if n == 0:
        return positions

    ai_ids = {s['id'] for s in services if s.get('isAIPersona')}
    service_id_set = {s['id'] for s in services}

    # Grid spacing — enough for interaction labels between boxes
    grid_gap_x = 100
    grid_gap_y = 80
    satellite_dist = 70  # gap from service edge to actor centre

    # Cell size adapts to entities in this view
    has_ai = bool(ai_ids)
    max_ew = ai_w if has_ai else svc_w
    max_eh = ai_h if has_ai else svc_h
    cell_w = max_ew + grid_gap_x
    cell_h = max_eh + grid_gap_y

    # === 1. SERVICE GRID ===

    grid_cols = max(1, min(n, math.ceil(math.sqrt(n * 1.5))))
    grid_rows = math.ceil(n / grid_cols)

    col_map = assign_columns(view)
    ordered = sorted(services, key=lambda s: (
        1 if s.get('serviceType') == 'support' else 0,
        col_map.get(s['id'], 0),
        s['name'],
    ))

    # Leave margin for satellite actors around the grid
    actor_zone = satellite_dist + 70
    grid_left = margin + actor_zone
    grid_top = margin + title_h + actor_zone

    for idx, svc in enumerate(ordered):
        row = idx // grid_cols
        col = idx % grid_cols
        items_in_row = min(grid_cols, n - row * grid_cols)
        row_offset = (grid_cols - items_in_row) * cell_w / 2

        is_ai = svc['id'] in ai_ids
        w = ai_w if is_ai else svc_w
        h = ai_h if is_ai else svc_h

        cx = grid_left + row_offset + col * cell_w + max_ew / 2
        cy = grid_top + row * cell_h + max_eh / 2

        positions[svc['id']] = {
            'x': cx - w / 2, 'y': cy - h / 2,
            'w': w, 'h': h, 'cx': cx, 'cy': cy,
        }

    # Grid centroid (used to compute outward direction)
    svc_pos = {k: v for k, v in positions.items() if k in service_id_set}
    gb_cx = sum(p['cx'] for p in svc_pos.values()) / len(svc_pos)
    gb_cy = sum(p['cy'] for p in svc_pos.values()) / len(svc_pos)

    # === 2. ASSIGN EACH ACTOR TO ITS PRIMARY SERVICE ===

    all_interactions = view.get('interactions', [])
    all_actors = list(view.get('humanActors', [])) + list(view.get('systemActors', []))

    def find_primary_service(actor):
        """Weighted scoring: owner(10), other assoc(1), interaction(2)."""
        weights = {}
        for assoc in actor.get('serviceAssociations', []):
            sid = assoc['serviceId']
            if sid not in service_id_set:
                continue
            w = 10 if assoc.get('associationType') == 'owner' else 1
            weights[sid] = weights.get(sid, 0) + w
        aid = actor['id']
        for inter in all_interactions:
            if inter['sourceId'] == aid and inter['targetId'] in service_id_set:
                weights[inter['targetId']] = weights.get(inter['targetId'], 0) + 2
            elif inter['targetId'] == aid and inter['sourceId'] in service_id_set:
                weights[inter['sourceId']] = weights.get(inter['sourceId'], 0) + 2
        if not weights:
            return None
        return max(weights, key=weights.get)

    svc_actors = {}   # service_id -> [actor, ...]
    unassigned = []
    for actor in all_actors:
        primary = find_primary_service(actor)
        if primary:
            svc_actors.setdefault(primary, []).append(actor)
        else:
            unassigned.append(actor)

    # === 3. PLACE ACTORS AS SATELLITES ===

    placed_positions = []  # (x, y) for collision avoidance

    for sid, actors in svc_actors.items():
        sp = positions[sid]
        scx, scy = sp['cx'], sp['cy']
        sw, sh = sp['w'], sp['h']

        # Outward angle: from grid centroid through this service
        dx = scx - gb_cx
        dy = scy - gb_cy
        if abs(dx) < 1 and abs(dy) < 1:
            base_angle = -math.pi / 2  # default: above
        else:
            base_angle = math.atan2(dy, dx)

        na = len(actors)
        fan = min(math.pi * 0.4, max(math.pi / 6, (na - 1) * math.pi / 5))

        for i, actor in enumerate(actors):
            if na == 1:
                angle = base_angle
            else:
                t = (i / (na - 1)) - 0.5   # range -0.5 … +0.5
                angle = base_angle + t * fan

            # Distance from service centre to its edge at this angle
            ca, sa = abs(math.cos(angle)), abs(math.sin(angle))
            if ca < 0.001:
                edge_d = sh / 2
            elif sa < 0.001:
                edge_d = sw / 2
            else:
                edge_d = min(sw / (2 * ca), sh / (2 * sa))

            total_d = edge_d + satellite_dist
            ax = scx + total_d * math.cos(angle)
            ay = scy + total_d * math.sin(angle)

            # Collision avoidance with previously placed actors
            for px, py in placed_positions:
                if math.sqrt((ax - px) ** 2 + (ay - py) ** 2) < 80:
                    ax += 45 * math.cos(angle)
                    ay += 45 * math.sin(angle)

            placed_positions.append((ax, ay))
            positions[actor['id']] = {'cx': ax, 'cy': ay, 'r': 30}

    # Unassigned actors (no visible connections) — float above grid
    if unassigned:
        g_top = min(p['y'] for p in svc_pos.values())
        for i, actor in enumerate(unassigned):
            ax = gb_cx + (i - (len(unassigned) - 1) / 2) * 120
            ay = g_top - satellite_dist
            positions[actor['id']] = {'cx': ax, 'cy': ay, 'r': 30}

    # External services on far right
    exts = view.get('externalInteractions', [])
    if exts:
        g_right = max(p['x'] + p['w'] for p in svc_pos.values())
        g_top = min(p['y'] for p in svc_pos.values())
        ext_x = g_right + 100
        for i, ext in enumerate(exts):
            ey = g_top + 20 + i * (ext_h + 30)
            positions[ext['id']] = {
                'x': ext_x, 'y': ey, 'w': ext_w, 'h': ext_h,
                'cx': ext_x + ext_w / 2, 'cy': ey + ext_h / 2,
            }

    return positions


# ---------------------------------------------------------------------------
# Layout Engine — Graphviz (Primary)
# ---------------------------------------------------------------------------

def graphviz_available():
    """Check if Graphviz neato engine is installed."""
    return shutil.which('neato') is not None


def build_dot_graph(view, style):
    """Build Graphviz DOT representation of the canonical model view."""
    PT_PER_INCH = 72
    ai_ids = {s['id'] for s in view['services'] if s.get('isAIPersona')}
    service_ids = {s['id'] for s in view['services']}

    lines = ['digraph G {']
    lines.append('  graph [layout=neato, overlap=prism, overlap_scaling=1.5,')
    lines.append('         splines=curved, sep="+25", esep="+10"];')
    lines.append('  node [fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')

    # Service nodes — sized to match our SVG entity dimensions
    svc_w, svc_h = 195, 95
    ai_w, ai_h = 225, 130
    for svc in view['services']:
        is_ai = svc['id'] in ai_ids
        w = (ai_w if is_ai else svc_w) / PT_PER_INCH
        h = (ai_h if is_ai else svc_h) / PT_PER_INCH
        lines.append(f'  "{svc["id"]}" [width={w:.4f}, height={h:.4f}, '
                     f'fixedsize=true, shape=box];')

    # Human actors — circular collision area
    actor_d = 80 / PT_PER_INCH
    for actor in view.get('humanActors', []):
        lines.append(f'  "{actor["id"]}" [width={actor_d:.4f}, '
                     f'height={actor_d:.4f}, fixedsize=true, shape=circle];')

    # System actors
    for actor in view.get('systemActors', []):
        lines.append(f'  "{actor["id"]}" [width={actor_d:.4f}, '
                     f'height={actor_d:.4f}, fixedsize=true, shape=circle];')

    # External services
    ext_w = 160 / PT_PER_INCH
    ext_h = 56 / PT_PER_INCH
    for ext in view.get('externalInteractions', []):
        lines.append(f'  "{ext["id"]}" [width={ext_w:.4f}, height={ext_h:.4f}, '
                     f'fixedsize=true, shape=box];')

    # Interaction edges
    for inter in view.get('interactions', []):
        src = inter['sourceId']
        tgt = inter['targetId']
        label = inter.get('label', '').replace('"', '\\"')
        src_is_svc = src in service_ids
        tgt_is_svc = tgt in service_ids
        if src_is_svc and tgt_is_svc:
            length, weight = 3.5, 1
        else:
            length, weight = 2.0, 2
        lines.append(f'  "{src}" -> "{tgt}" [label="{label}", '
                     f'len={length:.1f}, weight={weight}];')

    # Invisible affinity edges: keep actors near associated services
    all_actors = (list(view.get('humanActors', []))
                  + list(view.get('systemActors', [])))
    for actor in all_actors:
        connected = set()
        for inter in view.get('interactions', []):
            if inter['sourceId'] == actor['id'] and inter['targetId'] in service_ids:
                connected.add(inter['targetId'])
            elif inter['targetId'] == actor['id'] and inter['sourceId'] in service_ids:
                connected.add(inter['sourceId'])
        for assoc in actor.get('serviceAssociations', []):
            sid = assoc['serviceId']
            if sid in service_ids and sid not in connected:
                w = 4 if assoc.get('associationType') == 'owner' else 2
                lines.append(f'  "{actor["id"]}" -> "{sid}" '
                             f'[style=invis, len=1.8, weight={w}];')
        if not connected and not any(
            a['serviceId'] in service_ids
            for a in actor.get('serviceAssociations', [])
        ) and view['services']:
            lines.append(f'  "{actor["id"]}" -> "{view["services"][0]["id"]}" '
                         f'[style=invis, len=4.0, weight=1];')

    lines.append('}')
    return '\n'.join(lines)


def run_graphviz(dot_string, engine='neato'):
    """Run Graphviz layout engine, return parsed JSON."""
    result = subprocess.run(
        [engine, '-Tjson'],
        input=dot_string,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Graphviz error: {result.stderr.strip()}")
    return json.loads(result.stdout)


def parse_edge_spline(pos_str, tx_fn, ty_fn):
    """Parse Graphviz edge pos string into an SVG path.

    Graphviz pos format: [s,sx,sy] [e,ex,ey] x0,y0 x1,y1 ...
    Points form cubic B-spline segments (3n+1 control points).
    """
    tokens = pos_str.strip().split(' ')
    end_arrow = None
    start_arrow = None
    raw_points = []

    for token in tokens:
        if token.startswith('e,'):
            coords = token[2:].split(',')
            end_arrow = (float(coords[0]), float(coords[1]))
        elif token.startswith('s,'):
            coords = token[2:].split(',')
            start_arrow = (float(coords[0]), float(coords[1]))
        else:
            coords = token.split(',')
            if len(coords) >= 2:
                try:
                    raw_points.append((float(coords[0]), float(coords[1])))
                except ValueError:
                    continue

    if not raw_points:
        return None

    # Start at arrow start or first control point
    sx, sy = start_arrow if start_arrow else raw_points[0]
    path = f'M {tx_fn(sx):.1f},{ty_fn(sy):.1f}'

    if len(raw_points) < 4:
        # Not enough for cubic bezier — draw a line
        ex, ey = end_arrow if end_arrow else raw_points[-1]
        path += f' L {tx_fn(ex):.1f},{ty_fn(ey):.1f}'
        return path

    # Cubic bezier segments: start then groups of 3 (cp1, cp2, endpoint)
    i = 1
    while i + 2 <= len(raw_points):
        cp1 = raw_points[i]
        cp2 = raw_points[i + 1]
        if i + 3 > len(raw_points) and end_arrow:
            ep = end_arrow
        elif i + 2 < len(raw_points):
            ep = raw_points[i + 2]
        else:
            ep = end_arrow if end_arrow else raw_points[-1]
        path += (f' C {tx_fn(cp1[0]):.1f},{ty_fn(cp1[1]):.1f}'
                 f' {tx_fn(cp2[0]):.1f},{ty_fn(cp2[1]):.1f}'
                 f' {tx_fn(ep[0]):.1f},{ty_fn(ep[1]):.1f}')
        i += 3

    return path


def compute_layout_graphviz(view, style):
    """Compute layout using Graphviz for professional node placement and
    edge routing. Returns (positions, edge_paths)."""
    margin = 60
    title_h = 65

    dot = build_dot_graph(view, style)
    gv = run_graphviz(dot)

    # Bounding box: "llx,lly,urx,ury" in points
    bb = [float(x) for x in gv.get('bb', '0,0,800,600').split(',')]
    bb_llx, bb_lly, bb_urx, bb_ury = bb

    # Coordinate transforms (Graphviz y-up → SVG y-down)
    def tx(x):
        return x - bb_llx + margin

    def ty(y):
        return (bb_ury - y) + margin + title_h

    ai_ids = {s['id'] for s in view['services'] if s.get('isAIPersona')}
    service_ids = {s['id'] for s in view['services']}

    # Build object lookup by _gvid
    obj_by_gvid = {}
    for obj in gv.get('objects', []):
        gvid = obj.get('_gvid')
        if gvid is not None:
            obj_by_gvid[gvid] = obj

    # Parse node positions
    positions = {}
    for obj in gv.get('objects', []):
        name = obj.get('name', '')
        if not name or 'pos' not in obj:
            continue
        pos_parts = obj['pos'].split(',')
        cx = tx(float(pos_parts[0]))
        cy = ty(float(pos_parts[1]))

        if name in service_ids:
            is_ai = name in ai_ids
            w = 225 if is_ai else 195
            h = 130 if is_ai else 95
            positions[name] = {
                'x': cx - w / 2, 'y': cy - h / 2,
                'w': w, 'h': h, 'cx': cx, 'cy': cy,
            }
        elif any(a['id'] == name for a in view.get('humanActors', [])):
            positions[name] = {'cx': cx, 'cy': cy, 'r': 30}
        elif any(a['id'] == name for a in view.get('systemActors', [])):
            positions[name] = {'cx': cx, 'cy': cy, 'r': 30}
        else:
            ext_match = [e for e in view.get('externalInteractions', [])
                         if e['id'] == name]
            if ext_match:
                w, h = 160, 56
                positions[name] = {
                    'x': cx - w / 2, 'y': cy - h / 2,
                    'w': w, 'h': h, 'cx': cx, 'cy': cy,
                }

    # Parse edge paths
    edge_paths = {}
    for edge in gv.get('edges', []):
        if edge.get('style') == 'invis':
            continue
        tail_obj = obj_by_gvid.get(edge.get('tail'), {})
        head_obj = obj_by_gvid.get(edge.get('head'), {})
        tail_name = tail_obj.get('name', '')
        head_name = head_obj.get('name', '')
        if not tail_name or not head_name:
            continue

        pos_str = edge.get('pos', '')
        if not pos_str:
            continue

        svg_path = parse_edge_spline(pos_str, tx, ty)

        label_pos = None
        lp_str = edge.get('lp', '')
        if lp_str:
            lp_parts = lp_str.split(',')
            label_pos = (tx(float(lp_parts[0])), ty(float(lp_parts[1])))

        edge_label = edge.get('label', '')
        key = (tail_name, head_name, edge_label)
        edge_paths[key] = {'path': svg_path, 'label_pos': label_pos}

    return positions, edge_paths


def compute_layout(view, style):
    """Compute layout — Graphviz when available, manual fallback otherwise."""
    if graphviz_available():
        try:
            return compute_layout_graphviz(view, style)
        except Exception as e:
            print(f"Warning: Graphviz failed ({e}). Using manual layout.",
                  file=sys.stderr)
    else:
        print("Note: Install graphviz for better diagram layout.",
              file=sys.stderr)
    return compute_layout_manual(view, style), {}


def compute_canvas_size(positions, margin=60):
    """Compute canvas dimensions from entity positions."""
    if not positions:
        return 800, 600

    max_x = 0
    max_y = 0
    for p in positions.values():
        if 'x' in p:
            max_x = max(max_x, p['x'] + p['w'])
            max_y = max(max_y, p['y'] + p['h'])
        elif 'cx' in p:
            r = p.get('r', 30)
            max_x = max(max_x, p['cx'] + r + 60)
            max_y = max(max_y, p['cy'] + r + 40)

    canvas_w = int(max_x + margin + 200)  # room for legend
    canvas_h = int(max_y + margin + 140)  # room for legend
    return max(800, canvas_w), max(500, canvas_h)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def render_legend(canvas_w, canvas_h, style, has_ai=True):
    """Render a two-column legend in the bottom-right corner."""
    legend_w = 340
    legend_h = 130
    lx = canvas_w - legend_w - 30
    ly = canvas_h - legend_h - 20

    items = [f'<g class="legend" transform="translate({lx},{ly})">']
    items.append(f'  <rect x="0" y="0" width="{legend_w}" height="{legend_h}" rx="8" '
                 f'fill="white" stroke="#E5E7EB" stroke-width="1"/>')
    items.append(f'  <text x="12" y="18" font-family="{FONT_FAMILY}" font-size="11" '
                 f'font-weight="700" fill="#1A202C">Legend</text>')

    # Column 1: Entity types
    row_y = 36
    items.append(f'  <rect x="12" y="{row_y}" width="16" height="11" rx="3" '
                 f'fill="#F8F9FA" stroke="#2C5282" stroke-width="1.5"/>')
    items.append(f'  <text x="34" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                 f'font-size="9" fill="#475569">Business Service</text>')

    if has_ai:
        row_y += 18
        items.append(f'  <rect x="12" y="{row_y}" width="16" height="11" rx="3" '
                     f'fill="#F3E8FF" stroke="#6B21A8" stroke-width="1.5"/>')
        items.append(f'  <text x="34" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                     f'font-size="9" fill="#475569">AI Persona</text>')

    row_y += 18
    # Stick figure mini
    items.append(f'  <circle cx="20" cy="{row_y + 3}" r="3" fill="none" stroke="#92400E" stroke-width="1"/>')
    items.append(f'  <line x1="20" y1="{row_y + 6}" x2="20" y2="{row_y + 12}" stroke="#92400E" stroke-width="1"/>')
    items.append(f'  <text x="34" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                 f'font-size="9" fill="#475569">Human Actor</text>')

    row_y += 18
    items.append(f'  <rect x="14" y="{row_y}" width="12" height="10" rx="1.5" '
                 f'fill="none" stroke="#0C4A6E" stroke-width="1"/>')
    items.append(f'  <text x="34" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                 f'font-size="9" fill="#475569">System</text>')

    row_y += 18
    items.append(f'  <rect x="12" y="{row_y}" width="16" height="11" rx="3" fill="white" '
                 f'stroke="#94A3B8" stroke-width="1" stroke-dasharray="3,2"/>')
    items.append(f'  <text x="34" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                 f'font-size="9" fill="#475569">External Service</text>')

    # Column 2: Interaction types
    col2 = 165
    row_y = 36
    line_items = [
        ('Invocation', '#2C5282', None),
        ('Data Flow', '#059669', None),
        ('Escalation', '#DC2626', '5,3'),
        ('Support', '#15803D', '2,3'),
        ('External', '#94A3B8', '4,3'),
    ]
    for label, color, dash in line_items:
        d = f' stroke-dasharray="{dash}"' if dash else ''
        items.append(f'  <line x1="{col2}" y1="{row_y + 5}" x2="{col2 + 22}" '
                     f'y2="{row_y + 5}" stroke="{color}" stroke-width="2"{d}/>')
        # Small arrowhead
        items.append(f'  <path d="M {col2 + 18} {row_y + 2} L {col2 + 22} {row_y + 5} '
                     f'L {col2 + 18} {row_y + 8}" fill="none" stroke="{color}" stroke-width="1.2"/>')
        items.append(f'  <text x="{col2 + 28}" y="{row_y + 9}" font-family="{FONT_FAMILY}" '
                     f'font-size="9" fill="#475569">{label}</text>')
        row_y += 18

    items.append('</g>')
    return '\n'.join(items)


# ---------------------------------------------------------------------------
# Main Rendering Pipeline
# ---------------------------------------------------------------------------

def render_diagram(view, style):
    """Render a complete service architecture SVG diagram."""
    positions, edge_paths = compute_layout(view, style)
    canvas_w, canvas_h = compute_canvas_size(positions)
    margin = 60

    ai_ids = {s['id'] for s in view['services'] if s.get('isAIPersona')}

    svg = []

    # SVG header
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" '
               f'width="{canvas_w}" height="{canvas_h}">')
    svg.append(f'  <title>{esc(view["title"])}</title>')
    svg.append(f'  <desc>AI-First Business Service Architecture. {esc(view.get("breadcrumb", ""))}</desc>')

    # Defs
    svg.append(svg_defs(style))

    # Background
    svg.append(f'  <rect width="{canvas_w}" height="{canvas_h}" fill="#FAFBFC"/>')

    # Title
    svg.append(f'  <text x="{margin}" y="{margin + 18}" font-family="{FONT_FAMILY}" '
               f'font-size="20" font-weight="700" fill="#1A202C">{esc(view["title"])}</text>')
    svg.append(f'  <text x="{margin}" y="{margin + 36}" font-family="{FONT_FAMILY}" '
               f'font-size="11" font-weight="500" fill="#64748B" letter-spacing="0.3">'
               f'{esc(view.get("breadcrumb", ""))}</text>')

    # --- Render interactions (behind everything) ---
    svg.append('  <!-- Interactions -->')
    used_labels = []
    pair_counts = {}
    for interaction in view.get('interactions', []):
        key = (interaction['sourceId'], interaction['targetId'],
               interaction.get('label', ''))
        edge_data = edge_paths.get(key)
        svg.append(render_interaction(interaction, positions, style,
                                      used_labels, pair_counts, edge_data))

    # --- Render services ---
    svg.append('  <!-- Services -->')
    for service in view['services']:
        sid = service['id']
        if sid not in positions:
            continue
        p = positions[sid]
        if 'x' not in p:
            continue

        if service.get('isAIPersona'):
            persona = next((ap for ap in view.get('aiPersonas', [])
                           if ap.get('serviceId') == sid), None)
            svg.append(render_ai_persona_box(p['x'], p['y'], p['w'], p['h'],
                                              service, persona, style))
        else:
            svg.append(render_service_box(p['x'], p['y'], p['w'], p['h'],
                                           service, style))

    # --- Render human actors ---
    svg.append('  <!-- Human Actors -->')
    for actor in view.get('humanActors', []):
        if actor['id'] not in positions:
            continue
        p = positions[actor['id']]
        svg.append(render_human_actor(p['cx'], p['cy'], actor, style))

    # --- Render system actors ---
    svg.append('  <!-- System Actors -->')
    for actor in view.get('systemActors', []):
        if actor['id'] not in positions:
            continue
        p = positions[actor['id']]
        svg.append(render_system_actor(p['cx'], p['cy'], actor, style))

    # --- Render external services ---
    svg.append('  <!-- External Services -->')
    for ext in view.get('externalInteractions', []):
        if ext['id'] not in positions:
            continue
        p = positions[ext['id']]
        svg.append(render_external_service(p['x'], p['y'], p['w'], p['h'], ext, style))

    # Legend
    svg.append(render_legend(canvas_w, canvas_h, style, bool(view.get('aiPersonas'))))

    svg.append('</svg>')
    return '\n'.join(svg)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]

    with open(model_path, 'r') as f:
        model = json.load(f)

    style = load_style_config()

    if sys.argv[2] == '--l0':
        view = filter_level0_view(model)
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'enterprise-l0.svg'
    else:
        parent_id = sys.argv[2]
        level = int(sys.argv[3])
        output_path = sys.argv[4] if len(sys.argv) > 4 else f'{parent_id}-l{level}.svg'
        view = filter_service_architecture_view(model, parent_id, level)

    svg = render_diagram(view, style)

    with open(output_path, 'w') as f:
        f.write(svg)

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
