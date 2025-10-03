# features/orgchart/builder.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import pandas as pd

# Back-compat public type: Node = (name, designation)
Node = Tuple[str, str]

def _clean(s: Optional[str]) -> str:
    return (s or "").strip()

def build_upward_chain(row: pd.Series) -> List[Node]:
    """Two-level minimal chain (kept for back-compat)."""
    client_name = _clean(row.get("client_name"))
    client_desg = _clean(row.get("client_designation"))
    mgr_name    = _clean(row.get("reporting_manager"))
    mgr_desg    = _clean(row.get("reporting_manager_designation"))
    chain: List[Node] = []
    if client_name:
        chain.append((client_name, client_desg))
    if mgr_name:
        chain.append((mgr_name, mgr_desg))
    return chain

def build_upward_chain_multilevel(
    df: pd.DataFrame,
    seed_row: pd.Series,
    max_hops: int = 8,
) -> List[Node]:
    """Legacy multi-level (names only). Prefer build_upward_chain_to_ceo."""
    name_map: dict[str, pd.Series] = {}
    if "client_name" in df.columns:
        for _, r in df.iterrows():
            nm = _clean(r.get("client_name")).lower()
            if nm and nm not in name_map:
                name_map[nm] = r

    chain = build_upward_chain(seed_row)
    if not chain:
        return chain

    visited = set(n.lower() for n, _ in chain)
    hops = 1
    current_row = seed_row
    while hops < max_hops:
        next_mgr = _clean(current_row.get("reporting_manager"))
        next_des = _clean(current_row.get("reporting_manager_designation"))
        if not next_mgr:
            break
        if next_mgr.lower() in visited:
            break
        chain.append((next_mgr, next_des))
        visited.add(next_mgr.lower())
        current_row = name_map.get(next_mgr.lower())
        if current_row is None:
            break
        hops += 1
    return chain

# ---------- New: CEO chain carrying Business Unit ----------
RichNode = Dict[str, str]  # keys: name, designation, business_unit

def build_upward_chain_to_ceo(
    df: pd.DataFrame,
    seed_row: pd.Series,
    max_hops: int = 12,
) -> List[RichNode]:
    """
    Build list from selected person up to CEO (persona -> ... -> CEO).
    Each item carries business_unit for coloring.
    """
    def pack(r: pd.Series) -> RichNode:
        return {
            "name": _clean(r.get("client_name")),
            "designation": _clean(r.get("client_designation")),
            "business_unit": _clean(r.get("business_unit")),
        }

    # Build lookup by client_name (case-insensitive)
    name_map: dict[str, pd.Series] = {}
    if "client_name" in df.columns:
        for _, r in df.iterrows():
            nm = _clean(r.get("client_name")).lower()
            if nm and nm not in name_map:
                name_map[nm] = r

    chain: List[RichNode] = []
    cur = seed_row
    hops = 0
    seen: set[str] = set()

    while cur is not None and hops < max_hops:
        node = pack(cur)
        if not node["name"]:
            break
        low = node["name"].lower()
        if low in seen:
            break
        chain.append(node)
        seen.add(low)
        mgr = _clean(cur.get("reporting_manager"))
        if not mgr:
            break  # reached top
        cur = name_map.get(mgr.lower())
        hops += 1

    return chain

# ---------- New: CEO → … → persona path tree + persona’s direct reports ----------
def build_ceo_path_with_reportees_tree(
    df: pd.DataFrame,
    seed_row: pd.Series,
    max_hops: int = 12,
    include_reports_depth: int = 1,  # for now: only direct reports
) -> dict:
    """
    Build an ECharts tree dict where:
      - Root is CEO
      - The unique path CEO → … → persona is expanded
      - At the persona node, attach all direct reportees as children

    Node shape (ECharts-ready):
      { name: str, value: str (designation), bu: str, children: [...] }
    """
    def pack_row(r: pd.Series) -> dict:
        return {
            "name": _clean(r.get("client_name")),
            "value": _clean(r.get("client_designation")),
            "bu": _clean(r.get("business_unit")),
            "children": [],
        }

    # Maps
    name_map: dict[str, pd.Series] = {}
    reports_map: dict[str, list[pd.Series]] = {}
    for _, r in df.iterrows():
        nm = _clean(r.get("client_name"))
        if nm:
            nl = nm.lower()
            name_map.setdefault(nl, r)
        mgr = _clean(r.get("reporting_manager"))
        if mgr:
            reports_map.setdefault(mgr.lower(), []).append(r)

    # Build path persona -> ... -> CEO
    path_rows: List[pd.Series] = []
    cur = seed_row
    hops = 0
    seen = set()
    while cur is not None and hops < max_hops:
        nm = _clean(cur.get("client_name"))
        if not nm or nm.lower() in seen:
            break
        path_rows.append(cur)
        seen.add(nm.lower())
        mgr = _clean(cur.get("reporting_manager"))
        if not mgr:
            break
        cur = name_map.get(mgr.lower())
        hops += 1

    if not path_rows:
        # fallback: just the persona
        return pack_row(seed_row)

    # Reverse to CEO → … → persona
    path_rows = list(reversed(path_rows))

    # Build the linear path
    root = pack_row(path_rows[0])
    cursor = root
    for r in path_rows[1:]:
        nxt = pack_row(r)
        cursor["children"].append(nxt)
        cursor = nxt

    # Attach direct reportees at persona node
    persona_row = path_rows[-1]
    persona_name = _clean(persona_row.get("client_name"))
    if include_reports_depth >= 1 and persona_name:
        direct = reports_map.get(persona_name.lower(), [])
        for rep in direct:
            # Skip adding the persona itself (just in case)
            if _clean(rep.get("client_name")).lower() == persona_name.lower():
                continue
            cursor["children"].append(pack_row(rep))

    return root

def choose_person_id(row: pd.Series) -> str:
    """Prefer unique email as person id, else client_name."""
    email = _clean(row.get("email_address"))
    return email if email else _clean(row.get("client_name"))
