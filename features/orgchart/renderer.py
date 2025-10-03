# features/orgchart/renderer.py
from typing import List, Tuple, Optional, Dict, Union
import streamlit as st
from streamlit_echarts import st_echarts

# Accept either (name, designation) or rich dicts with BU
SimpleNode = Tuple[str, str]
RichNode = Dict[str, str]  # keys: name, designation, business_unit

_PALETTE = [
    "#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#6b7280", "#8b5cf6",
    "#059669", "#f97316", "#dc2626", "#0ea5e9", "#22c55e", "#a855f7",
]

def _assign_colors(bus: List[str]) -> Dict[str, str]:
    uniq, seen = [], set()
    for b in bus:
        b = (b or "").strip()
        if b and b not in seen:
            uniq.append(b); seen.add(b)
    return {b: _PALETTE[i % len(_PALETTE)] for i, b in enumerate(uniq)}

# ---------- Existing preview renderer (linear list) ----------
def _to_tree_data_and_legend(chain: List[Union[SimpleNode, RichNode]]):
    if not chain:
        return {}, {}
    rich: List[RichNode] = []
    for n in chain:
        if isinstance(n, tuple):
            rich.append({"name": n[0], "designation": n[1], "business_unit": ""})
        else:
            rich.append({
                "name": (n.get("name") or "").strip(),
                "designation": (n.get("designation") or "").strip(),
                "business_unit": (n.get("business_unit") or "").strip(),
            })
    bu_colors = _assign_colors([r["business_unit"] for r in rich])

    def mk_node(r: RichNode) -> dict:
        d = {"name": r["name"], "value": r["designation"], "children": []}
        col = bu_colors.get(r["business_unit"])
        if col:
            d["itemStyle"] = {"color": "#FFFFFF", "borderColor": col, "borderWidth": 2}
            d["label"] = {"color": "#213547"}
        return d

    root = mk_node(rich[0]); cursor = root
    for i in range(1, len(rich)):
        nxt = mk_node(rich[i])
        cursor["children"].append(nxt); cursor = nxt
    return root, bu_colors

def _render_legend(bu_colors: Dict[str, str]):
    if not bu_colors: return
    pills = []
    for bu, col in bu_colors.items():
        pills.append(
            f"""<span style="display:inline-flex;align-items:center;gap:8px;
                padding:4px 10px;border-radius:999px;background:#fff;
                border:1px solid #e6eaf0;margin:2px 6px 6px 0;
                box-shadow:0 1px 2px rgba(0,0,0,.04);">
                <span style="width:12px;height:12px;border-radius:3px;background:{col};display:inline-block;"></span>
                <span style="font-size:12px;color:#213547;">{bu}</span>
            </span>"""
        )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;margin-bottom:6px;">{"".join(pills)}</div>', unsafe_allow_html=True)

def _base_option(tree_data: dict) -> dict:
    return {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
            "formatter": "{b}: {c}",
            "borderColor": "#E0E7EF", "backgroundColor": "#FFFFFF",
            "textStyle": {"color": "#213547"},
            "extraCssText": "box-shadow: 0 6px 16px rgba(0,0,0,0.12);",
            "rich": {"c": {"fontSize": 12, "color": "#5b6b7a", "padding": [0,0,0,6]}},
        },
        "series": [{
            "type": "tree", 
            "data": [tree_data],
            "layout": "orthogonal",
            "orient": "vertical",
            "top": "10%",
            "left": "5%", 
            "bottom": "10%", 
            "right": "15%",
            "expandAndCollapse": True, 
            "initialTreeDepth": -1,
            "animationDuration": 300, 
            "animationDurationUpdate": 300,
            "nodeGap": 100,
            "emphasis": {
                    "focus": "descendant",
                    "itemStyle": {"borderColor": "#0072FF", "borderWidth": 1},
                    "label": {"fontSize": 12, "fontWeight": "bold"},
                },
            "edgeShape": "polyline",
            "edgeForkPosition": "100%",
            "symbol": "Rect", 
            "symbolSize": [155, 30],
            "itemStyle": {
                "borderColor": "#E0E7EF", 
                "borderWidth": 1, 
                "color": "#FFFFFF",
                "shadowBlur": 3, 
                "shadowColor": "rgba(0,0,0,0.06)"
                },
            "label": {
                "show": True, 
                "position": "inside", 
                "verticalAlign": "middle",
                "overflow": "break",
                "align": "center", 
                "fontSize": 12, 
                "fontWeight": "bold",
                "lineHeight": 1, 
                "color": "#213547"
                },
            "lineStyle": {
                "color": "#AAB4C3",
                "curveness": 0.0,
                },
            "leaves": {
                "label": {
                    "position": "inside", 
                    "align": "center",
                    "fontSize": 12,
                    }
                    }
            
        }],
    }

def render_upward_graph(
    chain: List[Union[SimpleNode, RichNode]], height: int = 680, title: Optional[str] = None
) -> None:
    if not chain:
        st.info("No org data available for this selection."); return
    data, bu_colors = _to_tree_data_and_legend(chain)
    if title: st.markdown(f"**{title}**")
    _render_legend(bu_colors)
    st_echarts(options=_base_option(data), height=f"{height}px", key=f"org_up_{hash(str(chain))}")

# ---------- New: full tree renderer (CEO→…→persona + reportees) ----------
def _collect_bus(node: dict, acc: List[str]):
    bu = node.get("bu", "") or node.get("business_unit", "")
    if bu: acc.append(bu)
    for ch in node.get("children", []):
        _collect_bus(ch, acc)

def _apply_colors(node: dict, bu_colors: Dict[str, str]):
    bu = node.get("bu", "") or node.get("business_unit", "")
    if bu in bu_colors:
        node.setdefault("itemStyle", {})
        node["itemStyle"].update({"color": "#FFFFFF", "borderColor": bu_colors[bu], "borderWidth": 2})
        node.setdefault("label", {"color": "#213547"})
    for ch in node.get("children", []):
        _apply_colors(ch, bu_colors)

def render_org_tree(tree_data: dict, height: int = 1520, title: Optional[str] = None) -> None:
    if not tree_data:
        st.info("No org data available for this selection."); return
    # Gather BUs and colorize nodes
    acc: List[str] = []
    _collect_bus(tree_data, acc)
    bu_colors = _assign_colors(acc)
    _apply_colors(tree_data, bu_colors)

    if title: st.markdown(f"**{title}**")
    _render_legend(bu_colors)
    st_echarts(options=_base_option(tree_data), height=f"{height}px", key=f"org_tree_{hash(str(tree_data))}")
