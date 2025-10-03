import streamlit as st
from streamlit_echarts import st_echarts
from collections import defaultdict

st.set_page_config(page_title="CEO → Person → Reportees (Focused Org)", layout="wide")
st.title("Focused Org: CEO → Selected Person → Direct Reports")

# ---------------- 1) Sample org (id, name, title, manager_id, dept) ----------------
people = [
    ("u1", "Alice Rodrigues", "Chief Executive Officer", None, "Exec"),
    ("u2", "Vikram Rao", "Chief Technology Officer", "u1", "Engineering"),
    ("u3", "Neha Sharma", "Chief Financial Officer", "u1", "Finance"),
    ("u4", "Martha Lewis", "Chief Operating Officer", "u1", "Operations"),
    ("u5", "Javier Gomez", "SVP, Platform Engineering", "u2", "Engineering"),
    ("u6", "Riya Iyer", "VP, Product Engineering", "u2", "Engineering"),
    ("u7", "Daniel Cho", "VP, Data & AI", "u2", "Engineering"),
    ("u8", "Priya Patel", "SVP, Corporate Finance", "u3", "Finance"),
    ("u9", "Owen Lee", "VP, FP&A", "u3", "Finance"),
    ("u10","Sara Khan", "VP, Global Operations", "u4", "Operations"),
    ("u11","Tom Becker", "VP, Customer Success Ops", "u4", "Operations"),
    ("u12","Yuki Tanaka", "Director, DevEx", "u5", "Engineering"),
    ("u13","Arjun Mehta", "Director, SRE", "u5", "Engineering"),
    ("u14","Sophia Wang", "Director, Web Apps", "u6", "Engineering"),
    ("u15","Rohit Singh", "Director, Mobile Apps", "u6", "Engineering"),
    ("u16","Anita George", "Director, ML Platform", "u7", "Engineering"),
    ("u17","Leo Martins", "Director, Data Platform", "u7", "Engineering"),
    ("u18","Mira Desai", "Director, Treasury", "u8", "Finance"),
    ("u19","Jonas Weber", "Director, Controllership", "u8", "Finance"),
    ("u20","Ishaan Kapoor", "Director, Planning", "u9", "Finance"),
    ("u21","Emily Davis", "Director, Logistics", "u10", "Operations"),
    ("u22","Hammad Noor", "Director, Quality & Safety", "u10", "Operations"),
    ("u23","Kavya Nair", "Director, Support Ops", "u11", "Operations"),
    ("u24","Nora Quinn", "Eng Manager – Tooling", "u12", "Engineering"),
    ("u25","Vikash Gupta", "Eng Manager – CI/CD", "u12", "Engineering"),
    ("u26","Wei Zhang", "SRE Manager – Reliability", "u13", "Engineering"),
    ("u27","Irene Costa", "SRE Manager – Resilience", "u13", "Engineering"),
    ("u28","Aditya Rao", "PM – Web Core", "u14", "Engineering"),
    ("u29","Ria Banerjee", "Eng Manager – Web FE", "u14", "Engineering"),
    ("u30","Kevin Brown", "Eng Manager – Android", "u15", "Engineering"),
    ("u31","Zainab Ali", "Eng Manager – iOS", "u15", "Engineering"),
    ("u32","Martin Silva", "Manager – MLOps", "u16", "Engineering"),
    ("u33","Harini Krish", "Manager – Feature Engg", "u16", "Engineering"),
    ("u34","Diego Perez", "Mgr – Ingestion", "u17", "Engineering"),
    ("u35","Olivia White", "Mgr – Warehousing", "u17", "Engineering"),
    ("u36","Raj Malhotra", "Manager – Treasury Ops", "u18", "Finance"),
    ("u37","Hiro Aoki", "Manager – Reporting", "u19", "Finance"),
    ("u38","Megan Ross", "Manager – Budgeting", "u20", "Finance"),
    ("u39","Violet Kim", "Manager – Route Planning", "u21", "Operations"),
    ("u40","Sanjay Kumar", "Manager – Vendor Mgmt", "u21", "Operations"),
    ("u41","Elena Petrova", "Manager – QA Audits", "u22", "Operations"),
    ("u42","Arman Singh", "Manager – Escalations", "u23", "Operations"),
    ("u43","Devon Hart", "Senior Engineer", "u29", "Engineering"),
    ("u44","Aisha Khan", "Engineer II", "u29", "Engineering"),
    ("u45","Ruben Diaz", "Engineer I", "u29", "Engineering"),
    ("u46","Pooja S", "Engineer I", "u29", "Engineering"),
    ("u47","Nikhil P", "Engineer I", "u29", "Engineering"),
]

DEPT_COLOR = {"Exec":"#FF6B6B","Engineering":"#45B7D1","Finance":"#FFD166","Operations":"#06D6A0"}
DEFAULT_COLOR = "#778DA9"

# ---------------- 2) Indexes ----------------
by_id = {pid: {"id":pid,"name":name,"title":title,"mgr":mgr,"dept":dept} for pid,name,title,mgr,dept in people}
name_to_id = {f"{p['name']} ({p['title']})": pid for pid,p in by_id.items()}
children_map = defaultdict(list)
for pid,p in by_id.items():
    if p["mgr"] is not None:
        children_map[p["mgr"]].append(pid)

def label(p): return f"{p['name']}\n{p['title']}"

def root_of(pid):
    """Walk to the top-most ancestor (CEO)."""
    cur = pid
    while by_id[cur]["mgr"] is not None:
        cur = by_id[cur]["mgr"]
    return cur

def path_root_to(pid):
    """Return list of IDs from CEO to pid."""
    chain = [pid]
    while by_id[chain[-1]]["mgr"] is not None:
        chain.append(by_id[chain[-1]]["mgr"])
    return list(reversed(chain))

def focused_tree(selected_id):
    """
    Build ONLY:
      CEO → ... → selected
      and selected's direct reports
    """
    chain = path_root_to(selected_id)

    def build_at_index(i):
        node_id = chain[i]
        p = by_id[node_id]
        node = {
            "name": label(p),
            "itemStyle": {"color": DEPT_COLOR.get(p["dept"], DEFAULT_COLOR)}
        }

        # continue the spine towards the selected person
        if i + 1 < len(chain):
            node["children"] = [build_at_index(i+1)]

        # at the selected node: add direct reports (only)
        if node_id == selected_id:
            node["itemStyle"].update({"borderColor":"#222","borderWidth":3})
            drs = children_map.get(node_id, [])
            if drs:
                extra = [{
                    "name": label(by_id[c]),
                    "itemStyle": {"color": DEPT_COLOR.get(by_id[c]["dept"], DEFAULT_COLOR)}
                } for c in drs]
                node["children"] = (node.get("children", []) + extra)

        return node

    # start at CEO (chain[0])
    return build_at_index(0)

# ---------------- 3) UI ----------------
with st.sidebar:
    st.header("Pick a person")
    sel_name = st.selectbox("Person", sorted(name_to_id.keys()))
    node_w = st.slider("Node width", 120, 240, 170, 10)
    node_h = st.slider("Node height", 40, 80, 54, 2)
    fork_pos = st.slider("Edge fork position (%)", 45, 80, 63, 1)

selected_id = name_to_id[sel_name]
tree_data = focused_tree(selected_id)

# ---------------- 4) ECharts options ----------------
options = {
    "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
    "series": [{
        "type": "tree",
        "layout": "orthogonal",
        "orient": "vertical",
        "data": [tree_data],

        # tighter outer margins
        "top": "2%", "left": "2%", "bottom": "2%", "right": "2%",

        # smaller node boxes
        "symbol": "rect",
        "symbolSize": [120, 40],                # ↓ width, height

        # bring siblings closer horizontally
        "edgeForkPosition": "30%",              # was f"{fork_pos}%"

        "label": {
            "position": "inside",
            "align": "center",
            "verticalAlign": "middle",
            "color": "#ffffff",
            "fontSize": 11,                     # ↓ text size also shrinks boxes
            "padding": [2, 6, 2, 6]             # ↓ inner padding
        },
        "lineStyle": {"width": 2, "color": "#A3A3A3"},
        "expandAndCollapse": True,
        "initialTreeDepth": 20,
        "animationDuration": 400,
        "edgeShape": "polyline",
    }]
}

st_echarts(options=options, height="580px")
st.caption("This view shows only the CEO→…→selected path, plus the selected person's direct reportees.")
