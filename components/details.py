# components/details.py
from __future__ import annotations
import difflib
import pandas as pd
import streamlit as st
from utils.db import get_engine

# ---- Icon set (your URLs) ----
ICON_NAME = "https://img.icons8.com/?size=100&id=99268&format=png&color=000000"
ICON_TITLE = "https://img.icons8.com/?size=100&id=109233&format=png&color=000000"
ICON_LOCATION = "https://img.icons8.com/?size=100&id=59830&format=png&color=000000"
ICON_LINKEDIN = "https://img.icons8.com/?size=100&id=Uj9DyJeLazL6&format=png&color=000000"

# ---- Data loaders ----
@st.cache_data(ttl=300, show_spinner=False)
def load_linkedin_clients() -> pd.DataFrame:
    cols = [
        "client_name",
        "client_avatar",
        "client_present_title",
        "client_city",
        "client_url",
    ]
    eng = get_engine()
    sql = f"SELECT {', '.join(cols)} FROM scout.linkedin_clients_data"
    with eng.begin() as conn:
        df = pd.read_sql(sql, conn)
    for c in cols:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str).str.strip()
    df["__norm_name"] = df["client_name"].str.lower()
    return df

def _initials(name: str) -> str:
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "?"
    return (parts[0][:1] + (parts[1][:1] if len(parts) > 1 else "")).upper()

def _find_profile(ldf: pd.DataFrame, person_name: str) -> dict | None:
    if not person_name:
        return None
    q = person_name.strip().lower()
    if not q:
        return None

    exact = ldf[ldf["__norm_name"] == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    contains = ldf[ldf["__norm_name"].str.contains(q, na=False)]
    if not contains.empty:
        contains = contains.assign(_len=(contains["__norm_name"].str.len()))
        return contains.sort_values("_len").iloc[0].to_dict()

    choices = ldf["__norm_name"].tolist()
    best = difflib.get_close_matches(q, choices, n=1, cutoff=0.82)
    if best:
        row = ldf[ldf["__norm_name"] == best[0]]
        if not row.empty:
            return row.iloc[0].to_dict()
    return None

# ---- Tiny HTML helpers ----
def _row(icon_url: str, text_html: str) -> str:
    if not text_html:
        return ""
    icon = (
        f'<img src="{icon_url}" width="16" height="16" '
        f'style="opacity:.85;flex:0 0 16px;" />'
        if icon_url.startswith("http") else ""
    )
    return f"""
      <div style="display:flex;gap:8px;align-items:center;margin:6px 0;">
        {icon}
        <div style="font-size:13px;color:#213547;line-height:1.35">{text_html}</div>
      </div>
    """

# ---- Public renderer ----
def render_avatar_only(
    person_name: str,
    *,
    height: int,           # required: px height for this row
    avatar_size: int | None = None,  # if None, auto-fit inside height
) -> None:
    """
    Row that shows only the avatar, centered. No card chrome.
    """
    ldf = load_linkedin_clients()
    prof = _find_profile(ldf, person_name) or {}
    name = prof.get("client_name") or person_name or "Unknown"
    avatar = (prof.get("client_avatar") or "").strip()

    # auto avatar size (keep a small margin)
    if avatar_size is None:
        avatar_size = max(64, min(280, height - 24))

    # wrapper = f"min-height:{height}px;display:flex;align-items:center;justify-content:center;"
    # st.markdown(f'<div style="{wrapper}">', unsafe_allow_html=True)

    if avatar.startswith("http"):
        st.markdown(
            f"""
            <div style="
                width:{avatar_size}px;height:{avatar_size}px;
                border-radius:14px;overflow:hidden;
                background:#f4f6fa;border:1px solid #e7ecf3;
            ">
              <img src="{avatar}" alt="avatar"
                   style="width:100%;height:100%;object-fit:cover;display:block;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        initials = _initials(name)
        st.markdown(
            f'''
            <svg width="{avatar_size}" height="{avatar_size}" xmlns="http://www.w3.org/2000/svg"
                 style="border-radius:14px;border:1px solid #e7ecf3;background:#E6F0FF;">
              <text x="50%" y="54%" text-anchor="middle"
                    font-size="{int(avatar_size*0.36)}" font-weight="700" fill="#21406A">{initials}</text>
            </svg>
            ''',
            unsafe_allow_html=True,
        )

    # st.markdown("</div>", unsafe_allow_html=True)


def render_info_only(
    person_name: str,
    *,
    height: int,                     # required: px height for this row
    show_card: bool = True,          # set False if you want zero chrome
    max_width: int = 520,
) -> None:
    """
    Row that shows only the textual info with icons, vertically centered.
    """
    ldf = load_linkedin_clients()
    prof = _find_profile(ldf, person_name) or {}

    name  = prof.get("client_name") or person_name or "Unknown"
    title = prof.get("client_present_title") or ""
    city  = prof.get("client_city") or ""
    url   = prof.get("client_url") or ""

    # outer = f"min-height:{height}px;display:flex;align-items:center;justify-content:center;"
    # st.markdown(f'<div style="{outer}">', unsafe_allow_html=True)
    st.markdown("---")

    card_open = ""
    card_close = ""
    if show_card:
        card_open = f"""
        <div style="
          width:100%;max-width:{max_width}px;
          background:#fff;border:1px solid #e6eaf0;border-radius:14px;
          padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06);
        ">
        """
        card_close = "</div>"

    # st.markdown(card_open, unsafe_allow_html=True)

    def _row(icon_url: str, text_html: str) -> str:
        if not text_html:
            return ""
        icon = (
            f'<img src="{icon_url}" width="16" height="16" style="opacity:.85;flex:0 0 16px;" />'
            if icon_url.startswith("http") else ""
        )
        return f"""
          <div style="display:flex;gap:10px;align-items:center;margin:7px 0;">
            {icon}
            <div style="font-size:13px;color:#213547;line-height:1.35">{text_html}</div>
          </div>
        """

    html = ""
    html += _row(ICON_NAME,   f"<strong>{name}</strong>")
    if title:
        html += _row(ICON_TITLE,  title)
    if city:
        html += _row(ICON_LOCATION, city)
    if url.startswith("http"):
        html += _row(
            ICON_LINKEDIN,
            f'<a href="{url}" target="_blank" style="color:#0a66c2;text-decoration:none;">Open LinkedIn</a>'
        )
    st.markdown(html, unsafe_allow_html=True)

    # st.markdown(card_close, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
