# utils/layout.py
from pathlib import Path
import streamlit as st

def apply_global_style() -> None:
    """
    Inject global CSS from asset/styles.css (read as UTF-8) and keep
    the app from crashing on Windows encoding issues.
    """
    css = ""
    css_path = Path(__file__).resolve().parents[1] / "asset" / "styles.css"

    try:
        css = css_path.read_text(encoding="utf-8")           # preferred
    except UnicodeDecodeError:
        # handle files saved with BOM or odd characters
        try:
            css = css_path.read_text(encoding="utf-8-sig")
        except Exception:
            css = css_path.read_bytes().decode("utf-8", "ignore")  # last resort
    except FileNotFoundError:
        # optional: no CSS file, just continue
        css = ""

    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_topbar(user: dict):
    """Minimal top bar: right-aligned name badge only."""
    first = user.get("first_name", "User")
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-spacer"></div>
          <span class="name-badge">Hi, {first}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
