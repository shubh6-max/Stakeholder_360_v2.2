# utils/__init__.py
from .layout import apply_global_style
import streamlit as st

# Automatically apply styles when utils is imported
try:
    apply_global_style()
except Exception as e:
    st.warning(f"⚠️ Could not apply global styles: {e}")
