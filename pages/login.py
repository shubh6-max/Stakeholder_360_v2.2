# pages/login.py
import re
import time
import streamlit as st
from datetime import datetime, timedelta

from utils.layout import apply_global_style
from utils.auth import login_user, is_authenticated

st.set_page_config(page_title="Login | Stakeholder 360", layout="centered")
apply_global_style()

# ---- Helpers ----
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def _init_login_state():
    ss = st.session_state
    ss.setdefault("login_attempts", 0)
    ss.setdefault("login_lock_until", None)
    # keep input keys initialized so we can reliably read them
    ss.setdefault("login_email", "")
    ss.setdefault("login_password", "")

def _locked_message(lock_until: datetime) -> str:
    remaining = int((lock_until - datetime.utcnow()).total_seconds())
    mins = max(0, remaining // 60)
    secs = max(0, remaining % 60)
    return f"Too many failed attempts. Please try again in {mins}m {secs}s."

def _check_lockout() -> bool:
    lock_until = st.session_state.get("login_lock_until")
    if lock_until and datetime.utcnow() < lock_until:
        st.error(_locked_message(lock_until))
        return True
    return False

def _record_failed_attempt():
    st.session_state["login_attempts"] += 1
    if st.session_state["login_attempts"] >= 5:
        # lock for 5 minutes
        st.session_state["login_lock_until"] = datetime.utcnow() + timedelta(minutes=5)

def _reset_attempts_on_success():
    st.session_state["login_attempts"] = 0
    st.session_state["login_lock_until"] = None

# ---- Init ----
_init_login_state()

# ---- Already authenticated? ----
if is_authenticated():
    st.success("You're already logged in.")
    st.switch_page("pages/main_app.py")

# ---- UI (simple inputs, no st.form) ----
# brand / logo (keeps your brand-box CSS)
st.markdown(
    """
    <div class="brand-box" style="margin-bottom:8px;">
      <img class="brand-logo"
           src="https://upload.wikimedia.org/wikipedia/commons/8/88/MathCo_Logo.png"
           alt="MathCo Logo" />
    </div>
    """,
    unsafe_allow_html=True,
)

icon_url = "https://img.icons8.com/?size=100&id=GEeJqVN0aRrU&format=png&color=000000"

st.markdown(
    f"""
    <div class="center-text" style="font-size:40px; font-weight:bold;margin-bottom: 15px;">
      <img src="{icon_url}" width="45" height="45" style="vertical-align: middle; margin-right: 0px;">
      <span>Sign in</span>
    </div>
    """,
    unsafe_allow_html=True
)

locked = _check_lockout()

# Simple standalone inputs (not inside a form)
col1,col2,col3=st.columns([1,2,1])
with col2:
    email = st.text_input("Email", value=st.session_state.get("login_email", ""), placeholder="✉ Email", key="login_email",label_visibility="collapsed")

col4,col5,col6=st.columns([1,2,1])
with col5:
    password = st.text_input("Password", type="password", value=st.session_state.get("login_password", ""), placeholder="••••••••", key="login_password",label_visibility="collapsed")

# Actions row (two columns) — keeps alignment with your CSS .auth-actions# Actions row (two columns) — keeps alignment with your CSS .auth-actions
st.markdown("<div class='auth-actions'>", unsafe_allow_html=True)
# outer 3-column layout to center content
left, center, right = st.columns([1, 2, 1])

with center:
    # inner 2-column layout for side-by-side buttons
    colA, colB = st.columns([1, 1])

    with colA:
        do_login = st.button("**Login**", use_container_width=True)

    with colB:
        go_signup = st.button("**Create account**", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Button handling
if go_signup:
    # Navigate to signup as a real page switch
    st.switch_page("pages/signup.py")

elif do_login and not locked:
    # read values (ensure we read latest from session_state)
    email_val = (st.session_state.get("login_email") or "").strip().lower()
    password_val = st.session_state.get("login_password") or ""

    # Basic validations
    if not email_val or not EMAIL_RE.match(email_val):
        st.error("Please enter a valid email address.")
    elif not password_val:
        st.error("Please enter your password.")
    else:
        try:
            ok = login_user(email_val, password_val)
            if ok:
                _reset_attempts_on_success()
                st.success("✅ Login successful! Redirecting…")
                time.sleep(0.25)
                st.switch_page("pages/main_app.py")
            else:
                _record_failed_attempt()
                if _check_lockout():
                    pass  # lock message already shown
                else:
                    st.error("❌ Invalid email or password.")
        except Exception:
            st.error("Something went wrong while logging in. Please try again.")


# wrapper card end
st.markdown("</div>", unsafe_allow_html=True)
