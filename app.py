# import streamlit as st
# from utils.layout import apply_global_style

# st.set_page_config(page_title="Stakeholder 360", layout="centered")

# # Apply global styles
# apply_global_style()

# # Page-specific styles
# st.markdown(
#     """
#     <style>
#     /* Logo card */
#     .logo-card {
#         max-width: 500px;
#         padding: 20px;
#         border-radius: 20px;
#         background: white;
#         box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
#         margin: 100px auto 0 auto;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#     }
#     .logo-card img {
#         width: 100%;
#         max-width: 300px;
#         height: auto;
#     }

#     /* Title & subtitle */
#     .welcome-title {
#         font-size: 58px;
#         font-weight: bold;
#         margin: 30px 0;
#         color: #333;
#     }
#     .welcome-subtitle {
#         font-size: 30px;
#         margin-bottom: 0px;
#         color: #555;
#     }
#     </style>

#     <div class="center-container">
#         <div class="logo-card">
#             <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/MathCo_Logo.png" />
#         </div>
#         <div class="welcome-title">Stakeholder 360</div>
#         <div class="welcome-subtitle">Please login or sign up to continue</div>
#     """,
#     unsafe_allow_html=True
# )

# # Buttons
# if st.button("🔐 Login"):
#     st.switch_page("pages/login.py")

# if st.button("📝 Sign Up"):
#     st.switch_page("pages/signup.py")

# st.markdown("</div>", unsafe_allow_html=True)


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

st.markdown(
    """
    <div class="center-text" style="font-size:28px; font-weight:bold;">
        <h2>➜] Sign in</h2>
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
