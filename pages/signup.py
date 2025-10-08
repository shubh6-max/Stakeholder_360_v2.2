# pages/signup.py
import re
import time
import streamlit as st
from typing import List

from utils.layout import apply_global_style
from utils.auth import signup_user, is_authenticated

from utils.page_config import set_common_page_config
set_common_page_config(page_title="Sign Up | Stakeholder 360", layout="centered")
# st.set_page_config(page_title="Sign Up | Stakeholder 360", layout="centered")
apply_global_style()

# ---------- Validators ----------
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

WEAK_PASSWORDS = {
    "password", "Password", "Password1", "passw0rd", "123456", "12345678",
    "qwerty", "letmein", "welcome", "admin", "iloveyou", "abc123"
}

def validate_password(pw: str) -> List[str]:
    """Return a list of unmet rules (empty list means strong)."""
    rules_failed = []
    if len(pw) < 8:
        rules_failed.append("• At least 8 characters")
    if not re.search(r"[A-Z]", pw):
        rules_failed.append("• At least one uppercase letter (A-Z)")
    if not re.search(r"[a-z]", pw):
        rules_failed.append("• At least one lowercase letter (a-z)")
    if not re.search(r"[0-9]", pw):
        rules_failed.append("• At least one digit (0-9)")
    if not re.search(r"[^\w]", pw):  # special char
        rules_failed.append("• At least one special character (!@#$...)")
    if pw in WEAK_PASSWORDS:
        rules_failed.append("• Avoid very common passwords")
    return rules_failed

# ---------- Already authenticated? ----------
if is_authenticated():
    st.info("You're already logged in.")
    st.switch_page("pages/main_app.py")

# ---------- UI ----------
st.markdown(
    """
    <div class="brand-box">
      <img class="brand-logo"
           src="https://upload.wikimedia.org/wikipedia/commons/8/88/MathCo_Logo.png"
           alt="MathCo Logo" />
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="center-text" style="font-size:28px; font-weight:bold; margin-bottom:20px;">
        Welcome, Create an account
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Simple inputs (no form) ----
col1,col2,col3=st.columns([1,3,1])
with col2:
    first_name = st.text_input("First Name", key="signup_first_name",placeholder="First Name",label_visibility="collapsed")

col4,col5,col6=st.columns([1,3,1])
with col5:
    last_name  = st.text_input("Last Name", key="signup_last_name",placeholder="Last Name",label_visibility="collapsed")

col7,col8,col9=st.columns([1,3,1])
with col8:
    email= st.text_input("Email", placeholder="you@company.com", key="signup_email",label_visibility="collapsed")

col10,col11,col12=st.columns([1,3,1])
with col11:
    pw_col1, pw_col2 = st.columns(2)
    with pw_col1:
        password = st.text_input("Password", type="password", placeholder="Min 8 characters", key="signup_password",label_visibility="collapsed")
    with pw_col2:
        confirm  = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_confirm",label_visibility="collapsed")

# ---- Actions: two centered buttons ----
left, center, right = st.columns([1, 2, 1])
with center:
    b1, b2 = st.columns([1, 1])
    with b1:
        create_account = st.button("**Create account**", key="signup_submit", use_container_width=True)
    with b2:
        back_to_login = st.button("**Back to Login**", key="back_to_login", use_container_width=True)

# ---- Processing ----
if create_account:
    if not (first_name and last_name and email and password and confirm):
        st.error("Please fill in all fields.")
    elif not EMAIL_RE.match(email.strip()):
        st.error("Please enter a valid email address.")
    elif password != confirm:
        st.error("Passwords do not match.")
    else:
        pw_issues = validate_password(password)
        if pw_issues:
            st.error("Please choose a stronger password:\n\n" + "\n".join(pw_issues))
        else:
            try:
                signup_user(first_name.strip(), last_name.strip(), email.strip().lower(), password)
                st.success("✅ Account created successfully! Redirecting to login…")
                time.sleep(0.5)
                st.switch_page("pages/login.py")
            except ValueError as ve:
                st.warning(str(ve))  # e.g. duplicate email
            except Exception:
                st.error("Something went wrong while creating your account. Please try again.")

if back_to_login:
    st.switch_page("pages/login.py")
