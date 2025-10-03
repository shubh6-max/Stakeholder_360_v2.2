# utils/auth.py
import os
import bcrypt
import jwt
import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from utils.db import get_engine
from dotenv import load_dotenv
import hashlib

# Ensure .env is loaded even if this module imports first
load_dotenv()

JWT_ALGO = "HS256"


def _get_secret_key() -> str:
    """
    Load SECRET_KEY from env and guarantee a non-empty string.
    Raises RuntimeError if missing.
    """
    key = os.getenv("SECRET_KEY")
    if not key or not key.strip():
        raise RuntimeError("SECRET_KEY is missing or empty. Set it in your .env / deployment secrets.")
    return key


def secret_fingerprint() -> str:
    """
    Return a short, non-sensitive fingerprint of the SECRET_KEY to confirm both pages
    are using the same key. Safe to print in UI for debugging.
    """
    key = _get_secret_key()
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{h[:4]}…{h[-4:]}"


# -------------------------
# Password Hashing & Verify
# -------------------------
def hash_password(password: str) -> str:
    """Hash plain-text password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Check if plain password matches stored bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# -------------------------
# User Signup & Login
# -------------------------
def signup_user(first_name: str, last_name: str, email: str, password: str) -> None:
    """
    Register a new user with role='user'.
    Raises ValueError if email already exists.
    """
    engine = get_engine()
    hashed_pw = hash_password(password)

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scout.users (first_name, last_name, email, password_hash, role)
                    VALUES (:first_name, :last_name, :email, :password_hash, 'user')
                """),
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email.lower(),
                    "password_hash": hashed_pw,
                },
            )
    except IntegrityError:
        # email is UNIQUE in DB
        raise ValueError("Email already registered. Please log in.")


def login_user(email: str, password: str) -> bool:
    """
    Authenticate a user.
    On success: stores session_state with auth=True and user dict, updates last_login.
    Returns True if login successful, False otherwise.
    """
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT id, first_name, last_name, email, password_hash, role
                FROM scout.users
                WHERE email = :email
            """),
            {"email": email.lower()},
        ).fetchone()

    if row and verify_password(password, row.password_hash):
        # Save session
        st.session_state["auth"] = True
        st.session_state["user"] = {
            "id": row.id,
            "first_name": row.first_name,
            "last_name": row.last_name,
            "email": row.email,
            "role": row.role,
        }

        # Update last_login timestamp (column exists per Option A)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE scout.users SET last_login = now() WHERE id = :id"),
                {"id": row.id},
            )

        return True
    return False


def logout():
    """Clear user session."""
    for k in ["auth", "user"]:
        if k in st.session_state:
            del st.session_state[k]


def is_authenticated() -> bool:
    """Check if a user is logged in."""
    return st.session_state.get("auth", False)


# -------------------------
# JWT Helpers (cross-tab)
# -------------------------
def issue_jwt(user: dict, expiry_minutes: int = 120) -> str:
    """Generate JWT for user with expiry (default 120 minutes)."""
    secret = _get_secret_key()
    payload = {
        "sub": str(user["id"]),        # <-- must be a STRING
        "role": user["role"],
        "iat": datetime.utcnow(),      # issued at (optional but good)
        "exp": datetime.utcnow() + timedelta(minutes=expiry_minutes),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGO)


# utils/auth.py  — PATCH these two functions

def verify_jwt(token: str) -> Optional[dict]:
    """Verify JWT and return decoded payload, or None if invalid/expired."""
    try:
        secret = _get_secret_key()
        # 30s leeway to tolerate minor clock skew
        return jwt.decode(token, secret, algorithms=[JWT_ALGO], leeway=30)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


from typing import Tuple, Optional

def verify_jwt_verbose(token: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Return (claims, reason). `reason` is a human-readable string that includes
    the exception type and a short message for debugging.
    """
    try:
        secret = _get_secret_key()
        data = jwt.decode(token, secret, algorithms=[JWT_ALGO], leeway=30)
        return data, None
    except jwt.ExpiredSignatureError as e:
        return None, f"expired: {e.__class__.__name__}"
    except jwt.InvalidTokenError as e:
        # Includes DecodeError, InvalidSignatureError, ImmatureSignatureError, etc.
        # Safe to show exception type; avoids leaking secrets.
        return None, f"invalid: {e.__class__.__name__} - {str(e)}"
    except Exception as e:
        return None, f"decode_error: {e.__class__.__name__} - {str(e)}"


# -------------------------
# Page Guard
# -------------------------
def require_auth(role: str | None = None):
    """
    Guard function for Streamlit pages.
    Redirects to login if user not authenticated or role mismatch.
    """
    if not is_authenticated():
        st.warning("⚠️ You must log in first.")
        st.switch_page("pages/login.py")
        st.stop()

    if role and st.session_state["user"]["role"] != role:
        st.error("🚫 You are not authorized to view this page.")
        st.stop()
