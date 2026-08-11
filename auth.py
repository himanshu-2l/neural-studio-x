# ─────────────────────────────────────────────────────────────
# Neural Studio X — Auth Layer
# JWT-based session management via streamlit-authenticator
# ─────────────────────────────────────────────────────────────
import os
import streamlit as st

# Load env variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit_authenticator as stauth
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False


# Pre-hashed passwords (bcrypt). In production load from DB or env vars.
# To regenerate: stauth.Hasher(['your_password']).generate()
CREDENTIALS = {
    "usernames": {
        "himanshu": {
            "name":     "Himanshu",
            "email":    "himanshurathore212.2l@gmail.com",
            "password": "$2b$12$KDRaPlE/DLbVMc.Lm8TSH.hYxkHaH8AuMKzCHgVXiRQ0kxhRfvnDe",  # neural2026
        },
        "guest": {
            "name":     "Guest User",
            "email":    "guest@neuralstudiox.ai",
            "password": "$2b$12$FiVHFJgQ0X5X.1/2L8yHYegWe5V1FNGi.xE9Iz5u4mOAqDpEhEt.S",  # guest123
        }
    }
}

COOKIE_CONFIG = {
    "name":       "neural_studio_x_session",
    "key":        os.getenv("COOKIE_SECRET", "nsx-super-secret-key-change-in-prod"),
    "expiry_days": 1,
}


def build_authenticator():
    """Build and return a streamlit-authenticator Authenticate object."""
    if not HAS_AUTH:
        return None
    return stauth.Authenticate(
        CREDENTIALS,
        COOKIE_CONFIG["name"],
        COOKIE_CONFIG["key"],
        COOKIE_CONFIG["expiry_days"],
    )


def render_login(authenticator) -> tuple[str | None, bool]:
    """
    Render login widget. Returns (username, is_authenticated).
    Handles both old (tuple) and new (session_state) streamlit-authenticator APIs.
    Falls back gracefully if not installed.
    """
    if not HAS_AUTH or authenticator is None:
        return "guest", True

    try:
        result = authenticator.login(
            location="sidebar",
            fields={
                "Form name": "🔐 Neural Studio X Login",
                "Username":  "Username",
                "Password":  "Password",
                "Login":     "Login"
            }
        )
        # Older API returns (name, auth_status, username)
        if result is not None:
            name, auth_status, username = result
        else:
            # Newer API stores state in st.session_state
            auth_status = st.session_state.get("authentication_status")
            username    = st.session_state.get("username")
            name        = st.session_state.get("name")
    except Exception:
        # Ultimate fallback
        return "guest", True

    if auth_status is False:
        st.sidebar.error("❌ Incorrect username or password.")
        return None, False
    elif auth_status is None or not auth_status:
        st.sidebar.info("Enter credentials to access Neural Studio X.")
        return None, False
    else:
        return username, True



def render_logout(authenticator, username: str):
    """Render logout button in sidebar."""
    if HAS_AUTH and authenticator:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 Logged in as **{username}**")
        authenticator.logout("🚪 Logout", location="sidebar")
