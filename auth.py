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
            "password": "$2b$12$pcdu1r7C2VZcYQoVYg5d/OseTSTIfh8RSBgdaqpXw8ewcnHALSkHW",  # neural2026
        },
        "guest": {
            "name":     "Guest User",
            "email":    "guest@neuralstudiox.ai",
            "password": "$2b$12$lOtMfRdVI3avF5HSZL9QBuVh0cEG5O3kg8nin0BoK8CWpuY.3LwOG",  # guest123
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
        auto_hash=False
    )



def render_login(authenticator) -> tuple[str | None, bool]:
    """
    Render login widget. Returns (username, is_authenticated).
    Handles new streamlit-authenticator API and adds a Quick Demo Login button.
    """
    if not HAS_AUTH or authenticator is None:
        return "guest", True

    # Check if already authenticated in session state
    auth_status = st.session_state.get("authentication_status")
    username    = st.session_state.get("username")

    if auth_status is True and username:
        return username, True

    # Render a Quick Demo Login button
    st.sidebar.markdown("### ⚡ Quick Access")
    if st.sidebar.button("🔑 Direct Demo Login (guest)", use_container_width=True):
        st.session_state["authentication_status"] = True
        st.session_state["username"] = "guest"
        st.session_state["name"] = "Guest User"
        st.rerun()

    st.sidebar.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)

    try:
        # For newer streamlit-authenticator versions, rendering location='sidebar' returns None
        # and sets st.session_state['authentication_status'], st.session_state['username']
        authenticator.login(
            location="sidebar",
            fields={
                "Form name": "🔐 Login to Studio X",
                "Username":  "Username",
                "Password":  "Password",
                "Login":     "Login"
            }
        )
        auth_status = st.session_state.get("authentication_status")
        username    = st.session_state.get("username")

        # If user just successfully logged in, trigger a rerun manually
        # to render the main app immediately (bypasses a streamlit-authenticator dict bug)
        if auth_status is True:
            st.rerun()
    except Exception:
        # Graceful fallback to guest if anything breaks
        return "guest", True

    if auth_status is False:
        st.sidebar.error("❌ Incorrect credentials.")
        st.sidebar.caption("Demo credentials:\n- User: `guest` | Pass: `guest123`\n- User: `himanshu` | Pass: `neural2026`")
        return None, False
    elif auth_status is None:
        st.sidebar.info("Please login or use Direct Demo Login.")
        st.sidebar.caption("Demo credentials:\n- User: `guest` | Pass: `guest123`")
        return None, False
    else:
        return username, True





def render_logout(authenticator, username: str):
    """Render logout button in sidebar."""
    if HAS_AUTH and authenticator:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 Logged in as **{username}**")
        authenticator.logout("🚪 Logout", location="sidebar")
