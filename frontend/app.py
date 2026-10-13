import streamlit as st

st.set_page_config(page_title="Federated Hospital Collaborative Platform", page_icon="🌐", layout="centered", initial_sidebar_state="collapsed")

# Inject CSS to completely hide the sidebar toggle button and polish the UI
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        [data-testid="stDeployButton"] {display: none;}
        
        /* Subtle Animated Gradient Background */
        .stApp {
            background: linear-gradient(-45deg, rgba(30, 144, 255, 0.05), rgba(0, 255, 127, 0.05), rgba(138, 43, 226, 0.05), rgba(30, 144, 255, 0.05));
            background-size: 400% 400%;
            animation: gradientBG 20s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        div[data-testid="stButton"] > button {
            height: 60px;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Add top padding
st.write("")
st.write("")
st.write("")

st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Federated Hospital Collaborative Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 1.2rem; margin-bottom: 4rem;'>Secure Collaborative Diagnosis Network</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    if st.button("🏥 Local Hospital", use_container_width=True, type="primary"):
        st.switch_page("pages/1_🏥_Local_Dashboard.py")

with col2:
    if st.button("🛡️ Admin", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🌍_Global_Analytics.py")

with col3:
    if st.button("🩺 Live Diagnosis", use_container_width=True, type="primary"):
        st.switch_page("pages/3_🩺_Live_Diagnostics.py")
