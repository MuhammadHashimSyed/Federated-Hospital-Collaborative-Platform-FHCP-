import streamlit as st
import requests

st.set_page_config(page_title="Live Diagnostics", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        [data-testid="stDeployButton"] {display: none;}

        .stTextInput > div > div > input { border-radius: 6px; }

        div[data-testid="stButton"] > button[kind="primary"] {
            height: 50px;
            font-size: 16px !important;
            font-weight: 600 !important;
            border-radius: 8px;
        }

        /* Clinical report card */
        .report-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-top: 1rem;
        }

        .finding-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #2d3748;
        }

        .confidence-badge {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fc8181;
        }

        .healthy-badge {
            color: #68d391;
        }

        .verdict-sick {
            background: linear-gradient(90deg, #742a2a, #2d1515);
            border-left: 4px solid #fc8181;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }

        .verdict-healthy {
            background: linear-gradient(90deg, #1a4731, #0f2d1e);
            border-left: 4px solid #68d391;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000/api"

if 'diag_jwt_token' not in st.session_state:
    st.session_state.diag_jwt_token = None

# ─── LOGIN GATE ────────────────────────────────────────────────────────────────
if not st.session_state.diag_jwt_token:
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🏠 Return Home", use_container_width=True):
            st.switch_page("app.py")

    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center;'>🩺 Live Diagnostic Tool</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 3rem;'>Physician Authentication required for inference.</p>", unsafe_allow_html=True)

    colA, colB, colC = st.columns([1, 1.5, 1])
    with colB:
        st.markdown("### Secure Login")
        username = st.text_input("Physician ID (Username)")
        password = st.text_input("Access Passcode (Password)", type="password")

        st.write("")
        if st.button("Authenticate & Access", type="primary", use_container_width=True):
            resp = requests.post(f"{API_BASE_URL}/auth/token/", data={"username": username, "password": password})
            if resp.status_code == 200:
                token = resp.json()['access']
                st.session_state.diag_jwt_token = token
                st.rerun()
            else:
                st.error("Invalid credentials. Please verify your Physician ID and Passcode.")

# ─── AUTHENTICATED VIEW ────────────────────────────────────────────────────────
else:
    col_nav1, col_space, col_logout = st.columns([2, 6, 1])
    with col_nav1:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
    with col_logout:
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state.diag_jwt_token = None
            st.rerun()

    st.markdown("<h1>🩺 Live Diagnostic Tool</h1>", unsafe_allow_html=True)
    st.markdown("Upload a Chest X-Ray to run inference through the **Federated DenseNet121 Model**.")
    st.write("")

    headers = {"Authorization": f"Bearer {st.session_state.diag_jwt_token}"}

    col_upload, col_results = st.columns([1, 1.4])

    with col_upload:
        uploaded_file = st.file_uploader("Choose a Chest X-Ray image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

            if st.button("🔬 Run Diagnostic Inference", type="primary", use_container_width=True):
                with st.spinner("Running DenseNet121 federated model inference…"):
                    uploaded_file.seek(0)
                    files = {'image': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/diagnostics/predict/",
                            headers=headers,
                            files=files
                        )
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot reach backend. Make sure the Django server is running.")
                        st.stop()

                try:
                    err_detail = resp.json().get('detail', resp.text)
                except Exception:
                    err_detail = resp.text[:200]

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state['last_predictions'] = data
                    st.rerun()
                elif resp.status_code == 503:
                    st.error(f"⚠️ Model not ready: {err_detail}")
                else:
                    st.error(f"Inference failed (HTTP {resp.status_code}): {err_detail}")

    with col_results:
        if 'last_predictions' in st.session_state:
            data = st.session_state['last_predictions']
            predictions = data.get('predictions', {})
            threshold = data.get('threshold', 0.69)
            total = data.get('total_findings', 0)

            st.markdown("### 📋 Clinical Diagnostic Report")

            # ── Overall Verdict ──────────────────────────────────────────────
            if total == 0:
                st.markdown("""
                <div class='verdict-healthy'>
                    <h3 style='color:#68d391; margin:0;'>✅ No Significant Pathology Detected</h3>
                    <p style='color:#a0aec0; margin:0.3rem 0 0;'>
                        No findings exceeded the 69% confidence threshold.
                        The X-Ray appears within normal limits for the screened conditions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                pathology_list = ", ".join(predictions.keys())
                st.markdown(f"""
                <div class='verdict-sick'>
                    <h3 style='color:#fc8181; margin:0;'>⚠️ Pathology Detected — {total} Finding(s)</h3>
                    <p style='color:#feb2b2; margin:0.3rem 0 0;'>
                        {pathology_list}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── Per-Finding Detail ───────────────────────────────────────────
            if total > 0:
                st.write("")
                for label, prob in predictions.items():
                    pct = prob * 100
                    # Color gradient: 69–79 amber, 80–89 orange, 90+ red
                    if pct >= 90:
                        bar_color = "#fc4444"
                        icon = "🔴"
                    elif pct >= 80:
                        bar_color = "#f97316"
                        icon = "🟠"
                    else:
                        bar_color = "#f59e0b"
                        icon = "🟡"

                    st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                            <span style='font-size:1rem; font-weight:600;'>{icon} {label}</span>
                            <span style='font-size:1.1rem; font-weight:700; color:{bar_color};'>{pct:.1f}%</span>
                        </div>
                        <div style='background:#2d3748; border-radius:999px; height:10px; overflow:hidden;'>
                            <div style='width:{pct:.1f}%; height:100%; background:{bar_color};
                                        border-radius:999px; transition: width 0.4s ease;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                st.caption("🏥 This report is generated by a federated AI model trained across multiple hospital nodes. "
                           "Clinical judgment by a qualified radiologist is required before any treatment decision.")
            else:
                st.info("Upload an image and click **Run Diagnostic Inference** to see the clinical report here.")
