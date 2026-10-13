import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import json

st.set_page_config(page_title="Admin Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        /* Input styling */
        .stTextInput > div > div > input { border-radius: 6px; }
        [data-testid="stDeployButton"] {display: none;}

        /* Main page primary button */
        div[data-testid="stMainBlockContainer"] div[data-testid="stButton"] > button[kind="primary"] {
            height: 50px;
            font-size: 16px !important;
            font-weight: 600 !important;
            border-radius: 8px;
        }

        /* Sidebar — remove all default button chrome */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            display: block;
            width: 100% !important;
            text-align: left !important;
            background: transparent !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 1rem !important;
            font-size: 14.5px !important;
            font-weight: 500 !important;
            color: inherit !important;
            transition: background 0.18s ease;
            box-shadow: none !important;
            height: auto !important;
            margin: 2px 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(128,128,128,0.13) !important;
        }
        /* Active nav item */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: rgba(99,102,241,0.15) !important;
            color: #818cf8 !important;
            font-weight: 600 !important;
            border-left: 3px solid #818cf8 !important;
            padding-left: calc(1rem - 3px) !important;
        }
    </style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000/api"

if 'admin_jwt_token' not in st.session_state:
    st.session_state.admin_jwt_token = None
if 'admin_page' not in st.session_state:
    st.session_state.admin_page = "metrics"

# ── LOGIN SCREEN ──────────────────────────────────────────────────────────────
if not st.session_state.admin_jwt_token:
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🏠 Return Home", use_container_width=True):
            st.switch_page("app.py")

    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center;'>🛡️ Admin Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 3rem;'>Central Administration Authentication.</p>", unsafe_allow_html=True)

    colA, colB, colC = st.columns([1, 1.5, 1])
    with colB:
        st.markdown("### Secure Login")
        username = st.text_input("Admin ID (Username)")
        password = st.text_input("Admin Passcode (Password)", type="password")
        st.write("")
        if st.button("Authenticate & Access", type="primary", use_container_width=True):
            resp = requests.post(f"{API_BASE_URL}/auth/token/", data={"username": username, "password": password})
            if resp.status_code == 200:
                st.session_state.admin_jwt_token = resp.json()['access']
                st.rerun()
            else:
                st.error("Invalid credentials. Please verify your Admin ID and Passcode.")

# ── AUTHENTICATED VIEW ────────────────────────────────────────────────────────
else:
    headers = {"Authorization": f"Bearer {st.session_state.admin_jwt_token}"}

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛡️ Admin Panel")
        st.markdown("---")

        nav_items = [
            ("metrics",   "📊 Performance Metrics & Controls"),
            ("hospitals", "🏥 Registered Hospitals"),
            ("history",   "📜 Round History"),
            ("logs",      "📝 System Logs"),
        ]

        for key, label in nav_items:
            btn_type = "primary" if st.session_state.admin_page == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.admin_page = key
                st.rerun()

        st.markdown("---")
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state.admin_jwt_token = None
            st.rerun()

    # ── PAGE: PERFORMANCE METRICS & CONTROLS ─────────────────────────────────
    if st.session_state.admin_page == "metrics":
        st.markdown("<h1>📊 Performance Metrics & Controls</h1>", unsafe_allow_html=True)
        st.write("")

        colA, colB = st.columns([2, 1])

        with colB:
            st.markdown("### Round Controls")
            resp_cur = requests.get(f"{API_BASE_URL}/rounds/current/", headers=headers)
            if resp_cur.status_code == 200:
                cur = resp_cur.json()
                st.markdown(f"**Active Round:** Round {cur['round_number']} — `{cur['status']}`")
                resp_h = requests.get(f"{API_BASE_URL}/analytics/hospitals/", headers=headers)
                total_hospitals = len(resp_h.json()) if resp_h.status_code == 200 else 0
                st.metric("Registered Hospitals", total_hospitals)
            else:
                st.info("No active round.")

            st.write("")
            if st.button("▶️ Initialize Next Training Round", type="primary", use_container_width=True):
                resp = requests.post(f"{API_BASE_URL}/rounds/start/", headers=headers)
                if resp.status_code == 201:
                    st.success(f"✅ Started Round {resp.json()['round_number']}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.text}")

            st.markdown("---")
            st.markdown("### FedAvg Aggregation")
            st.caption("Manually trigger Federated Averaging once hospitals have submitted weights.")
            if st.button("Run FedAvg & Aggregate Global Model", type="primary", use_container_width=True):
                with st.spinner("Running Federated Averaging algorithm..."):
                    resp_agg = requests.post(f"{API_BASE_URL}/rounds/aggregate/", headers=headers)
                if resp_agg.status_code == 200:
                    result = resp_agg.json()
                    st.success(f"✅ {result['detail']}")
                    st.metric("New Global Mean AUROC", f"{result['global_mean_auroc']:.4f}")
                    st.info(f"Aggregated **{result['submissions_aggregated']}** hospital submission(s).")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    try:
                        st.warning(resp_agg.json().get('detail', 'Aggregation failed.'))
                    except:
                        st.error(f"Error: {resp_agg.text}")

            st.markdown("---")
            st.markdown("### ⚠️ Danger Zone")
            if st.button("Reset Platform (Start from Round 1)", type="secondary", use_container_width=True):
                resp = requests.post(f"{API_BASE_URL}/rounds/reset/", headers=headers)
                if resp.status_code == 200:
                    st.success("Platform has been successfully reset! History cleared.")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"Failed to reset: {resp.text}")

        with colA:
            if st.button("🔄 Refresh Metrics"):
                st.rerun()

            resp = requests.get(f"{API_BASE_URL}/analytics/metrics/", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data)
                    df_plot = df.dropna(subset=['global_mean_auroc'])
                    if not df_plot.empty:
                        fig = px.line(df_plot, x="round_number", y="global_mean_auroc",
                                      title="Global Mean AUROC Progress", markers=True)
                        fig.update_layout(xaxis_title="Round", yaxis_title="Mean AUROC")
                        st.plotly_chart(fig, use_container_width=True)

                    latest_auroc = data[-1].get('global_mean_auroc')
                    latest_auroc_str = f"{latest_auroc:.4f}" if latest_auroc is not None else "Pending..."
                    st.metric(label="Latest Global Mean AUROC", value=latest_auroc_str)

                    global_metrics = data[-1].get('global_metrics', {})
                    if global_metrics:
                        with st.expander("View Global Pathology Breakdown", expanded=True):
                            st.table(pd.DataFrame([global_metrics], index=["Global AUROC Score"]))
                else:
                    st.info("No completed rounds yet. Initialize a round and wait for hospitals to submit weights.")
            else:
                st.error("Failed to load metrics.")

    # ── PAGE: REGISTERED HOSPITALS ────────────────────────────────────────────
    elif st.session_state.admin_page == "hospitals":
        st.markdown("<h1>🏥 Registered Hospitals</h1>", unsafe_allow_html=True)
        st.write("")

        with st.expander("➕ Register New Hospital Node", expanded=True):
            rCol1, rCol2, rCol3 = st.columns(3)
            with rCol1:
                new_username = st.text_input("Login Username", key="reg_username")
            with rCol2:
                new_hospital_name = st.text_input("Hospital Display Name", key="reg_hospital_name")
            with rCol3:
                new_password = st.text_input("Set Password", type="password", key="reg_password")

            if st.button("Register Hospital Node", type="primary", use_container_width=True):
                if not new_username or not new_password:
                    st.warning("Username and password are required.")
                else:
                    resp_reg = requests.post(
                        f"{API_BASE_URL}/hospitals/register/",
                        headers=headers,
                        data={"username": new_username, "password": new_password, "hospital_name": new_hospital_name}
                    )
                    if resp_reg.status_code == 201:
                        st.success(f"✅ {resp_reg.json()['detail']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(resp_reg.json().get('detail', 'Registration failed.'))

        st.markdown("---")
        st.markdown("### Currently Registered Hospitals")
        resp = requests.get(f"{API_BASE_URL}/analytics/hospitals/", headers=headers)
        if resp.status_code == 200:
            hospitals = resp.json()
            if hospitals:
                df_hosp = pd.DataFrame(hospitals)
                if 'date_joined' in df_hosp.columns:
                    df_hosp['date_joined'] = pd.to_datetime(df_hosp['date_joined']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(df_hosp, use_container_width=True)
                st.metric("Total Active Hospital Nodes", len(hospitals))

                st.markdown("#### Remove Hospital Node")
                del_options = {f"{h.get('hospital_name') or h['username']} (ID: {h['id']})": h['id'] for h in hospitals}
                selected_del = st.selectbox("Select hospital to remove", list(del_options.keys()), key="del_hospital")
                if st.button("Delete Selected Hospital", type="secondary"):
                    del_id = del_options[selected_del]
                    resp_del = requests.delete(f"{API_BASE_URL}/hospitals/{del_id}/delete/", headers=headers)
                    if resp_del.status_code == 200:
                        st.success(f"✅ {resp_del.json()['detail']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(resp_del.json().get('detail', 'Deletion failed.'))
            else:
                st.info("No hospitals registered yet. Use the form above to add one.")
        else:
            st.error("Failed to load hospitals.")

    # ── PAGE: ROUND HISTORY ───────────────────────────────────────────────────
    elif st.session_state.admin_page == "history":
        st.markdown("<h1>📜 Round History</h1>", unsafe_allow_html=True)
        st.write("")

        resp = requests.get(f"{API_BASE_URL}/analytics/metrics/", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                for round_data in reversed(data):
                    auroc_val = round_data.get('global_mean_auroc')
                    auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "Pending"
                    with st.expander(f"Round {round_data['round_number']} — Global AUROC: {auroc_str}", expanded=(round_data == data[-1])):
                        st.write("**Participating Hospitals:**")
                        subs = round_data.get("local_submissions", [])
                        if subs:
                            for s in subs:
                                st.write(f"**Hospital Node:** {s.get('hospital', 'Unknown')} (Mean AUROC: {s.get('local_mean_auroc', 0):.4f})")
                                metrics_raw = s.get("metrics_json", {})
                                if isinstance(metrics_raw, str):
                                    try:
                                        metrics = json.loads(metrics_raw)
                                    except:
                                        metrics = {}
                                else:
                                    metrics = metrics_raw if metrics_raw else {}
                                if metrics:
                                    st.table(pd.DataFrame([metrics], index=["Pathology Score"]))
                        else:
                            st.write("No submissions recorded.")
            else:
                st.info("No history available.")
        else:
            st.error("Failed to load round history.")

    # ── PAGE: SYSTEM LOGS ─────────────────────────────────────────────────────
    elif st.session_state.admin_page == "logs":
        colL1, colL2 = st.columns([8, 2])
        with colL1:
            st.markdown("<h1>📝 System Logs</h1>", unsafe_allow_html=True)
        with colL2:
            st.write("")
            st.write("")
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        resp_logs = requests.get(f"{API_BASE_URL}/analytics/logs/", headers=headers)
        if resp_logs.status_code == 200:
            logs = resp_logs.json()
            if logs:
                df_logs = pd.DataFrame(logs)
                if 'timestamp' in df_logs.columns:
                    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                if set(['timestamp', 'actor', 'action', 'ip_address']).issubset(df_logs.columns):
                    df_logs = df_logs[['timestamp', 'actor', 'action', 'ip_address']]
                st.dataframe(df_logs, use_container_width=True, height=600)
            else:
                st.info("No system activity logged yet.")
        else:
            st.error("Failed to load system logs.")
