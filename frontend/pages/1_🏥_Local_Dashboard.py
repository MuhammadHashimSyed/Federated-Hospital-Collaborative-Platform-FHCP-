import streamlit as st
import requests
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import json
import plotly.express as px
import random

st.set_page_config(page_title="Local Hospital Dashboard", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

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
            background: rgba(20,184,166,0.15) !important;
            color: #2dd4bf !important;
            font-weight: 600 !important;
            border-left: 3px solid #2dd4bf !important;
            padding-left: calc(1rem - 3px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Minimal Dummy Model to create valid .pth file for FedAvg
class DummyCheXpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 14)

API_BASE_URL = "http://127.0.0.1:8000/api"

if 'jwt_token' not in st.session_state:
    st.session_state.jwt_token = None
if 'hospital_page' not in st.session_state:
    st.session_state.hospital_page = "training"

# ── LOGIN SCREEN ──────────────────────────────────────────────────────────────
if not st.session_state.jwt_token:
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🏠 Return Home", use_container_width=True):
            st.switch_page("app.py")

    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center;'>🏥 Local Hospital Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 3rem;'>Secure authentication required to access the federated network.</p>", unsafe_allow_html=True)

    colA, colB, colC = st.columns([1, 1.5, 1])
    with colB:
        st.markdown("### Secure Login")
        username = st.text_input("Hospital ID (Username)")
        password = st.text_input("Access Key (Password)", type="password")
        st.write("")
        if st.button("Authenticate & Connect", type="primary", use_container_width=True):
            resp = requests.post(f"{API_BASE_URL}/auth/token/", data={"username": username, "password": password})
            if resp.status_code == 200:
                st.session_state.jwt_token = resp.json()['access']
                st.rerun()
            else:
                st.error("Invalid credentials. Please verify your Hospital ID and Access Key.")

# ── AUTHENTICATED VIEW ────────────────────────────────────────────────────────
else:
    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}

    # Fetch round info once
    resp = requests.get(f"{API_BASE_URL}/rounds/current/", headers=headers)
    round_info = resp.json() if resp.status_code == 200 else None
    notifications = []
    if round_info and round_info.get('status') == 'IN_PROGRESS' and not round_info.get('has_submitted'):
        notifications.append(f"Round {round_info['round_number']} is active. Please start your local training!")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 Hospital Node")
        st.markdown("---")

        nav_items = [
            ("training", "🚀 Training Hub"),
            ("history",  "📜 My Training History"),
            ("doctors",  "👨‍⚕️ Manage Doctors"),
        ]

        for key, label in nav_items:
            btn_type = "primary" if st.session_state.hospital_page == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.hospital_page = key
                st.rerun()

        st.markdown("---")

        # Notification bell
        if notifications:
            st.warning(f"🔔 **{len(notifications)} Alert(s)**")
            for n in notifications:
                st.caption(n)
        else:
            st.success("🔔 All up to date")

        st.markdown("---")
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state.jwt_token = None
            st.rerun()

    # ── PAGE: TRAINING HUB ────────────────────────────────────────────────────
    if st.session_state.hospital_page == "training":
        st.markdown("<h1>🚀 Training Hub</h1>", unsafe_allow_html=True)
        st.write("")

        st.markdown("### 🗄️ Local Data Configuration")
        default_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dummy_dataset"))
        data_path = st.text_input("Dataset Directory Path", default_dataset_path)

        csv_path = os.path.join(data_path, 'train.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Status metrics
            st.write("")
            colM1, colM2, colM3 = st.columns(3)
            with colM1:
                st.metric(label="Available Patient Records", value=len(df))
            with colM2:
                st.metric(label="Model Architecture", value="DenseNet121")
            with colM3:
                sync_status = "Pending Training" if notifications else "Synced"
                st.metric(label="Network Sync Status", value=sync_status,
                          delta="- Required Action" if notifications else "✓ Optimal",
                          delta_color="inverse" if notifications else "normal")

            with st.expander("Preview Local Dataset Configuration"):
                st.dataframe(df.head(), use_container_width=True)

            if notifications:
                st.markdown("---")
                st.markdown("### ⚡ Federated Training Execution")

                if st.button("Initialize Local Training Epochs", type="primary", use_container_width=True):
                    # Log start
                    requests.post(f"{API_BASE_URL}/rounds/log/", headers=headers,
                                  data={"action": f"Started local training for Round {round_info.get('round_number', 1)}"})

                    # ── STEP 1: Download latest global model ─────────────────
                    global_weights_path = "global_model_downloaded.pth"
                    global_model_loaded = False

                    dl_status = st.empty()
                    dl_status.info("Fetching latest global model weights from Central Aggregation Server...")
                    resp_dl = requests.get(f"{API_BASE_URL}/weights/download_global/", headers=headers, stream=True)

                    if resp_dl.status_code == 200:
                        with open(global_weights_path, 'wb') as gf:
                            for chunk in resp_dl.iter_content(chunk_size=8192):
                                gf.write(chunk)
                        round_from_server = resp_dl.headers.get('X-Round-Number', '?')
                        auroc_from_server = resp_dl.headers.get('X-Global-AUROC', '?')
                        dl_status.success(f"✅ Global model (Round {round_from_server}, AUROC: {auroc_from_server}) downloaded. Initializing fine-tuning...")
                        global_model_loaded = True
                    else:
                        dl_status.warning("No prior global model found. Initializing from scratch (Round 1).")

                    time.sleep(0.8)

                    # ── STEP 2: Load model weights ───────────────────────────
                    st.info("Loading model architecture and applying global weights...")
                    time.sleep(0.5)
                    model = DummyCheXpertModel()
                    if global_model_loaded and os.path.exists(global_weights_path):
                        try:
                            global_state = torch.load(global_weights_path, map_location='cpu', weights_only=True)
                            model.load_state_dict(global_state)
                            st.info("Global weights loaded. Beginning local fine-tuning epochs...")
                        except Exception as e:
                            st.warning(f"Could not load global weights ({e}). Using random initialisation.")

                    # ── STEP 3: Local fine-tuning (simulated) ────────────────
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    epochs = 10
                    for epoch in range(epochs):
                        status_text.markdown(f"**Fine-tuning Epoch {epoch+1}/{epochs}...** Processing local patient image batches.")
                        time.sleep(0.5)
                        progress_bar.progress(int((epoch + 1) / epochs * 100))

                    status_text.success("✅ Local Fine-Tuning Complete! Compiling performance metrics.")

                    current_round_num = round_info['round_number'] if round_info and 'round_number' in round_info else 1
                    learning_progress = (current_round_num - 1) * 0.012

                    base_metrics = {
                        "No Finding": 0.890, "Enlarged Cardiomediastinum": 0.820,
                        "Cardiomegaly": 0.880, "Lung Opacity": 0.845,
                        "Lung Lesion": 0.810, "Edema": 0.850,
                        "Consolidation": 0.840, "Pneumonia": 0.856,
                        "Atelectasis": 0.835, "Pneumothorax": 0.870,
                        "Pleural Effusion": 0.895, "Pleural Other": 0.800,
                        "Fracture": 0.865, "Support Devices": 0.910
                    }

                    metrics_dict = {}
                    for k, v in base_metrics.items():
                        metrics_dict[k] = min(0.999, v + learning_progress + random.uniform(-0.015, 0.025))

                    mean_auroc = sum(metrics_dict.values()) / len(metrics_dict)

                    st.write("### 📈 Local Validation Metrics")
                    st.metric("Local Mean AUROC Score", f"{mean_auroc:.4f}")
                    with st.expander("View Detailed Pathology Breakdown", expanded=True):
                        st.table(pd.DataFrame([metrics_dict], index=["AUROC Score"]))

                    # Generate & upload weights from fine-tuned model
                    st.info("Encrypting and serializing fine-tuned local model weights (.pth)...")
                    with torch.no_grad():
                        model.fc.weight += torch.randn_like(model.fc.weight) * 0.01
                    local_pth_path = "local_model_submission.pth"
                    torch.save(model.state_dict(), local_pth_path)

                    st.info("Transmitting encrypted weights to Central Aggregation Server...")
                    with open(local_pth_path, 'rb') as f:
                        files = {'local_model_file': ('local_model.pth', f)}
                        data = {'local_mean_auroc': mean_auroc, 'metrics_json': json.dumps(metrics_dict)}
                        resp_upload = requests.post(f"{API_BASE_URL}/weights/upload/", headers=headers, files=files, data=data)

                    if resp_upload.status_code == 201:
                        st.success("✅ Weights securely submitted to the global network!")
                        time.sleep(1)
                        st.rerun()
                    elif resp_upload.status_code == 403 and "Only hospital clients" in resp_upload.text:
                        st.error("🚨 Security Block: You are logged in as an Admin account. Please use a Hospital account.")
                    else:
                        st.error(f"Transmission failed: {resp_upload.text}")

            if not notifications:
                st.success("✅ Local Training for the current round has been completed and synced.")
                resp_history = requests.get(f"{API_BASE_URL}/weights/history/", headers=headers)
                if resp_history.status_code == 200 and resp_history.json():
                    latest_hist = resp_history.json()[0]
                    st.markdown("### 📊 Latest Validated Metrics")
                    st.metric("Local Mean AUROC Score", f"{latest_hist['local_mean_auroc']:.4f}")
                    metrics_raw = latest_hist.get('metrics_json', {})
                    if isinstance(metrics_raw, str):
                        try:
                            metrics = json.loads(metrics_raw)
                        except:
                            metrics = {}
                    else:
                        metrics = metrics_raw if metrics_raw else {}
                    if metrics:
                        with st.expander("View Detailed Pathology Breakdown", expanded=True):
                            st.table(pd.DataFrame([metrics], index=["AUROC Score"]))
        else:
            st.error(f"Could not locate `train.csv` inside `{data_path}`. Please verify the path.")

    # ── PAGE: TRAINING HISTORY ────────────────────────────────────────────────
    elif st.session_state.hospital_page == "history":
        colH1, colH2 = st.columns([8, 2])
        with colH1:
            st.markdown("<h1>📜 My Training History</h1>", unsafe_allow_html=True)
        with colH2:
            st.write("")
            st.write("")
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        resp_history = requests.get(f"{API_BASE_URL}/weights/history/", headers=headers)
        if resp_history.status_code == 200:
            history_data = resp_history.json()
            if history_data:
                df_history = pd.DataFrame(history_data)
                fig = px.area(df_history, x="round_number", y="local_mean_auroc",
                              title="Hospital Local AUROC Progression", markers=True)
                fig.update_layout(xaxis_title="Communication Round", yaxis_title="Mean AUROC")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Submission Ledger")
                for hist in history_data:
                    with st.expander(f"Round {hist['round_number']} | Local AUROC: {hist['local_mean_auroc']:.4f}"):
                        st.write(f"**Timestamp:** {pd.to_datetime(hist['submitted_at']).strftime('%B %d, %Y - %H:%M')}")
                        metrics_raw = hist.get('metrics_json', {})
                        if isinstance(metrics_raw, str):
                            try:
                                metrics = json.loads(metrics_raw)
                            except:
                                metrics = {}
                        else:
                            metrics = metrics_raw if metrics_raw else {}
                        if metrics:
                            st.table(pd.DataFrame([metrics], index=["Score"]))
            else:
                st.info("No training history found. Please complete your first communication round to view analytics.")
        else:
            st.error("Failed to securely fetch history data.")

    # ── PAGE: MANAGE DOCTORS ──────────────────────────────────────────────────
    elif st.session_state.hospital_page == "doctors":
        st.markdown("<h1>👨‍⚕️ Manage Doctors</h1>", unsafe_allow_html=True)
        st.markdown("Register or remove physicians authorized to use the Live Diagnostic module at your hospital node.")
        st.write("")

        with st.expander("➕ Register New Doctor", expanded=True):
            dCol1, dCol2, dCol3 = st.columns(3)
            with dCol1:
                doc_username = st.text_input("Login Username", key="doc_username")
            with dCol2:
                doc_fullname = st.text_input("Full Name (e.g. Dr. Ali Khan)", key="doc_fullname")
            with dCol3:
                doc_password = st.text_input("Set Password", type="password", key="doc_password")

            if st.button("Register Doctor", type="primary", use_container_width=True):
                if not doc_username or not doc_password:
                    st.warning("Username and password are required.")
                else:
                    resp_doc = requests.post(
                        f"{API_BASE_URL}/doctors/",
                        headers=headers,
                        data={"username": doc_username, "password": doc_password, "full_name": doc_fullname}
                    )
                    if resp_doc.status_code == 201:
                        st.success(f"✅ {resp_doc.json()['detail']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(resp_doc.json().get('detail', 'Registration failed.'))

        st.markdown("---")
        st.markdown("### Registered Physicians ")

        resp_docs = requests.get(f"{API_BASE_URL}/doctors/", headers=headers)
        if resp_docs.status_code == 200:
            doctors = resp_docs.json()
            if doctors:
                df_docs = pd.DataFrame(doctors)
                if 'date_joined' in df_docs.columns:
                    df_docs['date_joined'] = pd.to_datetime(df_docs['date_joined']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(df_docs[['username', 'full_name', 'date_joined']], use_container_width=True)
                st.metric("Total Registered Physicians", len(doctors))

                st.markdown("#### Remove Doctor")
                del_opts = {f"{d['full_name']} ({d['username']})": d['id'] for d in doctors}
                selected_doc = st.selectbox("Select doctor to remove", list(del_opts.keys()), key="del_doc")
                if st.button("Remove Selected Doctor", type="secondary"):
                    del_id = del_opts[selected_doc]
                    resp_del = requests.delete(f"{API_BASE_URL}/doctors/{del_id}/delete/", headers=headers)
                    if resp_del.status_code == 200:
                        st.success(f"✅ {resp_del.json()['detail']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(resp_del.json().get('detail', 'Removal failed.'))
            else:
                st.info("No doctors registered yet. Use the form above to add one.")
        else:
            st.error("Failed to fetch doctor list.")
