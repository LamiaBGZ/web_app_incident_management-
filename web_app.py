import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
from datetime import datetime
import io
import zipfile
import plotly.express as px
import plotly.graph_objects as go

# IA
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# =========================
# CONFIG & THEME
# =========================
st.set_page_config(
    page_title="ClassroomAI — Inspection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ClassroomAI — Système d'inspection IA des salles de classe"}
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --blue-600:   #2563EB;
    --blue-500:   #3B82F6;
    --blue-400:   #60A5FA;
    --blue-100:   #DBEAFE;
    --blue-50:    #EFF6FF;
    --slate-900:  #0F172A;
    --slate-700:  #334155;
    --slate-500:  #64748B;
    --slate-300:  #CBD5E1;
    --slate-100:  #F1F5F9;
    --slate-50:   #F8FAFC;
    --white:      #FFFFFF;
    --red-500:    #EF4444;
    --red-100:    #FEE2E2;
    --green-500:  #22C55E;
    --green-100:  #DCFCE7;
    --amber-500:  #F59E0B;
    --amber-100:  #FEF3C7;
    --radius-lg:  14px;
    --radius-md:  10px;
    --shadow-sm:  0 1px 3px rgba(15,23,42,.07), 0 1px 2px rgba(15,23,42,.04);
    --shadow-md:  0 4px 16px rgba(15,23,42,.08), 0 2px 6px rgba(15,23,42,.04);
}

/* ── Base typography ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--slate-700);
}

/* ── App background ── */
.stApp {
    background-color: var(--slate-50) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--slate-100) !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem !important; }

/* ── Main content padding ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Cards ── */
.card {
    background: var(--white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(203,213,225,.5);
}

/* ── KPI cards ── */
.kpi-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
.kpi-card {
    background: var(--white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(203,213,225,.5);
    display: flex;
    flex-direction: column;
    gap: .4rem;
}
.kpi-icon { font-size: 1.6rem; line-height:1; }
.kpi-label { font-size:.75rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color: var(--slate-500); margin:0; }
.kpi-value { font-size:2rem; font-weight:700; color: var(--slate-900); margin:0; line-height:1.1; }
.kpi-sub   { font-size:.78rem; color: var(--slate-500); margin:0; }
.kpi-badge { display:inline-block; padding:.15rem .55rem; border-radius:99px; font-size:.7rem; font-weight:600; }
.badge-green { background:var(--green-100); color:#15803D; }
.badge-red   { background:var(--red-100);   color:#B91C1C; }
.badge-blue  { background:var(--blue-100);  color:var(--blue-600); }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: .6rem;
    margin-bottom: 1.1rem;
}
.section-header h2 {
    font-size: 1.1rem; font-weight: 700;
    color: var(--slate-900); margin: 0;
}
.section-tag {
    font-size: .68rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--blue-600);
    background: var(--blue-50); border-radius: 99px;
    padding: .15rem .55rem;
}

/* ── Divider ── */
.divider { height:1px; background:var(--slate-100); margin:1.75rem 0; }

/* ── Streamlit buttons → override ── */
.stButton > button {
    background: var(--blue-600) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .875rem !important;
    padding: .55rem 1.25rem !important;
    transition: background .18s, box-shadow .18s, transform .1s !important;
    box-shadow: 0 1px 4px rgba(37,99,235,.25) !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Danger button (reset) ── */
.danger-btn > button {
    background: var(--red-500) !important;
    box-shadow: 0 1px 4px rgba(239,68,68,.25) !important;
}
.danger-btn > button:hover {
    background: #DC2626 !important;
    box-shadow: 0 4px 12px rgba(239,68,68,.35) !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: var(--slate-50) !important;
    color: var(--slate-700) !important;
    border: 1.5px solid var(--slate-300) !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--blue-50) !important;
    border-color: var(--blue-400) !important;
    color: var(--blue-600) !important;
    transform: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--white) !important;
    border: 2px dashed var(--slate-300) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue-400) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--blue-500), var(--blue-400)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--blue-100) !important;
    border-radius: 99px !important;
}

/* ── Alerts / info boxes ── */
.stSuccess > div {
    background: var(--green-100) !important;
    color: #166534 !important;
    border-left: 4px solid var(--green-500) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
}
.stError > div {
    background: var(--red-100) !important;
    color: #991B1B !important;
    border-left: 4px solid var(--red-500) !important;
    border-radius: var(--radius-md) !important;
}
.stInfo > div {
    background: var(--blue-50) !important;
    color: var(--blue-600) !important;
    border-left: 4px solid var(--blue-500) !important;
    border-radius: var(--radius-md) !important;
}
.stWarning > div {
    background: var(--amber-100) !important;
    color: #92400E !important;
    border-left: 4px solid var(--amber-500) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Metric tweaks ── */
[data-testid="metric-container"] {
    background: transparent !important;
}

/* ── Data table ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--slate-100) !important;
}

/* ── Sidebar nav links ── */
.nav-item {
    display: flex; align-items: center; gap: .65rem;
    padding: .55rem .9rem;
    border-radius: var(--radius-md);
    margin-bottom: .25rem;
    cursor: pointer;
    font-size: .875rem; font-weight: 500;
    color: var(--slate-500);
    transition: background .15s, color .15s;
}
.nav-item:hover   { background: var(--slate-100); color: var(--slate-900); }
.nav-item.active  { background: var(--blue-50);   color: var(--blue-600);  font-weight:600; }
.nav-icon { font-size: 1rem; }

/* ── Sidebar logo ── */
.sidebar-logo {
    padding: .25rem 1rem 1.25rem;
    border-bottom: 1px solid var(--slate-100);
    margin-bottom: 1rem;
}
.sidebar-logo-title {
    font-size: 1rem; font-weight: 700; color: var(--slate-900); margin:0;
}
.sidebar-logo-sub {
    font-size: .72rem; color: var(--slate-500); margin:0;
}

/* ── Tag chips ── */
.chip {
    display:inline-block; padding:.15rem .6rem;
    border-radius:99px; font-size:.72rem; font-weight:600;
    margin-right:.25rem;
}
.chip-red   { background:var(--red-100);   color:#B91C1C; }
.chip-green { background:var(--green-100); color:#166534; }
.chip-blue  { background:var(--blue-100);  color:var(--blue-600); }
</style>
""", unsafe_allow_html=True)


# =========================
# CACHE / MODELS
# =========================
@st.cache_resource
def load_yolo():
    return YOLO("../yolov8n.pt")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

yolo_model = load_yolo()
blip_processor, blip_model = load_blip()

if "db" not in st.session_state:
    st.session_state["db"] = pd.DataFrame()
if "page" not in st.session_state:
    st.session_state["page"] = "upload"


# =========================
# UTILS & REGLES METIERS  (unchanged)
# =========================
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def extract_text(image):
    try:
        text = pytesseract.image_to_string(image).strip()
        return text if text else "Aucun"
    except:
        return "Erreur OCR"

def generate_caption(image):
    try:
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except:
        return ""

def detect_quality_anomalies(cv_img):
    anomalies = []
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
        anomalies.append("Photo floue")
    brightness = np.mean(gray)
    if brightness < 50:
        anomalies.append("Photo sombre")
    elif brightness > 200:
        anomalies.append("Photo surexposée")
    return anomalies

def check_classroom_rules(detected_objects, caption):
    anomalies_globales = []
    persons   = [d for d in detected_objects if d['nom'] == 'person']
    backpacks = [d for d in detected_objects if d['nom'] == 'backpack']
    tables    = [d for d in detected_objects if d['nom'] in ['dining table', 'desk']]
    tvs       = [d for d in detected_objects if d['nom'] == 'tv']

    if len(persons) != 9:
        anomalies_globales.append(f"Effectif incorrect ({len(persons)} au lieu de 9)")
    if len(tvs) == 0:
        anomalies_globales.append("TV absente/éteinte")
    if "open window" in caption.lower():
        anomalies_globales.append("Fenêtre ouverte")

    for bp in backpacks:
        bp_y_bottom = bp['box'][3]
        for table in tables:
            table_y_top = table['box'][1]
            if bp_y_bottom < table_y_top + 30:
                bp['anomalie_specifique'] = "🚨 Cartable posé SUR une table"
                break
    return anomalies_globales

def process_image_to_rows(file):
    image  = Image.open(file).convert("RGB")
    cv_img = pil_to_cv2(image)
    text   = extract_text(image)
    caption = generate_caption(image)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = yolo_model(cv_img)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name   = yolo_model.names[cls_id]
            conf   = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            detected_objects.append(
                {"nom": name, "confiance": round(conf, 2), "box": coords, "anomalie_specifique": "Aucune"})

    anomalies_qualite         = detect_quality_anomalies(cv_img)
    anomalies_metier_globales = check_classroom_rules(detected_objects, caption)
    toutes_anomalies_globales = anomalies_qualite + anomalies_metier_globales
    anomalies_globales_str    = " | ".join(toutes_anomalies_globales) if toutes_anomalies_globales else "Conforme"

    rows = []
    base_info = {
        "Nom du Fichier": file.name,
        "Date": date_str,
        "Anomalies Globales (Salle)": anomalies_globales_str,
        "Texte Extrait (OCR)": text,
        "Description Globale": caption
    }

    if not detected_objects:
        row = base_info.copy()
        row.update({"Objet": "Aucun", "Confiance (%)": 0.0, "Anomalie Spécifique (Objet)": "Aucune"})
        rows.append(row)
    else:
        for obj in detected_objects:
            row = base_info.copy()
            row.update({
                "Objet": obj["nom"],
                "Confiance (%)": obj["confiance"] * 100,
                "Anomalie Spécifique (Objet)": obj["anomalie_specifique"]
            })
            rows.append(row)
    return rows


# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <p class="sidebar-logo-title">🏫 ClassroomAI</p>
        <p class="sidebar-logo-sub">Inspection IA — Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "upload":    ("📤", "Upload & Analyse"),
        "dashboard": ("📊", "Dashboard"),
        "data":      ("🗄️",  "Base de données"),
        "export":    ("📥", "Export"),
    }

    for key, (icon, label) in pages.items():
        active = "active" if st.session_state["page"] == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    # Status badge
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    n = len(st.session_state["db"]["Nom du Fichier"].unique()) if not st.session_state["db"].empty else 0
    badge_cls  = "badge-green" if n > 0 else "badge-blue"
    badge_txt  = f"{n} salle{'s' if n != 1 else ''} en base"
    st.markdown(f"""
    <div style="padding:.4rem 1rem;">
        <span class="kpi-badge {badge_cls}">{badge_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Reset button at bottom
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️  Vider la base", use_container_width=True):
            st.session_state["db"] = pd.DataFrame()
            st.session_state["page"] = "upload"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PLOTLY THEME HELPER
# =========================
PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_family="DM Sans",
    font_color="#334155",
    title_font_size=14,
    title_font_color="#0F172A",
    margin=dict(t=48, b=24, l=24, r=24),
)


# =========================
# PAGE: UPLOAD & ANALYSE
# =========================
page = st.session_state["page"]

if page == "upload":
    st.markdown("""
    <div class="section-header">
        <h2>Upload & Analyse</h2>
        <span class="section-tag">IA YOLO + BLIP + OCR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("##### 📤 Déposer les photos des salles de classe")
    uploaded_files = st.file_uploader(
        "Glissez-déposez vos images ici",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;">
            <span class="kpi-badge badge-blue">{len(uploaded_files)} fichier{'s' if len(uploaded_files)>1 else ''} sélectionné{'s' if len(uploaded_files)>1 else ''}</span>
        </div>
        """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze = st.button("🔍  Lancer l'analyse", use_container_width=True)

    if analyze:
        if not uploaded_files:
            st.error("⚠️ Veuillez d'abord sélectionner au moins une image.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            status_txt = st.empty()
            progress   = st.progress(0)
            all_new_rows = []

            for i, file in enumerate(uploaded_files):
                status_txt.markdown(f"**Analyse en cours…** `{file.name}` ({i+1}/{len(uploaded_files)})")
                all_new_rows.extend(process_image_to_rows(file))
                progress.progress((i + 1) / len(uploaded_files))

            new_df = pd.DataFrame(all_new_rows)
            if st.session_state["db"].empty:
                st.session_state["db"] = new_df
            else:
                st.session_state["db"] = pd.concat([st.session_state["db"], new_df], ignore_index=True)

            status_txt.empty()
            progress.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.success(f"✅ Analyse terminée — {len(uploaded_files)} image(s) traitée(s) et ajoutée(s) à la base.")
            st.balloons()

            col_nav, _ = st.columns([1, 3])
            with col_nav:
                if st.button("📊  Voir le Dashboard →", use_container_width=True):
                    st.session_state["page"] = "dashboard"
                    st.rerun()


# =========================
# PAGE: DASHBOARD
# =========================
elif page == "dashboard":
    st.markdown("""
    <div class="section-header">
        <h2>Dashboard Analytique</h2>
        <span class="section-tag">TEMPS RÉEL</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 Aucune donnée disponible. Commencez par analyser des images dans la section **Upload & Analyse**.")
    else:
        df = st.session_state["db"]
        df_images = df[["Nom du Fichier", "Anomalies Globales (Salle)"]].drop_duplicates()

        total_salles   = len(df_images)
        salles_conf    = len(df_images[df_images["Anomalies Globales (Salle)"] == "Conforme"])
        salles_nonconf = total_salles - salles_conf
        taux_conf      = (salles_conf / total_salles) * 100

        # ── KPI cards ──────────────────────────────────────────────────────────
        conf_badge  = f'<span class="kpi-badge badge-green">✓ Bon</span>'   if taux_conf >= 70 else \
                      f'<span class="kpi-badge badge-red">⚠ À surveiller</span>'

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-icon">🏫</span>
                <p class="kpi-label">Salles inspectées</p>
                <p class="kpi-value">{total_salles}</p>
                <p class="kpi-sub">Sessions analysées</p>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">✅</span>
                <p class="kpi-label">Salles conformes</p>
                <p class="kpi-value" style="color:#16A34A">{salles_conf}</p>
                <p class="kpi-sub"><span class="kpi-badge badge-red">{salles_nonconf} non-conforme{'s' if salles_nonconf != 1 else ''}</span></p>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">📈</span>
                <p class="kpi-label">Taux de conformité</p>
                <p class="kpi-value" style="color:{'#16A34A' if taux_conf>=70 else '#DC2626'}">{taux_conf:.1f}%</p>
                <p class="kpi-sub">{conf_badge}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            anomalies_globales_list = df_images["Anomalies Globales (Salle)"].str.split(" | ").explode()
            anomalies_globales_list = anomalies_globales_list[anomalies_globales_list != "Conforme"]
            anomalies_specifiques   = df[df["Anomalie Spécifique (Objet)"] != "Aucune"]["Anomalie Spécifique (Objet)"]
            toutes_anomalies        = pd.concat([anomalies_globales_list, anomalies_specifiques])

            if not toutes_anomalies.empty:
                df_anom = toutes_anomalies.value_counts().reset_index()
                df_anom.columns = ["Anomalie", "Occurrences"]
                fig_anom = px.bar(
                    df_anom, x="Occurrences", y="Anomalie", orientation="h",
                    title="🚨 Classement des infractions",
                    color="Occurrences",
                    color_continuous_scale=["#FEE2E2", "#EF4444", "#B91C1C"],
                )
                fig_anom.update_layout(
                    **PLOT_LAYOUT,
                    yaxis={"categoryorder": "total ascending", "title": ""},
                    xaxis_title="Nombre d'occurrences",
                    coloraxis_showscale=False,
                    bargap=0.35,
                )
                fig_anom.update_traces(marker_line_width=0)
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.success("🎉 Aucune anomalie détectée — toutes les salles sont conformes !")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df_objets = df[df["Objet"] != "Aucun"]
            if not df_objets.empty:
                fig_pie = px.pie(
                    df_objets, names="Objet",
                    title="🎒 Répartition des objets détectés",
                    hole=0.46,
                    color_discrete_sequence=[
                        "#2563EB","#3B82F6","#60A5FA","#93C5FD",
                        "#DBEAFE","#1D4ED8","#1E40AF","#BFDBFE"
                    ],
                )
                fig_pie.update_layout(**PLOT_LAYOUT)
                fig_pie.update_traces(textposition="outside", textinfo="label+percent")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Aucun objet détecté dans la base actuelle.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Conformité timeline (si plusieurs images) ─────────────────────────
        if total_salles > 1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df_timeline = df_images.copy()
            df_timeline["Conforme"] = (df_timeline["Anomalies Globales (Salle)"] == "Conforme").astype(int)
            df_timeline["Indice"] = range(1, len(df_timeline) + 1)

            fig_line = px.line(
                df_timeline, x="Indice", y="Conforme",
                title="📉 Conformité par salle (1 = conforme, 0 = non-conforme)",
                markers=True,
                color_discrete_sequence=["#2563EB"],
            )
            fig_line.update_layout(
                **PLOT_LAYOUT,
                yaxis=dict(tickvals=[0, 1], ticktext=["Non conforme", "Conforme"], title=""),
                xaxis_title="Salle n°",
            )
            fig_line.update_traces(marker_size=8, line_width=2.5)
            st.plotly_chart(fig_line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PAGE: BASE DE DONNÉES
# =========================
elif page == "data":
    st.markdown("""
    <div class="section-header">
        <h2>Base de données d'inspection</h2>
        <span class="section-tag">DÉTAILS BRUTS</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 La base est vide. Analysez d'abord des images dans **Upload & Analyse**.")
    else:
        df = st.session_state["db"]

        # Summary chips
        n_rows   = len(df)
        n_imgs   = df["Nom du Fichier"].nunique()
        n_anom   = len(df[df["Anomalie Spécifique (Objet)"] != "Aucune"])
        st.markdown(f"""
        <div style="margin-bottom:1rem;display:flex;gap:.5rem;flex-wrap:wrap;">
            <span class="chip chip-blue">📄 {n_rows} lignes</span>
            <span class="chip chip-blue">🏫 {n_imgs} images</span>
            <span class="chip chip-red">🚨 {n_anom} anomalie{'s' if n_anom!=1 else ''} objet</span>
        </div>
        """, unsafe_allow_html=True)

        # Filters
        with st.expander("🔍 Filtres"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                files_list = ["Tous"] + sorted(df["Nom du Fichier"].unique().tolist())
                sel_file   = st.selectbox("Fichier", files_list)
            with col_f2:
                sel_obj = st.selectbox("Objet", ["Tous"] + sorted(df["Objet"].unique().tolist()))

        df_view = df.copy()
        if sel_file != "Tous":
            df_view = df_view[df_view["Nom du Fichier"] == sel_file]
        if sel_obj != "Tous":
            df_view = df_view[df_view["Objet"] == sel_obj]

        def highlight_anomalies(val):
            color = "#DC2626" if isinstance(val, str) and val != "Aucune" else ""
            return f"color: {color}; font-weight: {'600' if color else '400'}"

        st.markdown('<div class="card" style="padding:.75rem">', unsafe_allow_html=True)
        st.dataframe(
            df_view.style.map(highlight_anomalies, subset=["Anomalie Spécifique (Objet)"]),
            use_container_width=True,
            height=420,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PAGE: EXPORT
# =========================
elif page == "export":
    st.markdown("""
    <div class="section-header">
        <h2>Export des données</h2>
        <span class="section-tag">CSV · ZIP</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 Aucune donnée à exporter. Analysez d'abord des images dans **Upload & Analyse**.")
    else:
        df = st.session_state["db"]

        col_e1, col_e2 = st.columns(2, gap="large")

        with col_e1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ⬇️ Export CSV global")
            st.markdown(f"Toutes les données consolidées en un seul fichier CSV `inspection_salles.csv`.")
            st.markdown(f"<br>", unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Télécharger le CSV",
                csv, "inspection_salles.csv", "text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_e2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ⬇️ Export ZIP (par salle)")
            st.markdown(f"Un CSV individuel par image, regroupés dans une archive ZIP.")
            st.markdown(f"<br>", unsafe_allow_html=True)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for nom_fichier in df["Nom du Fichier"].unique():
                    df_fichier = df[df["Nom du Fichier"] == nom_fichier]
                    zf.writestr(f"{nom_fichier}.csv", df_fichier.to_csv(index=False))
            st.download_button(
                "⬇️  Télécharger le ZIP",
                zip_buffer.getvalue(), "details_salles.zip",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Aperçu rapide** — {len(df)} lignes · {df['Nom du Fichier'].nunique()} images")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)