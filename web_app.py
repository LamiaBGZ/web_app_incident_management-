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
    initial_sidebar_state="collapsed",
    menu_items={"About": "ClassroomAI — Système d'inspection IA des salles de classe"}
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

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

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; color: var(--slate-700); }
.stApp { background-color: var(--slate-50) !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding: 2rem 1.5rem 3rem !important; max-width: 1400px !important; }

.card {
    background: var(--white); border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md); padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem; border: 1px solid rgba(203,213,225,.5);
}

.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin-bottom:1.5rem; }
.kpi-card {
    background: var(--white); border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md); padding: 1.4rem 1.6rem;
    border: 1px solid rgba(203,213,225,.5);
    display: flex; flex-direction: column; gap: .4rem;
}
.kpi-icon  { font-size: 1.6rem; line-height:1; }
.kpi-label { font-size:.75rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color: var(--slate-500); margin:0; }
.kpi-value { font-size:2rem; font-weight:700; color: var(--slate-900); margin:0; line-height:1.1; }
.kpi-sub   { font-size:.78rem; color: var(--slate-500); margin:0; }
.kpi-badge { display:inline-block; padding:.15rem .55rem; border-radius:99px; font-size:.7rem; font-weight:600; }
.badge-green { background:var(--green-100); color:#15803D; }
.badge-red   { background:var(--red-100);   color:#B91C1C; }
.badge-blue  { background:var(--blue-100);  color:var(--blue-600); }

.section-header { display: flex; align-items: center; gap: .6rem; margin-bottom: 1.1rem; }
.section-header h2 { font-size: 1.1rem; font-weight: 700; color: var(--slate-900); margin: 0; }
.section-tag {
    font-size: .68rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--blue-600);
    background: var(--blue-50); border-radius: 99px; padding: .15rem .55rem;
}

.divider { height:1px; background:var(--slate-100); margin:1.75rem 0; }

button[kind="primary"] {
    background: var(--blue-600) !important; color: #fff !important;
    border: none !important; border-radius: var(--radius-md) !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: .875rem !important; padding: .55rem 1.25rem !important;
    transition: background .18s, box-shadow .18s, transform .1s !important;
    box-shadow: 0 1px 4px rgba(37,99,235,.25) !important;
}
button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,.35) !important;
    transform: translateY(-1px) !important;
}
button[kind="secondary"] {
    background: var(--white) !important; color: var(--slate-500) !important;
    border: 1px solid var(--slate-200) !important; border-radius: var(--radius-md) !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: .875rem !important; padding: .55rem 1.25rem !important;
    transition: all .15s !important;
}
button[kind="secondary"]:hover { background: var(--slate-50) !important; color: var(--slate-900) !important; }

.danger-btn > button {
    background: var(--red-500) !important; color: white !important;
    box-shadow: 0 1px 4px rgba(239,68,68,.25) !important; border: none !important;
}
.danger-btn > button:hover { background: #DC2626 !important; }

[data-testid="stDownloadButton"] > button {
    background: var(--slate-50) !important; color: var(--slate-700) !important;
    border: 1.5px solid var(--slate-300) !important; box-shadow: none !important;
}
[data-testid="stFileUploader"] {
    background: var(--white) !important; border: 2px dashed var(--slate-300) !important;
    border-radius: var(--radius-lg) !important; padding: 1rem !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--blue-500), var(--blue-400)) !important;
    border-radius: 99px !important;
}

.stSuccess > div { background: var(--green-100) !important; color: #166534 !important; border-left: 4px solid var(--green-500) !important; border-radius: var(--radius-md) !important; }
.stError   > div { background: var(--red-100)   !important; color: #991B1B  !important; border-left: 4px solid var(--red-500)   !important; border-radius: var(--radius-md) !important; }
.stInfo    > div { background: var(--blue-50)   !important; color: var(--blue-600) !important; border-left: 4px solid var(--blue-500) !important; border-radius: var(--radius-md) !important; }

[data-testid="stDataFrame"] { border-radius: var(--radius-lg) !important; overflow: hidden !important; box-shadow: var(--shadow-sm) !important; border: 1px solid var(--slate-100) !important; }

.chip { display:inline-block; padding:.15rem .6rem; border-radius:99px; font-size:.72rem; font-weight:600; margin-right:.25rem; }
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
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


yolo_model = load_yolo()
blip_processor, blip_model = load_blip()

if "db" not in st.session_state: st.session_state["db"] = pd.DataFrame()
if "page" not in st.session_state: st.session_state["page"] = "upload"


# =========================
# UTILS & RÈGLES MÉTIER
# =========================

# Traduction des noms d'objets pour le tableau de bord
CLASS_MAP = {
    "dining table": "Table",
    "desk": "Bureau",
    "chair": "Chaise",
    "tv": "Écran/TV",
    "laptop": "Ordinateur portable",
    "person": "Élève/Prof",
    "book": "Livre",
    "bottle": "Bouteille",
    "cell phone": "Téléphone",
    "backpack": "Cartable/Sac",
    "cup": "Gobelet"
}

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def extract_text(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image).strip()
        return text if text else "Aucun"
    except Exception:
        return "Erreur OCR"

def generate_caption(image: Image.Image) -> str:
    try:
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        return ""

def detect_quality_anomalies(cv_img: np.ndarray) -> list[str]:
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


# ─── Seuils configurables ────────────────────────────────────────────────────
TV_BRIGHTNESS_THRESHOLD = 45     # en-dessous → écran éteint / noir
CHAIR_ASPECT_RATIO_MARGIN = 1.1  # width > height * marge → chaise renversée
EXPECTED_PERSONS = 9             # effectif réglementaire
PERSON_STANDING_RATIO = 2.0      # height / width > ce seuil → personne debout


def check_classroom_rules(detected_objects: list[dict], caption: str, cv_img: np.ndarray) -> list[str]:
    """
    Vérifie l'ensemble des règles métier de la salle.
    """
    anomalies_globales: list[str] = []

    # ── Catégories d'objets détectés (utilisation des noms originaux pour la logique)
    persons = [d for d in detected_objects if d["nom"] == "person"]
    backpacks = [d for d in detected_objects if d["nom"] == "backpack"]
    tables = [d for d in detected_objects if d["nom"] in ("dining table", "desk")]
    chairs = [d for d in detected_objects if d["nom"] == "chair"]
    tvs = [d for d in detected_objects if d["nom"] == "tv"]
    unexpected_items = [d for d in detected_objects if d["nom"] in ("book", "bottle", "cell phone", "cup")]

    # ── 1. Effectif ───────────────────────────────────────────────────────────
    if len(persons) != EXPECTED_PERSONS:
        anomalies_globales.append(f"Effectif incorrect ({len(persons)} au lieu de {EXPECTED_PERSONS})")

    # ── 2. État de la TV (luminosité du crop OpenCV) ──────────────────────────
    if not tvs:
        anomalies_globales.append("TV absente/éteinte")
    else:
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray_img.shape

        for tv in tvs:
            x1, y1, x2, y2 = tv["box"]
            x1c, y1c = max(0, int(x1)), max(0, int(y1))
            x2c, y2c = min(img_w, int(x2)), min(img_h, int(y2))

            tv_crop = gray_img[y1c:y2c, x1c:x2c]
            if tv_crop.size == 0:
                anomalies_globales.append("TV absente/éteinte")
                continue

            if float(np.mean(tv_crop)) < TV_BRIGHTNESS_THRESHOLD:
                anomalies_globales.append("TV absente/éteinte")

    # ── 3. Fenêtre ouverte (caption BLIP) ────────────────────────────────────
    if "open window" in caption.lower():
        anomalies_globales.append("Fenêtre ouverte")

    # ── 4. Déchets par terre (Analyse de texte BLIP) ─────────────────────────
    if any(word in caption.lower() for word in ["trash", "garbage", "litter", "dirty floor", "debris"]):
        anomalies_globales.append("Déchets par terre")

    # ── 5. Cartable posé SUR une table (géométrie YOLO) ──────────────────────
    for bp in backpacks:
        bp_y_bottom = bp["box"][3]
        for table in tables:
            table_y_top = table["box"][1]
            if bp_y_bottom < table_y_top + 30:
                bp["anomalie_specifique"] = "🚨 Cartable posé SUR une table"
                break

    # ── 6. Chaise renversée (rapport d'aspect de la bbox) ────────────────────
    for chair in chairs:
        x1, y1, x2, y2 = chair["box"]
        width, height = x2 - x1, y2 - y1
        if height > 0 and width > height * CHAIR_ASPECT_RATIO_MARGIN:
            chair["anomalie_specifique"] = "🚨 Chaise renversée"

    # ── 7. Personne debout (Analyse du ratio hauteur/largeur) ────────────────
    for person in persons:
        x1, y1, x2, y2 = person["box"]
        width, height = x2 - x1, y2 - y1
        if width > 0 and (height / width) > PERSON_STANDING_RATIO:
            person["anomalie_specifique"] = "🚨 Personne debout"
            if "Personne debout" not in anomalies_globales:
                anomalies_globales.append("Personne debout")

    # ── 8. Objets inattendus (Téléphone, Livre, Bouteille) ───────────────────
    for item in unexpected_items:
        item["anomalie_specifique"] = "🚨 Objet inattendu"
        if "Objet inattendu" not in anomalies_globales:
            anomalies_globales.append("Objet inattendu")

    # ── 9. Objet abandonné (Cartable au sol) ─────────────────────────────────
    for bp in backpacks:
        if bp["anomalie_specifique"] == "Aucune": # S'il n'est pas déjà détecté "SUR la table"
            bp["anomalie_specifique"] = "🚨 Objet abandonné"
            if "Objet abandonné" not in anomalies_globales:
                anomalies_globales.append("Objet abandonné")

    return anomalies_globales


def process_image_to_rows(file) -> list[dict]:
    image = Image.open(file).convert("RGB")
    cv_img = pil_to_cv2(image)
    text = extract_text(image)
    caption = generate_caption(image)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = yolo_model(cv_img)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            detected_objects.append({
                "nom": name,
                "confiance": round(conf, 2),
                "box": coords,
                "anomalie_specifique": "Aucune"
            })

    anomalies_qualite = detect_quality_anomalies(cv_img)
    anomalies_metier_globales = check_classroom_rules(detected_objects, caption, cv_img)

    toutes_anomalies_globales = anomalies_qualite + anomalies_metier_globales
    anomalies_globales_str = (
        " | ".join(toutes_anomalies_globales) if toutes_anomalies_globales else "Conforme"
    )

    base_info = {
        "Nom du Fichier": file.name,
        "Date": date_str,
        "Anomalies Globales (Salle)": anomalies_globales_str,
        "Texte Extrait (OCR)": text,
        "Description Globale": caption,
    }

    rows: list[dict] = []
    if not detected_objects:
        row = base_info.copy()
        row.update({"Objet": "Aucun", "Confiance (%)": 0.0, "Anomalie Spécifique (Objet)": "Aucune"})
        rows.append(row)
    else:
        for obj in detected_objects:
            row = base_info.copy()
            # Traduction du nom pour l'affichage via CLASS_MAP
            nom_francais = CLASS_MAP.get(obj["nom"], obj["nom"].capitalize())
            row.update({
                "Objet": nom_francais,
                "Confiance (%)": round(obj["confiance"] * 100, 1),
                "Anomalie Spécifique (Objet)": obj["anomalie_specifique"],
            })
            rows.append(row)

    return rows


# =========================
# TOP NAVIGATION
# =========================
col_logo, col_badge = st.columns([2, 1])
with col_logo:
    st.markdown("<h3 style='margin:0; padding-top:5px;'>🏫 ClassroomAI</h3>", unsafe_allow_html=True)
with col_badge:
    n = len(st.session_state["db"]["Nom du Fichier"].unique()) if not st.session_state["db"].empty else 0
    badge_cls = "badge-green" if n > 0 else "badge-blue"
    st.markdown(
        f'<div style="text-align:right;margin-top:10px;">'
        f'<span class="kpi-badge {badge_cls}">{n} salle(s)</span></div>',
        unsafe_allow_html=True
    )

st.markdown("<hr style='margin:.75rem 0;'>", unsafe_allow_html=True)

nav_cols = st.columns(4)
pages = {
    "upload": ("📤", "Upload"),
    "dashboard": ("📊", "Dashboard"),
    "data": ("🗄️", "Base"),
    "export": ("📥", "Export"),
}
for i, (key, (icon, label)) in enumerate(pages.items()):
    with nav_cols[i]:
        btn_type = "primary" if st.session_state["page"] == key else "secondary"
        if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state["page"] = key
            st.rerun()

with st.expander("⚙️ Options avancées"):
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Vider la base de données", use_container_width=True):
        st.session_state["db"] = pd.DataFrame()
        st.session_state["page"] = "upload"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# =========================
# PLOTLY THEME
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
# PAGE ROUTING
# =========================
page = st.session_state["page"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : UPLOAD & ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
if page == "upload":
    st.markdown("""
    <div class="section-header">
        <h2>Upload & Analyse</h2>
        <span class="section-tag">IA YOLO + BLIP + OCR</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("##### 📤 Déposer les photos des salles")
    uploaded_files = st.file_uploader(
        "Glissez-déposez vos images ici",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(
            f'<div style="margin-bottom:1rem;">'
            f'<span class="kpi-badge badge-blue">'
            f'{len(uploaded_files)} fichier{"s" if len(uploaded_files) > 1 else ""} sélectionné{"s" if len(uploaded_files) > 1 else ""}'
            f'</span></div>',
            unsafe_allow_html=True
        )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze = st.button("🔍 Lancer l'analyse", use_container_width=True, type="primary")

    if analyze:
        if not uploaded_files:
            st.error("⚠️ Veuillez d'abord sélectionner au moins une image.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            status_txt = st.empty()
            progress_bar = st.progress(0)
            all_new_rows = []

            for i, file in enumerate(uploaded_files):
                status_txt.markdown(
                    f"**Analyse en cours…** `{file.name}` ({i + 1}/{len(uploaded_files)})"
                )
                all_new_rows.extend(process_image_to_rows(file))
                progress_bar.progress((i + 1) / len(uploaded_files))

            new_df = pd.DataFrame(all_new_rows)
            st.session_state["db"] = (
                new_df if st.session_state["db"].empty
                else pd.concat([st.session_state["db"], new_df], ignore_index=True)
            )

            status_txt.empty()
            progress_bar.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.success(f"✅ Analyse terminée — {len(uploaded_files)} image(s) traitée(s).")
            st.balloons()

            col_nav, _ = st.columns([1, 2])
            with col_nav:
                if st.button("📊 Voir le Dashboard →", use_container_width=True, type="secondary"):
                    st.session_state["page"] = "dashboard"
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "dashboard":
    st.markdown("""
    <div class="section-header">
        <h2>Dashboard Analytique</h2>
        <span class="section-tag">TEMPS RÉEL</span>
    </div>""", unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 Aucune donnée disponible. Commencez par analyser des images dans **Upload**.")
    else:
        df = st.session_state["db"]
        df_images = df[["Nom du Fichier", "Anomalies Globales (Salle)"]].drop_duplicates()

        total_salles = len(df_images)
        salles_conf = len(df_images[df_images["Anomalies Globales (Salle)"] == "Conforme"])
        salles_nonconf = total_salles - salles_conf
        taux_conf = (salles_conf / total_salles) * 100
        conf_badge = (
            '<span class="kpi-badge badge-green">✓ Bon</span>'
            if taux_conf >= 70 else
            '<span class="kpi-badge badge-red">⚠ À surveiller</span>'
        )

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
                <p class="kpi-sub">
                    <span class="kpi-badge badge-red">
                        {salles_nonconf} non-conforme{"s" if salles_nonconf != 1 else ""}
                    </span>
                </p>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">📈</span>
                <p class="kpi-label">Taux de conformité</p>
                <p class="kpi-value" style="color:{'#16A34A' if taux_conf >= 70 else '#DC2626'}">{taux_conf:.1f}%</p>
                <p class="kpi-sub">{conf_badge}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Graphique anomalies
        st.markdown('<div class="card">', unsafe_allow_html=True)
        anom_glob = df_images["Anomalies Globales (Salle)"].str.split(" | ").explode()
        anom_glob = anom_glob[anom_glob != "Conforme"]
        anom_spec = df[df["Anomalie Spécifique (Objet)"] != "Aucune"]["Anomalie Spécifique (Objet)"]
        all_anom = pd.concat([anom_glob, anom_spec])

        if not all_anom.empty:
            df_anom = all_anom.value_counts().reset_index()
            df_anom.columns = ["Anomalie", "Occurrences"]
            fig_anom = px.bar(
                df_anom, x="Occurrences", y="Anomalie", orientation="h",
                title="🚨 Classement des infractions (globales & spécifiques)",
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

        # Graphique objets
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_objets = df[df["Objet"] != "Aucun"]
        if not df_objets.empty:
            fig_pie = px.pie(
                df_objets, names="Objet",
                title="🎒 Répartition des objets détectés",
                hole=0.46,
                color_discrete_sequence=[
                    "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD",
                    "#DBEAFE", "#1D4ED8", "#1E40AF", "#BFDBFE"
                ],
            )
            fig_pie.update_layout(**PLOT_LAYOUT)
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Aucun objet détecté.")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : BASE DE DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "data":
    st.markdown("""
    <div class="section-header">
        <h2>Base de données</h2>
        <span class="section-tag">DÉTAILS BRUTS</span>
    </div>""", unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 La base est vide.")
    else:
        df = st.session_state["db"]
        n_rows = len(df)
        n_imgs = df["Nom du Fichier"].nunique()
        n_anom = len(df[df["Anomalie Spécifique (Objet)"] != "Aucune"])

        st.markdown(
            f'<div style="margin-bottom:1rem;display:flex;gap:.5rem;flex-wrap:wrap;">'
            f'<span class="chip chip-blue">📄 {n_rows} lignes</span>'
            f'<span class="chip chip-blue">🏫 {n_imgs} images</span>'
            f'<span class="chip chip-red">🚨 {n_anom} anomalie(s) objet</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔍 Filtres"):
            sel_file = st.selectbox("Fichier", ["Tous"] + sorted(df["Nom du Fichier"].unique().tolist()))
            sel_obj = st.selectbox("Objet", ["Tous"] + sorted(df["Objet"].unique().tolist()))

        df_view = df.copy()
        if sel_file != "Tous": df_view = df_view[df_view["Nom du Fichier"] == sel_file]
        if sel_obj != "Tous": df_view = df_view[df_view["Objet"] == sel_obj]


        def highlight_anomalies(val):
            color = "#DC2626" if isinstance(val, str) and val != "Aucune" else ""
            weight = "600" if color else "400"
            return f"color: {color}; font-weight: {weight}"


        st.markdown('<div class="card" style="padding:.75rem">', unsafe_allow_html=True)
        st.dataframe(
            df_view.style.map(highlight_anomalies, subset=["Anomalie Spécifique (Objet)"]),
            use_container_width=True,
            height=400,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : EXPORT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "export":
    st.markdown("""
    <div class="section-header">
        <h2>Export des données</h2>
        <span class="section-tag">CSV · ZIP</span>
    </div>""", unsafe_allow_html=True)

    if st.session_state["db"].empty:
        st.info("📭 Aucune donnée à exporter.")
    else:
        df = st.session_state["db"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⬇️ Export CSV global")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger le CSV", csv, "inspection_salles.csv", "text/csv",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⬇️ Export ZIP (par salle)")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for nom_fichier in df["Nom du Fichier"].unique():
                zf.writestr(
                    f"{nom_fichier}.csv",
                    df[df["Nom du Fichier"] == nom_fichier].to_csv(index=False)
                )
        st.download_button(
            "⬇️ Télécharger le ZIP", zip_buffer.getvalue(), "details_salles.zip",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)