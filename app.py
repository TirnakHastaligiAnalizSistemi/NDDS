# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import os
import time
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import streamlit.components.v1 as components

# SAYFA AYARLARI

st.set_page_config(
    page_title="NDDS | Nail Disease Detection",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ÇEVİRİLER

LANGUAGES = {
    "Türkçe": "tr",
    "English": "en",
    "Español": "es",
    "Deutsch": "de",
    "Azərbaycan": "az"
}

TEXTS = {
    "tr": {
        "language_title": "Dil Seçimi",
        "language_subtitle": "Lütfen uygulama dilini seçin",
        "continue_button": "Devam Et",
        "main_title": "NDDS",
        "main_subtitle": "Nail Disease Detection System",
        "main_desc": "DenseNet121 tabanlı tek model ile sağlıklı / hastalıklı tırnak ve hastalık tipi analizi",
        "upload_label": "Bir tırnak fotoğrafı yükleyin",
        "slider_label": "Sağlıklı kabul eşiği",
        "analyzing": "Görüntü analiz ediliyor...",
        "analysis_result": "🔍 Analiz Sonucu",
        "healthy_prob": "Sağlıklı olasılığı",
        "harmful_prob": "Hastalıklı olasılığı",
        "healthy_msg": "Tırnak genel olarak sağlıklı görünüyor.",
        "harmful_msg": "⚠ Tırnakta hastalık belirtisi olabilir.",
        "detected_disease": "🎯 Tespit Edilen Hastalık",
        "model_prob": "Model olasılığı",
        "medical_explanation": "📘 Tıbbi Açıklama",
        "risk_distribution": "📊 Sistemik Risk Dağılımı",
        "all_probs": "🔎 Tüm sınıf olasılıklarını göster",
        "class_col": "Sınıf",
        "prob_col": "Olasılık",
        "uploaded_image": "Yüklenen Görüntü",
        "warning": "Uyarı: Bu sistem yalnızca yapay zekâ destekli ön değerlendirme amaçlıdır. Kesin tanı için uzman hekime başvurulmalıdır.",
        "healthy_class_text": "Görüntü model tarafından sağlıklı sınıfa daha yakın bulunmuştur. Bu sonuç klinik tanı yerine geçmez.",
        "no_explanation": "Bu sınıf için açıklama bulunamadı.",
        "healthy_not_found": "❌ TRAIN_DIR içinde 'healthy' sınıfı bulunamadı. Klasör isimlerini kontrol edin."
    },
    "en": {
        "language_title": "Language Selection",
        "language_subtitle": "Please choose the application language",
        "continue_button": "Continue",
        "main_title": "NDDS",
        "main_subtitle": "Nail Disease Detection System",
        "main_desc": "Single DenseNet121-based model for healthy / diseased nail and disease type analysis",
        "upload_label": "Upload a nail image",
        "slider_label": "Healthy classification threshold",
        "analyzing": "Analyzing image...",
        "analysis_result": "🔍 Analysis Result",
        "healthy_prob": "Healthy probability",
        "harmful_prob": "Diseased probability",
        "healthy_msg": "The nail appears generally healthy.",
        "harmful_msg": "⚠ There may be signs of nail disease.",
        "detected_disease": "🎯 Detected Disease",
        "model_prob": "Model probability",
        "medical_explanation": "📘 Medical Explanation",
        "risk_distribution": "📊 Systemic Risk Distribution",
        "all_probs": "🔎 Show all class probabilities",
        "class_col": "Class",
        "prob_col": "Probability",
        "uploaded_image": "Uploaded Image",
        "warning": "Warning: This system is intended only for AI-assisted preliminary evaluation. Please consult a medical specialist for a definitive diagnosis.",
        "healthy_class_text": "The image was found closer to the healthy class by the model. This result does not replace a clinical diagnosis.",
        "no_explanation": "No explanation found for this class.",
        "healthy_not_found": "❌ 'healthy' class could not be found in TRAIN_DIR. Please check folder names."
    },
    "es": {
        "language_title": "Selección de idioma",
        "language_subtitle": "Por favor seleccione el idioma de la aplicación",
        "continue_button": "Continuar",
        "main_title": "NDDS",
        "main_subtitle": "Sistema de Detección de Enfermedades de Uñas",
        "main_desc": "Modelo único basado en DenseNet121 para análisis de uñas sanas / enfermas y tipo de enfermedad",
        "upload_label": "Suba una imagen de la uña",
        "slider_label": "Umbral de clasificación saludable",
        "analyzing": "Analizando imagen...",
        "analysis_result": "🔍 Resultado del análisis",
        "healthy_prob": "Probabilidad de saludable",
        "harmful_prob": "Probabilidad de enfermedad",
        "healthy_msg": "La uña parece generalmente saludable.",
        "harmful_msg": "⚠ Puede haber signos de enfermedad en la uña.",
        "detected_disease": "🎯 Enfermedad detectada",
        "model_prob": "Probabilidad del modelo",
        "medical_explanation": "📘 Explicación médica",
        "risk_distribution": "📊 Distribución del riesgo sistémico",
        "all_probs": "🔎 Mostrar todas las probabilidades de clase",
        "class_col": "Clase",
        "prob_col": "Probabilidad",
        "uploaded_image": "Imagen cargada",
        "warning": "Advertencia: Este sistema es solo para evaluación preliminar asistida por inteligencia artificial. Para un diagnóstico definitivo, consulte a un especialista.",
        "healthy_class_text": "La imagen fue clasificada por el modelo como más cercana a la clase saludable. Este resultado no sustituye un diagnóstico clínico.",
        "no_explanation": "No se encontró explicación para esta clase.",
        "healthy_not_found": "❌ No se encontró la clase 'healthy' en TRAIN_DIR. Verifique los nombres de las carpetas."
    },
    "de": {
        "language_title": "Sprachauswahl",
        "language_subtitle": "Bitte wählen Sie die Sprache der Anwendung",
        "continue_button": "Weiter",
        "main_title": "NDDS",
        "main_subtitle": "System zur Erkennung von Nagelerkrankungen",
        "main_desc": "Einzelnes DenseNet121-basiertes Modell zur Analyse gesunder / erkrankter Nägel und des Krankheitstyps",
        "upload_label": "Laden Sie ein Nagelbild hoch",
        "slider_label": "Schwellenwert für gesund",
        "analyzing": "Bild wird analysiert...",
        "analysis_result": "🔍 Analyseergebnis",
        "healthy_prob": "Wahrscheinlichkeit gesund",
        "harmful_prob": "Wahrscheinlichkeit krank",
        "healthy_msg": "Der Nagel wirkt insgesamt gesund.",
        "harmful_msg": "⚠ Es könnten Anzeichen einer Nagelerkrankung vorliegen.",
        "detected_disease": "🎯 Erkannte Erkrankung",
        "model_prob": "Modellwahrscheinlichkeit",
        "medical_explanation": "📘 Medizinische Erklärung",
        "risk_distribution": "📊 Systemische Risikoverteilung",
        "all_probs": "🔎 Alle Klassenwahrscheinlichkeiten anzeigen",
        "class_col": "Klasse",
        "prob_col": "Wahrscheinlichkeit",
        "uploaded_image": "Hochgeladenes Bild",
        "warning": "Warnung: Dieses System dient nur der KI-gestützten Vorbewertung. Für eine endgültige Diagnose wenden Sie sich bitte an einen Facharzt.",
        "healthy_class_text": "Das Bild wurde vom Modell näher an der gesunden Klasse eingeordnet. Dieses Ergebnis ersetzt keine klinische Diagnose.",
        "no_explanation": "Für diese Klasse wurde keine Erklärung gefunden.",
        "healthy_not_found": "❌ Die Klasse 'healthy' wurde im TRAIN_DIR nicht gefunden. Bitte prüfen Sie die Ordnernamen."
    },
    "az": {
        "language_title": "Dil Seçimi",
        "language_subtitle": "Zəhmət olmasa tətbiq dilini seçin",
        "continue_button": "Davam et",
        "main_title": "NDDS",
        "main_subtitle": "Dırnaq Xəstəliklərini Aşkarlama Sistemi",
        "main_desc": "DenseNet121 əsaslı tək model ilə sağlam / xəstə dırnaq və xəstəlik növünün analizi",
        "upload_label": "Dırnaq şəkli yükləyin",
        "slider_label": "Sağlam qəbul həddi",
        "analyzing": "Şəkil analiz edilir...",
        "analysis_result": "🔍 Analiz Nəticəsi",
        "healthy_prob": "Sağlam ehtimalı",
        "harmful_prob": "Xəstə ehtimalı",
        "healthy_msg": "Dırnaq ümumiyyətlə sağlam görünür.",
        "harmful_msg": "⚠ Dırnaqlarda xəstəlik əlaməti ola bilər.",
        "detected_disease": "🎯 Aşkar Edilən Xəstəlik",
        "model_prob": "Model ehtimalı",
        "medical_explanation": "📘 Tibbi İzahat",
        "risk_distribution": "📊 Sistemik Risk Paylanması",
        "all_probs": "🔎 Bütün sinif ehtimallarını göstər",
        "class_col": "Sinif",
        "prob_col": "Ehtimal",
        "uploaded_image": "Yüklənmiş Şəkil",
        "warning": "Xəbərdarlıq: Bu sistem yalnız süni intellekt dəstəkli ilkin qiymətləndirmə üçün nəzərdə tutulmuşdur. Dəqiq diaqnoz üçün mütəxəssis həkimə müraciət edilməlidir.",
        "healthy_class_text": "Şəkil model tərəfindən sağlam sinfə daha yaxın tapılmışdır. Bu nəticə klinik diaqnozu əvəz etmir.",
        "no_explanation": "Bu sinif üçün izahat tapılmadı.",
        "healthy_not_found": "❌ TRAIN_DIR içində 'healthy' sinifi tapılmadı. Qovluq adlarını yoxlayın."
    }
}

# SESSION STATE

if "selected_language" not in st.session_state:
    st.session_state.selected_language = None

if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

# DİL SEÇİMİ EKRANI

if st.session_state.selected_language is None:
    st.markdown(
        """
        <div style='text-align:center; padding-top:80px; padding-bottom:30px;'>
            <h1 style='color:#0b3d3b;'>🌍 Dil / Language / Idioma / Sprache</h1>
            <p style='font-size:18px; color:#4f6363;'>Please choose your preferred language</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_label = st.selectbox(
        "Select Language / Dil Seçin / Seleccione Idioma / Sprache wählen / Dil Seçimi",
        list(LANGUAGES.keys())
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Continue", use_container_width=True):
            st.session_state.selected_language = LANGUAGES[selected_label]
            st.rerun()

    st.stop()

lang = st.session_state.selected_language
T = TEXTS[lang]


# AÇILIŞ EKRANI

def show_intro():
    intro_html = """
    <div id="intro-screen">
        <div id="main-text">NDDS</div>
        <div id="brand-text">Nail Disease Detection System</div>
    </div>

    <style>
    html, body, [class*="css"] {
        margin: 0;
        padding: 0;
    }

    #intro-screen {
        width: 100%;
        height: 90vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #081c24, #0d2d38, #123d4a);
        border-radius: 24px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 0 30px rgba(0,0,0,0.25);
    }

    #main-text {
        font-family: 'Arial', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        letter-spacing: 2px;
        opacity: 0;
        animation: fadeIn 2s ease forwards;
    }

    #brand-text {
        font-family: 'Arial', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        color: #7fffd4;
        letter-spacing: 8px;
        opacity: 0;
        transform: scale(0.5);
        margin-top: 25px;
        animation: brandGrow 2.2s ease forwards;
        animation-delay: 2.2s;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(18px); }
        100% { opacity: 1; transform: translateY(0px); }
    }

    @keyframes brandGrow {
        0% { opacity: 0; transform: scale(0.4); }
        100% { opacity: 1; transform: scale(1.8); }
    }
    </style>
    """
    components.html(intro_html, height=650)

if not st.session_state.intro_shown:
    show_intro()
    time.sleep(4.8)
    st.session_state.intro_shown = True
    st.rerun()

# BAŞLIK

st.markdown(
    f"""
    <div style='text-align:center; padding-top: 10px; padding-bottom: 10px;'>
        <h1 style='margin-bottom:0; color:#0b3d3b;'>🧬 {T["main_title"]}</h1>
        <h3 style='margin-top:8px; color:#3d6b6b;'>{T["main_subtitle"]}</h3>
        <p style='font-size:16px; color:#4f6363;'>
            {T["main_desc"]}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# MODEL YOLLARI
MODEL_PATH = "best_densenet121.keras"
TRAIN_DIR  = "data/train"

# MODEL DRIVE'DAN İNDİR
import gdown
if not os.path.exists(MODEL_PATH):
    with st.spinner("Model yükleniyor, lütfen bekleyin..."):
        gdown.download(
            "https://drive.google.com/uc?id=1fwfLi4kgPYm3XNmWOF3LmYw82QFp5Bu2",
            MODEL_PATH,
            quiet=False
        )


# MODEL YÜKLEME
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()


# SINIF İSİMLERİ
@st.cache_data
def load_class_names(train_dir):
    raw_classes = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    return sorted(raw_classes)

CLASS_NAMES = load_class_names(TRAIN_DIR)
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

if "healthy" not in CLASS_NAMES_LOWER:
    st.error(T["healthy_not_found"])
    st.stop()


# SİSTEMİK RİSKLER

SYSTEMIC_RISKS = {
    "psoriasis": {
        "psoriatic_arthritis": 0.40,
        "psoriasis_vulgaris": 0.65,
        "metabolic_syndrome": 0.15,
        "cardiovascular_risk": 0.10
    },
    "acral_lentiginous_melanoma": {
        "alm_nail_involvement": 0.25,
        "alm_ethnic_prevalence": 0.30
    },
    "onychomycosis": {
        "diabetes": 0.25,
        "vascular_disease": 0.15,
        "advanced_age": 0.35,
        "immune_deficiency": 0.07
    },
    "clubbing": {
        "lung_disease": 0.50,
        "cardiovascular": 0.15,
        "liver_gi": 0.25,
        "endocrine": 0.10
    },
    "blue_finger": {
        "peripheral_cyanosis": 0.45,
        "cardiac_disease": 0.12,
        "pulmonary_disease": 0.12,
        "renal_hematologic": 0.07,
        "trauma": 0.28
    },
    "pitting": {
        "psoriasis": 0.75,
        "alopecia_areata": 0.15,
        "eczema_atopic_dermatitis": 0.15,
        "reiter_psoriatic_arthritis": 0.10
    }
}


# SİSTEMİK RİSK ETİKET ÇEVİRİLERİ

SYSTEMIC_RISK_LABELS = {
    "tr": {
        "psoriatic_arthritis": "Psoriatik artrit",
        "psoriasis_vulgaris": "Psoriasis vulgaris",
        "metabolic_syndrome": "Metabolik sendrom",
        "cardiovascular_risk": "Kardiyovasküler risk",
        "alm_nail_involvement": "ALM tırnak tutulumu",
        "alm_ethnic_prevalence": "ALM etnik prevalansı",
        "diabetes": "Diyabet",
        "vascular_disease": "Damar hastalığı",
        "advanced_age": "İleri yaş",
        "immune_deficiency": "Bağışıklık yetmezliği",
        "lung_disease": "Akciğer hastalığı",
        "cardiovascular": "Kardiyovasküler",
        "liver_gi": "Karaciğer / Gastrointestinal",
        "endocrine": "Endokrin",
        "peripheral_cyanosis": "Periferik siyanoz",
        "cardiac_disease": "Kalp hastalığı",
        "pulmonary_disease": "Akciğer hastalığı",
        "renal_hematologic": "Böbrek / Hematolojik",
        "trauma": "Travma",
        "psoriasis": "Psoriasis",
        "alopecia_areata": "Alopesi areata",
        "eczema_atopic_dermatitis": "Egzama / Atopik dermatit",
        "reiter_psoriatic_arthritis": "Reiter sendromu / Psoriatik artrit"
    },
    "en": {
        "psoriatic_arthritis": "Psoriatic arthritis",
        "psoriasis_vulgaris": "Psoriasis vulgaris",
        "metabolic_syndrome": "Metabolic syndrome",
        "cardiovascular_risk": "Cardiovascular risk",
        "alm_nail_involvement": "ALM nail involvement",
        "alm_ethnic_prevalence": "ALM ethnic prevalence",
        "diabetes": "Diabetes",
        "vascular_disease": "Vascular disease",
        "advanced_age": "Advanced age",
        "immune_deficiency": "Immune deficiency",
        "lung_disease": "Lung disease",
        "cardiovascular": "Cardiovascular",
        "liver_gi": "Liver / GI",
        "endocrine": "Endocrine",
        "peripheral_cyanosis": "Peripheral cyanosis",
        "cardiac_disease": "Cardiac disease",
        "pulmonary_disease": "Pulmonary disease",
        "renal_hematologic": "Renal / hematologic",
        "trauma": "Trauma",
        "psoriasis": "Psoriasis",
        "alopecia_areata": "Alopecia areata",
        "eczema_atopic_dermatitis": "Eczema / Atopic dermatitis",
        "reiter_psoriatic_arthritis": "Reiter syndrome / Psoriatic arthritis"
    },
    "es": {
        "psoriatic_arthritis": "Artritis psoriásica",
        "psoriasis_vulgaris": "Psoriasis vulgar",
        "metabolic_syndrome": "Síndrome metabólico",
        "cardiovascular_risk": "Riesgo cardiovascular",
        "alm_nail_involvement": "Afectación ungueal por ALM",
        "alm_ethnic_prevalence": "Prevalencia étnica de ALM",
        "diabetes": "Diabetes",
        "vascular_disease": "Enfermedad vascular",
        "advanced_age": "Edad avanzada",
        "immune_deficiency": "Inmunodeficiencia",
        "lung_disease": "Enfermedad pulmonar",
        "cardiovascular": "Cardiovascular",
        "liver_gi": "Hepático / Gastrointestinal",
        "endocrine": "Endocrino",
        "peripheral_cyanosis": "Cianosis periférica",
        "cardiac_disease": "Enfermedad cardíaca",
        "pulmonary_disease": "Enfermedad pulmonar",
        "renal_hematologic": "Renal / hematológico",
        "trauma": "Traumatismo",
        "psoriasis": "Psoriasis",
        "alopecia_areata": "Alopecia areata",
        "eczema_atopic_dermatitis": "Eccema / Dermatitis atópica",
        "reiter_psoriatic_arthritis": "Síndrome de Reiter / Artritis psoriásica"
    },
    "de": {
        "psoriatic_arthritis": "Psoriasis-Arthritis",
        "psoriasis_vulgaris": "Psoriasis vulgaris",
        "metabolic_syndrome": "Metabolisches Syndrom",
        "cardiovascular_risk": "Kardiovaskuläres Risiko",
        "alm_nail_involvement": "ALM-Nagelbeteiligung",
        "alm_ethnic_prevalence": "Ethnische Prävalenz von ALM",
        "diabetes": "Diabetes",
        "vascular_disease": "Gefäßerkrankung",
        "advanced_age": "Hohes Alter",
        "immune_deficiency": "Immunschwäche",
        "lung_disease": "Lungenerkrankung",
        "cardiovascular": "Kardiovaskulär",
        "liver_gi": "Leber / Gastrointestinal",
        "endocrine": "Endokrin",
        "peripheral_cyanosis": "Periphere Zyanose",
        "cardiac_disease": "Herzerkrankung",
        "pulmonary_disease": "Lungenerkrankung",
        "renal_hematologic": "Renal / Hämatologisch",
        "trauma": "Trauma",
        "psoriasis": "Psoriasis",
        "alopecia_areata": "Alopecia areata",
        "eczema_atopic_dermatitis": "Ekzem / Atopische Dermatitis",
        "reiter_psoriatic_arthritis": "Reiter-Syndrom / Psoriasis-Arthritis"
    },
    "az": {
        "psoriatic_arthritis": "Psoriaz artriti",
        "psoriasis_vulgaris": "Psoriaz vulgaris",
        "metabolic_syndrome": "Metabolik sindrom",
        "cardiovascular_risk": "Kardiovaskulyar risk",
        "alm_nail_involvement": "ALM dırnaq tutulması",
        "alm_ethnic_prevalence": "ALM etnik yayılması",
        "diabetes": "Diabet",
        "vascular_disease": "Damar xəstəliyi",
        "advanced_age": "Yaşlılıq",
        "immune_deficiency": "İmmun çatışmazlığı",
        "lung_disease": "Ağciyər xəstəliyi",
        "cardiovascular": "Kardiovaskulyar",
        "liver_gi": "Qaraciyər / Mədə-bağırsaq",
        "endocrine": "Endokrin",
        "peripheral_cyanosis": "Periferik sianoz",
        "cardiac_disease": "Ürək xəstəliyi",
        "pulmonary_disease": "Ağciyər xəstəliyi",
        "renal_hematologic": "Böyrək / Hematoloji",
        "trauma": "Travma",
        "psoriasis": "Psoriaz",
        "alopecia_areata": "Alopesiya areata",
        "eczema_atopic_dermatitis": "Ekzema / Atopik dermatit",
        "reiter_psoriatic_arthritis": "Reiter sindromu / Psoriaz artriti"
    }
}


# AÇIKLAMALAR

EXPLANATIONS = {
    "psoriasis": {
        "tr": "Tırnak lezyonları sedef hastalığında sık görülür. Çukurlaşma, onikoliz ve keratin birikimi gözlenebilir.",
        "en": "Nail lesions are common in psoriasis. Pitting, onycholysis, and keratin accumulation may be observed.",
        "es": "Las lesiones ungueales son comunes en la psoriasis. Pueden observarse hoyuelos, onicólisis y acumulación de queratina.",
        "de": "Nagelveränderungen treten bei Psoriasis häufig auf. Grübchen, Onycholyse und Keratinansammlungen können beobachtet werden.",
        "az": "Dırnaq lezyonları psoriazda tez-tez müşahidə olunur. Çuxurlaşma, onikoliz və keratin yığılması görülə bilər."
    },
    "acral_lentiginous_melanoma": {
        "tr": "Acral Lentiginous Melanoma, tırnak yatağı gibi bölgelerde görülen önemli bir melanom türüdür.",
        "en": "Acral Lentiginous Melanoma is an important melanoma subtype seen in regions such as the nail bed.",
        "es": "El melanoma lentiginoso acral es un subtipo importante de melanoma observado en regiones como el lecho ungueal.",
        "de": "Das akral-lentiginöse Melanom ist ein wichtiger Melanom-Subtyp, der unter anderem im Nagelbett auftreten kann.",
        "az": "Akral Lentiginöz Melanoma, dırnaq yatağı kimi sahələrdə görülən mühüm bir melanom növüdür."
    },
    "onychomycosis": {
        "tr": "Tırnak mantarı diyabet, ileri yaş ve dolaşım bozuklukları ile ilişkili olabilir.",
        "en": "Onychomycosis may be associated with diabetes, older age, and circulatory disorders.",
        "es": "La onicomicosis puede estar relacionada con diabetes, edad avanzada y trastornos circulatorios.",
        "de": "Onychomykose kann mit Diabetes, höherem Alter und Durchblutungsstörungen zusammenhängen.",
        "az": "Dırnaq göbələyi diabet, yaşlılıq və qan dövranı pozğunluqları ilə əlaqəli ola bilər."
    },
    "clubbing": {
        "tr": "Clubbing, akciğer ve kalp-damar hastalıkları ile ilişkili olabilir.",
        "en": "Clubbing may be associated with pulmonary and cardiovascular diseases.",
        "es": "El hipocratismo digital puede estar relacionado con enfermedades pulmonares y cardiovasculares.",
        "de": "Trommelschlegelfinger können mit Lungen- und Herz-Kreislauf-Erkrankungen zusammenhängen.",
        "az": "Clubbing ağciyər və ürək-damar xəstəlikləri ilə əlaqəli ola bilər."
    },
    "blue_finger": {
        "tr": "Mavi tırnak görünümü dolaşım veya oksijenlenme sorunları ile ilişkili olabilir.",
        "en": "Blue nail appearance may be associated with circulation or oxygenation problems.",
        "es": "La apariencia azulada de la uña puede estar relacionada con problemas de circulación u oxigenación.",
        "de": "Ein bläuliches Nagelbild kann mit Durchblutungs- oder Sauerstoffproblemen zusammenhängen.",
        "az": "Mavi dırnaq görünüşü qan dövranı və ya oksigenləşmə problemləri ilə əlaqəli ola bilər."
    },
    "healthy": {
        "tr": "Görüntü model tarafından sağlıklı sınıfa daha yakın bulunmuştur. Bu sonuç klinik tanı yerine geçmez.",
        "en": "The image was found closer to the healthy class by the model. This result does not replace a clinical diagnosis.",
        "es": "La imagen fue clasificada por el modelo como más cercana a la clase saludable. Este resultado no sustituye un diagnóstico clínico.",
        "de": "Das Bild wurde vom Modell näher an der gesunden Klasse eingeordnet. Dieses Ergebnis ersetzt keine klinische Diagnose.",
        "az": "Şəkil model tərəfindən sağlam sinfə daha yaxın tapılmışdır. Bu nəticə klinik diaqnozu əvəz etmir."
    },
    "pitting": {
        "tr": "Tırnak çukurlaşması, en sık sedef hastalığı (psoriasis) olmak üzere alopesi areata ve egzama gibi hastalıklarda görülen, tırnak yüzeyinde küçük çukurcuklarla karakterize bir bulgudur.",
        "de": "Pitting ist ein Befund, der am häufigsten bei Psoriasis sowie auch bei Alopecia areata und Ekzemen auftritt und durch kleine Grübchen auf der Nageloberfläche gekennzeichnet ist.",
        "es": "El pitting es un hallazgo que se observa con mayor frecuencia en la psoriasis, así como en la alopecia areata y el eccema, caracterizado por pequeñas depresiones en la superficie de la uña.",
        "en": "Pitting is a finding most commonly seen in psoriasis, as well as in alopecia areata and eczema, characterized by small depressions on the nail surface.",
        "az": "Dırnaq çuxurlaşması ən çox psoriazda, eləcə də alopesiya areata və ekzemada müşahidə olunan, dırnaq səthindəki kiçik çuxurcuqlarla xarakterizə olunan bir tapıntıdır."
    }
}


# GÖRSEL HAZIRLAMA

def load_and_prepare(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# TAHMİN

def predict_pipeline(img_array, healthy_threshold=0.50):
    preds = model.predict(img_array, verbose=0)[0]

    class_probs = {
        CLASS_NAMES_LOWER[i]: float(preds[i])
        for i in range(len(CLASS_NAMES_LOWER))
    }

    healthy_prob = class_probs["healthy"]
    harmful_prob = 1.0 - healthy_prob

    best_idx = int(np.argmax(preds))
    best_class = CLASS_NAMES_LOWER[best_idx]
    best_prob = float(preds[best_idx])

    if healthy_prob >= healthy_threshold:
        status = "Healthy"
        detected_class = "healthy"
        detected_prob = healthy_prob
        systemic_results = None
    else:
        status = "Harmful"

        if best_class == "healthy":
            non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
            detected_class = max(non_healthy, key=non_healthy.get)
            detected_prob = non_healthy[detected_class]
        else:
            detected_class = best_class
            detected_prob = best_prob

        systemic_map = SYSTEMIC_RISKS.get(detected_class, {})
        systemic_results = {
            k: detected_prob * v for k, v in systemic_map.items()
        } if systemic_map else None

    return {
        "status": status,
        "healthy_probability": healthy_prob,
        "harmful_probability": harmful_prob,
        "detailed_class": detected_class,
        "detailed_prob": detected_prob,
        "systemic": systemic_results,
        "class_probs": class_probs
    }


# ARAYÜZ

st.markdown("---")

uploaded = st.file_uploader(T["upload_label"], type=["jpg", "jpeg", "png"])

healthy_threshold = st.slider(
    T["slider_label"],
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

if uploaded is not None:
    st.image(uploaded, caption=T["uploaded_image"], use_container_width=True)
    img_arr = load_and_prepare(uploaded)

    with st.spinner(T["analyzing"]):
        result = predict_pipeline(img_arr, healthy_threshold)

    st.markdown(f"### {T['analysis_result']}")
    st.write(f"**{T['healthy_prob']}:** {result['healthy_probability']:.2%}")
    st.write(f"**{T['harmful_prob']}:** {result['harmful_probability']:.2%}")

    if result["status"] == "Healthy":
        st.success(T["healthy_msg"])
    else:
        st.error(T["harmful_msg"])

        disease = result["detailed_class"]
        st.write(f"### {T['detected_disease']}: **{disease.replace('_', ' ').title()}**")
        st.write(f"**{T['model_prob']}:** {result['detailed_prob']:.2%}")

        st.write(f"### {T['medical_explanation']}")
        st.write(EXPLANATIONS.get(disease, {}).get(lang, T["no_explanation"]))

        if result["systemic"]:
            st.write(f"### {T['risk_distribution']}")

            labels_raw = list(result["systemic"].keys())
            labels = [
                SYSTEMIC_RISK_LABELS.get(lang, {}).get(label, label.replace("_", " ").title())
                for label in labels_raw
            ]
            values = list(result["systemic"].values())

            total = sum(values)
            if total > 0:
                values = [v / total for v in values]
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(values, labels=labels, autopct="%1.1f%%")
                ax.axis("equal")
                st.pyplot(fig)

    with st.expander(T["all_probs"]):
        df = pd.DataFrame({
            T["class_col"]: CLASS_NAMES_LOWER,
            T["prob_col"]: [result["class_probs"][c] for c in CLASS_NAMES_LOWER]
        }).sort_values(by=T["prob_col"], ascending=False)

        st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption(T["warning"])
