import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json, os

st.set_page_config(
    page_title="DES Lithium Recovery Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container{padding-top:1.5rem}
    .badge-ok  {background:#d4edda;color:#155724;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-mid {background:#fff3cd;color:#856404;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-low {background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .li-card   {background:linear-gradient(135deg,#e3f2fd,#f1f8e9);border:1px solid #90caf9;
                border-radius:12px;padding:16px 20px;margin-bottom:10px}
    .tip-box   {background:#f1f3f5;border-left:3px solid #4c6ef5;border-radius:6px;
                padding:12px 16px;font-size:13px;line-height:1.8;margin-top:8px}
    .section-header{font-size:12px;font-weight:700;color:#6c757d;letter-spacing:.07em;
                    text-transform:uppercase;margin-bottom:6px}
    div[data-testid="stMetric"]{background:#f8f9fa;border-radius:10px;padding:10px 14px}
    .stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 16px}
    /* Tab colour differentiation */
    .stTabs [data-baseweb="tab-list"] button:nth-child(1){border-bottom:3px solid #4c6ef5}
    .stTabs [data-baseweb="tab-list"] button:nth-child(2){border-bottom:3px solid #37b24d}
    .stTabs [data-baseweb="tab-list"] button:nth-child(3){border-bottom:3px solid #f76707}
    .stTabs [data-baseweb="tab-list"] button:nth-child(4){border-bottom:3px solid #7950f2}
    .stTabs [data-baseweb="tab-list"] button:nth-child(5){border-bottom:3px solid #1098ad}
    .stTabs [data-baseweb="tab-list"] button:nth-child(6){border-bottom:3px solid #e64980}
    .stTabs [data-baseweb="tab-list"] button:nth-child(7){border-bottom:3px solid #f59f00}
    .stTabs [data-baseweb="tab-list"] button:nth-child(8){border-bottom:3px solid #74b816}
    .stTabs [data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"]{background:rgba(76,110,245,0.12);color:#4c6ef5;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"]{background:rgba(55,178,77,0.12);color:#37b24d;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"]{background:rgba(247,103,7,0.12);color:#f76707;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"]{background:rgba(121,80,242,0.12);color:#7950f2;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(5)[aria-selected="true"]{background:rgba(16,152,173,0.12);color:#1098ad;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(6)[aria-selected="true"]{background:rgba(230,73,128,0.12);color:#e64980;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(7)[aria-selected="true"]{background:rgba(245,159,0,0.12);color:#f59f00;font-weight:700}
    .stTabs [data-baseweb="tab-list"] button:nth-child(8)[aria-selected="true"]{background:rgba(116,184,22,0.12);color:#74b816;font-weight:700}
    /* Rec card styles */
    .rec-card{background:#1e293b;border-left:4px solid #4c6ef5;border-radius:10px;
              padding:14px 18px;margin-bottom:10px}
    .rec-card-green{background:#1e293b;border-left:4px solid #37b24d;border-radius:10px;
                    padding:14px 18px;margin-bottom:10px}
    .rec-card-orange{background:#1e293b;border-left:4px solid #f76707;border-radius:10px;
                     padding:14px 18px;margin-bottom:10px}
    .rec-card-red{background:#1e293b;border-left:4px solid #e64980;border-radius:10px;
                  padding:14px 18px;margin-bottom:10px}
    .rec-title{font-size:14px;font-weight:700;color:#f1f5f9;margin-bottom:4px}
    .rec-body{font-size:12px;color:#94a3b8;line-height:1.7}
    .tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;
         font-weight:600;margin-right:4px;margin-bottom:4px}
    .tag-blue{background:#1e3a5f;color:#60a5fa}
    .tag-green{background:#14532d;color:#4ade80}
    .tag-orange{background:#431407;color:#fb923c}
    .model-badge{background:#e8f5e9;color:#2e7d32;padding:5px 12px;border-radius:8px;
                 font-size:12px;font-weight:600;display:inline-block;margin-bottom:10px}
    .warn-badge {background:#fff8e1;color:#f57f17;padding:5px 12px;border-radius:8px;
                 font-size:12px;font-weight:600;display:inline-block;margin-bottom:10px;margin-left:8px}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=" Loading Li-specific XGBoost models...")
def load_models():
    from model_data import load_all_models, ENCODERS, MODEL_RESULTS
    models, le_hba, le_hbd = load_all_models()
    return models, le_hba, le_hbd, ENCODERS, MODEL_RESULTS

models, le_hba, le_hbd, ENCODERS, MODEL_RESULTS = load_models()

LI_HBAS = set(ENCODERS['li_hbas'])
LI_HBDS = set(ENCODERS['li_hbds'])
ALL_HBAS = sorted(le_hba.classes_.tolist())
ALL_HBDS = sorted(le_hbd.classes_.tolist())

# HBAs/HBDs known best for Li from LIB cathodes (literature-curated)
BEST_LI_HBAS = [h for h in ['choline chloride','lactic acid','citric acid',
    'betaine','nicotinamide','glycine','dl-tartaric acid','malic acid',
    'guanidine hydrochloride','acetylcholine chloride'] if h in le_hba.classes_]
BEST_LI_HBDS = [h for h in ['ethylene glycol','oxalic acid','levulinic acid',
    'glycerol','lactic acid','malonic acid','glutaric acid','dl-malic acid',
    'citric acid','dl-tartaric acid','urea','phenol','acetic acid'] if h in le_hbd.classes_]

# Li-specific source materials
LI_SOURCES = {
    "LIB cathode (NMC/NCA)":    {"bench_li": 95, "bench_co": 92, "bench_ni": 90, "bench_mn": 88},
    "LIB cathode (LFP)":        {"bench_li": 97, "bench_co":  0, "bench_ni":  0, "bench_mn":  0},
    "LIB cathode (LCO)":        {"bench_li": 98, "bench_co": 95, "bench_ni":  0, "bench_mn":  0},
    "LIB cathode (LMO)":        {"bench_li": 93, "bench_co":  0, "bench_ni":  0, "bench_mn": 91},
    "Mixed battery black mass":  {"bench_li": 88, "bench_co": 82, "bench_ni": 80, "bench_mn": 78},
    "Li-ion pouch cell":         {"bench_li": 91, "bench_co": 85, "bench_ni": 83, "bench_mn": 81},
    "EV battery module":         {"bench_li": 86, "bench_co": 80, "bench_ni": 78, "bench_mn": 75},
}

OX_BOOST  = {"None":0, "H₂O₂":9, "Citric acid":6, "Ascorbic acid":8, "FeSO₄":5}
ASSIST_B  = {"Conventional":0, "Microwave":8, "Ultrasound":6, "Microwave + Ultrasound":12}
SOLID_LIQ = {"1:10":1.0, "1:20":1.2, "1:50":1.5, "1:100":1.8}
PRETREAT  = {"None":0, "Calcination (500°C)":5, "Mechanical crushing":3,
             "Calcination + crushing":8, "Sieving (<75μm)":4}
LEACH_TIME= {"30 min":0.85, "60 min":1.0, "90 min":1.1, "120 min":1.15, "180 min":1.18}

REGEN_METHODS = {
    "Vacuum evaporation":          {"li_recovery":0.60, "cycles":6,  "energy":"medium"},
    "HBD replenishment":           {"li_recovery":0.70, "cycles":5,  "energy":"low"},
    "Evaporation + replenishment": {"li_recovery":0.85, "cycles":8,  "energy":"high"},
    "Anti-solvent precipitation":  {"li_recovery":0.88, "cycles":7,  "energy":"medium"},
    "Membrane separation":         {"li_recovery":0.80, "cycles":10, "energy":"high"},
    "No regeneration":             {"li_recovery":0.00, "cycles":1,  "energy":"none"},
}

LI_DES_SYSTEMS = [
    dict(name="ChCl : Oxalic acid (1:1)",    li_eff=97, cost=0.72, green=0.82, visc=45),
    dict(name="ChCl : Lactic acid (1:2)",    li_eff=95, cost=0.80, green=0.85, visc=38),
    dict(name="ChCl : Levulinic acid (1:2)", li_eff=98, cost=0.65, green=0.78, visc=112),
    dict(name="ChCl : Malonic acid (1:1)",   li_eff=96, cost=0.70, green=0.80, visc=89),
    dict(name="ChCl : Citric acid (1:1)",    li_eff=93, cost=0.68, green=0.83, visc=850),
    dict(name="Lactic acid : Betaine (3:1)", li_eff=94, cost=0.75, green=0.88, visc=386),
    dict(name="ChCl : Ethylene glycol (1:2)",li_eff=88, cost=0.90, green=0.90, visc=14),
    dict(name="ChCl : Glycerol (1:2)",       li_eff=86, cost=0.92, green=0.92, visc=113),
    dict(name="ChCl : Tartaric acid (1:1)",  li_eff=95, cost=0.71, green=0.81, visc=76),
    dict(name="ChCl : Glutaric acid (1:1)",  li_eff=94, cost=0.73, green=0.79, visc=65),
    dict(name="Betaine : Citric acid (1:1)", li_eff=92, cost=0.69, green=0.84, visc=920),
    dict(name="Nicotinamide : Oxalic (1:1)", li_eff=96, cost=0.60, green=0.76, visc=55),
]

# ── Prediction functions ───────────────────────────────────────────────────────
def predict_all(hba, hbd, ratio, temp_K, water):
    try:
        hba_enc    = le_hba.transform([hba])[0]
        hbd_enc    = le_hbd.transform([hbd])[0]
        is_li_hba  = int(hba in LI_HBAS)
        is_li_hbd  = int(hbd in LI_HBDS)
    except ValueError as e:
        return None, str(e)
    X = np.array([[hba_enc, hbd_enc, ratio, temp_K, water, is_li_hba, is_li_hbd]])
    preds = {}
    for target, (model, log_t) in models.items():
        val = float(model.predict(X)[0])
        val = np.expm1(val) if log_t else val
        # Redox potential is always negative (-1.9 to -0.9 V) — do NOT clamp to 0
        if target == 'reduction_potential_V_vs_SHE':
            preds[target] = round(float(val), 4)
        else:
            preds[target] = max(0, round(float(val), 4))
    return preds, None

def predict_li_recovery(preds, temp_K, source, oxidant, assist, sl_ratio, pretreat, leach_time):
    """Combine XGBoost-predicted properties with process parameters for Li recovery %."""
    src = LI_SOURCES[source]
    base = src["bench_li"]

    visc   = preds['viscosity_cP']
    ph     = preds['pH_acidity']
    redox  = preds['reduction_potential_V_vs_SHE']
    li_sc  = preds.get('li_extraction_score', 60)

    # Viscosity penalty: >200 cP reduces mass transfer
    visc_factor = max(0.70, 1 - max(0, visc - 50) / 3000)
    # pH bonus: pH < 3 dissolves Li oxides faster
    ph_factor   = 1.0 + max(0, (3 - ph) * 0.03) if ph < 5 else max(0.85, 1 - (ph-5)*0.04)
    # Redox: more negative helps reduce Co3+→Co2+ releasing Li
    redox_factor = 1.0 + max(0, -redox - 0.9) * 0.04  # bonus for redox < -0.9 V
    # Temperature: optimal 60-90°C
    temp_C = temp_K - 273.15
    temp_factor = 0.82 + min(0.18, (temp_C - 20) / 400)
    # Process boosts
    ox_boost  = OX_BOOST[oxidant] / 100
    as_boost  = ASSIST_B[assist]  / 100
    sl_factor = SOLID_LIQ[sl_ratio]
    pt_boost  = PRETREAT[pretreat] / 100
    lt_factor = LEACH_TIME[leach_time]

    # Li extraction score from dedicated model (0-100)
    li_score_weight = li_sc / 100

    eff = (base * visc_factor * ph_factor * redox_factor * temp_factor * sl_factor * lt_factor
           * (1 + ox_boost) * (1 + as_boost) * (1 + pt_boost))
    # Blend with dedicated Li model score
    eff = 0.65 * eff + 0.35 * (li_score_weight * base)
    return min(99.5, max(20, round(eff, 1)))

def sweep_temps(hba, hbd, ratio, water):
    temps = np.arange(278.15, 378.15, 5)
    results = {'temps_C': (temps - 273.15).tolist()}
    try:
        hba_enc   = le_hba.transform([hba])[0]
        hbd_enc   = le_hbd.transform([hbd])[0]
        is_li_hba = int(hba in LI_HBAS)
        is_li_hbd = int(hbd in LI_HBDS)
    except ValueError:
        return None
    X = np.array([[hba_enc, hbd_enc, ratio, t, water, is_li_hba, is_li_hbd] for t in temps])
    for prop in ['viscosity_cP', 'pH_acidity', 'density_kg_m3', 'li_extraction_score']:
        model, log_t = models[prop]
        vals = model.predict(X)
        if prop == 'reduction_potential_V_vs_SHE':
            results[prop] = [float(v) for v in vals]
        else:
            results[prop] = [max(0, float(np.expm1(v) if log_t else v)) for v in vals]
    return results

def mc_uncertainty(hba, hbd, ratio, temp_K, water, n=80):
    samples = {'viscosity_cP': [], 'li_extraction_score': []}
    for _ in range(n):
        p, _ = predict_all(hba, hbd,
            ratio  * (1 + np.random.normal(0, 0.03)),
            temp_K + np.random.normal(0, 2),
            float(np.clip(water + np.random.normal(0, 0.01), 0, 1)))
        if p:
            samples['viscosity_cP'].append(p['viscosity_cP'])
            samples['li_extraction_score'].append(p.get('li_extraction_score', 0))
    return samples

def get_feature_importance():
    names = ['HBA identity','HBD identity','Molar ratio','Temperature','Water content',
             'Is Li-HBA','Is Li-HBD']
    fi = {}
    for target, (model, _) in models.items():
        fi[target] = dict(zip(names, model.feature_importances_.tolist()))
    return fi

def is_pareto(systems):
    flags = []
    for i, s in enumerate(systems):
        dominated = any(
            j!=i and o["li_eff"]>=s["li_eff"] and o["cost"]<=s["cost"]
            and o["green"]>=s["green"]
            and (o["li_eff"]>s["li_eff"] or o["cost"]<s["cost"] or o["green"]>s["green"])
            for j,o in enumerate(systems))
        flags.append(not dominated)
    return flags

def build_li_tips(preds, temp_K, source, oxidant, assist, pretreat):
    tips = []
    visc  = preds['viscosity_cP']
    ph    = preds['pH_acidity']
    redox = preds['reduction_potential_V_vs_SHE']
    li_sc = preds.get('li_extraction_score', 0)
    temp_C = temp_K - 273.15

    if visc > 200:
        tips.append(f" **High viscosity ({visc:.0f} cP)** — add 10–20% water or raise temperature to improve Li⁺ ion mobility and leaching kinetics.")
    elif visc < 30:
        tips.append(f"**Excellent viscosity ({visc:.0f} cP)** — good Li⁺ mass transfer expected.")
    if ph > 4:
        tips.append(f"**pH {ph:.1f} too high** — switch to a more acidic HBD (oxalic acid, malonic acid) to dissolve Li₂CO₃ and LiCoO₂ effectively.")
    elif ph < 2:
        tips.append(f"**Highly acidic (pH {ph:.1f})** — excellent for dissolving Li oxide phases.")
    if redox > -1.0:
        tips.append(f"**Reduction potential weak ({redox:.2f} V vs SHE)** — add H₂O₂ or ascorbic acid to reduce Co³⁺→Co²⁺ and release bound Li⁺.")
    elif redox < -1.5:
        tips.append(f"**Strong reducing conditions ({redox:.2f} V vs SHE)** — excellent for Co³⁺ reduction and Li⁺ release from cathode lattice.")
    if temp_C < 50:
        tips.append(f"**Temperature {temp_C:.0f}°C is low** — raise to 60–90°C for optimal Li leaching rate from cathode materials.")
    if oxidant == "None" and "NMC" in source or "NCA" in source or "LCO" in source:
        tips.append("**Add reductant** — H₂O₂ or ascorbic acid significantly boosts Li recovery from NMC/NCA/LCO by reducing transition metals.")
    if pretreat == "None":
        tips.append("**Calcination pre-treatment** at 500°C removes organic binder (PVDF) and improves cathode dissolution by 5–8%.")
    if li_sc < 50:
        tips.append(f"**Li suitability score low ({li_sc:.0f}/100)** — consider ChCl:Oxalic acid or ChCl:Levulinic acid which score >75.")
    if not tips:
        tips.append("**Optimal conditions** — this DES is well-suited for Li extraction from LIB cathodes.")
    return tips[:4]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Li Recovery Predictor")
    st.markdown("XGBoost models trained on **5,790 DES data points** with Li-specific feature engineering.")
    st.divider()

    filter_li = st.toggle("Show Li-optimised compounds only", value=True)
    hba_opts = BEST_LI_HBAS if filter_li else ALL_HBAS
    hbd_opts = BEST_LI_HBDS if filter_li else ALL_HBDS

    hba = st.selectbox("HBA (hydrogen bond acceptor)",
                        hba_opts,
                        index=hba_opts.index('choline chloride') if 'choline chloride' in hba_opts else 0)
    hbd = st.selectbox("HBD (hydrogen bond donor)",
                        hbd_opts,
                        index=hbd_opts.index('oxalic acid') if 'oxalic acid' in hbd_opts else 0)
    ratio = st.slider("Molar ratio HBA:HBD", 0.05, 10.0, 1.0, 0.05)
    water = st.slider("Water content (mol fraction)", 0.0, 0.50, 0.0, 0.01)

    st.markdown("### Process Conditions")
    temp_C_val = st.slider("Temperature (°C)", 20, 120, 70, 5)
    temp_K_val = temp_C_val + 273.15

    st.markdown("### Li Recovery Parameters")
    source   = st.selectbox("Battery source material", list(LI_SOURCES.keys()))
    oxidant  = st.selectbox("Reductant/oxidant additive", list(OX_BOOST.keys()))
    assist   = st.selectbox("Assist method", list(ASSIST_B.keys()))
    sl_ratio = st.selectbox("Solid:Liquid ratio", list(SOLID_LIQ.keys()), index=1)
    pretreat = st.selectbox("Pre-treatment", list(PRETREAT.keys()))
    leach_t  = st.selectbox("Leaching time", list(LEACH_TIME.keys()), index=1)

# ── Run predictions ────────────────────────────────────────────────────────────
preds, err = predict_all(hba, hbd, ratio, temp_K_val, water)
if err:
    st.error(f"Prediction error: {err}"); st.stop()

visc_pred  = preds['viscosity_cP']
dens_pred  = preds['density_kg_m3']
ph_pred    = preds['pH_acidity']
redox_pred = preds['reduction_potential_V_vs_SHE']
li_score   = preds.get('li_extraction_score', 0)
li_rec     = predict_li_recovery(preds, temp_K_val, source, oxidant, assist,
                                  sl_ratio, pretreat, leach_t)
bench_li   = LI_SOURCES[source]["bench_li"]
tips       = build_li_tips(preds, temp_K_val, source, oxidant, assist, pretreat)
fi         = get_feature_importance()
mc         = mc_uncertainty(hba, hbd, ratio, temp_K_val, water)

grade_css  = "badge-ok" if li_rec>=90 else ("badge-mid" if li_rec>=75 else "badge-low")
grade      = "Pass Excellent" if li_rec>=90 else (" Acceptable" if li_rec>=75 else "Fail Improve conditions")
li_flag    = "Li-optimised" if (hba in LI_HBAS and hbd in LI_HBDS) else " Not Li-specific"
li_flag_css = "model-badge" if (hba in LI_HBAS and hbd in LI_HBDS) else "warn-badge"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# DES Lithium Recovery Predictor")
st.markdown("**5 XGBoost models** · Viscosity · Density · pH · Redox potential · **Li extraction score** · Trained on 5,790 real data points")
col_b1, col_b2 = st.columns([1,5])
with col_b1:
    st.markdown(f'<span class="model-badge">XGBoost v2 — Li-specific</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown(f'<span class="{li_flag_css}">{li_flag}</span>', unsafe_allow_html=True)

tabs = st.tabs(["Li Recovery","DES Properties","Feature Importance",
                "Property Sweep","Reuse Simulator","DES Ranker","Recommendations","Model Info"])

# ════════ TAB 1 — LI RECOVERY (main tab) ════════
with tabs[0]:
    # Top metrics row
    delta_col = "green" if li_rec >= bench_li else "red"
    delta_sym  = "▲" if li_rec >= bench_li else "▼"
    visc_col   = "#37b24d" if visc_pred < 100 else ("#f76707" if visc_pred < 300 else "#f03e3e")
    ph_col     = "#37b24d" if ph_pred < 3 else ("#f76707" if ph_pred < 5 else "#f03e3e")
    redox_col  = "#37b24d" if redox_pred < -1.3 else ("#f76707" if redox_pred < -1.0 else "#f03e3e")
    li_sc_col  = "#37b24d" if li_score >= 70 else ("#f76707" if li_score >= 50 else "#f03e3e")

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">LI RECOVERY</div>
        <div style="font-size:28px;font-weight:700;color:#f1f5f9">{li_rec:.1f}<span style="font-size:16px">%</span></div>
        <div style="font-size:12px;color:{delta_col};margin-top:4px">{delta_sym} {abs(li_rec-bench_li):.1f}% vs benchmark ({bench_li}%)</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">LI SUIT. SCORE</div>
        <div style="font-size:28px;font-weight:700;color:{li_sc_col}">{li_score:.1f}<span style="font-size:14px;color:#64748b">/100</span></div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">XGBoost Li model</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">VISCOSITY</div>
        <div style="font-size:28px;font-weight:700;color:{visc_col}">{visc_pred:.0f}<span style="font-size:14px"> cP</span></div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">{"Good" if visc_pred<100 else ("Moderate" if visc_pred<300 else "High")}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">pH</div>
        <div style="font-size:28px;font-weight:700;color:{ph_col}">{ph_pred:.2f}</div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">{"Acidic (ideal)" if ph_pred<3 else ("Moderate" if ph_pred<6 else "Basic")}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">REDOX POTENTIAL</div>
        <div style="font-size:28px;font-weight:700;color:{redox_col}">{redox_pred:.3f}<span style="font-size:14px"> V</span></div>
        <div style="font-size:12px;color:{redox_col};margin-top:4px">{"Reducing" if redox_pred < -1.2 else ("Moderate" if redox_pred < -0.9 else "Weak")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<span class="{grade_css}">{grade}</span>', unsafe_allow_html=True)
    st.markdown(f"**Benchmark for {source}:** {bench_li}%")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Li recovery gauge
        st.markdown('<div class="section-header">Li recovery gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=li_rec,
            delta={'reference': bench_li, 'valueformat': '.1f', 'suffix': '%'},
            title={'text': f"Li Recovery vs Benchmark ({bench_li}%)", 'font': {'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#4c6ef5"},
                'steps': [
                    {'range': [0,  60], 'color': "#fee2e2"},
                    {'range': [60, 75], 'color': "#fef9c3"},
                    {'range': [75,100], 'color': "#dcfce7"},
                ],
                'threshold': {'line': {'color': "#ef4444", 'width': 3},
                              'thickness': 0.8, 'value': bench_li}
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10),
                                 paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Li suitability breakdown
        st.markdown('<div class="section-header">Li suitability components</div>', unsafe_allow_html=True)
        visc_s  = max(0, min(100, (1 - max(0, visc_pred-50)/3000)*100))
        ph_s    = max(0, min(100, (1 - max(0, ph_pred-1)/7)*100))
        redox_s = max(0, min(100, (-redox_pred - 0.9) / 1.0 * 100))  # range -1.9 to -0.9
        comp_df = pd.DataFrame({
            'Component': ['Viscosity score', 'pH score', 'Redox score'],
            'Value': [round(visc_s,1), round(ph_s,1), round(redox_s,1)],
            'Weight': ['35%', '40%', '25%'],
            'Status': [
                'Pass' if visc_s > 70 else ('' if visc_s > 40 else 'Fail'),
                'Pass' if ph_s   > 70 else ('' if ph_s   > 40 else 'Fail'),
                'Pass' if redox_s> 50 else ('' if redox_s> 25 else 'Fail'),
            ]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with col2:
        # Optimisation tips
        st.markdown('<div class="section-header">Li-specific optimisation advice</div>', unsafe_allow_html=True)
        tip_html = "".join(f"<p style='margin-bottom:8px'>{t}</p>" for t in tips)
        st.markdown(f'<div class="tip-box">{tip_html}</div>', unsafe_allow_html=True)

        # Co/Ni/Mn co-recovery
        st.markdown('<div class="section-header">Co-metal recovery estimates</div>', unsafe_allow_html=True)
        src_data = LI_SOURCES[source]
        metals = {'Li': li_rec, 'Co': src_data['bench_co'], 'Ni': src_data['bench_ni'], 'Mn': src_data['bench_mn']}
        metals = {k: v for k,v in metals.items() if v > 0}
        fig_metals = go.Figure(go.Bar(
            x=list(metals.keys()), y=list(metals.values()),
            marker_color=['#4c6ef5','#37b24d','#f76707','#7950f2'][:len(metals)],
            text=[f"{v:.0f}%" for v in metals.values()], textposition='outside'
        ))
        fig_metals.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                  yaxis=dict(range=[0,110], ticksuffix="%"),
                                  xaxis_title="Metal", yaxis_title="Est. Recovery %",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_metals, use_container_width=True)

    st.divider()

    # Sensitivity analysis for Li recovery
    st.markdown('<div class="section-header">Sensitivity — impact on Li recovery (%)</div>', unsafe_allow_html=True)

    def li_rec_for(hba2=hba, hbd2=hbd, ratio2=ratio, temp2=temp_K_val,
                   water2=water, ox2=oxidant, as2=assist, sl2=sl_ratio,
                   pt2=pretreat, lt2=leach_t):
        p2, _ = predict_all(hba2, hbd2, ratio2, temp2, water2)
        if p2 is None: return li_rec
        return predict_li_recovery(p2, temp2, source, ox2, as2, sl2, pt2, lt2)

    sens = {
        "Temp +20°C":           li_rec_for(temp2=temp_K_val+20) - li_rec,
        "Water +0.1 mol frac":  li_rec_for(water2=min(0.5, water+0.1)) - li_rec,
        "Add H₂O₂ reductant":   li_rec_for(ox2="H₂O₂") - li_rec,
        "Add microwave assist":  li_rec_for(as2="Microwave") - li_rec,
        "Calcination pretreat":  li_rec_for(pt2="Calcination (500°C)") - li_rec,
        "Extend to 120 min":     li_rec_for(lt2="120 min") - li_rec,
        "Ratio ×2":              li_rec_for(ratio2=ratio*2) - li_rec,
    }
    fig_sens = go.Figure(go.Bar(
        x=list(sens.values()), y=list(sens.keys()), orientation="h",
        marker_color=["#37b24d" if v>=0 else "#f03e3e" for v in sens.values()],
        text=[f"{v:+.1f}%" for v in sens.values()], textposition="outside"
    ))
    fig_sens.update_layout(height=260, margin=dict(l=0,r=80,t=10,b=10),
                            xaxis_title="Δ Li recovery (%)",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sens, use_container_width=True)

    # MC uncertainty
    st.markdown('<div class="section-header">Monte Carlo uncertainty (80 passes, ±3% input noise)</div>',
                unsafe_allow_html=True)
    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        if mc['viscosity_cP']:
            mc_s = sorted(mc['viscosity_cP'])
            fig_mc = px.histogram(x=mc['viscosity_cP'], nbins=15, color_discrete_sequence=["#74c0fc"])
            fig_mc.add_vline(x=visc_pred, line_dash="dash", line_color="#4c6ef5",
                             annotation_text=f"{visc_pred:.0f} cP")
            fig_mc.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                                  showlegend=False, xaxis_title="Viscosity (cP)",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(f"90% CI: {mc_s[int(len(mc_s)*0.05)]:.0f} – {mc_s[int(len(mc_s)*0.95)]:.0f} cP")
    with col_mc2:
        if mc['li_extraction_score']:
            mc_ls = sorted(mc['li_extraction_score'])
            fig_mc2 = px.histogram(x=mc['li_extraction_score'], nbins=15, color_discrete_sequence=["#a9e34b"])
            fig_mc2.add_vline(x=li_score, line_dash="dash", line_color="#37b24d",
                              annotation_text=f"{li_score:.1f}/100")
            fig_mc2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                                   showlegend=False, xaxis_title="Li suitability score",
                                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mc2, use_container_width=True)
            st.caption(f"90% CI: {mc_ls[int(len(mc_ls)*0.05)]:.1f} – {mc_ls[int(len(mc_ls)*0.95)]:.1f}")


# ════════ NEW — PROCESS OPTIMISER & DEEP ANALYSIS (still inside Tab 1) ════════

    st.markdown("---")
    st.markdown("### Deep Analysis & Optimisation")

    ana_tabs = st.tabs(["Process Window","Condition Heatmap",
                         "Co-Metal Selectivity","Leaching Kinetics","DES Comparison"])

    # ── Analysis sub-tab 1: Process Window ─────────────────────────────────────
    with ana_tabs[0]:
        st.markdown("#### Optimal operating window for maximum Li recovery")
        temps_grid  = np.arange(30, 115, 5)
        waters_grid = np.linspace(0, 0.45, 10)
        z_rec = []
        for wg in waters_grid:
            row = []
            for tg in temps_grid:
                pg, _ = predict_all(hba, hbd, ratio, tg + 273.15, float(wg))
                if pg:
                    row.append(predict_li_recovery(pg, tg+273.15, source,
                                                    oxidant, assist, sl_ratio, pretreat, leach_t))
                else:
                    row.append(0)
            z_rec.append(row)

        fig_hw = go.Figure(go.Heatmap(
            x=temps_grid.tolist(),
            y=[f"{w:.2f}" for w in waters_grid],
            z=z_rec,
            colorscale="RdYlGn",
            zmin=55, zmax=99,
            colorbar=dict(title="Li Recovery %", ticksuffix="%"),
            hovertemplate="Temp: %{x}°C<br>Water: %{y} mol frac<br>Li Recovery: %{z:.1f}%<extra></extra>"
        ))
        fig_hw.add_scatter(x=[temp_C_val], y=[f"{water:.2f}"],
                           mode="markers", marker=dict(color="white", size=12,
                           symbol="star", line=dict(color="#4c6ef5", width=2)),
                           name="Current conditions", showlegend=True)
        fig_hw.update_layout(
            title="Li Recovery (%) — Temperature vs Water Content Heatmap",
            xaxis_title="Temperature (°C)", yaxis_title="Water content (mol frac)",
            height=360, margin=dict(l=0,r=0,t=50,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_hw, use_container_width=True)

        # Find best conditions
        best_val = 0; best_t = temp_C_val; best_w = water
        for wi, wg in enumerate(waters_grid):
            for ti, tg in enumerate(temps_grid):
                if z_rec[wi][ti] > best_val:
                    best_val = z_rec[wi][ti]; best_t = tg; best_w = wg
        st.success(f"**Optimal window:** {best_t}°C · {best_w:.2f} mol frac water → **{best_val:.1f}% Li recovery** "
                   f"(current: {li_rec:.1f}% at {temp_C_val}°C, {water:.2f} mol frac)")

    # ── Analysis sub-tab 2: Condition Heatmap (ratio vs temp) ──────────────────
    with ana_tabs[1]:
        st.markdown("#### Li recovery vs Molar Ratio × Temperature")
        ratios_g = np.linspace(0.25, 6.0, 12)
        temps_g2 = np.arange(40, 105, 5)
        z_rt = []
        for rg in ratios_g:
            row = []
            for tg in temps_g2:
                pg, _ = predict_all(hba, hbd, float(rg), tg+273.15, water)
                if pg:
                    row.append(predict_li_recovery(pg, tg+273.15, source,
                                                    oxidant, assist, sl_ratio, pretreat, leach_t))
                else:
                    row.append(0)
            z_rt.append(row)

        fig_rt = go.Figure(go.Heatmap(
            x=temps_g2.tolist(),
            y=[f"{r:.2f}" for r in ratios_g],
            z=z_rt,
            colorscale="Viridis",
            colorbar=dict(title="Li Recovery %", ticksuffix="%"),
            hovertemplate="Temp: %{x}°C<br>Ratio: %{y}<br>Li Recovery: %{z:.1f}%<extra></extra>"
        ))
        fig_rt.add_scatter(x=[temp_C_val], y=[f"{ratio:.2f}"],
                           mode="markers", marker=dict(color="red", size=12, symbol="star"),
                           name="Current", showlegend=True)
        fig_rt.update_layout(
            title="Li Recovery (%) — Molar Ratio vs Temperature",
            xaxis_title="Temperature (°C)", yaxis_title="HBA:HBD Molar Ratio",
            height=360, margin=dict(l=0,r=0,t=50,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rt, use_container_width=True)

        col_rt1, col_rt2 = st.columns(2)
        with col_rt1:
            best_rt = 0; best_r2 = ratio; best_t2 = temp_C_val
            for ri, rg in enumerate(ratios_g):
                for ti, tg in enumerate(temps_g2):
                    if z_rt[ri][ti] > best_rt:
                        best_rt = z_rt[ri][ti]; best_r2 = rg; best_t2 = tg
            st.info(f"**Best ratio:temp combo:** {best_r2:.2f} ratio at {best_t2}°C → **{best_rt:.1f}% Li**")
        with col_rt2:
            # Show how current ratio compares at current temp
            ratio_scan = [predict_all(hba, hbd, float(r), temp_K_val, water)[0] for r in ratios_g]
            rec_scan   = [predict_li_recovery(p, temp_K_val, source, oxidant, assist,
                                               sl_ratio, pretreat, leach_t) if p else 0
                          for p in ratio_scan]
            fig_rs = go.Figure(go.Scatter(x=ratios_g.tolist(), y=rec_scan,
                                           mode='lines+markers',
                                           line=dict(color='#4c6ef5', width=2.5)))
            fig_rs.add_vline(x=ratio, line_dash='dash', line_color='#f03e3e',
                              annotation_text=f"Current {ratio:.2f}")
            fig_rs.update_layout(title=f"Li Recovery vs Ratio @ {temp_C_val}°C",
                                  height=220, margin=dict(l=0,r=0,t=40,b=10),
                                  xaxis_title="Molar ratio", yaxis_title="Li Recovery %",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rs, use_container_width=True)

    # ── Analysis sub-tab 3: Co-Metal Selectivity ────────────────────────────────
    with ana_tabs[2]:
        st.markdown("#### Predicted metal recovery & Li selectivity")
        src_d = LI_SOURCES[source]

        # Li selectivity = Li recovery / average of co-metals
        co_metals = {k:v for k,v in {
            'Co': src_d['bench_co'], 'Ni': src_d['bench_ni'], 'Mn': src_d['bench_mn']
        }.items() if v > 0}
        avg_co = np.mean(list(co_metals.values())) if co_metals else 0

        # Scale co-metals by same factors affecting Li (viscosity, temp, pH)
        visc_f = max(0.72, 1 - max(0, visc_pred-50)/3000)
        temp_f = 0.82 + min(0.18, (temp_C_val-20)/400)
        metals_pred = {'Li': li_rec}
        for m, bench in co_metals.items():
            metals_pred[m] = min(99, round(bench * visc_f * temp_f * (1+OX_BOOST[oxidant]/100), 1))

        selectivity_li = li_rec / (avg_co * visc_f * temp_f + 0.01) if avg_co > 0 else 999

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_sel = go.Figure()
            metal_colors = {'Li':'#4c6ef5','Co':'#37b24d','Ni':'#f76707','Mn':'#7950f2'}
            for metal, val in metals_pred.items():
                fig_sel.add_trace(go.Bar(
                    name=metal, x=[metal], y=[val],
                    marker_color=metal_colors.get(metal,'#adb5bd'),
                    text=[f"{val:.1f}%"], textposition='outside'
                ))
            fig_sel.update_layout(
                title="Predicted Metal Recovery from " + source,
                height=300, margin=dict(l=0,r=0,t=50,b=10),
                yaxis=dict(range=[0,110], ticksuffix="%"),
                showlegend=False, barmode='group',
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sel, use_container_width=True)

        with col_s2:
            # Radar of Li vs co-metals
            if co_metals:
                cat_sel = ['Li'] + list(co_metals.keys())
                val_sel = [metals_pred['Li']] + [metals_pred.get(m, 0) for m in co_metals]
                fig_rad = go.Figure(go.Scatterpolar(
                    r=val_sel+[val_sel[0]], theta=cat_sel+[cat_sel[0]],
                    fill='toself', fillcolor='rgba(76,110,245,0.25)',
                    line=dict(color='#4c6ef5', width=2.5)))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(range=[0,110])),
                    title="Metal Recovery Radar",
                    height=280, margin=dict(l=20,r=20,t=50,b=10),
                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_rad, use_container_width=True)

        # Selectivity stats
        sel_cols = st.columns(len(metals_pred))
        for col_i, (metal, val) in zip(sel_cols, metals_pred.items()):
            col_i.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:12px;text-align:center">
              <div style="font-size:11px;color:#94a3b8;font-weight:600">{metal}</div>
              <div style="font-size:24px;font-weight:700;color:#f1f5f9">{val:.1f}%</div>
              <div style="font-size:11px;color:#64748b">{"Li target" if metal=="Li" else "co-leached"}</div>
            </div>""", unsafe_allow_html=True)

        if selectivity_li < 1.1:
            st.warning(f"**Low Li selectivity ({selectivity_li:.2f}×)** — co-metals leaching at similar rate. "
                       "Consider lower temperature or adjusted pH to improve separation.")
        elif selectivity_li > 1.2:
            st.success(f"**Good Li selectivity ({selectivity_li:.2f}×)** — Li leaching preferentially over co-metals.")

    # ── Analysis sub-tab 4: Leaching Kinetics ──────────────────────────────────
    with ana_tabs[3]:
        st.markdown("#### Simulated Li leaching kinetics over time")
        st.caption("Based on first-order kinetic model calibrated to viscosity and temperature predictions.")

        times = np.linspace(0, 180, 50)
        # Rate constant depends on viscosity (diffusion-controlled) and temperature (Arrhenius)
        k_base  = 0.035
        visc_k  = k_base * max(0.3, 1 - (visc_pred - 25) / 2000)
        temp_k  = visc_k * np.exp(0.02 * (temp_C_val - 25))
        li_kin  = li_rec * (1 - np.exp(-temp_k * times))

        # Compare across temps
        kin_fig = go.Figure()
        comp_temps = [40, 60, 80, 100]
        colors_k = ['#94a3b8','#60a5fa','#4c6ef5','#1d4ed8']
        for ct, ck_col in zip(comp_temps, colors_k):
            pg_k, _ = predict_all(hba, hbd, ratio, ct+273.15, water)
            if pg_k:
                rec_k = predict_li_recovery(pg_k, ct+273.15, source,
                                             oxidant, assist, sl_ratio, pretreat, leach_t)
                visc_k2 = k_base * max(0.3, 1 - (pg_k['viscosity_cP']-25)/2000)
                temp_k2 = visc_k2 * np.exp(0.02*(ct-25))
                y_k = rec_k * (1 - np.exp(-temp_k2 * times))
                width = 2.5 if ct == temp_C_val else 1.5
                dash  = "solid" if ct == temp_C_val else "dot"
                kin_fig.add_trace(go.Scatter(
                    x=times, y=y_k, mode='lines',
                    name=f"{ct}°C ({rec_k:.0f}% final)",
                    line=dict(color=ck_col, width=width, dash=dash)))

        # Add leach time markers
        for lt_label, lt_factor in LEACH_TIME.items():
            lt_mins = int(lt_label.split()[0])
            kin_fig.add_vline(x=lt_mins, line_dash="dot", line_color="#475569",
                               line_width=1,
                               annotation_text=lt_label,
                               annotation_font_size=10)

        kin_fig.add_hline(y=li_rec, line_dash="dash", line_color="#f03e3e",
                           annotation_text=f"Current: {li_rec:.1f}%",
                           annotation_font_size=10)
        kin_fig.update_layout(
            title=f"Li Leaching Kinetics — {hba} : {hbd}",
            xaxis_title="Time (min)", yaxis_title="Li Leaching Efficiency (%)",
            height=340, margin=dict(l=0,r=0,t=50,b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0,105])
        )
        st.plotly_chart(kin_fig, use_container_width=True)

        t90 = -np.log(1 - 0.9/li_rec * (li_rec/li_rec)) / temp_k if li_rec > 0 else 999
        t90_approx = -np.log(0.1) / temp_k if temp_k > 0 else 999
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">EST. TIME TO 90% OF MAX</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9">{min(180,int(t90_approx))} min</div>
        </div>""", unsafe_allow_html=True)
        col_k2.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">RATE CONSTANT k</div>
          <div style="font-size:22px;font-weight:700;color:#60a5fa">{temp_k:.4f} min⁻¹</div>
        </div>""", unsafe_allow_html=True)
        col_k3.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">KINETIC CONTROL</div>
          <div style="font-size:16px;font-weight:700;color:#f1f5f9">
            {"Diffusion" if visc_pred>200 else "Mixed" if visc_pred>80 else "Reaction"}</div>
          <div style="font-size:11px;color:#64748b">(based on viscosity)</div>
        </div>""", unsafe_allow_html=True)

    # ── Analysis sub-tab 5: DES Comparison ─────────────────────────────────────
    with ana_tabs[4]:
        st.markdown("#### Compare current DES against top Li candidates")
        compare_systems = [
            ("choline chloride", "oxalic acid",    1.0),
            ("choline chloride", "levulinic acid", 2.0),
            ("choline chloride", "malonic acid",   1.0),
            ("choline chloride", "lactic acid",    2.0),
            ("choline chloride", "dl-malic acid",  1.0),
            ("choline chloride", "ethylene glycol",2.0),
            ("lactic acid",      "betaine",        3.0),
        ]
        compare_data = []
        for c_hba, c_hbd, c_ratio in compare_systems:
            if c_hba in le_hba.classes_ and c_hbd in le_hbd.classes_:
                cp, _ = predict_all(c_hba, c_hbd, c_ratio, temp_K_val, water)
                if cp:
                    c_rec = predict_li_recovery(cp, temp_K_val, source,
                                                 oxidant, assist, sl_ratio, pretreat, leach_t)
                    compare_data.append({
                        'System': f"{c_hba.replace('choline chloride','ChCl')} : {c_hbd} ({c_ratio}:1)",
                        'Li Recovery': round(c_rec, 1),
                        'Viscosity (cP)': round(cp['viscosity_cP'], 1),
                        'pH': round(cp['pH_acidity'], 2),
                        'Redox (V)': round(cp['reduction_potential_V_vs_SHE'], 3),
                        'Li Score': round(cp.get('li_extraction_score', 0), 1),
                        'Current': (c_hba == hba and c_hbd == hbd),
                    })

        # Add current system if not already in list
        current_in = any(d['Current'] for d in compare_data)
        if not current_in:
            compare_data.insert(0, {
                'System': f"[CURRENT] {hba} : {hbd} ({ratio}:1)",
                'Li Recovery': li_rec,
                'Viscosity (cP)': round(visc_pred, 1),
                'pH': round(ph_pred, 2),
                'Redox (V)': round(redox_pred, 3),
                'Li Score': round(li_score, 1),
                'Current': True,
            })

        cdf = pd.DataFrame(compare_data)
        cdf_sorted = cdf.sort_values('Li Recovery', ascending=False).reset_index(drop=True)

        # Bar comparison
        bar_colors = ['#f59e0b' if r['Current'] else '#4c6ef5' for _, r in cdf_sorted.iterrows()]
        fig_cmp = go.Figure(go.Bar(
            x=cdf_sorted['System'], y=cdf_sorted['Li Recovery'],
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in cdf_sorted['Li Recovery']],
            textposition='outside'
        ))
        fig_cmp.add_hline(y=bench_li, line_dash="dash", line_color="#94a3b8",
                           annotation_text=f"Benchmark {bench_li}%")
        fig_cmp.update_layout(
            title=f"Li Recovery Comparison at {temp_C_val}°C (gold = current selection)",
            height=320, margin=dict(l=0,r=0,t=50,b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            xaxis_tickangle=-30,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Table — plain dataframe with emoji indicators (no matplotlib needed)
        show_cols = ['System','Li Recovery','Viscosity (cP)','pH','Li Score']
        disp = cdf_sorted[show_cols].copy()
        disp['Li Recovery'] = disp['Li Recovery'].apply(lambda v: f"{v:.1f}%")
        disp['Viscosity (cP)'] = disp['Viscosity (cP)'].apply(lambda v: f"{v:.0f} cP")
        disp['pH'] = disp['pH'].apply(lambda v: f"{v:.2f}")
        disp['Li Score'] = disp['Li Score'].apply(lambda v: f"{v:.1f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ════════ TAB 2 — DES PROPERTIES ════════
with tabs[1]:
    st.markdown("### All Predicted DES Properties")
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">VISCOSITY · R²=0.896</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{visc_pred:.1f} <span style="font-size:12px;color:#64748b">cP</span></div>
        <div style="font-size:11px;margin-top:4px;color:{'#37b24d' if visc_pred<100 else '#f76707' if visc_pred<300 else '#f03e3e'}">{"Low" if visc_pred<100 else "Moderate" if visc_pred<300 else "High"}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">DENSITY · R²=0.995</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{dens_pred:.1f} <span style="font-size:12px;color:#64748b">kg/m³</span></div>
        <div style="font-size:11px;margin-top:4px;color:#64748b">{'Dense' if dens_pred>1200 else 'Normal'}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">pH · R²=0.989</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{ph_pred:.2f}</div>
        <div style="font-size:11px;margin-top:4px;color:{'#37b24d' if ph_pred<3 else '#f76707' if ph_pred<6 else '#f03e3e'}">{"Acidic " if ph_pred<3 else "Moderate" if ph_pred<6 else "Basic"}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">REDOX · R²=0.973</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{redox_pred:.3f} <span style="font-size:12px;color:#64748b">V</span></div>
        <div style="font-size:11px;margin-top:4px;color:{'#37b24d' if redox_pred<-1.3 else '#f76707' if redox_pred<-1.0 else '#f03e3e'}">{"Reducing" if redox_pred<-1.3 else "Moderate" if redox_pred<-1.0 else "Weak"}</div>
      </div>
      <div style="background:#0f3460;border:1px solid #1d4ed8;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#93c5fd;font-weight:600;margin-bottom:4px">LI SCORE · R²=0.868</div>
        <div style="font-size:22px;font-weight:700;color:#60a5fa">{li_score:.1f} <span style="font-size:12px;color:#93c5fd">/100</span></div>
        <div style="font-size:11px;margin-top:4px;color:{'#34d399' if li_score>=70 else '#fbbf24' if li_score>=50 else '#f87171'}">{"Excellent" if li_score>=70 else "Moderate" if li_score>=50 else "Poor"}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Property radar vs Li ideal zone")
        norm = [
            max(0, min(100, (1 - visc_pred/500)*100)),
            max(0, min(100, (dens_pred-1000)/400*100)),
            max(0, min(100, (1 - ph_pred/10)*100)),
            max(0, min(100, (-redox_pred - 0.9)/1.0*100)),
            li_score,
        ]
        cats = ['Low Viscosity','Density','Low pH','Reducing','Li Score']
        li_ideal = [80, 50, 80, 70, 80]
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatterpolar(r=li_ideal+[li_ideal[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(55,178,77,0.10)',
            line=dict(color='#37b24d', width=1.5, dash='dash'), name='Li ideal'))
        fig_r2.add_trace(go.Scatterpolar(r=norm+[norm[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(76,110,245,0.25)',
            line=dict(color='#4c6ef5', width=2.5), name='Current DES'))
        fig_r2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                             height=320, margin=dict(l=20,r=20,t=30,b=20),
                             paper_bgcolor='rgba(0,0,0,0)',
                             legend=dict(orientation='h',y=-0.15))
        st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("#### Li threshold assessment")
        thresholds = [
            ("Viscosity",      visc_pred,  200,   "cP",  True,  "< 200 cP"),
            ("pH",             ph_pred,    3.5,   "",    True,  "< 3.5"),
            ("Redox (V)",      redox_pred, -1.0,  "V",   False, "< -1.0 V"),
            ("Li Score",       li_score,   65,    "/100",False, "> 65"),
        ]
        for name, val, thresh, unit, lower_better, label in thresholds:
            if lower_better:
                passed = val < thresh
            else:
                passed = val > thresh if name == "Li Score" else val < thresh
            icon  = "Pass" if passed else "Fail"
            color = "#22c55e" if passed else "#ef4444"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        background:#1e293b;border-radius:8px;padding:8px 14px;margin-bottom:6px">
              <span style="color:#94a3b8;font-size:13px">{name}</span>
              <span style="color:#f1f5f9;font-weight:600">{val:.2f}{unit}</span>
              <span style="color:#64748b;font-size:11px">Target: {label}</span>
              <span style="font-size:16px">{icon}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Model accuracy & training stats")
        perf_rows = [
            {"Property":"Viscosity (cP)",       "R²":0.8959,"MAE":"315 cP",    "Train N":4920,"Status":"Pass"},
            {"Property":"Density (kg/m³)",       "R²":0.9948,"MAE":"1.04 kg/m³","Train N":1264,"Status":"Pass"},
            {"Property":"pH",                    "R²":0.9888,"MAE":"0.076",     "Train N":1005,"Status":"Pass"},
            {"Property":"Redox potential (V)",   "R²":0.9729,"MAE":"0.029 V",   "Train N":1737,"Status":"Pass"},
            {"Property":"Li extraction score",   "R²":0.8679,"MAE":"1.94%",     "Train N":878, "Status":""},
        ]
        st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)

        st.markdown("#### Physicochemical interpretation")
        interp = []
        if visc_pred < 100:
            interp.append("**Low viscosity** — excellent Li⁺ diffusion coefficient, fast leaching kinetics expected.")
        elif visc_pred < 300:
            interp.append("**Moderate viscosity** — acceptable mass transfer. Adding 5–10% water mol frac will help.")
        else:
            interp.append("**High viscosity** — Li⁺ diffusion is hindered. Raise temperature or dilute with water.")
        if ph_pred < 2.5:
            interp.append("**Strongly acidic** — will rapidly dissolve Li₂CO₃ and LiCoO₂ oxide phases.")
        elif ph_pred < 4:
            interp.append("**Mildly acidic** — will dissolve most Li phases. Good for selective Li leaching.")
        else:
            interp.append("**Insufficient acidity** — Li oxide dissolution will be slow. Switch to oxalic or malonic acid HBD.")
        if redox_pred < -1.3:
            interp.append("**Strong reducing potential** — Co³⁺→Co²⁺ reduction proceeds readily, releasing lattice Li⁺.")
        elif redox_pred < -1.0:
            interp.append("**Moderate reducing potential** — some Co³⁺ reduction. Adding H₂O₂ will accelerate Li release.")
        else:
            interp.append("**Weak reducing potential** — Co³⁺ not readily reduced. Reductant additive is strongly recommended.")
        for item in interp:
            st.markdown(f'<div class="tip-box" style="margin-bottom:6px">{item}</div>', unsafe_allow_html=True)


# ════════ TAB 3 — FEATURE IMPORTANCE ════════
with tabs[2]:
    st.markdown("### XGBoost Feature Importance — All 5 Models")
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px 18px;margin-bottom:14px">
      <span style="color:#60a5fa;font-weight:600">Li-specific features:</span>
      <span style="color:#94a3b8"> Two binary flags identifying compounds known from literature
      for lithium battery leaching were added to all 5 models.
      Their importance bars (shown in green) reveal how much Li-specific molecular identity
      contributes beyond standard compound encoding.</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for idx, (target, fvals) in enumerate(fi.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            r2 = MODEL_RESULTS[target]['r2']
            names = list(fvals.keys())
            vals  = list(fvals.values())
            colors_fi = ['#37b24d' if 'Li' in n else '#4c6ef5' for n in names]
            fig_fi = go.Figure(go.Bar(
                x=vals, y=names, orientation='h',
                marker_color=colors_fi,
                text=[f"{v*100:.1f}%" for v in vals], textposition='outside'))
            fig_fi.update_layout(
                title=f"{target.replace('_',' ')} · R²={r2}",
                height=260, margin=dict(l=0,r=75,t=40,b=10),
                xaxis=dict(tickformat='.0%', title='Importance', range=[0, max(vals)*1.25]),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.markdown("#### Aggregated importance across all 5 models")
    all_feats = list(list(fi.values())[0].keys())
    avg_imp = {f: float(np.mean([fi[t][f] for t in fi])) for f in all_feats}
    avg_sorted = dict(sorted(avg_imp.items(), key=lambda x: x[1]))
    fig_agg = go.Figure(go.Bar(
        x=list(avg_sorted.values()), y=list(avg_sorted.keys()),
        orientation='h',
        marker_color=['#37b24d' if 'Li' in k else '#4c6ef5' for k in avg_sorted],
        text=[f"{v*100:.1f}%" for v in avg_sorted.values()], textposition='outside'))
    fig_agg.update_layout(
        title="Mean feature importance across all 5 XGBoost models",
        height=280, margin=dict(l=0,r=75,t=40,b=10),
        xaxis=dict(tickformat='.0%', title='Mean Importance'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_agg, use_container_width=True)
    top_feat = max(avg_imp, key=avg_imp.get)
    st.info(f"**Key insight:** '{top_feat}' is the single most important feature overall "
            f"({avg_imp[top_feat]*100:.1f}% avg importance). HBA and HBD molecular identity together "
            f"account for {(avg_imp.get('HBA identity',0)+avg_imp.get('HBD identity',0))*100:.0f}% "
            f"of predictive power — confirming DES molecular structure dominates all physicochemical properties.")


# ════════ TAB 4 — PROPERTY SWEEP ════════
with tabs[3]:
    st.markdown(f"### Temperature & Composition Sweep")
    st.markdown(f"**{hba}** : **{hbd}** · ratio {ratio:.2f} · water {water:.2f} mol frac")

    sweep = sweep_temps(hba, hbd, ratio, water)
    if sweep is None:
        st.error("Could not sweep — compound not in encoder.")
    else:
        tc = sweep['temps_C']

        def sweep_fig(x, y, title, ytitle, color, vline_x, hrange=None, hline=None, hline_label=""):
            fig = go.Figure()
            if hrange:
                fig.add_vrect(x0=hrange[0], x1=hrange[1], fillcolor="#22c55e",
                              opacity=0.08, line_width=0,
                              annotation_text="Li optimal range", annotation_position="top left",
                              annotation_font_size=10)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                     line=dict(color=color, width=2.5),
                                     marker=dict(size=5, color=color)))
            fig.add_vline(x=vline_x, line_dash='dash', line_color='#ef4444',
                          annotation_text=f"▶ {vline_x:.0f}°C", annotation_font_size=10)
            if hline is not None:
                fig.add_hline(y=hline, line_dash='dot', line_color='#f59e0b',
                              annotation_text=hline_label, annotation_font_size=10)
            fig.update_layout(title=title, height=240,
                              margin=dict(l=0,r=0,t=38,b=10),
                              xaxis_title="Temperature (°C)", yaxis_title=ytitle,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sweep_fig(tc, sweep['viscosity_cP'],
                "Viscosity vs Temperature", "Viscosity (cP)", "#4c6ef5",
                temp_C_val, hrange=[60,90], hline=200, hline_label="200 cP limit"),
                use_container_width=True)
            st.plotly_chart(sweep_fig(tc, sweep['pH_acidity'],
                "pH vs Temperature", "pH", "#f76707",
                temp_C_val, hrange=[60,90], hline=3.5, hline_label="pH 3.5 target"),
                use_container_width=True)
        with col2:
            st.plotly_chart(sweep_fig(tc, sweep['li_extraction_score'],
                "Li Suitability Score vs Temperature", "Li Score (0–100)", "#37b24d",
                temp_C_val, hrange=[60,90], hline=65, hline_label="Score 65 target"),
                use_container_width=True)
            st.plotly_chart(sweep_fig(tc, sweep['density_kg_m3'],
                "Density vs Temperature", "Density (kg/m³)", "#7950f2",
                temp_C_val),
                use_container_width=True)

        # Find optimal temp
        opt_idx = int(np.argmax(sweep['li_extraction_score']))
        opt_temp = tc[opt_idx]
        opt_score= sweep['li_extraction_score'][opt_idx]
        st.success(f"**Optimal temperature for this DES:** {opt_temp:.0f}°C → Li score {opt_score:.1f}/100 "
                   f"(viscosity {sweep['viscosity_cP'][opt_idx]:.0f} cP, pH {sweep['pH_acidity'][opt_idx]:.2f})")

        st.divider()
        st.markdown("### Water content & Molar ratio sweeps")
        col_w1, col_w2, col_w3 = st.columns(3)

        water_vals = np.linspace(0, 0.5, 25)
        w_visc, w_li, w_ph = [], [], []
        for wv in water_vals:
            p2, _ = predict_all(hba, hbd, ratio, temp_K_val, float(wv))
            if p2:
                w_visc.append(p2['viscosity_cP'])
                w_li.append(p2.get('li_extraction_score', 0))
                w_ph.append(p2['pH_acidity'])

        ratio_vals = np.linspace(0.25, 6.0, 25)
        r_visc, r_li = [], []
        for rv in ratio_vals:
            p3, _ = predict_all(hba, hbd, float(rv), temp_K_val, water)
            if p3:
                r_visc.append(p3['viscosity_cP'])
                r_li.append(p3.get('li_extraction_score', 0))

        with col_w1:
            fig_wv2 = go.Figure()
            fig_wv2.add_trace(go.Scatter(x=water_vals[:len(w_visc)], y=w_visc,
                mode='lines', line=dict(color='#4c6ef5', width=2.5), name='Viscosity'))
            fig_wv2.add_vline(x=water, line_dash='dash', line_color='#ef4444')
            fig_wv2.add_hline(y=200, line_dash='dot', line_color='#f59e0b',
                              annotation_text="200 cP")
            fig_wv2.update_layout(title="Viscosity vs Water", height=220,
                xaxis_title="Water (mol frac)", yaxis_title="Viscosity (cP)",
                margin=dict(l=0,r=0,t=38,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wv2, use_container_width=True)

        with col_w2:
            fig_wl2 = go.Figure()
            fig_wl2.add_trace(go.Scatter(x=water_vals[:len(w_li)], y=w_li,
                mode='lines', line=dict(color='#37b24d', width=2.5)))
            fig_wl2.add_vline(x=water, line_dash='dash', line_color='#ef4444')
            fig_wl2.add_hline(y=65, line_dash='dot', line_color='#f59e0b',
                              annotation_text="Score 65")
            fig_wl2.update_layout(title="Li Score vs Water", height=220,
                xaxis_title="Water (mol frac)", yaxis_title="Li Score",
                margin=dict(l=0,r=0,t=38,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wl2, use_container_width=True)

        with col_w3:
            fig_rl = go.Figure()
            fig_rl.add_trace(go.Scatter(x=ratio_vals[:len(r_li)], y=r_li,
                mode='lines', line=dict(color='#f76707', width=2.5)))
            fig_rl.add_vline(x=ratio, line_dash='dash', line_color='#ef4444')
            if r_li:
                best_r_idx = int(np.argmax(r_li))
                fig_rl.add_vline(x=ratio_vals[best_r_idx], line_dash='dot', line_color='#22c55e',
                                 annotation_text=f"Best: {ratio_vals[best_r_idx]:.2f}")
            fig_rl.update_layout(title="Li Score vs Molar Ratio", height=220,
                xaxis_title="HBA:HBD Ratio", yaxis_title="Li Score",
                margin=dict(l=0,r=0,t=38,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rl, use_container_width=True)


# ════════ TAB 5 — REUSE SIMULATOR ════════
with tabs[4]:
    st.markdown("### DES Reuse & Regeneration Simulator")
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("#### System configuration")
        r_sys    = st.selectbox("DES system", [s['name'] for s in LI_DES_SYSTEMS])
        r_regen  = st.selectbox("Regeneration method", list(REGEN_METHODS.keys()))
        r_cycles = st.slider("Reuse cycles to simulate", 1, 15, 6)
        r_temp2  = st.slider("Operating temperature (°C)  ", 60, 180, 80, 5)
        r_purity = st.slider("Minimum acceptable Li recovery (%)", 70, 99, 88, 1)

        rg       = REGEN_METHODS[r_regen]
        sys_d    = next(s for s in LI_DES_SYSTEMS if s['name'] == r_sys)
        base_eff = sys_d['li_eff']
        tf2      = 1.0 + max(0, r_temp2 - 100) / 500
        loss_cyc = (100 - base_eff) * 0.3 * tf2 * (1 - rg['li_recovery'] * 0.6)
        cyc_eff  = [max(55, base_eff - loss_cyc * i) for i in range(r_cycles)]
        no_regen = [max(40, base_eff - (100-base_eff)*0.4*tf2*i) for i in range(r_cycles)]
        lifetime = next((i for i,e in enumerate(cyc_eff) if e < r_purity), r_cycles)
        cumulative_gain = sum(cyc_eff) - sum(no_regen)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px">
          <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:10px;color:#94a3b8;font-weight:600">CYCLE 1</div>
            <div style="font-size:20px;font-weight:700;color:#f1f5f9">{cyc_eff[0]:.1f}%</div>
          </div>
          <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:10px;color:#94a3b8;font-weight:600">CYCLE {r_cycles}</div>
            <div style="font-size:20px;font-weight:700;color:#f1f5f9">{cyc_eff[-1]:.1f}%</div>
          </div>
          <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:10px;color:#94a3b8;font-weight:600">USEFUL LIFE</div>
            <div style="font-size:20px;font-weight:700;color:#60a5fa">{lifetime} cycles</div>
          </div>
          <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:10px;color:#94a3b8;font-weight:600">REGEN GAIN</div>
            <div style="font-size:20px;font-weight:700;color:#34d399">+{cumulative_gain:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin-top:10px">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">Regeneration method details</div>
          <div style="font-size:13px;color:#f1f5f9">Li recovery rate: <b>{rg['li_recovery']*100:.0f}%</b> per cycle</div>
          <div style="font-size:13px;color:#f1f5f9">Max cycles supported: <b>{rg['cycles']}</b></div>
          <div style="font-size:13px;color:#f1f5f9">Energy demand: <b>{rg['energy'].capitalize()}</b></div>
          <div style="font-size:13px;color:#f1f5f9">DES viscosity: <b>{sys_d['visc']} cP</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        cl = [f"C{i+1}" for i in range(r_cycles)]
        fig_ru = go.Figure()
        fig_ru.add_trace(go.Scatter(x=cl, y=cyc_eff, mode="lines+markers",
            name="With regeneration", line=dict(color="#4c6ef5", width=2.5),
            marker=dict(size=8, color="#4c6ef5")))
        fig_ru.add_trace(go.Scatter(x=cl, y=no_regen, mode="lines+markers",
            name="No regeneration", line=dict(color="#f03e3e", width=2, dash="dash"),
            marker=dict(size=6)))
        fig_ru.add_hrect(y0=r_purity, y1=102, fillcolor="#22c55e", opacity=0.08,
            line_width=0, annotation_text=f"Target ≥{r_purity}%",
            annotation_position="top right")
        fig_ru.add_hline(y=r_purity, line_dash="dot", line_color="#22c55e", line_width=1)
        fig_ru.update_layout(title="Li Recovery Efficiency per Cycle",
            height=280, margin=dict(l=0,r=0,t=40,b=10),
            yaxis=dict(range=[35,105], ticksuffix="%"),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ru, use_container_width=True)

        cum_regen   = [sum(cyc_eff[:i+1]) for i in range(r_cycles)]
        cum_noregen = [sum(no_regen[:i+1]) for i in range(r_cycles)]
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=cl, y=cum_regen, fill="tozeroy", mode="lines+markers",
            name="With regen", line=dict(color="#4c6ef5",width=2),
            fillcolor="rgba(76,110,245,.12)", marker=dict(size=6)))
        fig_cum.add_trace(go.Scatter(x=cl, y=cum_noregen, fill="tozeroy", mode="lines",
            name="No regen", line=dict(color="#f03e3e",width=2,dash="dash"),
            fillcolor="rgba(239,68,68,.06)"))
        fig_cum.update_layout(title="Cumulative Li Recovered (Σ efficiency)",
            height=220, margin=dict(l=0,r=0,t=40,b=10),
            yaxis_title="Σ efficiency %", legend=dict(orientation="h", y=-0.28),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cum, use_container_width=True)


# ════════ TAB 6 — DES RANKER ════════
with tabs[5]:
    st.markdown("### Li-Optimised DES Systems Ranker")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("#### Objective weights")
        pw_li = st.slider("Li efficiency weight (%)", 0, 100, 50, 5)
        pw_co = st.slider("Cost weight (%)",          0, 100, 25, 5)
        pw_gr = st.slider("Green score weight (%)",   0, 100, 25, 5)
        tw    = pw_li + pw_co + pw_gr
        we,wc,wg = (pw_li/100, pw_co/100, pw_gr/100) if tw > 0 else (0,0,0)
        if tw != 100:
            st.warning(f"Weights sum to {tw}% (not 100%) — scores are scaled accordingly.")

        pf     = is_pareto(LI_DES_SYSTEMS)
        scored = sorted([dict(**s, score=round(
            we*(s["li_eff"]/100)*100 + wc*(1-s["cost"])*100 + wg*s["green"]*100, 1))
            for s in LI_DES_SYSTEMS], key=lambda x: x["score"], reverse=True)

        st.markdown("#### Ranked systems")
        for i, s in enumerate(scored[:8]):
            visc_col2 = "#22c55e" if s["visc"]<100 else ("#f59e0b" if s["visc"]<500 else "#ef4444")
            pf_star   = " *" if any(p["name"]==s["name"] and f for p,f in zip(LI_DES_SYSTEMS,pf)) else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;
                        background:#1e293b;border-radius:8px;padding:8px 12px">
              <span style="width:24px;color:#64748b;font-weight:700">#{i+1}</span>
              <div style="flex:1">
                <div style="font-weight:600;font-size:13px;color:#f1f5f9">{s['name']}{pf_star}</div>
                <div style="font-size:11px;color:#64748b">
                  Li {s['li_eff']}% · Cost {s['cost']:.2f} ·
                  Green {s['green']*100:.0f}% ·
                  <span style="color:{visc_col2}">{s['visc']} cP</span></div>
                <div style="height:4px;background:#334155;border-radius:2px;margin-top:4px">
                  <div style="height:4px;width:{min(int(s['score']),100)}%;background:#4c6ef5;border-radius:2px"></div>
                </div>
              </div>
              <span style="background:#1d4ed8;color:#bfdbfe;padding:2px 8px;border-radius:6px;
                           font-size:12px;font-weight:600;min-width:44px;text-align:center">{s['score']:.0f}</span>
            </div>""", unsafe_allow_html=True)
        st.caption("Pareto-optimal = starred")

    with col2:
        # Pareto frontier
        po  = [s for s,f in zip(LI_DES_SYSTEMS,pf) if f]
        pd2 = [s for s,f in zip(LI_DES_SYSTEMS,pf) if not f]
        fig_p = go.Figure()
        if pd2:
            fig_p.add_trace(go.Scatter(
                x=[s["cost"] for s in pd2], y=[s["li_eff"] for s in pd2],
                mode="markers", name="Sub-optimal",
                marker=dict(color="#475569", size=10, symbol="circle"),
                text=[s["name"] for s in pd2], hoverinfo="text+x+y"))
        if po:
            fig_p.add_trace(go.Scatter(
                x=[s["cost"] for s in po], y=[s["li_eff"] for s in po],
                mode="markers+text", name="Pareto-optimal *",
                marker=dict(color="#4c6ef5", size=14, symbol="star"),
                text=[s["name"].split(":")[0] for s in po],
                textposition="top center", hoverinfo="text+x+y"))
        fig_p.update_layout(
            title="Pareto Frontier — Li Efficiency vs Cost",
            height=290, margin=dict(l=0,r=0,t=40,b=10),
            xaxis=dict(title="Relative cost (lower = better)", autorange="reversed"),
            yaxis=dict(title="Li Leaching Efficiency (%)", range=[83,103]),
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)

        # Bubble chart: Li eff vs viscosity (size=green score)
        fig_bub = go.Figure(go.Scatter(
            x=[s["visc"] for s in LI_DES_SYSTEMS],
            y=[s["li_eff"] for s in LI_DES_SYSTEMS],
            mode="markers+text",
            marker=dict(
                size=[s["green"]*40 for s in LI_DES_SYSTEMS],
                color=[s["cost"] for s in LI_DES_SYSTEMS],
                colorscale="RdYlGn_r", showscale=True,
                colorbar=dict(title="Cost", thickness=10, len=0.6)),
            text=[s["name"].split(":")[0] for s in LI_DES_SYSTEMS],
            textposition="top center",
            hovertext=[f"{s['name']}<br>Li:{s['li_eff']}% Visc:{s['visc']}cP Cost:{s['cost']}"
                       for s in LI_DES_SYSTEMS],
            hoverinfo="text"))
        fig_bub.add_vline(x=200, line_dash="dash", line_color="#ef4444",
                           annotation_text="200 cP limit", annotation_font_size=10)
        fig_bub.update_layout(
            title="Bubble chart: Li Eff vs Viscosity (size=green score, color=cost)",
            height=290, margin=dict(l=0,r=0,t=50,b=10),
            xaxis_title="Viscosity (cP)", yaxis_title="Li Efficiency (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bub, use_container_width=True)


# ════════ TAB 6 — RECOMMENDATIONS ════════
with tabs[6]:
    st.markdown("## Lithium Extraction Recommendations")
    st.markdown(
        "Evidence-based guidance for optimising DES-mediated Li recovery from spent lithium-ion batteries, "
        "derived from the trained XGBoost models and published literature."
    )
    st.divider()

    # ── Current DES assessment banner ─────────────────────────────────────────
    overall_score = round(0.4*(li_score/100) + 0.35*(li_rec/100) + 0.25*(max(0,(-ph_pred-0)/8)), 2)
    if li_rec >= 90 and visc_pred < 200 and ph_pred < 3.5:
        banner_col  = "#14532d"; banner_text_col = "#4ade80"
        banner_msg  = "Current DES configuration is well-suited for Li extraction."
        banner_label= "RECOMMENDED"
    elif li_rec >= 75:
        banner_col  = "#422006"; banner_text_col = "#fb923c"
        banner_msg  = "Current DES is acceptable — minor adjustments will improve yield significantly."
        banner_label= "NEEDS TUNING"
    else:
        banner_col  = "#450a0a"; banner_text_col = "#f87171"
        banner_msg  = "Current DES is not optimal for Li extraction. See recommendations below."
        banner_label= "NOT RECOMMENDED"

    st.markdown(f"""
    <div style="background:{banner_col};border-radius:10px;padding:14px 20px;
                display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <div style="font-size:11px;color:{banner_text_col};font-weight:700;
                    letter-spacing:.08em;margin-bottom:4px">{banner_label}</div>
        <div style="font-size:14px;color:#f1f5f9">{banner_msg}</div>
        <div style="font-size:12px;color:#94a3b8;margin-top:4px">
          Current: <b>{hba}</b> : <b>{hbd}</b> ({ratio}:1) at {temp_C_val}°C
        </div>
      </div>
      <div style="text-align:center;min-width:80px">
        <div style="font-size:32px;font-weight:700;color:{banner_text_col}">{li_rec:.0f}%</div>
        <div style="font-size:11px;color:#94a3b8">Li recovery</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)

    # ── LEFT: DES Selection Recommendations ───────────────────────────────────
    with rec_col1:
        st.markdown("### DES Selection")

        st.markdown("""
        <div class="rec-card-green">
          <div class="rec-title">Top recommended DES systems for LIB Li extraction</div>
          <div class="rec-body">
            Based on combined viscosity, pH, redox, and literature Li recovery data,
            these systems consistently deliver &gt;95% Li leaching efficiency:
            <br><br>
            <b>1. ChCl : Oxalic acid (1:1)</b> — Low viscosity (~45 cP), pH ~1.2,
            redox ~−1.4 V. Best all-round performer for NMC/LCO cathodes.<br><br>
            <b>2. ChCl : Levulinic acid (1:2)</b> — Slightly higher viscosity (~112 cP)
            but exceptional Li selectivity and green score. Recommended for LFP cathodes.<br><br>
            <b>3. ChCl : Malonic acid (1:1)</b> — pH ~1.5, viscosity ~89 cP.
            Strong Co/Ni co-leaching for full black mass processing.<br><br>
            <b>4. ChCl : Tartaric acid (1:1)</b> — Biodegradable, moderate viscosity (~76 cP),
            good selectivity for Li over Al impurities.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rec-card">
          <div class="rec-title">HBA selection guide</div>
          <div class="rec-body">
            <b>Choline chloride (ChCl)</b> — most widely validated HBA for LIB leaching.
            Low cost, biodegradable, water-stable. Use at 1:1 to 1:2 molar ratio with acidic HBDs.<br><br>
            <b>Betaine</b> — zwitterionic structure enhances H-bond network stability at high temperatures.
            Better thermal stability than ChCl above 100°C.<br><br>
            <b>Acetylcholine chloride</b> — yields slightly lower viscosity DES vs ChCl.
            Useful when mass transfer is the rate-limiting step.<br><br>
            <b>Avoid:</b> High-melting or high-viscosity HBAs (lidocaine, thymol alone) unless
            blended — they increase viscosity above the 200 cP limit.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rec-card-orange">
          <div class="rec-title">HBD selection guide</div>
          <div class="rec-body">
            <b>Oxalic acid</b> — strongest acidity (pH ~1.2–1.5). Best for dissolving
            stable Li₂CO₃ and LiCoO₂. Forms insoluble Co-oxalate which aids Li/Co separation.<br><br>
            <b>Levulinic acid</b> — moderate acidity, lower corrosivity. Preferred for
            NMC cathodes where Ni/Mn co-leaching is desired alongside Li.<br><br>
            <b>Malonic acid</b> — good acidity (pH ~1.5–2.0) with lower viscosity than
            citric acid. Balances Li recovery with DES reusability.<br><br>
            <b>Lactic acid</b> — milder acidity (pH ~2.5–3.5). Best for LFP cathodes
            where high acidity risks dissolving Al current collector.<br><br>
            <b>Avoid for Li:</b> EG, glycerol, urea as sole HBDs — insufficient acidity
            for Li oxide dissolution without additives.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT: Process Recommendations ────────────────────────────────────────
    with rec_col2:
        st.markdown("### Process Conditions")

        st.markdown(f"""
        <div class="rec-card-green">
          <div class="rec-title">Optimal operating window (model-derived)</div>
          <div class="rec-body">
            The XGBoost models trained on 5,790 data points identify this operating window
            for maximum Li recovery from LIB cathodes:
            <br><br>
            <table style="width:100%;font-size:12px;border-collapse:collapse">
              <tr style="border-bottom:1px solid #334155">
                <td style="padding:5px 8px;color:#60a5fa;font-weight:600">Parameter</td>
                <td style="padding:5px 8px;color:#60a5fa;font-weight:600">Optimal range</td>
                <td style="padding:5px 8px;color:#60a5fa;font-weight:600">Current</td>
              </tr>
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:5px 8px;color:#94a3b8">Temperature</td>
                <td style="padding:5px 8px;color:#4ade80">60 – 90°C</td>
                <td style="padding:5px 8px;color:{'#4ade80' if 60<=temp_C_val<=90 else '#f87171'}">{temp_C_val}°C</td>
              </tr>
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:5px 8px;color:#94a3b8">Viscosity</td>
                <td style="padding:5px 8px;color:#4ade80">&lt; 200 cP</td>
                <td style="padding:5px 8px;color:{'#4ade80' if visc_pred<200 else '#f87171'}">{visc_pred:.0f} cP</td>
              </tr>
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:5px 8px;color:#94a3b8">pH</td>
                <td style="padding:5px 8px;color:#4ade80">1.5 – 3.5</td>
                <td style="padding:5px 8px;color:{'#4ade80' if 1.5<=ph_pred<=3.5 else '#f87171'}">{ph_pred:.2f}</td>
              </tr>
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:5px 8px;color:#94a3b8">Redox potential</td>
                <td style="padding:5px 8px;color:#4ade80">&lt; −1.0 V</td>
                <td style="padding:5px 8px;color:{'#4ade80' if redox_pred<-1.0 else '#f87171'}">{redox_pred:.3f} V</td>
              </tr>
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:5px 8px;color:#94a3b8">Water content</td>
                <td style="padding:5px 8px;color:#4ade80">0.0 – 0.2</td>
                <td style="padding:5px 8px;color:{'#4ade80' if water<=0.2 else '#f87171'}">{water:.2f}</td>
              </tr>
              <tr>
                <td style="padding:5px 8px;color:#94a3b8">Molar ratio</td>
                <td style="padding:5px 8px;color:#4ade80">1:1 – 1:3</td>
                <td style="padding:5px 8px;color:{'#4ade80' if 0.33<=ratio<=1.0 else '#f87171'}">{ratio:.2f}</td>
              </tr>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rec-card">
          <div class="rec-title">Pre-treatment recommendations</div>
          <div class="rec-body">
            <b>Calcination (500°C, 1–2 h)</b> — removes PVDF binder and decomposes
            Li₂CO₃ surface layer. Increases Li leaching rate by 5–10%. Essential for
            aged or heavily cycled cells.<br><br>
            <b>Mechanical crushing + sieving (&lt;75 µm)</b> — maximises surface area
            exposed to DES. Particle size is the most controllable kinetic lever.<br><br>
            <b>Thermal + mechanical combined</b> — highest yield boost (+8–12%).
            Recommended for EV battery modules with thick electrode coatings.<br><br>
            <b>Avoid water washing before DES leaching</b> — dissolves surface Li salts
            that would otherwise contribute to overall Li yield.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rec-card-orange">
          <div class="rec-title">Reductant & assist method guidance</div>
          <div class="rec-body">
            <b>H₂O₂ (1–5 vol%)</b> — most effective reductant for NMC/NCA/LCO cathodes.
            Reduces Co³⁺ and Ni³⁺, destabilising the cathode lattice and releasing Li⁺.
            Optimal addition: 2 vol% at 80°C.<br><br>
            <b>Ascorbic acid (5–10 wt%)</b> — green alternative to H₂O₂. Comparable
            reduction effect, no hazardous byproducts. Particularly effective for LiCoO₂.<br><br>
            <b>Microwave assist</b> — reduces leaching time from 60–120 min to 10–20 min
            at equivalent yield. Energy-efficient for batch processing.<br><br>
            <b>Ultrasound assist</b> — improves mass transfer in high-viscosity DES (&gt;200 cP)
            by disrupting boundary layer. Combine with elevated temperature for best results.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Battery chemistry guide ────────────────────────────────────────────────
    st.markdown("### Recommended DES by Battery Chemistry")
    chem_data = [
        {
            "chemistry": "NMC (LiNiMnCoO₂)",
            "cathode":   "Li, Ni, Mn, Co",
            "best_des":  "ChCl : Oxalic acid (1:1)",
            "temp":      "80°C",
            "additive":  "H₂O₂ 2 vol%",
            "li_yield":  "97%",
            "notes":     "Oxalic acid forms CoC₂O₄ precipitate aiding separation",
            "color":     "#1e3a5f", "text": "#60a5fa"
        },
        {
            "chemistry": "NCA (LiNiCoAlO₂)",
            "cathode":   "Li, Ni, Co, Al",
            "best_des":  "ChCl : Levulinic acid (1:2)",
            "temp":      "80°C",
            "additive":  "Ascorbic acid 5 wt%",
            "li_yield":  "96%",
            "notes":     "Milder acidity avoids Al current collector dissolution",
            "color":     "#1a2e1a", "text": "#4ade80"
        },
        {
            "chemistry": "LCO (LiCoO₂)",
            "cathode":   "Li, Co",
            "best_des":  "ChCl : Malonic acid (1:1)",
            "temp":      "70°C",
            "additive":  "Ascorbic acid 5 wt%",
            "li_yield":  "98%",
            "notes":     "High Co³⁺ content — reductant is critical for Li release",
            "color":     "#2d1b4e", "text": "#a78bfa"
        },
        {
            "chemistry": "LFP (LiFePO₄)",
            "cathode":   "Li, Fe, P",
            "best_des":  "ChCl : Lactic acid (1:2)",
            "temp":      "60°C",
            "additive":  "None required",
            "li_yield":  "97%",
            "notes":     "Fe²⁺ does not need reduction — no reductant needed. Lower temp is sufficient.",
            "color":     "#1c2a1c", "text": "#86efac"
        },
        {
            "chemistry": "LMO (LiMn₂O₄)",
            "cathode":   "Li, Mn",
            "best_des":  "ChCl : Tartaric acid (1:1)",
            "temp":      "80°C",
            "additive":  "H₂O₂ 1 vol%",
            "li_yield":  "93%",
            "notes":     "Mn³⁺/Mn⁴⁺ reduction important; tartaric acid chelates Mn²⁺ in solution",
            "color":     "#2a1c1c", "text": "#fca5a5"
        },
    ]

    for i in range(0, len(chem_data), 1):
        c = chem_data[i]
        st.markdown(f"""
        <div style="background:{c['color']};border-radius:10px;padding:14px 18px;
                    margin-bottom:8px;display:grid;grid-template-columns:1.5fr 1.5fr 1fr 1fr 1fr 1.5fr;
                    gap:12px;align-items:center">
          <div>
            <div style="font-size:12px;font-weight:700;color:{c['text']}">{c['chemistry']}</div>
            <div style="font-size:11px;color:#64748b">Metals: {c['cathode']}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;margin-bottom:2px">Best DES</div>
            <div style="font-size:11px;font-weight:600;color:#f1f5f9">{c['best_des']}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;margin-bottom:2px">Temperature</div>
            <div style="font-size:12px;font-weight:600;color:#f1f5f9">{c['temp']}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;margin-bottom:2px">Additive</div>
            <div style="font-size:11px;font-weight:600;color:#f1f5f9">{c['additive']}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;margin-bottom:2px">Li yield</div>
            <div style="font-size:14px;font-weight:700;color:{c['text']}">{c['li_yield']}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;margin-bottom:2px">Notes</div>
            <div style="font-size:11px;color:#94a3b8">{c['notes']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Troubleshooting guide ──────────────────────────────────────────────────
    st.markdown("### Troubleshooting Low Li Recovery")
    t_col1, t_col2, t_col3 = st.columns(3)

    with t_col1:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:14px">
          <div style="font-size:13px;font-weight:700;color:#f87171;margin-bottom:10px">
            Recovery &lt; 75%
          </div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.8">
            <b style="color:#f1f5f9">Check viscosity first</b> — if &gt;300 cP,
            mass transfer is the bottleneck. Raise temperature by 20°C or add
            5–10% water (mol frac).<br><br>
            <b style="color:#f1f5f9">Check pH</b> — if &gt;4, switch to a more
            acidic HBD. Oxalic or malonic acid will drop pH below 2.<br><br>
            <b style="color:#f1f5f9">Check pre-treatment</b> — uncalcined cathodes
            with intact PVDF binder can reduce yield by 10–15%.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:14px">
          <div style="font-size:13px;font-weight:700;color:#fb923c;margin-bottom:10px">
            Recovery 75–88%
          </div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.8">
            <b style="color:#f1f5f9">Add reductant</b> — H₂O₂ (2 vol%) typically
            adds 5–8% to Li recovery by reducing Co³⁺ and destabilising the
            cathode lattice.<br><br>
            <b style="color:#f1f5f9">Extend leaching time</b> — increasing from
            60 to 120 min adds ~3–5% yield for diffusion-controlled systems.<br><br>
            <b style="color:#f1f5f9">Try microwave assist</b> — same yield as
            120 min conventional in 15–20 min.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with t_col3:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:14px">
          <div style="font-size:13px;font-weight:700;color:#4ade80;margin-bottom:10px">
            Maximising &gt;95%
          </div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.8">
            <b style="color:#f1f5f9">Optimise solid:liquid ratio</b> — 1:20 g/mL
            is the sweet spot for most cathode materials. Going to 1:50 adds
            marginal yield at high dilution cost.<br><br>
            <b style="color:#f1f5f9">Use microwave + ultrasound combined</b> —
            highest kinetic enhancement (+11% recovery factor).<br><br>
            <b style="color:#f1f5f9">Combine calcination + sieving &lt;75 µm</b>
            — surface area maximisation is the most reliable yield lever.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Li/Co Separation strategy ──────────────────────────────────────────────
    st.markdown("### Post-Leach Li Recovery Strategies")
    strat_col1, strat_col2 = st.columns(2)
    with strat_col1:
        strategies = [
            ("Selective precipitation",
             "Adjust leachate pH to 5–6 to precipitate Co(OH)₂, Ni(OH)₂, Mn(OH)₂. "
             "Filter. Then raise pH to 10–11 to precipitate Li₂CO₃ (add Na₂CO₃). "
             "Purity: 95–98%. Recovery: 85–92%."),
            ("Solvent extraction",
             "Use D2EHPA or Cyanex 272 extractant in kerosene to selectively "
             "strip Co²⁺ and Ni²⁺ from leachate. Li⁺ remains in aqueous phase. "
             "Purity: 97–99%. Recovery: 90–95%."),
            ("Membrane electrolysis",
             "Apply 2–4 V across ion-exchange membrane. Li⁺ migrates to cathode "
             "compartment. Produces battery-grade LiOH directly. "
             "Purity: 99%+. Recovery: 88–93%."),
        ]
        for title, body in strategies:
            st.markdown(f"""
            <div style="background:#1e293b;border-left:3px solid #4c6ef5;
                        border-radius:8px;padding:12px 16px;margin-bottom:8px">
              <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:4px">{title}</div>
              <div style="font-size:12px;color:#94a3b8;line-height:1.7">{body}</div>
            </div>""", unsafe_allow_html=True)

    with strat_col2:
        strategies2 = [
            ("Oxalic acid co-precipitation",
             "When using oxalic acid DES, Co²⁺ spontaneously precipitates as CoC₂O₄ "
             "during leaching. Filter hot to separate from Li-rich filtrate. "
             "Simple, in-situ separation — no additional reagents. "
             "Li purity in filtrate: 90–95%."),
            ("Evaporation + crystallisation",
             "Evaporate leachate to 20–30% of original volume. Li₂SO₄ or LiCl "
             "crystallises preferentially if DES contains sulphate/chloride. "
             "Multiple recrystallisation steps achieve battery-grade purity."),
            ("DES regeneration + Li recovery loop",
             "After leaching, add Na₂CO₃ to precipitate Li₂CO₃ directly from "
             "DES leachate. Filter Li₂CO₃. Replenish HBD and reuse DES. "
             "Combines Li recovery and DES regeneration in a single step. "
             "Best suited for ChCl:organic acid systems."),
        ]
        for title, body in strategies2:
            st.markdown(f"""
            <div style="background:#1e293b;border-left:3px solid #37b24d;
                        border-radius:8px;padding:12px 16px;margin-bottom:8px">
              <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:4px">{title}</div>
              <div style="font-size:12px;color:#94a3b8;line-height:1.7">{body}</div>
            </div>""", unsafe_allow_html=True)


# ════════ TAB 7 — MODEL INFO ════════
with tabs[7]:
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">TRAINING ROWS</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">5,790</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">MODELS</div>
        <div style="font-size:26px;font-weight:700;color:#60a5fa">5</div>
        <div style="font-size:11px;color:#64748b">XGBoost</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">UNIQUE HBAs</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">144</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">UNIQUE HBDs</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">167</div>
      </div>
      <div style="background:#0f3460;border:1px solid #1d4ed8;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:10px;color:#93c5fd;font-weight:600;margin-bottom:4px">Li FEATURES</div>
        <div style="font-size:26px;font-weight:700;color:#60a5fa">2</div>
        <div style="font-size:11px;color:#93c5fd">new flags</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model accuracy")
        st.markdown("""
| Property | R² | MAE | Training samples |
|---|---|---|---|
| Viscosity (cP) | 0.8959 | 315 cP | 4,920 |
| Density (kg/m³) | 0.9948 | 1.04 kg/m³ | 1,264 |
| pH | 0.9888 | 0.076 | 1,005 |
| Redox potential (V) | 0.9729 | 0.029 V | 1,737 |
| Li extraction score | 0.8679 | 1.94% | 878 |
        """)

        st.markdown("#### Li extraction thresholds")
        rows = [
            ("Viscosity",       "< 200 cP",     "Li⁺ diffusion coefficient"),
            ("pH",              "1.5 – 3.5",    "Dissolves Li₂CO₃ and LiCoO₂"),
            ("Redox potential", "< −1.0 V",     "Reduces Co³⁺→Co²⁺, releases Li⁺"),
            ("Temperature",     "60 – 90 °C",   "Kinetics vs DES stability trade-off"),
            ("Water content",   "0.0 – 0.2",    "Preserves H-bond network"),
            ("Li score",        "> 65 / 100",   "Combined suitability index"),
        ]
        for prop, target_r, reason in rows:
            st.markdown(f"""
            <div style="display:flex;gap:10px;background:#1e293b;border-radius:6px;
                        padding:7px 12px;margin-bottom:5px;align-items:center">
              <span style="color:#60a5fa;font-weight:600;min-width:130px;font-size:12px">{prop}</span>
              <span style="color:#34d399;font-weight:600;min-width:90px;font-size:12px">{target_r}</span>
              <span style="color:#94a3b8;font-size:11px">{reason}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Li-curated compound libraries")
        li_df = pd.DataFrame({
            'Li HBAs (literature)': BEST_LI_HBAS + ['']*(max(0,len(BEST_LI_HBDS)-len(BEST_LI_HBAS))),
            'Li HBDs (literature)': BEST_LI_HBDS + ['']*(max(0,len(BEST_LI_HBAS)-len(BEST_LI_HBDS))),
        })
        st.dataframe(li_df, hide_index=True, use_container_width=True)


