import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Obesidade — FIAP POSTECH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIAP_PINK  = "#ED145B"
FIAP_PINK2 = "#C4004A"

THEME = {
    "dark": {
        "bg":"#0D0D0D","bg2":"#1A1A1A","card":"#1E1E1E","kpi":"#252525",
        "text":"#FFFFFF","subtext":"#ED145B","label":"#ED145B",
        "border":"#333333","accent":"#ED145B","accent2":"#C4004A",
        "plot_bg":"#1A1A1A","paper_bg":"#1A1A1A","grid":"#2E2E2E",
        "font":"#F0F0F0","positive":"#ED145B","negative":"#4D9DE0",
        "sidebar":"#111111","input_bg":"#252525","input_txt":"#FFFFFF",
        "tag_bg":"#252525","tag_txt":"#FFFFFF",
    },
    "light": {
        "bg":"#FFFFFF","bg2":"#F7F7F7","card":"#F0F0F0","kpi":"#F7F7F7",
        "text":"#111111","subtext":"#444444","label":"#111111",
        "border":"#DDDDDD","accent":"#ED145B","accent2":"#C4004A",
        "plot_bg":"#FFFFFF","paper_bg":"#FFFFFF","grid":"#EEEEEE",
        "font":"#111111","positive":"#ED145B","negative":"#1A73E8",
        "sidebar":"#F5F5F5","input_bg":"#FFFFFF","input_txt":"#111111",
        "tag_bg":"#ED145B","tag_txt":"#FFFFFF",
    },
}

OBESITY_ORDER = [
    "Insufficient_Weight","Normal_Weight",
    "Overweight_Level_I","Overweight_Level_II",
    "Obesity_Type_I","Obesity_Type_II","Obesity_Type_III",
]
OBESITY_LABELS = {
    "Insufficient_Weight":"Abaixo do Peso","Normal_Weight":"Peso Normal",
    "Overweight_Level_I":"Sobrepeso I","Overweight_Level_II":"Sobrepeso II",
    "Obesity_Type_I":"Obesidade I","Obesity_Type_II":"Obesidade II",
    "Obesity_Type_III":"Obesidade III",
}
OB_SHORT = ["Ab.Peso","Peso Norm.","Sobr.I","Sobr.II","Ob.I","Ob.II","Ob.III"]
OB_COLORS = ["#4D9DE0","#2ECC71","#F39C12","#E67E22","#ED145B","#C0392B","#7B0D1E"]

# IMC removido das features de treino e predicao
ALL_FEATURES = [
    "Gender","Age","Height","Weight","BMI",
    "family_history","FAVC","FCVC","NCP","CAEC",
    "SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS",
]
FORM_FEATURES = [
    "Gender","Age","family_history","FAVC","FCVC","NCP","CAEC",
    "SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS",
]

# Features exibidas em Feature Importance e SHAP (sem variaveis antropometricas)
DISPLAY_FEATURES = [
    "Gender","Age","family_history","FAVC","FCVC","NCP","CAEC",
    "SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS",
]
CAT_COLS = ["Gender","family_history","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"]
FEAT_LABELS = {
    "Gender":"Genero","Age":"Idade","Height":"Altura","Weight":"Peso",
    "family_history":"Historico Familiar","FAVC":"Alimentos Caloricos",
    "FCVC":"Consumo de Vegetais","NCP":"Refeicoes por Dia",
    "CAEC":"Lanches entre Refeicoes","SMOKE":"Fumante",
    "CH2O":"Consumo de Agua","SCC":"Monitora Calorias","FAF":"Atividade Fisica",
    "TUE":"Uso de Eletronicos","CALC":"Consumo de Alcool","MTRANS":"Meio de Transporte",
}

@st.cache_data
def load_data():
    df = pd.read_csv("data/Obesity.csv")
    df["BMI"] = (df["Weight"] / (df["Height"] ** 2)).round(2)
    df["Obesity_Label"] = df["Obesity"].map(OBESITY_LABELS)
    for col in ["FCVC","NCP","CH2O","FAF","TUE"]:
        df[col] = df[col].round().astype(int)
    df["Obesity"] = pd.Categorical(df["Obesity"], categories=OBESITY_ORDER, ordered=True)
    df["Obesity_Label"] = pd.Categorical(
        df["Obesity_Label"],
        categories=[OBESITY_LABELS[k] for k in OBESITY_ORDER], ordered=True)
    return df

@st.cache_resource
def train_model(df):
    le_dict = {}
    df_enc  = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    le_target = LabelEncoder()
    le_target.fit(OBESITY_ORDER)
    df_enc["target"] = le_target.transform(df["Obesity"].astype(str))
    X = df_enc[ALL_FEATURES]
    y = df_enc["target"]
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    acc_test  = accuracy_score(y_test, y_pred)
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    report    = classification_report(y_test, y_pred,
                    target_names=[str(c) for c in le_target.classes_], output_dict=True)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[:300])
    return dict(model=model, le_dict=le_dict, le_target=le_target,
                X_test=X_test, y_test=y_test, acc_test=acc_test,
                cv_scores=cv_scores, report=report,
                shap_vals=shap_vals, X_shap=X_test[:300])

df = load_data()
import pickle, os

@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

with st.spinner("Carregando modelo..."):
    res = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    modo = st.radio("Tema", ["Dark Mode","Light Mode"], horizontal=True)
    T    = THEME["dark"] if "Dark" in modo else THEME["light"]
    st.markdown("---")
    st.markdown("### Filtros EDA")
    genero = st.multiselect("Genero",
        options=df["Gender"].unique().tolist(),
        default=df["Gender"].unique().tolist(),
        format_func=lambda x: "Masculino" if x=="Male" else "Feminino")
    faixa_idade = st.slider("Faixa Etaria",
        int(df["Age"].min()), int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())))
    classes = st.multiselect("Classe de Obesidade",
        options=OBESITY_ORDER, default=OBESITY_ORDER,
        format_func=lambda x: OBESITY_LABELS[x])
    historico = st.multiselect("Historico Familiar",
        options=["yes","no"], default=["yes","no"],
        format_func=lambda x: "Sim" if x=="yes" else "Nao")
    st.markdown("---")
    top_n = st.slider("Top N variaveis SHAP", min_value=3, max_value=14, value=7)
    st.markdown("---")
    st.caption("FIAP POSTECH · Tech Challenge 4")

mask = (
    df["Gender"].isin(genero)
    & df["Age"].between(*faixa_idade)
    & df["Obesity"].isin(classes)
    & df["family_history"].isin(historico)
)
dff       = df[mask].copy()
total     = len(dff)
total_all = len(df)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def apply_theme(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["font"], family="Arial", size=12),
        xaxis=dict(gridcolor=T["grid"], linecolor=T["border"],
                   tickfont=dict(color=T["font"]),
                   title=dict(font=dict(color=T["font"]))),
        yaxis=dict(gridcolor=T["grid"], linecolor=T["border"],
                   tickfont=dict(color=T["font"]),
                   title=dict(font=dict(color=T["font"]))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["font"])),
        margin=dict(l=10, r=30, t=40, b=10),
    )
    return fig

def kpi(col, label, value, delta="", color=None):
    vc = color if color else T["accent"]
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val" style="color:{vc}">{value}</div>'
        f'<div class="kpi-dlt">{delta}</div>'
        f'</div>', unsafe_allow_html=True)

def subtitle(text):
    return f'<p style="color:{T["accent"]};font-weight:700;font-size:14px;margin:16px 0 8px">{text}</p>'

# ── CSS ───────────────────────────────────────────────────────────────────────
bg=T["bg"]; bg2=T["bg2"]; card=T["card"]; kbg=T["kpi"]
txt=T["text"]; sub=T["subtext"]; brd=T["border"]
acc=T["accent"]; acc2=T["accent2"]; side=T["sidebar"]
ibg=T["input_bg"]; itxt=T["input_txt"]
tbg=T["tag_bg"]; ttxt=T["tag_txt"]

st.markdown(f"""<style>
  .stApp {{background:{bg} !important}}
  .stApp p, .stApp h1,.stApp h2,.stApp h3 {{color:{txt} !important}}

  /* Sidebar */
  section[data-testid="stSidebar"] {{background:{side} !important;border-right:1px solid {brd}}}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span {{color:{acc} !important}}

  /* Labels de inputs — rosa com grafia branca */
  label, .stSelectbox label, .stNumberInput label,
  .stSlider label, .stMultiSelect label, .stRadio label {{
    color:{acc} !important;font-weight:600 !important;font-size:13px !important
  }}

  /* Selectbox — fundo escuro, texto branco */
  .stSelectbox > div > div,
  .stSelectbox > div > div > div {{
    background:{ibg} !important;color:{itxt} !important;
    border:1px solid {brd} !important;border-radius:6px !important;
  }}
  .stSelectbox > div > div > div > div {{color:{itxt} !important}}
  .stSelectbox svg {{fill:{acc} !important}}

  /* Dropdown popup options */
  [data-baseweb="popover"] ul li,
  [data-baseweb="menu"] li {{
    background:{ibg} !important;color:{itxt} !important;
  }}
  [data-baseweb="popover"] ul li:hover,
  [data-baseweb="menu"] li:hover {{
    background:{acc} !important;color:#fff !important;
  }}

  /* Number input */
  .stNumberInput input {{
    background:{ibg} !important;color:{itxt} !important;
    border:1px solid {brd} !important;border-radius:6px !important;
    font-size:15px !important;font-weight:600 !important;
  }}
  .stNumberInput button {{
    background:{ibg} !important;color:{itxt} !important;
    border:1px solid {brd} !important;
  }}

  /* Multiselect container — fundo escuro */
  .stMultiSelect > div > div,
  .stMultiSelect > div > div > div {{
    background:{ibg} !important;border:1px solid {brd} !important;border-radius:6px !important;
  }}
  /* Tags dentro do multiselect */
  .stMultiSelect span[data-baseweb="tag"] {{
    background:{tbg} !important;color:{ttxt} !important;
    border:1px solid {brd} !important;
  }}
  /* Texto dentro da tag */
  .stMultiSelect span[data-baseweb="tag"] span {{color:{ttxt} !important}}
  /* X dentro da tag — rosa */
  .stMultiSelect span[data-baseweb="tag"] svg {{fill:{acc} !important}}
  /* Botão X de limpar tudo — rosa, sem afetar fundo */
  .stMultiSelect button[aria-label*="Clear"] svg,
  .stMultiSelect button[title*="Clear"] svg,
  .stMultiSelect [role="button"] svg {{fill:{acc} !important}}
  /* Seta dropdown — rosa */
  .stMultiSelect [data-baseweb="select"] svg:last-of-type {{fill:{acc} !important}}
  /* Placeholder e texto digitado */
  .stMultiSelect input {{color:{itxt} !important;background:transparent !important}}

  /* Abas */
  .stTabs [data-baseweb="tab-list"] {{background:{bg2};border-radius:8px;padding:4px;gap:4px;border:1px solid {brd}}}
  .stTabs [data-baseweb="tab"] {{
    background:transparent;color:{acc};border-radius:6px;
    padding:8px 20px;font-size:13px;font-weight:600;border:none
  }}
  .stTabs [aria-selected="true"] {{background:{acc} !important;color:#fff !important}}
  /* Texto da aba selecionada em branco */
  .stTabs [aria-selected="true"] p,
  .stTabs [aria-selected="true"] span {{color:#fff !important}}

  /* KPI cards */
  .kpi-card {{
    background:{kbg};border:1px solid {brd};border-radius:10px;
    padding:18px 12px;text-align:center;margin-bottom:8px;
    min-height:108px;display:flex;flex-direction:column;
    justify-content:center;align-items:center;border-top:3px solid {acc};
  }}
  .kpi-lbl {{font-size:10px;color:{acc};margin-bottom:6px;
    text-transform:uppercase;letter-spacing:.08em;font-weight:700}}
  .kpi-val {{font-size:22px;font-weight:700;line-height:1.2;color:{txt}}}
  .kpi-dlt {{font-size:10px;color:{acc};margin-top:4px}}

  .sec-title {{
    font-size:12px;font-weight:700;color:{acc};
    text-transform:uppercase;letter-spacing:.1em;
    margin:20px 0 12px;padding-left:12px;border-left:3px solid {acc};
  }}
  .fiap-header {{
    background:linear-gradient(135deg,{acc} 0%,{acc2} 100%);
    padding:20px 28px;border-radius:12px;margin-bottom:24px;
  }}
  .fiap-title {{font-size:22px;font-weight:700;color:#fff;margin:0}}
  .fiap-sub   {{font-size:13px;color:rgba(255,255,255,.85);margin-top:4px}}
  .pred-box {{
    background:{card};border:1px solid {brd};
    border-top:4px solid {acc};border-radius:12px;
    padding:28px 20px;text-align:center;margin:10px 0;
  }}
  .pred-class {{font-size:26px;font-weight:700;margin:10px 0}}
  .pred-conf  {{font-size:14px;color:{acc};margin-top:6px;font-weight:600}}
  .stButton button {{
    background:{acc};color:#fff;border:none;border-radius:8px;
    font-weight:700;font-size:14px;padding:12px 0;
  }}
  .stButton button:hover {{background:{acc2}}}
  .stCaption, small {{color:{acc} !important}}

  /* Slider — thumb e track em rosa forte */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background:{acc} !important;
    border-color:{acc} !important;
  }}
  [data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(2) {{
    background:{acc} !important;
  }}
  [data-testid="stSlider"] div[class*="thumb"] {{
    background:{acc} !important;
    border-color:{acc} !important;
    box-shadow:0 0 0 4px {acc}33 !important;
  }}
  [data-testid="stSlider"] div[class*="track"] > div {{
    background:{acc} !important;
  }}

  /* Header e toolbar — fundo igual ao app */
  header[data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"] {{
    background:{bg} !important;
    border-bottom:1px solid {brd} !important;
  }}
  [data-testid="stDecoration"] {{ display:none !important; }}
</style>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="fiap-header">'
    f'<div class="fiap-title">🏥 Dashboard Preditivo de Obesidade</div>'
    f'<div class="fiap-sub">FIAP POSTECH · Tech Challenge 4 · XGBoost + SHAP · '
    f'{total:,} de {total_all:,} registros selecionados</div>'
    f'</div>', unsafe_allow_html=True)

# ── KPIs MODELO ───────────────────────────────────────────────────────────────
acc_pct = res["acc_test"] * 100
cv_pct  = res["cv_scores"].mean() * 100
cv_std  = res["cv_scores"].std() * 100

st.markdown('<div class="sec-title">Performance do Modelo</div>', unsafe_allow_html=True)
m1,m2,m3,m4,m5 = st.columns(5)
kpi(m1,"Algoritmo",      "XGBoost",         "300 arvores | profundidade 5")
kpi(m2,"Acuracia Teste", f"{acc_pct:.1f}%", "holdout 20% dos dados")
kpi(m3,"Acuracia CV",    f"{cv_pct:.1f}%",  f"desvio +/- {cv_std:.1f}% | 5 folds")
kpi(m4,"Classes",        "7",               "niveis de obesidade previstos")
kpi(m5,"Dataset",        f"{total_all:,}",  "registros · 17 variaveis")
st.markdown("<br>", unsafe_allow_html=True)

# ── KPIs DADOS ────────────────────────────────────────────────────────────────
pct_obeso = dff["Obesity"].isin(["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]).sum()/total*100 if total else 0
bmi_medio = dff["BMI"].mean() if total else 0
bmi_delta = bmi_medio - df["BMI"].mean()
pct_fam   = (dff["family_history"]=="yes").sum()/total*100 if total else 0
pct_ob30  = (dff["BMI"]>=30).sum()/total*100 if total else 0
faf_med   = dff["FAF"].mean() if total else 0
pct_favc  = (dff["FAVC"]=="yes").sum()/total*100 if total else 0
pct_smoke = (dff["SMOKE"]=="yes").sum()/total*100 if total else 0

st.markdown('<div class="sec-title">Resumo dos Dados Filtrados</div>', unsafe_allow_html=True)
g1,g2,g3,g4,g5,g6,g7,g8 = st.columns(8)
kpi(g1,"Registros",        f"{total:,}",        f"de {total_all:,} total")
kpi(g2,"Taxa de Obesos",   f"{pct_obeso:.1f}%", f"{int(pct_obeso*total/100)} casos")
kpi(g3,"IMC Medio",        f"{bmi_medio:.1f}",  f'{"+" if bmi_delta>=0 else ""}{bmi_delta:.1f} vs geral')
kpi(g4,"IMC acima de 30",  f"{pct_ob30:.1f}%",  "obesos clinicos")
kpi(g5,"Hist. Familiar",   f"{pct_fam:.1f}%",   "com historico positivo")
kpi(g6,"Atividade Fisica", f"{faf_med:.2f}",    "media (escala 0 a 3)")
kpi(g7,"Alim. Calorico",   f"{pct_favc:.1f}%",  "consomem frequentemente")
kpi(g8,"Fumantes",         f"{pct_smoke:.1f}%", "do grupo selecionado")
st.markdown("---")

aba_pred, aba_eda, aba_hab, aba_feat = st.tabs([
    "  Predicao Interativa  ",
    "  Perfil Geral  ",
    "  Habitos de Vida  ",
    "  Importancia e SHAP  ",
])

# ══ ABA 1 — PREDICAO ═════════════════════════════════════════════════════════
with aba_pred:
    st.markdown('<div class="sec-title">Insira os dados para prever o nivel de obesidade</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{acc};font-size:13px">Todos os campos sao obrigatorios. Altura e peso sao inferidos pela media do dataset — apenas habitos e perfil pessoal influenciam a predicao.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown('<div class="sec-title">Dados Pessoais</div>', unsafe_allow_html=True)
        p_gender = st.selectbox("Genero",
            options=["Male","Female"],
            format_func=lambda x: "Masculino" if x=="Male" else "Feminino",
            key="gender_sel")
        p_age = st.number_input(
            "Idade (anos inteiros, minimo 14, maximo 61)",
            min_value=14, max_value=61, value=25, step=1, key="age_inp")
        p_fam = st.selectbox("Possui historico familiar de sobrepeso?",
            options=["yes","no"],
            format_func=lambda x: "Sim" if x=="yes" else "Nao",
            key="fam_sel")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="sec-title">Medidas Corporais (informativas)</div>', unsafe_allow_html=True)
        p_height = st.number_input(
            "Altura (metros, ex: 1.70)",
            min_value=1.45, max_value=1.98, value=1.70, step=0.01,
            format="%.2f", key="height_inp")
        p_weight = st.number_input(
            "Peso (quilogramas, ex: 75.0)",
            min_value=39.0, max_value=173.0, value=75.0, step=0.5,
            format="%.1f", key="weight_inp")
        p_bmi = round(p_weight / (p_height ** 2), 1)
        bmi_cat = (
            "Abaixo do Peso" if p_bmi < 18.5 else
            "Peso Normal"    if p_bmi < 25.0 else
            "Sobrepeso"      if p_bmi < 30.0 else
            "Obeso"
        )
        bmi_color = (
            "#4D9DE0" if p_bmi < 18.5 else
            "#2ECC71" if p_bmi < 25.0 else
            "#F39C12" if p_bmi < 30.0 else
            "#ED145B"
        )
        st.markdown(
            f'<div class="kpi-card" style="margin-top:8px">'
            f'<div class="kpi-lbl">IMC Calculado</div>'
            f'<div class="kpi-val" style="color:{bmi_color};font-size:28px">{p_bmi}</div>'
            f'<div class="kpi-dlt">{bmi_cat}</div>'
            f'</div>', unsafe_allow_html=True)
        st.caption("Peso e Altura sao utilizados pelo modelo para classificacao. O IMC e calculado automaticamente e exibido como referencia.")

    with p2:
        st.markdown('<div class="sec-title">Habitos Alimentares</div>', unsafe_allow_html=True)
        p_favc = st.selectbox("Consome alimentos muito caloricos frequentemente?",
            options=["yes","no"],
            format_func=lambda x: "Sim" if x=="yes" else "Nao",
            key="favc_sel")
        p_fcvc = st.selectbox("Frequencia de consumo de vegetais nas refeicoes",
            options=[1,2,3],
            format_func=lambda x: {
                1:"1 - Raramente consome",
                2:"2 - Consome as vezes",
                3:"3 - Consome sempre"}[x],
            index=1, key="fcvc_sel")
        p_ncp = st.selectbox("Numero de refeicoes principais por dia",
            options=[1,2,3,4],
            format_func=lambda x: {
                1:"1 - Uma refeicao",
                2:"2 - Duas refeicoes",
                3:"3 - Tres refeicoes",
                4:"4 - Quatro ou mais"}[x],
            index=2, key="ncp_sel")
        p_caec = st.selectbox("Com que frequencia consome lanches entre refeicoes?",
            options=["no","Sometimes","Frequently","Always"],
            format_func=lambda x: {
                "no":"Nao consome lanches",
                "Sometimes":"As vezes",
                "Frequently":"Frequentemente",
                "Always":"Sempre"}[x],
            key="caec_sel")
        p_ch2o = st.selectbox("Consumo diario de agua",
            options=[1,2,3],
            format_func=lambda x: {
                1:"1 - Menos de 1 litro por dia",
                2:"2 - Entre 1 e 2 litros por dia",
                3:"3 - Mais de 2 litros por dia"}[x],
            index=1, key="ch2o_sel")

    with p3:
        st.markdown('<div class="sec-title">Estilo de Vida</div>', unsafe_allow_html=True)
        p_smoke = st.selectbox("E fumante?",
            options=["no","yes"],
            format_func=lambda x: "Nao e fumante" if x=="no" else "Sim, e fumante",
            key="smoke_sel")
        p_scc = st.selectbox("Monitora a ingestao calorica diaria?",
            options=["no","yes"],
            format_func=lambda x: "Nao monitora" if x=="no" else "Sim, monitora",
            key="scc_sel")
        p_faf = st.selectbox("Frequencia de atividade fisica por semana",
            options=[0,1,2,3],
            format_func=lambda x: {
                0:"0 - Nenhuma atividade",
                1:"1 - 1 a 2 vezes por semana",
                2:"2 - 3 a 4 vezes por semana",
                3:"3 - Todos os dias"}[x],
            index=1, key="faf_sel")
        p_tue = st.selectbox("Horas diarias com dispositivos eletronicos",
            options=[0,1,2],
            format_func=lambda x: {
                0:"0 - Ate 2 horas por dia",
                1:"1 - Entre 3 e 5 horas por dia",
                2:"2 - Mais de 5 horas por dia"}[x],
            index=1, key="tue_sel")
        p_calc = st.selectbox("Com que frequencia consome bebidas alcoolicas?",
            options=["no","Sometimes","Frequently","Always"],
            format_func=lambda x: {
                "no":"Nao consome alcool",
                "Sometimes":"As vezes",
                "Frequently":"Frequentemente",
                "Always":"Sempre"}[x],
            key="calc_sel")
        p_trans = st.selectbox("Qual o principal meio de transporte utilizado?",
            options=["Public_Transportation","Automobile","Walking","Motorbike","Bike"],
            format_func=lambda x: {
                "Public_Transportation":"Transporte Publico",
                "Automobile":"Automovel",
                "Walking":"A pe",
                "Motorbike":"Motocicleta",
                "Bike":"Bicicleta"}[x],
            key="trans_sel")

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("Prever Nivel de Obesidade", use_container_width=True)

    if btn:
        le = res["le_dict"]
        def enc(col, val): return int(le[col].transform([str(val)])[0])

        p_bmi_model = round(float(p_weight) / (float(p_height) ** 2), 2)
        entrada = pd.DataFrame([{
            "Gender":         enc("Gender", p_gender),
            "Age":            float(p_age),
            "Height":         float(p_height),
            "Weight":         float(p_weight),
            "BMI":            p_bmi_model,
            "family_history": enc("family_history", p_fam),
            "FAVC":           enc("FAVC", p_favc),
            "FCVC":           float(p_fcvc),
            "NCP":            float(p_ncp),
            "CAEC":           enc("CAEC", p_caec),
            "SMOKE":          enc("SMOKE", p_smoke),
            "CH2O":           float(p_ch2o),
            "SCC":            enc("SCC", p_scc),
            "FAF":            float(p_faf),
            "TUE":            float(p_tue),
            "CALC":           enc("CALC", p_calc),
            "MTRANS":         enc("MTRANS", p_trans),
        }])

        proba      = res["model"].predict_proba(entrada)[0]
        pred_idx   = int(proba.argmax())
        pred_class = res["le_target"].classes_[pred_idx]
        pred_label = OBESITY_LABELS.get(pred_class, pred_class)
        pred_color = OB_COLORS[OBESITY_ORDER.index(pred_class)] if pred_class in OBESITY_ORDER else acc
        conf       = float(proba[pred_idx]) * 100

        r1, r2 = st.columns([1,2])
        with r1:
            st.markdown(
                f'<div class="pred-box">'
                f'<div class="kpi-lbl">Classificacao Prevista pelo Modelo</div>'
                f'<div class="pred-class" style="color:{pred_color}">{pred_label}</div>'
                f'<div class="pred-conf">Confianca: {conf:.1f}%</div>'
                f'<hr style="border:none;border-top:1px solid {brd};margin:14px 0">'
                f'<div class="kpi-lbl">Idade: {int(p_age)} anos | '
                f'{"Masculino" if p_gender=="Male" else "Feminino"}</div>'
                f'</div>', unsafe_allow_html=True)

        with r2:
            st.markdown(subtitle("Probabilidade por classe (%)"), unsafe_allow_html=True)
            labels_ord = [OBESITY_LABELS.get(c,c) for c in res["le_target"].classes_]
            proba_pct  = [round(float(p)*100,1) for p in proba]
            cores_ord  = [OB_COLORS[OBESITY_ORDER.index(c)] if c in OBESITY_ORDER else "#aaa"
                          for c in res["le_target"].classes_]
            prob_df = pd.DataFrame({"Classe":labels_ord,"Prob":proba_pct,"Cor":cores_ord})
            prob_df = prob_df.sort_values("Prob", ascending=True)

            fig_p = go.Figure()
            for _, row in prob_df.iterrows():
                fig_p.add_trace(go.Bar(
                    x=[row["Prob"]], y=[row["Classe"]],
                    orientation="h", marker_color=row["Cor"],
                    showlegend=False,
                    hovertemplate=f"<b>{row['Classe']}</b>: {row['Prob']:.1f}%<extra></extra>"))
            anns = [dict(x=row["Prob"]+1, y=row["Classe"],
                         text=f"<b>{row['Prob']:.1f}%</b>",
                         showarrow=False, xanchor="left",
                         font=dict(color=T["font"],size=12))
                    for _, row in prob_df.iterrows()]
            fig_p.update_layout(
                xaxis=dict(title=dict(text="Probabilidade (%)",font=dict(color=T["font"])),
                           range=[0,115],tickfont=dict(color=T["font"])),
                yaxis=dict(tickfont=dict(color=T["font"])),
                annotations=anns)
            st.plotly_chart(apply_theme(fig_p, 300), use_container_width=True)

        st.markdown(f'<div class="sec-title">Impacto de cada variavel nesta predicao (SHAP) — Top {top_n}</div>', unsafe_allow_html=True)
        try:
            sv_ind = shap.TreeExplainer(res["model"]).shap_values(entrada)
            if isinstance(sv_ind, list):
                shap_row = np.array([sv_ind[c][0] for c in range(len(sv_ind))]).T[:,pred_idx]
            elif sv_ind.ndim == 3:
                shap_row = sv_ind[0,:,pred_idx]
            else:
                shap_row = sv_ind[0]

            form_idx = [ALL_FEATURES.index(f) for f in FORM_FEATURES]
            shap_df  = pd.DataFrame({
                "Variavel": [FEAT_LABELS.get(f,f) for f in FORM_FEATURES],
                "SHAP":     shap_row[form_idx],
            })
            mais  = shap_df[shap_df["SHAP"]>0].nlargest(top_n,"SHAP").sort_values("SHAP")
            menos = shap_df[shap_df["SHAP"]<0].nsmallest(top_n,"SHAP").copy()
            menos["ABS"] = menos["SHAP"].abs()
            menos = menos.sort_values("ABS")

            # Grafico unificado com slider nativo Plotly (feature 1+3)
            shap_sorted = shap_df.sort_values("SHAP")
            cores_shap  = [acc if v >= 0 else "#4D9DE0" for v in shap_sorted["SHAP"]]
            custom_shap = [
                f"{'Aumenta' if v>=0 else 'Reduz'} a probabilidade<br>Impacto: {'+'if v>=0 else ''}{v:.4f}"
                for v in shap_sorted["SHAP"]
            ]

            # Slider nativo Plotly para Top N — sem frames, restyle por visibilidade
            n_total  = len(shap_sorted)
            all_y    = shap_sorted["Variavel"].tolist()
            all_x    = shap_sorted["SHAP"].tolist()
            all_cor  = cores_shap
            all_cust = custom_shap
            all_txt  = [f"{'+'if v>=0 else ''}{v:.3f}" for v in shap_sorted["SHAP"]]

            steps = []
            for n in range(3, n_total+1):
                # Pega os n mais extremos (n//2 positivos + n//2 negativos)
                pos = shap_sorted[shap_sorted["SHAP"]>0].tail(n//2 + n%2)
                neg = shap_sorted[shap_sorted["SHAP"]<0].head(n//2)
                subset = pd.concat([neg, pos]).sort_values("SHAP")
                sy = subset["Variavel"].tolist()
                sx = subset["SHAP"].tolist()
                sc = [acc if v>=0 else "#4D9DE0" for v in sx]
                st2 = [f"{'+'if v>=0 else ''}{v:.3f}" for v in sx]
                scu = [f"{'Aumenta' if v>=0 else 'Reduz'} a probabilidade<br>Impacto: {'+'if v>=0 else ''}{v:.4f}" for v in sx]
                steps.append(dict(
                    method="restyle", label=str(n),
                    args=[{"x":[sx],"y":[sy],"marker.color":[sc],
                           "customdata":[scu],"text":[st2]}]
                ))

            n_init = min(top_n, n_total)
            init_step = steps[n_init-3]
            init_x = init_step["args"][0]["x"][0]
            init_y = init_step["args"][0]["y"][0]
            init_c = init_step["args"][0]["marker.color"][0]
            init_cu= init_step["args"][0]["customdata"][0]
            init_t = init_step["args"][0]["text"][0]

            fig_shap = go.Figure(go.Bar(
                x=init_x, y=init_y,
                orientation="h",
                marker_color=init_c,
                customdata=init_cu,
                hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
                text=init_t,
                textposition="outside",
            ))
            fig_shap.update_layout(
                sliders=[dict(
                    active=n_init-3,
                    currentvalue=dict(
                        prefix="Variaveis exibidas: ",
                        font=dict(color=T["font"], size=12),
                        visible=True, xanchor="left",
                    ),
                    pad=dict(t=40, b=10),
                    steps=steps,
                    bgcolor=T["card"],
                    bordercolor=brd,
                    tickcolor=T["font"],
                    font=dict(color=T["font"]),
                )],
                xaxis=dict(
                    title=dict(text="Impacto SHAP — positivo aumenta, negativo reduz", font=dict(color=T["font"])),
                    tickfont=dict(color=T["font"]),
                    zeroline=True, zerolinecolor=brd, zerolinewidth=1,
                ),
                yaxis=dict(tickfont=dict(color=T["font"])),
                showlegend=False,
                margin=dict(l=10,r=30,t=20,b=80),
            )
            st.plotly_chart(apply_theme(fig_shap, 420), use_container_width=True)
            st.caption("Arraste o slider abaixo do grafico para controlar quantas variaveis exibir. Vermelho = aumenta risco | Azul = reduz risco.")
        except Exception as e:
            st.warning(f"SHAP nao disponivel: {e}")

# ══ ABA 2 — PERFIL GERAL ═════════════════════════════════════════════════════
with aba_eda:
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown(subtitle("Distribuicao de registros por classe de obesidade"), unsafe_allow_html=True)
        cnt = dff.groupby("Obesity_Label",observed=True).size().reset_index(name="Qtd")
        cnt["Pct"] = (cnt["Qtd"]/total*100).round(1)
        fig = go.Figure(go.Bar(
            x=cnt["Obesity_Label"].tolist(), y=cnt["Qtd"].tolist(),
            marker_color=OB_COLORS[:len(cnt)],
            text=[f"{q} ({p}%)" for q,p in zip(cnt["Qtd"],cnt["Pct"])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} registros<extra></extra>"))
        fig.update_layout(showlegend=False,
            xaxis=dict(title=dict(text="Classe de Obesidade",font=dict(color=T["font"])),
                       tickangle=-20,tickfont=dict(color=T["font"])),
            yaxis=dict(title=dict(text="Numero de Registros",font=dict(color=T["font"])),
                       tickfont=dict(color=T["font"])))
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with c2:
        g1c, g2c = st.columns(2)
        with g1c:
            st.markdown(subtitle("Genero"), unsafe_allow_html=True)
            gen = dff["Gender"].map({"Male":"Masculino","Female":"Feminino"}).value_counts().reset_index()
            gen.columns = ["Genero","count"]
            fig = go.Figure(go.Pie(
                labels=gen["Genero"], values=gen["count"], hole=0.55,
                marker_colors=["#4D9DE0","#ED145B"],
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>"))
            fig.update_traces(textfont=dict(color="#FFFFFF"))
            st.plotly_chart(apply_theme(fig,240), use_container_width=True)
        with g2c:
            st.markdown(subtitle("Hist. Familiar"), unsafe_allow_html=True)
            hf = dff["family_history"].map({"yes":"Com historico","no":"Sem historico"}).value_counts().reset_index()
            hf.columns = ["Historico","count"]
            fig = go.Figure(go.Pie(
                labels=hf["Historico"], values=hf["count"], hole=0.55,
                marker_colors=["#ED145B","#2ECC71"],
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>"))
            fig.update_traces(textfont=dict(color="#FFFFFF"))
            st.plotly_chart(apply_theme(fig,240), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(subtitle("IMC por classe — mediana e dispersao"), unsafe_allow_html=True)
        fig = px.box(dff, x="Obesity_Label", y="BMI", color="Obesity_Label",
                     color_discrete_sequence=OB_COLORS, points=False,
                     labels={"Obesity_Label":"Classe","BMI":"IMC (kg/m²)"})
        fig.update_layout(showlegend=False, xaxis_tickangle=-25,
            xaxis=dict(title=dict(text="Classe de Obesidade",font=dict(color=T["font"])),tickfont=dict(color=T["font"])),
            yaxis=dict(title=dict(text="IMC (kg/m²)",font=dict(color=T["font"])),tickfont=dict(color=T["font"])))
        st.plotly_chart(apply_theme(fig,360), use_container_width=True)

    with c4:
        st.markdown(subtitle("Idade mediana por classe de obesidade"), unsafe_allow_html=True)
        medians = dff.groupby("Obesity_Label",observed=True)["Age"].median().reset_index()
        medians.columns = ["Classe","Mediana"]
        fig = go.Figure(go.Bar(
            x=medians["Classe"].tolist(), y=medians["Mediana"].tolist(),
            marker_color=OB_COLORS[:len(medians)],
            text=[f"{v:.1f} anos" for v in medians["Mediana"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Mediana: %{y:.1f} anos<extra></extra>"))
        fig.update_layout(showlegend=False, xaxis_tickangle=-25,
            xaxis=dict(title=dict(text="Classe de Obesidade",font=dict(color=T["font"])),tickfont=dict(color=T["font"])),
            yaxis=dict(title=dict(text="Idade Mediana (anos)",font=dict(color=T["font"])),tickfont=dict(color=T["font"])))
        st.plotly_chart(apply_theme(fig,360), use_container_width=True)

# ══ ABA 3 — HABITOS ══════════════════════════════════════════════════════════
with aba_hab:
    OB_LABELS_SHORT = [OBESITY_LABELS[k] for k in OBESITY_ORDER]
    HAB_COLS  = ["FAF","CH2O","FCVC","NCP","TUE"]
    HAB_NAMES = ["Atividade Fisica (0-3)","Agua/dia (1-3)","Vegetais (1-3)","Refeicoes/dia (1-4)","Eletronicos (0-2)"]

    # ── 1. Heatmap com hover customizado (feature 4) ──────────────────────────
    st.markdown(subtitle("Mapa de calor dos habitos por classe — passe o mouse para detalhes"), unsafe_allow_html=True)
    heat_df = dff.groupby("Obesity_Label", observed=True)[HAB_COLS].agg(["mean","std"]).round(3)
    heat_df.index = [OBESITY_LABELS.get(str(i),str(i)) for i in heat_df.index]
    z_vals, text_vals, custom_vals = [], [], []
    for hab, hab_name in zip(HAB_COLS, HAB_NAMES):
        mean_global = df[hab].mean()
        z_row, txt_row, cust_row = [], [], []
        for classe in heat_df.index:
            m   = heat_df.loc[classe,(hab,"mean")]
            std = heat_df.loc[classe,(hab,"std")]
            delta = m - mean_global
            z_row.append(m)
            txt_row.append(f"{m:.2f}")
            cust_row.append(f"Media: {m:.2f}<br>Desvio padrao: {std:.2f}<br>Media geral: {mean_global:.2f}<br>Diferenca vs geral: {'+'if delta>=0 else ''}{delta:.2f}")
        z_vals.append(z_row)
        text_vals.append(txt_row)
        custom_vals.append(cust_row)

    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=heat_df.index.tolist(),
        y=HAB_NAMES,
        colorscale="RdBu_r",
        text=text_vals, texttemplate="%{text}",
        customdata=custom_vals,
        hovertemplate="<b>%{y}</b><br>Classe: %{x}<br>%{customdata}<extra></extra>",
        colorbar=dict(title="Valor medio", tickfont=dict(color=T["font"]))
    ))
    fig_heat.update_layout(
        xaxis=dict(tickfont=dict(color=T["font"]), title=dict(text="Classe de Obesidade", font=dict(color=T["font"]))),
        yaxis=dict(tickfont=dict(color=T["font"]), title=dict(text="Habito", font=dict(color=T["font"]))),
    )
    st.plotly_chart(apply_theme(fig_heat, 360), use_container_width=True)

    # ── 2. updatemenus — botoes dentro do grafico (feature 2) ────────────────
    st.markdown(subtitle("Analise por habito — selecione o habito nos botoes do grafico"), unsafe_allow_html=True)
    st.caption("Use os botoes no topo do grafico para alternar entre os habitos de vida.")

    classes_ord = [OBESITY_LABELS[k] for k in OBESITY_ORDER if OBESITY_LABELS[k] in dff["Obesity_Label"].cat.categories]
    hab_means_all = {h: [] for h in HAB_COLS}
    hab_means_geral = {}
    for h in HAB_COLS:
        media_geral = df[h].mean()
        hab_means_geral[h] = media_geral
        for cls in classes_ord:
            sub = dff[dff["Obesity_Label"]==cls]
            hab_means_all[h].append(round(sub[h].mean(),3) if len(sub) else 0)

    traces = []
    for i, (h, hn) in enumerate(zip(HAB_COLS, HAB_NAMES)):
        vals   = hab_means_all[h]
        geral  = hab_means_geral[h]
        deltas = [round(v - geral, 3) for v in vals]
        custom = [f"Media: {v:.2f}<br>Media geral: {geral:.2f}<br>Diferenca: {'+'if d>=0 else ''}{d:.2f}" for v,d in zip(vals,deltas)]
        cores  = [acc if v >= geral else "#4D9DE0" for v in vals]
        traces.append(go.Bar(
            name=hn,
            x=classes_ord,
            y=vals,
            visible=(i==0),
            marker_color=cores,
            customdata=custom,
            hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        ))

    buttons = []
    for i, hn in enumerate(HAB_NAMES):
        vis = [j==i for j in range(len(HAB_NAMES))]
        buttons.append(dict(label=hn.split(" (")[0], method="update",
                            args=[{"visible": vis},
                                  {"title": f"<span style='color:{T['font']}'>{hn} por classe de obesidade</span>"}]))

    fig_menu = go.Figure(data=traces)
    fig_menu.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.0, y=1.18,
            xanchor="left", yanchor="top",
            buttons=buttons,
            bgcolor=T["card"],
            bordercolor=acc,
            borderwidth=1,
            font=dict(color=T["font"], size=11),
            pad=dict(r=4, t=4),
        )],
        xaxis=dict(tickangle=-20, tickfont=dict(color=T["font"]),
                   title=dict(text="Classe de Obesidade", font=dict(color=T["font"]))),
        yaxis=dict(tickfont=dict(color=T["font"]),
                   title=dict(text="Valor medio", font=dict(color=T["font"]))),
        showlegend=False,
        margin=dict(l=10,r=30,t=80,b=10),
    )
    st.plotly_chart(apply_theme(fig_menu, 400), use_container_width=True)

    # ── 3. Histograma com rangeslider (feature 6) ────────────────────────────
    st.markdown(subtitle("Distribuicao de idades — arraste o seletor abaixo para filtrar o intervalo"), unsafe_allow_html=True)
    st.caption("O mini-grafico abaixo e um seletor de intervalo. Arraste as bordas para filtrar a faixa de idade.")

    fig_range = go.Figure()
    for cls, cor in zip(classes_ord, OB_COLORS[:len(classes_ord)]):
        sub = dff[dff["Obesity_Label"]==cls]["Age"]
        if len(sub) == 0: continue
        fig_range.add_trace(go.Histogram(
            x=sub.tolist(), name=cls,
            marker_color=cor, opacity=0.72,
            nbinsx=20, bingroup=1,
            hovertemplate=f"<b>{cls}</b><br>Idade: %{{x}}<br>Contagem: %{{y}}<extra></extra>",
        ))
    fig_range.update_layout(
        barmode="overlay",
        xaxis=dict(
            title=dict(text="Idade (anos)", font=dict(color=T["font"])),
            tickfont=dict(color=T["font"]),
            rangeslider=dict(
                visible=True,
                thickness=0.07,
                bgcolor=T["card"],
                bordercolor=brd,
                borderwidth=1,
            ),
            range=[14, 61],
        ),
        yaxis=dict(
            title=dict(text="Contagem", font=dict(color=T["font"])),
            tickfont=dict(color=T["font"]),
        ),
        legend=dict(font=dict(color=T["font"]), orientation="h", y=1.08, x=0),
        margin=dict(l=10,r=30,t=50,b=60),
    )
    st.plotly_chart(apply_theme(fig_range, 400), use_container_width=True)

    # ── 4. Subplots sincronizados — alcool, fumante, monit. (feature 5) ──────
    st.markdown(subtitle("Comportamentos de risco por classe — graficos sincronizados (zoom em um afeta todos)"), unsafe_allow_html=True)
    st.caption("Faca zoom em qualquer grafico — os tres sincronizam automaticamente pelo eixo X.")

    pct_alc  = dff.groupby("Obesity_Label",observed=True).apply(lambda x: round((x["CALC"]!="no").mean()*100,1)).reindex(classes_ord).fillna(0)
    pct_fum  = dff.groupby("Obesity_Label",observed=True).apply(lambda x: round((x["SMOKE"]=="yes").mean()*100,1)).reindex(classes_ord).fillna(0)
    pct_cal  = dff.groupby("Obesity_Label",observed=True).apply(lambda x: round((x["FAVC"]=="yes").mean()*100,1)).reindex(classes_ord).fillna(0)
    pct_mon  = dff.groupby("Obesity_Label",observed=True).apply(lambda x: round((x["SCC"]=="yes").mean()*100,1)).reindex(classes_ord).fillna(0)

    fig_sub = make_subplots(
        rows=1, cols=3,
        shared_xaxes=True,
        subplot_titles=["Consome Alcool (%)", "Fumantes (%)", "Alim. Calorico (%)"],
        horizontal_spacing=0.08,
    )

    for col_idx, (vals, cor, label) in enumerate([
        (pct_alc,  acc,       "Alcool"),
        (pct_fum,  "#4D9DE0", "Fumante"),
        (pct_cal,  "#2ECC71", "Alim.Cal."),
    ], start=1):
        fig_sub.add_trace(go.Bar(
            x=classes_ord,
            y=vals.tolist(),
            name=label,
            marker_color=cor,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>"+label+": %{y:.1f}%<extra></extra>",
        ), row=1, col=col_idx)

    fig_sub.update_layout(
        showlegend=False,
        yaxis=dict(range=[0,115], tickfont=dict(color=T["font"]),
                   title=dict(text="% do grupo", font=dict(color=T["font"]))),
        yaxis2=dict(range=[0,115], tickfont=dict(color=T["font"])),
        yaxis3=dict(range=[0,115], tickfont=dict(color=T["font"])),
        xaxis=dict(tickangle=-30, tickfont=dict(color=T["font"])),
        xaxis2=dict(tickangle=-30, tickfont=dict(color=T["font"])),
        xaxis3=dict(tickangle=-30, tickfont=dict(color=T["font"])),
        margin=dict(l=10,r=10,t=50,b=10),
    )
    for ann in fig_sub.layout.annotations:
        ann.font = dict(color=T["font"], size=13)
    st.plotly_chart(apply_theme(fig_sub, 420), use_container_width=True)


# ══ ABA 4 — IMPORTANCIA E SHAP ═══════════════════════════════════════════════
with aba_feat:
    st.markdown(subtitle("Importancia das variaveis no modelo XGBoost (ganho medio)"), unsafe_allow_html=True)
    # Filtrar apenas features comportamentais (excluir Height, Weight, BMI)
    display_idx = [ALL_FEATURES.index(f) for f in DISPLAY_FEATURES]
    fi = pd.DataFrame({
        "Feature":     [FEAT_LABELS.get(f,f) for f in DISPLAY_FEATURES],
        "Importancia": res["model"].feature_importances_[display_idx],
    }).sort_values("Importancia")
    fi_median  = fi["Importancia"].median()
    colors_fi  = [acc if v > fi_median else "#555555" for v in fi["Importancia"]]
    fig = go.Figure(go.Bar(
        x=fi["Importancia"].tolist(), y=fi["Feature"].tolist(),
        orientation="h", marker_color=colors_fi,
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>"))
    fig.update_layout(showlegend=False,
        xaxis=dict(title=dict(text="Importancia (ganho medio)",font=dict(color=T["font"])),tickfont=dict(color=T["font"])),
        yaxis=dict(tickfont=dict(color=T["font"])))
    st.plotly_chart(apply_theme(fig,480), use_container_width=True)

    st.markdown(subtitle(f"SHAP Global — impacto medio de cada variavel por classe (Top {top_n})"), unsafe_allow_html=True)
    st.caption("Calculado sobre 300 amostras do conjunto de teste. Cores mais intensas = maior impacto medio naquela classe.")
    try:
        sv = res["shap_vals"]
        feat_labels_all = [FEAT_LABELS.get(f,f) for f in ALL_FEATURES]
        class_labels    = [OBESITY_LABELS.get(c,c) for c in res["le_target"].classes_]
        if isinstance(sv,list):
            shap_mat = np.array([np.abs(s).mean(axis=0) for s in sv]).T
        elif sv.ndim==3:
            shap_mat = np.abs(sv).mean(axis=0)
        else:
            shap_mat = np.abs(sv).mean(axis=0).reshape(-1,1)
            class_labels = ["Global"]
        shap_df_g = pd.DataFrame(shap_mat, index=feat_labels_all, columns=class_labels)
        # Filtrar apenas features comportamentais (excluir Height, Weight, BMI)
        display_labels = [FEAT_LABELS.get(f,f) for f in DISPLAY_FEATURES]
        shap_df_g = shap_df_g.loc[shap_df_g.index.isin(display_labels)]
        top_idx   = shap_df_g.mean(axis=1).nlargest(top_n).index
        shap_top  = shap_df_g.loc[top_idx]

        fig = go.Figure(go.Heatmap(
            z=shap_top.values.tolist(),
            x=class_labels,
            y=top_idx.tolist(),
            colorscale="Reds",
            text=[[f"{v:.3f}" for v in row] for row in shap_top.values],
            texttemplate="%{text}",
            hovertemplate="Variavel: %{y}<br>Classe: %{x}<br>|SHAP|: %{z:.4f}<extra></extra>"))
        fig.update_layout(
            xaxis=dict(title=dict(text="Classe de Obesidade",font=dict(color=T["font"])),
                       tickangle=-30,tickfont=dict(color=T["font"])),
            yaxis=dict(title=dict(text="Variavel",font=dict(color=T["font"])),
                       tickfont=dict(color=T["font"])))
        st.plotly_chart(apply_theme(fig,420), use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP global: {e}")

    st.markdown(subtitle("Relatorio de Classificacao por Classe"), unsafe_allow_html=True)
    rep = res.get("report", {})
    rep_rows = []
    for cls in res["le_target"].classes_:
        r = rep.get(str(cls),{})
        rep_rows.append({
            "Classe":   OBESITY_LABELS.get(cls,cls),
            "Precisao": f"{r.get('precision',0):.3f}",
            "Recall":   f"{r.get('recall',0):.3f}",
            "F1-Score": f"{r.get('f1-score',0):.3f}",
            "Suporte":  int(r.get("support",0)),
        })
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Dashboard Preditivo de Obesidade v5 | XGBoost + SHAP | FIAP POSTECH · Tech Challenge 4")
