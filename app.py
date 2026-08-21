import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Talent Matching AI - FGV People Analytics",
    page_icon="🎯",
    layout="wide"
)

# Estilização CSS customizada
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Talent Matching AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema Inteligente de Análise de Aderência de Candidatos (LinkedIn x Vagas)</div>', unsafe_allow_html=True)

# --- FUNÇÕES DE SUPORTE ---
def extrair_texto_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    texto = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texto += t + "\n"
    return texto

def estimar_anos_experiencia(texto):
    # Procura padrões de datas no texto do LinkedIn
    anos_encontrados = re.findall(r'\b(20[0-2][0-9]|19[8-9][0-9])\b', texto)
    if len(anos_encontrados) >= 2:
        anos = [int(a) for a in anos_encontrados]
        ano_min = min(anos)
        ano_max = max(anos)
        exp = max(1.0, float(ano_max - ano_min))
        return min(exp, 25.0)
    return 4.0

@st.cache_resource
def treinar_modelo_base():
    n = 600
    np.random.seed(42)
    vaga_ref = "Buscamos profissional de dados com experiencia em Python, SQL, Machine Learning, Power BI, GCP e analise de negocios."
    vaga_exp_ref = 4.0

    perfis_base = [
        "Coordenador de BI com foco em Power BI, SQL avancado, Python, GCP, Databricks e gestao de times de dados.",
        "Analytics Engineer Senior com experiencia em Data Lake GCP, BigQuery, Dataform e pipelines de dados.",
        "Cientista de dados com foco em Python, Machine Learning, modelos preditivos e SQL avançado em empresas de tecnologia.",
        "Analista de Dados com experiência em SQL, PowerBI, Excel avançado e dashboards de negócios para área comercial.",
        "Desenvolvedor Software Backend Python, API REST, Docker, PostgreSQL e microsserviços cloud.",
        "Engenheiro de Machine Learning com vivência em MLOps, infraestrutura de modelos em nuvem, Python e CI/CD.",
        "Estagiário de Administração apaixonado por estatística, análise de dados e suporte ao time de vendas.",
        "Gerente de Produtos de Dados com visão de negócios, People Analytics, liderança de equipes e métodos ágeis."
    ]
    indices = np.random.choice(len(perfis_base), size=n)
    ruidos = [" Forte atuação corporativa.", " Foco em entregas estratégicas.", " Vivência em grandes projetos."]
    textos_candidatos = [perfis_base[i] + np.random.choice(ruidos) for i in indices]
    anos_experiencia = np.round(np.clip(np.random.gamma(shape=2.5, scale=1.8, size=n), 0.5, 20.0), 1)

    pesos_tecnicos = [95, 92, 90, 68, 55, 88, 30, 78]
    score_base = np.array([pesos_tecnicos[i] for i in indices])
    fator_exp = np.where(
        anos_experiencia < vaga_exp_ref,
        (anos_experiencia / vaga_exp_ref)**1.5,
        1.0 + 0.05 * np.log1p(np.maximum(0, anos_experiencia - vaga_exp_ref))
    )
    ruido = np.random.normal(0, 4.0, size=n)
    y_real = np.clip(score_base * fator_exp + ruido, 0, 100)

    df_train = pd.DataFrame({"texto": textos_candidatos, "exp": anos_experiencia, "y": y_real})
    X_tr_df, _, y_tr, _ = train_test_split(df_train[["texto", "exp"]], df_train["y"], test_size=0.3, random_state=42)

    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    tfidf.fit([vaga_ref] + list(X_tr_df["texto"]))

    v_tr = tfidf.transform(X_tr_df["texto"]).toarray()
    sc_exp = StandardScaler()
    e_tr = sc_exp.fit_transform(X_tr_df[["exp"]])

    X_nn = np.hstack([v_tr, e_tr])
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42, early_stopping=True)
    mlp.fit(X_nn, y_tr)

    return tfidf, sc_exp, mlp

tfidf_model, scaler_exp_model, mlp_model = treinar_modelo_base()

# --- LAYOUT PRINCIPAL (2 COLUNAS) ---
col_candidato, col_vaga = st.columns(2)

with col_candidato:
    st.subheader("👤 1. Perfil do Candidato (LinkedIn)")
    opcao_input = st.radio("Como deseja fornecer o perfil?", ["Upload de PDF do LinkedIn", "Cole o Texto do Perfil"])
    
    texto_perfil = ""
    if opcao_input == "Upload de PDF do LinkedIn":
        file_pdf = st.file_uploader("Selecione o PDF exportado do LinkedIn", type=["pdf"])
        if file_pdf is not None:
            texto_perfil = extrair_texto_pdf(file_pdf)
            st.success("✅ PDF lido com sucesso!")
    else:
        texto_perfil = st.text_area("Cole o resumo/histórico do LinkedIn aqui:", height=200, placeholder="Ex: Sou Coordenador de BI com experiência em Power BI, SQL, Python...")
    
    exp_estimada = estimar_anos_experiencia(texto_perfil) if texto_perfil else 4.0
    exp_candidato = st.number_input("Tempo de experiência acumulado (anos):", min_value=0.5, max_value=30.0, value=float(exp_estimada), step=0.5)

with col_vaga:
    st.subheader("💼 2. Descrição da Vaga Alvo")
    vaga_pre = st.selectbox(
        "Selecione um modelo de vaga ou escolha 'Personalizada':",
        [
            "Coordenador de BI / Analytics Engineer",
            "Cientista de Dados / Machine Learning",
            "Analista de Dados Pleno/Sênior",
            "Personalizada (Digitar Descrição)"
        ]
    )

    if vaga_pre == "Coordenador de BI / Analytics Engineer":
        vaga_texto = "Buscamos Coordenador de BI e Analytics Engineer com experiência em Power BI, SQL avançado, Python, Databricks, GCP (BigQuery, Dataform), gestão de equipes de dados, governança e estruturação de Data Lakes."
        vaga_exp = 4.0
    elif vaga_pre == "Cientista de Dados / Machine Learning":
        vaga_texto = "Buscamos Cientista de Dados com experiência em Python, Machine Learning, Redes Neurais, SQL, MLOps e modelagem preditiva em produção para People Analytics e negócios."
        vaga_exp = 4.0
    elif vaga_pre == "Analista de Dados Pleno/Sênior":
        vaga_texto = "Buscamos Analista de Dados com domínio em SQL, Power BI, construção de dashboards, Excel avançado, análise de KPIs e suporte a decisão de negócios."
        vaga_exp = 3.0
    else:
        vaga_texto = st.text_area("Digite a descrição da vaga personalizada:", height=150, placeholder="Descreva os requisitos, ferramentas e qualificações exigidas...")
        vaga_exp = st.number_input("Anos de experiência exigidos pela vaga:", min_value=0.5, max_value=20.0, value=3.0, step=0.5)

st.markdown("---")

# --- BOTÃO DE AÇÃO & RESULTADOS ---
if st.button("🚀 Calcular Aderência de Match", type="primary", use_container_width=True):
    if not texto_perfil.strip():
        st.warning("⚠️ Por favor, forneça o perfil do candidato (via PDF ou texto) antes de calcular.")
    elif not vaga_texto.strip():
        st.warning("⚠️ Por favor, forneça a descrição da vaga.")
    else:
        # Processamento
        candidato_vec = tfidf_model.transform([texto_perfil])
        vaga_vec = tfidf_model.transform([vaga_texto])

        sim_cosseno = cosine_similarity(candidato_vec, vaga_vec).flatten()[0]
        fator_exp = np.clip(exp_candidato / vaga_exp, 0.2, 1.2)
        score_baseline = np.clip(sim_cosseno * 100 * fator_exp, 0, 100)

        candidato_exp_scaled = scaler_exp_model.transform([[exp_candidato]])
        X_cand_nn = np.hstack([candidato_vec.toarray(), candidato_exp_scaled])
        score_mlp = np.clip(mlp_model.predict(X_cand_nn)[0], 0, 100)

        st.subheader("📊 Resultado da Análise de Aderência")

        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="Match Rede Neural (MLP)", value=f"{score_mlp:.1f}%")
        with res_col2:
            st.metric(label="Match Baseline (TF-IDF)", value=f"{score_baseline:.1f}%")
        with res_col3:
            st.metric(label="Similaridade de Texto", value=f"{sim_cosseno*100:.1f}%")

        # Classificação de Aderência
        if score_mlp >= 75.0:
            st.success("🟢 **ALTA ADERÊNCIA (Perfil Fortemente Recomendado)**: O candidato possui competências e experiência altamente alinhadas com as exigências da vaga.")
        elif score_mlp >= 50.0:
            st.info("🟡 **ADERÊNCIA MODERADA (Apto para Entrevista)**: O candidato possui sólida base analítica/técnica e potencial de adaptação rápida para o cargo.")
        else:
            st.error("🔴 **BAIXA ADERÊNCIA**: O histórico do candidato apresenta divergência relevante dos requisitos específicos exigidos para este cargo.")

        # Gráfico comparativo
        fig, ax = plt.subplots(figsize=(8, 3))
        modelos = ["Baseline (TF-IDF)", "Rede Neural (MLP)"]
        scores = [score_baseline, score_mlp]
        cores = ["#DD8452", "#4C72B0"]
        
        bars = ax.barh(modelos, scores, color=cores, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score de Match (%)")
        ax.set_title("Comparação dos Modelos de Aderência")
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va="center", fontweight="bold")
            
        st.pyplot(fig)
