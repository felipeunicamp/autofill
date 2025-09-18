import streamlit as st
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain import hub
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import io
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="🤖 Preenchimento Automático de Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🤖 Sistema de Preenchimento Automático Q&A</h1>
    <p>Faça upload do seu arquivo Excel e deixe a IA preencher as respostas automaticamente!</p>
</div>
""", unsafe_allow_html=True)

# Inicializar session state
if 'base_dados' not in st.session_state:
    st.session_state.base_dados = None
if 'arquivo_upload' not in st.session_state:
    st.session_state.arquivo_upload = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'resultados_preenchimento' not in st.session_state:
    st.session_state.resultados_preenchimento = None

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    # Configuração da API Key
    st.subheader("🔑 API Key do Google")
    api_key = st.text_input(
        "Digite sua API Key do Google Gemini:",
        type="password",
        help="Necessária para usar o modelo de IA"
    )

    if api_key:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(
                temperature=0,
                google_api_key=api_key,
                model='gemini-2.0-flash-exp'
            )
            st.success("✅ API Key configurada!")
        except Exception as e:
            st.error(f"❌ Erro na API Key: {e}")

    st.divider()

    # Configurações de busca
    st.subheader("🎯 Configurações de Busca")
    limite_similaridade = st.slider(
        "Limite de Similaridade (%)",
        min_value=30,
        max_value=100,
        value=70,
        help="Quanto maior, mais restritiva a busca"
    )

    num_resultados_similares = st.slider(
        "Número de Resultados Similares",
        min_value=1,
        max_value=10,
        value=5,
        help="Quantas perguntas similares considerar"
    )


# Funções auxiliares
@st.cache_data
def carregar_base_dados(arquivo):
    """Carrega e processa a base de dados histórica"""
    try:
        if arquivo.name.endswith('.xlsx'):
            df = pd.read_excel(arquivo)
        elif arquivo.name.endswith('.csv'):
            df = pd.read_csv(arquivo)
        else:
            return None, "Formato não suportado"

        # Verificar se tem as colunas necessárias
        colunas_necessarias = ['Pergunta', 'Resposta']
        colunas_alternativas = [
            ['pergunta', 'resposta'],
            ['question', 'answer'],
            ['perguntas', 'respostas'],
            ['Question', 'Answer']
        ]

        # Tentar encontrar as colunas corretas
        colunas_encontradas = None
        for alternativa in [colunas_necessarias] + colunas_alternativas:
            if all(col in df.columns for col in alternativa):
                colunas_encontradas = alternativa
                break

        if not colunas_encontradas:
            # Tentar mapear automaticamente
            colunas_df = df.columns.tolist()
            if len(colunas_df) >= 2:
                df = df.rename(columns={
                    colunas_df[0]: 'Pergunta',
                    colunas_df[1]: 'Resposta'
                })
                colunas_encontradas = ['Pergunta', 'Resposta']

        if colunas_encontradas:
            if colunas_encontradas != ['Pergunta', 'Resposta']:
                df = df.rename(columns={
                    colunas_encontradas[0]: 'Pergunta',
                    colunas_encontradas[1]: 'Resposta'
                })

            # Limpar dados
            df = df.dropna(subset=['Pergunta', 'Resposta'])
            df['Pergunta'] = df['Pergunta'].astype(str)
            df['Resposta'] = df['Resposta'].astype(str)

            return df, None
        else:
            return None, "Colunas 'Pergunta' e 'Resposta' não encontradas"

    except Exception as e:
        return None, f"Erro ao carregar arquivo: {e}"


def buscar_resposta_fuzzy(df, texto_busca, limite=70):
    """Busca usando fuzzy matching"""
    try:
        # Match exato primeiro
        matches_exatos = df[df['Pergunta'].str.lower() == texto_busca.lower()]
        if not matches_exatos.empty:
            return {
                'resposta': matches_exatos.iloc[0]['Resposta'],
                'pergunta_encontrada': matches_exatos.iloc[0]['Pergunta'],
                'similaridade': 100,
                'tipo_match': 'exato'
            }

        # Busca fuzzy
        perguntas = df['Pergunta'].tolist()
        melhor_match = process.extractOne(texto_busca, perguntas, scorer=fuzz.token_sort_ratio)

        if melhor_match and melhor_match[1] >= limite:
            pergunta_encontrada = melhor_match[0]
            linha = df[df['Pergunta'] == pergunta_encontrada]

            return {
                'resposta': linha.iloc[0]['Resposta'],
                'pergunta_encontrada': pergunta_encontrada,
                'similaridade': melhor_match[1],
                'tipo_match': 'aproximado'
            }

        return None

    except Exception as e:
        st.error(f"Erro na busca fuzzy: {e}")
        return None


def buscar_perguntas_similares(df, texto_busca, num_resultados=5, limite=30):
    """Busca múltiplas perguntas similares"""
    try:
        perguntas = df['Pergunta'].tolist()
        matches = process.extract(texto_busca, perguntas, scorer=fuzz.token_sort_ratio, limit=num_resultados)

        resultados = []
        for pergunta, similaridade in matches:
            if similaridade >= limite:
                linha = df[df['Pergunta'] == pergunta]
                if not linha.empty:
                    resultados.append({
                        'pergunta': pergunta,
                        'resposta': linha.iloc[0]['Resposta'],
                        'similaridade': similaridade
                    })

        resultados.sort(key=lambda x: x['similaridade'], reverse=True)
        return resultados

    except Exception as e:
        st.error(f"Erro na busca de perguntas similares: {e}")
        return []


def gerar_resposta_ia(pergunta, df, llm, limite_similaridade, num_resultados):
    """Gera resposta usando IA baseada na base de dados"""
    if not llm:
        return "❌ LLM não configurado"

    # Busca direta
    resultado_direto = buscar_resposta_fuzzy(df, pergunta, limite=limite_similaridade)

    if resultado_direto and resultado_direto['similaridade'] >= limite_similaridade:
        return {
            'resposta': resultado_direto['resposta'],
            'confianca': resultado_direto['similaridade'],
            'tipo': 'match_direto',
            'fonte': resultado_direto['pergunta_encontrada']
        }

    # Busca similares
    resultados_similares = buscar_perguntas_similares(df, pergunta, num_resultados, limite=25)

    if resultados_similares:
        contexto = ""
        for i, resultado in enumerate(resultados_similares):
            contexto += f"""
Pergunta {i + 1} (Similaridade: {resultado['similaridade']}%): {resultado['pergunta']}
Resposta {i + 1}: {resultado['resposta']}
---
"""

        template = f"""
        Você é um assistente especializado que analisa perguntas e respostas de um banco de dados.

        PERGUNTA DO USUÁRIO: {pergunta}

        PERGUNTAS E RESPOSTAS SIMILARES ENCONTRADAS:
        {contexto}

        INSTRUÇÕES:
        1. Analise a pergunta do usuário e compare com as perguntas similares
        2. Se encontrar uma resposta adequada, use ela
        3. Se precisar combinar informações, faça de forma coerente
        4. Baseie-se APENAS nas informações fornecidas
        5. Seja claro e direto
        6. Se não houver informação suficiente, diga que não foi possível encontrar

        RESPOSTA (apenas o texto da resposta, sem explicações adicionais):
        """

        try:
            resposta = llm.invoke(template)
            return {
                'resposta': resposta.content,
                'confianca': max([r['similaridade'] for r in resultados_similares]),
                'tipo': 'ia_baseada',
                'fonte': f"{len(resultados_similares)} perguntas similares"
            }
        except Exception as e:
            return {
                'resposta': f"Erro ao gerar resposta: {e}",
                'confianca': 0,
                'tipo': 'erro',
                'fonte': 'N/A'
            }

    return {
        'resposta': "Não foi possível encontrar informações relacionadas na base de dados.",
        'confianca': 0,
        'tipo': 'nao_encontrado',
        'fonte': 'N/A'
    }


# Interface principal
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 1. Carregar Base de Dados Histórica")

    arquivo_base = st.file_uploader(
        "Faça upload da sua base de dados (Excel/CSV):",
        type=['xlsx', 'xls', 'csv'],
        help="Arquivo deve conter colunas 'Pergunta' e 'Resposta'"
    )

    if arquivo_base:
        base_dados, erro = carregar_base_dados(arquivo_base)

        if erro:
            st.markdown(f'<div class="error-box">❌ {erro}</div>', unsafe_allow_html=True)
        else:
            st.session_state.base_dados = base_dados
            st.markdown(f'<div class="success-box">✅ Base carregada: {len(base_dados)} registros</div>',
                        unsafe_allow_html=True)

            # Preview da base
            with st.expander("👀 Preview da Base de Dados"):
                st.dataframe(base_dados.head(10))

with col2:
    st.header("📤 2. Upload do Arquivo para Preencher")

    arquivo_preencher = st.file_uploader(
        "Faça upload do arquivo para preencher:",
        type=['xlsx', 'xls', 'csv'],
        help="Arquivo deve conter pelo menos uma coluna 'Pergunta'"
    )

    if arquivo_preencher:
        try:
            if arquivo_preencher.name.endswith('.xlsx'):
                df_preencher = pd.read_excel(arquivo_preencher)
            else:
                df_preencher = pd.read_csv(arquivo_preencher)

            st.session_state.arquivo_upload = df_preencher
            st.markdown(f'<div class="success-box">✅ Arquivo carregado: {len(df_preencher)} linhas</div>',
                        unsafe_allow_html=True)

            # Preview do arquivo
            with st.expander("👀 Preview do Arquivo"):
                st.dataframe(df_preencher.head(10))

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Erro ao carregar: {e}</div>', unsafe_allow_html=True)

# Seção de processamento
if st.session_state.base_dados is not None and st.session_state.arquivo_upload is not None and st.session_state.llm is not None:
    st.header("🚀 3. Processamento Automático")

    df_base = st.session_state.base_dados
    df_upload = st.session_state.arquivo_upload.copy()

    # Identificar coluna de perguntas
    colunas_pergunta = [col for col in df_upload.columns if 'pergunta' in col.lower() or 'question' in col.lower()]

    if not colunas_pergunta:
        colunas_pergunta = [df_upload.columns[0]]  # Usar primeira coluna como fallback

    coluna_pergunta = st.selectbox(
        "Selecione a coluna que contém as perguntas:",
        df_upload.columns.tolist(),
        index=df_upload.columns.tolist().index(colunas_pergunta[0]) if colunas_pergunta else 0
    )

    # Verificar se já existe coluna de resposta
    tem_coluna_resposta = any('resposta' in col.lower() or 'answer' in col.lower() for col in df_upload.columns)

    col_proc1, col_proc2 = st.columns([1, 1])

    with col_proc1:
        processar_tudo = st.button("🤖 Processar Todas as Perguntas", type="primary")

    with col_proc2:
        if tem_coluna_resposta:
            processar_vazias = st.button("📝 Processar Apenas Respostas Vazias")
        else:
            processar_vazias = False

    if processar_tudo or processar_vazias:
        # Criar coluna de resposta se não existir
        if 'Resposta' not in df_upload.columns:
            df_upload['Resposta'] = ''

        progress_bar = st.progress(0)
        status_text = st.empty()

        resultados = []
        total_linhas = len(df_upload)

        for idx, row in df_upload.iterrows():
            pergunta = str(row[coluna_pergunta])

            # Pular se for processar apenas vazias e já tiver resposta
            if processar_vazias and pd.notna(row.get('Resposta', '')) and str(row.get('Resposta', '')).strip():
                continue

            status_text.text(f"Processando: {pergunta[:50]}...")

            resultado = gerar_resposta_ia(
                pergunta,
                df_base,
                st.session_state.llm,
                limite_similaridade,
                num_resultados_similares
            )

            df_upload.at[idx, 'Resposta'] = resultado['resposta']

            resultados.append({
                'pergunta': pergunta,
                'resposta': resultado['resposta'],
                'confianca': resultado['confianca'],
                'tipo': resultado['tipo'],
                'fonte': resultado['fonte']
            })

            progress_bar.progress((idx + 1) / total_linhas)

        st.session_state.resultados_preenchimento = {
            'dataframe': df_upload,
            'detalhes': resultados
        }

        status_text.text("✅ Processamento concluído!")

        # Mostrar estatísticas
        st.subheader("📊 Estatísticas do Processamento")

        tipos_resposta = {}
        for r in resultados:
            tipo = r['tipo']
            tipos_resposta[tipo] = tipos_resposta.get(tipo, 0) + 1

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            st.metric("Total Processado", len(resultados))

        with col_stat2:
            st.metric("Matches Diretos", tipos_resposta.get('match_direto', 0))

        with col_stat3:
            st.metric("IA Baseada", tipos_resposta.get('ia_baseada', 0))

        with col_stat4:
            st.metric("Não Encontrado", tipos_resposta.get('nao_encontrado', 0))

# Seção de resultados
if st.session_state.resultados_preenchimento is not None:
    st.header("📋 4. Resultados e Download")

    df_resultado = st.session_state.resultados_preenchimento['dataframe']
    detalhes = st.session_state.resultados_preenchimento['detalhes']

    # Tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["📊 Resultado Final", "🔍 Detalhes do Processamento", "📈 Análise de Qualidade"])

    with tab1:
        st.subheader("Arquivo Preenchido")
        st.dataframe(df_resultado, use_container_width=True)

        # Download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_resultado.to_excel(writer, sheet_name='Perguntas_Respostas', index=False)

        st.download_button(
            label="📥 Download Arquivo Preenchido (Excel)",
            data=buffer.getvalue(),
            file_name=f"arquivo_preenchido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with tab2:
        st.subheader("Detalhes do Processamento")

        for detalhe in detalhes:
            with st.expander(f"📝 {detalhe['pergunta'][:60]}... (Confiança: {detalhe['confianca']}%)"):
                st.write(f"**Pergunta:** {detalhe['pergunta']}")
                st.write(f"**Resposta:** {detalhe['resposta']}")
                st.write(f"**Tipo de Match:** {detalhe['tipo']}")
                st.write(f"**Fonte:** {detalhe['fonte']}")
                st.write(f"**Confiança:** {detalhe['confianca']}%")

    with tab3:
        st.subheader("Análise de Qualidade das Respostas")

        # Gráfico de distribuição de confiança
        confiances = [d['confianca'] for d in detalhes]

        if confiances:
            import plotly.express as px

            fig = px.histogram(
                x=confiances,
                nbins=20,
                title="Distribuição de Confiança das Respostas",
                labels={'x': 'Confiança (%)', 'y': 'Quantidade'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Métricas de qualidade
            col_qual1, col_qual2, col_qual3 = st.columns(3)

            with col_qual1:
                st.metric("Confiança Média", f"{np.mean(confiances):.1f}%")

            with col_qual2:
                alta_confianca = len([c for c in confiances if c >= 80])
                st.metric("Alta Confiança (≥80%)", f"{alta_confianca}/{len(confiances)}")

            with col_qual3:
                baixa_confianca = len([c for c in confiances if c < 50])
                st.metric("Baixa Confiança (<50%)", f"{baixa_confianca}/{len(confiances)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🤖 Sistema de Preenchimento Automático Q&A | Desenvolvido com Streamlit & LangChain
</div>
""", unsafe_allow_html=True)