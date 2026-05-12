
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------------------
st.set_page_config(page_title="Modelo de Scoring de Ativos", layout="wide")

st.title("📊 Modelo de Análise e Scoring de Ativos")
st.markdown("Integração de **Análise Técnica (60%)** e **Análise Fundamentalista (40%)** para geração de sinal operacional.")
st.write("---")

# -------------------------------------------------------------------------
# FUNÇÕES DE PROCESSAMENTO
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600) # Guarda os dados em cache por 1 hora para ficar rápido
def carregar_dados(ticker):
    # Adiciona o sufixo .SA automaticamente para ações brasileiras se não houver
    if not ticker.endswith(".SA") and not ticker.endswith(".US"):
        ticker_busca = f"{ticker}.SA"
    else:
        ticker_busca = ticker
        
    acao = yf.Ticker(ticker_busca)
    df = acao.history(period="1y")
    
    # Coleta de algumas métricas fundamentalistas básicas
    try:
        info = acao.info
    except:
        info = {}
        
    return df, info

def calcular_indicadores(df):
    # Canal de Donchian (20 períodos)
    df['Donchian_Max'] = df['High'].rolling(20).max()
    df['Donchian_Min'] = df['Low'].rolling(20).min()
    
    # Média Móvel Simples (21 períodos)
    df['SMA_21'] = df['Close'].rolling(21).mean()
    
    # Média de Volume (21 períodos)
    df['Vol_SMA_21'] = df['Volume'].rolling(21).mean()
    
    # Proxy simples de HiLo (Média de máximas e mínimas de 4 períodos)
    df['HiLo_High'] = df['High'].rolling(4).mean()
    df['HiLo_Low'] = df['Low'].rolling(4).mean()
    # Lógica HiLo: Se o fechamento for maior que a média das máximas, vira compra
    df['HiLo_Status'] = df['Close'] > df['HiLo_High'] 
    
    return df

# -------------------------------------------------------------------------
# INTERFACE DO USUÁRIO (BARRA LATERAL / INPUT)
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    ativo_input = st.text_input("Digite o Ticker do Ativo:", value="JHSF3").upper()
    botao_analisar = st.button("Executar Análise", use_container_width=True)

# -------------------------------------------------------------------------
# MOTOR DE EXECUÇÃO E SCORING
# -------------------------------------------------------------------------
if ativo_input:
    try:
        with st.spinner(f"Coletando dados de {ativo_input}..."):
            df, info = carregar_dados(ativo_input)
            
        if df.empty:
            st.error("Nenhum dado encontrado para este Ticker. Verifique se o código está correto.")
        else:
            df = calcular_indicadores(df)
            linha_atual = df.iloc[-1]
            linha_anterior = df.iloc[-2]
            
            # --- CÁLCULO DO SCORE TÉCNICO (Máx: 60 pts) ---
            score_tec = 0
            motivos_tec = []
            
            # 1. Preço vs Média (15 pts)
            if linha_atual['Close'] > linha_atual['SMA_21']:
                score_tec += 15
                motivos_tec.append("Preço acima da Média Móvel de 21 dias (+15)")
            
            # 2. Inclinação da Média (15 pts)
            if linha_atual['SMA_21'] > linha_anterior['SMA_21']:
                score_tec += 15
                motivos_tec.append("Média Móvel de 21 dias inclinada para cima (+15)")
                
            # 3. Proximidade/Rompimento do Donchian Superior (15 pts)
            # Considera positivo se estiver a pelo menos 3% do topo do canal
            if linha_atual['Close'] >= (linha_atual['Donchian_Max'] * 0.97):
                score_tec += 15
                motivos_tec.append("Preço testando/rompendo o topo do Canal de Donchian (+15)")
                
            # 4. Força do Volume (15 pts)
            if linha_atual['Volume'] > linha_atual['Vol_SMA_21']:
                score_tec += 15
                motivos_tec.append("Volume diário superior à média de 21 dias (+15)")

            # --- CÁLCULO DO SCORE FUNDAMENTALISTA (Máx: 40 pts) ---
            score_fund = 0
            motivos_fund = []
            
            # Coleta segura de chaves do yfinance
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0)
            
            # 1. P/L Saudável (15 pts)
            if pe_ratio > 0 and pe_ratio < 20:
                score_fund += 15
                motivos_fund.append(f"P/L atrativo: {pe_ratio:.2f}x (+15)")
            elif pe_ratio >= 20:
                motivos_fund.append(f"P/L esticado: {pe_ratio:.2f}x (0)")
            else:
                motivos_fund.append("P/L negativo ou indisponível (0)")
                
            # 2. ROE (15 pts)
            if roe and roe > 0.10: # ROE > 10%
                score_fund += 15
                motivos_fund.append(f"ROE forte: {(roe*100):.1f}% (+15)")
            else:
                motivos_fund.append("ROE abaixo de 10% ou indisponível (0)")
                
            # 3. P/VP (10 pts)
            if pb_ratio > 0 and pb_ratio < 3:
                score_fund += 10
                motivos_fund.append(f"P/VP justo: {pb_ratio:.2f}x (+10)")
            else:
                motivos_fund.append("P/VP elevado ou indisponível (0)")
                
            # --- RESULTADO FINAL ---
            score_final = score_tec + score_fund
            
            if score_final >= 75:
                status, cor, alert_type = "COMPRA", "#22c55e", "success"
            elif score_final >= 45:
                status, cor, alert_type = "NEUTRA", "#eab308", "warning"
            else:
                status, cor, alert_type = "VENDA", "#ef4444", "error"

            # -----------------------------------------------------------------
            # EXIBIÇÃO NO PAINEL PRINCIPAL
            # -----------------------------------------------------------------
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.metric("Cotação Atual", f"R$ {linha_atual['Close']:.2f}", 
                          f"{(linha_atual['Close'] - linha_anterior['Close']):.2f} R$")
                st.markdown(f"### Status Final: <span style='color:{cor}'>{status}</span>", unsafe_allow_html=True)
                st.progress(score_final / 100.0)
                st.write(f"**Pontuação Total: {score_final}/100**")
                
            with col2:
                st.subheader("Resumo dos Pesos")
                st.write(f"📈 **Técnico:** {score_tec}/60 pts")
                st.write(f"🏢 **Fundamentos:** {score_fund}/40 pts")
                
            with col3:
                # Alertas estilizados baseados na indicação
                if alert_type == "success":
                    st.success("🎯 **Gatilho de Entrada Otimizado:** O gráfico demonstra forte momentum aliado a múltiplos saudáveis de suporte.")
                elif alert_type == "warning":
                    st.warning("⚠️ **Atenção:** Sinais mistos. O ativo pode estar passando por correção ou os múltiplos exigem cautela.")
                else:
                    st.error("🛑 **Viés Negativo:** Indicadores em tendência de baixa e fraqueza nos múltiplos fundamentais.")

            st.write("---")
            
            # --- GRÁFICO INTERATIVO (PLOTLY) ---
            st.subheader("Visualização Gráfica (Preço + Canal de Donchian)")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Preço'
            ))
            
            # Canal de Donchian Superior e Inferior
            fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_Max'], line=dict(color='rgba(130, 130, 130, 0.5)', width=1, dash='dash'), name='Donchian Max'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_Min'], line=dict(color='rgba(130, 130, 130, 0.5)', width=1, dash='dash'), name='Donchian Min'))
            
            # Média de 21
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_21'], line=dict(color='orange', width=2), name='Média 21'))
            
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EXIBIÇÃO DETALHADA DOS CRITÉRIOS ---
            col_tec, col_fund = st.columns(2)
            with col_tec:
                st.markdown("#### Detalhamento Técnico")
                for m in motivos_tec:
                    st.write(f"✔️ {m}")
            with col_fund:
                st.markdown("#### Detalhamento Fundamentalista")
                for m in motivos_fund:
                    st.write(f"✔️ {m}")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o ativo: {e}")
