import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json

st.set_page_config(page_title="Modelo de Scoring de Ativos", layout="wide")

st.title("Modelo de Analise e Scoring de Ativos")
st.markdown("**Camada 1:** Setup HiLo | **Camada 2:** Candle + Volume | **Camada 3:** Contexto | **Fundamentalista**")
st.write("---")

def safe_float(value, default=0.0):
    try:
        v = float(value)
        return v if v == v else default
    except:
        return default

@st.cache_data(ttl=3600)
def carregar_dados(ticker):
    if "." not in ticker:
        ticker_busca = f"{ticker}.SA"
    else:
        ticker_busca = ticker
    acao = yf.Ticker(ticker_busca)
    df = acao.history(period="1y")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    try:
        info = acao.info
    except:
        info = {}
    return df, info

def calcular_hilo(df, periodos=6):
    mm_min = df['Low'].rolling(periodos).mean()
    mm_max = df['High'].rolling(periodos).mean()
    hilo = [np.nan] * len(df)
    modos = [np.nan] * len(df)
    modo = None
    for i in range(periodos, len(df)):
        close = df['Close'].iloc[i]
        mn = mm_min.iloc[i]
        mx = mm_max.iloc[i]
        if modo is None:
            modo = 'compra' if close > mn else 'venda'
        if modo == 'compra':
            if close < mn:
                modo = 'venda'
                hilo[i] = mx
            else:
                hilo[i] = mn
        else:
            if close > mx:
                modo = 'compra'
                hilo[i] = mn
            else:
                hilo[i] = mx
        modos[i] = modo
    df = df.copy()
    df['HiLo'] = hilo
    df['HiLo_modo'] = modos
    return df

def calcular_indicadores(df):
    df = df.copy()
    df['MM6']  = df['Close'].rolling(6).mean()
    df['MM21'] = df['Close'].rolling(21).mean()
    df['MM72'] = df['Close'].rolling(72).mean()
    df = calcular_hilo(df, periodos=6)
    df['Donchian_Max'] = df['High'].rolling(20).max()
    df['Donchian_Min'] = df['Low'].rolling(20).min()
    df['Vol_MM21'] = df['Volume'].rolling(21).mean()
    df['Corpo']    = abs(df['Close'] - df['Open'])
    df['Amplitude'] = df['High'] - df['Low']
    df['Corpo_Pct'] = np.where(df['Amplitude'] > 0, df['Corpo'] / df['Amplitude'], 0)
    df['Fech_Pos']  = np.where(df['Amplitude'] > 0, (df['Close'] - df['Low']) / df['Amplitude'], 0.5)
    return df

def detectar_reversao(df_v):
    if len(df_v) < 3:
        return None
    c_ant  = df_v.iloc[-3]
    c_rom  = df_v.iloc[-2]
    c_conf = df_v.iloc[-1]
    close_rom  = safe_float(c_rom['Close'])
    open_rom   = safe_float(c_rom['Open'])
    close_conf = safe_float(c_conf['Close'])
    mm6_rom    = safe_float(c_rom['MM6'])
    mm6_conf   = safe_float(c_conf['MM6'])
    hilo_ant   = safe_float(c_ant['HiLo'])
    if mm6_rom == 0 or hilo_ant == 0:
        return None
    cruzou_cima = open_rom < mm6_rom and close_rom > mm6_rom
    if cruzou_cima and close_rom > hilo_ant and close_conf > mm6_conf:
        return "COMPRA"
    cruzou_baixo = open_rom > mm6_rom and close_rom < mm6_rom
    if cruzou_baixo and close_rom < hilo_ant and close_conf < mm6_conf:
        return "VENDA"
    return None

@st.cache_data(ttl=3600)
def buscar_fundamentals_claude(ticker, api_key):
    try:
        prompt = (
            "Voce e um analista financeiro brasileiro. "
            "Retorne APENAS um JSON com os dados fundamentalistas "
            "mais recentes do ativo " + ticker + " listado na B3.\n"
            "Formato: {\"pl_ratio\": 12.5, \"roe_pct\": 18.3, \"pvp\": 1.4, \"disponivel\": true}\n"
            "Se nao tiver dados: {\"pl_ratio\": 0, \"roe_pct\": 0, \"pvp\": 0, \"disponivel\": false}\n"
            "Retorne APENAS o JSON."
        )
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 200,
                  "system": "Responda SOMENTE com JSON valido, sem texto adicional.",
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=15
        )
        if response.status_code == 200:
            txt = response.json()["content"][0]["text"]
            data = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
            if data.get("disponivel"):
                return data
        return None
    except:
        return None

with st.sidebar:
    st.header("Parametros")
    ativo_input = st.text_input("Ticker do Ativo:", value="JHSF3").upper()
    st.write("---")
    st.caption("API Key Anthropic (opcional)")
    api_key_input = st.text_input("Chave para fundamentalista via IA:", type="password")
    st.button("Executar Analise", use_container_width=True)

if ativo_input:
    try:
        with st.spinner(f"Coletando dados de {ativo_input}..."):
            df, info = carregar_dados(ativo_input)

        if df.empty:
            st.error("Nenhum dado encontrado. Verifique o ticker.")
        else:
            df = calcular_indicadores(df)
            df_v = df.dropna(subset=['MM6', 'MM72', 'HiLo', 'Vol_MM21', 'Donchian_Max'])

            if len(df_v) < 6:
                st.error("Dados insuficientes.")
            else:
                a  = df_v.iloc[-1]
                a2 = df_v.iloc[-2]

                close  = safe_float(a['Close'])
                open_  = safe_float(a['Open'])
                hilo   = safe_float(a['HiLo'])
                mm6    = safe_float(a['MM6'])
                mm72   = safe_float(a['MM72'])
                vol    = safe_float(a['Volume'])
                vol_mm = safe_float(a['Vol_MM21'])
                corpo_pct = safe_float(a['Corpo_Pct'])
                fech_pos  = safe_float(a['Fech_Pos'])
                modo_compra = a['HiLo_modo'] == 'compra'

                ultimos5 = df_v['Close'].tail(5)
                faixa5 = (ultimos5.max() - ultimos5.min()) / ultimos5.mean() if ultimos5.mean() > 0 else 0
                dist_mm72_pct = abs(close - mm72) / mm72 * 100 if mm72 > 0 else 0
                dist_hilo_pct = abs(close - hilo) / close * 100 if close > 0 else 0

                reversao = detectar_reversao(df_v)

                # CAMADA 1
                score_c1 = 0
                motivos_c1 = []

                if close > hilo:
                    score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA do HiLo - altista", "+10"))
                else:
                    motivos_c1.append(("X", "Preco ABAIXO do HiLo - baixista", "0"))

                if close > mm6:
                    score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA da MM6 - momentum positivo", "+10"))
                else:
                    motivos_c1.append(("X", "Preco ABAIXO da MM6 - momentum negativo", "0"))

                if modo_compra and hilo < mm6:
                    score_c1 += 10; motivos_c1.append(("V", "HiLo abaixo da MM6 - suporte confirmado", "+10"))
                elif not modo_compra and hilo > mm6:
                    score_c1 += 10; motivos_c1.append(("V", "HiLo acima da MM6 - resistencia confirmada", "+10"))
                else:
                    motivos_c1.append(("X", "HiLo e MM6 sem confirmacao direcional", "0"))

                if modo_compra and mm6 > mm72:
                    score_c1 += 10; motivos_c1.append(("V", "MM6 ACIMA da MM72 - cruzamento altista", "+10"))
                elif not modo_compra and mm6 < mm72:
                    score_c1 += 10; motivos_c1.append(("V", "MM6 ABAIXO da MM72 - cruzamento baixista", "+10"))
                else:
                    motivos_c1.append(("X", "Medias sem cruzamento direcional confirmado", "0"))

                if close > mm72:
                    score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA da MM72 - medio prazo altista", "+10"))
                else:
                    motivos_c1.append(("X", "Preco ABAIXO da MM72 - medio prazo baixista", "0"))

                # CAMADA 2
                bonus_c2 = 0.0
                motivos_c2 = []
                candle_alta = close > open_

                if (modo_compra and candle_alta) or (not modo_compra and not candle_alta):
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Candle alinhado com a tendencia", "+0.5"))
                else:
                    motivos_c2.append(("o", "Candle contra a tendencia", "0"))

                if corpo_pct >= 0.50:
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Corpo forte: {corpo_pct*100:.0f}% da amplitude", "+0.5"))
                else:
                    motivos_c2.append(("o", f"Corpo fraco: {corpo_pct*100:.0f}% da amplitude", "0"))

                if modo_compra and fech_pos >= 0.70:
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Fechamento proximo da maxima ({fech_pos*100:.0f}%)", "+0.5"))
                elif not modo_compra and fech_pos <= 0.30:
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Fechamento proximo da minima ({fech_pos*100:.0f}%)", "+0.5"))
                else:
                    motivos_c2.append(("o", f"Fechamento no meio da amplitude ({fech_pos*100:.0f}%)", "0"))

                if vol_mm > 0 and vol > vol_mm:
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Volume acima da MM21 ({vol/vol_mm:.1f}x)", "+0.5"))
                else:
                    motivos_c2.append(("o", "Volume abaixo da MM21", "0"))

                if vol_mm > 0 and vol > vol_mm * 1.5:
                    bonus_c2 += 0.5; motivos_c2.append(("V", f"Volume forte: {vol/vol_mm:.1f}x a media", "+0.5"))
                else:
                    motivos_c2.append(("o", "Volume sem confirmacao forte (< 1.5x)", "0"))

                # CAMADA 3
                alertas_c3 = []
                if faixa5 <= 0.05:
                    rom = "Rompeu para cima" if close > ultimos5.iloc[0] else ("Rompeu para baixo" if close < ultimos5.iloc[0] else "Ainda lateral")
                    alertas_c3.append(("alerta", f"Fase: CONSOLIDACAO - {rom}"))
                elif close > mm6 > mm72:
                    alertas_c3.append(("ok", "Fase: TENDENCIA DE ALTA"))
                elif close < mm6 < mm72:
                    alertas_c3.append(("ok", "Fase: TENDENCIA DE BAIXA"))
                else:
                    alertas_c3.append(("alerta", "Fase: POSSIVEL REVERSAO - medias desalinhadas"))

                alertas_c3.append(("info", f"Distancia da MM72: {dist_mm72_pct:.1f}% {'(esticado)' if dist_mm72_pct > 15 else '(normal)'}"))

                if faixa5 <= 0.05:
                    alertas_c3.append(("alerta", f"Lateral detectada: faixa 5 pregoes = {faixa5*100:.1f}%"))
                else:
                    alertas_c3.append(("info", f"Faixa 5 pregoes = {faixa5*100:.1f}% - sem consolidacao"))

                if dist_hilo_pct < 2.0:
                    alertas_c3.append(("urgente", f"ALERTA: HiLo a {dist_hilo_pct:.1f}% - virada iminente!"))
                else:
                    alertas_c3.append(("info", f"HiLo a {dist_hilo_pct:.1f}% - sem iminencia de virada"))

                # FUNDAMENTALISTA
                score_fund = 0
                motivos_fund = []
                pe_ratio = roe = pb_ratio = 0.0
                fonte_fund = ""

                if api_key_input:
                    with st.spinner("Buscando fundamentalista via IA..."):
                        dados_claude = buscar_fundamentals_claude(ativo_input, api_key_input)
                    if dados_claude:
                        pe_ratio = safe_float(dados_claude.get("pl_ratio", 0))
                        roe      = safe_float(dados_claude.get("roe_pct", 0)) / 100
                        pb_ratio = safe_float(dados_claude.get("pvp", 0))
                        fonte_fund = "Claude AI"

                if not fonte_fund:
                    pe_ratio = safe_float(info.get("trailingPE") or info.get("forwardPE") or 0)
                    roe      = safe_float(info.get("returnOnEquity") or 0)
                    pb_ratio = safe_float(info.get("priceToBook") or 0)
                    fonte_fund = "yfinance" if (pe_ratio > 0 or roe > 0 or pb_ratio > 0) else "Indisponivel"

                if 0 < pe_ratio < 20:
                    score_fund += 15; motivos_fund.append(("V", f"P/L atrativo: {pe_ratio:.2f}x", "+15"))
                elif pe_ratio >= 20:
                    motivos_fund.append(("X", f"P/L esticado: {pe_ratio:.2f}x", "0"))
                else:
                    motivos_fund.append(("!", "P/L indisponivel", "0"))

                if roe > 0.10:
                    score_fund += 15; motivos_fund.append(("V", f"ROE forte: {roe*100:.1f}%", "+15"))
                elif roe > 0:
                    motivos_fund.append(("X", f"ROE fraco: {roe*100:.1f}%", "0"))
                else:
                    motivos_fund.append(("!", "ROE indisponivel", "0"))

                if 0 < pb_ratio < 3:
                    score_fund += 10; motivos_fund.append(("V", f"P/VP justo: {pb_ratio:.2f}x", "+10"))
                elif pb_ratio >= 3:
                    motivos_fund.append(("X", f"P/VP elevado: {pb_ratio:.2f}x", "0"))
                else:
                    motivos_fund.append(("!", "P/VP indisponivel", "0"))

                # SCORE FINAL
                score_tec_norm = round(score_c1 * (60 / 50))
                score_bonus    = round(bonus_c2 * 2)
                score_final    = min(score_tec_norm + score_fund + score_bonus, 100)

                if score_final >= 75:
                    status, cor, alert_type = "COMPRA", "#22c55e", "success"
                elif score_final >= 45:
                    status, cor, alert_type = "NEUTRA", "#eab308", "warning"
                else:
                    status, cor, alert_type = "VENDA", "#ef4444", "error"

                # PAINEL
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    st.metric("Cotacao Atual", f"R$ {close:.2f}", f"{(close - safe_float(a2['Close'])):+.2f} R$")
                    st.markdown(f"### Status: <span style='color:{cor}'>{status}</span>", unsafe_allow_html=True)
                    st.progress(min(score_final / 100.0, 1.0))
                    st.write(f"**Pontuacao Total: {score_final}/100**")
                    st.caption(f"HiLo modo: {'Compra' if modo_compra else 'Venda'}")

                with col2:
                    st.subheader("Resumo dos Pesos")
                    st.write(f"Setup HiLo (C1): {score_tec_norm}/60 pts")
                    st.write(f"Bonus Candle+Vol (C2): +{score_bonus} pts")
                    st.write(f"Fundamentos: {score_fund}/40 pts")
                    st.write("---")
                    st.caption(f"Fonte: {fonte_fund}")
                    st.caption(f"P/L: {pe_ratio:.2f}x" if pe_ratio > 0 else "P/L: --")
                    st.caption(f"ROE: {roe*100:.1f}%" if roe > 0 else "ROE: --")
                    st.caption(f"P/VP: {pb_ratio:.2f}x" if pb_ratio > 0 else "P/VP: --")

                with col3:
                    if alert_type == "success":
                        st.success("COMPRA: Setup HiLo alinhado com momentum e fundamentos.")
                    elif alert_type == "warning":
                        st.warning("NEUTRO: Sinais mistos - aguardar confirmacao.")
                    else:
                        st.error("VENDA/EVITAR: Estrutura baixista ou multiplos fracos.")

                    if reversao == "COMPRA":
                        st.success("Padrao de reversao confirmado - COMPRA")
                    elif reversao == "VENDA":
                        st.error("Padrao de reversao confirmado - VENDA")

                    for tipo, txt in alertas_c3:
                        if tipo == "urgente":
                            st.warning(txt)
                        elif tipo == "alerta":
                            st.warning(txt)
                        else:
                            st.info(txt)

                st.write("---")

                # GRAFICO
                st.subheader("Visualizacao Grafica")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_v.index, open=df_v['Open'], high=df_v['High'],
                    low=df_v['Low'], close=df_v['Close'], name='Preco'
                ))
                hilo_compra = df_v['HiLo'].where(df_v['HiLo_modo'] == 'compra')
                hilo_venda  = df_v['HiLo'].where(df_v['HiLo_modo'] == 'venda')
                fig.add_trace(go.Scatter(x=df_v.index, y=hilo_compra,
                    line=dict(color='#00e676', width=2), name='HiLo Compra', connectgaps=False))
                fig.add_trace(go.Scatter(x=df_v.index, y=hilo_venda,
                    line=dict(color='#ff5252', width=2), name='HiLo Venda', connectgaps=False))
                fig.add_trace(go.Scatter(x=df_v.index, y=df_v['MM6'],
                    line=dict(color='#ffca28', width=1.5), name='MM6'))
                fig.add_trace(go.Scatter(x=df_v.index, y=df_v['MM72'],
                    line=dict(color='#00d4ff', width=2), name='MM72'))
                fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Donchian_Max'],
                    line=dict(color='rgba(130,130,130,0.4)', width=1, dash='dash'), name='Donchian Max'))
                fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Donchian_Min'],
                    line=dict(color='rgba(130,130,130,0.4)', width=1, dash='dash'), name='Donchian Min'))
                fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_rangeslider_visible=False, template="plotly_dark",
                    legend=dict(orientation="h", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                # DETALHAMENTO
                st.subheader("Detalhamento Completo")
                col_c1, col_c2, col_c3, col_fund = st.columns(4)

                with col_c1:
                    st.markdown("**Camada 1 - Setup HiLo**")
                    for tp, txt, pts in motivos_c1:
                        st.write(f"{tp} {txt} [{pts}]")

                with col_c2:
                    st.markdown("**Camada 2 - Candle + Volume**")
                    for tp, txt, pts in motivos_c2:
                        st.write(f"{tp} {txt} [{pts}]")

                with col_c3:
                    st.markdown("**Camada 3 - Contexto**")
                    for tipo, txt in alertas_c3:
                        st.write(f"- {txt}")
                    if reversao:
                        st.write(f"- Padrao de reversao: {reversao}")

                with col_fund:
                    st.markdown(f"**Fundamentalista** ({fonte_fund})")
                    for tp, txt, pts in motivos_fund:
                        st.write(f"{tp} {txt} [{pts}]")

    except Exception as e:
        st.error(f"Erro: {e}")
        st.exception(e)
