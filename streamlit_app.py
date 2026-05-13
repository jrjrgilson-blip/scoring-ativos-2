import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json

st.set_page_config(page_title="Modelo de Scoring de Ativos", layout="wide")

# -------------------------------------------------------------------------
# FUNCOES AUXILIARES
# -------------------------------------------------------------------------
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
    df['Corpo']     = abs(df['Close'] - df['Open'])
    df['Amplitude'] = df['High'] - df['Low']
    df['Corpo_Pct'] = np.where(df['Amplitude'] > 0, df['Corpo'] / df['Amplitude'], 0)
    df['Fech_Pos']  = np.where(df['Amplitude'] > 0, (df['Close'] - df['Low']) / df['Amplitude'], 0.5)
    return df

def detectar_reversao(df_v, janela=60):
    if len(df_v) < 3:
        return None
    df_scan = df_v.tail(janela + 2)
    datas = df_scan.index.tolist()
    valores = df_scan.reset_index(drop=True)
    ultimo_padrao = None
    for i in range(1, len(valores) - 1):
        c_ant  = valores.iloc[i - 1]
        c_rom  = valores.iloc[i]
        c_conf = valores.iloc[i + 1]
        close_rom  = safe_float(c_rom["Close"])
        open_rom   = safe_float(c_rom["Open"])
        close_conf = safe_float(c_conf["Close"])
        mm6_rom    = safe_float(c_rom["MM6"])
        mm6_conf   = safe_float(c_conf["MM6"])
        hilo_ant   = safe_float(c_ant["HiLo"])
        if mm6_rom == 0 or hilo_ant == 0:
            continue
        data_rom = datas[i]
        if hasattr(data_rom, "date"):
            data_rom = data_rom.date()
        # Candle anterior ao rompimento (contexto de onde veio)
        close_ant     = safe_float(c_ant["Close"])
        mm6_ant       = safe_float(c_ant["MM6"])
        hilo_modo_ant = c_ant["HiLo_modo"]  # modo ANTES do rompimento

        # COMPRA: vinha de baixo (anterior abaixo da MM6),
        # HiLo ainda em modo venda no candle anterior (virada real),
        # cruza MM6 para cima, fecha acima do HiLo anterior,
        # confirmacao fecha acima da MM6
        cruzou_cima    = open_rom < mm6_rom and close_rom > mm6_rom
        vinha_de_baixo = close_ant < mm6_ant
        # Aceita fechamento acima ou encostando no HiLo (tolerancia de 0.5%)
        tocou_hilo_compra = close_rom >= hilo_ant * 0.990
        if (cruzou_cima and vinha_de_baixo and
                hilo_modo_ant == "venda" and
                tocou_hilo_compra and close_conf > mm6_conf):
            ultimo_padrao = {"tipo": "COMPRA", "data": data_rom}

        # VENDA: vinha de cima (anterior acima da MM6),
        # HiLo ainda em modo compra no candle anterior (virada real),
        # cruza MM6 para baixo, fecha abaixo do HiLo anterior,
        # confirmacao fecha abaixo da MM6
        cruzou_baixo  = open_rom > mm6_rom and close_rom < mm6_rom
        vinha_de_cima = close_ant > mm6_ant
        # Aceita fechamento abaixo ou encostando no HiLo (tolerancia de 0.5%)
        tocou_hilo_venda = close_rom <= hilo_ant * 1.010
        if (cruzou_baixo and vinha_de_cima and
                hilo_modo_ant == "compra" and
                tocou_hilo_venda and close_conf < mm6_conf):
            ultimo_padrao = {"tipo": "VENDA", "data": data_rom}
    return ultimo_padrao

def calcular_score_completo(df_v):
    """Calcula todas as camadas e retorna dicionario com resultados."""
    if len(df_v) < 6:
        return None

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
        score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA do HiLo", "+10"))
    else:
        motivos_c1.append(("X", "Preco ABAIXO do HiLo", "0"))
    if close > mm6:
        score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA da MM6", "+10"))
    else:
        motivos_c1.append(("X", "Preco ABAIXO da MM6", "0"))
    if modo_compra and hilo < mm6:
        score_c1 += 10; motivos_c1.append(("V", "HiLo abaixo da MM6 - suporte", "+10"))
    elif not modo_compra and hilo > mm6:
        score_c1 += 10; motivos_c1.append(("V", "HiLo acima da MM6 - resistencia", "+10"))
    else:
        motivos_c1.append(("X", "HiLo e MM6 sem confirmacao", "0"))
    if modo_compra and mm6 > mm72:
        score_c1 += 10; motivos_c1.append(("V", "MM6 ACIMA da MM72 - altista", "+10"))
    elif not modo_compra and mm6 < mm72:
        score_c1 += 10; motivos_c1.append(("V", "MM6 ABAIXO da MM72 - baixista", "+10"))
    else:
        motivos_c1.append(("X", "Medias sem cruzamento confirmado", "0"))
    if close > mm72:
        score_c1 += 10; motivos_c1.append(("V", "Preco ACIMA da MM72", "+10"))
    else:
        motivos_c1.append(("X", "Preco ABAIXO da MM72", "0"))

    # CAMADA 2
    bonus_c2 = 0.0
    motivos_c2 = []
    candle_alta = close > open_
    if (modo_compra and candle_alta) or (not modo_compra and not candle_alta):
        bonus_c2 += 0.5; motivos_c2.append(("V", "Candle alinhado com tendencia", "+0.5"))
    else:
        motivos_c2.append(("o", "Candle contra a tendencia", "0"))
    if corpo_pct >= 0.50:
        bonus_c2 += 0.5; motivos_c2.append(("V", f"Corpo forte: {corpo_pct*100:.0f}%", "+0.5"))
    else:
        motivos_c2.append(("o", f"Corpo fraco: {corpo_pct*100:.0f}%", "0"))
    if modo_compra and fech_pos >= 0.70:
        bonus_c2 += 0.5; motivos_c2.append(("V", f"Fechamento proximo maxima ({fech_pos*100:.0f}%)", "+0.5"))
    elif not modo_compra and fech_pos <= 0.30:
        bonus_c2 += 0.5; motivos_c2.append(("V", f"Fechamento proximo minima ({fech_pos*100:.0f}%)", "+0.5"))
    else:
        motivos_c2.append(("o", f"Fechamento no meio ({fech_pos*100:.0f}%)", "0"))
    if vol_mm > 0 and vol > vol_mm:
        bonus_c2 += 0.5; motivos_c2.append(("V", f"Volume acima MM21 ({vol/vol_mm:.1f}x)", "+0.5"))
    else:
        motivos_c2.append(("o", "Volume abaixo MM21", "0"))
    if vol_mm > 0 and vol > vol_mm * 1.5:
        bonus_c2 += 0.5; motivos_c2.append(("V", f"Volume forte: {vol/vol_mm:.1f}x", "+0.5"))
    else:
        motivos_c2.append(("o", "Volume sem confirmacao forte", "0"))

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
        alertas_c3.append(("alerta", "Fase: POSSIVEL REVERSAO"))
    alertas_c3.append(("info", f"Distancia da MM72: {dist_mm72_pct:.1f}% {'(esticado)' if dist_mm72_pct > 15 else '(normal)'}"))
    if faixa5 <= 0.05:
        alertas_c3.append(("alerta", f"Lateral detectada: faixa 5 pregoes = {faixa5*100:.1f}%"))
    else:
        alertas_c3.append(("info", f"Faixa 5 pregoes = {faixa5*100:.1f}%"))
    if dist_hilo_pct < 2.0:
        alertas_c3.append(("urgente", f"ALERTA: HiLo a {dist_hilo_pct:.1f}% - virada iminente!"))
    else:
        alertas_c3.append(("info", f"HiLo a {dist_hilo_pct:.1f}% do preco"))

    score_tec_norm = round(score_c1 * (60 / 50))
    score_bonus    = round(bonus_c2 * 2)

    return {
        "close": close,
        "close_ant": safe_float(a2['Close']),
        "modo_compra": modo_compra,
        "score_c1": score_c1,
        "score_tec_norm": score_tec_norm,
        "score_bonus": score_bonus,
        "motivos_c1": motivos_c1,
        "motivos_c2": motivos_c2,
        "alertas_c3": alertas_c3,
        "reversao": reversao,
        "dist_hilo_pct": dist_hilo_pct,
        "faixa5": faixa5,
        "df_v": df_v,
    }

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

def calcular_fund(ticker, info, api_key_input):
    score_fund = 0
    motivos_fund = []
    pe_ratio = roe = pb_ratio = 0.0
    fonte_fund = ""

    if api_key_input:
        dados_claude = buscar_fundamentals_claude(ticker, api_key_input)
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

    return score_fund, motivos_fund, fonte_fund, pe_ratio, roe, pb_ratio

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Parametros")
    api_key_input = st.text_input("API Key Anthropic (opcional):", type="password",
                                   help="console.anthropic.com")

# -------------------------------------------------------------------------
# ABAS
# -------------------------------------------------------------------------
aba1, aba2 = st.tabs(["Analise Individual", "Ranking Multi-Ativo"])

# =========================================================================
# ABA 1: ANALISE INDIVIDUAL
# =========================================================================
with aba1:
    st.title("Analise Individual")
    st.markdown("**Camada 1:** Setup HiLo | **Camada 2:** Candle + Volume | **Camada 3:** Contexto | **Fundamentalista**")
    st.write("---")

    col_inp, col_btn = st.columns([3, 1])
    with col_inp:
        ativo_input = st.text_input("Ticker do Ativo:", value="JHSF3", key="ticker_individual").upper()
    with col_btn:
        st.write("")
        st.write("")
        analisar_btn = st.button("Analisar", use_container_width=True, key="btn_individual")

    if ativo_input:
        try:
            with st.spinner(f"Coletando dados de {ativo_input}..."):
                df, info = carregar_dados(ativo_input)

            if df.empty:
                st.error("Nenhum dado encontrado. Verifique o ticker.")
            else:
                df = calcular_indicadores(df)
                df_v = df.dropna(subset=['MM6', 'MM72', 'HiLo', 'Vol_MM21', 'Donchian_Max'])
                res = calcular_score_completo(df_v)

                if res is None:
                    st.error("Dados insuficientes.")
                else:
                    score_fund, motivos_fund, fonte_fund, pe_ratio, roe, pb_ratio = calcular_fund(
                        ativo_input, info, api_key_input)

                    score_final = min(res['score_tec_norm'] + res['score_bonus'] + score_fund, 100)

                    if score_final >= 75:
                        status, cor, alert_type = "COMPRA", "#22c55e", "success"
                    elif score_final >= 45:
                        status, cor, alert_type = "NEUTRA", "#eab308", "warning"
                    else:
                        status, cor, alert_type = "VENDA", "#ef4444", "error"

                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Cotacao Atual", f"R$ {res['close']:.2f}",
                                  f"{(res['close'] - res['close_ant']):+.2f} R$")
                        st.markdown(f"### Status: <span style='color:{cor}'>{status}</span>",
                                    unsafe_allow_html=True)
                        st.progress(min(score_final / 100.0, 1.0))
                        st.write(f"**Pontuacao Total: {score_final}/100**")
                        st.caption(f"HiLo modo: {'Compra' if res['modo_compra'] else 'Venda'}")

                    with col2:
                        st.subheader("Resumo dos Pesos")
                        st.write(f"Setup HiLo (C1): {res['score_tec_norm']}/60 pts")
                        st.write(f"Bonus Candle+Vol (C2): +{res['score_bonus']} pts")
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

                        rev = res["reversao"]
                        if rev:
                            data_rev = rev["data"]
                            if rev["tipo"] == "COMPRA":
                                st.success(f"Ultimo padrao de reversao: COMPRA em {data_rev}")
                            else:
                                st.error(f"Ultimo padrao de reversao: VENDA em {data_rev}")

                        for tipo, txt in res['alertas_c3']:
                            if tipo in ("urgente", "alerta"):
                                st.warning(txt)
                            else:
                                st.info(txt)

                    st.write("---")

                    # GRAFICO
                    st.subheader("Visualizacao Grafica")
                    df_v = res['df_v']
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df_v.index, open=df_v['Open'], high=df_v['High'],
                        low=df_v['Low'], close=df_v['Close'], name='Preco'))
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
                        for tp, txt, pts in res['motivos_c1']:
                            st.write(f"{tp} {txt} [{pts}]")
                    with col_c2:
                        st.markdown("**Camada 2 - Candle + Volume**")
                        for tp, txt, pts in res['motivos_c2']:
                            st.write(f"{tp} {txt} [{pts}]")
                    with col_c3:
                        st.markdown("**Camada 3 - Contexto**")
                        for tipo, txt in res['alertas_c3']:
                            st.write(f"- {txt}")
                        if res["reversao"]:
                            rv = res["reversao"]
                            st.write(f"- Ultimo padrao reversao: {rv["tipo"]} em {rv["data"]}")
                    with col_fund:
                        st.markdown(f"**Fundamentalista** ({fonte_fund})")
                        for tp, txt, pts in motivos_fund:
                            st.write(f"{tp} {txt} [{pts}]")

        except Exception as e:
            st.error(f"Erro: {e}")
            st.exception(e)

# =========================================================================
# ABA 2: RANKING MULTI-ATIVO
# =========================================================================
with aba2:
    st.title("Ranking Multi-Ativo")
    st.markdown("Analise varios ativos de uma vez e compare pelo score.")
    st.write("---")

    tickers_input = st.text_area(
        "Digite os tickers separados por virgula ou espaco:",
        value="PETR4, VALE3, ITUB4, BBAS3, WEGE3, JHSF3",
        height=80,
        help="Ex: PETR4, VALE3, ITUB4"
    )
    analisar_multi = st.button("Analisar Todos", use_container_width=True, key="btn_multi")

    if analisar_multi and tickers_input:
        tickers_lista = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]

        if not tickers_lista:
            st.warning("Digite ao menos um ticker.")
        else:
            resultados = []
            erros = []

            prog = st.progress(0)
            status_txt = st.empty()

            for i, ticker in enumerate(tickers_lista):
                status_txt.text(f"Analisando {ticker}... ({i+1}/{len(tickers_lista)})")
                try:
                    df, info = carregar_dados(ticker)
                    if df.empty:
                        erros.append(ticker)
                        continue
                    df = calcular_indicadores(df)
                    df_v = df.dropna(subset=['MM6', 'MM72', 'HiLo', 'Vol_MM21', 'Donchian_Max'])
                    res = calcular_score_completo(df_v)
                    if res is None:
                        erros.append(ticker)
                        continue

                    score_fund, _, _, _, _, _ = calcular_fund(ticker, info, api_key_input)
                    score_final = min(res['score_tec_norm'] + res['score_bonus'] + score_fund, 100)

                    if score_final >= 75:
                        sinal = "COMPRA"
                    elif score_final >= 45:
                        sinal = "NEUTRA"
                    else:
                        sinal = "VENDA"

                    variacao = ((res['close'] - res['close_ant']) / res['close_ant'] * 100) if res['close_ant'] > 0 else 0

                    resultados.append({
                        "Ticker": ticker,
                        "Cotacao": res['close'],
                        "Var%": variacao,
                        "Score": score_final,
                        "C1 HiLo": res['score_tec_norm'],
                        "C2 Bonus": res['score_bonus'],
                        "Fundamentos": score_fund,
                        "Sinal": sinal,
                        "Modo HiLo": "Compra" if res['modo_compra'] else "Venda",
                        "Reversao": (res["reversao"]["tipo"] + " " + str(res["reversao"]["data"])) if res["reversao"] else "--",
                        "HiLo Dist%": round(res['dist_hilo_pct'], 1),
                        "Consolidacao": "Sim" if res['faixa5'] <= 0.05 else "Nao",
                    })
                except Exception:
                    erros.append(ticker)

                prog.progress((i + 1) / len(tickers_lista))

            status_txt.empty()
            prog.empty()

            if erros:
                st.warning(f"Nao foi possivel analisar: {', '.join(erros)}")

            if resultados:
                df_rank = pd.DataFrame(resultados).sort_values("Score", ascending=False).reset_index(drop=True)
                df_rank.index += 1  # ranking comeca em 1

                # CARDS DE DESTAQUE
                col_top = st.columns(min(3, len(df_rank)))
                for idx, row in df_rank.head(3).iterrows():
                    cor = "#22c55e" if row['Sinal'] == "COMPRA" else ("#eab308" if row['Sinal'] == "NEUTRA" else "#ef4444")
                    with col_top[idx-1]:
                        st.markdown(f"### #{idx} {row['Ticker']}")
                        st.markdown(f"<span style='color:{cor}; font-size:20px; font-weight:bold'>{row['Sinal']}</span>", unsafe_allow_html=True)
                        st.metric("Score", f"{row['Score']}/100")
                        if row["Reversao"] != "--":
                            rev_cor = "green" if row["Reversao"].startswith("COMPRA") else "red"
                            st.markdown(f"<span style='color:{rev_cor}'>Reversao: {row["Reversao"]}</span>", unsafe_allow_html=True)
                        if row['HiLo Dist%'] < 2.0:
                            st.warning(f"HiLo a {row['HiLo Dist%']}% - virada iminente!")

                st.write("---")

                # TABELA COMPLETA
                st.subheader("Tabela Completa (ordenada por Score)")

                def colorir_sinal(val):
                    if val == "COMPRA":
                        return "color: #22c55e; font-weight: bold"
                    elif val == "VENDA":
                        return "color: #ef4444; font-weight: bold"
                    return "color: #eab308; font-weight: bold"

                def colorir_reversao(val):
                    if val == "COMPRA":
                        return "color: #22c55e; font-weight: bold"
                    elif val == "VENDA":
                        return "color: #ef4444; font-weight: bold"
                    return ""

                df_display = df_rank.copy()
                df_display['Cotacao']    = df_display['Cotacao'].apply(lambda x: f"R$ {x:.2f}")
                df_display['Var%']       = df_display['Var%'].apply(lambda x: f"{x:+.2f}%")
                df_display['HiLo Dist%'] = df_display['HiLo Dist%'].apply(lambda x: f"{x:.1f}%")

                def colorir_score(val):
                    if val >= 75:
                        return "color: #22c55e; font-weight: bold"
                    elif val >= 45:
                        return "color: #eab308; font-weight: bold"
                    return "color: #ef4444; font-weight: bold"

                styled = df_display.style \
                    .map(colorir_sinal, subset=['Sinal']) \
                    .map(colorir_reversao, subset=['Reversao']) \
                    .map(colorir_score, subset=['Score'])

                st.dataframe(styled, use_container_width=True)

                # GRAFICO DE BARRAS DO RANKING
                st.subheader("Grafico do Ranking")
                cores = ["#22c55e" if s == "COMPRA" else ("#eab308" if s == "NEUTRA" else "#ef4444")
                         for s in df_rank['Sinal']]
                fig_rank = go.Figure(go.Bar(
                    x=df_rank['Ticker'],
                    y=df_rank['Score'],
                    marker_color=cores,
                    text=df_rank['Score'],
                    textposition='outside'
                ))
                fig_rank.update_layout(
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(range=[0, 110]),
                    showlegend=False
                )
                st.plotly_chart(fig_rank, use_container_width=True)

                # ALERTAS DE REVERSAO E IMINENCIA
                reversoes = df_rank[df_rank['Reversao'] != "--"]
                iminentes = df_rank[df_rank['HiLo Dist%'] < 2.0]

                if not reversoes.empty:
                    st.write("---")
                    st.subheader("Padroes de Reversao Detectados")
                    for _, row in reversoes.iterrows():
                        cor = "#22c55e" if row["Reversao"].startswith("COMPRA") else "#ef4444"
                        st.markdown(
                            f"<span style='color:{cor}; font-weight:bold'>"
                            f"{row["Ticker"]} — {row["Reversao"]}"
                            f"</span>",
                            unsafe_allow_html=True
                        )

                if not iminentes.empty:
                    st.write("---")
                    st.subheader("Alertas de Virada Iminente do HiLo")
                    for _, row in iminentes.iterrows():
                        st.warning(f"{row['Ticker']} — HiLo a {row['HiLo Dist%']}% do preco")
