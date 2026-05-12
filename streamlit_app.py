import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json

st.set_page_config(page_title=“Modelo de Scoring de Ativos”, layout=“wide”)

st.title(“📊 Modelo de Análise e Scoring de Ativos”)
st.markdown(”**Camada 1:** Setup HiLo · **Camada 2:** Candle + Volume · **Camada 3:** Contexto e Fase · **Fundamentalista**”)
st.write(”—”)

# ———————————————————————––

# FUNÇÕES AUXILIARES

# ———————————————————————––

def safe_float(value, default=0.0):
try:
v = float(value)
return v if v == v else default
except:
return default

@st.cache_data(ttl=3600)
def carregar_dados(ticker):
if “.” not in ticker:
ticker_busca = f”{ticker}.SA”
else:
ticker_busca = ticker
acao = yf.Ticker(ticker_busca)
df = acao.history(period=“1y”)
if df.index.tz is not None:
df.index = df.index.tz_localize(None)
try:
info = acao.info
except:
info = {}
return df, info

def calcular_indicadores(df):
df = df.copy()

```
# Médias móveis
df['MM6']  = df['Close'].rolling(6).mean()
df['MM21'] = df['Close'].rolling(21).mean()
df['MM72'] = df['Close'].rolling(72).mean()

# HiLo (média das máximas e mínimas de 4 períodos)
df['HiLo_High'] = df['High'].rolling(4).mean()
df['HiLo_Low']  = df['Low'].rolling(4).mean()
# HiLo sinaliza compra se fechamento > média das máximas, venda se < média das mínimas
df['HiLo'] = np.where(df['Close'] > df['HiLo_High'], df['HiLo_High'],
             np.where(df['Close'] < df['HiLo_Low'],  df['HiLo_Low'], np.nan))
df['HiLo'] = df['HiLo'].ffill()

# Canal de Donchian (20 períodos)
df['Donchian_Max'] = df['High'].rolling(20).max()
df['Donchian_Min'] = df['Low'].rolling(20).min()

# Volume médio
df['Vol_MM21'] = df['Volume'].rolling(21).mean()

# Corpo e amplitude do candle
df['Corpo']     = abs(df['Close'] - df['Open'])
df['Amplitude'] = df['High'] - df['Low']
df['Corpo_Pct'] = np.where(df['Amplitude'] > 0, df['Corpo'] / df['Amplitude'], 0)

# Posição do fechamento na amplitude (0=mínima, 1=máxima)
df['Fech_Pos'] = np.where(df['Amplitude'] > 0,
                          (df['Close'] - df['Low']) / df['Amplitude'], 0.5)

return df
```

@st.cache_data(ttl=3600)
def buscar_fundamentals_claude(ticker, api_key):
try:
prompt = f””“Você é um analista financeiro brasileiro. Retorne APENAS um JSON com os dados fundamentalistas mais recentes do ativo {ticker} listado na B3.

Formato exato (sem texto extra, sem markdown):
{{“pl_ratio”: 12.5, “roe_pct”: 18.3, “pvp”: 1.4, “disponivel”: true}}

Se não tiver dados confiáveis, retorne:
{{“pl_ratio”: 0, “roe_pct”: 0, “pvp”: 0, “disponivel”: false}}

Retorne APENAS o JSON.”””

```
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json",
                 "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 200,
              "system": "Responda SOMENTE com JSON válido, sem texto adicional.",
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
```

# ———————————————————————––

# SIDEBAR

# ———————————————————————––

with st.sidebar:
st.header(“⚙️ Parâmetros”)
ativo_input = st.text_input(“Ticker do Ativo:”, value=“JHSF3”).upper()
st.write(”—”)
st.caption(“🔑 API Key Anthropic (opcional)”)
api_key_input = st.text_input(
“Chave para análise fundamentalista via IA:”,
type=“password”,
help=“console.anthropic.com — se não informada, usa yfinance”
)
st.button(“Executar Análise”, use_container_width=True)

# ———————————————————————––

# EXECUÇÃO PRINCIPAL

# ———————————————————————––

if ativo_input:
try:
with st.spinner(f”Coletando dados de {ativo_input}…”):
df, info = carregar_dados(ativo_input)

```
    if df.empty:
        st.error("Nenhum dado encontrado. Verifique o ticker.")
    else:
        df = calcular_indicadores(df)
        cols_necessarias = ['MM6', 'MM72', 'HiLo', 'Vol_MM21', 'Donchian_Max']
        df_v = df.dropna(subset=cols_necessarias)

        if len(df_v) < 6:
            st.error("Dados insuficientes para calcular os indicadores.")
        else:
            a  = df_v.iloc[-1]   # candle atual
            a2 = df_v.iloc[-2]   # candle anterior

            close  = safe_float(a['Close'])
            open_  = safe_float(a['Open'])
            high   = safe_float(a['High'])
            low    = safe_float(a['Low'])
            hilo   = safe_float(a['HiLo'])
            mm6    = safe_float(a['MM6'])
            mm72   = safe_float(a['MM72'])
            vol    = safe_float(a['Volume'])
            vol_mm = safe_float(a['Vol_MM21'])
            corpo_pct = safe_float(a['Corpo_Pct'])
            fech_pos  = safe_float(a['Fech_Pos'])
            donchian_max = safe_float(a['Donchian_Max'])
            donchian_min = safe_float(a['Donchian_Min'])

            ultimos5 = df_v['Close'].tail(5)
            faixa5   = (ultimos5.max() - ultimos5.min()) / ultimos5.mean() if ultimos5.mean() > 0 else 0

            # ── CAMADA 1: Setup HiLo (cada condição = 10 pts → máx 50 pts) ──────
            score_c1 = 0
            motivos_c1 = []
            modo_compra = close > hilo  # define direção dominante

            # 1. Preço vs HiLo
            if close > hilo:
                score_c1 += 10
                motivos_c1.append(("✅", "Preço ACIMA do HiLo — tendência altista", "+10"))
            else:
                motivos_c1.append(("❌", "Preço ABAIXO do HiLo — tendência baixista", "0"))

            # 2. Preço vs MM6
            if close > mm6:
                score_c1 += 10
                motivos_c1.append(("✅", "Preço ACIMA da MM6 — momentum positivo", "+10"))
            else:
                motivos_c1.append(("❌", "Preço ABAIXO da MM6 — momentum negativo", "0"))

            # 3. HiLo vs MM6 (suporte em compra, resistência em venda)
            if modo_compra and hilo < mm6:
                score_c1 += 10
                motivos_c1.append(("✅", "HiLo abaixo da MM6 — suporte confirmado", "+10"))
            elif not modo_compra and hilo > mm6:
                score_c1 += 10
                motivos_c1.append(("✅", "HiLo acima da MM6 — resistência confirmada (venda)", "+10"))
            else:
                motivos_c1.append(("❌", "HiLo e MM6 sem confirmação direcional", "0"))

            # 4. MM6 vs MM72 (cruzamento)
            if close > hilo and mm6 > mm72:
                score_c1 += 10
                motivos_c1.append(("✅", "MM6 ACIMA da MM72 — cruzamento altista confirmado", "+10"))
            elif close <= hilo and mm6 < mm72:
                score_c1 += 10
                motivos_c1.append(("✅", "MM6 ABAIXO da MM72 — cruzamento baixista confirmado", "+10"))
            else:
                motivos_c1.append(("❌", "Médias sem cruzamento direcional confirmado", "0"))

            # 5. Preço vs MM72
            if close > mm72:
                score_c1 += 10
                motivos_c1.append(("✅", "Preço ACIMA da MM72 — tendência de médio prazo altista", "+10"))
            else:
                motivos_c1.append(("❌", "Preço ABAIXO da MM72 — tendência de médio prazo baixista", "0"))

            # ── CAMADA 2: Candle + Volume (bônus, cada = 0,5 pt → máx 5 pts) ────
            bonus_c2 = 0.0
            motivos_c2 = []

            # 6. Candle de alta ou baixa
            candle_alta = close > open_
            if (modo_compra and candle_alta) or (not modo_compra and not candle_alta):
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Candle {'de alta' if candle_alta else 'de baixa'} alinhado com a tendência", "+0.5"))
            else:
                motivos_c2.append(("⚪", f"Candle {'de alta' if candle_alta else 'de baixa'} contra a tendência", "0"))

            # 7. Força do corpo ≥ 50%
            if corpo_pct >= 0.50:
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Corpo forte: {corpo_pct*100:.0f}% da amplitude", "+0.5"))
            else:
                motivos_c2.append(("⚪", f"Corpo fraco: {corpo_pct*100:.0f}% da amplitude", "0"))

            # 8. Fechamento próximo da máxima (compra) ou mínima (venda)
            if modo_compra and fech_pos >= 0.70:
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Fechamento próximo da máxima ({fech_pos*100:.0f}% da amplitude)", "+0.5"))
            elif not modo_compra and fech_pos <= 0.30:
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Fechamento próximo da mínima ({fech_pos*100:.0f}% da amplitude)", "+0.5"))
            else:
                motivos_c2.append(("⚪", f"Fechamento no meio da amplitude ({fech_pos*100:.0f}%)", "0"))

            # 9. Volume acima da média
            if vol_mm > 0 and vol > vol_mm:
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Volume acima da MM21 ({vol/vol_mm:.1f}x)", "+0.5"))
            else:
                motivos_c2.append(("⚪", "Volume abaixo da MM21", "0"))

            # 10. Volume > 1,5x a média
            if vol_mm > 0 and vol > vol_mm * 1.5:
                bonus_c2 += 0.5
                motivos_c2.append(("✅", f"Volume forte: {vol/vol_mm:.1f}x a média — confirmação forte", "+0.5"))
            else:
                motivos_c2.append(("⚪", f"Volume sem confirmação forte (< 1.5x)", "0"))

            # ── CAMADA 3: Contexto e Fase (informativo, sem pontuação direta) ───
            alertas_c3 = []

            # 11. Fase do ativo
            dist_mm72_pct = abs(close - mm72) / mm72 * 100 if mm72 > 0 else 0
            if faixa5 <= 0.05:
                fase = "🔲 CONSOLIDAÇÃO"
                rompimento = "✅ Rompeu para cima" if close > ultimos5.iloc[0] else ("🔴 Rompeu para baixo" if close < ultimos5.iloc[0] else "⏳ Ainda lateral")
                alertas_c3.append(("🔲", f"Fase: CONSOLIDAÇÃO — {rompimento}", ""))
            elif close > mm6 > mm72:
                fase = "📈 TENDÊNCIA DE ALTA"
                alertas_c3.append(("📈", "Fase: TENDÊNCIA DE ALTA — estrutura de médias alinhada", ""))
            elif close < mm6 < mm72:
                fase = "📉 TENDÊNCIA DE BAIXA"
                alertas_c3.append(("📉", "Fase: TENDÊNCIA DE BAIXA — estrutura de médias alinhada", ""))
            else:
                fase = "🔄 POSSÍVEL REVERSÃO"
                alertas_c3.append(("🔄", "Fase: POSSÍVEL REVERSÃO — médias desalinhadas", ""))

            # 12. Distância da MM72
            alertas_c3.append(("📏", f"Distância da MM72: {dist_mm72_pct:.1f}% {'(esticado — atenção)' if dist_mm72_pct > 15 else '(dentro do normal)'}", ""))

            # 13. Detector de consolidação
            if faixa5 <= 0.05:
                alertas_c3.append(("⚠️", f"Lateral detectada: faixa dos 5 pregões = {faixa5*100:.1f}%", ""))
            else:
                alertas_c3.append(("✅", f"Sem consolidação: faixa dos 5 pregões = {faixa5*100:.1f}%", ""))

            # 14. Alerta de iminência HiLo
            dist_hilo_pct = abs(close - hilo) / close * 100 if close > 0 else 0
            if dist_hilo_pct < 2.0:
                alertas_c3.append(("🚨", f"ALERTA: HiLo a {dist_hilo_pct:.1f}% do preço — virada iminente!", ""))
            else:
                alertas_c3.append(("🟢", f"HiLo a {dist_hilo_pct:.1f}% do preço — sem iminência de virada", ""))

            # ── FUNDAMENTALISTA (40 pts) ───────────────────────────────────────
            score_fund = 0
            motivos_fund = []
            fonte_fund = ""
            pe_ratio = roe = pb_ratio = 0.0

            if api_key_input:
                with st.spinner("Buscando fundamentalista via IA..."):
                    dados_claude = buscar_fundamentals_claude(ativo_input, api_key_input)
                if dados_claude:
                    pe_ratio = safe_float(dados_claude.get("pl_ratio", 0))
                    roe      = safe_float(dados_claude.get("roe_pct", 0)) / 100
                    pb_ratio = safe_float(dados_claude.get("pvp", 0))
                    fonte_fund = "🤖 Claude AI"
                else:
                    fonte_fund = "⚠️ IA indisponível — usando yfinance"

            if not api_key_input or not fonte_fund.startswith("🤖"):
                pe_ratio = safe_float(info.get('trailingPE') or info.get('forwardPE') or 0)
                roe      = safe_float(info.get('returnOnEquity') or 0)
                pb_ratio = safe_float(info.get('priceToBook') or 0)
                fonte_fund = "📡 yfinance" if (pe_ratio > 0 or roe > 0 or pb_ratio > 0) else "⚠️ Indisponível"

            if 0 < pe_ratio < 20:
                score_fund += 15
                motivos_fund.append(("✅", f"P/L atrativo: {pe_ratio:.2f}x", "+15"))
            elif pe_ratio >= 20:
                motivos_fund.append(("❌", f"P/L esticado: {pe_ratio:.2f}x", "0"))
            else:
                motivos_fund.append(("⚠️", "P/L indisponível", "0"))

            if roe > 0.10:
                score_fund += 15
                motivos_fund.append(("✅", f"ROE forte: {roe*100:.1f}%", "+15"))
            elif roe > 0:
                motivos_fund.append(("❌", f"ROE fraco: {roe*100:.1f}%", "0"))
            else:
                motivos_fund.append(("⚠️", "ROE indisponível", "0"))

            if 0 < pb_ratio < 3:
                score_fund += 10
                motivos_fund.append(("✅", f"P/VP justo: {pb_ratio:.2f}x", "+10"))
            elif pb_ratio >= 3:
                motivos_fund.append(("❌", f"P/VP elevado: {pb_ratio:.2f}x", "0"))
            else:
                motivos_fund.append(("⚠️", "P/VP indisponível", "0"))

            # ── SCORE FINAL ───────────────────────────────────────────────────
            # C1 normalizado para 60 pts · bônus C2 até 5 pts · fund 40 pts
            score_tec_base = score_c1  # já é de 0-50, escala para 0-60
            score_tec_norm = round(score_tec_base * (60 / 50))
            score_bonus    = round(bonus_c2 * 2)  # bônus até 10 pts extras
            score_final    = min(score_tec_norm + score_fund + score_bonus, 100)

            if score_final >= 75:
                status, cor, alert_type = "COMPRA", "#22c55e", "success"
            elif score_final >= 45:
                status, cor, alert_type = "NEUTRA", "#eab308", "warning"
            else:
                status, cor, alert_type = "VENDA", "#ef4444", "error"

            # ── PAINEL PRINCIPAL ──────────────────────────────────────────────
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.metric("Cotação Atual", f"R$ {close:.2f}",
                          f"{(close - safe_float(a2['Close'])):+.2f} R$")
                st.markdown(f"### Status: <span style='color:{cor}'>{status}</span>",
                            unsafe_allow_html=True)
                st.progress(min(score_final / 100.0, 1.0))
                st.write(f"**Pontuação Total: {score_final}/100**")
                st.caption(fase)

            with col2:
                st.subheader("Resumo dos Pesos")
                st.write(f"📐 **Setup HiLo (C1):** {score_tec_norm}/60 pts")
                st.write(f"🕯️ **Bônus Candle+Vol (C2):** +{score_bonus} pts")
                st.write(f"🏢 **Fundamentos:** {score_fund}/40 pts")
                st.write("---")
                st.caption(f"Fonte: {fonte_fund}")
                st.caption(f"P/L: {pe_ratio:.2f}x" if pe_ratio > 0 else "P/L: —")
                st.caption(f"ROE: {roe*100:.1f}%" if roe > 0 else "ROE: —")
                st.caption(f"P/VP: {pb_ratio:.2f}x" if pb_ratio > 0 else "P/VP: —")

            with col3:
                if alert_type == "success":
                    st.success("🎯 **Compra:** Setup HiLo alinhado com momentum e fundamentos.")
                elif alert_type == "warning":
                    st.warning("⚠️ **Neutro:** Sinais mistos — aguardar confirmação.")
                else:
                    st.error("🛑 **Venda/Evitar:** Estrutura baixista ou múltiplos fracos.")

                # Alertas da Camada 3
                for ico, txt, _ in alertas_c3:
                    if "ALERTA" in txt or "iminente" in txt:
                        st.warning(f"{ico} {txt}")
                    else:
                        st.info(f"{ico} {txt}")

            st.write("---")

            # ── GRÁFICO ───────────────────────────────────────────────────────
            st.subheader("Visualização Gráfica")

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df_v.index, open=df_v['Open'], high=df_v['High'],
                low=df_v['Low'], close=df_v['Close'], name='Preço'
            ))
            fig.add_trace(go.Scatter(x=df_v.index, y=df_v['HiLo'],
                line=dict(color='#00d4ff', width=2), name='HiLo'))
            fig.add_trace(go.Scatter(x=df_v.index, y=df_v['MM6'],
                line=dict(color='#ffca28', width=1.5), name='MM6'))
            fig.add_trace(go.Scatter(x=df_v.index, y=df_v['MM72'],
                line=dict(color='#ff5252', width=2), name='MM72'))
            fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Donchian_Max'],
                line=dict(color='rgba(130,130,130,0.4)', width=1, dash='dash'), name='Donchian Max'))
            fig.add_trace(go.Scatter(x=df_v.index, y=df_v['Donchian_Min'],
                line=dict(color='rgba(130,130,130,0.4)', width=1, dash='dash'), name='Donchian Min'))

            fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                              xaxis_rangeslider_visible=False, template="plotly_dark",
                              legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # ── DETALHAMENTO ──────────────────────────────────────────────────
            st.subheader("Detalhamento Completo")
            col_c1, col_c2, col_c3, col_fund = st.columns(4)

            with col_c1:l:
                st.markdown("**📐 Camada 1 — Setup HiLo**")
                for ico, txt, pts in motivos_c1:
                    st.write(f"{ico} {txt} `{pts}`")

            with col_c2:
                st.markdown("**🕯️ Camada 2 — Candle + Volume**")
                for ico, txt, pts in motivos_c2:
                    st.write(f"{ico} {txt} `{pts}`")

            with col_c3:
                st.markdown("**🗺️ Camada 3 — Contexto**")
                for ico, txt, _ in alertas_c3:
                    st.write(f"{ico} {txt}")

            with col_fund:
                st.markdown(f"**🏢 Fundamentalista** _{fonte_fund}_")
                for ico, txt, pts in motivos_fund:
                    st.write(f"{ico} {txt} `{pts}`")

except Exception as e:
    st.error(f"Erro: {e}")
    st.exception(e)
```