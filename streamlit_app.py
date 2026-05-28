#!/usr/bin/env python3
"""
Streamlit Stock Analyzer with Sentiment Analysis
Author: Vikas Ramaswamy

Streamlit web application for sentiment-enhanced stock analysis.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

_VALID_TICKER = re.compile(r'^[A-Z0-9]{1,5}(\.[A-Z]{1,2})?$')

def is_valid_ticker(symbol: str) -> bool:
    return bool(_VALID_TICKER.match(symbol))

from stock_analyzer_unified import UnifiedStockAnalyzer

# Set up the stock analyzer
analyzer = UnifiedStockAnalyzer(use_realtime_sentiment=True)
if analyzer.use_realtime_sentiment:
    st.info("🔴 **LIVE**: Using real-time sentiment analysis from news sources")
else:
    st.warning("⚠️ Using simulated sentiment. Run setup_apis.py for real-time data")

# Page config
st.set_page_config(
    page_title="Stock Analyzer - Vikas Ramaswamy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def calculate_expected_return(predicted_price, current_price):
    """Figure out the expected return as a percentage."""
    if predicted_price and current_price > 0:
        return ((predicted_price - current_price) / current_price) * 100
    return 0


def build_stock_chart(symbol, result):
    """
    3-panel Plotly chart: candlestick + MAs + prediction, RSI, volume.
    Professional dark-compatible design — no text clutter on the canvas.
    """
    try:
        import yfinance as yf
        from datetime import timedelta

        raw = yf.Ticker(symbol).history(period="6mo")
        if raw is None or len(raw) < 20:
            return None

        data = analyzer.calculate_indicators(raw)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.58, 0.22, 0.20],
        )

        # ── Panel 1: Candlestick ──────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'],   close=data['Close'],
            name="OHLC",
            increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
            hoverinfo='x+y',
        ), row=1, col=1)

        # MA20
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MA_20'],
            name="MA 20",
            line=dict(color="#ff9800", width=1.5, dash='solid'),
            hovertemplate="MA20: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # MA50
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MA_50'],
            name="MA 50",
            line=dict(color="#42a5f5", width=1.5, dash='solid'),
            hovertemplate="MA50: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # Predicted price — star with error bars for confidence range
        if result.get('predicted_price'):
            next_day = data.index[-1] + timedelta(days=1)
            pred = result['predicted_price']

            reliability = result.get('ml_reliability', 'low')
            marker_color = (
                "#ce93d8" if reliability == 'high'
                else "#ffb74d" if reliability == 'low'
                else "#78909c"
            )

            # Error bar range from confidence interval
            err_plus, err_minus = 0, 0
            if result.get('prediction_interval'):
                lo, hi = result['prediction_interval']
                err_plus  = hi - pred
                err_minus = pred - lo

            fig.add_trace(go.Scatter(
                x=[next_day],
                y=[pred],
                mode='markers',
                name="Predicted",
                marker=dict(
                    symbol='star',
                    size=18,
                    color=marker_color,
                    line=dict(color='white', width=1.2),
                ),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[err_plus],
                    arrayminus=[err_minus],
                    color=marker_color,
                    thickness=2,
                    width=10,
                    visible=True,
                ),
                hovertemplate=(
                    f"<b>Predicted</b>: ${pred:.2f}<br>"
                    f"Range: ${pred - err_minus:.2f} – ${pred + err_plus:.2f}"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

        # ── Panel 2: RSI ──────────────────────────────────────────────────
        # Overbought/oversold fills
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)",
                      line_width=0, row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name="RSI",
            line=dict(color="#ec407a", width=1.5),
            fill='tozeroy',
            fillcolor='rgba(236,64,122,0.05)',
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

        # Reference lines
        for level, color in [(70, "rgba(239,83,80,0.5)"), (50, "rgba(180,180,180,0.3)"), (30, "rgba(38,166,154,0.5)")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          line_width=1, row=2, col=1)

        # ── Panel 3: Volume ───────────────────────────────────────────────
        bar_colors = [
            "rgba(38,166,154,0.7)" if c >= o else "rgba(239,83,80,0.7)"
            for c, o in zip(data['Close'], data['Open'])
        ]
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name="Volume",
            marker_color=bar_colors,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=3, col=1)

        # ── Layout ────────────────────────────────────────────────────────
        fig.update_layout(
            height=560,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12, color="#aaa"),
            legend=dict(
                orientation='h',
                yanchor='top', y=-0.06,
                xanchor='center', x=0.5,
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
                font=dict(size=12),
            ),
            margin=dict(l=10, r=20, t=20, b=10),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(30,30,30,0.9)',
                bordercolor='rgba(100,100,100,0.5)',
                font=dict(size=12, color='white'),
            ),
        )

        # Price axis — dollar prefix, clean grid
        fig.update_yaxes(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            row=1, col=1,
        )
        # RSI axis
        fig.update_yaxes(
            range=[0, 100],
            tickvals=[30, 50, 70],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            row=2, col=1,
        )
        # Volume axis — abbreviated (K/M)
        fig.update_yaxes(
            tickformat=".2s",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            row=3, col=1,
        )
        # X axes
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.08)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='rgba(180,180,180,0.4)',
            spikethickness=1,
        )
        # Only show x-axis tick labels on bottom panel
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        return fig

    except Exception:
        return None

# ── CSS — design system ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default sidebar toggle & Streamlit branding ──────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Top nav bar wrapper ────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.4rem 0;
    border-bottom: 1px solid rgba(130,130,130,0.15);
    margin-bottom: 0.5rem;
}
.app-logo { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px; }
.app-logo span { color: #6366f1; }
.app-tagline { font-size: 0.8rem; opacity: 0.45; margin-top: 2px; }
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}
.live-dot {
    width: 6px; height: 6px; background: #10b981;
    border-radius: 50%; animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ── Tab bar ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid rgba(130,130,130,0.15);
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.65rem 1.6rem;
    font-weight: 500;
    font-size: 0.9rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    background: transparent !important;
    color: rgba(150,150,150,0.7);
    transition: color 0.15s, border-color 0.15s;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #6366f1 !important;
    color: #6366f1 !important;
    background: transparent !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: inherit;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── KPI cards ──────────────────────────────────────────────────────── */
.professional-metric {
    border: 1px solid rgba(130,130,130,0.18);
    border-radius: 12px;
    padding: 1rem 0.9rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    min-height: 90px;
    display: flex; flex-direction: column; justify-content: center;
    margin-bottom: 0.5rem;
}
.professional-metric:hover {
    border-color: rgba(99,102,241,0.45);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: -0.5px; margin-bottom: 0.2rem; line-height: 1.1;
}
.metric-label {
    font-size: 0.7rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.9px; opacity: 0.5;
}

/* ── Recommendation badge ───────────────────────────────────────────── */
.rec-badge {
    border-radius: 14px; padding: 1.2rem 2rem;
    text-align: center; margin: 1rem 0;
}
.rec-badge h3 { font-size: 1.4rem; font-weight: 700; margin: 0 0 0.25rem 0; }
.rec-badge p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
.recommendation-strong-buy  { background: linear-gradient(135deg,#065f46,#059669); color:#fff; }
.recommendation-buy         { background: linear-gradient(135deg,#0e7490,#0891b2); color:#fff; }
.recommendation-hold        { background: linear-gradient(135deg,#92400e,#d97706); color:#fff; }
.recommendation-sell        { background: linear-gradient(135deg,#991b1b,#dc2626); color:#fff; }
.recommendation-strong-sell { background: linear-gradient(135deg,#7f1d1d,#b91c1c); color:#fff; }

/* ── Sentiment ──────────────────────────────────────────────────────── */
.sentiment-positive { color: #34d399; font-weight: 600; }
.sentiment-negative { color: #f87171; font-weight: 600; }
.sentiment-neutral  { color: #fbbf24; font-weight: 600; }

/* ── News ticker ────────────────────────────────────────────────────── */
.news-ticker {
    border: 1px solid rgba(130,130,130,0.18);
    border-radius: 8px; padding: 0.65rem 1rem;
    margin: 0.8rem 0; overflow: hidden;
    white-space: nowrap; position: relative;
}
.news-ticker::before {
    content: 'LIVE'; position: absolute; left: 10px; top: 50%;
    transform: translateY(-50%);
    background: #dc2626; color: white;
    padding: 2px 7px; border-radius: 3px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px; z-index: 2;
}
.news-item {
    display: inline-block;
    animation: scroll 40s linear infinite;
    padding-left: 65px; font-size: 0.85rem; font-weight: 500; opacity: 0.75;
}
@keyframes scroll { 0%{transform:translateX(100%);} 100%{transform:translateX(-100%);} }

/* ── Signal / reason tags ───────────────────────────────────────────── */
.reason-tag {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.22);
    padding: 3px 9px; border-radius: 20px;
    font-size: 0.78rem; margin: 2px 2px;
}

/* ── Feature grid (overview) ────────────────────────────────────────── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem; margin: 1.5rem 0;
}
.feature-card {
    border: 1px solid rgba(130,130,130,0.18);
    border-radius: 14px; padding: 1.4rem 1.2rem;
    transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover { border-color: rgba(99,102,241,0.4); transform: translateY(-3px); }
.feature-icon  { font-size: 1.8rem; margin-bottom: 0.6rem; }
.feature-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.4rem; }
.feature-desc  { font-size: 0.82rem; opacity: 0.55; line-height: 1.55; }

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    text-align: center; padding: 4rem 0; opacity: 0.4;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-state p { font-size: 1rem; }

/* ── Section divider ────────────────────────────────────────────────── */
.section-divider { height:1px; background:rgba(130,130,130,0.12); margin:1.2rem 0; border:none; }

/* ── Portfolio table header ─────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── App header ────────────────────────────────────────────────────────────────
live_badge = (
    '<span class="live-badge"><span class="live-dot"></span>LIVE</span>'
    if analyzer.use_realtime_sentiment
    else '<span style="opacity:0.4;font-size:0.8rem;">Demo mode</span>'
)
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-logo">📈 Stock<span>Analyzer</span></div>
        <div class="app-tagline">ML predictions · Technical analysis · Sentiment · by Vikas Ramaswamy</div>
    </div>
    <div>{live_badge}</div>
</div>
""", unsafe_allow_html=True)


# ── Helper: render complete analysis for one symbol ───────────────────────────

def render_stock_analysis(symbol):
    """Renders KPIs, recommendation badge, chart, signals, sentiment for a symbol."""

    # News ticker
    sent = analyzer.get_sentiment_data(symbol)
    headlines = sent.get('top_headlines', [])
    if headlines:
        st.markdown(
            f'<div class="news-ticker"><div class="news-item">'
            f'{"  ·  ".join(headlines[:5])}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Progress steps
    bar = st.progress(0, text="Fetching market data…")
    result = None
    try:
        bar.progress(35, text="Running sentiment analysis…")
        bar.progress(65, text="Training AI model…")
        result = analyzer.analyze_stock(symbol)
        bar.progress(100, text="Complete")
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(f"Analysis failed for **{symbol}**: {e}")
        return

    if not result:
        st.error(f"No data found for **{symbol}**. Check the ticker is listed on a major exchange.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────
    expected_return = calculate_expected_return(
        result['predicted_price'], result['current_price']
    )
    c1, c2, c3, c4, c5 = st.columns(5)

    def kpi(col, value, label, color=None):
        style = f'style="color:{color};"' if color else ''
        col.markdown(
            f'<div class="professional-metric">'
            f'<div class="metric-value" {style}>{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    kpi(c1, f"${result['current_price']:.2f}", "Current Price")
    kpi(
        c2,
        f"${result['predicted_price']:.2f}" if result['predicted_price'] else "N/A",
        "Predicted Price",
        "#34d399" if (result.get('predicted_price') or 0) > result['current_price'] else "#f87171",
    )
    kpi(c3, f"{expected_return:+.1f}%", "Expected Return",
        "#34d399" if expected_return >= 0 else "#f87171")
    kpi(c4, f"{result['score']}/{result['max_score']}", "Score",
        "#34d399" if result['score'] >= 7 else "#fbbf24" if result['score'] >= 5 else "#f87171")
    kpi(c5, f"{result['sentiment_score']:+.3f}", "Sentiment",
        "#34d399" if result['sentiment_score'] > 0.1
        else "#f87171" if result['sentiment_score'] < -0.1
        else "#fbbf24")

    # ── Recommendation badge ──────────────────────────────────────────────
    rec   = result['recommendation']
    arrow = '⬆' if 'BUY' in rec else '⬇' if 'SELL' in rec else '↔'
    st.markdown(
        f'<div class="rec-badge recommendation-{rec.lower().replace(" ","-")}">'
        f'<h3>{arrow} {rec}</h3>'
        f'<p>Score: {result["score"]}/{result["max_score"]} &nbsp;·&nbsp;'
        f'Sentiment: {result["sentiment_score"]:+.3f} &nbsp;·&nbsp;'
        f'Source: {result["sentiment_data"].get("source","N/A")}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chart (left) + Technicals (right) ────────────────────────────────
    chart_col, signal_col = st.columns([3, 1])

    with chart_col:
        fig = build_stock_chart(symbol, result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            reliability = result.get('ml_reliability', 'low')
            captions = {
                'high':       "🟢 Model R² acceptable — ★ = next-day prediction, error bar = ±1σ range.",
                'low':        "🟡 Model confidence low — ★ prediction is a weak signal, weight the MAs more.",
                'unreliable': "⚠️ Model unreliable — ignore ★ prediction, rely on MA crossovers and RSI.",
            }
            st.caption(captions.get(reliability, ''))
        else:
            st.info("Chart unavailable for this symbol.")

    with signal_col:
        st.markdown("**Why this score**")
        tags = "".join(
            f'<span class="reason-tag">{r}</span>' for r in result['reasons']
        )
        st.markdown(tags, unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        reliability = result.get('ml_reliability', 'low')
        rsi_c = "#34d399" if 30 <= result['rsi'] <= 70 else "#fbbf24"
        r2_c  = {'high': '#34d399', 'low': '#fbbf24', 'unreliable': '#f87171'}.get(reliability, '#fbbf24')
        r2_lbl = {'high': 'Reliable', 'low': 'Weak', 'unreliable': 'Unreliable'}.get(reliability, '')
        vol_c = "#34d399" if result['volume_ratio'] > 1.2 else "inherit"

        st.markdown(f"""
        <div class="professional-metric" style="margin-bottom:.45rem;">
            <div class="metric-value" style="color:{rsi_c};font-size:1.35rem;">{result['rsi']:.1f}</div>
            <div class="metric-label">RSI</div>
        </div>
        <div class="professional-metric" style="margin-bottom:.45rem;">
            <div class="metric-value" style="font-size:1.05rem;">${result['ma_20']:.2f} / ${result['ma_50']:.2f}</div>
            <div class="metric-label">MA20 / MA50</div>
        </div>
        <div class="professional-metric" style="margin-bottom:.45rem;">
            <div class="metric-value" style="color:{vol_c};font-size:1.2rem;">{result['volume_ratio']:.2f}×</div>
            <div class="metric-label">Volume Ratio</div>
        </div>
        <div class="professional-metric">
            <div class="metric-value" style="color:{r2_c};font-size:1.1rem;">{result['model_accuracy']:.2f} R²</div>
            <div class="metric-label">Model ({r2_lbl})</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentiment expander ────────────────────────────────────────────────
    with st.expander("Sentiment & news detail"):
        s1, s2 = st.columns(2)
        ss = result['sentiment_score']
        sc = "positive" if ss > 0.1 else "negative" if ss < -0.1 else "neutral"
        st_text = "Positive" if ss > 0.1 else "Negative" if ss < -0.1 else "Neutral"
        with s1:
            st.markdown(
                f'<div class="sentiment-{sc}">'
                f'<h4>Sentiment: {st_text}</h4>'
                f'<p>Score: {ss:.3f} · Source: {result["sentiment_data"].get("source","N/A")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(max(0.0, min(1.0, (ss + 1) / 2)))
        with s2:
            topics = result['sentiment_data'].get('key_topics', [])
            if topics:
                st.write("**Topics:**", " · ".join(topics[:5]))
            st.write(f"**Articles:** {result['sentiment_data'].get('news_count', 0)}")

    # ── Scoring explainer ─────────────────────────────────────────────────
    with st.expander("How this score was calculated"):
        st.markdown("""
**Technical indicators** (up to 5 pts)
- Price vs MA20 (+1), Price vs MA50 (+1), MA20 vs MA50 (+1)
- RSI 30–70 range (+1), volatility vs own history (+1)

**AI model** (0–2 pts, gated by R²)
- R² > 0.1 → full +2 pts · R² > −0.3 → +1 pt · R² ≤ −0.3 → 0 pts (excluded)

**Sentiment** (0–2 pts)
- Score > 0.10 → +2 pts · Score > 0.05 → +1 pt · Score < −0.10 → −1 pt
        """)


# ── Navigation tabs ───────────────────────────────────────────────────────────
POPULAR = ['TSLA', 'NVDA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN', 'NFLX']

tab_overview, tab_analyze, tab_portfolio = st.tabs([
    "  Overview  ",
    "  Analyze Stock  ",
    "  Portfolio  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Live Market Data</div>
            <div class="feature-desc">5 years of OHLCV history from Yahoo Finance. Candlestick chart with MA20/MA50 overlays, RSI panel, and volume bars.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">ML Price Prediction</div>
            <div class="feature-desc">Random Forest trained on returns (not raw price). Confidence band from 200 individual tree votes. R²-gated scoring prevents unreliable signals.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📰</div>
            <div class="feature-title">Live Sentiment</div>
            <div class="feature-desc">Real headlines via NewsAPI or yfinance.news. VADER sentiment scoring. Three-tier fallback — no static data.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">9-Point Scoring</div>
            <div class="feature-desc">Technical indicators, ML model (reliability-gated), and news sentiment combined into a single actionable score with STRONG BUY → STRONG SELL output.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### How it works")

    w1, w2, w3, w4 = st.columns(4)
    for col, step, title, desc in [
        (w1, "01", "Enter a ticker", "Type any symbol (AAPL, BRK.A, RELIANCE.NS) in the Analyze tab."),
        (w2, "02", "Fetch & compute", "5 years of daily OHLCV fetched. MA, RSI, MACD, volatility and momentum computed."),
        (w3, "03", "Train & predict", "Random Forest trained on % returns via TimeSeriesSplit. Prediction + confidence band generated."),
        (w4, "04", "Score & recommend", "Technical + ML + sentiment signals scored 0–9. Final recommendation with full reasoning shown."),
    ]:
        col.markdown(f"""
        <div class="professional-metric" style="min-height:130px;">
            <div style="font-size:1.8rem;font-weight:700;opacity:0.18;margin-bottom:0.3rem;">{step}</div>
            <div class="metric-value" style="font-size:1rem;">{title}</div>
            <div style="font-size:0.78rem;opacity:0.5;margin-top:0.3rem;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.caption("⚠️ Educational tool only. Not financial advice. Always do your own research before investing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE STOCK
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:

    # Search bar
    search_c, btn_c = st.columns([5, 1])
    with search_c:
        symbol_input = st.text_input(
            "ticker",
            placeholder="Enter ticker symbol — e.g. TSLA, AAPL, BRK.A, RELIANCE.NS",
            label_visibility="collapsed",
            key="analyze_input",
        ).upper().strip()
    with btn_c:
        analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)

    # Popular stock quick-select
    st.markdown("<div style='margin:0.4rem 0 0.8rem 0;font-size:0.8rem;opacity:0.5;'>Quick select:</div>", unsafe_allow_html=True)
    quick_cols = st.columns(len(POPULAR))
    for i, sym in enumerate(POPULAR):
        if quick_cols[i].button(sym, key=f"qs_{sym}", use_container_width=True):
            st.session_state.analyze_symbol = sym
            st.rerun()

    if analyze_btn and symbol_input:
        if not is_valid_ticker(symbol_input):
            st.error(f"'{symbol_input}' doesn't look like a valid ticker. Use 1–5 uppercase letters, e.g. TSLA or BRK.A")
        else:
            st.session_state.analyze_symbol = symbol_input
            st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if 'analyze_symbol' in st.session_state:
        sym = st.session_state.analyze_symbol
        st.markdown(f"#### {sym}")
        render_stock_analysis(sym)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🔍</div>
            <p>Enter a ticker symbol above or pick a popular stock to get started</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:

    p_input = st.text_area(
        "Symbols",
        placeholder="TSLA, AAPL, NVDA, GOOGL, MSFT",
        height=80,
        label_visibility="collapsed",
        key="portfolio_input",
    )
    run_portfolio = st.button("Analyse Portfolio", type="primary")

    if run_portfolio and p_input:
        symbols = [s.strip().upper() for s in p_input.split(',') if s.strip()]
        st.session_state.portfolio_symbols = symbols

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if 'portfolio_symbols' in st.session_state:
        symbols = st.session_state.portfolio_symbols
        st.markdown(f"#### Portfolio — {len(symbols)} stocks")

        portfolio_results = []
        prog = st.progress(0, text=f"Analysing {symbols[0]}…")

        for i, sym in enumerate(symbols):
            prog.progress((i + 1) / len(symbols), text=f"Analysing {sym}… ({i+1}/{len(symbols)})")
            try:
                r = analyzer.analyze_stock(sym)
                if r:
                    r['expected_return'] = calculate_expected_return(
                        r['predicted_price'], r['current_price']
                    )
                    portfolio_results.append(r)
            except Exception as e:
                st.warning(f"Could not analyse {sym}: {e}")

        prog.empty()

        if not portfolio_results:
            st.error("No results returned. Check your ticker symbols.")
        else:
            portfolio_results.sort(key=lambda x: x['score'], reverse=True)

            # ── Summary KPIs ──────────────────────────────────────────────
            avg_return    = np.mean([r['expected_return'] for r in portfolio_results])
            avg_sentiment = np.mean([r['sentiment_score'] for r in portfolio_results])
            avg_score     = np.mean([r['score'] for r in portfolio_results])
            strong_buys   = sum(1 for r in portfolio_results if r['recommendation'] == 'STRONG BUY')

            s1, s2, s3, s4 = st.columns(4)
            for col, val, lbl, color in [
                (s1, len(portfolio_results), "Stocks", None),
                (s2, f"{avg_return:+.1f}%", "Avg Expected Return",
                     "#34d399" if avg_return >= 0 else "#f87171"),
                (s3, f"{avg_score:.1f}/9", "Avg Score",
                     "#34d399" if avg_score >= 7 else "#fbbf24" if avg_score >= 5 else "#f87171"),
                (s4, f"{avg_sentiment:+.3f}", "Avg Sentiment",
                     "#34d399" if avg_sentiment > 0.1 else "#f87171" if avg_sentiment < -0.1 else "#fbbf24"),
            ]:
                style = f'style="color:{color};"' if color else ''
                col.markdown(
                    f'<div class="professional-metric">'
                    f'<div class="metric-value" {style}>{val}</div>'
                    f'<div class="metric-label">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Sortable results table ────────────────────────────────────
            df = pd.DataFrame([{
                'Symbol':        r['symbol'],
                'Price':         f"${r['current_price']:.2f}",
                'Predicted':     f"${r['predicted_price']:.2f}" if r['predicted_price'] else "N/A",
                'Exp. Return':   f"{r['expected_return']:+.1f}%",
                'Sentiment':     f"{r['sentiment_score']:+.3f}",
                'Model R²':      f"{r['model_accuracy']:.2f}",
                'Score':         f"{r['score']}/{r['max_score']}",
                'Recommendation': r['recommendation'],
            } for r in portfolio_results])

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Recommendation distribution bar ───────────────────────────
            rec_counts = {}
            for r in portfolio_results:
                rec_counts[r['recommendation']] = rec_counts.get(r['recommendation'], 0) + 1

            rec_order  = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
            rec_colors = {'STRONG BUY': '#059669', 'BUY': '#0891b2',
                          'HOLD': '#d97706', 'SELL': '#dc2626', 'STRONG SELL': '#b91c1c'}

            fig_dist = go.Figure(go.Bar(
                x=[rc for rc in rec_order if rc in rec_counts],
                y=[rec_counts[rc] for rc in rec_order if rc in rec_counts],
                marker_color=[rec_colors[rc] for rc in rec_order if rc in rec_counts],
                text=[rec_counts[rc] for rc in rec_order if rc in rec_counts],
                textposition='outside',
            ))
            fig_dist.update_layout(
                height=240,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                xaxis=dict(showgrid=False),
                font=dict(family="Inter, sans-serif", size=12),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💼</div>
            <p>Enter comma-separated symbols above and click Analyse Portfolio</p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-divider' style='margin-top:3rem;'>", unsafe_allow_html=True)
st.caption("© 2025 Vikas Ramaswamy · Educational tool only · Not financial advice · Data via Yahoo Finance & NewsAPI")
