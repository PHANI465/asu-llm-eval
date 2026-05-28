# =============================================================================
# dashboard/app.py
# ASU LLM Evaluation — Professional Dashboard  (Dark / Light theme toggle)
#
# Run with:  streamlit run dashboard/app.py
# =============================================================================

import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Page config  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASU LLM Eval Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Theme palettes
# ─────────────────────────────────────────────────────────────────────────────
DARK = {
    "name":        "dark",
    "app_bg":      "#0F1117",
    "sidebar_bg":  "#13151F",
    "card":        "#1A1D2E",
    "card2":       "#1E2235",
    "border":      "#2D3250",
    "border_hi":   "#3D4577",
    "gold":        "#FFC627",
    "maroon":      "#8C1D40",
    "green":       "#00E676",
    "red":         "#FF5252",
    "blue":        "#448AFF",
    "teal":        "#00BCD4",
    "purple":      "#CE93D8",
    "orange":      "#FFB74D",
    "text":        "#FFFFFF",
    "text2":       "#C8D0E0",
    "muted":       "#8892A4",
    "chart_bg":    "#1A1D2E",
    "chart_grid":  "#2D3250",
    "hero_grad":   "linear-gradient(135deg, #2A0A1A 0%, #1A1D2E 55%, #0F1530 100%)",
}

LIGHT = {
    "name":        "light",
    "app_bg":      "#F0F3FA",
    "sidebar_bg":  "#FFFFFF",
    "card":        "#FFFFFF",
    "card2":       "#F8FAFF",
    "border":      "#DDE3F0",
    "border_hi":   "#B8C5E0",
    "gold":        "#C98A00",
    "maroon":      "#8C1D40",
    "green":       "#1B8A4C",
    "red":         "#C0392B",
    "blue":        "#1A6BC9",
    "teal":        "#0097A7",
    "purple":      "#7B1FA2",
    "orange":      "#E65100",
    "text":        "#1A1D2E",
    "text2":       "#3A4060",
    "muted":       "#6B7A99",
    "chart_bg":    "#FFFFFF",
    "chart_grid":  "#E8ECF5",
    "hero_grad":   "linear-gradient(135deg, #6B0A26 0%, #8C1D40 40%, #A0234D 100%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Session state — theme toggle
# ─────────────────────────────────────────────────────────────────────────────
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "dark"

T = DARK if st.session_state.theme_name == "dark" else LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CSS injection
#
# Strategy:
#   • .streamlit/config.toml sets Streamlit's base theme to "light"
#     → all Streamlit widgets render with dark text on white by default
#   • Light mode: only minor overrides (ASU brand colors, layout tweaks)
#   • Dark mode: comprehensive override of EVERY background/text/border
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(t: dict) -> None:
    is_dark = t["name"] == "dark"

    # Shared rules for both themes
    shared = f"""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        #MainMenu, footer {{ visibility: hidden; }}
        .stDeployButton {{ display: none; }}

        .block-container {{
            padding: 1rem 1.8rem 2rem !important;
            max-width: 1700px !important;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }}
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {t["app_bg"]}; }}
        ::-webkit-scrollbar-thumb {{ background: {t["border_hi"]}; border-radius: 3px; }}

        /* ── Remove white gaps injected around st.html() blocks ── */
        [data-testid="stHtml"] {{
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }}
        div.element-container:has([data-testid="stHtml"]) {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}
        /* Tighten global vertical rhythm */
        .element-container {{
            margin-bottom: 0.15rem !important;
        }}

        .stButton > button {{
            background: {t["gold"]} !important;
            color: {"#000" if is_dark else "#fff"} !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}
        .stButton > button:hover {{ opacity: 0.86 !important; }}

        [data-testid="stDataFrame"] {{
            border-radius: 12px !important;
            overflow: hidden !important;
        }}
        [data-testid="stExpander"] {{
            border-radius: 12px !important;
        }}
        hr {{
            border-color: {t["border"]} !important;
            margin: 0.6rem 0 !important;
        }}
    """

    # Dark-mode-only overrides (light mode is handled by config.toml base theme)
    dark_only = f"""
        /* ── App shell ── */
        .stApp, .main {{
            background-color: {t["app_bg"]} !important;
            color: {t["text"]} !important;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
            border-right: 1px solid {t["border"]} !important;
        }}
        [data-testid="stSidebar"] *:not(button) {{
            color: {t["text"]} !important;
        }}

        /* ── All text ── */
        p, span, div, label, li, td, th {{
            color: {t["text2"]} !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {t["text"]} !important;
        }}
        .stMarkdown, .stMarkdown * {{
            color: {t["text2"]} !important;
        }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: {t["muted"]} !important;
        }}

        /* ── Metric cards ── */
        [data-testid="stMetric"] {{
            background: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
        }}
        [data-testid="stMetricLabel"] > div {{
            color: {t["muted"]} !important;
            font-size: 0.7rem !important;
            letter-spacing: 1.1px !important;
            text-transform: uppercase !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {t["text"]} !important;
            font-size: 1.55rem !important;
            font-weight: 800 !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            background: {t["card"]} !important;
            border-radius: 10px !important;
            padding: 4px !important;
            border: 1px solid {t["border"]} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {t["muted"]} !important;
            border-radius: 8px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {t["border_hi"]} !important;
            color: {t["text"]} !important;
        }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span {{
            color: {t["text"]} !important;
        }}

        /* ── Alerts ── */
        [data-testid="stAlert"] {{
            border: 1px solid {t["border"]} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stAlert"] * {{
            color: {t["text"]} !important;
        }}

        /* ── Toggle widget ── */
        [data-testid="stToggle"] label, [data-testid="stToggle"] span {{
            color: {t["text"]} !important;
        }}

        /* ── Success / info / error banners ── */
        .element-container .stSuccess,
        .element-container .stInfo,
        .element-container .stError,
        .element-container .stWarning {{
            background-color: {t["card"]} !important;
        }}
    """ if is_dark else f"""
        /* ── Light mode: ASU brand overrides on top of config.toml base ── */
        .stApp {{ background-color: {t["app_bg"]} !important; }}
        [data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
            border-right: 1px solid {t["border"]} !important;
        }}
        [data-testid="stMetric"] {{
            border-radius: 14px !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07) !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            border-radius: 10px !important;
            padding: 4px !important;
        }}
    """

    st.markdown(f"<style>{shared}{dark_only}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3b. Color utility — Plotly does NOT accept 8-digit hex (#RRGGBBAA).
#     Use this helper wherever a translucent color is needed in Plotly traces.
# ─────────────────────────────────────────────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert '#RRGGBB' → 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Paths & data loaders
# ─────────────────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_DIR, ".."))
_REPORT_PATH = os.path.join(_ROOT, "results", "latest_report.json")
_DB_PATH     = os.path.join(_ROOT, "results", "eval_history.db")


@st.cache_data(ttl=30)
def load_report() -> dict | None:
    if not os.path.exists(_REPORT_PATH):
        return None
    try:
        with open(_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_history() -> pd.DataFrame:
    if not os.path.exists(_DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(_DB_PATH)
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='eval_runs'"
        ).fetchone()
        if not tbl:
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql_query(
            "SELECT * FROM eval_runs ORDER BY run_timestamp ASC", conn
        )
        conn.close()
        if not df.empty:
            df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HTML component builders  (all use the active theme dict `T`)
# ─────────────────────────────────────────────────────────────────────────────

def _hero(overall, ts, commit, n_q, test_mode) -> str:
    badge_color = T["green"] if overall == "PASS" else T["red"]
    mode_label  = "⚡ TEST MODE — 10 questions" if test_mode else "🚀 Full Run — 100 questions"
    ts_display  = ts[:19].replace("T", " ") if ts else "—"
    shadow      = f"0 0 28px {badge_color}66, inset 0 0 16px {badge_color}18"

    return f"""
    <div style="
        background: {T["hero_grad"]};
        border: 1px solid {T["border"]};
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 1.4rem;
        position: relative; overflow: hidden;
    ">
        <div style="
            position:absolute; top:-60px; right:-60px;
            width:240px; height:240px;
            background: radial-gradient({T["gold"]}22 0%, transparent 70%);
            border-radius:50%; pointer-events:none;
        "></div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:18px;">
            <div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                    <span style="font-size:1.7rem;">🎓</span>
                    <span style="color:#FFFFFF; font-size:1.5rem; font-weight:900;
                                 letter-spacing:0.3px; font-family:Inter,sans-serif;">
                        ASU LLM Evaluation Dashboard
                    </span>
                </div>
                <p style="color:rgba(255,255,255,0.65); font-size:0.8rem; margin:0 0 14px;">
                    Automated RAG quality monitoring · GPT-4o · RAGAS · ChromaDB
                </p>
                <div style="display:flex; gap:22px; flex-wrap:wrap;">
                    <span style="color:rgba(255,255,255,0.5); font-size:0.75rem;">
                        🕐 <span style="color:#FFF; font-weight:500;">{ts_display}</span>
                    </span>
                    <span style="color:rgba(255,255,255,0.5); font-size:0.75rem;">
                        🔖 <span style="color:#FFF; font-weight:500;">{commit}</span>
                    </span>
                    <span style="color:rgba(255,255,255,0.5); font-size:0.75rem;">
                        ❓ <span style="color:#FFF; font-weight:500;">{n_q} questions</span>
                    </span>
                    <span style="color:{T["gold"]}; font-size:0.75rem; font-weight:700;">{mode_label}</span>
                </div>
            </div>
            <div style="text-align:center;">
                <div style="
                    background: {badge_color}22;
                    border: 2.5px solid {badge_color};
                    color: {badge_color};
                    font-size: 2.1rem; font-weight: 900;
                    padding: 14px 36px;
                    border-radius: 16px;
                    letter-spacing: 4px;
                    box-shadow: {shadow};
                    font-family: Inter, sans-serif;
                ">{overall}</div>
                <p style="color:rgba(255,255,255,0.45); font-size:0.62rem;
                           margin:6px 0 0; letter-spacing:1.5px;">OVERALL RESULT</p>
            </div>
        </div>
    </div>
    """


def _kpi_card(label, value_str, threshold_str, passed, sublabel="") -> str:
    border_top = T["green"] if passed else T["red"]
    status     = ("✓ PASS", T["green"]) if passed else ("✗ FAIL", T["red"])
    is_dark    = T["name"] == "dark"
    glow       = f"box-shadow: 0 0 14px {border_top}2A, 0 4px 20px rgba(0,0,0,0.35);" if is_dark \
                 else f"box-shadow: 0 2px 12px rgba(0,0,0,0.08);"
    card_bg    = f"linear-gradient(145deg, {T['card']}, {T['card2']})"

    return f"""
    <div style="
        background: {card_bg};
        border: 1px solid {T['border']};
        border-top: 4px solid {border_top};
        border-radius: 14px;
        padding: 18px 16px 14px;
        min-height: 128px;
        {glow}
        position: relative; overflow: hidden;
    ">
        <div style="
            position:absolute; bottom:-20px; right:-20px;
            width:70px; height:70px;
            background: radial-gradient({border_top}18 0%, transparent 70%);
            border-radius:50%; pointer-events:none;
        "></div>
        <p style="color:{T['muted']}; font-size:0.67rem; letter-spacing:1.3px;
                  text-transform:uppercase; font-weight:600; margin:0 0 8px;">{label}</p>
        <p style="color:{T['text']}; font-size:1.8rem; font-weight:800;
                  margin:0 0 8px; line-height:1; font-family:Inter,sans-serif;">{value_str}</p>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{T['muted']}; font-size:0.67rem;">threshold: {threshold_str}</span>
            <span style="color:{status[1]}; font-size:0.73rem; font-weight:700;">{status[0]}</span>
        </div>
        <p style="color:{T['muted']}; font-size:0.62rem; margin:5px 0 0;">{sublabel}</p>
    </div>
    """


def _section_label(title: str, subtitle: str = "") -> str:
    sub = f'<p style="color:{T["muted"]}; font-size:0.76rem; margin:2px 0 0;">{subtitle}</p>' \
          if subtitle else ""
    return f"""
    <div style="border-left:4px solid {T["gold"]}; padding-left:12px; margin:0.2rem 0 1rem;">
        <p style="color:{T["text"]}; font-size:1.02rem; font-weight:700;
                  margin:0; font-family:Inter,sans-serif;">{title}</p>
        {sub}
    </div>
    """


def _gate_pill(name: str, passed: bool) -> str:
    color = T["green"] if passed else T["red"]
    icon  = "●"
    label = name.replace("_", " ").title()
    return f"""
    <span style="
        display:inline-flex; align-items:center; gap:5px;
        background:{color}18; border:1.5px solid {color}66;
        color:{color}; border-radius:20px;
        padding:5px 13px; font-size:0.73rem; font-weight:700;
        letter-spacing:0.3px; margin:3px;
        box-shadow: 0 0 8px {color}22;
    "><span style="font-size:0.45rem; line-height:1;">{icon}</span> {label}</span>
    """


def _stat_row(label: str, value: str, color: str = None) -> str:
    c = color or T["text"]
    return f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding:7px 0; border-bottom:1px solid {T['border']};">
        <span style="color:{T['muted']}; font-size:0.78rem;">{label}</span>
        <span style="color:{c}; font-size:0.82rem; font-weight:600;">{value}</span>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _chart_layout(title: str = "", height: int = 280) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=T["text"], size=13, family="Inter")),
        plot_bgcolor=T["chart_bg"],
        paper_bgcolor=T["chart_bg"],
        font=dict(color=T["muted"], size=11, family="Inter"),
        xaxis=dict(
            gridcolor=T["chart_grid"], tickfont=dict(color=T["muted"]),
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor=T["chart_grid"], tickfont=dict(color=T["muted"]),
            showgrid=True, zeroline=False,
        ),
        legend=dict(
            bgcolor=T["card2"], bordercolor=T["border"], borderwidth=1,
            font=dict(color=T["text"], size=11),
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        margin=dict(l=10, r=10, t=44, b=10),
        height=height,
        hovermode="x unified",
    )


def _trend_fig(df: pd.DataFrame, series: list, title: str, thresholds: list = None) -> go.Figure:
    """
    series    : [(col_name, color, label), ...]
    thresholds: [(label, value, color, dash), ...]
    """
    fig = go.Figure()

    for col, color, label in series:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["run_timestamp"], y=df[col],
            name=label,
            line=dict(color=color, width=2.5, shape="spline"),
            mode="lines+markers",
            marker=dict(size=9, color=color,
                        line=dict(color=T["chart_bg"], width=2)),
            fill="tozeroy",
            fillcolor=_hex_rgba(color, 0.08),
            hovertemplate=f"<b>{label}</b>: %{{y:.4f}}<extra></extra>",
        ))

    if thresholds:
        for tlabel, tval, tcolor, tdash in thresholds:
            fig.add_hline(
                y=tval, line_dash=tdash, line_color=tcolor, line_width=1.5,
                annotation_text=f"  {tlabel}: {tval}",
                annotation_position="top left",
                annotation_font=dict(color=tcolor, size=10),
            )

    fig.update_layout(**_chart_layout(title))
    return fig


def _bar_fig(df: pd.DataFrame, col: str, title: str, threshold: float,
             threshold_label: str, color: str) -> go.Figure:
    labels = df["run_timestamp"].dt.strftime("%m-%d %H:%M")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=df[col],
        marker_color=[T["green"] if v <= threshold else T["red"] for v in df[col]]
        if "latency" in col or "hallucination" in col or "cost" in col
        else [T["green"] if v >= threshold else T["red"] for v in df[col]],
        text=[f"{v:.3f}" for v in df[col]],
        textposition="outside",
        textfont=dict(color=T["text"], size=10),
        hovertemplate=f"<b>{title}</b>: %{{y:.4f}}<extra></extra>",
        name=col,
    ))
    fig.add_hline(
        y=threshold, line_dash="dot", line_color=T["gold"], line_width=1.5,
        annotation_text=f"  {threshold_label}: {threshold}",
        annotation_font=dict(color=T["gold"], size=10),
    )
    fig.update_layout(**_chart_layout(title, height=260))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Gate table with styling
# ─────────────────────────────────────────────────────────────────────────────
_GATE_FORMAT = {
    "hallucination_rate": ("Hallucination Rate",  lambda v: f"{v:.4f}", "<=", ""),
    "answer_relevancy":   ("Answer Relevancy",    lambda v: f"{v:.4f}", ">=", ""),
    "faithfulness":       ("Faithfulness",        lambda v: f"{v:.4f}", ">=", ""),
    "context_precision":  ("Context Precision",   lambda v: f"{v:.4f}", ">=", ""),
    "latency_p95":        ("Latency P95",         lambda v: f"{v:.3f}", "<=", "s"),
    "cost_per_query":     ("Cost Per Query",      lambda v: f"{v:.4f}", "<=", "$"),
}


def _gate_dataframe(gates: dict) -> None:
    rows = []
    for gate_name, info in gates.items():
        cfg    = _GATE_FORMAT.get(gate_name, (gate_name, str, "?", ""))
        label, fmt_fn, sym, unit = cfg
        val    = info.get("value")
        thr    = info.get("threshold")
        passed = info.get("passed", False)

        v_str = (f"{unit}{fmt_fn(val)}" if unit == "$" else f"{fmt_fn(val)}{unit}") \
                if val is not None else "N/A"
        t_str = (f"{unit}{thr}" if unit == "$" else f"{thr}{unit}") \
                if thr is not None else "N/A"

        rows.append({
            "Gate":      label,
            "Value":     v_str,
            "Must be":   sym,
            "Threshold": t_str,
            "Status":    "✓  PASS" if passed else "✗  FAIL",
        })

    df = pd.DataFrame(rows)

    def _style_status(val: str) -> str:
        if "PASS" in val:
            return f"background-color:{T['green']}22; color:{T['green']}; font-weight:700;"
        if "FAIL" in val:
            return f"background-color:{T['red']}22; color:{T['red']}; font-weight:700;"
        return ""

    def _style_gate(val: str) -> str:
        return f"color:{T['text']}; font-weight:600;"

    def _style_val(val: str) -> str:
        return f"color:{T['gold']}; font-weight:600;"

    styled = (
        df.style
        .map(_style_status, subset=["Status"])
        .map(_style_gate,   subset=["Gate"])
        .map(_style_val,    subset=["Value"])
        .set_properties(**{"background-color": T["card"], "color": T["text2"],
                           "border-color": T["border"]})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Page sections
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(report) -> None:
    with st.sidebar:
        # ── Theme toggle ──
        st.caption("DISPLAY")
        prev_theme = st.session_state.get("theme_name", "dark")
        is_light   = st.toggle("☀  Light Mode", value=(prev_theme == "light"))
        new_theme  = "light" if is_light else "dark"
        if new_theme != prev_theme:
            st.session_state.theme_name = new_theme
            st.rerun()          # re-inject CSS immediately — no manual refresh needed

        st.divider()

        # ── Run info — what ran last ──
        # (this panel shows metadata from results/latest_report.json,
        #  written automatically after each `python src/run_eval.py` call)
        st.caption("RUN INFO")
        if report is None:
            st.warning("No report found.\nRun `python src/run_eval.py` first.")
        else:
            ts     = report.get("run_timestamp", "—")[:19].replace("T", " ")
            com    = report.get("commit_id", "—")
            n      = report.get("total_questions", "—")
            mode   = report.get("test_mode", False)
            f_list = report.get("failed_gates", [])
            p_list = report.get("gate_results", {}).get("passed_gates", [])
            total  = len(p_list) + len(f_list)

            st.markdown(f"🕐 **When:** {ts}")
            st.markdown(f"🔖 **Commit:** `{com}`")
            st.markdown(f"❓ **Questions:** {n}")
            st.markdown(f"⚡ **Mode:** {'Test (10 Qs)' if mode else 'Full (100 Qs)'}")
            st.markdown(f"✅ **Gates:** {len(p_list)} / {total} passed")

            st.divider()
            if f_list:
                st.error("Failed: " + ", ".join(f_list))
            else:
                st.success("All gates passed ✓")

        st.divider()

        # ── Refresh ──
        st.caption("Cache TTL: 30s")
        if st.button("⟳  Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_kpi_row(metrics: dict, gates: dict) -> None:
    st.html(_section_label("Key Metrics",
        "Live scores from the latest evaluation run"))

    kpi_defs = [
        # (metric_key, gate_key, label, fmt_fn, threshold_fmt, sublabel)
        ("hallucination_rate",  "hallucination_rate",
         "Hallucination Rate", lambda v: f"{v:.4f}", lambda t: f"≤ {t}", "lower is better"),
        ("answer_relevancy",    "answer_relevancy",
         "Answer Relevancy",   lambda v: f"{v:.4f}", lambda t: f"≥ {t}", "higher is better"),
        ("faithfulness",        "faithfulness",
         "Faithfulness",       lambda v: f"{v:.4f}", lambda t: f"≥ {t}", "higher is better"),
        ("context_precision",   "context_precision",
         "Context Precision",  lambda v: f"{v:.4f}", lambda t: f"≥ {t}", "higher is better"),
        ("latency_p95_seconds", "latency_p95",
         "Latency P95",        lambda v: f"{v:.3f}s", lambda t: f"≤ {t}s", "95th percentile"),
        ("cost_per_query_usd",  "cost_per_query",
         "Cost Per Query",     lambda v: f"${v:.4f}", lambda t: f"≤ ${t}", "blended GPT-4o rate"),
    ]

    cols = st.columns(6, gap="small")
    for col, (mkey, gkey, label, vfmt, tfmt, sub) in zip(cols, kpi_defs):
        val    = metrics.get(mkey)
        ginfo  = gates.get(gkey, {})
        passed = ginfo.get("passed", False)
        thr    = ginfo.get("threshold")
        v_str  = vfmt(val) if val is not None else "N/A"
        t_str  = tfmt(thr) if thr is not None else "—"
        col.html(_kpi_card(label, v_str, t_str, passed, sub))


def render_gate_section(gates: dict) -> None:
    st.html(_section_label("Quality Gates",
        "All thresholds configured in config.yaml"))

    col_pills, col_table = st.columns([1, 2], gap="large")

    with col_pills:
        st.caption("GATE STATUS")
        pills_html = "".join(_gate_pill(n, i.get("passed", False)) for n, i in gates.items())
        st.html(
            f'<div style="background:{T["card"]}; border:1px solid {T["border"]}; '
            f'border-radius:12px; padding:16px; line-height:2.2;">{pills_html}</div>'
        )

    with col_table:
        st.caption("THRESHOLD DETAIL")
        _gate_dataframe(gates)


def render_trend_charts(history: pd.DataFrame) -> None:
    st.html(_section_label("Trend Over Time",
        f"Historical metrics across {len(history)} evaluation run(s)"))

    if history.empty or len(history) < 1:
        st.info("Trend charts appear after the first evaluation run. "
                "Run `python src/run_eval.py` to generate data.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Faithfulness & Relevancy",
        "🔴  Hallucination Rate",
        "⚡  Latency P95",
        "💰  Cost Per Query",
    ])

    with tab1:
        fig = _trend_fig(
            history,
            series=[
                ("faithfulness",     T["green"],  "Faithfulness"),
                ("answer_relevancy", T["blue"],   "Answer Relevancy"),
                ("context_precision",T["purple"], "Context Precision"),
            ],
            title="Faithfulness, Relevancy & Precision",
            thresholds=[
                ("faithfulness min",    0.80, T["green"],  "dash"),
                ("relevancy min",       0.75, T["blue"],   "dot"),
                ("precision min",       0.60, T["purple"], "longdash"),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dashed lines show minimum thresholds · higher is better for all three")

    with tab2:
        fig = _trend_fig(
            history,
            series=[("hallucination_rate", T["red"], "Hallucination Rate")],
            title="Hallucination Rate  (1 − avg faithfulness)",
            thresholds=[("max allowed", 0.05, T["gold"], "dash")],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gold dashed line = max threshold (0.05) · lower is better")

    with tab3:
        if "latency_p95_seconds" in history.columns:
            fig = _bar_fig(
                history, "latency_p95_seconds",
                "Latency P95 (seconds)", 3.0, "max threshold", T["teal"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green bars = within 3.0 s SLA · Red bars = exceeded")

    with tab4:
        if "cost_per_query_usd" in history.columns:
            fig = _bar_fig(
                history, "cost_per_query_usd",
                "Cost Per Query (USD)", 0.02, "max threshold", T["orange"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green bars = within $0.02 budget · Red bars = exceeded")


def render_failures(failures: list) -> None:
    st.html(_section_label("Sample Failures",
        "Questions where faithfulness < 0.5"))

    if not failures:
        st.html(
            f'<div style="background:{_hex_rgba(T["green"],0.08)}; '
            f'border:1px solid {_hex_rgba(T["green"],0.3)}; '
            f'border-radius:12px; padding:16px 20px; color:{T["green"]}; font-weight:600;">'
            f'&#10003; &nbsp; No sample failures — all answers met the faithfulness threshold.'
            f'</div>'
        )
        return

    st.html(
        f'<div style="background:{_hex_rgba(T["red"],0.08)}; '
        f'border:1px solid {_hex_rgba(T["red"],0.3)}; '
        f'border-radius:10px; padding:12px 16px; color:{T["red"]}; font-weight:600; margin-bottom:12px;">'
        f'&#9888; &nbsp; {len(failures)} question(s) had low faithfulness scores</div>'
    )
    for i, f in enumerate(failures, 1):
        q     = f.get("question", "N/A")
        a     = f.get("answer", "N/A")
        faith = f.get("faithfulness")
        cat   = f.get("category", "—")
        err   = f.get("error")
        label = q[:85] + ("…" if len(q) > 85 else "")

        with st.expander(f"Failure {i} · [{cat}]  {label}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")
            c1, c2 = st.columns(2)
            c1.metric("Faithfulness", f"{faith:.4f}" if faith is not None else "N/A")
            c2.metric("Category", cat)
            if err:
                st.error(f"Evaluation error: {err}")


def render_history_table(history: pd.DataFrame) -> None:
    st.html(_section_label("All Evaluation Runs",
        f"{len(history)} run(s) stored in eval_history.db"))

    if history.empty:
        st.info("No run history yet. History accumulates with each evaluation.")
        return

    cols_wanted = [
        "run_timestamp", "commit_id", "total_questions",
        "hallucination_rate", "faithfulness", "answer_relevancy",
        "context_precision", "latency_p95_seconds", "cost_per_query_usd",
        "overall_result",
    ]
    show_cols = [c for c in cols_wanted if c in history.columns]
    disp = history[show_cols].copy().sort_values("run_timestamp", ascending=False)

    for fc in ["hallucination_rate", "faithfulness", "answer_relevancy",
               "context_precision", "latency_p95_seconds", "cost_per_query_usd"]:
        if fc in disp.columns:
            disp[fc] = disp[fc].round(4)

    disp = disp.rename(columns={
        "run_timestamp":       "Timestamp",
        "commit_id":           "Commit",
        "total_questions":     "Qs",
        "hallucination_rate":  "Hallucination",
        "faithfulness":        "Faithfulness",
        "answer_relevancy":    "Relevancy",
        "context_precision":   "Precision",
        "latency_p95_seconds": "Latency P95 (s)",
        "cost_per_query_usd":  "Cost/Q ($)",
        "overall_result":      "Result",
    })

    def _style_result(val):
        if val == "PASS":
            return f"background-color:{T['green']}22; color:{T['green']}; font-weight:700;"
        if val == "FAIL":
            return f"background-color:{T['red']}22; color:{T['red']}; font-weight:700;"
        return ""

    styled = (
        disp.style
        .map(_style_result, subset=["Result"])
        .set_properties(**{"background-color": T["card"], "color": T["text2"]})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Re-resolve theme in case toggle was just clicked
    global T
    T = DARK if st.session_state.get("theme_name", "dark") == "dark" else LIGHT

    inject_css(T)

    report  = load_report()
    history = load_history()

    render_sidebar(report)

    if report is None:
        st.html(_hero("—", "", "—", 0, False))
        st.warning(
            "No evaluation results found.  \n"
            "Run `python src/run_eval.py` from the project root to generate the first report."
        )
        return

    metrics      = report.get("metrics", {})
    gate_results = report.get("gate_results", {})
    gates        = gate_results.get("gates", {})
    failures     = report.get("sample_failures", [])

    # Hero banner
    st.html(_hero(
        report.get("overall_result", "—"),
        report.get("run_timestamp", ""),
        report.get("commit_id", "—"),
        report.get("total_questions", 0),
        report.get("test_mode", False),
    ))

    # KPI cards
    render_kpi_row(metrics, gates)

    st.divider()

    # Gate status
    render_gate_section(gates)

    st.divider()

    # Trend charts
    render_trend_charts(history)

    st.divider()

    # Sample failures
    render_failures(failures)

    st.divider()

    # Run history table
    render_history_table(history)

    # Footer
    st.html(
        f'<div style="text-align:center; padding:20px 0 8px; '
        f'color:{T["muted"]}; font-size:0.7rem; letter-spacing:0.5px;">'
        f'ASU LLM Evaluation Pipeline &middot; LangChain &middot; ChromaDB &middot; RAGAS &middot; Streamlit &middot; GPT-4o'
        f'</div>'
    )


main()
