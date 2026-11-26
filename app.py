"""
Streamlit Quant-Dashboard für die vektorisierte Monte-Carlo-Karrieresimulation
(Investment Banking / M&A, Quant Trading & Hedgefonds).

Dark-Mode Bloomberg/Quant-Ästhetik mit 4 interaktiven Analyse-Tabs:
- Tab 1: Übersicht & Strategie-Empfehlung (KPIs, 3-Pfeile-Portfolio, Stacked Bar)
- Tab 2: Abischnitt-Heatmap (1,3 bis 2,5) & Downside-Protection-Analyse
- Tab 3: Sankey-Flow & Finanz-Risiko (Total Comp, Schuldenlast, Risk-Return)
- Tab 4: Sensitivitätsanalyse (Tornado-Chart), Annahmen-Editor & CSV-Export
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ABITUR_SWEEP_GRADES,
    ApplicantProfile,
    OUTCOME_COLORS,
    OUTCOME_DROPOUT,
    OUTCOME_IB,
    OUTCOME_LABELS,
    OUTCOME_QUANT,
    OUTCOME_SHORT_LABELS,
    SimConfig,
    TRACK_IDS,
    TRACK_META,
    get_default_config,
)
from engine import (
    SimulationResult,
    TrackSimArrays,
    build_sankey_data,
    run_abitur_sweep,
    run_sensitivity_analysis,
    run_simulation,
)

# =============================================================================
# PAGE CONFIG & QUANT / BLOOMBERG DARK-MODE CSS
# =============================================================================

st.set_page_config(
    page_title="Quant Career Monte-Carlo Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_BG = "#0F172A"
PLOTLY_PAPER = "#111827"
PLOTLY_GRID = "#1E293B"
PLOTLY_FONT = "#E2E8F0"
MONO_FONT = "'JetBrains Mono', 'SF Mono', 'Fira Code', Menlo, monospace"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .stApp {{
        background-color: #0B0F19;
        color: #F3F4F6;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0F1523;
        border-right: 1px solid #1E293B;
    }}
    .terminal-header {{
        background: linear-gradient(90deg, #111827 0%, #0F172A 60%, #172033 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .terminal-title {{
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }}
    .terminal-sub {{
        font-size: 0.82rem;
        color: #94A3B8;
        margin: 2px 0 0 0;
    }}
    .terminal-pills {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .pill {{
        font-family: {MONO_FONT};
        font-size: 0.74rem;
        padding: 4px 10px;
        border-radius: 999px;
        background: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
    }}
    .pill-amber {{
        border-color: #F59E0B;
        color: #FCD34D;
        background: rgba(245, 158, 11, 0.12);
    }}
    .pill-cyan {{
        border-color: #00D4FF;
        color: #67E8F9;
        background: rgba(0, 212, 255, 0.10);
    }}
    .pill-green {{
        border-color: #10B981;
        color: #6EE7B7;
        background: rgba(16, 185, 129, 0.12);
    }}
    .pill-red {{
        border-color: #EF4444;
        color: #FCA5A5;
        background: rgba(239, 68, 68, 0.12);
    }}
    .kpi-card {{
        background: #111827;
        border: 1px solid #1E293B;
        border-top: 3px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
    }}
    .kpi-card.cyan {{ border-top-color: #00D4FF; }}
    .kpi-card.green {{ border-top-color: #10B981; }}
    .kpi-card.purple {{ border-top-color: #A855F7; }}
    .kpi-card.rose {{ border-top-color: #FB7185; }}
    .kpi-label {{
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-family: {MONO_FONT};
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.15;
    }}
    .kpi-value-sm {{
        font-family: {MONO_FONT};
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.25;
    }}
    .kpi-delta {{
        font-family: {MONO_FONT};
        font-size: 0.76rem;
        color: #10B981;
        margin-top: 6px;
    }}
    .kpi-sub {{
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 5px;
    }}
    .strategy-box {{
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid #1E293B;
        border-left: 4px solid #00D4FF;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0 18px 0;
        font-size: 0.89rem;
        line-height: 1.55;
        color: #E2E8F0;
    }}
    .sidebar-badge {{
        font-family: {MONO_FONT};
        font-size: 0.75rem;
        padding: 6px 10px;
        border-radius: 6px;
        margin-top: 4px;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid #1E293B;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #111827;
        border: 1px solid #1E293B;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.86rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #1E293B !important;
        color: #F59E0B !important;
        border-color: #F59E0B !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE & CACHED SIMULATION WRAPPERS
# =============================================================================

if "sim_config" not in st.session_state:
    st.session_state.sim_config = get_default_config()
if "config_modified" not in st.session_state:
    st.session_state.config_modified = False


def _config_fingerprint(cfg: SimConfig) -> str:
    payload = {
        "tracks": {k: asdict(v) for k, v in cfg.tracks.items()},
        "profile_multipliers": cfg.profile_multipliers,
        "scholarship_params": cfg.scholarship_params,
        "correlation_params": cfg.correlation_params,
        "outcome_comp_eur": cfg.outcome_comp_eur,
        "score_weights": cfg.score_weights,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _profile_key(profile: ApplicantProfile) -> str:
    return json.dumps(asdict(profile), sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_run_simulation(
    profile_json: str,
    n_runs: int,
    seed: int,
    cfg_hash: str,
    _cfg: SimConfig,
) -> SimulationResult:
    profile = ApplicantProfile(**json.loads(profile_json))
    return run_simulation(profile=profile, n_runs=n_runs, seed=seed, cfg=_cfg, keep_arrays=True)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_run_sweep(
    profile_json: str,
    n_runs: int,
    seed: int,
    cfg_hash: str,
    _cfg: SimConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profile = ApplicantProfile(**json.loads(profile_json))
    return run_abitur_sweep(profile=profile, n_runs=n_runs, seed=seed, cfg=_cfg)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_run_sensitivity(
    profile_json: str,
    target_scope: str,
    n_runs: int,
    seed: int,
    cfg_hash: str,
    _cfg: SimConfig,
) -> pd.DataFrame:
    profile = ApplicantProfile(**json.loads(profile_json))
    return run_sensitivity_analysis(
        profile=profile,
        target_scope=target_scope,
        n_runs=n_runs,
        seed=seed,
        cfg=_cfg,
    )


def _fmt_eur(val: float) -> str:
    return f"{val:,.0f} €".replace(",", ".")


def _fmt_pct(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f} %".replace(".", ",")


def _apply_dark_layout(fig: go.Figure, height: int = 430, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC", family="Inter, sans-serif"), x=0.01),
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_FONT, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=24, t=52, b=46),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(color="#F8FAFC", family=MONO_FONT, size=12)),
    )
    return fig


# =============================================================================
# SIDEBAR: BEWERBER-PROFIL & ENGINE-STEUERUNG
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="border-bottom:1px solid #1E293B; padding-bottom:10px; margin-bottom:12px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#F59E0B; letter-spacing:0.1em;">
                QUANT CAREER TERMINAL // v1.0
            </div>
            <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-top:2px;">
                Bewerber-Profil & Parameter
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_mode = st.radio(
        "Simulations-Modus",
        options=["Einzel-Schnitt", "Sweep (1,3 bis 2,5)"],
        index=0,
        horizontal=True,
        help="Einzel-Schnitt analysiert einen exakten Abischnitt im Detail. Der Sweep vergleicht alle 13 Schnitte (1,3 bis 2,5 in 0,1er-Schritten).",
    )

    if sim_mode == "Einzel-Schnitt":
        abitur_grade = st.slider(
            "Abiturschnitt",
            min_value=1.3,
            max_value=2.5,
            value=1.5,
            step=0.1,
            format="%.1f",
        )
    else:
        st.caption("Sweep aktiv: Alle 13 Abischnitte (1,3 bis 2,5) werden berechnet.")
        abitur_grade = st.slider(
            "Fokus-Abiturschnitt (für Detail-Sankey & Tornado)",
            min_value=1.3,
            max_value=2.5,
            value=1.5,
            step=0.1,
            format="%.1f",
        )

    st.markdown("##### Fachnoten (Abiturpunkte 0–15)")
    col_m, col_p = st.columns(2)
    with col_m:
        math_points = st.slider("Mathe (Pkt.)", 0, 15, 13, 1)
    with col_p:
        physics_points = st.slider("Physik (Pkt.)", 0, 15, 13, 1)
    language_points = st.slider(
        "Sprache / Deutsch-Englisch (Pkt., für ETH-Regel)",
        0,
        15,
        11,
        1,
        help="ETH Zürich prüfungsfrei: Durchschnitt aus Mathe + Physik + Sprache >= 10,5 Punkte (70 %).",
    )

    eth_avg = (math_points + physics_points + language_points) / 3.0
    eth_pct = (eth_avg / 15.0) * 100.0
    if eth_avg >= 10.5:
        st.markdown(
            f'<div class="sidebar-badge" style="background:rgba(16,185,129,0.12);border-color:#10B981;color:#6EE7B7;">'
            f'ETH-Regel: {eth_avg:.1f}/15 Pkt ({eth_pct:.0f} %) — Prüfungsfrei</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="sidebar-badge" style="background:rgba(239,68,68,0.12);border-color:#EF4444;color:#FCA5A5;">'
            f'ETH-Regel: {eth_avg:.1f}/15 Pkt ({eth_pct:.0f} %) — Aufnahmeprüfung</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Standardisierte Tests")
    sat_score = st.slider("SAT-Score", 1200, 1600, 1540, 10)
    act_score = st.slider("ACT-Score", 26, 36, 35, 1)
    hsg_percentile = st.slider("HSG-Eignungstest-Perzentil (%)", 0, 100, 80, 1)

    st.markdown("##### Resilienz & Studienerfolg")
    discipline = st.slider(
        "Lern-Disziplin & Quant-Resilienz",
        min_value=0.10,
        max_value=1.00,
        value=0.75,
        step=0.05,
        format="%.2f",
        help="Steuert massiv die Abbrecherquote im Physik-Grundstudium & ETH-Basisprüfung, Uni-GPA und Praktika-Akquise.",
    )

    st.markdown("##### Profil-Faktoren")
    has_ma_internship = st.checkbox("M&A-Schülerpraktikum (Boutique / Eichenfels)", value=True)
    has_jugend_gruendet = st.checkbox("Top 3 % Jugend gründet", value=True)
    has_tradity = st.checkbox("Tradity e.V. (Finanzbildung an Schule)", value=True)
    is_first_gen = st.checkbox("First-Gen-Student (erster Akademiker)", value=True)
    has_gdb_100 = st.checkbox("GdB 100 (Ohrmuscheldysplasie 3. Grades)", value=True)

    st.markdown("##### Monte-Carlo-Engine")
    n_runs = st.slider(
        "Simulationsanzahl (Runs pro Track)",
        min_value=20_000,
        max_value=250_000,
        value=100_000,
        step=10_000,
        format="%d",
    )
    seed = int(st.number_input("Random Seed (Reproduzierbarkeit)", min_value=0, max_value=999_999, value=42, step=1))

profile = ApplicantProfile(
    abitur_grade=round(float(abitur_grade), 1),
    math_points=int(math_points),
    physics_points=int(physics_points),
    language_points=int(language_points),
    sat_score=int(sat_score),
    act_score=int(act_score),
    hsg_percentile=int(hsg_percentile),
    discipline=float(discipline),
    has_ma_internship=bool(has_ma_internship),
    has_jugend_gruendet=bool(has_jugend_gruendet),
    has_tradity=bool(has_tradity),
    is_first_gen=bool(is_first_gen),
    has_gdb_100=bool(has_gdb_100),
)

cfg: SimConfig = st.session_state.sim_config
cfg_hash = _config_fingerprint(cfg)
p_json = _profile_key(profile)

# Hauptsimulation ausführen (gecached, ~0,18 s für 100.000 Runs)
with st.spinner(f"Simuliere {n_runs:,} vektorisierte Karriereverläufe ...".replace(",", ".")):
    sim_res: SimulationResult = cached_run_simulation(p_json, n_runs, seed, cfg_hash, cfg)

# Abischnitt-Sweep (1,3 bis 2,5) wird für Tab 2 und Sweep-Modus immer bereitgestellt (gecached)
sweep_runs = n_runs if n_runs <= 100_000 else 100_000
with st.spinner(f"Berechne Abischnitt-Sweep 1,3–2,5 (13 × {sweep_runs:,} Runs) ...".replace(",", ".")):
    track_sweep_df, port_sweep_df = cached_run_sweep(p_json, sweep_runs, seed, cfg_hash, cfg)


# =============================================================================
# TOP TERMINAL HEADER
# =============================================================================

override_pill = (
    '<span class="pill pill-red">CUSTOM CONFIG AKTIV</span>'
    if st.session_state.config_modified
    else '<span class="pill">CONFIG: DEFAULT</span>'
)

st.markdown(
    f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Monte-Carlo Karrieresimulation: Tier-1 IB, Quant Trading & Hedgefonds</div>
            <div class="terminal-sub">
                Vektorisierte 5-Stufen-Pipeline (Zulassung → Finanzierung → Abschluss/GPA → Top-Master → Karriere-Outcome) über 6 Uni-Tracks & 20 3-Pfeile-Portfolios
            </div>
        </div>
        <div class="terminal-pills">
            <span class="pill pill-amber">ABI {profile.abitur_grade:.1f} {'(SWEEP)' if sim_mode != 'Einzel-Schnitt' else ''}</span>
            <span class="pill pill-cyan">{n_runs:,} RUNS / TRACK</span>
            <span class="pill pill-green">STIPENDIUM: {sim_res.scholarship_prob * 100:.1f} %</span>
            <span class="pill">ENGINE: {sim_res.elapsed_seconds * 1000:.0f} ms</span>
            {override_pill}
        </div>
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)


# =============================================================================
# 4 PFLICHT-TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Tab 1 · Übersicht & Strategie-Empfehlung",
    "Tab 2 · Abischnitt-Heatmap (1,3–2,5)",
    "Tab 3 · Sankey-Flow & Finanz-Risiko",
    "Tab 4 · Sensitivitätsanalyse & Annahmen-Editor",
])


# =============================================================================
# TAB 1: ÜBERSICHT & STRATEGIE-EMPFEHLUNG
# =============================================================================
with tab1:
    bp = sim_res.best_portfolio
    bs = sim_res.track_summaries[sim_res.best_single_track_id]
    delta_pp = (bp.target_ib_quant_prob - bs.target_ib_quant_prob) * 100.0
    best_ra_port = max(sim_res.portfolios, key=lambda x: x.risk_adjusted_score)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Ziel-Chance (Tier-1 IB + Quant/HF)</div>
                <div class="kpi-value">{_fmt_pct(bp.target_ib_quant_prob * 100)}</div>
                <div class="kpi-delta">+{delta_pp:.1f} %-Pkt. vs. bestem Single-Track</div>
                <div class="kpi-sub">IB: {_fmt_pct(bp.ib_prob * 100)} · Quant: {_fmt_pct(bp.quant_prob * 100)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card cyan">
                <div class="kpi-label">Bester Single-Track</div>
                <div class="kpi-value-sm">{TRACK_META[bs.track_id]['short_name']}</div>
                <div class="kpi-delta" style="color:#00D4FF;">{_fmt_pct(bs.target_ib_quant_prob * 100)} Ziel-Chance</div>
                <div class="kpi-sub">Score: {bs.risk_adjusted_score:.1f}/100 · Abbruch: {_fmt_pct(bs.outcome_probs[OUTCOME_LABELS[OUTCOME_DROPOUT]] * 100)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card green">
                <div class="kpi-label">Bester 3-Pfeile-Portfolio-Mix</div>
                <div class="kpi-value-sm">{bp.label}</div>
                <div class="kpi-delta">Abdeckung: {_fmt_pct(bp.coverage_rate * 100)} zugel. & fin.</div>
                <div class="kpi-sub">Risk-Adj. Score: {bp.risk_adjusted_score:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card purple">
                <div class="kpi-label">Median-Schuldenlast (Portfolio)</div>
                <div class="kpi-value">{_fmt_eur(bp.median_debt_eur)}</div>
                <div class="kpi-sub">Single-Best ({TRACK_META[bs.track_id]['short_name']}): {_fmt_eur(bs.median_debt_eur)}</div>
                <div class="kpi-sub">UK Post-Brexit Absolv.: {_fmt_eur(sim_res.track_summaries['uk_top'].median_debt_grad_eur)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card rose">
                <div class="kpi-label">Median-Einstiegsgehalt (J1)</div>
                <div class="kpi-value">{_fmt_eur(bp.median_comp_eur)}</div>
                <div class="kpi-delta" style="color:#FCD34D;">P90 Upside: {_fmt_eur(bp.p90_comp_eur)}</div>
                <div class="kpi-sub">Brutto Total Comp, EUR-normalisiert</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dynamisches Strategie-Briefing
    us_s = sim_res.track_summaries["us_elite"]
    uk_s = sim_res.track_summaries["uk_top"]
    phys_s = sim_res.track_summaries["de_physik"]
    bwl_s = sim_res.track_summaries["de_bwl_wima"]
    usage_str = " | ".join(
        f"<b>{TRACK_META[tid]['short_name']}</b>: {share * 100:.1f} %"
        for tid, share in bp.track_usage_shares.items()
    )
    st.markdown(
        f"""
        <div class="strategy-box">
            <b>Strategie-Empfehlung (Abiturschnitt {profile.abitur_grade:.1f}, Disziplin {profile.discipline:.2f}):</b>
            Die parallele 3-Pfeile-Bewerbung auf <b>{bp.label}</b> steigert die Tier-1-IB- & Quant/Hedgefonds-Chance
            von <b>{_fmt_pct(bs.target_ib_quant_prob * 100)}</b> (bester Einzel-Track) auf <b>{_fmt_pct(bp.target_ib_quant_prob * 100)}</b>
            bei <b>{_fmt_pct(bp.coverage_rate * 100)}</b> Zulassungs- & Finanzierungsabdeckung.
            Tatsächliche Einschreibungs-Verteilung im Portfolio: {usage_str}.<br>
            <span style="color:#94A3B8;">
            • <b>Upside vs. Bottlenecks:</b> US Elite ({_fmt_pct(us_s.admit_rate * 100)} Zulassung, {_fmt_pct(us_s.finance_rate_cond * 100)} Full-Aid-Finanzierung) wirkt als kostenloses High-Upside-Ticket.
            UK Top-Unis scheitern im Post-Brexit-Bachelor zu {_fmt_pct((1.0 - uk_s.finance_rate_cond) * 100)} an den ~35–45k £/Jahr Gebühren (besser als 1-J-Master nach DE-Bachelor!).
            DE Top-Physik ({_fmt_pct(phys_s.target_ib_quant_prob * 100)} Ziel-Chance, 100 % NC-frei) schützt gegen NC-/Stipendienrisiken;
            höchster Risk-Adjusted Score im Portfolio-Set: <b>{best_ra_port.label}</b> ({best_ra_port.risk_adjusted_score:.1f}/100, Abbruch nur {_fmt_pct(best_ra_port.dropout_prob * 100)}).
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gestapeltes Balkendiagramm: Outcome-Verteilung pro Track + Optimaler 3-Pfeile-Mix
    bar_labels = [TRACK_META[tid]["short_name"] for tid in TRACK_IDS] + ["★ 3-Pfeile-Mix"]
    fig_stack = go.Figure()

    for idx, outcome_lbl in enumerate(OUTCOME_LABELS):
        y_vals = [sim_res.track_summaries[tid].outcome_probs[outcome_lbl] * 100.0 for tid in TRACK_IDS]
        y_vals.append(bp.outcome_probs[outcome_lbl] * 100.0)
        fig_stack.add_trace(
            go.Bar(
                name=OUTCOME_SHORT_LABELS[idx],
                x=bar_labels,
                y=y_vals,
                marker_color=OUTCOME_COLORS[outcome_lbl],
                text=[f"{v:.1f}%" if v >= 4.5 else "" for v in y_vals],
                textposition="inside",
                textfont=dict(color="#0B0F19" if idx in (0, 1) else "#F8FAFC", size=11, family=MONO_FONT),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{outcome_lbl}: <b>%{{y:.2f}} %</b><extra></extra>"
                ),
            )
        )

    fig_stack.update_layout(
        barmode="stack",
        yaxis=dict(title="Wahrscheinlichkeit (%)", range=[0, 100], gridcolor=PLOTLY_GRID, ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11)),
    )
    _apply_dark_layout(
        fig_stack,
        height=440,
        title=f"Karriere-Outcome-Verteilung (100 % normiert) — Einzel-Tracks vs. Optimaler 3-Pfeile-Mix (Abi {profile.abitur_grade:.1f})",
    )
    st.plotly_chart(fig_stack, width="stretch", theme=None)

    # Falls Sweep-Modus aktiv: zusätzliche kompakte Sweep-Strategie-Tabelle direkt in Tab 1
    if sim_mode == "Sweep (1,3 bis 2,5)":
        st.markdown("##### Sweep-Übersicht: Optimaler Single-Track & 3-Pfeile-Mix über alle Abischnitte (1,3–2,5)")
        st.dataframe(port_sweep_df, width="stretch", hide_index=True)

    col_tbl1, col_tbl2 = st.columns([1.35, 1.0])
    with col_tbl1:
        st.markdown("##### Detail-Kennzahlen aller 6 Uni-Tracks")
        display_cols = [
            "Track", "Zulassung (%)", "Finanzierung bed. (%)", "Abschluss bed. (%)",
            "Top-Master bed. (%)", "Tier-1 IB / M&A (%)", "Top Quant / HF (%)",
            "Ziel gesamt (IB+Quant %)", "Median Schulden (EUR)", "Median Gehalt J1 (EUR)",
            "Risk-Adjusted Score",
        ]
        st.dataframe(sim_res.summary_df[display_cols], width="stretch", hide_index=True)

    with col_tbl2:
        st.markdown("##### Top-5 Strategien: 3-Pfeile-Portfolio-Mix")
        st.dataframe(sim_res.portfolio_df.head(5), width="stretch", hide_index=True)


# =============================================================================
# TAB 2: ABISCHNITT-HEATMAP (1,3 BIS 2,5) & DOWNSIDE PROTECTION
# =============================================================================
with tab2:
    c_opt1, c_opt2 = st.columns([2, 1])
    with c_opt1:
        heat_metric = st.selectbox(
            "Heatmap-Kennzahl",
            options=[
                "Ziel gesamt (IB+Quant %)",
                "Tier-1 IB / M&A (%)",
                "Top Quant / HF (%)",
                "Risk-Adjusted Score",
                "Abbruch/Stopp (%)",
            ],
            index=0,
        )
    with c_opt2:
        include_portfolio_row = st.checkbox("Optimalen 3-Pfeile-Mix als oberste Zeile anzeigen", value=True)

    # Heatmap-Matrix konstruieren
    grades_list = [round(float(g), 1) for g in ABITUR_SWEEP_GRADES]
    x_labels = [f"{g:.1f}".replace(".", ",") for g in grades_list]
    y_track_ids = list(TRACK_IDS)
    y_labels = [TRACK_META[tid]["short_name"] for tid in y_track_ids]

    pivot_df = track_sweep_df.pivot(index="track_id", columns="Abiturschnitt", values=heat_metric).loc[y_track_ids, grades_list]
    z_matrix = pivot_df.values.tolist()

    if include_portfolio_row:
        port_col_map = {
            "Ziel gesamt (IB+Quant %)": "Portfolio Ziel (IB+Quant %)",
            "Tier-1 IB / M&A (%)": "Portfolio Tier-1 IB (%)",
            "Top Quant / HF (%)": "Portfolio Quant/HF (%)",
            "Risk-Adjusted Score": "Portfolio Risk-Adjusted Score",
            "Abbruch/Stopp (%)": "Portfolio Abbruch (%)",
        }
        p_vals = port_sweep_df.set_index("Abiturschnitt").loc[grades_list, port_col_map[heat_metric]].tolist()
        z_matrix = z_matrix + [p_vals]
        y_labels = y_labels + ["★ 3-Pfeile-Mix"]

    is_risk_metric = heat_metric == "Abbruch/Stopp (%)"
    colorscale = (
        [[0.0, "#064E3B"], [0.45, "#B45309"], [1.0, "#DC2626"]]
        if is_risk_metric
        else [[0.0, "#0F172A"], [0.25, "#1E3A5F"], [0.55, "#0284C7"], [0.80, "#10B981"], [1.0, "#F59E0B"]]
    )
    unit = "" if "Score" in heat_metric else "%"
    text_matrix = [[f"{v:.1f}{unit}" for v in row] for row in z_matrix]

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, family=MONO_FONT, color="#F8FAFC"),
            colorscale=colorscale,
            colorbar=dict(title=dict(text=heat_metric, font=dict(size=11)), ticksuffix=unit),
            hovertemplate="Track: <b>%{y}</b><br>Abiturschnitt: <b>%{x}</b><br>"
                          + f"{heat_metric}: <b>%{{z:.2f}}{unit}</b><extra></extra>",
        )
    )

    # Vertikale Markierung des aktuell gewählten Fokus-Abiturschnitts
    sel_idx = grades_list.index(profile.abitur_grade) if profile.abitur_grade in grades_list else None
    if sel_idx is not None:
        fig_heat.add_shape(
            type="rect",
            x0=sel_idx - 0.5,
            x1=sel_idx + 0.5,
            y0=-0.5,
            y1=len(y_labels) - 0.5,
            line=dict(color="#F59E0B", width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )

    fig_heat.update_layout(
        xaxis=dict(title="Abiturschnitt (0,1er-Schritte)", side="bottom"),
        yaxis=dict(title="", automargin=True),
    )
    _apply_dark_layout(
        fig_heat,
        height=430,
        title=f"Abischnitt-Heatmap (1,3 bis 2,5): {heat_metric} — Gelber Rahmen = gewählter Schnitt ({profile.abitur_grade:.1f})",
    )
    st.plotly_chart(fig_heat, width="stretch", theme=None)

    # Downside-Protection-Liniendiagramm: Physik-Pfad vs. BWL-Pfade
    phys_series = track_sweep_df[track_sweep_df["track_id"] == "de_physik"].sort_values("Abiturschnitt")
    bwl_series = track_sweep_df[track_sweep_df["track_id"] == "de_bwl_wima"].sort_values("Abiturschnitt")
    hsg_series = track_sweep_df[track_sweep_df["track_id"] == "hsg"].sort_values("Abiturschnitt")
    eth_series = track_sweep_df[track_sweep_df["track_id"] == "eth_physik"].sort_values("Abiturschnitt")
    us_series = track_sweep_df[track_sweep_df["track_id"] == "us_elite"].sort_values("Abiturschnitt")
    port_series = port_sweep_df.sort_values("Abiturschnitt")

    phys_25 = float(phys_series[phys_series["Abiturschnitt"] == 2.5]["Ziel gesamt (IB+Quant %)"].iloc[0])
    bwl_25 = float(bwl_series[bwl_series["Abiturschnitt"] == 2.5]["Ziel gesamt (IB+Quant %)"].iloc[0])
    bwl_13 = float(bwl_series[bwl_series["Abiturschnitt"] == 1.3]["Ziel gesamt (IB+Quant %)"].iloc[0])
    phys_13 = float(phys_series[phys_series["Abiturschnitt"] == 1.3]["Ziel gesamt (IB+Quant %)"].iloc[0])
    spread_25 = phys_25 - bwl_25

    fig_line = go.Figure()

    # Portfolio-Envelope (gestrichelt oben)
    fig_line.add_trace(
        go.Scatter(
            x=port_series["Abiturschnitt"],
            y=port_series["Portfolio Ziel (IB+Quant %)"],
            mode="lines+markers",
            name="★ Optimaler 3-Pfeile-Mix",
            line=dict(color="#F8FAFC", width=2.5, dash="dash"),
            marker=dict(size=6),
            hovertemplate="3-Pfeile-Mix | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )
    # DE Top-Physik (Downside Protection Floor)
    fig_line.add_trace(
        go.Scatter(
            x=phys_series["Abiturschnitt"],
            y=phys_series["Ziel gesamt (IB+Quant %)"],
            mode="lines+markers",
            name="DE Top-Physik -> Master (Downside Floor)",
            line=dict(color="#10B981", width=3.5),
            marker=dict(size=8),
            hovertemplate="DE Top-Physik | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )
    # DE BWL / WiMa
    fig_line.add_trace(
        go.Scatter(
            x=bwl_series["Abiturschnitt"],
            y=bwl_series["Ziel gesamt (IB+Quant %)"],
            mode="lines+markers",
            name="DE Top-BWL / WiMa (Mannheim / WHU / FS)",
            line=dict(color="#A855F7", width=3.2),
            marker=dict(size=7),
            hovertemplate="DE BWL/WiMa | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )
    # HSG
    fig_line.add_trace(
        go.Scatter(
            x=hsg_series["Abiturschnitt"],
            y=hsg_series["Ziel gesamt (IB+Quant %)"],
            mode="lines+markers",
            name="HSG St. Gallen",
            line=dict(color="#F59E0B", width=2.2),
            marker=dict(size=5),
            hovertemplate="HSG St. Gallen | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )
    # ETH
    fig_line.add_trace(
        go.Scatter(
            x=eth_series["Abiturschnitt"],
            y=eth_series["Ziel gesamt (IB+Quant %)"],
            mode="lines+markers",
            name="ETH Zürich (Physik)",
            line=dict(color="#38BDF8", width=2.2),
            marker=dict(size=5),
            hovertemplate="ETH Zürich | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )
    # US Elite
    fig_line.add_trace(
        go.Scatter(
            x=us_series["Abiturschnitt"],
            y=us_series["Ziel gesamt (IB+Quant %)"],
            mode="lines+markers",
            name="US Elite (Need-Blind)",
            line=dict(color="#00D4FF", width=1.8, dash="dot"),
            marker=dict(size=5),
            hovertemplate="US Elite | Abi %{x:.1f}: <b>%{y:.2f} %</b><extra></extra>",
        )
    )

    fig_line.add_vline(
        x=profile.abitur_grade,
        line_width=1.5,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text=f"Fokus: {profile.abitur_grade:.1f}",
        annotation_font=dict(color="#FCD34D", family=MONO_FONT, size=11),
    )

    fig_line.update_layout(
        xaxis=dict(title="Abiturschnitt", dtick=0.1, gridcolor=PLOTLY_GRID),
        yaxis=dict(title="Tier-1 IB + Top Quant / HF (%)", gridcolor=PLOTLY_GRID, ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11)),
    )
    _apply_dark_layout(
        fig_line,
        height=430,
        title="Downside Protection: DE Top-Physik (NC-frei + Top-Master) vs. BWL- & Elite-Pfade über Abischnitt 1,3–2,5",
    )
    st.plotly_chart(fig_line, width="stretch", theme=None)

    dp_color = "#10B981" if spread_25 >= 0 else "#F59E0B"
    st.markdown(
        f"""
        <div class="strategy-box" style="border-left-color:{dp_color};">
            <b>Downside-Protection-Befund (Physik vs. BWL):</b>
            Während <b>DE Top-BWL / WiMa</b> zwischen Abischnitt 1,3 (<b>{bwl_13:.1f} %</b>) und 2,5 (<b>{bwl_25:.1f} %</b>)
            um <b>{bwl_13 - bwl_25:.1f} %-Punkte</b> einbricht (NC-Druck, geringere Stipendienquote & deutsches IB-CV-Screening),
            bleibt <b>DE Top-Physik → Top-Master</b> praktisch invariant zwischen <b>{phys_13:.1f} %</b> und <b>{phys_25:.1f} %</b>
            (Veränderung nur {abs(phys_13 - phys_25):.1f} %-Punkte). Bei Abischnitt 2,5 liefert der Physik-Pfad einen
            <b>Downside-Protection-Vorteil von {spread_25:+.1f} %-Punkten</b> gegenüber reinem DE-BWL/WiMa!
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 3: SANKEY-FLOW & FINANZ-RISIKO
# =============================================================================
with tab3:
    sankey_options = {
        "portfolio_best": f"★ Optimaler 3-Pfeile-Mix ({sim_res.best_portfolio.label})",
        **{tid: TRACK_META[tid]["name"] for tid in TRACK_IDS},
    }
    col_sk1, col_sk2 = st.columns([2, 1])
    with col_sk1:
        selected_sankey_key = st.selectbox(
            "Pipeline-Pfad für Sankey-Flussdiagramm",
            options=list(sankey_options.keys()),
            format_func=lambda k: sankey_options[k],
            index=0,
        )
    with col_sk2:
        fin_filter = st.radio(
            "Finanz-Risiko Stichprobe",
            options=["Alle Runs", "Nur Absolventen"],
            index=0,
            horizontal=True,
        )

    if selected_sankey_key == "portfolio_best":
        sk_arr: TrackSimArrays = sim_res.best_portfolio_arrays
        sk_src_label = f"3-Pfeile-Mix ({n_runs:,})".replace(",", ".")
    else:
        sk_arr = sim_res.track_arrays[selected_sankey_key]
        sk_src_label = f"{TRACK_META[selected_sankey_key]['short_name']} ({n_runs:,})".replace(",", ".")

    sk = build_sankey_data(sk_arr, sk_src_label)
    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="#0B0F19", width=1.2),
                    label=sk["node_labels"],
                    color=sk["node_colors"],
                    hovertemplate="<b>%{label}</b><br>Runs: %{value:,.0f}<extra></extra>",
                ),
                link=dict(
                    source=sk["sources"],
                    target=sk["targets"],
                    value=sk["values"],
                    color=sk["link_colors"],
                    hovertemplate="%{source.label} → %{target.label}<br><b>%{value:,.0f} Runs</b><extra></extra>",
                ),
            )
        ]
    )
    _apply_dark_layout(
        fig_sankey,
        height=440,
        title=f"5-Stufen Karriere-Flussdiagramm (N = {n_runs:,} Runs, Abi {profile.abitur_grade:.1f}) — {sankey_options[selected_sankey_key]}".replace(",", "."),
    )
    st.plotly_chart(fig_sankey, width="stretch", theme=None)

    # Finanz-Risiko: Exakte Boxplots über alle 100k+ Runs (P5/P25/Med/P75/P95/Mean) + Verteilungs-Histogramm
    fin_tracks: List[Tuple[str, str, TrackSimArrays, str]] = [
        (tid, TRACK_META[tid]["short_name"], sim_res.track_arrays[tid], TRACK_META[tid]["color"])
        for tid in TRACK_IDS
    ] + [
        ("portfolio_best", "★ 3-Pfeile-Mix", sim_res.best_portfolio_arrays, "#F8FAFC")
    ]

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        fig_comp = go.Figure()
        for _, lbl, arr, color in fin_tracks:
            vals = arr.comp_eur[arr.graduated] if (fin_filter == "Nur Absolventen" and np.any(arr.graduated)) else arr.comp_eur
            p5, q1, med, q3, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
            mean_v = float(np.mean(vals, dtype=np.float64))
            fig_comp.add_trace(
                go.Box(
                    name=lbl,
                    q1=[float(q1)],
                    median=[float(med)],
                    q3=[float(q3)],
                    lowerfence=[float(p5)],
                    upperfence=[float(p95)],
                    mean=[mean_v],
                    marker_color=color,
                    line=dict(color=color, width=2),
                    fillcolor="rgba(15, 23, 42, 0.7)",
                    hoverinfo="y+name",
                )
            )
        fig_comp.update_layout(
            showlegend=False,
            yaxis=dict(title="Total Comp Jahr 1 (EUR)", gridcolor=PLOTLY_GRID, tickformat=",.0f", ticksuffix=" €"),
            xaxis=dict(title=""),
        )
        _apply_dark_layout(fig_comp, height=390, title=f"Brutto-Einstiegsgehalt J1 (P5–P95 Boxplot · {fin_filter})")
        st.plotly_chart(fig_comp, width="stretch", theme=None)

    with col_f2:
        fig_debt = go.Figure()
        for _, lbl, arr, color in fin_tracks:
            vals = arr.debt_eur[arr.graduated] if (fin_filter == "Nur Absolventen" and np.any(arr.graduated)) else arr.debt_eur
            p5, q1, med, q3, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
            mean_v = float(np.mean(vals, dtype=np.float64))
            fig_debt.add_trace(
                go.Box(
                    name=lbl,
                    q1=[float(q1)],
                    median=[float(med)],
                    q3=[float(q3)],
                    lowerfence=[float(p5)],
                    upperfence=[float(p95)],
                    mean=[mean_v],
                    marker_color=color,
                    line=dict(color=color, width=2),
                    fillcolor="rgba(15, 23, 42, 0.7)",
                    hoverinfo="y+name",
                )
            )
        fig_debt.update_layout(
            showlegend=False,
            yaxis=dict(title="Gesamtschuldenlast (EUR)", gridcolor=PLOTLY_GRID, tickformat=",.0f", ticksuffix=" €"),
            xaxis=dict(title=""),
        )
        _apply_dark_layout(fig_debt, height=390, title=f"Schuldenlast bei Berufseinstieg (P5–P95 Boxplot · {fin_filter})")
        st.plotly_chart(fig_debt, width="stretch", theme=None)

    # Histogramme für den ausgewählten Sankey-Track + Risk-Return-Matrix
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        comp_sel = sk_arr.comp_eur[sk_arr.graduated] if (fin_filter == "Nur Absolventen" and np.any(sk_arr.graduated)) else sk_arr.comp_eur
        # Exaktes NumPy-Histogramm über ALLE n_runs (null Browser-Lag!)
        c_max = float(np.percentile(comp_sel, 99.2))
        counts_c, edges_c = np.histogram(np.clip(comp_sel, 0.0, c_max), bins=36)
        mids_c = 0.5 * (edges_c[:-1] + edges_c[1:])
        pct_c = (counts_c / max(1, len(comp_sel))) * 100.0
        fig_hc = go.Figure(
            data=[
                go.Bar(
                    x=mids_c,
                    y=pct_c,
                    marker_color="#00D4FF",
                    opacity=0.82,
                    hovertemplate="Gehalt ~%{x:,.0f} €<br>Anteil: <b>%{y:.2f} %</b><extra></extra>",
                )
            ]
        )
        med_c_sel = float(np.median(comp_sel))
        fig_hc.add_vline(
            x=med_c_sel,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Median: {_fmt_eur(med_c_sel)}",
            annotation_font=dict(color="#FCD34D", family=MONO_FONT, size=11),
        )
        fig_hc.update_layout(
            xaxis=dict(title="Total Compensation Jahr 1 (EUR)", tickformat=",.0f", ticksuffix=" €", gridcolor=PLOTLY_GRID),
            yaxis=dict(title="Häufigkeit (%)", ticksuffix="%", gridcolor=PLOTLY_GRID),
            bargap=0.05,
        )
        _apply_dark_layout(fig_hc, height=340, title=f"Gehalts-Histogramm: {sankey_options[selected_sankey_key]}")
        st.plotly_chart(fig_hc, width="stretch", theme=None)

    with col_h2:
        debt_sel = sk_arr.debt_eur[sk_arr.graduated] if (fin_filter == "Nur Absolventen" and np.any(sk_arr.graduated)) else sk_arr.debt_eur
        d_max = max(10_000.0, float(np.percentile(debt_sel, 99.2)))
        counts_d, edges_d = np.histogram(np.clip(debt_sel, 0.0, d_max), bins=32)
        mids_d = 0.5 * (edges_d[:-1] + edges_d[1:])
        pct_d = (counts_d / max(1, len(debt_sel))) * 100.0
        fig_hd = go.Figure(
            data=[
                go.Bar(
                    x=mids_d,
                    y=pct_d,
                    marker_color="#A855F7",
                    opacity=0.82,
                    hovertemplate="Schulden ~%{x:,.0f} €<br>Anteil: <b>%{y:.2f} %</b><extra></extra>",
                )
            ]
        )
        med_d_sel = float(np.median(debt_sel))
        fig_hd.add_vline(
            x=med_d_sel,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Median: {_fmt_eur(med_d_sel)}",
            annotation_font=dict(color="#FCD34D", family=MONO_FONT, size=11),
        )
        fig_hd.update_layout(
            xaxis=dict(title="Gesamtschuldenlast (EUR)", tickformat=",.0f", ticksuffix=" €", gridcolor=PLOTLY_GRID),
            yaxis=dict(title="Häufigkeit (%)", ticksuffix="%", gridcolor=PLOTLY_GRID),
            bargap=0.05,
        )
        _apply_dark_layout(fig_hd, height=340, title=f"Schulden-Histogramm: {sankey_options[selected_sankey_key]}")
        st.plotly_chart(fig_hd, width="stretch", theme=None)


# =============================================================================
# TAB 4: SENSITIVITÄTSANALYSE (TORNADO), ANNAHMEN-EDITOR & CSV-EXPORT
# =============================================================================
with tab4:
    scope_options = {
        "portfolio_best": f"★ Optimaler 3-Pfeile-Mix ({sim_res.best_portfolio.label})",
        **{tid: TRACK_META[tid]["name"] for tid in TRACK_IDS},
    }
    selected_scope = st.selectbox(
        "Zielgröße für Tornado-Sensitivitätsanalyse (Wirkung auf Tier-1 IB + Quant/HF %)",
        options=list(scope_options.keys()),
        format_func=lambda k: scope_options[k],
        index=0,
    )

    sens_runs = min(n_runs, 60_000)
    with st.spinner("Berechne Tornado-Sensitivität mit Common Random Numbers ..."):
        sens_df = cached_run_sensitivity(p_json, selected_scope, sens_runs, seed, cfg_hash, cfg)

    base_prob = float(sens_df["Basis (%)"].iloc[0])

    fig_tornado = go.Figure()
    fig_tornado.add_trace(
        go.Bar(
            name="Downside (Low-Szenario)",
            y=sens_df["Hebel"],
            x=sens_df["Delta Low (%-Pkt.)"],
            orientation="h",
            marker_color="#EF4444",
            text=[f"{v:+.1f} pp" if abs(v) >= 0.1 else "" for v in sens_df["Delta Low (%-Pkt.)"]],
            textposition="outside",
            textfont=dict(size=11, family=MONO_FONT, color="#FCA5A5"),
            customdata=np.stack([sens_df["Low-Szenario"], sens_df["Low (%)"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}: <b>%{customdata[1]:.2f} %</b> (%{x:+.2f} %-Pkt.)<extra></extra>",
        )
    )
    fig_tornado.add_trace(
        go.Bar(
            name="Upside (High-Szenario)",
            y=sens_df["Hebel"],
            x=sens_df["Delta High (%-Pkt.)"],
            orientation="h",
            marker_color="#10B981",
            text=[f"{v:+.1f} pp" if abs(v) >= 0.1 else "" for v in sens_df["Delta High (%-Pkt.)"]],
            textposition="outside",
            textfont=dict(size=11, family=MONO_FONT, color="#6EE7B7"),
            customdata=np.stack([sens_df["High-Szenario"], sens_df["High (%)"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}: <b>%{customdata[1]:.2f} %</b> (%{x:+.2f} %-Pkt.)<extra></extra>",
        )
    )
    fig_tornado.add_vline(x=0.0, line_width=2, line_color="#F59E0B")

    max_abs = max(5.0, float(max(sens_df["Delta Low (%-Pkt.)"].abs().max(), sens_df["Delta High (%-Pkt.)"].abs().max())) * 1.25)
    fig_tornado.update_layout(
        barmode="overlay",
        xaxis=dict(
            title="Effekt auf Tier-1 IB + Quant/HF-Wahrscheinlichkeit (%-Punkte ggü. Basis)",
            range=[-max_abs, max_abs],
            gridcolor=PLOTLY_GRID,
            zeroline=False,
        ),
        yaxis=dict(title="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    _apply_dark_layout(
        fig_tornado,
        height=440,
        title=f"Tornado-Chart: Strategische Hebel (Basis = {base_prob:.2f} % · {scope_options[selected_scope]})",
    )
    st.plotly_chart(fig_tornado, width="stretch", theme=None)

    with st.expander("Sensitivitäts-Tabelle im Detail anzeigen", expanded=False):
        st.dataframe(sens_df.iloc[::-1].reset_index(drop=True), width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # AUFKLAPPBARER ANNAHMEN-EDITOR (config.py live inspizieren & überschreiben)
    # -------------------------------------------------------------------------
    with st.expander("Annahmen-Editor: Alle Wahrscheinlichkeiten & Parameter aus config.py testen", expanded=False):
        st.caption(
            "Alle Werte stammen direkt aus `config.py`. Du kannst jede Wahrscheinlichkeit, Gebühr oder "
            "Multiplikator-Annahme direkt bearbeiten und über 'Annahmen übernehmen' sofort neu simulieren."
        )

        track_edit_df = pd.DataFrame(cfg.tracks_as_records())
        edited_tracks_df = st.data_editor(
            track_edit_df,
            width="stretch",
            hide_index=True,
            disabled=["track_id", "Track"],
            key="track_params_editor",
            column_config={
                "track_id": st.column_config.TextColumn("ID"),
                "base_admit": st.column_config.NumberColumn("Basis Zulassung", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
                "base_finance": st.column_config.NumberColumn("Basis Finanz.", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
                "finance_with_scholarship": st.column_config.NumberColumn("Finanz. m. Stip.", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
                "base_dropout": st.column_config.NumberColumn("Basis Abbruch", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
                "dropout_discipline_sensitivity": st.column_config.NumberColumn("Disziplin-Hebel", min_value=0.05, max_value=2.0, step=0.02, format="%.2f"),
                "top_master_prob_high_gpa": st.column_config.NumberColumn("Top-Master (GPA<=1,7)", min_value=0.0, max_value=1.0, step=0.02, format="%.2f"),
                "top_master_prob_mid_gpa": st.column_config.NumberColumn("Top-Master (GPA<=2,1)", min_value=0.0, max_value=1.0, step=0.02, format="%.2f"),
                "placement_ib": st.column_config.NumberColumn("Basis IB", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "placement_quant": st.column_config.NumberColumn("Basis Quant", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "placement_tier2": st.column_config.NumberColumn("Basis Tier-2", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "placement_tech": st.column_config.NumberColumn("Basis Tech", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "top_master_ib_mult": st.column_config.NumberColumn("Master IB-Mult", min_value=0.5, max_value=6.0, step=0.05, format="%.2f"),
                "top_master_quant_mult": st.column_config.NumberColumn("Master Quant-Mult", min_value=0.5, max_value=6.0, step=0.05, format="%.2f"),
                "debt_no_scholarship_eur": st.column_config.NumberColumn("Schulden o. Stip. (€)", min_value=0.0, max_value=250_000.0, step=1000.0, format="%.0f"),
                "debt_with_scholarship_eur": st.column_config.NumberColumn("Schulden m. Stip. (€)", min_value=0.0, max_value=250_000.0, step=1000.0, format="%.0f"),
                "debt_top_master_addon_eur": st.column_config.NumberColumn("Master-Zusatzschuld (€)", min_value=0.0, max_value=150_000.0, step=1000.0, format="%.0f"),
                "comp_region_mult": st.column_config.NumberColumn("Gehalt-Region-Mult", min_value=0.5, max_value=2.5, step=0.05, format="%.2f"),
            },
        )

        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            st.markdown("**Profil-Multiplikatoren**")
            pm_df = pd.DataFrame([{"Parameter": k, "Wert": v} for k, v in cfg.profile_multipliers.items()])
            edited_pm_df = st.data_editor(
                pm_df,
                width="stretch",
                hide_index=True,
                disabled=["Parameter"],
                key="profile_mult_editor",
            )
        with col_ed2:
            st.markdown("**Begabtenförderungswerk (Stipendien-Kurve)**")
            sp_df = pd.DataFrame([{"Parameter": k, "Wert": v} for k, v in cfg.scholarship_params.items()])
            edited_sp_df = st.data_editor(
                sp_df,
                width="stretch",
                hide_index=True,
                disabled=["Parameter"],
                key="scholarship_editor",
            )

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Annahmen übernehmen & neu simulieren", type="primary", width="stretch"):
                new_cfg = cfg.clone()
                new_cfg.apply_track_records(edited_tracks_df.to_dict(orient="records"))
                for row in edited_pm_df.to_dict(orient="records"):
                    new_cfg.profile_multipliers[row["Parameter"]] = float(row["Wert"])
                for row in edited_sp_df.to_dict(orient="records"):
                    new_cfg.scholarship_params[row["Parameter"]] = float(np.clip(float(row["Wert"]), 0.0, 1.0))
                st.session_state.sim_config = new_cfg
                st.session_state.config_modified = True
                st.rerun()
        with btn_c2:
            if st.button("Standardwerte aus config.py wiederherstellen", width="stretch"):
                st.session_state.sim_config = get_default_config()
                st.session_state.config_modified = False
                st.rerun()

    # -------------------------------------------------------------------------
    # CSV-EXPORT
    # -------------------------------------------------------------------------
    st.markdown("##### CSV-Export der aggregierten Simulationsergebnisse")
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        st.download_button(
            label=f"Track-Zusammenfassung (Abi {profile.abitur_grade:.1f}) · CSV",
            data=sim_res.summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"career_mc_tracks_abi_{profile.abitur_grade:.1f}.csv",
            mime="text/csv",
            width="stretch",
        )
    with exp_c2:
        st.download_button(
            label="Kompletter Abischnitt-Sweep (1,3–2,5) · CSV",
            data=track_sweep_df.to_csv(index=False).encode("utf-8"),
            file_name="career_mc_abitur_sweep_1.3_2.5.csv",
            mime="text/csv",
            width="stretch",
        )
    with exp_c3:
        st.download_button(
            label="Tornado-Sensitivitätsanalyse · CSV",
            data=sens_df.to_csv(index=False).encode("utf-8"),
            file_name=f"career_mc_sensitivity_abi_{profile.abitur_grade:.1f}.csv",
            mime="text/csv",
            width="stretch",
        )
