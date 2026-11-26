"""
Vektorisierte Monte-Carlo-Karrieresimulations-Engine (NumPy + Pandas).

Architekturprinzipien:
- 100 % vektorisiert über alle N Runs (KEINE Python-for-Schleifen über Runs).
- Gaussian-Copula-Korrelation über gemeinsame latente Bewerberfaktoren
  (z_academic, z_interview, z_scholarship), sodass parallele Bewerbungen im
  "3-Pfeile-Portfolio-Mix" realistisch korreliert sind bei exakt erhaltenen
  Randwahrscheinlichkeiten pro Track.
- Common Random Numbers (CRN) für den Abischnitt-Sweep (1,3 bis 2,5) und die
  Tornado-Sensitivitätsanalyse zur Varianzreduktion und glatten Vergleichbarkeit.
- Ziel-Laufzeit: < 0,5 s (strikt < 1,5 s) für 100.000 Runs über alle 6 Tracks +
  sämtliche 20 möglichen 3-Pfeile-Portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from statistics import NormalDist
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import (
    ABITUR_SWEEP_GRADES,
    ApplicantProfile,
    OUTCOME_DROPOUT,
    OUTCOME_IB,
    OUTCOME_LABELS,
    OUTCOME_QUANT,
    OUTCOME_TECH,
    OUTCOME_TIER2,
    SimConfig,
    TRACK_IDS,
    TRACK_META,
    TrackConfig,
    get_default_config,
)

_STD_NORM = NormalDist(0.0, 1.0)


# =============================================================================
# 1. MATHEMATISCHE HILFSFUNKTIONEN (VEKTORISIERT & EXAKT)
# =============================================================================

def _norm_ppf(p: float) -> float:
    """Exaktes Quantil der Standardnormalverteilung für skalare Wahrscheinlichkeit p."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    return _STD_NORM.inv_cdf(float(np.clip(p, 1e-9, 1.0 - 1e-9)))


def _correlated_bernoulli(
    p: float,
    z_shared: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Zieht korrelierte Bernoulli(p)-Variablen mittels Gaussian Copula:
        Z = sqrt(rho) * z_shared + sqrt(1 - rho) * eps   ~ N(0, 1)
        Event = (Z < Phi^{-1}(p))
    Bewahrt exakt die Randwahrscheinlichkeit P(Event) = p.
    """
    n = z_shared.shape[0]
    if p <= 0.0:
        return np.zeros(n, dtype=bool)
    if p >= 1.0:
        return np.ones(n, dtype=bool)
    thresh = np.float32(_norm_ppf(p))
    w_s = np.float32(np.sqrt(rho))
    w_i = np.float32(np.sqrt(max(0.0, 1.0 - rho)))
    z = w_s * z_shared + w_i * rng.standard_normal(n, dtype=np.float32)
    return z < thresh


def _correlated_bernoulli_vec(
    p_vec: np.ndarray,
    z_shared: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vektorisierte korrelierte Bernoulli-Ziehung für Array-Wahrscheinlichkeiten p_vec
    via Acklam-Approximation der inversen Normal-CDF (Fehler < 1.2e-9) in float32/64.
    """
    n = z_shared.shape[0]
    p_clipped = np.clip(p_vec.astype(np.float64), 1e-7, 1.0 - 1e-7)
    thresh = _acklam_norm_ppf(p_clipped).astype(np.float32)
    w_s = np.float32(np.sqrt(rho))
    w_i = np.float32(np.sqrt(max(0.0, 1.0 - rho)))
    z = w_s * z_shared + w_i * rng.standard_normal(n, dtype=np.float32)
    res = z < thresh
    res = np.where(p_vec <= 0.0, False, np.where(p_vec >= 1.0, True, res))
    return res


def _acklam_norm_ppf(p: np.ndarray) -> np.ndarray:
    """Vektorisierte inverse Standardnormal-CDF nach Peter J. Acklam (rein NumPy)."""
    a = np.array([
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ])
    b = np.array([
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ])
    d = np.array([
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ])

    p_low = 0.02425
    p_high = 1.0 - p_low
    x = np.empty_like(p, dtype=np.float64)

    low_mask = p < p_low
    high_mask = p > p_high
    mid_mask = (~low_mask) & (~high_mask)

    if np.any(low_mask):
        q = np.sqrt(-2.0 * np.log(p[low_mask]))
        x[low_mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                      ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    if np.any(mid_mask):
        q = p[mid_mask] - 0.5
        r = q * q
        x[mid_mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                      (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    if np.any(high_mask):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high_mask]))
        x[high_mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return x


def _compute_run_scores(
    outcome: np.ndarray,
    comp_eur: np.ndarray,
    debt_eur: np.ndarray,
    cfg: SimConfig,
) -> Tuple[np.ndarray, float]:
    """
    Berechnet den per-Run Career Score (0-100) sowie den aggregierten
    Risk-Adjusted Career Score (Mean - lambda * Downside-Semideviation).
    """
    sw = cfg.score_weights
    util_lut = np.array(
        [sw["util_ib"], sw["util_quant"], sw["util_tier2"], sw["util_tech"], sw["util_dropout"]],
        dtype=np.float32,
    )
    util = util_lut[outcome]
    comp_bonus = np.clip(
        (comp_eur - np.float32(sw["comp_floor_eur"])) / np.float32(sw["comp_step_eur"]),
        np.float32(0.0),
        np.float32(sw["comp_max_bonus"]),
    )
    debt_pen = np.clip(
        (debt_eur / np.float32(100_000.0)) * np.float32(sw["debt_penalty_per_100k"]),
        np.float32(0.0),
        np.float32(sw["debt_max_penalty"]),
    )
    run_score = np.clip(
        np.float32(sw["util_weight"]) * util + comp_bonus - debt_pen,
        np.float32(0.0),
        np.float32(100.0),
    )

    mean_score = float(np.mean(run_score, dtype=np.float64))
    downside = np.minimum(np.float32(0.0), run_score - np.float32(mean_score))
    downside_semi_std = float(np.sqrt(np.mean(downside * downside, dtype=np.float64)))
    risk_adj = float(np.clip(mean_score - sw["downside_risk_lambda"] * downside_semi_std, 0.0, 100.0))
    return run_score, risk_adj


# =============================================================================
# 2. ERGEBNIS-DATACLASSES
# =============================================================================

@dataclass
class TrackSimArrays:
    """Roh-Arrays (Länge n_runs) eines einzelnen Tracks für Sankey, Boxplots & Portfolios."""
    track_id: str
    admitted: np.ndarray          # bool
    financed: np.ndarray          # bool (Subset von admitted)
    graduated: np.ndarray         # bool (Subset von financed)
    has_top_master: np.ndarray    # bool (Subset von graduated)
    has_scholarship: np.ndarray   # bool
    uni_gpa: np.ndarray           # float32 (1.0 - 4.0)
    outcome: np.ndarray           # int8 (0..4)
    debt_eur: np.ndarray          # float32
    comp_eur: np.ndarray          # float32
    run_score: np.ndarray         # float32


@dataclass
class TrackSummary:
    """Aggregierte Kennzahlen eines Tracks (oder Portfolios)."""
    track_id: str
    name: str
    admit_rate: float
    finance_rate_cond: float      # P(Finanziert | Zugelassen)
    finance_rate_uncond: float    # P(Zugelassen & Finanziert)
    grad_rate_cond: float         # P(Abschluss | Finanziert)
    top_master_rate_cond: float   # P(Top-Master | Abschluss)
    outcome_probs: Dict[str, float]  # Summe exakt = 1.0
    target_ib_quant_prob: float   # Tier-1 IB + Top Quant / Hedgefonds
    median_debt_eur: float
    median_debt_grad_eur: float
    mean_debt_eur: float
    median_comp_eur: float
    median_comp_grad_eur: float
    p90_comp_eur: float
    mean_comp_eur: float
    risk_adjusted_score: float
    conditional_risk_adj_score: float  # Score bedingt auf (Zugelassen & Finanziert)


@dataclass
class PortfolioSummary:
    """Aggregierte Kennzahlen eines 3-Pfeile-Portfolio-Mixes."""
    track_ids: Tuple[str, str, str]
    label: str
    coverage_rate: float          # Anteil Runs mit >= 1 zugelassenem & finanziertem Track
    outcome_probs: Dict[str, float]
    target_ib_quant_prob: float
    ib_prob: float
    quant_prob: float
    dropout_prob: float
    median_debt_eur: float
    median_comp_eur: float
    p90_comp_eur: float
    risk_adjusted_score: float
    track_usage_shares: Dict[str, float]  # Wie oft welcher der 3 Pfeile im Portfolio gewählt wurde


@dataclass
class SimulationResult:
    """Vollständiges Simulationsergebnis für ein Profil."""
    profile: ApplicantProfile
    n_runs: int
    seed: int
    elapsed_seconds: float
    scholarship_prob: float
    track_summaries: Dict[str, TrackSummary]
    track_arrays: Optional[Dict[str, TrackSimArrays]]
    best_single_track_id: str
    portfolios: List[PortfolioSummary]
    best_portfolio: PortfolioSummary
    best_portfolio_arrays: Optional[TrackSimArrays]
    summary_df: pd.DataFrame
    portfolio_df: pd.DataFrame


# =============================================================================
# 3. STIPENDIEN- & ZULASSUNGSFUNKTIONEN
# =============================================================================

def compute_scholarship_probability(profile: ApplicantProfile, cfg: SimConfig) -> float:
    """
    Wahrscheinlichkeit für ein deutsches Begabtenförderungswerk
    (Studienstiftung des dt. Volkes, sdw, KAS/FES/FNS/HBS).
    Stark abhängig vom Abischnitt (hoch bis 1,5, steiler Abfall ab 1,7) sowie
    First-Gen, Engagement (Tradity, Jugend gründet) & Resilienz (GdB 100).
    """
    sp = cfg.scholarship_params
    pm = cfg.profile_multipliers

    base = float(np.interp(
        profile.abitur_grade,
        [1.3, 1.5, 1.7, 2.0, 2.5],
        [
            sp["base_prob_abi_13"],
            sp["base_prob_abi_15"],
            sp["base_prob_abi_17"],
            sp["base_prob_abi_20"],
            sp["base_prob_abi_25"],
        ],
    ))

    mult = 1.0
    if profile.is_first_gen:
        mult *= pm["scholarship_first_gen_mult"]
    if profile.has_tradity:
        mult *= pm["scholarship_tradity_mult"]
    if profile.has_jugend_gruendet:
        mult *= pm["scholarship_jugend_gruendet_mult"]
    if profile.has_gdb_100:
        mult *= pm["scholarship_gdb_100_mult"]
    if profile.has_ma_internship:
        mult *= 1.05
    if profile.discipline >= 0.75:
        mult *= 1.04

    return float(np.clip(base * mult, 0.005, sp["max_prob_cap"]))


def _compute_admit_prob_us_elite(profile: ApplicantProfile, tc: TrackConfig, cfg: SimConfig) -> float:
    """US Need-Blind Full-Aid Portfolio (~7 Eliten): extrem selektiv, Spike- & Test-getrieben."""
    pm = cfg.profile_multipliers
    abi_mult = float(np.interp(
        profile.abitur_grade,
        [1.3, 1.5, 1.7, 1.9, 2.2, 2.5],
        [1.35, 1.00, 0.58, 0.30, 0.12, 0.05],
    ))
    test_mult = float(np.interp(
        profile.effective_sat,
        [1200.0, 1350.0, 1450.0, 1500.0, 1540.0, 1600.0],
        [0.15, 0.35, 0.72, 1.02, 1.22, 1.38],
    ))
    stem_avg = 0.5 * (profile.math_points + profile.physics_points)
    stem_mult = float(np.interp(stem_avg, [6.0, 10.0, 12.0, 14.0, 15.0], [0.55, 0.85, 1.00, 1.12, 1.18]))

    spike_mult = 1.0
    if profile.is_first_gen:
        spike_mult *= pm["us_first_gen_admit_mult"]
    if profile.has_ma_internship:
        spike_mult *= pm["us_ma_internship_admit_mult"]
    if profile.has_jugend_gruendet:
        spike_mult *= pm["us_jugend_gruendet_admit_mult"]
    if profile.has_tradity:
        spike_mult *= pm["us_tradity_admit_mult"]
    if profile.has_gdb_100:
        spike_mult *= pm["us_gdb_100_resilience_mult"]

    raw = tc.base_admit * abi_mult * test_mult * stem_mult * spike_mult
    return float(np.clip(raw, 0.002, 0.25))


def _compute_admit_prob_hsg(profile: ApplicantProfile, tc: TrackConfig, cfg: SimConfig) -> float:
    """HSG St. Gallen: primär HSG-Eignungstest + Video-Interview, schwach NC-abhängig."""
    pm = cfg.profile_multipliers
    # Logistische Test-Kurve zentriert um das 68. Perzentil (normiert auf 1.0 bei Perzentil 80)
    logistic_80 = 1.0 / (1.0 + np.exp(-(80.0 - 68.0) / 11.5))
    logistic_p = 1.0 / (1.0 + np.exp(-(float(profile.hsg_percentile) - 68.0) / 11.5))
    test_factor = logistic_p / logistic_80

    abi_factor = float(np.interp(profile.abitur_grade, [1.3, 1.7, 2.1, 2.5], [1.06, 1.00, 0.92, 0.84]))
    base_prob = tc.base_admit * test_factor * abi_factor

    interview_add = 0.0
    if profile.has_ma_internship:
        interview_add += pm["hsg_ma_interview_boost"]
    if profile.has_jugend_gruendet:
        interview_add += 0.6 * pm["hsg_ec_interview_boost"]
    if profile.has_tradity:
        interview_add += 0.4 * pm["hsg_ec_interview_boost"]

    return float(np.clip(base_prob + interview_add * test_factor, 0.02, 0.95))


def _compute_admit_prob_eth(profile: ApplicantProfile, tc: TrackConfig) -> float:
    """
    ETH Zürich B.Sc. Physik:
    Prüfungsfrei (base_admit ~ 100 %), wenn Durchschnitt aus Mathe + Physik + Sprache >= 10,5 Pkt (70 %).
    Darunter ist die anspruchsvolle ETH-Aufnahmeprüfung erforderlich.
    """
    if profile.eth_exam_free:
        return float(np.clip(tc.base_admit, 0.0, 1.0))
    stem_avg = 0.5 * (profile.math_points + profile.physics_points)
    stem_factor = float(np.interp(stem_avg, [5.0, 8.0, 10.0, 12.0, 14.0], [0.25, 0.65, 0.95, 1.25, 1.45]))
    disc_factor = 0.55 + 0.65 * profile.discipline
    exam_pass = 0.30 * tc.base_admit * stem_factor * disc_factor
    return float(np.clip(exam_pass, 0.04, 0.68))


def _compute_admit_prob_uk(profile: ApplicantProfile, tc: TrackConfig) -> float:
    """UK Top-Unis (LSE, Oxford, Imperial, Warwick): streng abischnittabhängige UCAS-Offers."""
    abi_factor = float(np.interp(
        profile.abitur_grade,
        [1.3, 1.4, 1.5, 1.7, 1.9, 2.2, 2.5],
        [1.18, 0.94, 0.72, 0.34, 0.14, 0.05, 0.02],
    ))
    stem_avg = 0.5 * (profile.math_points + profile.physics_points)
    stem_factor = float(np.interp(stem_avg, [8.0, 11.0, 13.0, 15.0], [0.55, 0.85, 1.05, 1.18]))
    test_factor = float(np.interp(profile.effective_sat, [1300.0, 1450.0, 1540.0, 1600.0], [0.80, 0.96, 1.06, 1.12]))
    ps_boost = 1.0
    if profile.has_ma_internship:
        ps_boost += 0.10
    if profile.has_jugend_gruendet:
        ps_boost += 0.06
    if profile.has_gdb_100:
        ps_boost += 0.04
    return float(np.clip(tc.base_admit * abi_factor * stem_factor * test_factor * ps_boost, 0.004, 0.78))


# =============================================================================
# 4. KERN-TRACK-SIMULATION (100 % VEKTORISIERT)
# =============================================================================

def _simulate_track(
    tid: str,
    profile: ApplicantProfile,
    cfg: SimConfig,
    n_runs: int,
    z_acad: np.ndarray,
    z_inter: np.ndarray,
    z_schol: np.ndarray,
    has_scholarship: np.ndarray,
    rng: np.random.Generator,
) -> TrackSimArrays:
    """
    Simuliert die 5-stufige Karriere-Pipeline eines einzelnen Uni-Tracks
    vollständig vektorisiert über `n_runs` Durchläufe.
    """
    tc = cfg.tracks[tid]
    pm = cfg.profile_multipliers
    cp = cfg.correlation_params
    rho_acad = cp["rho_academic"]
    rho_inter = cp["rho_interview"]
    rho_schol = cp["rho_scholarship"]

    stem_avg = 0.5 * (profile.math_points + profile.physics_points)
    stem_strength = (stem_avg - 12.0) / 3.0  # normiert: 12 Pkt -> 0.0, 15 Pkt -> +1.0

    # Spezielle Subpfad-Flags für Track 5 (DE Top-BWL / WiMa: 0=keine, 1=Mannheim BWL, 2=WHU/FS, 3=Mannheim WiMa)
    subpath_bwl = np.zeros(n_runs, dtype=np.int8)

    # -------------------------------------------------------------------------
    # STUFE 1 (Zulassung) & STUFE 2 (Finanzierbarkeit)
    # -------------------------------------------------------------------------
    if tid == "de_bwl_wima":
        # Explizite Modellierung der 3 realen DE-Target-Subpfade:
        # (1) Uni Mannheim BWL (harter NC + GdB-100-Härtefallbonus + EC-Bonus, 0 EUR Gebühren)
        # (2) WHU / Frankfurt School (Assessment Center, ~45k EUR Gebühren via Stipendium oder UGV)
        # (3) Uni Mannheim WiMa (NC-entspannte Quant-Alternative, 0 EUR Gebühren)
        scale = tc.base_admit / 0.88 if 0.88 > 0 else 1.0
        ec_frac = (
            (0.60 if profile.has_ma_internship else 0.0)
            + (0.25 if profile.has_jugend_gruendet else 0.0)
            + (0.15 if profile.has_tradity else 0.0)
        )
        eff_abi_ma = profile.abitur_grade - pm["mannheim_ec_abi_bonus"] * ec_frac
        if profile.has_gdb_100:
            # Härtefallantrag / Nachteilsausgleich (GdB 100) verbessert den effektiven NC-Rang spürbar,
            # garantiert aber bei sehr hohem Schnitt (> 2,1) nicht automatisch einen Platz.
            eff_abi_ma_gdb = eff_abi_ma - pm["mannheim_gdb_100_abi_bonus"]
            p_ma_bwl = float(np.interp(
                eff_abi_ma_gdb,
                [0.9, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5],
                [0.96, 0.92, 0.80, 0.56, 0.34, 0.18, 0.10, 0.05],
            )) * scale
        else:
            p_ma_bwl = float(np.interp(
                eff_abi_ma,
                [1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5],
                [0.95, 0.90, 0.74, 0.38, 0.12, 0.03, 0.01],
            )) * scale
        p_ma_bwl = float(np.clip(p_ma_bwl, 0.005, 0.98))

        # WHU / Frankfurt School: holistisches Assessment Center + SAT/ACT + ECs
        sat_bonus = float(np.clip((profile.effective_sat - 1380.0) / 1800.0, -0.08, 0.10))
        p_whu_fs = (
            float(np.interp(profile.abitur_grade, [1.3, 1.6, 1.9, 2.2, 2.5], [0.68, 0.54, 0.36, 0.20, 0.10]))
            + pm["whu_fs_ec_admit_boost"] * ec_frac
            + sat_bonus
            + (0.03 if profile.has_gdb_100 else 0.0)
        ) * scale
        p_whu_fs = float(np.clip(p_whu_fs, 0.03, 0.90))

        # Uni Mannheim Wirtschaftsmathematik (WiMa): NC-entspannter, aber Mathe-selektiv
        wima_abi = profile.abitur_grade - (0.20 if profile.has_gdb_100 else 0.0)
        math_wima_mult = float(np.interp(profile.math_points, [6.0, 9.0, 11.0, 13.0, 15.0], [0.40, 0.78, 0.96, 1.04, 1.08]))
        p_ma_wima = float(np.interp(
            wima_abi,
            [1.3, 1.7, 2.0, 2.2, 2.4, 2.5],
            [0.88, 0.78, 0.58, 0.36, 0.18, 0.10],
        )) * math_wima_mult * scale
        p_ma_wima = float(np.clip(p_ma_wima, 0.02, 0.94))

        # Hohe gemeinsame Zulassungskorrelation innerhalb DE BWL/WiMa (gleiche Bewerbungsunterlagen)
        z_de_bwl = np.float32(0.7071) * (z_inter + z_acad)
        adm_ma_bwl = _correlated_bernoulli(p_ma_bwl, z_de_bwl, 0.65, rng)
        adm_whu_fs = _correlated_bernoulli(p_whu_fs, z_de_bwl, 0.65, rng)
        adm_ma_wima = _correlated_bernoulli(p_ma_wima, z_de_bwl, 0.60, rng)
        admitted = adm_ma_bwl | adm_whu_fs | adm_ma_wima

        # Subpfad-Finanzierung:
        # Öffentlich (Mannheim BWL / WiMa): gebührenfrei + BAföG
        fin_scale = tc.base_finance / 0.84 if 0.84 > 0 else 1.0
        p_fin_pub_vec = np.where(
            has_scholarship,
            np.clip(tc.finance_with_scholarship, 0.0, 0.995),
            np.clip(0.955 * fin_scale + (0.015 if profile.is_first_gen else 0.0), 0.05, 0.99),
        )
        # Privat (WHU / Frankfurt School): Stipendium oder UGV (Brain Capital / Dt. Bildung)
        p_fin_priv_vec = np.where(
            has_scholarship,
            np.clip(tc.finance_with_scholarship * 0.98, 0.0, 0.99),
            np.clip(0.76 * fin_scale + 0.08 * (profile.discipline - 0.5), 0.05, 0.95),
        )
        fin_pub = _correlated_bernoulli_vec(p_fin_pub_vec, z_schol, rho_schol * 0.4, rng)
        fin_priv = _correlated_bernoulli_vec(p_fin_priv_vec, z_schol, rho_schol * 0.6, rng)

        ok_ma_bwl = adm_ma_bwl & fin_pub
        ok_whu_fs = adm_whu_fs & fin_priv
        ok_ma_wima = adm_ma_wima & fin_pub

        # Rationale Wahl innerhalb des Tracks:
        # (1) Mannheim BWL (kostenfrei & Top-IB) > (2) WHU/FS (Top-IB via UGV) > (3) Mannheim WiMa (Quant & kostenfrei)
        subpath_bwl = np.where(
            ok_ma_bwl, np.int8(1),
            np.where(ok_whu_fs, np.int8(2),
                     np.where(ok_ma_wima, np.int8(3), np.int8(0)))
        )
        financed = subpath_bwl > 0

    else:
        if tid == "us_elite":
            p_admit = _compute_admit_prob_us_elite(profile, tc, cfg)
            admitted = _correlated_bernoulli(p_admit, z_inter, rho_inter, rng)
        elif tid == "hsg":
            p_admit = _compute_admit_prob_hsg(profile, tc, cfg)
            admitted = _correlated_bernoulli(p_admit, z_inter, rho_inter, rng)
        elif tid == "eth_physik":
            p_admit = _compute_admit_prob_eth(profile, tc)
            admitted = _correlated_bernoulli(p_admit, z_acad, rho_acad, rng)
        elif tid == "de_physik":
            p_admit = float(np.clip(tc.base_admit, 0.0, 1.0))
            admitted = _correlated_bernoulli(p_admit, z_acad, 0.1, rng)
        elif tid == "uk_top":
            p_admit = _compute_admit_prob_uk(profile, tc)
            z_uk = np.float32(0.7071) * (z_acad + z_inter)
            admitted = _correlated_bernoulli(p_admit, z_uk, 0.40, rng)
        else:
            raise ValueError(f"Unbekannter Track: {tid}")

        # Stufe 2: Finanzierbarkeit (bedingt auf Zulassung)
        p_fin_no_schol = tc.base_finance
        p_fin_with_schol = tc.finance_with_scholarship

        if tid == "us_elite":
            if profile.is_first_gen:
                p_fin_no_schol += 0.025  # Need-Blind Full Aid bei niedrigem Familieneinkommen
        elif tid == "hsg":
            if profile.is_first_gen:
                p_fin_no_schol += 0.07   # Auslands-BAföG-Berechtigung
            # CH-Werkstudentenjob (gut bezahlt ~30 CHF/h) korreliert mit Disziplin & Praxis
            p_fin_no_schol += 0.12 * (profile.discipline - 0.5)
            if profile.has_ma_internship:
                p_fin_no_schol += 0.04
        elif tid == "eth_physik":
            if profile.is_first_gen:
                p_fin_no_schol += 0.07   # Auslands-BAföG
            # Im 1. Jahr (Basisprüfung) kaum Nebenjob möglich -> geringerer Disziplin-Job-Effekt
            p_fin_no_schol += 0.05 * (profile.discipline - 0.5)
        elif tid == "de_physik":
            if profile.is_first_gen:
                p_fin_no_schol += 0.015  # Inlands-BAföG
        elif tid == "uk_top":
            # Post-Brexit: Selbst für First-Gen extrem wenige Vollstipendien im Bachelor
            if profile.is_first_gen:
                p_fin_no_schol += 0.015
                p_fin_with_schol += 0.03

        p_fin_no_schol = float(np.clip(p_fin_no_schol, 0.01, 0.995))
        p_fin_with_schol = float(np.clip(p_fin_with_schol, 0.01, 0.998))

        p_fin_vec = np.where(has_scholarship, p_fin_with_schol, p_fin_no_schol)
        fin_draw = _correlated_bernoulli_vec(p_fin_vec, z_schol, rho_schol * 0.5, rng)
        financed = admitted & fin_draw

    # -------------------------------------------------------------------------
    # STUFE 3: Studienabschluss & Uni-GPA (inkl. realer Abbrecherquoten)
    # -------------------------------------------------------------------------
    is_stem_heavy = tid in ("eth_physik", "de_physik")
    if tid == "de_bwl_wima":
        # Mannheim WiMa (subpath == 3) hat deutlich höhere Mathe-Abbruchquote als BWL/WHU
        eff_res_bwl = (profile.discipline - 0.60) + 0.08 * stem_strength
        eff_res_wima = (profile.discipline - 0.60) + 0.25 * stem_strength
        p_drop_bwl = float(np.clip(tc.base_dropout * 0.82 - tc.dropout_discipline_sensitivity * eff_res_bwl, 0.03, 0.65))
        p_drop_wima = float(np.clip(tc.base_dropout * 1.75 - (tc.dropout_discipline_sensitivity * 1.55) * eff_res_wima, 0.06, 0.80))
        p_grad_vec = np.where(subpath_bwl == 3, 1.0 - p_drop_wima, 1.0 - p_drop_bwl)
        grad_draw = _correlated_bernoulli_vec(p_grad_vec, z_acad, rho_acad, rng)
    else:
        eff_resilience = (profile.discipline - 0.60) + (0.28 * stem_strength if is_stem_heavy else 0.08 * stem_strength)
        p_drop = float(np.clip(tc.base_dropout - tc.dropout_discipline_sensitivity * eff_resilience, 0.02, 0.88))
        grad_draw = _correlated_bernoulli(1.0 - p_drop, z_acad, rho_acad, rng)

    graduated = financed & grad_draw

    # Uni-GPA (deutsche Notenskala 1,0 = Bestnote bis 3,7; niedriger = besser)
    # Wichtig: Bei DE Top-Physik hängt Uni-GPA NICHT vom Abischnitt ab, sondern von Disziplin, Mathe/Physik & z_acad!
    gpa_shift = 0.0
    if tid == "us_elite":
        gpa_shift = -0.22  # US Ivy/Elite Grade Inflation
    elif tid in ("eth_physik", "de_physik"):
        gpa_shift = 0.08   # Strengere Physik-Notengebung
    elif tid == "uk_top":
        gpa_shift = -0.08  # First Class / 2:1 Umrechnung

    mean_gpa = float(np.clip(
        2.18 + gpa_shift - 0.82 * (profile.discipline - 0.50) - 0.045 * (stem_avg - 11.5),
        1.22,
        2.90,
    ))
    w_s = np.float32(np.sqrt(rho_acad))
    w_i = np.float32(np.sqrt(1.0 - rho_acad))
    z_gpa = w_s * z_acad + w_i * rng.standard_normal(n_runs, dtype=np.float32)
    # Höheres z_acad => bessere (niedrigere) Note
    uni_gpa = np.clip(np.float32(mean_gpa) - np.float32(0.34) * z_gpa, np.float32(1.0), np.float32(3.7))

    # -------------------------------------------------------------------------
    # STUFE 4: Optionaler Top-Master (ETH MSc QF, HSG MBF, US MFE, Oxford/Imperial)
    # -------------------------------------------------------------------------
    # Bei DE Top-Physik ist der Abischnitt ab hier komplett irrelevant — nur Uni-GPA & Profil zählen!
    sat_gre_boost = float(np.clip((profile.effective_sat - 1440.0) / 1500.0, -0.06, 0.09))
    profile_master_boost = (
        (0.05 if profile.has_ma_internship else 0.0)
        + (0.03 if profile.has_jugend_gruendet else 0.0)
        + (0.02 if profile.has_tradity else 0.0)
    )
    p_master_vec = np.where(
        uni_gpa <= np.float32(1.70),
        np.float32(tc.top_master_prob_high_gpa),
        np.where(
            uni_gpa <= np.float32(2.10),
            np.float32(tc.top_master_prob_mid_gpa),
            np.float32(tc.top_master_prob_mid_gpa * 0.28),
        ),
    )
    # Stipendium erleichtert Finanzierung eines 1-2-jährigen Auslands-Top-Masters (z. B. UK 1-J-Master / ETH / HSG)
    p_master_vec = np.clip(
        p_master_vec
        + np.float32(sat_gre_boost + profile_master_boost)
        + np.where(has_scholarship, np.float32(0.07), np.float32(0.0)),
        np.float32(0.02),
        np.float32(0.95),
    )
    master_draw = _correlated_bernoulli_vec(p_master_vec, z_acad, rho_acad * 0.5, rng)
    has_top_master = graduated & master_draw

    # -------------------------------------------------------------------------
    # STUFE 5: Praktika & Finales Karriere-Outcome (0=IB, 1=Quant, 2=Tier2, 3=Tech, 4=Abbruch)
    # -------------------------------------------------------------------------
    if tid == "de_bwl_wima":
        # Subpfad-spezifische Placement-Profile:
        # Mannheim BWL (1) & WHU/FS (2): starkes IB; Mannheim WiMa (3): starkes Quant
        w_ib = np.where(subpath_bwl == 3, np.float32(tc.placement_ib * 0.58), np.float32(tc.placement_ib * 1.12))
        w_quant = np.where(subpath_bwl == 3, np.float32(tc.placement_quant * 2.65), np.float32(tc.placement_quant * 0.65))
        w_tier2 = np.full(n_runs, np.float32(tc.placement_tier2), dtype=np.float32)
        w_tech = np.where(subpath_bwl == 3, np.float32(tc.placement_tech * 1.30), np.float32(tc.placement_tech * 0.90))
    else:
        w_ib = np.full(n_runs, np.float32(tc.placement_ib), dtype=np.float32)
        w_quant = np.full(n_runs, np.float32(tc.placement_quant), dtype=np.float32)
        w_tier2 = np.full(n_runs, np.float32(tc.placement_tier2), dtype=np.float32)
        w_tech = np.full(n_runs, np.float32(tc.placement_tech), dtype=np.float32)

    # Top-Master-Sprungbrett (besonders massiv für DE Top-Physik -> ETH QF / HSG MBF / US MFE / Oxbridge)
    w_ib = w_ib * np.where(has_top_master, np.float32(tc.top_master_ib_mult), np.float32(1.0))
    w_quant = w_quant * np.where(has_top_master, np.float32(tc.top_master_quant_mult), np.float32(1.0))

    # Uni-GPA-Effekt auf Top-Recruiting (<= 1,7 stark belohnt)
    gpa_ib_mult = np.clip(np.float32(1.0) + np.float32(0.55) * (np.float32(1.85) - uni_gpa), np.float32(0.45), np.float32(1.55))
    gpa_quant_mult = np.clip(np.float32(1.0) + np.float32(0.65) * (np.float32(1.85) - uni_gpa), np.float32(0.40), np.float32(1.65))
    w_ib = w_ib * gpa_ib_mult
    w_quant = w_quant * gpa_quant_mult

    # Bewerber-Profil- & Praktika-Snowball-Multiplikatoren (Skalar -> extrem schnell)
    ib_prof_mult = 0.72 + 0.56 * profile.discipline
    if profile.has_ma_internship:
        ib_prof_mult *= pm["recruiting_ma_ib_mult"]
    if profile.has_jugend_gruendet:
        ib_prof_mult *= pm["recruiting_jg_ib_mult"]
    if profile.has_tradity:
        ib_prof_mult *= pm["recruiting_tradity_fin_mult"]

    # Realistisches deutsches IB-HR-Abitur-Screening bei reinem BWL-Track:
    # BB/EB-Banken in Frankfurt screenen bei BWL-Bewerbern das Abitur (starker Abfall > 1,8),
    # während bei Physik + Top-Master (DE Top-Physik / ETH) der Abischnitt komplett irrelevant ist!
    if tid == "de_bwl_wima":
        bwl_abi_screen = float(np.interp(
            profile.abitur_grade,
            [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
            [1.06, 1.00, 0.90, 0.76, 0.62, 0.48, 0.36],
        ))
        # WiMa-Absolventen (subpath == 3) profitieren stärker vom Quant-Signal als reine BWLer
        abi_screen_vec = np.where(subpath_bwl == 3, np.float32(0.5 + 0.5 * bwl_abi_screen), np.float32(bwl_abi_screen))
        w_ib = w_ib * abi_screen_vec
    elif tid == "hsg":
        # HSG-Marke puffert Abischnitt im IB deutlich besser ab, leichter Effekt ab > 2,0
        hsg_abi_screen = float(np.interp(profile.abitur_grade, [1.3, 1.8, 2.2, 2.5], [1.03, 1.00, 0.90, 0.80]))
        w_ib = w_ib * np.float32(hsg_abi_screen)

    quant_prof_mult = (
        (0.62 + 0.76 * profile.discipline)
        * float(np.interp(stem_avg, [8.0, 12.0, 15.0], [0.55, 1.00, 1.32]))
        * float(np.interp(profile.effective_sat, [1300.0, 1480.0, 1560.0], [0.80, 1.00, 1.16]))
        * (1.04 if profile.has_tradity else 1.0)
        * (1.05 if profile.has_ma_internship else 1.0)
    )

    w_ib = w_ib * np.float32(ib_prof_mult) * np.exp(np.float32(0.22) * z_inter)
    w_quant = w_quant * np.float32(quant_prof_mult) * np.exp(np.float32(0.24) * z_acad + np.float32(0.08) * z_inter)

    # Strenge Normalisierung auf 1,0 für alle 4 Abschluss-Outcomes
    w_sum = np.maximum(w_ib + w_quant + w_tier2 + w_tech, np.float32(1e-8))
    c1 = w_ib / w_sum
    c2 = c1 + (w_quant / w_sum)
    c3 = c2 + (w_tier2 / w_sum)

    u_career = rng.random(n_runs, dtype=np.float32)
    grad_outcome = np.where(
        u_career < c1, np.int8(OUTCOME_IB),
        np.where(
            u_career < c2, np.int8(OUTCOME_QUANT),
            np.where(u_career < c3, np.int8(OUTCOME_TIER2), np.int8(OUTCOME_TECH)),
        ),
    )
    outcome = np.where(graduated, grad_outcome, np.int8(OUTCOME_DROPOUT))

    # -------------------------------------------------------------------------
    # STUFE 6: Finanzmetriken (Schuldenlast EUR, Total Comp J1 EUR, Score)
    # -------------------------------------------------------------------------
    if tid == "de_bwl_wima":
        # Mannheim (Subpfad 1 & 3) ist gebührenfrei (~6,5k BAföG), WHU/FS (Subpfad 2) hat ~40k UGV-Last
        debt_scale = tc.debt_no_scholarship_eur / 18_000.0 if 18_000.0 > 0 else 1.0
        base_debt = np.where(
            subpath_bwl == 2,
            np.where(has_scholarship, np.float32(12_000.0 * debt_scale), np.float32(40_000.0 * debt_scale)),
            np.where(has_scholarship, np.float32(1_800.0 * debt_scale), np.float32(6_500.0 * debt_scale)),
        )
    else:
        base_debt = np.where(
            has_scholarship,
            np.float32(tc.debt_with_scholarship_eur),
            np.float32(tc.debt_no_scholarship_eur),
        )

    master_debt = np.where(
        has_top_master,
        np.where(has_scholarship, np.float32(tc.debt_top_master_addon_eur * 0.55), np.float32(tc.debt_top_master_addon_eur)),
        np.float32(0.0),
    )
    debt_noise = np.clip(
        np.float32(1.0) + np.float32(0.20) * rng.standard_normal(n_runs, dtype=np.float32),
        np.float32(0.45),
        np.float32(1.85),
    )
    grad_debt = (base_debt + master_debt) * debt_noise

    # Nicht-Absolventen:
    # - Nicht zugelassen: 0 EUR Track-Schulden
    # - Finanzierungs-Stopp: geringe Bewerbungs-/Test-/Sunk-Costs (max. 5.500 EUR)
    # - Studienabbruch: anteilige Schulden (~38 % der Basis-Schulden)
    sunk_fin_debt = np.minimum(base_debt * np.float32(0.10), np.float32(5_500.0)) * debt_noise
    drop_debt = base_debt * np.float32(0.38) * debt_noise

    debt_eur = np.where(
        graduated, grad_debt,
        np.where(
            financed, drop_debt,
            np.where(admitted, sunk_fin_debt, np.float32(0.0)),
        ),
    )

    # Brutto-Einstiegsgehalt (Total Comp Jahr 1 in EUR)
    oc = cfg.outcome_comp_eur
    med_lut = np.array([oc[lbl]["median"] for lbl in OUTCOME_LABELS], dtype=np.float32)
    sig_lut = np.array([oc[lbl]["sigma"] for lbl in OUTCOME_LABELS], dtype=np.float32)

    base_med = med_lut[outcome]
    base_sig = sig_lut[outcome]

    # Regionaler Gehaltsmultiplikator (Top-Master aus DE öffnet London/Zürich/NYC-Gehaltsbänder)
    region_mult = np.full(n_runs, np.float32(tc.comp_region_mult), dtype=np.float32)
    if tid in ("de_physik", "de_bwl_wima"):
        region_mult = np.where(has_top_master, region_mult + np.float32(0.16), region_mult)

    # Talent-/GPA-Prämie für Absolventen
    gpa_comp_mult = np.clip(
        np.float32(1.0) + np.float32(0.055) * (np.float32(1.80) - uni_gpa) + np.float32(0.035) * z_acad,
        np.float32(0.85),
        np.float32(1.24),
    )
    eff_med = np.where(graduated, base_med * region_mult * gpa_comp_mult, base_med)
    z_comp = rng.standard_normal(n_runs, dtype=np.float32)
    comp_eur = eff_med * np.exp(base_sig * z_comp)

    run_score, _ = _compute_run_scores(outcome, comp_eur, debt_eur, cfg)

    return TrackSimArrays(
        track_id=tid,
        admitted=admitted,
        financed=financed,
        graduated=graduated,
        has_top_master=has_top_master,
        has_scholarship=has_scholarship,
        uni_gpa=uni_gpa,
        outcome=outcome,
        debt_eur=debt_eur,
        comp_eur=comp_eur,
        run_score=run_score,
    )


def _summarize_track(arr: TrackSimArrays, cfg: SimConfig) -> TrackSummary:
    """Aggregiert die 100k+ Arrays eines Tracks in kompakte Kennzahlen."""
    n = float(arr.outcome.shape[0])
    counts = np.bincount(arr.outcome, minlength=5).astype(np.float64)
    probs = counts / n
    # Numerisch exakte Normierung auf Summe 1.0
    probs = probs / probs.sum()
    outcome_probs = {OUTCOME_LABELS[i]: float(probs[i]) for i in range(5)}

    admit_cnt = int(np.sum(arr.admitted))
    fin_cnt = int(np.sum(arr.financed))
    grad_cnt = int(np.sum(arr.graduated))
    master_cnt = int(np.sum(arr.has_top_master))

    admit_rate = admit_cnt / n
    fin_cond = (fin_cnt / admit_cnt) if admit_cnt > 0 else 0.0
    fin_uncond = fin_cnt / n
    grad_cond = (grad_cnt / fin_cnt) if fin_cnt > 0 else 0.0
    master_cond = (master_cnt / grad_cnt) if grad_cnt > 0 else 0.0

    _, risk_adj = _compute_run_scores(arr.outcome, arr.comp_eur, arr.debt_eur, cfg)

    # Bedingter Risk-Adjusted Score (gegeben zugelassen & finanziert) für rationale Portfolio-Rangfolge
    if fin_cnt >= 25:
        _, cond_risk_adj = _compute_run_scores(
            arr.outcome[arr.financed],
            arr.comp_eur[arr.financed],
            arr.debt_eur[arr.financed],
            cfg,
        )
    else:
        cond_risk_adj = risk_adj

    med_debt = float(np.median(arr.debt_eur))
    mean_debt = float(np.mean(arr.debt_eur, dtype=np.float64))
    med_comp = float(np.median(arr.comp_eur))
    p90_comp = float(np.percentile(arr.comp_eur, 90))
    mean_comp = float(np.mean(arr.comp_eur, dtype=np.float64))

    if grad_cnt > 0:
        med_debt_grad = float(np.median(arr.debt_eur[arr.graduated]))
        med_comp_grad = float(np.median(arr.comp_eur[arr.graduated]))
    else:
        med_debt_grad = 0.0
        med_comp_grad = med_comp

    target_prob = outcome_probs[OUTCOME_LABELS[OUTCOME_IB]] + outcome_probs[OUTCOME_LABELS[OUTCOME_QUANT]]

    return TrackSummary(
        track_id=arr.track_id,
        name=TRACK_META[arr.track_id]["name"],
        admit_rate=admit_rate,
        finance_rate_cond=fin_cond,
        finance_rate_uncond=fin_uncond,
        grad_rate_cond=grad_cond,
        top_master_rate_cond=master_cond,
        outcome_probs=outcome_probs,
        target_ib_quant_prob=target_prob,
        median_debt_eur=med_debt,
        median_debt_grad_eur=med_debt_grad,
        mean_debt_eur=mean_debt,
        median_comp_eur=med_comp,
        median_comp_grad_eur=med_comp_grad,
        p90_comp_eur=p90_comp,
        mean_comp_eur=mean_comp,
        risk_adjusted_score=risk_adj,
        conditional_risk_adj_score=cond_risk_adj,
    )


# =============================================================================
# 5. "3-PFEILE-PORTFOLIO-MIX" OPTIMIERUNG (ALLE 20 KOMBINATIONEN)
# =============================================================================

def _evaluate_portfolios(
    track_arrays: Dict[str, TrackSimArrays],
    track_summaries: Dict[str, TrackSummary],
    cfg: SimConfig,
    keep_best_arrays: bool = True,
) -> Tuple[List[PortfolioSummary], PortfolioSummary, Optional[TrackSimArrays]]:
    """
    Evaluiert alle C(6, 3) = 20 möglichen parallelen 3-Track-Bewerbungsportfolios.

    Rationales Entscheidungsmodell in jedem Monte-Carlo-Run i:
    Unter den Tracks des Portfolios, die sowohl zugelassen als auch finanzierbar sind
    (`admitted & financed`), schreibt sich der Bewerber in den Track mit dem höchsten
    bedingten Risk-Adjusted Career Score ein.
    """
    n_runs = next(iter(track_arrays.values())).outcome.shape[0]

    # Präferenzrangfolge der Tracks (bedingt auf Zulassung & Finanzierbarkeit)
    pref_order = sorted(
        TRACK_IDS,
        key=lambda t: (
            track_summaries[t].conditional_risk_adj_score,
            track_summaries[t].target_ib_quant_prob,
        ),
        reverse=True,
    )
    pref_rank = {tid: idx for idx, tid in enumerate(pref_order)}

    # Fallback-Werte für den Fall, dass 0 von 3 Tracks zugelassen & finanziert sind
    fallback_comp = np.full(
        n_runs,
        np.float32(cfg.outcome_comp_eur[OUTCOME_LABELS[OUTCOME_DROPOUT]]["median"]),
        dtype=np.float32,
    )
    fallback_debt = np.zeros(n_runs, dtype=np.float32)

    portfolios: List[PortfolioSummary] = []
    best_combo: Optional[Tuple[str, str, str]] = None
    best_metric = -1.0

    for raw_combo in combinations(TRACK_IDS, 3):
        # Nach rationaler Präferenz sortieren: a > b > c
        a, b, c = sorted(raw_combo, key=lambda t: pref_rank[t])
        arr_a = track_arrays[a]
        arr_b = track_arrays[b]
        arr_c = track_arrays[c]

        pick_a = arr_a.financed
        pick_b = (~pick_a) & arr_b.financed
        pick_c = (~pick_a) & (~pick_b) & arr_c.financed
        any_ok = pick_a | pick_b | pick_c

        outcome = np.where(
            pick_a, arr_a.outcome,
            np.where(pick_b, arr_b.outcome,
                     np.where(pick_c, arr_c.outcome, np.int8(OUTCOME_DROPOUT)))
        )
        debt = np.where(
            pick_a, arr_a.debt_eur,
            np.where(pick_b, arr_b.debt_eur,
                     np.where(pick_c, arr_c.debt_eur, fallback_debt))
        )
        comp = np.where(
            pick_a, arr_a.comp_eur,
            np.where(pick_b, arr_b.comp_eur,
                     np.where(pick_c, arr_c.comp_eur, fallback_comp))
        )

        counts = np.bincount(outcome, minlength=5).astype(np.float64)
        probs = counts / float(n_runs)
        probs = probs / probs.sum()
        outcome_probs = {OUTCOME_LABELS[i]: float(probs[i]) for i in range(5)}

        ib_p = outcome_probs[OUTCOME_LABELS[OUTCOME_IB]]
        q_p = outcome_probs[OUTCOME_LABELS[OUTCOME_QUANT]]
        target_p = ib_p + q_p
        drop_p = outcome_probs[OUTCOME_LABELS[OUTCOME_DROPOUT]]
        _, risk_adj = _compute_run_scores(outcome, comp, debt, cfg)

        label = f"{TRACK_META[a]['short_name']} + {TRACK_META[b]['short_name']} + {TRACK_META[c]['short_name']}"
        usage = {
            a: float(np.mean(pick_a)),
            b: float(np.mean(pick_b)),
            c: float(np.mean(pick_c)),
        }

        ps = PortfolioSummary(
            track_ids=(a, b, c),
            label=label,
            coverage_rate=float(np.mean(any_ok)),
            outcome_probs=outcome_probs,
            target_ib_quant_prob=target_p,
            ib_prob=ib_p,
            quant_prob=q_p,
            dropout_prob=drop_p,
            median_debt_eur=float(np.median(debt)),
            median_comp_eur=float(np.median(comp)),
            p90_comp_eur=float(np.percentile(comp, 90)),
            risk_adjusted_score=risk_adj,
            track_usage_shares=usage,
        )
        portfolios.append(ps)

        # Optimierungsziel: Primär Tier-1 IB + Quant/HF-Wahrscheinlichkeit, sekundär Risk-Adjusted Score
        score_key = target_p * 100.0 + 0.05 * risk_adj
        if score_key > best_metric:
            best_metric = score_key
            best_combo = (a, b, c)

    portfolios.sort(key=lambda p: (p.target_ib_quant_prob, p.risk_adjusted_score), reverse=True)
    best_portfolio = portfolios[0]

    best_arrays: Optional[TrackSimArrays] = None
    if keep_best_arrays and best_combo is not None:
        a, b, c = best_portfolio.track_ids
        arr_a, arr_b, arr_c = track_arrays[a], track_arrays[b], track_arrays[c]
        pick_a = arr_a.financed
        pick_b = (~pick_a) & arr_b.financed
        pick_c = (~pick_a) & (~pick_b) & arr_c.financed
        any_adm = arr_a.admitted | arr_b.admitted | arr_c.admitted
        any_fin = pick_a | pick_b | pick_c

        grad = np.where(pick_a, arr_a.graduated, np.where(pick_b, arr_b.graduated, np.where(pick_c, arr_c.graduated, False)))
        master = np.where(pick_a, arr_a.has_top_master, np.where(pick_b, arr_b.has_top_master, np.where(pick_c, arr_c.has_top_master, False)))
        schol = arr_a.has_scholarship
        gpa = np.where(pick_a, arr_a.uni_gpa, np.where(pick_b, arr_b.uni_gpa, np.where(pick_c, arr_c.uni_gpa, np.float32(3.0))))
        outcome = np.where(pick_a, arr_a.outcome, np.where(pick_b, arr_b.outcome, np.where(pick_c, arr_c.outcome, np.int8(OUTCOME_DROPOUT))))
        debt = np.where(pick_a, arr_a.debt_eur, np.where(pick_b, arr_b.debt_eur, np.where(pick_c, arr_c.debt_eur, fallback_debt)))
        comp = np.where(pick_a, arr_a.comp_eur, np.where(pick_b, arr_b.comp_eur, np.where(pick_c, arr_c.comp_eur, fallback_comp)))
        rscore, _ = _compute_run_scores(outcome, comp, debt, cfg)

        best_arrays = TrackSimArrays(
            track_id="portfolio_best",
            admitted=any_adm,
            financed=any_fin,
            graduated=grad,
            has_top_master=master,
            has_scholarship=schol,
            uni_gpa=gpa,
            outcome=outcome,
            debt_eur=debt,
            comp_eur=comp,
            run_score=rscore,
        )

    return portfolios, best_portfolio, best_arrays


# =============================================================================
# 6. ÖFFENTLICHE ENGINE-APIS (SINGLE RUN, ABITUR-SWEEP, SENSITIVITÄT)
# =============================================================================

def run_simulation(
    profile: ApplicantProfile,
    n_runs: int = 100_000,
    seed: int = 42,
    cfg: Optional[SimConfig] = None,
    keep_arrays: bool = True,
) -> SimulationResult:
    """
    Führt eine vollständige vektorisierte Monte-Carlo-Simulation mit `n_runs`
    Durchläufen über alle 6 Uni-Tracks + alle 20 3-Pfeile-Portfolios durch.
    """
    t0 = time.perf_counter()
    if cfg is None:
        cfg = get_default_config()

    rng = np.random.default_rng(seed)

    # Gemeinsame latente Faktoren (Gaussian Copula über alle 6 Tracks)
    z_acad = rng.standard_normal(n_runs, dtype=np.float32)
    z_inter = rng.standard_normal(n_runs, dtype=np.float32)
    z_schol = rng.standard_normal(n_runs, dtype=np.float32)

    # Gemeinsamer Stipendien-Erfolg (Begabtenförderungswerk)
    p_schol = compute_scholarship_probability(profile, cfg)
    has_scholarship = _correlated_bernoulli(p_schol, z_schol, cfg.correlation_params["rho_scholarship"], rng)

    track_arrays: Dict[str, TrackSimArrays] = {}
    track_summaries: Dict[str, TrackSummary] = {}

    for tid in TRACK_IDS:
        arr = _simulate_track(
            tid=tid,
            profile=profile,
            cfg=cfg,
            n_runs=n_runs,
            z_acad=z_acad,
            z_inter=z_inter,
            z_schol=z_schol,
            has_scholarship=has_scholarship,
            rng=rng,
        )
        track_arrays[tid] = arr
        track_summaries[tid] = _summarize_track(arr, cfg)

    best_single_id = max(
        TRACK_IDS,
        key=lambda t: (track_summaries[t].target_ib_quant_prob, track_summaries[t].risk_adjusted_score),
    )

    portfolios, best_portfolio, best_portfolio_arrays = _evaluate_portfolios(
        track_arrays=track_arrays,
        track_summaries=track_summaries,
        cfg=cfg,
        keep_best_arrays=keep_arrays,
    )

    # Zusammenfassende DataFrames für Tabellen & CSV-Export
    summary_rows = []
    for tid in TRACK_IDS:
        s = track_summaries[tid]
        summary_rows.append({
            "track_id": tid,
            "Track": TRACK_META[tid]["short_name"],
            "Zulassung (%)": round(s.admit_rate * 100.0, 2),
            "Finanzierung bed. (%)": round(s.finance_rate_cond * 100.0, 2),
            "Abschluss bed. (%)": round(s.grad_rate_cond * 100.0, 2),
            "Top-Master bed. (%)": round(s.top_master_rate_cond * 100.0, 2),
            "Tier-1 IB / M&A (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_IB]] * 100.0, 2),
            "Top Quant / HF (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_QUANT]] * 100.0, 2),
            "Ziel gesamt (IB+Quant %)": round(s.target_ib_quant_prob * 100.0, 2),
            "Tier-2 Fin/Cons (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_TIER2]] * 100.0, 2),
            "Tech/Forschung (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_TECH]] * 100.0, 2),
            "Abbruch/Stopp (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_DROPOUT]] * 100.0, 2),
            "Median Schulden (EUR)": round(s.median_debt_eur, 0),
            "Median Schulden Absolv. (EUR)": round(s.median_debt_grad_eur, 0),
            "Median Gehalt J1 (EUR)": round(s.median_comp_eur, 0),
            "P90 Gehalt J1 (EUR)": round(s.p90_comp_eur, 0),
            "Risk-Adjusted Score": round(s.risk_adjusted_score, 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    portfolio_rows = []
    for rank_idx, p in enumerate(portfolios, start=1):
        portfolio_rows.append({
            "Rang": rank_idx,
            "3-Pfeile-Portfolio": p.label,
            "Abdeckung Zugel.&Fin. (%)": round(p.coverage_rate * 100.0, 2),
            "Tier-1 IB / M&A (%)": round(p.ib_prob * 100.0, 2),
            "Top Quant / HF (%)": round(p.quant_prob * 100.0, 2),
            "Ziel gesamt (IB+Quant %)": round(p.target_ib_quant_prob * 100.0, 2),
            "Abbruch/Stopp (%)": round(p.dropout_prob * 100.0, 2),
            "Median Schulden (EUR)": round(p.median_debt_eur, 0),
            "Median Gehalt J1 (EUR)": round(p.median_comp_eur, 0),
            "P90 Gehalt J1 (EUR)": round(p.p90_comp_eur, 0),
            "Risk-Adjusted Score": round(p.risk_adjusted_score, 2),
        })
    portfolio_df = pd.DataFrame(portfolio_rows)

    elapsed = time.perf_counter() - t0
    return SimulationResult(
        profile=profile,
        n_runs=n_runs,
        seed=seed,
        elapsed_seconds=elapsed,
        scholarship_prob=p_schol,
        track_summaries=track_summaries,
        track_arrays=track_arrays if keep_arrays else None,
        best_single_track_id=best_single_id,
        portfolios=portfolios,
        best_portfolio=best_portfolio,
        best_portfolio_arrays=best_portfolio_arrays if keep_arrays else None,
        summary_df=summary_df,
        portfolio_df=portfolio_df,
    )


def run_abitur_sweep(
    profile: ApplicantProfile,
    n_runs: int = 100_000,
    seed: int = 42,
    cfg: Optional[SimConfig] = None,
    grades: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Führt einen vollständigen Abischnitt-Sweep (1,3 bis 2,5 in 0,1er-Schritten)
    mit Common Random Numbers (identischer Seed) durch.

    Rückgabe:
    - track_sweep_df: 13 × 6 Zeilen (alle Tracks × alle Abischnitte)
    - portfolio_sweep_df: 13 Zeilen (optimaler 3-Pfeile-Mix je Abischnitt)
    """
    if cfg is None:
        cfg = get_default_config()
    if grades is None:
        grades = ABITUR_SWEEP_GRADES

    track_rows: List[Dict[str, Any]] = []
    port_rows: List[Dict[str, Any]] = []

    for g in grades:
        g_val = round(float(g), 1)
        p_step = replace(profile, abitur_grade=g_val)
        res = run_simulation(p_step, n_runs=n_runs, seed=seed, cfg=cfg, keep_arrays=False)

        for tid in TRACK_IDS:
            s = res.track_summaries[tid]
            track_rows.append({
                "Abiturschnitt": g_val,
                "track_id": tid,
                "Track": TRACK_META[tid]["short_name"],
                "Track_Full": TRACK_META[tid]["name"],
                "Zulassung (%)": round(s.admit_rate * 100.0, 2),
                "Finanzierung bed. (%)": round(s.finance_rate_cond * 100.0, 2),
                "Tier-1 IB / M&A (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_IB]] * 100.0, 2),
                "Top Quant / HF (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_QUANT]] * 100.0, 2),
                "Ziel gesamt (IB+Quant %)": round(s.target_ib_quant_prob * 100.0, 2),
                "Abbruch/Stopp (%)": round(s.outcome_probs[OUTCOME_LABELS[OUTCOME_DROPOUT]] * 100.0, 2),
                "Median Schulden (EUR)": round(s.median_debt_eur, 0),
                "Median Gehalt J1 (EUR)": round(s.median_comp_eur, 0),
                "Risk-Adjusted Score": round(s.risk_adjusted_score, 2),
            })

        bp = res.best_portfolio
        bs = res.track_summaries[res.best_single_track_id]
        port_rows.append({
            "Abiturschnitt": g_val,
            "Bester Single-Track": TRACK_META[res.best_single_track_id]["short_name"],
            "Single Ziel (%)": round(bs.target_ib_quant_prob * 100.0, 2),
            "Optimaler 3-Pfeile-Mix": bp.label,
            "Portfolio Tier-1 IB (%)": round(bp.ib_prob * 100.0, 2),
            "Portfolio Quant/HF (%)": round(bp.quant_prob * 100.0, 2),
            "Portfolio Ziel (IB+Quant %)": round(bp.target_ib_quant_prob * 100.0, 2),
            "Portfolio Abbruch (%)": round(bp.dropout_prob * 100.0, 2),
            "Portfolio Median Schulden (EUR)": round(bp.median_debt_eur, 0),
            "Portfolio Median Gehalt (EUR)": round(bp.median_comp_eur, 0),
            "Portfolio Risk-Adjusted Score": round(bp.risk_adjusted_score, 2),
            "Stipendium-Chance (%)": round(res.scholarship_prob * 100.0, 2),
        })

    return pd.DataFrame(track_rows), pd.DataFrame(port_rows)


def run_sensitivity_analysis(
    profile: ApplicantProfile,
    target_scope: str = "portfolio_best",
    n_runs: int = 50_000,
    seed: int = 42,
    cfg: Optional[SimConfig] = None,
) -> pd.DataFrame:
    """
    Tornado-Sensitivitätsanalyse mit Common Random Numbers (CRN).
    Vergleicht alle zentralen strategischen Hebel relativ zum Basisprofil.

    `target_scope`: `"portfolio_best"` oder eine `track_id` aus `TRACK_IDS`.
    """
    if cfg is None:
        cfg = get_default_config()

    def _eval_metric(p: ApplicantProfile) -> float:
        r = run_simulation(p, n_runs=n_runs, seed=seed, cfg=cfg, keep_arrays=False)
        if target_scope == "portfolio_best":
            return r.best_portfolio.target_ib_quant_prob * 100.0
        return r.track_summaries[target_scope].target_ib_quant_prob * 100.0

    base_val = _eval_metric(profile)

    # Definition der Hebel: (Anzeigename, Low-Profil, High-Profil, Low-Label, High-Label)
    levers: List[Tuple[str, ApplicantProfile, ApplicantProfile, str, str]] = [
        (
            "Lern-Disziplin & Quant-Resilienz (±20 %)",
            replace(profile, discipline=float(np.clip(profile.discipline * 0.80, 0.10, 1.00))),
            replace(profile, discipline=float(np.clip(profile.discipline * 1.20, 0.10, 1.00))),
            f"Disziplin {np.clip(profile.discipline * 0.80, 0.10, 1.00):.2f}",
            f"Disziplin {np.clip(profile.discipline * 1.20, 0.10, 1.00):.2f}",
        ),
        (
            "Abiturschnitt (±0,3 Notenstufen)",
            replace(profile, abitur_grade=float(np.clip(round(profile.abitur_grade + 0.3, 1), 1.3, 2.5))),
            replace(profile, abitur_grade=float(np.clip(round(profile.abitur_grade - 0.3, 1), 1.3, 2.5))),
            f"Abi {np.clip(round(profile.abitur_grade + 0.3, 1), 1.3, 2.5):.1f}",
            f"Abi {np.clip(round(profile.abitur_grade - 0.3, 1), 1.3, 2.5):.1f}",
        ),
        (
            "Mathe & Physik Abiturnote (±2 Pkt.)",
            replace(
                profile,
                math_points=int(np.clip(profile.math_points - 2, 0, 15)),
                physics_points=int(np.clip(profile.physics_points - 2, 0, 15)),
            ),
            replace(
                profile,
                math_points=int(np.clip(profile.math_points + 2, 0, 15)),
                physics_points=int(np.clip(profile.physics_points + 2, 0, 15)),
            ),
            f"M/P {max(0, profile.math_points - 2)}/{max(0, profile.physics_points - 2)} Pkt",
            f"M/P {min(15, profile.math_points + 2)}/{min(15, profile.physics_points + 2)} Pkt",
        ),
        (
            "M&A-Schülerpraktikum (AUS vs. AN)",
            replace(profile, has_ma_internship=False),
            replace(profile, has_ma_internship=True),
            "Ohne M&A-Praktikum",
            "Mit M&A-Praktikum",
        ),
        (
            "SAT-Score (±100 Pkt. / ACT ±2)",
            replace(
                profile,
                sat_score=int(np.clip(profile.sat_score - 100, 1200, 1600)),
                act_score=int(np.clip(profile.act_score - 2, 26, 36)),
            ),
            replace(
                profile,
                sat_score=int(np.clip(profile.sat_score + 100, 1200, 1600)),
                act_score=int(np.clip(profile.act_score + 2, 26, 36)),
            ),
            f"SAT {max(1200, profile.sat_score - 100)} / ACT {max(26, profile.act_score - 2)}",
            f"SAT {min(1600, profile.sat_score + 100)} / ACT {min(36, profile.act_score + 2)}",
        ),
        (
            "HSG-Eignungstest-Perzentil (±15 %-Pkt.)",
            replace(profile, hsg_percentile=int(np.clip(profile.hsg_percentile - 15, 0, 100))),
            replace(profile, hsg_percentile=int(np.clip(profile.hsg_percentile + 15, 0, 100))),
            f"HSG {max(0, profile.hsg_percentile - 15)} %",
            f"HSG {min(100, profile.hsg_percentile + 15)} %",
        ),
        (
            "First-Gen-Student (AUS vs. AN)",
            replace(profile, is_first_gen=False),
            replace(profile, is_first_gen=True),
            "Kein First-Gen",
            "First-Gen aktiv",
        ),
        (
            "GdB 100 / Härtefallantrag (AUS vs. AN)",
            replace(profile, has_gdb_100=False),
            replace(profile, has_gdb_100=True),
            "Ohne GdB 100",
            "Mit GdB 100",
        ),
        (
            "Jugend gründet + Tradity e.V. (AUS vs. AN)",
            replace(profile, has_jugend_gruendet=False, has_tradity=False),
            replace(profile, has_jugend_gruendet=True, has_tradity=True),
            "Ohne JG & Tradity",
            "Mit JG & Tradity",
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for name, p_low, p_high, lbl_low, lbl_high in levers:
        v_low = _eval_metric(p_low)
        v_high = _eval_metric(p_high)
        delta_low = v_low - base_val
        delta_high = v_high - base_val
        spread = abs(v_high - v_low)
        rows.append({
            "Hebel": name,
            "Basis (%)": round(base_val, 2),
            "Low-Szenario": lbl_low,
            "High-Szenario": lbl_high,
            "Low (%)": round(v_low, 2),
            "High (%)": round(v_high, 2),
            "Delta Low (%-Pkt.)": round(delta_low, 2),
            "Delta High (%-Pkt.)": round(delta_high, 2),
            "Gesamt-Spread (%-Pkt.)": round(spread, 2),
        })

    df = pd.DataFrame(rows).sort_values("Gesamt-Spread (%-Pkt.)", ascending=True).reset_index(drop=True)
    return df


def build_sankey_data(
    arr: TrackSimArrays,
    source_label: str,
) -> Dict[str, Any]:
    """
    Erzeugt exakte, massenerhaltende Knoten- & Flussdaten für das interaktive
    Tab-3-Sankey-Diagramm:
      Ebene 0: Bachelor-Track
      Ebene 1: Zulassung & Finanzierung (Zugelassen & Finanziert | Finanzierungs-Stopp | Nicht zugelassen)
      Ebene 2: Abschluss & Master-Sprung (Top-Master | Standard-Master/Direkt | Studienabbruch)
      Ebene 3: Finales Karriere-Outcome (alle 5 Zielzustände)
    """
    admitted = arr.admitted
    financed = arr.financed
    graduated = arr.graduated
    top_master = arr.has_top_master
    outcome = arr.outcome

    n_not_admitted = int(np.sum(~admitted))
    n_fin_stop = int(np.sum(admitted & (~financed)))
    n_fin_ok = int(np.sum(financed))

    n_dropout_study = int(np.sum(financed & (~graduated)))
    n_top_master = int(np.sum(graduated & top_master))
    n_std_grad = int(np.sum(graduated & (~top_master)))

    node_labels = [
        source_label,                                  # 0
        "Zugelassen & Finanziert",                     # 1
        "Finanzierungs-Stopp",                         # 2
        "Nicht zugelassen",                            # 3
        "Top-Master (ETH/HSG/US/UK)",                  # 4
        "Standard-Master / Direkt",                    # 5
        "Studienabbruch",                              # 6
        OUTCOME_LABELS[OUTCOME_IB],                    # 7
        OUTCOME_LABELS[OUTCOME_QUANT],                 # 8
        OUTCOME_LABELS[OUTCOME_TIER2],                 # 9
        OUTCOME_LABELS[OUTCOME_TECH],                  # 10
        OUTCOME_LABELS[OUTCOME_DROPOUT],               # 11
    ]

    node_colors = [
        "#38BDF8",  # 0 Source
        "#10B981",  # 1 Finanziert
        "#F97316",  # 2 Finanzierungs-Stopp
        "#64748B",  # 3 Nicht zugelassen
        "#00D4FF",  # 4 Top-Master
        "#6366F1",  # 5 Standard/Direkt
        "#EF4444",  # 6 Studienabbruch
        "#F59E0B",  # 7 Tier-1 IB
        "#00D4FF",  # 8 Quant/HF
        "#6366F1",  # 9 Tier-2
        "#10B981",  # 10 Tech/Forschung
        "#EF4444",  # 11 Abbruch/Stopp
    ]

    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    link_colors: List[str] = []

    def _add(src: int, dst: int, val: int, rgba: str) -> None:
        if val > 0:
            sources.append(src)
            targets.append(dst)
            values.append(val)
            link_colors.append(rgba)

    # Ebene 0 -> Ebene 1
    _add(0, 1, n_fin_ok, "rgba(16, 185, 129, 0.35)")
    _add(0, 2, n_fin_stop, "rgba(249, 115, 22, 0.35)")
    _add(0, 3, n_not_admitted, "rgba(100, 116, 139, 0.30)")

    # Ebene 1 (Fehlschläge) -> Ebene 3 (Abbruch / Finanzierungs-Stopp)
    _add(2, 11, n_fin_stop, "rgba(249, 115, 22, 0.35)")
    _add(3, 11, n_not_admitted, "rgba(100, 116, 139, 0.25)")

    # Ebene 1 (Zugelassen & Finanziert) -> Ebene 2
    _add(1, 4, n_top_master, "rgba(0, 212, 255, 0.40)")
    _add(1, 5, n_std_grad, "rgba(99, 102, 241, 0.35)")
    _add(1, 6, n_dropout_study, "rgba(239, 68, 68, 0.35)")

    # Ebene 2 (Studienabbruch) -> Ebene 3 (Abbruch / Stopp)
    _add(6, 11, n_dropout_study, "rgba(239, 68, 68, 0.35)")

    # Ebene 2 (Absolventen) -> Ebene 3 (Outcomes 0..3)
    out_rgba = [
        "rgba(245, 158, 11, 0.45)",  # IB
        "rgba(0, 212, 255, 0.45)",   # Quant
        "rgba(99, 102, 241, 0.35)",  # Tier-2
        "rgba(16, 185, 129, 0.35)",  # Tech
    ]
    tm_mask = graduated & top_master
    st_mask = graduated & (~top_master)

    for code in range(4):
        n_tm_out = int(np.sum(tm_mask & (outcome == code)))
        n_st_out = int(np.sum(st_mask & (outcome == code)))
        _add(4, 7 + code, n_tm_out, out_rgba[code])
        _add(5, 7 + code, n_st_out, out_rgba[code])

    return {
        "node_labels": node_labels,
        "node_colors": node_colors,
        "sources": sources,
        "targets": targets,
        "values": values,
        "link_colors": link_colors,
    }
