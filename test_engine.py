"""
Unit-Tests für die vektorisierte Monte-Carlo-Karrieresimulations-Engine (`engine.py`).

Verifiziert:
1. Fehlerfreie Ausführung von 100.000 Runs über alle 6 Tracks + 20 Portfolios.
2. Strikt reproduzierbare Ergebnisse bei identischem Random Seed.
3. Performance-Anforderung: 100.000 Runs in < 1,5 s (strikt < 2,0 s).
4. Exakte Wahrscheinlichkeits-Normierung (Summe aller 5 Outcomes == 100 % / 1,0).
5. Fachliche Monotonie- & Plausibilitätsprüfungen (ETH-70%-Regel, DE-Physik NC-Freiheit,
   Disziplin-Effekt auf Physik-Abbruch, GdB-100-Bonus bei Mannheim, US First-Gen/SAT,
   UK Post-Brexit-Finanzierungsengpass, Sankey-Massenerhaltung, Config-Override).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from config import (
    ABITUR_SWEEP_GRADES,
    ApplicantProfile,
    OUTCOME_DROPOUT,
    OUTCOME_LABELS,
    TRACK_IDS,
    get_default_config,
)
from engine import (
    build_sankey_data,
    run_abitur_sweep,
    run_sensitivity_analysis,
    run_simulation,
)


def test_100k_performance_and_no_nans():
    """100.000 Simulationen über alle 6 Tracks + 20 Portfolios müssen < 2,0 s (< 1,5 s) laufen."""
    profile = ApplicantProfile()
    t0 = time.perf_counter()
    res = run_simulation(profile, n_runs=100_000, seed=42, keep_arrays=True)
    wall_time = time.perf_counter() - t0

    assert res.n_runs == 100_000
    assert wall_time < 2.0, f"Laufzeit zu hoch: {wall_time:.3f}s (Limit: 2.0s)"
    assert res.elapsed_seconds < 1.5, f"Engine-Laufzeit überschritt 1.5s-Ziel: {res.elapsed_seconds:.3f}s"

    assert len(res.track_summaries) == 6
    assert len(res.portfolios) == 20  # C(6, 3) = 20
    assert not res.summary_df.isna().any().any()
    assert not res.portfolio_df.isna().any().any()

    for tid in TRACK_IDS:
        arr = res.track_arrays[tid]
        assert arr.outcome.shape == (100_000,)
        assert np.all(np.isfinite(arr.debt_eur))
        assert np.all(np.isfinite(arr.comp_eur))
        assert np.all(np.isfinite(arr.run_score))
        assert np.all(arr.debt_eur >= 0.0)
        assert np.all(arr.comp_eur > 0.0)


def test_reproducibility_with_fixed_seed():
    """Zwei Läufe mit identischem Seed müssen bit-genau identische Ergebnisse liefern."""
    profile = ApplicantProfile(abitur_grade=1.6, discipline=0.80)
    res1 = run_simulation(profile, n_runs=100_000, seed=1337, keep_arrays=True)
    res2 = run_simulation(profile, n_runs=100_000, seed=1337, keep_arrays=True)

    for tid in TRACK_IDS:
        s1 = res1.track_summaries[tid]
        s2 = res2.track_summaries[tid]
        assert s1.outcome_probs == s2.outcome_probs
        assert s1.median_debt_eur == s2.median_debt_eur
        assert s1.median_comp_eur == s2.median_comp_eur
        assert s1.risk_adjusted_score == s2.risk_adjusted_score
        np.testing.assert_array_equal(res1.track_arrays[tid].outcome, res2.track_arrays[tid].outcome)

    assert res1.best_portfolio.track_ids == res2.best_portfolio.track_ids
    assert res1.best_portfolio.target_ib_quant_prob == res2.best_portfolio.target_ib_quant_prob


def test_probabilities_sum_to_100_percent():
    """Alle 5 Karriere-Outcomes müssen für jeden Track und jedes Portfolio exakt auf 100 % summieren."""
    profile = ApplicantProfile(abitur_grade=1.8)
    res = run_simulation(profile, n_runs=100_000, seed=42, keep_arrays=False)

    for tid in TRACK_IDS:
        probs = res.track_summaries[tid].outcome_probs
        assert set(probs.keys()) == set(OUTCOME_LABELS)
        total = sum(probs.values())
        assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9), f"{tid}: Summe={total}"
        for lbl, val in probs.items():
            assert 0.0 <= val <= 1.0, f"{tid}/{lbl}: ungültige Wahrscheinlichkeit {val}"

    for port in res.portfolios:
        total = sum(port.outcome_probs.values())
        assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9), f"{port.label}: Summe={total}"


def test_eth_70_percent_rule():
    """ETH Zürich: >= 10,5 Pkt Schnitt (Mathe/Physik/Sprache) => prüfungsfrei (~100 %); darunter Aufnahmeprüfung."""
    p_pass = ApplicantProfile(math_points=12, physics_points=11, language_points=9)   # Schnitt 10.67 >= 10.5
    p_fail = ApplicantProfile(math_points=10, physics_points=10, language_points=10)  # Schnitt 10.00 < 10.5

    assert p_pass.eth_exam_free is True
    assert p_fail.eth_exam_free is False

    r_pass = run_simulation(p_pass, n_runs=25_000, seed=7, keep_arrays=False)
    r_fail = run_simulation(p_fail, n_runs=25_000, seed=7, keep_arrays=False)

    assert r_pass.track_summaries["eth_physik"].admit_rate == 1.0
    assert r_fail.track_summaries["eth_physik"].admit_rate < 0.55


def test_de_physik_downside_protection_and_discipline_effect():
    """DE Top-Physik bleibt auch bei Abi 2,5 zu 100 % NC-frei; Disziplin reduziert Grundstudium-Abbruch drastisch."""
    p_low_disc = ApplicantProfile(abitur_grade=2.5, discipline=0.35, math_points=10, physics_points=10)
    p_high_disc = ApplicantProfile(abitur_grade=2.5, discipline=0.90, math_points=14, physics_points=14)

    r_low = run_simulation(p_low_disc, n_runs=30_000, seed=42, keep_arrays=False)
    r_high = run_simulation(p_high_disc, n_runs=30_000, seed=42, keep_arrays=False)

    assert r_low.track_summaries["de_physik"].admit_rate == 1.0
    assert r_high.track_summaries["de_physik"].admit_rate == 1.0
    # Höhere Disziplin & Mathe/Physik erhöhen die Abschlussquote im Physikstudium massiv
    assert r_high.track_summaries["de_physik"].grad_rate_cond > r_low.track_summaries["de_physik"].grad_rate_cond + 0.30
    # Bei Abi 2,5 schützt DE Top-Physik klar gegenüber DE BWL/WiMa (Downside Protection)
    assert (
        r_high.track_summaries["de_physik"].target_ib_quant_prob
        > r_high.track_summaries["de_bwl_wima"].target_ib_quant_prob + 0.15
    )


def test_profile_factors_and_uk_post_brexit():
    """GdB 100 boostet Mannheim/DE BWL; First-Gen+SAT boosten US Elite; UK Bachelor scheitert meist an Finanzierung."""
    p_gdb_on = ApplicantProfile(abitur_grade=2.0, has_gdb_100=True)
    p_gdb_off = ApplicantProfile(abitur_grade=2.0, has_gdb_100=False)
    r_on = run_simulation(p_gdb_on, n_runs=30_000, seed=42, keep_arrays=False)
    r_off = run_simulation(p_gdb_off, n_runs=30_000, seed=42, keep_arrays=False)
    assert r_on.track_summaries["de_bwl_wima"].admit_rate > r_off.track_summaries["de_bwl_wima"].admit_rate + 0.05

    p_us_strong = ApplicantProfile(abitur_grade=1.4, sat_score=1560, act_score=35, is_first_gen=True)
    p_us_weak = ApplicantProfile(abitur_grade=1.4, sat_score=1320, act_score=29, is_first_gen=False)
    r_us_s = run_simulation(p_us_strong, n_runs=30_000, seed=42, keep_arrays=False)
    r_us_w = run_simulation(p_us_weak, n_runs=30_000, seed=42, keep_arrays=False)
    assert r_us_s.track_summaries["us_elite"].admit_rate > r_us_w.track_summaries["us_elite"].admit_rate * 2.5

    # UK Post-Brexit Bachelor: extrem niedrige Finanzierungsquote & hohe Absolventen-Schulden
    uk = r_us_s.track_summaries["uk_top"]
    assert uk.finance_rate_cond < 0.25
    assert uk.median_debt_grad_eur > 50_000.0


def test_portfolio_dominates_single_tracks_and_sankey_conservation():
    """Der optimale 3-Pfeile-Mix schlägt jeden einzelnen Track; Sankey-Flüsse sind 100 % massenerhaltend."""
    profile = ApplicantProfile(abitur_grade=1.6)
    res = run_simulation(profile, n_runs=50_000, seed=42, keep_arrays=True)

    best_single = res.track_summaries[res.best_single_track_id]
    assert res.best_portfolio.target_ib_quant_prob >= best_single.target_ib_quant_prob

    # Sankey-Massenerhaltung für alle 6 Tracks + bestes Portfolio prüfen
    for tid in TRACK_IDS:
        sk = build_sankey_data(res.track_arrays[tid], tid)
        out_from_root = sum(v for s, v in zip(sk["sources"], sk["values"]) if s == 0)
        in_to_outcomes = sum(v for t, v in zip(sk["targets"], sk["values"]) if t >= 7)
        assert out_from_root == 50_000
        assert in_to_outcomes == 50_000

    sk_port = build_sankey_data(res.best_portfolio_arrays, "3-Pfeile-Portfolio")
    assert sum(v for t, v in zip(sk_port["targets"], sk_port["values"]) if t >= 7) == 50_000


def test_sweep_sensitivity_and_config_override():
    """Abischnitt-Sweep (1,3–2,5), Sensitivitätsanalyse und Live-Config-Override laufen sauber."""
    profile = ApplicantProfile(abitur_grade=1.6)
    track_sw, port_sw = run_abitur_sweep(profile, n_runs=10_000, seed=42)
    assert len(port_sw) == len(ABITUR_SWEEP_GRADES) == 13
    assert len(track_sw) == 13 * 6
    assert not track_sw.isna().any().any()

    sens_df = run_sensitivity_analysis(profile, target_scope="portfolio_best", n_runs=10_000, seed=42)
    assert len(sens_df) == 9
    assert not sens_df.isna().any().any()

    # Annahmen-Editor Override testen
    cfg = get_default_config()
    records = cfg.tracks_as_records()
    for row in records:
        if row["track_id"] == "uk_top":
            row["base_finance"] = 0.90
            row["finance_with_scholarship"] = 0.95
    cfg.apply_track_records(records)
    r_override = run_simulation(profile, n_runs=10_000, seed=42, cfg=cfg, keep_arrays=False)
    assert r_override.track_summaries["uk_top"].finance_rate_cond > 0.85
