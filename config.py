"""
Konfigurationsmodul für die Monte-Carlo-Karrieresimulation (IB / Quant / Hedgefonds).

Enthält sämtliche dokumentierten Wahrscheinlichkeitsfunktionen, Studiengebühren,
Lebenshaltungskosten, Schwellenwerte, Profil-Multiplikatoren, Gehalts- und
Schuldenverteilungen sowie Risiko-Score-Gewichtungen.

Alle Parameter sind sauber gekapselt in Dataclasses/Dictionaries, sodass sie sowohl
von der vektorisierten Engine (`engine.py`) genutzt als auch im Dashboard-Tab 4
(Annahmen-Editor) transparent eingesehen und live überschrieben werden können.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any
import numpy as np


# =============================================================================
# 1. KONSTANTEN: OUTCOMES, FARBEN & TRACK-METADATEN
# =============================================================================

OUTCOME_LABELS: List[str] = [
    "Tier-1 IB / M&A",
    "Top Quant / Hedgefonds",
    "Tier-2 Finance / Consulting / Corporate Finance",
    "Tech / Forschung / Sonstiges",
    "Abbruch / Finanzierungs-Stopp",
]

OUTCOME_SHORT_LABELS: List[str] = [
    "Tier-1 IB / M&A",
    "Top Quant / HF",
    "Tier-2 Fin / Cons.",
    "Tech / Forschung",
    "Abbruch / Stopp",
]

OUTCOME_IB = 0
OUTCOME_QUANT = 1
OUTCOME_TIER2 = 2
OUTCOME_TECH = 3
OUTCOME_DROPOUT = 4

OUTCOME_COLORS: Dict[str, str] = {
    "Tier-1 IB / M&A": "#F59E0B",                          # Bloomberg Amber/Gold
    "Top Quant / Hedgefonds": "#00D4FF",                   # Electric Quant Cyan
    "Tier-2 Finance / Consulting / Corporate Finance": "#6366F1",  # Indigo
    "Tech / Forschung / Sonstiges": "#10B981",             # Emerald
    "Abbruch / Finanzierungs-Stopp": "#EF4444",            # Crimson Risk
}

TRACK_IDS: List[str] = [
    "us_elite",
    "hsg",
    "eth_physik",
    "de_physik",
    "de_bwl_wima",
    "uk_top",
]

TRACK_META: Dict[str, Dict[str, str]] = {
    "us_elite": {
        "name": "US Elite (Need-Blind Full Aid)",
        "short_name": "US Elite",
        "subtitle": "Harvard, Yale, Princeton, MIT, Dartmouth, Amherst (Double Major Physics & Econ)",
        "color": "#00D4FF",
        "region": "US",
    },
    "hsg": {
        "name": "HSG St. Gallen (BWL / Quant Econ)",
        "short_name": "HSG St. Gallen",
        "subtitle": "B.A. BWL / VWL / Quantitative Economics (Assessmentjahr + IB-Pipeline)",
        "color": "#F59E0B",
        "region": "CH",
    },
    "eth_physik": {
        "name": "ETH Zürich (B.Sc. Physik)",
        "short_name": "ETH Zürich",
        "subtitle": "B.Sc. Physik -> ETH MSc Quantitative Finance / Quant-Hedgefonds",
        "color": "#38BDF8",
        "region": "CH",
    },
    "de_physik": {
        "name": "DE Top-Physik -> Top-Master",
        "short_name": "DE Top-Physik",
        "subtitle": "TUM, Heidelberg, LMU, KIT, RWTH (NC-frei) -> ETH QF / HSG MBF / US MFE / Oxbridge",
        "color": "#10B981",
        "region": "DE",
    },
    "de_bwl_wima": {
        "name": "DE Top-BWL / WiMa",
        "short_name": "DE BWL / WiMa",
        "subtitle": "Uni Mannheim (BWL & WiMa), WHU, Frankfurt School (NC/Härtefall & UGV)",
        "color": "#A855F7",
        "region": "DE",
    },
    "uk_top": {
        "name": "UK Top-Unis (Post-Brexit Bachelor)",
        "short_name": "UK Top-Unis",
        "subtitle": "LSE, Oxford, Imperial, Warwick (~35k-45k £/Jahr Overseas Fees)",
        "color": "#FB7185",
        "region": "UK",
    },
}

ABITUR_SWEEP_GRADES: np.ndarray = np.round(np.arange(1.3, 2.55, 0.1), 1)


# =============================================================================
# 2. BEWERBER-PROFIL DATACLASS
# =============================================================================

@dataclass
class ApplicantProfile:
    """Interaktives Bewerberprofil aus der Streamlit-Sidebar."""

    abitur_grade: float = 1.5           # 1.3 bis 2.5
    math_points: int = 13               # 0 bis 15 Abiturpunkte
    physics_points: int = 13            # 0 bis 15 Abiturpunkte
    language_points: int = 11           # 0 bis 15 Abiturpunkte (für ETH 70%-Regel)
    sat_score: int = 1540               # 1200 bis 1600
    act_score: int = 35                 # 26 bis 36
    hsg_percentile: int = 80            # 0 bis 100 %
    discipline: float = 0.75            # 0.1 bis 1.0 (Lern-Disziplin & Quant-Resilienz)

    # Profil-Faktoren (Standardmäßig aktiv)
    has_ma_internship: bool = True      # M&A-Schülerpraktikum (Boutique / Eichenfels)
    has_jugend_gruendet: bool = True    # Top 3 % Jugend gründet
    has_tradity: bool = True            # Tradity e.V. (Finanzbildung an Schule)
    is_first_gen: bool = True           # First-Gen-Student
    has_gdb_100: bool = True            # GdB 100 (Ohrmuscheldysplasie 3. Grades)

    @property
    def eth_avg_points(self) -> float:
        """Durchschnitt aus Mathe, Physik, Sprache für die ETH-Zürich-Regel (>= 10,5 Pkt = 70%)."""
        return (self.math_points + self.physics_points + self.language_points) / 3.0

    @property
    def eth_exam_free(self) -> bool:
        """True, wenn prüfungsfreie Zulassung an der ETH Zürich (>= 10,5 Punkte / 70%) erreicht."""
        return self.eth_avg_points >= 10.5

    @property
    def effective_sat(self) -> float:
        """
        Bestes standardisiertes US-Testergebnis auf SAT-Skala (1200-1600).
        Offizielle CollegeBoard/ACT-Konkordanztabelle: ACT 36~1590, 35~1540, 34~1500, 33~1460, ...
        """
        act_to_sat = {
            36: 1590, 35: 1540, 34: 1500, 33: 1460, 32: 1430,
            31: 1400, 30: 1370, 29: 1340, 28: 1310, 27: 1280, 26: 1240,
        }
        act_equiv = act_to_sat.get(int(self.act_score), 1240)
        return float(max(self.sat_score, act_equiv))


# =============================================================================
# 3. TRACK-PARAMETER & SIMULATIONS-KONFIGURATION
# =============================================================================

@dataclass
class TrackConfig:
    """
    Dokumentierte Kernparameter eines Studien-Tracks.
    Jeder Parameter ist im Tab 4 (Annahmen-Editor) einsehbar und live editierbar.
    """
    track_id: str
    # Stufe 1: Basis-Zulassungswahrscheinlichkeit (vor Profil- & Noten-Modifikation)
    base_admit: float
    # Stufe 2: Basis-Finanzierungssicherheit (ohne Stipendium bzw. Basiswert)
    base_finance: float
    # Stufe 2b: Finanzierungssicherheit MIT Begabtenförderungswerk / Vollstipendium
    finance_with_scholarship: float
    # Stufe 3: Basis-Abbrecherquote bei durchschnittlicher Disziplin (0,60) & Mathe (12 Pkt)
    base_dropout: float
    # Stufe 3b: Stärke des Disziplin-/Mathe-Hebels auf Abbruchquote
    dropout_discipline_sensitivity: float
    # Stufe 4: Wahrscheinlichkeit für Top-Master bei guter Bachelor-GPA (<= 1,7)
    top_master_prob_high_gpa: float
    # Stufe 4b: Wahrscheinlichkeit für Top-Master bei mittlerer Bachelor-GPA (1,7 - 2,1)
    top_master_prob_mid_gpa: float
    # Stufe 5: Bedingte Ziel-Basisgewichte bei Abschluss (ohne Top-Master-Boost)
    # [Tier-1 IB, Top Quant/HF, Tier-2 Fin/Cons, Tech/Forschung] (werden normalisiert)
    placement_ib: float
    placement_quant: float
    placement_tier2: float
    placement_tech: float
    # Stufe 5b: Zusätzlicher Boost bei Top-Master-Abschluss (Multiplikator auf IB & Quant)
    top_master_ib_mult: float
    top_master_quant_mult: float
    # Finanzmetriken (EUR): Typische Schuldenlast bei Abschluss ohne/mit Stipendium
    debt_no_scholarship_eur: float
    debt_with_scholarship_eur: float
    debt_top_master_addon_eur: float
    # Regionaler Gehaltsmultiplikator (z. B. US NYC/Quant vs. CH Zürich vs. London vs. DE)
    comp_region_mult: float
    # Kurze fachliche Dokumentation
    notes: str = ""


def default_track_configs() -> Dict[str, TrackConfig]:
    """
    Realistische, dokumentierte Standardparameter für alle 6 Uni-Tracks (Stand 2025/2026).
    """
    return {
        "us_elite": TrackConfig(
            track_id="us_elite",
            # Portfolio aus ~7 US Need-Blind Full-Aid Unis (Harvard, Yale, Princeton, MIT,
            # Dartmouth, Amherst, Bowdoin): Basis-Chance ~3,0 % für solide Internationals
            # ohne besondere Spikes; mit First-Gen + Spikes + 1540 SAT/35 ACT bis ~12-16 %.
            base_admit=0.030,
            # Need-Blind Full Aid deckt nahezu 100 % von Tuition + Room & Board für DE-Familien.
            base_finance=0.960,
            finance_with_scholarship=0.990,
            # Sehr hohe Abschlussquote an US-Eliten (~94-96 % Retention).
            base_dropout=0.055,
            dropout_discipline_sensitivity=0.10,
            # 4-jähriges Double Major -> direkter Wall-Street-/Quant-Einstieg ist Standard.
            top_master_prob_high_gpa=0.18,
            top_master_prob_mid_gpa=0.08,
            # Extrem starke Recruiting-Pipeline für Physics & Econ Double Major.
            placement_ib=0.38,
            placement_quant=0.33,
            placement_tier2=0.17,
            placement_tech=0.12,
            top_master_ib_mult=1.05,
            top_master_quant_mult=1.20,
            # Minimale Restschulden (Flüge, Bücher, Work-Study-Differenz).
            debt_no_scholarship_eur=6_500.0,
            debt_with_scholarship_eur=3_000.0,
            debt_top_master_addon_eur=12_000.0,
            comp_region_mult=1.35,  # US NYC / Chicago Wall Street & Quant TC
            notes="Need-Blind Full Aid (7-Uni-Slate); sehr niedrige Zulassung, extrem hohe Finanzierungssicherheit & Top-Placement.",
        ),
        "hsg": TrackConfig(
            track_id="hsg",
            # Zulassung primär über HSG-Zulassungstest + Video-Interview (weniger NC-abhängig).
            # Basis bei 80. Perzentil ~0,62 (wird dynamisch per Logistikfunktion skaliert).
            base_admit=0.60,
            # Engpass Finanzierung (~3.129 CHF/Sem. Ausländergebühr + ~1.650 CHF/M Lebenshaltung):
            # Ohne Stipendium über Auslands-BAföG + HSG-Darlehen + CH-Werkstudent machbar (~54 %).
            base_finance=0.54,
            # Mit Begabtenförderungswerk (Studienstiftung / sdw) + Auslands-BAföG sehr sicher (~94 %).
            finance_with_scholarship=0.94,
            # HSG Assessmentjahr siebt ~22-28 % aus.
            base_dropout=0.24,
            dropout_discipline_sensitivity=0.32,
            # Viele HSG-B.A. gehen nach Off-Cycle-Praktika direkt ins IB oder in MBF/SIM/LSE.
            top_master_prob_high_gpa=0.62,
            top_master_prob_mid_gpa=0.32,
            # DACH-Powerhouse für Tier-1 IB/M&A (London, Frankfurt, Zürich); geringerer Quant-Anteil.
            placement_ib=0.44,
            placement_quant=0.06,
            placement_tier2=0.36,
            placement_tech=0.14,
            top_master_ib_mult=1.22,
            top_master_quant_mult=1.40,
            debt_no_scholarship_eur=34_000.0,
            debt_with_scholarship_eur=13_000.0,
            debt_top_master_addon_eur=14_000.0,
            comp_region_mult=1.15,  # London / Zürich / Frankfurt Mix
            notes="Test-basiert (wenig NC-Druck); Finanzierung stark abhängig von Stipendium (Abi <= 1,5), BAföG & CH-Job.",
        ),
        "eth_physik": TrackConfig(
            track_id="eth_physik",
            # Prüfungsfrei (100 %) bei >= 10,5 Pkt (70 %) in Mathe/Physik/Sprache;
            # darunter ETH-Aufnahmeprüfung (~32 % Basis-Passrate).
            base_admit=1.00,
            # ~2.190 CHF/Sem. Ausländergebühr (Verdreifachung ab 2025) + Zürich Lebenshaltung (~1.900 CHF/M),
            # zudem wegen harter Basisprüfung im 1. Jahr kaum Nebenjob möglich.
            base_finance=0.49,
            finance_with_scholarship=0.92,
            # Harte Basisprüfung im 1. Jahr (~44 % Gesamt-Abbruch bei mittlerer Disziplin).
            base_dropout=0.45,
            dropout_discipline_sensitivity=0.56,
            # Nach ETH B.Sc. Physik: Top-Zugang zu ETH/UZH MSc Quantitative Finance & Top-MFE.
            top_master_prob_high_gpa=0.80,
            top_master_prob_mid_gpa=0.48,
            # Sehr starke Quant- & Hedgefonds-Pipeline; respektables IB-Placement (Zürich/London).
            placement_ib=0.16,
            placement_quant=0.30,
            placement_tier2=0.22,
            placement_tech=0.32,
            top_master_ib_mult=1.45,
            top_master_quant_mult=1.70,
            debt_no_scholarship_eur=36_000.0,
            debt_with_scholarship_eur=14_000.0,
            debt_top_master_addon_eur=8_000.0,   # ETH MSc QF relativ günstig (CH-Gebühren)
            comp_region_mult=1.18,               # Zürich / London Quant & Finance
            notes="Prüfungsfrei ab 70 % in Mathe/Physik/Sprache; hohe Zürcher Kosten & harte Basisprüfung, Top-Quant-Pipeline.",
        ),
        "de_physik": TrackConfig(
            track_id="de_physik",
            # 100 % NC-frei an TUM, Heidelberg, LMU, KIT, RWTH Aachen (Abischnitt komplett egal!).
            base_admit=1.00,
            # Keine Studiengebühren + Inlands-BAföG / HiWi / Nebenjob -> nahezu 100 % finanziert.
            base_finance=0.965,
            finance_with_scholarship=0.992,
            # DZHW-Realität: 35-50 % Abbrecherquote im Physik-Grundstudium (stark disziplinabhängig!).
            base_dropout=0.46,
            dropout_discipline_sensitivity=0.56,
            # Entscheidendes Sprungbrett: Bei Physik-GPA <= 1,7 extrem hohe Chancen auf
            # ETH MSc QF, HSG MBF, US MFE (Princeton/MIT/Baruch), Oxford/Imperial — Abi egal!
            top_master_prob_high_gpa=0.76,
            top_master_prob_mid_gpa=0.34,
            # Ohne Top-Master (Standard DE M.Sc. Physik): Solide Tech/Forschung/Consulting-Basis,
            # moderates direktes IB/Quant. MIT Top-Master: massiver Sprung!
            placement_ib=0.10,
            placement_quant=0.11,
            placement_tier2=0.29,
            placement_tech=0.50,
            top_master_ib_mult=3.10,
            top_master_quant_mult=3.80,
            # BAföG-Deckel in DE (max. 10.010 EUR, oft geringer durch Einmalrückzahlung/HiWi).
            debt_no_scholarship_eur=6_500.0,
            debt_with_scholarship_eur=1_500.0,
            # 1-2 Jahre Auslands-Top-Master (ETH/HSG/UK/US MFE) über KfW/Prodigy/DAAD/Rücklagen.
            debt_top_master_addon_eur=19_000.0,
            comp_region_mult=1.00,               # Basis DE; bei Top-Master automatischer Uplift
            notes="100 % NC-frei & sicher finanziert (Downside Protection); Grundstudium-Risiko, starkes Top-Master-Sprungbrett.",
        ),
        "de_bwl_wima": TrackConfig(
            track_id="de_bwl_wima",
            # Aggregierter Track aus Uni Mannheim BWL (harter NC + GdB/EC-Bonus),
            # Uni Mannheim WiMa (NC-entspannter, Quant-Alternative) & WHU/Frankfurt School (UGV).
            base_admit=0.88,
            # Mannheim gebührenfrei; WHU/FS (~45k EUR) via Stipendium oder UGV (Brain Capital / Dt. Bildung).
            base_finance=0.84,
            finance_with_scholarship=0.97,
            # Niedrigere Abbrecherquote in BWL (~14 %), moderat höher in WiMa (~32 %).
            base_dropout=0.18,
            dropout_discipline_sensitivity=0.30,
            # Hohe Top-Master-Chancen (HSG MBF, LSE, LBS, Bocconi, HEC, Mannheim MMM / MFE).
            top_master_prob_high_gpa=0.68,
            top_master_prob_mid_gpa=0.35,
            # Starkes deutsches IB/M&A-Placement (Mannheim/WHU/FS), Quant-Anteil über WiMa;
            # im BWL-Track bleibt der Abischnitt im deutschen BB/EB-CV-Screening relevant.
            placement_ib=0.22,
            placement_quant=0.06,
            placement_tier2=0.45,
            placement_tech=0.27,
            top_master_ib_mult=1.35,
            top_master_quant_mult=1.60,
            # Blend aus Mannheim (~6k EUR) und WHU/FS UGV-Äquivalent (~38k EUR); wird je Subpfad dynamisch!
            debt_no_scholarship_eur=18_000.0,
            debt_with_scholarship_eur=6_000.0,
            debt_top_master_addon_eur=16_000.0,
            comp_region_mult=1.03,
            notes="Mannheim BWL (NC + GdB-100-Härtefallbonus), Mannheim WiMa (Quant-Puffer) & WHU/FS (UGV/Brain Capital).",
        ),
        "uk_top": TrackConfig(
            track_id="uk_top",
            # UCAS-Angebote (Oxford, LSE, Imperial, Warwick) stark abischnittabhängig (1,0-1,4).
            base_admit=0.34,
            # Post-Brexit-Realität: 35.000-45.000 £/Jahr Tuition + London/Oxford Living (~185k EUR/3J),
            # kein UK Student Finance, kaum Vollstipendien -> massive Finanzierungs-Abbruchrate!
            base_finance=0.045,
            finance_with_scholarship=0.11,
            # Hohe UK-Abschlussquote sobald finanziert (~91-93 %).
            base_dropout=0.08,
            dropout_discipline_sensitivity=0.14,
            # Direkte London-City-Rekrutierung über Spring Weeks -> Summer Analyst.
            top_master_prob_high_gpa=0.22,
            top_master_prob_mid_gpa=0.12,
            # London City Target Powerhouse (LSE/Oxford/Imperial/Warwick).
            placement_ib=0.42,
            placement_quant=0.23,
            placement_tier2=0.23,
            placement_tech=0.12,
            top_master_ib_mult=1.10,
            top_master_quant_mult=1.25,
            # Enorme Schuldenlast bei privater Finanzierung im Post-Brexit-Bachelor.
            debt_no_scholarship_eur=105_000.0,
            debt_with_scholarship_eur=48_000.0,
            debt_top_master_addon_eur=22_000.0,
            comp_region_mult=1.20,  # London City IB / Quant TC
            notes="Post-Brexit-Bachelor extrem teuer (35-45k £/Jahr) -> Finanzierung als Hauptengpass; realistischer als 1-J-Master.",
        ),
    }


@dataclass
class SimConfig:
    """Gesamtkonfiguration der Monte-Carlo-Karrieresimulation."""

    tracks: Dict[str, TrackConfig] = field(default_factory=default_track_configs)

    # Profil-Multiplikatoren (transparent & editierbar)
    profile_multipliers: Dict[str, float] = field(default_factory=lambda: {
        # US Holistic Admissions Multiplikatoren
        "us_first_gen_admit_mult": 1.62,
        "us_ma_internship_admit_mult": 1.32,
        "us_jugend_gruendet_admit_mult": 1.28,
        "us_tradity_admit_mult": 1.16,
        "us_gdb_100_resilience_mult": 1.22,
        # Deutschland / Mannheim / BWL-Boni
        "mannheim_gdb_100_abi_bonus": 0.45,       # Effektiver Abischnitt-Bonus via Härtefall/Nachteilsausgleich
        "mannheim_ec_abi_bonus": 0.15,            # Bonus durch M&A-Praktikum + Jugend gründet
        "whu_fs_ec_admit_boost": 0.14,            # Assessment-Center-Boost für WHU/Frankfurt School
        # HSG Interview- & Profil-Boost
        "hsg_ma_interview_boost": 0.08,
        "hsg_ec_interview_boost": 0.06,
        # Begabtenförderungswerk (Studienstiftung / sdw / etc.)
        "scholarship_first_gen_mult": 1.35,
        "scholarship_tradity_mult": 1.22,
        "scholarship_jugend_gruendet_mult": 1.20,
        "scholarship_gdb_100_mult": 1.18,
        # Recruiting / Praktika-Snowball (Stufe 5)
        "recruiting_ma_ib_mult": 1.32,            # M&A-Schülerpraktikum -> frühes Boutique-/BB-Praktikum
        "recruiting_jg_ib_mult": 1.10,            # Top 3 % Jugend gründet -> Business-Signal
        "recruiting_tradity_fin_mult": 1.06,      # Tradity e.V. -> Finance-Engagement
    })

    # Begabtenförderungswerk Basis-Wahrscheinlichkeiten (Studienstiftung / sdw / Stiftungen)
    # Hohe Chance bei Abi <= 1,5, stark sinkend ab 1,7
    scholarship_params: Dict[str, float] = field(default_factory=lambda: {
        "base_prob_abi_13": 0.34,
        "base_prob_abi_15": 0.25,
        "base_prob_abi_17": 0.12,
        "base_prob_abi_20": 0.04,
        "base_prob_abi_25": 0.012,
        "max_prob_cap": 0.72,
    })

    # Korrelationsstruktur (Gaussian Copula zwischen Tracks im Portfolio)
    correlation_params: Dict[str, float] = field(default_factory=lambda: {
        "rho_academic": 0.45,       # Latente akademische / analytische Stärke über Tracks hinweg
        "rho_interview": 0.38,      # Latente Interview- / Networking-Performance
        "rho_scholarship": 0.65,    # Gemeinsamer Stipendien- / BAföG-Fördererfolg
    })

    # Basis-Brutto-Einstiegsgehälter (Total Comp Jahr 1 in EUR, Region DE = 1.00)
    # Log-Normal-Parameter: median_eur und sigma (Volatilität der Gehaltsspanne)
    outcome_comp_eur: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Tier-1 IB / M&A": {"median": 122_000.0, "sigma": 0.16},
        "Top Quant / Hedgefonds": {"median": 175_000.0, "sigma": 0.24},
        "Tier-2 Finance / Consulting / Corporate Finance": {"median": 80_000.0, "sigma": 0.15},
        "Tech / Forschung / Sonstiges": {"median": 62_000.0, "sigma": 0.18},
        "Abbruch / Finanzierungs-Stopp": {"median": 44_000.0, "sigma": 0.14},
    })

    # Risk-Adjusted Career Score (0-100) Gewichtungen
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "util_ib": 95.0,
        "util_quant": 100.0,
        "util_tier2": 62.0,
        "util_tech": 48.0,
        "util_dropout": 12.0,
        "util_weight": 0.80,
        "comp_floor_eur": 50_000.0,
        "comp_step_eur": 10_000.0,
        "comp_max_bonus": 20.0,
        "debt_penalty_per_100k": 28.0,
        "debt_max_penalty": 30.0,
        "downside_risk_lambda": 0.20,  # Risikoaversion für Downside-Semivarianz
    })

    def clone(self) -> "SimConfig":
        """Erstellt eine tiefe Kopie der Konfiguration für sicheres Überschreiben im Editor."""
        return deepcopy(self)

    def tracks_as_records(self) -> List[Dict[str, Any]]:
        """Gibt alle Track-Kernparameter als flache Tabelle für st.data_editor zurück."""
        rows: List[Dict[str, Any]] = []
        for tid in TRACK_IDS:
            tc = self.tracks[tid]
            rows.append({
                "track_id": tid,
                "Track": TRACK_META[tid]["short_name"],
                "base_admit": tc.base_admit,
                "base_finance": tc.base_finance,
                "finance_with_scholarship": tc.finance_with_scholarship,
                "base_dropout": tc.base_dropout,
                "dropout_discipline_sensitivity": tc.dropout_discipline_sensitivity,
                "top_master_prob_high_gpa": tc.top_master_prob_high_gpa,
                "top_master_prob_mid_gpa": tc.top_master_prob_mid_gpa,
                "placement_ib": tc.placement_ib,
                "placement_quant": tc.placement_quant,
                "placement_tier2": tc.placement_tier2,
                "placement_tech": tc.placement_tech,
                "top_master_ib_mult": tc.top_master_ib_mult,
                "top_master_quant_mult": tc.top_master_quant_mult,
                "debt_no_scholarship_eur": tc.debt_no_scholarship_eur,
                "debt_with_scholarship_eur": tc.debt_with_scholarship_eur,
                "debt_top_master_addon_eur": tc.debt_top_master_addon_eur,
                "comp_region_mult": tc.comp_region_mult,
            })
        return rows

    def apply_track_records(self, records: List[Dict[str, Any]]) -> None:
        """Übernimmt editierte Werte aus dem Tab-4-Annahmen-Editor sicher mit Bounds-Clamping."""
        prob_fields = {
            "base_admit", "base_finance", "finance_with_scholarship",
            "base_dropout", "top_master_prob_high_gpa", "top_master_prob_mid_gpa",
            "placement_ib", "placement_quant", "placement_tier2", "placement_tech",
        }
        pos_fields = {
            "dropout_discipline_sensitivity", "top_master_ib_mult",
            "top_master_quant_mult", "comp_region_mult",
        }
        nonneg_fields = {
            "debt_no_scholarship_eur", "debt_with_scholarship_eur",
            "debt_top_master_addon_eur",
        }
        for row in records:
            tid = row.get("track_id")
            if tid not in self.tracks:
                continue
            tc = self.tracks[tid]
            for f in prob_fields:
                if f in row and row[f] is not None:
                    setattr(tc, f, float(np.clip(float(row[f]), 0.0, 1.0)))
            for f in pos_fields:
                if f in row and row[f] is not None:
                    setattr(tc, f, float(np.clip(float(row[f]), 0.05, 10.0)))
            for f in nonneg_fields:
                if f in row and row[f] is not None:
                    setattr(tc, f, float(np.clip(float(row[f]), 0.0, 500_000.0)))


def get_default_config() -> SimConfig:
    """Factory-Funktion für die Standard-Simulationskonfiguration."""
    return SimConfig()
