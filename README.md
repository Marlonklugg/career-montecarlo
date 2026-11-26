# Quant Career Monte-Carlo Simulation (IB / M&A, Quant Trading & Hedgefonds)

Vollständig vektorisierte Monte-Carlo-Karrieresimulation (Python + NumPy + Pandas + Streamlit + Plotly) für **100.000 bis 250.000 Durchläufe pro Track** vom deutschen Abitur (1,3 bis 2,5) über Bachelor und optionalen Top-Master bis zum Berufseinstieg (Alter ca. 24–27).

---

## Projektarchitektur

```text
career_montecarlo/
├── config.py              # Alle dokumentierten Wahrscheinlichkeiten, Gebühren, Multiplikatoren & Dataclasses
├── engine.py              # 100 % vektorisierte NumPy-Engine (Gaussian Copula, CRN, 20 3-Pfeile-Portfolios)
├── app.py                 # Dark-Mode Quant/Bloomberg Streamlit-Dashboard mit 4 Analyse-Tabs
├── test_engine.py         # Pytest-Suite (100k Performance, Seed-Reproduzierbarkeit, 100 %-Normierung, Domain-Checks)
├── requirements.txt       # Python-Abhängigkeiten
├── .streamlit/config.toml # Native Dark-Mode Konfiguration
└── README.md
```

---

## Modellierte 5-Stufen-Pipeline & 6 Uni-Tracks

Jeder Durchlauf simuliert eine korrelierte 5-stufige Pipeline über gemeinsame latente Bewerberfaktoren (`z_academic`, `z_interview`, `z_scholarship` via Gaussian Copula):

1. **Stufe 1 – Zulassung:** Track-spezifische Selektionsfunktionen (US Need-Blind Holistic Review, HSG-Zulassungstest-Logistik, ETH-Zürich 70 %-Regel `>= 10,5 Pkt.` in Mathe/Physik/Sprache, NC-freie DE-Top-Physik, Mannheim NC + GdB-100-Härtefallbonus / WHU-FS / Mannheim WiMa, UK UCAS-Offers).
2. **Stufe 2 – Finanzierbarkeit:** Kombination aus Need-Based Full Aid (US), Begabtenförderungswerk (Studienstiftung / sdw – hoch bis Abi 1,5, steiler Abfall ab 1,7), Auslands-/Inlands-BAföG, HSG-/KfW-Darlehen, Umgekehrtem Generationenvertrag (WHU / Frankfurt School) und Post-Brexit-Realität im UK-Bachelor (~35–45k £/Jahr).
3. **Stufe 3 – Abschluss & Uni-GPA:** Reale Abbrecherquoten (35–50 % im deutschen Physik-Grundstudium & ETH-Basisprüfung, stark gesteuert durch Disziplin & Mathe/Physik-Faktor) und korrelierte Uni-GPA (1,0–3,7).
4. **Stufe 4 – Optionaler Top-Master:** Sprungbrett-Mechanik bei guter Uni-GPA (`<= 1,7`) für ETH MSc Quantitative Finance, HSG MBF, US MFE (Princeton/MIT/Baruch) und Oxford/Imperial/LSE (1-jähriger Master) — bei `DE Top-Physik` komplett unabhängig vom Abiturschnitt.
5. **Stufe 5 – Praktika & Finales Karriere-Outcome + Finanzmetriken:** Exakt einer von 5 Zielzuständen (`Tier-1 IB / M&A`, `Top Quant / Hedgefonds`, `Tier-2 Finance / Consulting / Corporate Finance`, `Tech / Forschung / Sonstiges`, `Abbruch / Finanzierungs-Stopp`) plus Gesamtschuldenlast (EUR), Total Compensation Jahr 1 (EUR) und Risk-Adjusted Career Score (0–100, mit Downside-Semideviation-Strafe).

### 3-Pfeile-Portfolio-Mix
Zusätzlich zu den 6 Einzel-Tracks evaluiert `engine.py` vektorisiert alle $\binom{6}{3} = 20$ parallelen 3-Track-Bewerbungsportfolios. In jedem Run schreibt sich der Bewerber rational in den am höchsten bewerteten Track der 3 Optionen ein, der sowohl **zugelassen als auch finanziert** ist.

---

## Lokale Ausführung & Tests

### 1. Virtuelle Umgebung einrichten & Abhängigkeiten installieren
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Unit-Tests & Performance-Verifikation ausführen
```bash
pytest -v
```
Verifiziert u. a.:
- 100.000 Runs fehlerfrei in deutlich unter 1,5 Sekunden (~0,18 s auf Apple Silicon)
- Bit-genaue Reproduzierbarkeit mit festem Seed
- Exakte 100,0 %-Summierung aller Outcome-Wahrscheinlichkeiten für alle Tracks & Portfolios
- Fachliche Plausibilität (ETH-70 %-Regel, DE-Physik Downside Protection, GdB-100-Bonus, UK Post-Brexit-Engpass, Sankey-Massenerhaltung)

### 3. Streamlit-Dashboard starten
```bash
streamlit run app.py
```
Die Anwendung ist anschließend lokal unter `http://localhost:8501` erreichbar.
