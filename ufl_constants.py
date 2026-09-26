"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Standortplanung: warum ist die Schranke hier fast exakt?"."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------
SITES_MIN, SITES_MAX, DEFAULT_SITES = 5, 40, 25
CUSTOMERS_MIN, CUSTOMERS_MAX, DEFAULT_CUSTOMERS = 10, 100, 50
FIXED_MIN, FIXED_MAX, DEFAULT_FIXED, FIXED_STEP = 25, 400, 100, 25       # Fixkosten-Faktor in Prozent
DEFAULT_SEED = 1
SEED_MAX = 2_000_000_000

NETS = {
    "map": "Karte (Zufallsnetz mit Standorten und Kunden)",
    "ties": "Gleichstandsnetz (ganzzahlige Kosten 1 bis 5, ohne Karte)",
    "add_trap": "Lehrnetz: Add-Falle (3 Standorte, 4 Kunden)",
    "drop_trap": "Lehrnetz: Drop-Falle (3 Standorte, 4 Kunden)",
    "stuck": "Lehrnetz: Interchange steckt fest (3 Standorte, 4 Kunden)",
    "both_trap": "Lehrnetz: beide Fallen (4 Standorte, 5 Kunden)",
    "lp_frac": "Lehrnetz: LP gebrochen (3 Standorte, 4 Kunden)",
    "proven": "Lehrnetz: Dual Ascent beweist das Optimum (3 Standorte, 4 Kunden)",
}
DEFAULT_NET = "map"
FIXED_NETS = tuple(k for k in NETS if k not in ("map", "ties"))

HEURISTICS = {"add": "Add (nacheinander öffnen)", "drop": "Drop (nacheinander schließen)", "interchange": "Interchange (öffnen, schließen, tauschen; Start = Add)"}
DEFAULT_HEURISTIC = "add"
DEFAULT_ORDER = "index"

# --- feste Seed-Mengen (dieselben wie in den Flussdemos; unabhängig vom Nutzer-Seed) ---------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
SERIES_SEEDS = DIST_SEEDS[:20]
SIZE_SEEDS = DIST_SEEDS[:5]
SERIES_FIXED = (25, 50, 100, 200, 400)
SIZES = ((10, 20), (20, 50), (30, 75), (40, 100))        # (Kandidaten, Kunden)

COLORS = {"open": "#2ca02c", "closed": "#9aa0a6", "frac": "#ff7f0e", "customer": "#1f77b4", "line": "#7f8c8d", "opt": "#111111",
          "weak": "#d62728", "strong": "#1f77b4", "dual": "#9467bd", "dual_adj": "#8c564b", "add": "#ff7f0e", "drop": "#2ca02c", "interchange": "#1f77b4"}
LABELS = {"weak": "schwache LP", "strong": "starke LP", "dual": "Dual Ascent", "dual_adj": "Dual Ascent mit Anpassung", "opt": "Optimum",
          "add": "Add", "drop": "Drop", "interchange": "Interchange"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net=DEFAULT_NET, sites=DEFAULT_SITES, customers=DEFAULT_CUSTOMERS, fixed=DEFAULT_FIXED, seed=DEFAULT_SEED, heuristic=DEFAULT_HEURISTIC, order=DEFAULT_ORDER)
PRESETS = {
    "🗺️ Standardnetz": {**_BASE},
    "➕ Add-Falle": {**_BASE, "net": "add_trap", "heuristic": "add"},
    "➖ Drop-Falle": {**_BASE, "net": "drop_trap", "heuristic": "drop"},
    "🔁 Interchange steckt fest": {**_BASE, "net": "stuck", "heuristic": "interchange"},
    "🧮 LP gebrochen": {**_BASE, "net": "lp_frac"},
    "🏷️ Hohe Fixkosten": {**_BASE, "fixed": 400},
    "🎲 Gleichstandsnetz": {**_BASE, "net": "ties", "sites": 30, "customers": 30},
    "🏙️ Großes Netz": {**_BASE, "sites": 40, "customers": 100},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py belegt.
PRESET_HELP = {
    "🗺️ Standardnetz": "25 Kandidaten, 50 Kunden, Fixkosten-Faktor 100 %: das Optimum kostet 39 967 und öffnet 8 Standorte. Die starke LP erreicht das Optimum (100,0 %), die schwache nur 63,6 %. Add endet bei 41 872 (4,8 % darüber), Drop und Interchange finden das Optimum. Dual Ascent erreicht 85,7 %, mit Anpassung 99,4 %.",
    "➕ Add-Falle": "3 Standorte, 4 Kunden: Add öffnet zuerst S2 (der billigste Einzelstandort) und bleibt bei 17; das Optimum 16 öffnet S1 und S3, Drop und Interchange finden es. Der Dual Ascent beweist es: Schranke 16, Auswahl 16.",
    "➖ Drop-Falle": "3 Standorte, 4 Kunden: Drop schließt sich in die Auswahl {S2, S3} (22); das Optimum 21 ist S1 allein, Add findet es. Der Dual Ascent kommt mit Anpassung nur auf 20 (ohne 19): die Lücke von 1 bleibt, bewiesen ist nichts.",
    "🔁 Interchange steckt fest": "3 Standorte, 4 Kunden: Add öffnet nur S1 (21). Von dort hilft kein einzelnes Öffnen, Schließen oder Tauschen; das Optimum {S2, S3} kostet 20, Drop findet es, der Dual Ascent beweist es (20 = 20).",
    "🧮 LP gebrochen": "3 Standorte, 4 Kunden: das Optimum ist S2 allein (25), die starke LP kommt nur auf 24,5 (98,0 %) und öffnet Standorte anteilig; die schwache LP auf 18,25 (73,0 %). Dual Ascent: 21, mit Anpassung 24.",
    "🏷️ Hohe Fixkosten": "Fixkosten-Faktor 400 %: das Optimum kostet 63 970 und öffnet nur 5 Standorte. Die schwache LP fällt auf 47,7 %, die starke bleibt bei 100,0 %. Add liegt 3,9 % darüber, Drop 1,9 %, Interchange trifft das Optimum. Dual Ascent ohne Anpassung nur 67,1 %, mit Anpassung 96,8 % (nach 5 Durchläufen).",
    "🎲 Gleichstandsnetz": "30 Standorte, 30 Kunden, Kosten 1 bis 5: das Optimum 47 öffnet 4 Standorte. Die starke LP erreicht nur 95,8 % (die schwache 70,3 %), Add liegt 2,1 % darüber, Drop 4,3 %, Interchange trifft das Optimum. Dual Ascent mit Anpassung 91,5 %.",
    "🏙️ Großes Netz": "40 Standorte, 100 Kunden: das Optimum 74 264 öffnet 14 Standorte. Die schwache LP erreicht 58,0 %, die starke 100,0 %; Add liegt 0,9 % darüber, Drop 0,3 %, Interchange trifft das Optimum. Dual Ascent mit Anpassung 98,8 %.",
}
