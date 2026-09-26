"""Jede Zahl, die README, Hilfetexte und Beschriftungen nennen, ist hier belegt: das Standardnetz (Seed 1), die Presets, die Lehrnetze von Hand und die Verteilungen über feste Netze
(Seeds ab 100000: 40 Netze für die Verteilung, 20 für die Fixkosten-Reihe, 5 je Größe).

Zahlen der LP sind Gleitkommawerte (Vergleich mit Toleranz); alle Kosten der Heuristiken und des Dual Ascent sind ganzzahlig und auf allen Plattformen dieselben. Gezählt wird nur, was der LP-WERT
hergibt (gleich dem Optimum), nie, welche Ecke der Löser bei mehreren gleich guten Lösungen wählt."""

import pytest

import ufl_constants as C
import ufl_dual as dl
import ufl_evaluation as ev
import ufl_exact as ex
import ufl_scenario as sc

STD = ev.Params("map", 25, 50, 100, 1)
PCT = pytest.approx


def _a(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.canonical(ev.Params(p["net"], p["sites"], p["customers"], p["fixed"], p["seed"], p["order"])))


def _pct(a, key):
    return 100.0 * a[key] / a["opt"]


@pytest.fixture(scope="module")
def dist_map():
    return ev.distribution(STD)


@pytest.fixture(scope="module")
def dist_ties():
    return ev.distribution(ev.Params("ties", 30, 30, 100, 1))


@pytest.fixture(scope="module")
def series_map():
    return {r["fixed"]: r for r in ev.fixed_series(STD)}


@pytest.fixture(scope="module")
def series_ties():
    return {r["fixed"]: r for r in ev.fixed_series(ev.Params("ties", 30, 30, 100, 1))}


@pytest.fixture(scope="module")
def size_rows():
    return {(r["m"], r["n"]): r for r in ev.sizes(STD)}


# --- Standardnetz und Presets -------------------------------------------------------------------------------------------------------------

def test_standard_net():
    """25 Kandidaten, 50 Kunden: Optimum 39 967 mit 8 offenen Standorten; starke LP 100,0 %, schwache 63,6 %; Add 41 872 (+4,8 %, 9 Züge), Drop (17 Züge) und Interchange (2 Züge) treffen das Optimum;
    Dual Ascent 34 252 (85,7 %), mit Anpassung 39 744 (99,4 %); die Auswahl aus den knappen Standorten kostet 41 464 (Lücke 1 720), mit Interchange 39 967."""
    a = ev.analyse(STD)
    assert (a["opt"], len(a["opt_set"])) == (39967, 8)
    assert _pct(a, "lp_strong") == PCT(100.0, abs=0.005) and _pct(a, "lp_weak") == PCT(63.57, abs=0.01)
    assert (a["add"].cost, len(a["add"].trace)) == (41872, 9) and a["add"].cost / a["opt"] == PCT(1.0477, abs=0.0001)
    assert a["drop"].cost == 39967 == a["interchange"].cost and a["polished"].cost == 39967
    assert (len(a["drop"].trace), len(a["interchange"].trace)) == (17, 2)
    assert (a["dual"].lower_bound, a["dual_adj"].lower_bound) == (34252, 39744) and 100.0 * a["dual"].lower_bound / a["opt"] == PCT(85.70, abs=0.01)
    assert (a["dual_adj"].primal_cost, a["dual_adj"].gap, a["dual_adj"].proven) == (41464, 1720, False) and a["dual_adj"].passes == 2


def test_the_four_customer_orders_change_the_bound_a_little():
    """Aufstieg / mit Anpassung im Standardnetz: in Reihenfolge 34 252 / 39 744, rückwärts 37 253 / 39 944, nächste zuerst 32 870 / 33 924, entfernteste zuerst 37 781 / 39 568."""
    got = {}
    for order in dl.ORDERS:
        a = ev.analyse(ev.Params("map", 25, 50, 100, 1, order))
        got[order] = (a["dual"].lower_bound, a["dual_adj"].lower_bound)
    assert got == {"index": (34252, 39744), "reverse": (37253, 39944), "near": (32870, 33924), "far": (37781, 39568)}


def test_preset_add_trap():
    a = _a("➕ Add-Falle")
    assert (a["opt"], a["opt_set"], a["add"].cost, a["add"].open_set, a["drop"].cost, a["interchange"].cost) == (16, (0, 2), 17, (0, 1), 16, 16)
    assert (a["dual_adj"].lower_bound, a["dual_adj"].primal_cost, a["dual_adj"].proven) == (16, 16, True)


def test_preset_drop_trap():
    a = _a("➖ Drop-Falle")
    assert (a["opt"], a["opt_set"], a["drop"].cost, a["drop"].open_set, a["add"].cost) == (21, (0,), 22, (1, 2), 21)
    assert (a["dual"].lower_bound, a["dual_adj"].lower_bound, a["dual_adj"].primal_cost, a["dual_adj"].proven) == (19, 20, 21, False)


def test_preset_interchange_stuck():
    a = _a("🔁 Interchange steckt fest")
    assert (a["opt"], a["opt_set"], a["add"].cost, a["add"].open_set, a["interchange"].cost, len(a["interchange"].trace), a["drop"].cost) == (20, (1, 2), 21, (0,), 21, 0, 20)
    assert (a["dual_adj"].lower_bound, a["dual_adj"].proven) == (20, True)


def test_preset_lp_fractional():
    a = _a("🧮 LP gebrochen")
    assert a["opt"] == 25 and a["opt_set"] == (1,) and a["lp_strong"] == PCT(24.5, abs=1e-6) and a["lp_weak"] == PCT(18.25, abs=1e-6)
    assert ex.fractional_sites(a["y_strong"]) and (a["dual"].lower_bound, a["dual_adj"].lower_bound) == (21, 24)


def test_preset_high_fixed_cost():
    a = _a("🏷️ Hohe Fixkosten")
    assert (a["opt"], len(a["opt_set"])) == (63970, 5) and _pct(a, "lp_weak") == PCT(47.65, abs=0.02) and _pct(a, "lp_strong") == PCT(100.0, abs=0.005)
    assert (a["add"].cost, a["drop"].cost, a["interchange"].cost) == (66436, 65182, 63970)
    assert (a["dual"].lower_bound, a["dual_adj"].lower_bound, a["dual_adj"].passes) == (42926, 61912, 5)


def test_preset_ties_net():
    a = _a("🎲 Gleichstandsnetz")
    assert (a["opt"], len(a["opt_set"])) == (47, 4) and _pct(a, "lp_strong") == PCT(95.84, abs=0.02) and _pct(a, "lp_weak") == PCT(70.28, abs=0.02)
    assert (a["add"].cost, a["drop"].cost, a["interchange"].cost, a["dual_adj"].lower_bound) == (48, 49, 47, 43)


def test_preset_large_net():
    a = _a("🏙️ Großes Netz")
    assert (a["opt"], len(a["opt_set"])) == (74264, 14) and _pct(a, "lp_weak") == PCT(57.97, abs=0.02) and _pct(a, "lp_strong") == PCT(100.0, abs=0.005)
    assert (a["add"].cost, a["drop"].cost, a["interchange"].cost, a["dual_adj"].lower_bound) == (74965, 74473, 74264, 73338)


def test_lehrnetze_proven_and_both_trap():
    p = ev.analyse(ev.Params("proven", 25, 50, 100, 1))
    assert (p["opt"], p["dual"].lower_bound, p["dual_adj"].lower_bound, p["dual_adj"].proven, p["add"].cost, p["drop"].cost) == (25, 23, 25, True, 28, 25)
    b = ev.analyse(ev.Params("both_trap", 25, 50, 100, 1))
    assert (b["opt"], b["add"].cost, b["drop"].cost, b["interchange"].cost) == (27, 28, 28, 27)


# --- Verteilung über 40 feste Kartennetze ----------------------------------------------------------------------------------------------------

def test_map_nets_strong_lp_is_almost_exact_and_weak_lp_is_not(dist_map):
    """40 Kartennetze (25 × 50, Fixkosten 100 %): starke LP im Mittel 99,99 % (schlechtestes Netz 99,65 %), der LP-Wert ist in 38 von 40 Netzen gleich dem Optimum; schwache LP im Mittel 61,0 %, schlechtestes Netz 69,3 %."""
    s, rows = dist_map["summary"], dist_map["rows"]
    assert s["count"] == 40 and s["tight"] == 38
    assert s["strong"] == PCT(99.99, abs=0.01) and s["strong_min"] == PCT(99.65, abs=0.02)
    assert s["weak"] == PCT(61.04, abs=0.05) and s["weak_min"] == PCT(53.68, abs=0.05) and max(r["weak"] for r in rows) == PCT(69.29, abs=0.05)
    assert min(r["n_open"] for r in rows) == 7 and max(r["n_open"] for r in rows) == 10 and s["n_open"] == PCT(8.68, abs=0.1)


def test_map_nets_heuristics(dist_map):
    """Abstand zum Optimum: Add im Mittel 1,87 % (Median 1,01, schlechtestes 7,96, exakt in 6 Netzen), Drop 0,11 % (0,00; 0,84; 27), Interchange 0,11 % (0,00; 1,37; 32);
    Drop ist in 32 Netzen besser als Add, in einem schlechter."""
    s, rows = dist_map["summary"], dist_map["rows"]
    assert (s["add"], s["add_median"], s["add_max"], s["add_exact"]) == (PCT(1.87, abs=0.01), PCT(1.01, abs=0.01), PCT(7.96, abs=0.01), 6)
    assert (s["drop"], s["drop_median"], s["drop_max"], s["drop_exact"]) == (PCT(0.11, abs=0.01), PCT(0.0, abs=0.01), PCT(0.84, abs=0.01), 27)
    assert (s["interchange"], s["interchange_median"], s["interchange_max"], s["interchange_exact"]) == (PCT(0.11, abs=0.01), PCT(0.0, abs=0.01), PCT(1.37, abs=0.01), 32)
    assert sum(1 for r in rows if r["add"] > r["drop"]) == 32 and sum(1 for r in rows if r["drop"] > r["add"]) == 1


def test_map_nets_dual_ascent(dist_map):
    """Dual Ascent ohne Anpassung im Mittel 87,8 % (schlechtestes Netz 76,7 %), mit Anpassung 99,4 % (schlechtestes 97,3 %; in 30 Netzen mindestens 99 %); die Auswahl aus den knappen Standorten liegt im Mittel 4,0 % über dem Optimum,
    bewiesen optimal in 4 Netzen, mit anschließendem Interchange 0,02 % (schlechtestes Netz 0,41 %, exakt in 35)."""
    s, rows = dist_map["summary"], dist_map["rows"]
    assert (s["dual"], s["dual_min"]) == (PCT(87.84, abs=0.01), PCT(76.75, abs=0.01)) and (s["dual_adj"], s["dual_adj_min"]) == (PCT(99.38, abs=0.01), PCT(97.27, abs=0.01))
    assert sum(1 for r in rows if r["dual_adj"] >= 99) == 30
    assert s["dual_primal"] == PCT(4.04, abs=0.01) and s["proven"] == 4
    assert (s["polished"], s["polished_max"]) == (PCT(0.02, abs=0.01), PCT(0.41, abs=0.01)) and sum(1 for r in rows if r["polished"] == 0) == 35


# --- Gleichstandsnetze -----------------------------------------------------------------------------------------------------------------------

def test_ties_nets_are_where_ufl_gets_hard(dist_ties):
    """40 Gleichstandsnetze (30 × 30): starke LP im Mittel 97,3 % (schlechtestes 95,3 %), LP-Wert gleich dem Optimum in 1 von 40; schwache LP 69,9 %; Dual Ascent mit Anpassung 93,6 % (schlechtestes 89,1 %), bewiesen in keinem Netz."""
    s = dist_ties["summary"]
    assert s["tight"] == 1 and s["strong"] == PCT(97.34, abs=0.02) and s["strong_min"] == PCT(95.29, abs=0.02) and s["weak"] == PCT(69.92, abs=0.05)
    assert (s["dual"], s["dual_adj"], s["dual_adj_min"], s["proven"]) == (PCT(82.07, abs=0.01), PCT(93.61, abs=0.01), PCT(89.13, abs=0.01), 0)


def test_ties_nets_heuristics_reverse_the_order_of_add_and_drop(dist_ties):
    """Abstand: Add 3,07 % (Median 2,17, schlechtestes 10,64, exakt in 9), Drop 5,02 % (4,26; 13,33; 2), Interchange 1,79 % (2,04; 8,51; 18). Anders als auf der Karte ist hier Add in 20 Netzen besser als Drop und nur in 8 schlechter:
    kein Verfahren gewinnt überall. Die Auswahl aus den knappen Standorten liegt 26,7 % über dem Optimum, mit Interchange 1,32 % (schlechtestes 6,25 %, exakt in 22)."""
    s, rows = dist_ties["summary"], dist_ties["rows"]
    assert (s["add"], s["add_median"], s["add_max"], s["add_exact"]) == (PCT(3.07, abs=0.01), PCT(2.17, abs=0.01), PCT(10.64, abs=0.01), 9)
    assert (s["drop"], s["drop_median"], s["drop_max"], s["drop_exact"]) == (PCT(5.02, abs=0.01), PCT(4.26, abs=0.01), PCT(13.33, abs=0.01), 2)
    assert (s["interchange"], s["interchange_median"], s["interchange_max"], s["interchange_exact"]) == (PCT(1.79, abs=0.01), PCT(2.04, abs=0.01), PCT(8.51, abs=0.01), 18)
    assert sum(1 for r in rows if r["add"] > r["drop"]) == 8 and sum(1 for r in rows if r["drop"] > r["add"]) == 20
    assert s["dual_primal"] == PCT(26.69, abs=0.01) and (s["polished"], s["polished_max"]) == (PCT(1.32, abs=0.01), PCT(6.25, abs=0.01)) and sum(1 for r in rows if r["polished"] == 0) == 22


# --- Fixkosten-Reihe (20 Netze je Wert) -------------------------------------------------------------------------------------------------------

def test_fixed_cost_series_on_map_nets(series_map):
    """Fixkosten 25 / 50 / 100 / 200 / 400 %: offene Standorte im Mittel 13,55 / 11,35 / 8,65 / 6,5 / 4,4; starke LP 100,0 % (LP-Wert = Optimum in 20 / 20 / 19 / 19 / 19 von 20); schwache LP 81,1 / 71,4 / 61,1 / 52,1 / 46,1 %;
    Dual Ascent 96,8 / 93,1 / 87,7 / 81,5 / 75,3 %, mit Anpassung 99,9 / 99,6 / 99,3 / 99,1 / 99,3 %; Add 0,58 / 1,09 / 1,58 / 2,28 / 3,85 %, Drop 0,02 / 0,10 / 0,11 / 0,55 / 0,99 %, Interchange 0,02 / 0,00 / 0,18 / 0,22 / 0,31 %."""
    fixed = (25, 50, 100, 200, 400)
    r = series_map
    assert [r[f]["n_open"] for f in fixed] == pytest.approx([13.55, 11.35, 8.65, 6.5, 4.4], abs=0.01)
    assert [r[f]["tight"] for f in fixed] == [20, 20, 19, 19, 19] and all(r[f]["strong"] > 99.98 for f in fixed)
    assert [r[f]["weak"] for f in fixed] == pytest.approx([81.08, 71.39, 61.10, 52.11, 46.11], abs=0.05)
    assert [r[f]["dual"] for f in fixed] == pytest.approx([96.84, 93.12, 87.73, 81.52, 75.33], abs=0.02)
    assert [r[f]["dual_adj"] for f in fixed] == pytest.approx([99.91, 99.63, 99.33, 99.14, 99.26], abs=0.02)
    assert [r[f]["add"] for f in fixed] == pytest.approx([0.58, 1.09, 1.58, 2.28, 3.85], abs=0.01)
    assert [r[f]["drop"] for f in fixed] == pytest.approx([0.02, 0.10, 0.11, 0.55, 0.99], abs=0.01)
    assert [r[f]["interchange"] for f in fixed] == pytest.approx([0.02, 0.0, 0.18, 0.22, 0.31], abs=0.01)
    assert r[25]["weak"] > r[50]["weak"] > r[100]["weak"] > r[200]["weak"] > r[400]["weak"] and r[25]["dual"] > r[400]["dual"] + 20 and min(r[f]["dual_adj"] for f in fixed) > 99.0


def test_fixed_cost_series_on_ties_nets(series_ties):
    """Gleichstandsnetze: starke LP 99,9 / 98,6 / 97,5 / 96,7 / 96,7 %, LP-Wert = Optimum in 19 / 3 / 1 / 1 / 2 von 20; schwache LP 97,6 / 82,6 / 70,2 / 61,7 / 56,4 %; Dual Ascent mit Anpassung 99,8 / 95,2 / 93,3 / 91,7 / 85,9 %."""
    fixed = (25, 50, 100, 200, 400)
    r = series_ties
    assert [r[f]["strong"] for f in fixed] == pytest.approx([99.92, 98.55, 97.53, 96.71, 96.65], abs=0.03)
    assert [r[f]["tight"] for f in fixed] == [19, 3, 1, 1, 2]
    assert [r[f]["weak"] for f in fixed] == pytest.approx([97.58, 82.62, 70.20, 61.69, 56.44], abs=0.05)
    assert [r[f]["dual_adj"] for f in fixed] == pytest.approx([99.84, 95.22, 93.31, 91.73, 85.87], abs=0.03)


# --- Größenreihe (5 Netze je Größe) ------------------------------------------------------------------------------------------------------------

def test_size_series(size_rows):
    """10 × 20 / 20 × 50 / 30 × 75 / 40 × 100: offene Standorte 4,0 / 7,6 / 10,6 / 13,0; starke LP 100,0 %, gleich dem Optimum in 5 / 5 / 5 / 4 von 5 Netzen; schwache LP 76,4 / 66,3 / 62,6 / 59,2 %;
    Bewertungen Add 40 / 144 / 290 / 484, Drop 49 / 183 / 414 / 739, Interchange 52 / 184 / 516 / 1 486; Abstand Interchange 0,00 / 0,12 / 0,01 / 0,28 %."""
    keys = ((10, 20), (20, 50), (30, 75), (40, 100))
    r = size_rows
    assert [round(r[k]["n_open"], 1) for k in keys] == pytest.approx([4.0, 7.6, 10.6, 13.0], abs=0.06)
    assert [r[k]["tight"] for k in keys] == [5, 5, 5, 4]
    assert [r[k]["weak"] for k in keys] == pytest.approx([76.42, 66.30, 62.64, 59.17], abs=0.05)
    assert [round(r[k]["add_evals"]) for k in keys] == [40, 144, 290, 484]
    assert [round(r[k]["drop_evals"]) for k in keys] == [49, 183, 414, 739]
    assert [round(r[k]["interchange_evals"]) for k in keys] == [52, 184, 516, 1486]
    assert [r[k]["interchange"] for k in keys] == pytest.approx([0.0, 0.12, 0.01, 0.28], abs=0.01)


def test_effort_counters_are_platform_stable_integers():
    """Bewertungen und Kantenoperationen sind Ganzzahlen aus der Rechnung, nicht aus Zeit oder Gleitkomma: Standardnetz Add 205, Drop 297, Interchange 491; Dual Ascent mit Anpassung 130 250 Kantenoperationen."""
    a = ev.analyse(STD)
    assert (a["add"].evals, a["drop"].evals, a["interchange"].evals, a["dual_adj"].ops) == (205, 297, 491, 130250)


def test_help_texts_only_name_numbers_from_the_presets_above():
    """Jede Voreinstellung hat einen Hilfetext, der ihre Kennzahlen nennt (oben belegt); hier nur: keine Lücke."""
    assert all(C.PRESET_HELP[k].strip() for k in C.PRESETS) and set(C.PRESET_HELP) == set(C.PRESETS)
