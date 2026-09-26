"""Presets: vollständig, in den Grenzen, und jedes Beispiel zeigt, was sein Name verspricht (die Zahlen selbst belegt test_claims.py)."""

import pytest

import ufl_constants as C
import ufl_evaluation as ev
import ufl_presets as P

KEYS = set(P.PRESET_KEYS)


def _params(p):
    return ev.Params(p["net"], p["sites"], p["customers"], p["fixed"], p["seed"], p["order"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["heuristic"] in C.HEURISTICS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["fixed"] - C.FIXED_MIN) % C.FIXED_STEP == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_each_preset_shows_the_effect_its_name_promises():
    a = {name: ev.analyse(ev.canonical(_params(p))) for name, p in C.PRESETS.items()}
    add = a["➕ Add-Falle"]
    assert add["add"].cost > add["opt"] and add["drop"].cost == add["opt"]
    drop = a["➖ Drop-Falle"]
    assert drop["drop"].cost > drop["opt"] and drop["add"].cost == drop["opt"]
    stuck = a["🔁 Interchange steckt fest"]
    assert stuck["interchange"].cost > stuck["opt"] and not stuck["interchange"].trace and stuck["drop"].cost == stuck["opt"]
    frac = a["🧮 LP gebrochen"]
    assert frac["lp_strong"] < frac["opt"] - 0.4
    high = a["🏷️ Hohe Fixkosten"]
    assert high["lp_weak"] < 0.5 * high["opt"] and high["lp_strong"] > high["opt"] - 1e-6
    ties = a["🎲 Gleichstandsnetz"]
    assert ties["lp_strong"] < ties["opt"] * 0.97 and ties["inst"].kind == "ties" and a["🏙️ Großes Netz"]["inst"].m == 40


def test_teaching_nets_ignore_the_random_controls():
    assert ev.canonical(ev.Params("add_trap", 10, 20, 400, 99)) == ev.canonical(ev.Params("add_trap", 40, 100, 25, 5))
    assert ev.canonical(ev.Params("map", 10, 20, 400, 99)) != ev.canonical(ev.Params("map", 40, 100, 25, 5))
