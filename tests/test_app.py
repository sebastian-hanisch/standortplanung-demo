"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, jede Heuristik, Randgrößen, Zug-Regler, ausgeblendete Regler, Permalink, Experimente auf Abruf."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ufl_constants as C
from ufl_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _step_slider(at):
    found = [s for s in at.slider if s.key == "ufl_step"]
    return found[0] if found else None


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def test_default_renders_without_exception():
    at = _run()
    assert _metric(at, "Optimum")[0] == "39 967" and _metric(at, "Dual Ascent: Schranke") == ["39 744"] and _metric(at, "Bewiesen optimal?") == ["nein"]
    assert _step_slider(at).value == 9 and _step_slider(at).max == 9
    assert any("Die starke LP ist ganzzahlig" in t for t in _texts(at)) and any("Add hält nach 9 Zügen bei 41 872" in t for t in _texts(at))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and _metric(at, "Dual Ascent: Schranke")


@pytest.mark.parametrize("net", list(C.NETS))
@pytest.mark.parametrize("heur", list(C.HEURISTICS))
def test_every_net_and_heuristic_renders(net, heur):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["heuristic_radio"] = heur
    at = _run(setup)
    assert not at.error and _metric(at, "Optimum")


def test_extreme_sizes_render():
    for vals in ((("sites_slider", C.SITES_MIN), ("customers_slider", C.CUSTOMERS_MIN), ("fixed_slider", C.FIXED_MIN)),
                 (("sites_slider", C.SITES_MAX), ("customers_slider", C.CUSTOMERS_MAX), ("fixed_slider", C.FIXED_MAX))):
        def setup(at, vals=vals):
            for key, value in vals:
                at.session_state[key] = value
        at = _run(setup)
        assert not at.error and _metric(at, "Dual Ascent: Schranke")


def test_step_slider_moves_through_frames():
    at = _run()
    top = int(_step_slider(at).max)
    for value in (0, 1, top // 2, top):
        _step_slider(at).set_value(value)
        at.run()
        assert not at.exception and _step_slider(at).value == value


def test_a_heuristic_without_moves_shows_no_slider_and_a_notice():
    at = _run(lambda a: _apply(a, C.PRESETS["🔁 Interchange steckt fest"]))
    assert _step_slider(at) is None and any("Kein Zug senkt die Kosten" in t for t in _texts(at))


def test_selecting_no_site_shows_a_notice():
    at = _run()
    at.multiselect(key="open_multi").set_value([])
    at.run()
    assert not at.exception and any("mindestens einen Standort" in t for t in _texts(at))


def test_hidden_controls_keep_their_values_across_a_net_switch():
    at = _run()
    at.sidebar.slider(key="sites_slider").set_value(12)
    at.run()
    at.sidebar.selectbox(key="net_select").set_value("add_trap")
    at.run()
    assert not at.exception and not [w for w in at.sidebar.slider if w.key == "sites_slider"]
    at.sidebar.selectbox(key="net_select").set_value("map")
    at.run()
    assert at.sidebar.slider(key="sites_slider").value == 12 and not at.exception


def test_permalink_settings_are_loaded_and_clamped():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["net"] = "ties"
    at.query_params["sites"] = "99"
    at.query_params["fixed"] = "130"
    at.query_params["heur"] = "drop"
    at.query_params["order"] = "far"
    at.run()
    assert not at.exception
    assert at.sidebar.slider(key="sites_slider").value == C.SITES_MAX and at.sidebar.slider(key="fixed_slider").value == 125
    assert at.session_state["heuristic_radio"] == "drop" and at.sidebar.radio(key="order_radio").value == "far"


def test_invalid_permalink_values_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["heur"] = "magic"
    at.query_params["net"] = "nirgendwo"
    at.run()
    assert not at.exception and at.session_state["heuristic_radio"] == C.DEFAULT_HEURISTIC and at.sidebar.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand(monkeypatch):
    import ufl_evaluation as ev
    s_orig, d_orig, f_orig = ev.sizes, ev.distribution, ev.fixed_series
    monkeypatch.setattr(ev, "sizes", lambda p: s_orig(p, seeds=C.SIZE_SEEDS[:1], sizes_=((10, 20), (15, 30))))
    monkeypatch.setattr(ev, "distribution", lambda p: d_orig(p, seeds=C.SWEEP_SEEDS[:3]))
    monkeypatch.setattr(ev, "fixed_series", lambda p: f_orig(p, seeds=C.SERIES_SEEDS[:2], fixed_values=(50, 100)))
    at = _run()
    for key in ("series_start", "dist_start", "sizes_start"):
        next(b for b in at.button if b.key == key).click().run()
        assert not at.exception, key
    assert any("Der Wert der starken LP ist in" in c.value for c in at.caption)


def test_source_has_explicit_chart_keys_and_locked_axes():
    app = APP.read_text(encoding="utf-8")
    assert all(re.search(r"plotly_chart\(.*key=", line) for line in app.splitlines() if "st.plotly_chart(" in line)
    viz = (ROOT / "ufl_visualization.py").read_text(encoding="utf-8")
    assert viz.count("return _base(fig") >= 8 and "def lock_axes" in viz
