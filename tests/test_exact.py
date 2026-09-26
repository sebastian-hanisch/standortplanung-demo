"""Exakte Rechnung: MILP gegen Brute Force, starke und schwache LP, Lehrnetz mit gebrochener LP von Hand."""

import pytest

import ufl_evaluation as ev
import ufl_exact as ex
import ufl_scenario as sc
from tests.test_heuristics import _tiny  # noqa: E402


@pytest.mark.parametrize("seed", range(30))
def test_milp_equals_brute_force(seed):
    inst = _tiny(seed, m=6, n=8)
    opt, sets = ex.brute_force(inst)
    z, open_set = ex.solve_mip(inst)
    assert z == opt and open_set in sets


@pytest.mark.parametrize("seed", range(20))
def test_lp_bounds_are_ordered(seed):
    """schwache LP <= starke LP <= Optimum: die starke Kopplung ist strenger."""
    inst = _tiny(seed, m=6, n=8)
    opt, _ = ex.brute_force(inst)
    weak, _yw = ex.solve_lp(inst, False)
    strong, _ys = ex.solve_lp(inst, True)
    assert weak <= strong + 1e-6 <= opt + 2e-6


def test_lp_of_the_fractional_teaching_net_by_hand():
    """Lehrnetz 'lp_frac': Optimum 25 (S2 allein), starke LP 24,5, schwache LP 18,25 (jeweils ein Bruchteil der Fixkosten)."""
    inst = sc.teaching("lp_frac")
    assert ex.brute_force(inst)[0] == 25
    strong, ys = ex.solve_lp(inst, True)
    weak, yw = ex.solve_lp(inst, False)
    assert strong == pytest.approx(24.5, abs=1e-6) and weak == pytest.approx(18.25, abs=1e-6)
    assert ex.fractional_sites(ys) and ex.fractional_sites(yw)


def test_weak_lp_is_strictly_below_the_strong_lp_on_the_teaching_nets():
    for name in sc.TEACHING:
        inst = sc.teaching(name)
        assert ex.solve_lp(inst, False)[0] < ex.solve_lp(inst, True)[0] - 1e-6


def test_fractional_sites_threshold():
    assert ex.fractional_sites((0.0, 1.0, 0.5, 1e-9, 1 - 1e-9, 0.3)) == (2, 5)


def test_solve_lp_and_mip_on_a_single_site_net():
    inst = sc.UFL("teaching", ("S1",), ("K1", "K2"), (), (), (), (7,), ((2, 3),))
    assert ex.solve_mip(inst) == (12, (0,)) and ex.solve_lp(inst)[0] == pytest.approx(12.0) and ex.brute_force(inst) == (12, [(0,)])


@pytest.mark.parametrize("seed", range(4))
def test_analyse_is_consistent_on_map_nets(seed):
    a = ev.analyse(ev.Params("map", 12, 25, 100, seed))
    opt = a["opt"]
    assert a["lp_weak"] <= a["lp_strong"] + 1e-6 <= opt + 2e-6
    assert a["dual"].lower_bound <= a["dual_adj"].lower_bound <= opt
    for k in ("add", "drop", "interchange"):
        assert a[k].cost >= opt
    assert a["polished"].cost >= opt and a["best_heuristic"].cost == min(a[k].cost for k in ("add", "drop", "interchange"))
