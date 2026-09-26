"""Dual Ascent: von Hand gerechnetes Beispiel, Dualzulässigkeit, Maximalität des Aufstiegs, Schranke <= Optimum <= primale Auswahl, Anpassung."""

import pytest

import ufl_dual as dl
import ufl_exact as ex
import ufl_scenario as sc
from tests.test_heuristics import _tiny  # noqa: E402  (kleine Zufallsnetze)


def _by_hand():
    """S1 (Fix 3) und S2 (Fix 6); K1 kostet 1 bzw. 5, K2 kostet 9 bzw. 2. Optimum: beide offen, 3 + 6 + 1 + 2 = 12."""
    return sc.UFL("teaching", ("S1", "S2"), ("K1", "K2"), (), (), (), (3, 6), ((1, 9), (5, 2)))


def test_dual_ascent_by_hand():
    """Start v = (1, 2). K1: bis S2 (Knick bei 5) oder S1 (Rest 3): +3 -> v1 = 4, S1 knapp. K2: bis S2 (Rest 6) vor S1 (Knick 9): +6 -> v2 = 8, S2 knapp. Schranke 12."""
    inst = _by_hand()
    res = dl.dual_ascent(inst, "index")
    assert res.v == (4, 8) and res.lower_bound == 12 and res.slack == (0, 0) and res.tight == (0, 1)
    assert [(s.customer, s.v_before, s.v_after, s.blocked_by, s.became_tight) for s in res.steps] == [(0, 1, 4, 0, (0,)), (1, 2, 8, 1, (1,))]
    assert res.primal_open == (0, 1) and res.primal_cost == 12 and res.gap == 0 and res.proven
    assert ex.brute_force(inst)[0] == 12


def test_free_sites_give_the_trivial_bound():
    """Kosten 0 für beide Standorte: der Aufstieg bleibt bei den Mindestkosten (1 + 2), und mehr als das kostet auch das Optimum nicht."""
    inst = sc.UFL("teaching", ("S1", "S2"), ("K1", "K2"), (), (), (), (0, 0), ((1, 9), (5, 2)))
    res = dl.dual_ascent(inst)
    assert res.lower_bound == 3                                # v = min-Kosten (1 + 2), beide Standorte kosten nichts


@pytest.mark.parametrize("order", list(dl.ORDERS))
@pytest.mark.parametrize("seed", range(15))
def test_ascent_is_dual_feasible_maximal_and_below_the_optimum(seed, order):
    inst = _tiny(seed, m=5, n=7)
    opt, _sets = ex.brute_force(inst)
    s, steps = dl.ascent(inst, order)
    assert dl.is_dual_feasible(inst, s.v)
    assert sum(s.v) <= opt
    for j in range(inst.n):                                    # maximal: jeder Kunde wird von einem knappen Standort blockiert
        assert any(inst.c[i][j] <= s.v[j] and s.slack[i] == 0 for i in range(inst.m))
    assert all(sl >= 0 for sl in s.slack) and all(s.slack[i] == inst.f[i] - sum(max(0, s.v[j] - inst.c[i][j]) for j in range(inst.n)) for i in range(inst.m))
    assert len(steps) == inst.n and sorted(st.customer for st in steps) == list(range(inst.n))


@pytest.mark.parametrize("order", list(dl.ORDERS))
@pytest.mark.parametrize("seed", range(15))
def test_adjustment_never_lowers_the_bound_and_keeps_feasibility(seed, order):
    inst = _tiny(seed, m=5, n=7)
    opt, _sets = ex.brute_force(inst)
    plain = dl.dual_ascent(inst, order, adjusted=False)
    adj = dl.dual_ascent(inst, order, adjusted=True)
    assert plain.lower_bound <= adj.lower_bound <= opt
    assert dl.is_dual_feasible(inst, adj.v) and sum(adj.v) == adj.lower_bound
    assert adj.primal_cost >= opt and (not adj.proven or adj.primal_cost == opt == adj.lower_bound)
    assert adj.gap == adj.primal_cost - adj.lower_bound >= 0


@pytest.mark.parametrize("seed", range(6))
def test_bound_is_below_the_optimum_on_map_and_ties_nets(seed):
    for inst in (sc.generate_map(12, 25, 100, seed), sc.generate_ties(12, 15, 100, seed)):
        opt, _ = ex.solve_mip(inst)
        for adjusted in (False, True):
            r = dl.dual_ascent(inst, "index", adjusted)
            assert dl.is_dual_feasible(inst, r.v) and r.lower_bound <= opt <= r.primal_cost


def test_dual_ascent_is_deterministic():
    inst = sc.generate_map(15, 30, 100, 4)
    assert dl.dual_ascent(inst, "far", True) == dl.dual_ascent(inst, "far", True)


def test_orders_are_permutations_and_differ():
    inst = sc.generate_map(15, 30, 100, 4)
    orders = {o: dl.customer_order(inst, o) for o in dl.ORDERS}
    assert all(sorted(v) == list(range(inst.n)) for v in orders.values()) and orders["reverse"] == orders["index"][::-1]
    assert len({tuple(v) for v in orders.values()}) == 4
    with pytest.raises(KeyError):
        dl.customer_order(inst, "zufall")


def test_teaching_net_where_dual_ascent_proves_the_optimum():
    """Lehrnetz 'proven': ohne Anpassung 23, mit Anpassung 25 = Optimum = die Kosten der knappen Standorte {S1, S2}."""
    inst = sc.teaching("proven")
    plain, adj = dl.dual_ascent(inst, "index", False), dl.dual_ascent(inst, "index", True)
    assert (plain.lower_bound, adj.lower_bound, adj.primal_cost, adj.primal_open, adj.proven) == (23, 25, 25, (0, 1), True)
    assert ex.brute_force(inst)[0] == 25 and not plain.proven and adj.passes == 1


def test_adjustment_counts_passes_and_ops_grow():
    inst = sc.generate_map(20, 40, 100, 3)
    plain, adj = dl.dual_ascent(inst, "index", False), dl.dual_ascent(inst, "index", True)
    assert plain.passes == 0 and adj.passes >= 1 and adj.ops > plain.ops
