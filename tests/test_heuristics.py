"""Add, Drop, Interchange: Lehrnetze von Hand, Brute Force auf Kleinstnetzen, Invarianten der Züge."""

import pytest

import ufl_evaluation as ev
import ufl_exact as ex
import ufl_heuristics as hu
import ufl_scenario as sc


def _tiny(seed, m=4, n=5, cmax=9, fmax=10):
    r = sc.SplitMix64(seed)
    c = tuple(tuple(1 + r.below(cmax) for _ in range(n)) for _ in range(m))
    f = tuple(2 + r.below(fmax - 1) for _ in range(m))
    return sc.UFL("teaching", tuple(f"S{i + 1}" for i in range(m)), tuple(f"K{j + 1}" for j in range(n)), (), (), (), f, c)


def test_cost_of_by_hand():
    """S1 kostet 3, S2 kostet 6: mit beiden offen holt sich jeder Kunde die billigere Zeile."""
    inst = sc.UFL("teaching", ("S1", "S2"), ("K1", "K2"), (), (), (), (3, 6), ((1, 9), (5, 2)))
    assert hu.cost_of(inst, {0}) == 3 + 1 + 9
    assert hu.cost_of(inst, {1}) == 6 + 5 + 2
    assert hu.cost_of(inst, {0, 1}) == 3 + 6 + 1 + 2
    assert hu.cost_of(inst, set()) is None
    assert hu.cost_split(inst, {0, 1}) == (9, 3) and hu.assignment(inst, {0, 1}) == (0, 1)


def test_add_trap_add_ends_above_the_optimum_drop_and_interchange_do_not():
    """Lehrnetz Add-Falle: Add öffnet S2 (billigster Einzelzug) und bleibt bei 17; das Optimum 16 ist {S1, S3}."""
    inst = sc.teaching("add_trap")
    opt, sets = ex.brute_force(inst)
    assert (opt, sets) == (16, [(0, 2)])
    add, drop, ich = hu.run(inst, "add"), hu.run(inst, "drop"), hu.run(inst, "interchange")
    assert (add.cost, add.open_set) == (17, (0, 1)) and (drop.cost, drop.open_set) == (16, (0, 2)) and (ich.cost, ich.open_set) == (16, (0, 2))
    assert [(m.kind, m.opened) for m in add.trace] == [("open", 1), ("open", 0)]


def test_drop_trap_drop_ends_above_the_optimum_add_finds_it():
    inst = sc.teaching("drop_trap")
    opt, sets = ex.brute_force(inst)
    assert (opt, sets) == (21, [(0,)])
    add, drop = hu.run(inst, "add"), hu.run(inst, "drop")
    assert (add.cost, add.open_set) == (21, (0,)) and (drop.cost, drop.open_set) == (22, (1, 2))


def test_interchange_gets_stuck_where_drop_finds_the_optimum():
    """Lehrnetz: Add öffnet nur S1 (21); der Optimum {S2, S3} (20) ist von dort kein einzelner Öffnen-, Schließen- oder Tauschzug entfernt, Drop findet es."""
    inst = sc.teaching("stuck")
    opt, sets = ex.brute_force(inst)
    assert (opt, sets) == (20, [(1, 2)])
    add, drop, ich = hu.run(inst, "add"), hu.run(inst, "drop"), hu.run(inst, "interchange")
    assert (add.cost, add.open_set) == (21, (0,)) and (ich.cost, ich.open_set, len(ich.trace)) == (21, (0,), 0) and drop.cost == 20


def test_both_trap_needs_interchange():
    inst = sc.teaching("both_trap")
    opt, sets = ex.brute_force(inst)
    assert (opt, sets) == (27, [(0, 1)])
    add, drop, ich = hu.run(inst, "add"), hu.run(inst, "drop"), hu.run(inst, "interchange")
    assert (add.cost, add.open_set, drop.cost, drop.open_set) == (28, (1, 3), 28, (1, 3))
    assert (ich.cost, ich.open_set) == (27, (0, 1)) and [m.kind for m in ich.trace] == ["swap"]


@pytest.mark.parametrize("seed", range(40))
def test_heuristics_are_valid_and_never_beat_the_optimum(seed):
    inst = _tiny(seed)
    opt, _sets = ex.brute_force(inst)
    for method in hu.METHODS:
        r = hu.run(inst, method)
        assert r.cost == hu.cost_of(inst, r.open_set) >= opt
        assert r.evals >= sum(m.evals for m in r.trace)          # die letzte Runde (kein Zug mehr) bewertet auch Nachbarn und gehört zu keinem Zug


@pytest.mark.parametrize("seed", range(20))
def test_trace_costs_fall_strictly_and_replay_to_the_result(seed):
    inst = _tiny(seed, m=5, n=6)
    for method in hu.METHODS:
        r = hu.run(inst, method)
        costs = [m.cost_after for m in r.trace]
        assert all(b < a for a, b in zip(costs, costs[1:]))
        assert all(m.cost_before is None or m.cost_after < m.cost_before for m in r.trace)
        st = ev.states(r)
        assert st[-1][0] == r.open_set and (st[-1][1] == r.cost or not r.trace)
        for (open_k, cost_k) in st:
            if cost_k is not None:
                assert hu.cost_of(inst, open_k) == cost_k


def test_add_starts_with_the_cheapest_single_site():
    inst = _tiny(3)
    first = hu.run(inst, "add").trace[0]
    singles = [hu.cost_of(inst, {i}) for i in range(inst.m)]
    assert first.cost_after == min(singles) and first.opened == singles.index(min(singles)) and first.cost_before is None


def test_drop_starts_with_everything_open_and_keeps_at_least_one():
    inst = _tiny(4, m=6, n=3)
    r = hu.run(inst, "drop")
    assert r.start_open == tuple(range(6)) and r.start_cost == hu.cost_of(inst, range(6)) and len(r.open_set) >= 1


def test_ties_break_towards_the_smaller_index():
    """Zwei gleiche Standorte: Add öffnet den mit dem kleineren Index."""
    inst = sc.UFL("teaching", ("S1", "S2"), ("K1",), (), (), (), (5, 5), ((2,), (2,)))
    assert hu.run(inst, "add").open_set == (0,)


def test_the_three_methods_differ_on_some_net_of_each_kind():
    """Zweig-Test: keine Nullspalte - Add, Drop und Interchange weichen auf mindestens einem der Lehrnetze voneinander ab."""
    seen = set()
    for name in sc.TEACHING:
        inst = sc.teaching(name)
        a, d, i = (hu.run(inst, m).cost for m in ("add", "drop", "interchange"))
        seen |= {("add>drop", a > d), ("drop>add", d > a), ("interchange<add", i < a), ("interchange>drop", i > d)}
    assert {("add>drop", True), ("drop>add", True), ("interchange<add", True), ("interchange>drop", True)} <= seen


def test_states_of_add_begin_with_nothing_open():
    inst = sc.teaching("add_trap")
    st = ev.states(hu.run(inst, "add"))
    assert st[0] == ((), None) and st[1][0] == (1,) and st[-1] == ((0, 1), 17)


def test_describe_move():
    inst = sc.teaching("both_trap")
    assert ev.describe_move(inst, hu.run(inst, "interchange").trace[0]) == "S4 schließen, S1 öffnen"
    assert ev.describe_move(inst, hu.run(inst, "add").trace[0]) == "S4 öffnen" and ev.describe_move(inst, hu.run(inst, "drop").trace[0]) == "S1 schließen"   # von Hand: Einzelkosten S4 29 < 33 < 33 < 34; Drop: 32 bei S1 und S3, Gleichstand -> S1
