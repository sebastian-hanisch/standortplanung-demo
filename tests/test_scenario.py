"""Szenario: Zufallsstrom, Netze, Lehrnetze - ganzzahlig, deterministisch, in den erwarteten Grenzen."""

import pytest

import ufl_constants as C
import ufl_scenario as sc


def test_splitmix64_stream_is_the_portfolio_standard():
    rng = sc.SplitMix64(1)
    assert [rng.next() for _ in range(2)] == [10451216379200822465, 13757245211066428519]


def test_distance_is_integer_euclid_in_tenths():
    assert sc.distance((0, 0), (3, 4)) == 50
    assert sc.distance((5, 5), (5, 5)) == 0
    assert sc.distance((0, 0), (1, 1)) == 14            # 10 * 1,414... abgerundet


def test_map_net_is_deterministic_and_integer():
    a = sc.generate_map(25, 50, 100, 1)
    b = sc.generate_map(25, 50, 100, 1)
    assert a == b and a != sc.generate_map(25, 50, 100, 2)
    assert (a.m, a.n, a.kind, a.has_map) == (25, 50, "map", True)
    assert all(isinstance(v, int) for v in a.f) and all(isinstance(v, int) for row in a.c for v in row)
    assert all(1 <= d <= 9 for d in a.demand)
    assert all(0 <= x < sc.MAP_W and 0 <= y < sc.MAP_W for x, y in a.site_pos + a.cust_pos)


def test_map_cost_is_demand_times_distance():
    inst = sc.generate_map(6, 9, 100, 3)
    for i in range(inst.m):
        for j in range(inst.n):
            assert inst.c[i][j] == inst.demand[j] * sc.distance(inst.site_pos[i], inst.cust_pos[j])


def test_fixed_costs_scale_with_the_factor_and_stay_in_the_60_to_140_percent_band():
    base = sc.generate_map(25, 50, 100, 5)
    assert all(sc.FIXED_BASE * 60 // 100 <= f <= sc.FIXED_BASE * 140 // 100 for f in base.f)
    double = sc.generate_map(25, 50, 200, 5)
    assert double.c == base.c and all(d == b * 2 for d, b in zip(double.f, base.f))       # 1750 * k // 100 * 200 // 100: glatt teilbar


def test_positions_and_costs_do_not_depend_on_the_fixed_factor():
    a, b = sc.generate_map(10, 20, 25, 7), sc.generate_map(10, 20, 400, 7)
    assert (a.site_pos, a.cust_pos, a.demand, a.c) == (b.site_pos, b.cust_pos, b.demand, b.c) and a.f != b.f


def test_ties_net_has_small_repeating_costs():
    inst = sc.generate_ties(30, 30, 100, 1)
    assert (inst.m, inst.n, inst.kind, inst.has_map) == (30, 30, "ties", False)
    assert all(1 <= v <= 5 for row in inst.c for v in row) and set(inst.f) <= {3, 4}
    assert sc.generate_ties(30, 30, 100, 1) == inst


@pytest.mark.parametrize("name", list(sc.TEACHING))
def test_teaching_nets_are_well_formed(name):
    inst = sc.teaching(name)
    assert inst.kind == "teaching" and not inst.has_map
    assert len(inst.c) == inst.m and all(len(row) == inst.n for row in inst.c)
    assert len(inst.site_names) == inst.m and len(inst.cust_names) == inst.n
    assert all(f > 0 for f in inst.f) and all(v > 0 for row in inst.c for v in row)


def test_unknown_teaching_net_raises():
    with pytest.raises(KeyError):
        sc.teaching("gibt-es-nicht")


def test_every_teaching_net_in_the_constants_exists():
    assert set(C.FIXED_NETS) == set(sc.TEACHING)
