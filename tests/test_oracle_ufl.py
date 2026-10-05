"""Orakel-Test: Add, Drop und Interchange gegen naive Neuimplementierungen (Auswahl, Kosten, Zugzahl, bewertete Nachbarn); starke LP gegen das selbst gelöste Dual; Dual-Ascent-Schranke darunter."""

import random

import numpy as np
import pytest
from scipy.optimize import linprog

import ufl_dual as dl
import ufl_exact as ex
import ufl_heuristics as hu
import ufl_scenario as sc


def _inst(seed):
    rng = random.Random(seed)
    m, n = rng.randint(2, 6), rng.randint(2, 7)
    if seed % 3 == 0:                                            # gleichstandsreich, auch Fixkosten 0
        c = [[rng.randint(0, 4) for _ in range(n)] for _ in range(m)]
        f = [rng.randint(0, 3) for _ in range(m)]
    else:
        c = [[rng.randint(1, 30) for _ in range(n)] for _ in range(m)]
        f = [rng.randint(1, 40) for _ in range(m)]
    return f, c, sc.UFL("teaching", tuple(f"S{i}" for i in range(m)), tuple(f"K{j}" for j in range(n)), (), (), (), tuple(f), tuple(tuple(r) for r in c))


def _cost(f, c, s):
    return sum(f[i] for i in s) + sum(min(c[i][j] for i in s) for j in range(len(c[0])))


def _add(f, c):
    cur, cc, trace, evals = [], None, 0, 0
    while len(cur) < len(f):
        opts = sorted((_cost(f, c, cur + [i]), i) for i in range(len(f)) if i not in cur)
        evals += len(opts)
        if cc is not None and opts[0][0] >= cc:
            break
        cur.append(opts[0][1])
        cc, trace = opts[0][0], trace + 1
    return tuple(sorted(cur)), cc, trace, evals


def _drop(f, c):
    cur, evals, trace = list(range(len(f))), 0, 0
    cc = _cost(f, c, cur)
    while len(cur) > 1:
        res = sorted((_cost(f, c, [x for x in cur if x != i]), i) for i in cur)
        evals += len(cur)
        if res[0][0] >= cc:
            break
        cur.remove(res[0][1])
        cc, trace = res[0][0], trace + 1
    return tuple(sorted(cur)), cc, trace, evals


def _inter(f, c, start):
    m, cur, evals, trace = len(f), set(start), 0, 0
    cc = _cost(f, c, cur)
    while True:
        cands = [(_cost(f, c, cur | {i}), 0, i, -1) for i in range(m) if i not in cur]
        if len(cur) > 1:
            cands += [(_cost(f, c, cur - {i}), 1, -1, i) for i in cur]
        cands += [(_cost(f, c, (cur - {a}) | {b}), 2, b, a) for a in cur for b in range(m) if b not in cur]
        evals += len(cands)
        best = min(cands)
        if best[0] >= cc:
            return tuple(sorted(cur)), cc, trace, evals
        if best[2] >= 0:
            cur.add(best[2])
        if best[3] >= 0:
            cur.discard(best[3])
        cc, trace = best[0], trace + 1


def _dual_optimum(f, c):
    """max Σ v_j mit Σ_j w_ij <= f_i, w_ij >= v_j - c_ij, w >= 0 (selbst aufgestellt, ohne ufl_dual)."""
    m, n = len(f), len(c[0])
    rows, rhs = [], []
    for i in range(m):
        r = np.zeros(n + m * n)
        r[n + i * n:n + (i + 1) * n] = 1
        rows.append(r)
        rhs.append(f[i])
        for j in range(n):
            q = np.zeros(n + m * n)
            q[j], q[n + i * n + j] = 1, -1
            rows.append(q)
            rhs.append(c[i][j])
    res = linprog([-1.0] * n + [0.0] * (m * n), A_ub=np.array(rows), b_ub=rhs, bounds=[(None, None)] * n + [(0, None)] * (m * n), method="highs")
    return -res.fun


@pytest.mark.parametrize("seed", range(40))
def test_heuristics_equal_naive_reimplementation(seed):
    f, c, inst = _inst(seed)
    a, d = hu.run(inst, "add"), hu.run(inst, "drop")
    assert (a.open_set, a.cost, len(a.trace), a.evals) == _add(f, c)
    assert (d.open_set, d.cost, len(d.trace), d.evals) == _drop(f, c)
    i = hu.run(inst, "interchange", a.open_set)
    assert (i.open_set, i.cost, len(i.trace), i.evals) == _inter(f, c, a.open_set)


@pytest.mark.parametrize("seed", range(25))
def test_strong_lp_equals_own_dual_and_dual_ascent_stays_below(seed):
    f, c, inst = _inst(seed)
    lp, _y = ex.solve_lp(inst, True)
    assert lp == pytest.approx(_dual_optimum(f, c), abs=1e-7)
    opt, _s = ex.solve_mip(inst)
    for order in dl.ORDERS:
        for adjusted in (False, True):
            r = dl.dual_ascent(inst, order, adjusted)
            assert r.lower_bound <= lp + 1e-7 and r.lower_bound <= opt <= r.primal_cost
            assert r.proven == (r.primal_cost == r.lower_bound)
