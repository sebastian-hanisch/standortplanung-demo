"""Exakte Rechnung und LP-Schranken (HiGHS über scipy): das Optimum, die starke und die schwache LP-Relaxation, Brute Force für Kleinstnetze.

Starke Formulierung: x_ij <= y_i für jedes Paar. Schwache Formulierung: Σ_j x_ij <= n · y_i (eine Zeile je Standort; ein Standort zahlt nur den Bruchteil
seiner Fixkosten, den er an Kunden liefert). Das ist dieselbe Frage wie im Fixkosten-Netzdesign, wo die schwache Kopplung die Schranke auf etwa 70 %
drückt.
"""

from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from ufl_heuristics import arrays, cost_of

EPS = 1e-6


def _model(inst, strong):
    m, n = inst.m, inst.n
    nv = m + m * n                     # y_0..y_{m-1}, dann x_ij zeilenweise (i * n + j)
    cost = np.concatenate([np.array(inst.f, dtype=float), np.array(inst.c, dtype=float).ravel()])
    rows, cols, vals = [], [], []
    lo, hi = [], []
    r = 0
    for j in range(n):                 # jeder Kunde genau einmal
        for i in range(m):
            rows.append(r); cols.append(m + i * n + j); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
    if strong:
        for i in range(m):
            for j in range(n):
                rows += [r, r]; cols += [m + i * n + j, i]; vals += [1.0, -1.0]
                lo.append(-np.inf); hi.append(0.0); r += 1
    else:
        for i in range(m):
            for j in range(n):
                rows.append(r); cols.append(m + i * n + j); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-float(n))
            lo.append(-np.inf); hi.append(0.0); r += 1
    return cost, coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr(), np.array(lo), np.array(hi)


def solve_lp(inst, strong=True):
    """LP-Relaxation: (Wert, y je Standort). Ein Wert unter dem Optimum ist die Schranke."""
    cost, a, lo, hi = _model(inst, strong)
    res = linprog(cost, method="highs", bounds=(0, 1), **_split(a, lo, hi))
    if res.status != 0:
        raise RuntimeError("LP nicht lösbar: " + str(res.message))
    return float(res.fun), tuple(float(v) for v in res.x[:inst.m])


def _split(a, lo, hi):
    """Gleichungen (lo == hi) und Ungleichungen (hi <= 0) für linprog trennen."""
    eq = [k for k in range(len(lo)) if lo[k] == hi[k]]
    ub = [k for k in range(len(lo)) if lo[k] != hi[k]]
    out = {"A_eq": a[eq], "b_eq": lo[eq]}
    if ub:
        out["A_ub"] = a[ub]
        out["b_ub"] = hi[ub]
    return out


def solve_mip(inst):
    """Optimum (ganzzahlige Kosten) und offene Standorte; die Standorte sind ganzzahlig, die Zuordnung ergibt sich von selbst."""
    cost, a, lo, hi = _model(inst, True)
    integrality = np.concatenate([np.ones(inst.m), np.zeros(inst.m * inst.n)])
    res = milp(cost, constraints=LinearConstraint(a, lo, hi), integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": 120, "mip_rel_gap": 0.0})
    if res.status != 0:
        raise RuntimeError("MILP nicht optimal gelöst: " + str(res.message))
    open_set = tuple(i for i in range(inst.m) if res.x[i] > 0.5)
    return int(round(res.fun)), open_set


def brute_force(inst):
    """Alle 2^m - 1 Auswahlen (nur für kleine m): Optimum und alle optimalen Auswahlen."""
    arr = arrays(inst)
    best, sets = None, []
    for k in range(1, inst.m + 1):
        for s in combinations(range(inst.m), k):
            v = cost_of(inst, s, arr)
            if best is None or v < best:
                best, sets = v, [s]
            elif v == best:
                sets.append(s)
    return best, sets


def fractional_sites(y):
    """Standorte, die die LP-Lösung nur anteilig öffnet."""
    return tuple(i for i, v in enumerate(y) if EPS < v < 1 - EPS)
