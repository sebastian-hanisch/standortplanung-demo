"""Standortwahl von Hand und mit Heuristiken: Add (nacheinander öffnen), Drop (nacheinander schließen), Interchange (öffnen, schließen, tauschen).

Jede Heuristik wählt in jedem Zug den besten Nachbarn (kleinste Gesamtkosten, Gleichstand: kleinster Index) und hält an, sobald kein Zug die Kosten
strikt senkt. Aufwand = Zahl der Kostenbewertungen (eine Bewertung kostet n Kunden mal die Zahl der offenen Standorte).
"""

from dataclasses import dataclass

import numpy as np


def arrays(inst):
    """Fixkosten und Kostenmatrix als numpy-Ganzzahl-Felder (einmal je Heuristik-Lauf)."""
    return np.array(inst.f, dtype=np.int64), np.array(inst.c, dtype=np.int64)


def cost_of(inst, open_set, arr=None):
    """Gesamtkosten einer Auswahl (Fixkosten + je Kunde der billigste offene Standort); None, wenn nichts offen ist."""
    idx = sorted(open_set)
    if not idx:
        return None
    f, c = arr if arr is not None else arrays(inst)
    return int(f[idx].sum() + c[idx].min(axis=0).sum())


def cost_split(inst, open_set):
    """(Fixkosten, Zuordnungskosten) der Auswahl."""
    idx = sorted(open_set)
    f, c = arrays(inst)
    return int(f[idx].sum()), int(c[idx].min(axis=0).sum())


def assignment(inst, open_set):
    """Standort je Kunde (Index in `inst`); Gleichstand: der kleinste Index."""
    idx = sorted(open_set)
    _f, c = arrays(inst)
    return tuple(int(idx[k]) for k in c[idx].argmin(axis=0))


@dataclass(frozen=True)
class Move:
    kind: str            # "open", "close", "swap"
    opened: int          # -1, falls nichts geöffnet wird
    closed: int          # -1, falls nichts geschlossen wird
    cost_before: int     # None beim ersten Zug von "keine offen"
    cost_after: int
    evals: int           # in diesem Zug bewertete Nachbarn


@dataclass(frozen=True)
class Result:
    method: str
    open_set: tuple
    cost: int
    start_open: tuple
    start_cost: int
    trace: tuple         # Moves
    evals: int


def _best(candidates, key_cost):
    """Bester Kandidat (Kosten, Schlüssel); Gleichstand entscheidet der kleinere Schlüssel."""
    return min(candidates, key=key_cost)


def add(inst, arr=None):
    """Add: mit dem billigsten Einzelstandort beginnen, dann jeweils den Standort öffnen, der die Gesamtkosten am stärksten senkt."""
    arr = arr or arrays(inst)
    cur = set()
    cost = None
    trace = []
    total = 0
    while True:
        cands = [(cost_of(inst, cur | {i}, arr), i) for i in range(inst.m) if i not in cur]
        if not cands:
            break
        total += len(cands)
        best_cost, best_i = _best(cands, lambda t: (t[0], t[1]))
        if cost is not None and best_cost >= cost:
            break
        trace.append(Move("open", best_i, -1, cost, best_cost, len(cands)))
        cur.add(best_i)
        cost = best_cost
    return Result("add", tuple(sorted(cur)), cost, (), None, tuple(trace), total)


def drop(inst, arr=None):
    """Drop: alle Standorte offen, dann jeweils den schließen, der die Gesamtkosten am stärksten senkt (mindestens einer bleibt offen)."""
    arr = arr or arrays(inst)
    cur = set(range(inst.m))
    cost = cost_of(inst, cur, arr)
    start_cost = cost
    trace = []
    total = 0
    while len(cur) > 1:
        cands = [(cost_of(inst, cur - {i}, arr), i) for i in sorted(cur)]
        total += len(cands)
        best_cost, best_i = _best(cands, lambda t: (t[0], t[1]))
        if best_cost >= cost:
            break
        trace.append(Move("close", -1, best_i, cost, best_cost, len(cands)))
        cur.discard(best_i)
        cost = best_cost
    return Result("drop", tuple(sorted(cur)), cost, tuple(range(inst.m)), start_cost, tuple(trace), total)


def interchange(inst, start_open, arr=None):
    """Interchange: von einer Startauswahl aus jeweils den besten Zug wählen: einen Standort öffnen, einen schließen oder einen tauschen."""
    arr = arr or arrays(inst)
    cur = set(start_open)
    cost = cost_of(inst, cur, arr)
    start_cost = cost
    trace = []
    total = 0
    while True:
        cands = []
        for i in range(inst.m):
            if i not in cur:
                cands.append((cost_of(inst, cur | {i}, arr), 0, i, -1))
        if len(cur) > 1:
            for i in sorted(cur):
                cands.append((cost_of(inst, cur - {i}, arr), 1, -1, i))
        for a in sorted(cur):
            for b in range(inst.m):
                if b not in cur:
                    cands.append((cost_of(inst, (cur - {a}) | {b}, arr), 2, b, a))
        total += len(cands)
        best = _best(cands, lambda t: t[:1] + (t[1], t[2], t[3]))
        if best[0] >= cost:
            break
        _c, kind, opened, closed = best
        trace.append(Move(("open", "close", "swap")[kind], opened, closed, cost, best[0], len(cands)))
        if opened >= 0:
            cur.add(opened)
        if closed >= 0:
            cur.discard(closed)
        cost = best[0]
    return Result("interchange", tuple(sorted(cur)), cost, tuple(sorted(start_open)), start_cost, tuple(trace), total)


METHODS = {"add": "Add (nacheinander öffnen)", "drop": "Drop (nacheinander schließen)", "interchange": "Interchange (öffnen, schließen, tauschen)"}


def run(inst, method, start=None):
    """Heuristik nach Namen; Interchange startet bei `start` (Standard: das Ergebnis von Add)."""
    arr = arrays(inst)
    if method == "add":
        return add(inst, arr)
    if method == "drop":
        return drop(inst, arr)
    if method == "interchange":
        if start is None:
            start = add(inst, arr).open_set
        return interchange(inst, start, arr)
    raise KeyError(method)
