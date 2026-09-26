"""Dual Ascent nach Erlenkotter (1978): eine untere Schranke ohne LP-Löser.

Das duale Programm der starken Formulierung lautet: maximiere die Summe der Kundenwerte v_j, sodass für jeden Standort i die Summe der Beiträge
max(0, v_j - c_ij) höchstens die Fixkosten f_i beträgt. Der Aufstieg beginnt bei v_j = billigster Standort und hebt jeden Kundenwert an, bis ein Standort
"voll bezahlt" (knapp) ist. Ein knapper Standort blockiert alle Kunden, die zu ihm beitragen; ein einmal blockierter Kunde bleibt blockiert, weil die Restbeträge
nur sinken. Ein Durchlauf über die Kunden genügt daher, und jede Zahl bleibt ganzzahlig.

Primale Lösung aus der Komplementarität: eröffnet werden nur knappe Standorte, und zwar so, dass kein Kunde zu zwei offenen Standorten beiträgt (Beitrag > 0).
Dann sind die Kosten gleich der Schranke und die Lösung ist bewiesen optimal; sonst bleibt eine Lücke.
"""

from dataclasses import dataclass

from ufl_heuristics import cost_of, arrays

ORDERS = {
    "index": "Kunden in Reihenfolge",
    "reverse": "Kunden rückwärts",
    "near": "nächste Kunden zuerst (kleinster Mindestabstand)",
    "far": "entfernteste Kunden zuerst",
}


def customer_order(inst, order):
    js = list(range(inst.n))
    mins = [min(inst.c[i][j] for i in range(inst.m)) for j in range(inst.n)]
    if order == "index":
        return js
    if order == "reverse":
        return js[::-1]
    if order == "near":
        return sorted(js, key=lambda j: (mins[j], j))
    if order == "far":
        return sorted(js, key=lambda j: (-mins[j], j))
    raise KeyError(order)


@dataclass(frozen=True)
class Raise:
    customer: int
    v_before: int
    v_after: int
    blocked_by: int      # Standort, der den Kunden blockiert (knapp), -1 nie
    became_tight: tuple  # Standorte, die in diesem Schritt knapp wurden


@dataclass(frozen=True)
class DualResult:
    order: str
    adjusted: bool       # mit Anpassungsschritt
    v: tuple             # Kundenwerte
    slack: tuple         # Restbetrag je Standort (0 = knapp)
    lower_bound: int
    tight: tuple         # knappe Standorte in der Reihenfolge, in der sie knapp wurden
    steps: tuple         # Raises
    primal_open: tuple
    primal_cost: int
    gap: int             # primale Kosten minus Schranke
    proven: bool         # gap == 0: die primale Lösung ist bewiesen optimal
    ops: int             # Kantenoperationen (geprüfte Kunde-Standort-Paare)
    passes: int          # Anpassungsdurchläufe mit Verbesserung


class _State:
    """Kundenwerte, Restbeträge und die Reihenfolge, in der die Standorte knapp wurden (veränderlich, mit Kopie für die Anpassung)."""

    def __init__(self, inst):
        m, n, c, f = inst.m, inst.n, inst.c, inst.f
        self.v = [min(c[i][j] for i in range(m)) for j in range(n)]
        self.slack = [f[i] - sum(max(0, self.v[j] - c[i][j]) for j in range(n)) for i in range(m)]
        if min(self.slack) < 0:
            raise ValueError("Startwerte verletzen die Dualbedingung (Fixkosten kleiner als die Startbeiträge)")
        self.tight = [i for i in range(m) if self.slack[i] == 0]
        self.ops = 0

    def copy(self):
        o = object.__new__(_State)
        o.v, o.slack, o.tight, o.ops = self.v[:], self.slack[:], self.tight[:], self.ops
        return o


def _raise_customer(inst, s, j):
    """Kundenwert j anheben, bis ein knapper Standort ihn blockiert; liefert (Blocker, in diesem Schritt knapp gewordene Standorte)."""
    m, c = inst.m, inst.c
    tight_now = []
    while True:
        s.ops += m
        blockers = [i for i in range(m) if c[i][j] <= s.v[j] and s.slack[i] == 0]
        if blockers:
            return blockers[0], tight_now
        delta = min([c[i][j] - s.v[j] for i in range(m) if c[i][j] > s.v[j]] + [s.slack[i] for i in range(m) if c[i][j] <= s.v[j]])
        for i in range(m):
            if c[i][j] <= s.v[j]:
                s.slack[i] -= delta
                if s.slack[i] == 0:
                    tight_now.append(i)
                    s.tight.append(i)
        s.v[j] += delta


def _lower_customer(inst, s, j, new_v):
    """Kundenwert j auf new_v senken: die frei werdenden Beträge gehen an die Standorte zurück (knapp gewesene Standorte verlieren ihre Markierung)."""
    for i in range(inst.m):
        freed = max(0, s.v[j] - inst.c[i][j]) - max(0, new_v - inst.c[i][j])
        if freed:
            was_tight = s.slack[i] == 0
            s.slack[i] += freed
            if was_tight and i in s.tight:
                s.tight.remove(i)
    s.v[j] = new_v


def ascent(inst, order="index"):
    s = _State(inst)
    steps = []
    for j in customer_order(inst, order):
        v0 = s.v[j]
        blocked, tight_now = _raise_customer(inst, s, j)
        steps.append(Raise(j, v0, s.v[j], blocked, tuple(tight_now)))
    return s, steps


def adjust(inst, s, order="index", max_passes=5):
    """Anpassung (Erlenkotter, vereinfacht): den Wert eines Kunden auf den niedrigsten knappen Standort senken, alle anderen erneut anheben; behalten, wenn die
    Summe steigt. Wiederholen, bis ein Durchlauf nichts mehr bringt. Liefert die Zahl der Durchläufe mit Verbesserung."""
    order_js = customer_order(inst, order)
    improved_passes = 0
    for _ in range(max_passes):
        improved = False
        for j in order_js:
            blockers = [inst.c[i][j] for i in range(inst.m) if s.slack[i] == 0 and inst.c[i][j] <= s.v[j]]
            if not blockers or min(blockers) >= s.v[j]:
                continue
            trial = s.copy()
            before = sum(trial.v)
            _lower_customer(inst, trial, j, min(blockers))
            for k in order_js:
                if k != j:
                    _raise_customer(inst, trial, k)
            _raise_customer(inst, trial, j)
            s.ops = trial.ops
            if sum(trial.v) > before:
                s.v, s.slack, s.tight = trial.v, trial.slack, trial.tight
                improved = True
        if not improved:
            break
        improved_passes += 1
    return improved_passes


def primal_from_tight(inst, v, tight_order):
    """Komplementarität: knappe Standorte in umgekehrter Reihenfolge des Knappwerdens öffnen, sofern kein Kunde zu beiden beiträgt."""
    contributes = [[j for j in range(inst.n) if v[j] > inst.c[i][j]] for i in range(inst.m)]
    used = set()
    opened = []
    for i in reversed(tight_order):
        if any(j in used for j in contributes[i]):
            continue
        opened.append(i)
        used.update(contributes[i])
    return tuple(sorted(opened))


def dual_ascent(inst, order="index", adjusted=False):
    s, steps = ascent(inst, order)
    passes = adjust(inst, s, order) if adjusted else 0
    lb = sum(s.v)
    opened = primal_from_tight(inst, s.v, s.tight)
    arr = arrays(inst)
    cost = cost_of(inst, opened, arr)
    if cost is None:
        opened = tuple(s.tight[:1])
        cost = cost_of(inst, opened, arr)
    return DualResult(order, adjusted, tuple(s.v), tuple(s.slack), lb, tuple(s.tight), tuple(steps), opened, cost, cost - lb, cost == lb, s.ops, passes)


def is_dual_feasible(inst, v):
    """Prüfung für die Tests: Σ_j max(0, v_j - c_ij) <= f_i für alle Standorte."""
    return all(sum(max(0, v[j] - inst.c[i][j]) for j in range(inst.n)) <= inst.f[i] for i in range(inst.m))
