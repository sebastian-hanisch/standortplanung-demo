"""Auswertung: alles, was die Oberfläche zu einem Netz zeigt, und die Experimente über feste Netze (Fixkosten-Reihe, Verteilung, Größenreihe)."""

import statistics
from dataclasses import dataclass

import ufl_constants as C
import ufl_dual as dl
import ufl_exact as ex
import ufl_heuristics as hu
import ufl_scenario as sc


@dataclass(frozen=True)
class Params:
    net: str
    sites: int
    customers: int
    fixed: int
    seed: int
    order: str = C.DEFAULT_ORDER


def build(p):
    """Das Netz zu den Parametern; feste Lehrnetze ignorieren die Zufallsregler."""
    if p.net == "map":
        return sc.generate_map(p.sites, p.customers, p.fixed, p.seed)
    if p.net == "ties":
        return sc.generate_ties(p.sites, p.customers, p.fixed, p.seed)
    return sc.teaching(p.net)


def canonical(p):
    """Lehrnetze rechnen unabhängig von den Reglern: gleiche Netze unter demselben Schlüssel."""
    if p.net in C.FIXED_NETS:
        return Params(p.net, C.DEFAULT_SITES, C.DEFAULT_CUSTOMERS, C.DEFAULT_FIXED, C.DEFAULT_SEED, p.order)
    return p


def analyse(p):
    """Netz, Optimum, LP-Schranken, die drei Heuristiken und Dual Ascent (mit und ohne Anpassung, dazu die Politur der primalen Lösung)."""
    inst = build(p)
    opt, opt_set = ex.solve_mip(inst)
    lp_s, y_s = ex.solve_lp(inst, True)
    lp_w, y_w = ex.solve_lp(inst, False)
    add = hu.run(inst, "add")
    drop = hu.run(inst, "drop")
    ich = hu.run(inst, "interchange", add.open_set)
    plain = dl.dual_ascent(inst, p.order, adjusted=False)
    adj = dl.dual_ascent(inst, p.order, adjusted=True)
    polished = hu.interchange(inst, adj.primal_open)
    return dict(inst=inst, opt=opt, opt_set=opt_set, lp_strong=lp_s, y_strong=y_s, lp_weak=lp_w, y_weak=y_w, add=add, drop=drop, interchange=ich,
                dual=plain, dual_adj=adj, polished=polished, best_heuristic=min((add, drop, ich), key=lambda r: (r.cost, r.method)))


def bound_rows(a):
    """Untere und obere Schranken: (Name, Wert, Prozent des Optimums, Art); unten: schwache LP, starke LP, Dual Ascent; oben: die Lösungen der Verfahren."""
    opt = a["opt"]
    vals = [
        ("schwache LP", a["lp_weak"], "untere Schranke"),
        ("starke LP", a["lp_strong"], "untere Schranke"),
        ("Dual Ascent", a["dual"].lower_bound, "untere Schranke"),
        ("Dual Ascent mit Anpassung", a["dual_adj"].lower_bound, "untere Schranke"),
        ("Optimum", opt, "Optimum"),
        ("Add", a["add"].cost, "Lösung"),
        ("Drop", a["drop"].cost, "Lösung"),
        ("Interchange", a["interchange"].cost, "Lösung"),
        ("Dual Ascent: knappe Standorte", a["dual_adj"].primal_cost, "Lösung"),
        ("... mit Interchange", a["polished"].cost, "Lösung"),
    ]
    return [(name, v, 100.0 * v / opt, kind) for name, v, kind in vals]


def gap_pct(cost, opt):
    return 100.0 * (cost - opt) / opt


def _row(inst, order=C.DEFAULT_ORDER):
    opt, opt_set = ex.solve_mip(inst)
    lp_s, y_s = ex.solve_lp(inst, True)
    lp_w, y_w = ex.solve_lp(inst, False)
    add = hu.run(inst, "add")
    drop = hu.run(inst, "drop")
    ich = hu.run(inst, "interchange", add.open_set)
    plain = dl.dual_ascent(inst, order, adjusted=False)
    adj = dl.dual_ascent(inst, order, adjusted=True)
    pol = hu.interchange(inst, adj.primal_open)
    return dict(opt=opt, n_open=len(opt_set), strong=100 * lp_s / opt, weak=100 * lp_w / opt, tight=abs(lp_s - opt) <= 1e-6 * opt,
                add=gap_pct(add.cost, opt), drop=gap_pct(drop.cost, opt), interchange=gap_pct(ich.cost, opt),
                add_evals=add.evals, drop_evals=drop.evals, interchange_evals=ich.evals,
                dual=100 * plain.lower_bound / opt, dual_adj=100 * adj.lower_bound / opt, dual_primal=gap_pct(adj.primal_cost, opt), proven=adj.proven,
                polished=gap_pct(pol.cost, opt), ops_adj=adj.ops)


def _mean(rows, key):
    return statistics.fmean(r[key] for r in rows)


def summary(rows):
    keys = ("strong", "weak", "dual", "dual_adj", "add", "drop", "interchange", "dual_primal", "polished", "n_open", "add_evals", "drop_evals", "interchange_evals", "ops_adj")
    out = {k: _mean(rows, k) for k in keys}
    out.update({f"{k}_median": statistics.median(r[k] for r in rows) for k in ("strong", "weak", "add", "drop", "interchange")})
    out.update({f"{k}_max": max(r[k] for r in rows) for k in ("add", "drop", "interchange", "polished")})
    out.update({f"{k}_min": min(r[k] for r in rows) for k in ("strong", "weak", "dual", "dual_adj")})
    out["tight"] = sum(1 for r in rows if r["tight"])
    out["proven"] = sum(1 for r in rows if r["proven"])
    for k in ("add", "drop", "interchange"):
        out[f"{k}_exact"] = sum(1 for r in rows if r[k] == 0)
    out["count"] = len(rows)
    return out


def distribution(p, seeds=C.SWEEP_SEEDS):
    """Verteilung über feste Netze: Karten- oder Gleichstandsnetz mit den gewählten Größen und Fixkosten (nicht dem Seed)."""
    kind = "ties" if p.net == "ties" else "map"
    gen = sc.generate_ties if kind == "ties" else sc.generate_map
    rows = [_row(gen(p.sites, p.customers, p.fixed, s), p.order) for s in seeds]
    return dict(rows=rows, summary=summary(rows), kind=kind)


def fixed_series(p, seeds=C.SERIES_SEEDS, fixed_values=C.SERIES_FIXED):
    """Mittelwerte über feste Netze je Fixkosten-Faktor (Größen aus den Reglern)."""
    kind = "ties" if p.net == "ties" else "map"
    gen = sc.generate_ties if kind == "ties" else sc.generate_map
    out = []
    for pct in fixed_values:
        rows = [_row(gen(p.sites, p.customers, pct, s), p.order) for s in seeds]
        out.append(dict(fixed=pct, **summary(rows)))
    return out


def sizes(p, seeds=C.SIZE_SEEDS, sizes_=C.SIZES):
    out = []
    for m, n in sizes_:
        rows = [_row(sc.generate_map(m, n, p.fixed, s), p.order) for s in seeds]
        out.append(dict(m=m, n=n, **summary(rows)))
    return out


# --- Züge einer Heuristik als Abfolge von Auswahlen (für den Regler "Zug für Zug") ------------------------------------------------

def states(result):
    """Offene Standorte vor dem ersten Zug und nach jedem Zug: Liste von (Auswahl, Kosten)."""
    cur = set(result.start_open)
    cost = result.start_cost
    out = [(tuple(sorted(cur)), cost)]
    for mv in result.trace:
        if mv.opened >= 0:
            cur.add(mv.opened)
        if mv.closed >= 0:
            cur.discard(mv.closed)
        out.append((tuple(sorted(cur)), mv.cost_after))
    return out


def describe_move(inst, mv):
    if mv.kind == "open":
        return f"{inst.site_names[mv.opened]} öffnen"
    if mv.kind == "close":
        return f"{inst.site_names[mv.closed]} schließen"
    return f"{inst.site_names[mv.closed]} schließen, {inst.site_names[mv.opened]} öffnen"
