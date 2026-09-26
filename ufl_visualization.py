"""Plotly-Abbildungen: Karte mit offenen Standorten und Zuordnung (oder Kostenmatrix bei Netzen ohne Karte), Kostenverlauf der Heuristiken, Schranken in Prozent des Optimums,
Experimente (Fixkosten-Reihe, Verteilung, Größenreihe). Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Karten haben gleichen Maßstab (scaleanchor)
mit automatischem Bereich; der Rand kommt über zwei unsichtbare Punkte (ein fest vorgegebener Bereich wird beim ersten Zeichnen in schmaler Breite eingefroren)."""

import plotly.graph_objects as go

import ufl_constants as C
import ufl_heuristics as hu


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_map(inst, open_set=(), y=None, height=520):
    """Karte: Kunden (Größe = Nachfrage), Standorte (grün = offen, grau = zu, orange = in der LP anteilig geöffnet), Linien von jedem Kunden zu seinem Standort."""
    fig = go.Figure()
    open_set = tuple(sorted(open_set))
    if open_set:
        assign = hu.assignment(inst, open_set)
        xs, ys = [], []
        for j, i in enumerate(assign):
            xs += [inst.cust_pos[j][0], inst.site_pos[i][0], None]
            ys += [inst.cust_pos[j][1], inst.site_pos[i][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLORS["line"], width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[p[0] for p in inst.cust_pos], y=[p[1] for p in inst.cust_pos], mode="markers", name="Kunde",
                             marker=dict(color=C.COLORS["customer"], size=[5 + 1.4 * d for d in inst.demand], opacity=0.75),
                             text=[f"{nm}: Nachfrage {d}" for nm, d in zip(inst.cust_names, inst.demand)], hoverinfo="text"))
    frac = [] if y is None else [i for i in range(inst.m) if 1e-6 < y[i] < 1 - 1e-6]
    closed = [i for i in range(inst.m) if i not in open_set and i not in frac]
    for name, idx, color, size in (("Standort zu", closed, C.COLORS["closed"], 12), ("Standort offen", list(open_set), C.COLORS["open"], 16), ("in der LP anteilig", frac, C.COLORS["frac"], 15)):
        if not idx:
            continue
        fig.add_trace(go.Scatter(x=[inst.site_pos[i][0] for i in idx], y=[inst.site_pos[i][1] for i in idx], mode="markers+text", name=name,
                                 marker=dict(symbol="square", color=color, size=size, line=dict(color="#111111", width=1)),
                                 text=[inst.site_names[i].replace("Standort ", "S") for i in idx], textposition="top center", textfont=dict(size=10),
                                 customdata=[[inst.f[i], "-" if y is None else f"{y[i]:.2f}".replace(".", ",")] for i in idx],
                                 hovertemplate="%{text}: Fixkosten %{customdata[0]}, LP-Wert y = %{customdata[1]}<extra></extra>"))
    pts = list(inst.site_pos) + list(inst.cust_pos)
    xs = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.add_trace(go.Scatter(x=[min(xs) - 6, max(xs) + 6], y=[min(ys_) - 6, max(ys_) + 6], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    return _base(fig, height)


def build_matrix(inst, open_set=(), height=None):
    """Kostenmatrix c_ij für Netze ohne Karte: eine Zeile je Standort (offene Zeilen fett gerahmt), je Kunde ist die billigste offene Zelle umrandet."""
    open_set = set(open_set)
    rows = [f"{inst.site_names[i]}{' · offen' if i in open_set else ''} (Fix {inst.f[i]})" for i in range(inst.m)]
    fig = go.Figure(go.Heatmap(z=[list(r) for r in inst.c], x=list(inst.cust_names), y=rows, text=[[str(v) for v in r] for r in inst.c], texttemplate="%{text}",
                               colorscale="Blues", reversescale=True, showscale=False, hovertemplate="%{y}, %{x}: %{z}<extra></extra>", xgap=2, ygap=2))
    if open_set:
        assign = hu.assignment(inst, sorted(open_set))
        for j, i in enumerate(assign):
            fig.add_shape(type="rect", xref="x", yref="y", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=C.COLORS["open"], width=3))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height or max(220, 60 + 42 * inst.m))


def build_cost_line(states, current, height=380):
    """Gesamtkosten nach jedem Zug; der aktuelle Zug ist markiert. `states`: [(Auswahl, Kosten)] (Kosten None vor dem ersten Zug von Add)."""
    xs = [k for k, (_s, c) in enumerate(states) if c is not None]
    ys = [c for _s, c in states if c is not None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C.COLORS["interchange"], width=2), name="Gesamtkosten"))
    if 0 <= current < len(states) and states[current][1] is not None:
        fig.add_trace(go.Scatter(x=[current], y=[states[current][1]], mode="markers", marker=dict(color=C.COLORS["drop"], size=14, symbol="diamond"), name="aktueller Zug"))
    fig.update_xaxes(title="Zug", dtick=1, range=[-0.5, len(states) - 0.5])
    fig.update_yaxes(title="Gesamtkosten")
    return _base(fig, height)


def build_bounds(rows, height=380):
    """Schranken und Lösungen in Prozent des Optimums (Linie bei 100)."""
    colors = {"untere Schranke": C.COLORS["strong"], "Optimum": C.COLORS["opt"], "Lösung": C.COLORS["add"]}
    labels = [r[0] for r in rows][::-1]
    vals = [r[2] for r in rows][::-1]
    cols = [colors[r[3]] for r in rows][::-1]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h", marker=dict(color=cols), text=[f"{v:.1f} %".replace(".", ",") for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f} % des Optimums<extra></extra>"))
    lo = min(vals)
    fig.update_xaxes(range=[max(0, lo - 10), max(vals) + 12], title="Prozent des Optimums (100 = Optimum)")
    fig.add_vline(x=100, line=dict(color=C.COLORS["opt"], dash="dash"))
    return _base(fig, height)


def build_series(series, title_x="Fixkosten-Faktor [%]", height=420):
    """Untere Schranken gegen den Fixkosten-Faktor (Mittel über feste Netze)."""
    xs = [r["fixed"] for r in series]
    fig = go.Figure()
    for key, name in (("strong", "starke LP"), ("dual_adj", "Dual Ascent mit Anpassung"), ("dual", "Dual Ascent"), ("weak", "schwache LP")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in series], mode="lines+markers", name=name, line=dict(color=C.COLORS[key], width=2)))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title="Schranke in % des Optimums (Mittel)", range=[30, 102])
    return _base(fig, height)


def build_gaps(series, height=380):
    """Abstand der Heuristiken zum Optimum gegen den Fixkosten-Faktor (Mittel über feste Netze)."""
    xs = [r["fixed"] for r in series]
    fig = go.Figure()
    for key in ("add", "drop", "interchange"):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in series], mode="lines+markers", name=C.LABELS[key], line=dict(color=C.COLORS[key], width=2)))
    fig.update_xaxes(title="Fixkosten-Faktor [%]")
    fig.update_yaxes(title="Abstand zum Optimum [%] (Mittel)")
    return _base(fig, height)


def build_dist(dist, height=420):
    """Verteilung über feste Netze: Kästen der unteren Schranken (Prozent des Optimums) und der Heuristik-Abstände."""
    rows = dist["rows"]
    fig = go.Figure()
    for key, name, color in (("weak", "schwache LP", C.COLORS["weak"]), ("strong", "starke LP", C.COLORS["strong"]), ("dual_adj", "Dual Ascent mit Anpassung", C.COLORS["dual_adj"])):
        fig.add_trace(go.Box(y=[r[key] for r in rows], name=name, marker_color=color, boxpoints="all", jitter=0.4, pointpos=0))
    fig.update_yaxes(title="Schranke in % des Optimums")
    return _base(fig, height)


def build_dist_gaps(dist, height=380):
    rows = dist["rows"]
    fig = go.Figure()
    for key in ("add", "drop", "interchange"):
        fig.add_trace(go.Box(y=[r[key] for r in rows], name=C.LABELS[key], marker_color=C.COLORS[key], boxpoints="all", jitter=0.4, pointpos=0))
    fig.update_yaxes(title="Abstand zum Optimum [%]")
    return _base(fig, height)


def build_sizes(rows, height=380):
    """Bewertungen der drei Heuristiken gegen die Netzgröße."""
    labels = [f"{r['m']} × {r['n']}" for r in rows]
    fig = go.Figure()
    for key in ("add", "drop", "interchange"):
        fig.add_trace(go.Bar(x=labels, y=[r[f"{key}_evals"] for r in rows], name=C.LABELS[key], marker_color=C.COLORS[key]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Kandidaten × Kunden")
    fig.update_yaxes(title="Kostenbewertungen (Mittel)")
    return _base(fig, height)
