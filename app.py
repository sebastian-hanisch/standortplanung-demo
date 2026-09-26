"""Standortplanung - warum ist die Schranke hier fast exakt? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Modell - die Standortplanung ohne Kapazitätsgrenze (Uncapacitated Facility Location) -
und lässt stattdessen das Beispiel wachsen.
Neues Stück (Wurzel der Standortplanungs-Linie) der "Konzepte"-Reihe: dieselbe Fixkosten-Entscheidung wie im Fixkosten-Netzdesign (Stück 10 der Netzwerkfluss-Linie), aber mit fast exakter Schranke.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import ufl_constants as C
import ufl_dual as dl
import ufl_evaluation as ev
from ufl_exact import fractional_sites
import ufl_heuristics as hu
from ufl_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from ufl_visualization import build_bounds, build_cost_line, build_dist, build_dist_gaps, build_gaps, build_map, build_matrix, build_series, build_sizes

st.set_page_config(page_title="Standortplanung – Sebastian Hanisch", layout="wide")


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return "–" if x is None else f"{int(round(x)):,}".replace(",", " ")


def _val(x):
    """Ganzzahl ohne Nachkommastellen, LP-Werte mit zwei."""
    return _int(x) if abs(x - round(x)) < 1e-9 else _f(x, 2)


def _pl(k, one, many):
    return f"{k} {one if k == 1 else many}"


def _pct(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


@st.cache_resource(show_spinner=False, max_entries=32)
def _analysis(p):
    return ev.analyse(p)


@st.cache_resource(show_spinner=False, max_entries=8)
def _series(p):
    return ev.fixed_series(p)


@st.cache_resource(show_spinner=False, max_entries=8)
def _dist(p):
    return ev.distribution(p)


@st.cache_resource(show_spinner=False, max_entries=8)
def _sizes(p):
    return ev.sizes(p)


st.title("📍 Standortplanung – warum ist die Schranke hier fast exakt?")
st.markdown(
    """
Welche von mehreren Kandidaten-Standorten soll man eröffnen, wenn jeder Kunde vom nächsten offenen Standort beliefert wird? Jeder Standort kostet **Fixkosten**, jede Belieferung kostet Nachfrage mal Entfernung; ohne Kapazitätsgrenze der Standorte
heißt das Modell **Uncapacitated Facility Location (UFL)**. Es ist dieselbe Ja/Nein-Entscheidung wie beim Fixkosten-Netzdesign (Stück 10 der Netzwerkfluss-Linie), und dort war die LP-Schranke schwach (etwa 70 % des Optimums).
Hier ist sie **fast exakt**: die LP-Relaxation liefert meist schon eine ganzzahlige Lösung. Die Demo zeigt, warum das so ist, welche einfachen Heuristiken (**Add**, **Drop**, **Interchange**) wie nah herankommen, wie **Dual Ascent** (Erlenkotter) eine Schranke ganz ohne LP-Löser liefert
und wann UFL trotzdem schwer wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - die Wurzel der Standortplanungs-Linie der \"Konzepte\"-Reihe - **ein** Modell an einem wachsenden Beispiel. "
    "Verwandt: das Fixkosten-Netzdesign (schwache Schranke), die k-Means-Demo (Depotwahl, aber ohne Fixkosten) und die Rettungsdienst-Demo (Standorte mit Verfügbarkeit)."
)

with st.expander("So funktioniert die Standortplanung", expanded=True):
    st.markdown(
        r"""
1. **Modell:** je Kandidat $i$ eine Ja/Nein-Entscheidung $y_i$ (Fixkosten $f_i$), je Kunde $j$ der Standort $x_{ij}$, der ihn beliefert (Kosten $c_{ij}$). Ohne Kapazität nimmt jeder Kunde den billigsten offenen Standort; die Frage ist nur, **welche** Standorte offen sind.
2. **Heuristiken:** *Add* öffnet nacheinander den Standort, der die Gesamtkosten am stärksten senkt; *Drop* startet mit allen Standorten und schließt nacheinander; *Interchange* öffnet, schließt oder tauscht, solange das hilft. Alle drei halten in einem lokalen Optimum an.
3. **Schranke von oben und unten:** die **starke LP** ($x_{ij}\le y_i$) und die **schwache LP** ($\sum_j x_{ij}\le n\,y_i$) sind untere Schranken; jede zulässige Auswahl ist eine obere. Liegen beide dicht beieinander, ist die Lösung fast bewiesen.
4. **Dual Ascent:** die Kundenwerte $v_j$ werden angehoben, bis ein Standort „voll bezahlt“ ist; die Summe der Werte ist eine untere Schranke, ganz ohne LP-Löser. Die knappen Standorte liefern zugleich eine Auswahl; ist ihr Wert gleich der Schranke, ist sie **bewiesen** optimal. Ein Anpassungsschritt (Kundenwert senken, alle anderen erneut anheben) hebt die Schranke stark.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox("Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
                           help="Ein Zufallsnetz mit Karte, ein Gleichstandsnetz mit vielen gleich guten Auswahlen (dort wird UFL schwer) oder ein kleines Lehrnetz, an dem sich ein Effekt von Hand nachrechnen lässt.")
    order = st.radio("Reihenfolge der Kunden im Dual Ascent", list(dl.ORDERS), key="order_radio", format_func=lambda k: dl.ORDERS[k],
                     help="Der Dual Ascent hebt die Kundenwerte nacheinander an; die Reihenfolge ändert die Schranke ein wenig (ohne Anpassungsschritt stärker als mit).")
    if net_key in ("map", "ties"):
        seed_widget("sites_slider")
        sites = st.slider("Kandidaten-Standorte", *bounds("sites_slider"), key="sites_slider")
        st.session_state[KEPT["sites_slider"]] = sites
        seed_widget("customers_slider")
        customers = st.slider("Kunden", *bounds("customers_slider"), key="customers_slider")
        st.session_state[KEPT["customers_slider"]] = customers
        seed_widget("fixed_slider")
        fixed = st.slider("Fixkosten-Faktor [%]", *bounds("fixed_slider"), key="fixed_slider", step=C.FIXED_STEP,
                          help="Skaliert die Fixkosten aller Standorte. Hohe Fixkosten: wenige offene Standorte; niedrige: viele. Die schwache LP-Schranke fällt mit den Fixkosten, die starke bleibt bei fast 100 %.")
        st.session_state[KEPT["fixed_slider"]] = fixed
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed (neue Positionen, Nachfragen und Fixkosten). Die Verteilungen über feste Netze weiter unten ändern sich dabei nicht.")
    else:
        sites = int(st.session_state.get(KEPT["sites_slider"], C.DEFAULT_SITES))
        customers = int(st.session_state.get(KEPT["customers_slider"], C.DEFAULT_CUSTOMERS))
        fixed = int(st.session_state.get(KEPT["fixed_slider"], C.DEFAULT_FIXED))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Kandidaten, Kunden, Fixkosten-Faktor und Seed gehören zu den Zufallsnetzen.")

sync_query_params({"net_select": net_key, "order_radio": order, "sites_slider": int(sites), "customers_slider": int(customers), "fixed_slider": int(fixed), "seed_input": int(seed),
                   "heuristic_radio": st.session_state.get("heuristic_radio", C.DEFAULT_HEURISTIC)})

params = ev.canonical(ev.Params(net_key, int(sites), int(customers), int(fixed), int(seed), order))
with st.spinner("Rechne..."):
    a = _analysis(params)
inst, opt, opt_set = a["inst"], a["opt"], a["opt_set"]
arr = hu.arrays(inst)


def _view(open_set, key, y=None, height=480):
    if inst.has_map:
        st.plotly_chart(build_map(inst, open_set, y, height=height), width="stretch", key=key)
    else:
        st.plotly_chart(build_matrix(inst, open_set), width="stretch", key=key)


st.markdown(
    f"Das Netz hat **{inst.m} Kandidaten** und **{inst.n} Kunden**. Das Optimum kostet **{_int(opt)}** und öffnet **{_pl(len(opt_set), 'Standort', 'Standorte')}** "
    f"({', '.join(inst.site_names[i].replace('Standort ', 'S') for i in opt_set)})."
)

# --- Selbst probieren ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Selbst probieren: welche Standorte öffnen?")
if st.session_state.get("ufl_open_owner") != params:
    st.session_state["open_multi"] = sorted(range(inst.m), key=lambda i: (inst.f[i], i))[:min(3, inst.m)]
    st.session_state["ufl_open_owner"] = params
chosen = st.multiselect("Offene Standorte", list(range(inst.m)), key="open_multi", format_func=lambda i: f"{inst.site_names[i]} (Fixkosten {_int(inst.f[i])})",
                        help="Voreingestellt sind die billigsten Standorte nach Fixkosten. Jeder Kunde wird vom nächsten offenen Standort beliefert.")
cl, cr = st.columns([3, 2])
with cl:
    _view(chosen, key="pick_view")
with cr:
    if chosen:
        fixed_cost, assign_cost = hu.cost_split(inst, chosen)
        total = fixed_cost + assign_cost
        m1, m2 = st.columns(2)
        m1.metric("Gesamtkosten", _int(total), delta=f"{_pct(ev.gap_pct(total, opt))} über dem Optimum" if total > opt else "Optimum", delta_color="off" if total == opt else "inverse")
        m2.metric("Optimum", _int(opt))
        m3, m4 = st.columns(2)
        m3.metric("Fixkosten", _int(fixed_cost), help="Summe der Fixkosten der offenen Standorte.")
        m4.metric("Belieferung", _int(assign_cost), help="Summe der Kosten der Kunden zu ihrem nächsten offenen Standort.")
        st.caption(f"{len(chosen)} von {inst.m} Standorten offen, im Optimum {len(opt_set)}. Mehr Standorte senken die Belieferung und erhöhen die Fixkosten.")
    else:
        st.info("Wählen Sie mindestens einen Standort: ohne offenen Standort kann niemand beliefert werden.")
st.caption("Grün: offener Standort; graue Quadrate sind geschlossen; die Linien führen von jedem Kunden (blau, Größe = Nachfrage) zu seinem Standort. Bei Netzen ohne Karte zeigt die Matrix die Kosten je Standort und Kunde, die billigste offene Zelle je Kunde ist grün umrandet (dunkel = billig).")

st.markdown("---")

# --- Heuristiken Zug für Zug ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zug für Zug: Add, Drop, Interchange")
heur = st.radio("Heuristik", list(C.HEURISTICS), key="heuristic_radio", format_func=lambda k: C.HEURISTICS[k], horizontal=True,
                help="Add öffnet nacheinander, Drop schließt nacheinander, Interchange öffnet, schließt oder tauscht (Start: das Ergebnis von Add). Jeder Zug ist der beste Nachbar; die Heuristik hält, wenn kein Zug die Kosten senkt.")
res = a[heur]
sts = ev.states(res)
N = len(res.trace)
owner = (params, heur)
if st.session_state.get("ufl_step_owner") != owner:
    st.session_state["ufl_step"] = N
    st.session_state["ufl_step_owner"] = owner
view_slot = st.empty()
if N > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Zug", 0, N, key="ufl_step", help="0 ist der Start (bei Add: nichts offen; bei Drop: alles offen; bei Interchange: das Ergebnis von Add).")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
elif N == 1:
    step, auto_play = 1, False
else:
    step, auto_play = 0, False
    st.info("Kein Zug senkt die Kosten: schon die Startauswahl ist ein lokales Optimum.")


def _render(k):
    with view_slot.container():
        open_k, cost_k = sts[k]
        c1, c2 = st.columns([3, 2])
        with c1:
            _view(open_k, key=f"heur_view_{heur}_{k}", height=460)
        with c2:
            st.plotly_chart(build_cost_line(sts, k, height=380), width="stretch", key=f"cost_{heur}_{k}")
        if k >= 1:
            mv = res.trace[k - 1]
            st.markdown(f"**Zug {k}:** {ev.describe_move(inst, mv)}")
            m2, m3 = st.columns(2)
            m2.metric("Kosten danach", _int(mv.cost_after), delta=None if mv.cost_before is None else f"{_int(mv.cost_after - mv.cost_before)}", delta_color="off")
            m3.metric("Bewertete Nachbarn", _int(mv.evals), help="Wie viele Auswahlen in diesem Zug bewertet wurden, um den besten Zug zu finden.")
        else:
            st.caption("Startauswahl.")


if N > 0:
    if auto_play:
        for k in range(0, N + 1):
            _render(k)
            time.sleep(min(0.5, 6.0 / max(N, 1)))
        step = N
    else:
        _render(step)
else:
    _render(0)
gap = ev.gap_pct(res.cost, opt)
if res.cost == opt:
    st.success(f"✅ {C.HEURISTICS[heur].split(' (')[0]} findet nach {_pl(N, 'Zug', 'Zügen')} das Optimum ({_int(opt)}); {_int(res.evals)} bewertete Nachbarn.")
else:
    st.warning(f"{C.HEURISTICS[heur].split(' (')[0]} hält nach {_pl(N, 'Zug', 'Zügen')} bei {_int(res.cost)}: {_pct(gap)} über dem Optimum ({_int(opt)}). Der bessere Zug wäre ein Tausch oder ein Zug über mehrere Standorte hinweg, den diese Heuristik nicht sieht.")

st.markdown("---")

# --- Schranken -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Schranke: wie nah kommt man an den Beweis?")
rows = ev.bound_rows(a)
st.plotly_chart(build_bounds(rows), width="stretch", key="bounds_chart")
st.table({"Wert": [r[0] for r in rows], "Kosten / Schranke": [_val(r[1]) for r in rows], "% des Optimums": [_f(r[2], 2) for r in rows]})
frac_s = fractional_sites(a["y_strong"])
frac_w = fractional_sites(a["y_weak"])
if frac_s:
    st.info(f"Die starke LP öffnet {len(frac_s)} Standorte nur anteilig ({', '.join(inst.site_names[i] for i in frac_s)}): hier reicht ihre Lösung nicht als Antwort, sie liefert nur die Schranke {_pct(100 * a['lp_strong'] / opt, 2)}.")
else:
    st.success(f"✅ Die starke LP ist ganzzahlig: ihre Lösung ist das Optimum ({_pct(100 * a['lp_strong'] / opt, 2)}). Die schwache LP öffnet dagegen {len(frac_w)} von {inst.m} Standorten anteilig und erreicht nur {_pct(100 * a['lp_weak'] / opt)}.")
lp_choice = st.radio("LP-Lösung anzeigen", ["starke", "schwache"], horizontal=True, key="lp_view", help="Der Wert y_i einer LP-Lösung sagt, zu welchem Bruchteil ein Standort offen ist. Orange: anteilig geöffnet, grün: ganz offen.")
y_show = a["y_strong"] if lp_choice == "starke" else a["y_weak"]
lp_open = tuple(i for i, v in enumerate(y_show) if v > 1 - 1e-6)
if inst.has_map:
    _view(lp_open, key=f"lp_view_{lp_choice}", y=y_show, height=460)
else:
    st.table({"Standort": list(inst.site_names), "y (LP)": [_f(v, 2) for v in y_show]})
st.caption(
    "Die **schwache LP** lässt einen Standort so weit offen sein, wie er Kunden beliefert (Σ x ≤ n·y): ein Standort, der wenige Kunden beliefert, zahlt nur einen kleinen Bruchteil seiner Fixkosten - die Schranke liegt tief. "
    "Die **starke LP** (x ≤ y je Kunde) verlangt für jeden Kunden den ganzen Standort: die Fixkosten werden fast voll bezahlt, und die Lösung ist meist ganzzahlig. Im Fixkosten-Netzdesign wirkt die starke Kopplung kaum, weil dort Kapazitäten geteilt werden."
)

dual = a["dual_adj"]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Dual Ascent: Schranke", _int(dual.lower_bound), delta=f"ohne Anpassung {_int(a['dual'].lower_bound)}", delta_color="off", help="Summe der Kundenwerte nach Aufstieg und Anpassung; ohne LP-Löser berechnet.")
d2.metric("Auswahl aus den knappen Standorten", _int(dual.primal_cost), delta=f"Lücke {_int(dual.gap)}", delta_color="off", help="Kosten der Auswahl, die die Komplementarität aus den knappen Standorten liest. Gleich der Schranke heißt: bewiesen optimal.")
d3.metric("Bewiesen optimal?", "ja" if dual.proven else "nein", help="Ja, wenn die Kosten der Auswahl gleich der Schranke sind.")
d4.metric("Kantenoperationen", _int(dual.ops), help="Geprüfte Kunde-Standort-Paare in Aufstieg und Anpassung.")
with st.expander("Dual Ascent Schritt für Schritt"):
    st.table({"Kunde": [inst.cust_names[s.customer] for s in dual.steps], "Wert vorher": [_int(s.v_before) for s in dual.steps],
              "Wert nach dem Aufstieg": [_int(s.v_after) for s in dual.steps], "blockiert durch": [inst.site_names[s.blocked_by] if s.blocked_by >= 0 else "-" for s in dual.steps],
              "dabei knapp geworden": [", ".join(inst.site_names[i] for i in s.became_tight) or "-" for s in dual.steps]})
    st.caption(f"Reihenfolge: {dl.ORDERS[dual.order]}. Die Tabelle zeigt den Aufstieg vor der Anpassung ({dual.passes} Anpassungsdurchläufe verbessern die Summe danach noch). Knappe Standorte in der Reihenfolge des Knappwerdens: {', '.join(inst.site_names[i] for i in dual.tight) or '-'}.")

st.markdown("---")

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wovon hängt die Schranke ab?")
st.caption("Untere Schranken und Heuristik-Abstände gegen den Fixkosten-Faktor (Mittel über 20 feste Netze der gewählten Größe; Karten- oder Gleichstandsnetz je nach Auswahl).")
if st.button("Fixkosten-Reihe durchrechnen (dauert einige Sekunden)", key="series_start"):
    st.session_state["series_on"] = True
if st.session_state.get("series_on"):
    with st.spinner("Rechne 5 Fixkosten-Faktoren × 20 Netze..."):
        series = _series(params)
    st.plotly_chart(build_series(series), width="stretch", key="series_chart")
    st.plotly_chart(build_gaps(series), width="stretch", key="gaps_chart")
    st.table({"Fixkosten-Faktor": [f"{r['fixed']} %" for r in series], "offene Standorte im Optimum": [_f(r["n_open"], 1) for r in series],
              "LP-Wert = Optimum in": [f"{r['tight']} von {r['count']} Netzen" for r in series],
              "schwache / starke LP / Dual Ascent [%]": [f"{_f(r['weak'], 1)} / {_f(r['strong'], 2)} / {_f(r['dual_adj'], 1)}" for r in series]})
    st.caption("Mit steigenden Fixkosten werden weniger Standorte geöffnet; die schwache Schranke fällt, die starke bleibt praktisch bei 100 %. Add wird mit den Fixkosten schlechter, weil es früh zu viele Standorte öffnet, die es später nicht mehr schließt.")

st.subheader("🔬 Gilt das in jedem Netz?")
st.caption("40 feste Netze mit den gewählten Größen: Schranken, Ganzzahligkeit der starken LP, Abstände der Heuristiken, Dual Ascent.")
if st.button("40 Netze durchrechnen (dauert einige Sekunden)", key="dist_start"):
    st.session_state["dist_on"] = True
if st.session_state.get("dist_on"):
    with st.spinner("Rechne 40 Netze..."):
        dist = _dist(params)
    s = dist["summary"]
    st.plotly_chart(build_dist(dist), width="stretch", key="dist_chart")
    st.plotly_chart(build_dist_gaps(dist), width="stretch", key="dist_gaps_chart")
    hl = ("add", "drop", "interchange")
    st.table({"Heuristik": [C.LABELS[k] for k in hl], "Abstand Mittel / Median / schlechtestes Netz [%]": [f"{_f(s[k], 2)} / {_f(s[k + '_median'], 2)} / {_f(s[k + '_max'], 2)}" for k in hl],
              "exakt in": [f"{s[k + '_exact']} von {s['count']} Netzen" for k in hl], "bewertete Nachbarn (Mittel)": [_int(s[k + "_evals"]) for k in hl]})
    st.caption(f"Der Wert der starken LP ist in {s['tight']} von {s['count']} Netzen gleich dem Optimum und erreicht im Mittel {_pct(s['strong'], 2)} (schlechtestes Netz {_pct(s['strong_min'], 2)}); die schwache LP {_pct(s['weak'])}. "
               f"Dual Ascent erreicht mit Anpassung {_pct(s['dual_adj'])} (ohne {_pct(s['dual'])}); die Auswahl aus den knappen Standorten liegt im Mittel {_pct(s['dual_primal'])} über dem Optimum, bewiesen optimal in {s['proven']} Netzen, "
               f"mit anschließendem Interchange {_pct(s['polished'], 2)}.")

st.subheader("🔬 Wie wächst der Aufwand mit dem Netz?")
st.caption("Bewertete Nachbarn der Heuristiken bei wachsender Größe (Mittel über 5 feste Kartennetze je Größe, Fixkosten-Faktor wie eingestellt).")
if st.button("Größen durchrechnen (dauert einige Sekunden)", key="sizes_start"):
    st.session_state["sizes_on"] = True
if st.session_state.get("sizes_on"):
    with st.spinner("Rechne 4 Größen × 5 Netze..."):
        srows = _sizes(params)
    st.plotly_chart(build_sizes(srows), width="stretch", key="sizes_chart")
    st.table({"Kandidaten × Kunden": [f"{r['m']} × {r['count']}" for r in srows], "offene im Optimum": [_f(r["n_open"], 1) for r in srows],
              "Add / Drop / Interchange Abstand [%]": [f"{_f(r['add'], 2)} / {_f(r['drop'], 2)} / {_f(r['interchange'], 2)}" for r in srows],
              "LP-Wert = Optimum in": [f"{r['tight']} von {r['count']} Netzen" for r in srows]})
    st.caption("Die Zahl der Bewertungen wächst mit der Zahl der Kandidaten (jeder Zug bewertet alle Nachbarn), bei Interchange stärker, weil es Tauschzüge über alle Paare prüft. Der Abstand zum Optimum bleibt klein.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer ansetzt |
|---|---|
| **Keine Kapazitätsgrenze** | Mit Kapazitäten je Standort (CFLP) und Single-Sourcing ist die LP-Schranke wieder schwach; die Lagrange-Relaxation ist das nächste Stück der Linie. |
| **Kosten linear in Entfernung und Nachfrage** | Realistisch sind Staffeln, Straßennetze und Lieferzeiten; hier ist $c_{ij}$ Nachfrage mal Luftlinie in Zehntel-Einheiten, ganzzahlig. |
| **Eine Ebene** | Zwei Ebenen (Werke → Verteilzentren → Filialen) und Routing (Location-Routing) sind nicht gebaut; sie überlappen mit der Tourenplanung. |
| **Feste Anzahl statt Fixkosten (p-Median)** | Wird die Zahl der offenen Standorte vorgegeben und sind die Fixkosten 0, ist das p-Median (k-Medoids mit Entfernungen); dafür gibt es kein eigenes Stück: die k-Means-Demo erklärt k-Medoids nur, die p-Center-Demo rechnet das p-Median exakt als Vergleich. |
| **Dual Ascent ohne volle DUALOC-Anpassung** | Die Anpassung ist vereinfacht (Kundenwert auf den niedrigsten knappen Standort senken, alle anderen erneut anheben, behalten, wenn die Summe steigt); Erlenkotters Original ist feiner und liefert oft eine noch bessere Schranke. |
| **Erzeugte Netze** | Gleichverteilte Standorte und Kunden auf einer Karte, keine Fremddaten; die Gleichstandsnetze sind eine Konstruktion, kein Praxisfall. |
"""
)
st.caption("Die Standortplanungs-Linie ist als Ganzes geplant: diese Demo als Wurzel, danach die kapazitierte Standortplanung mit Lagrange-Relaxation und die Hub-Standortplanung (p-Hub-Median).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell (stark).** Kandidaten $I$, Kunden $J$, Fixkosten $f_i$, Zuordnungskosten $c_{ij}$:
$$\min\sum_i f_iy_i+\sum_{i,j}c_{ij}x_{ij}\quad\text{u.d.N.}\quad\sum_i x_{ij}=1\ \ \forall j,\quad x_{ij}\le y_i\ \ \forall i,j,\quad y_i\in\{0,1\},\ x_{ij}\ge0.$$
Bei ganzzahligem $y$ ist $x$ von selbst ganzzahlig: jeder Kunde nimmt den billigsten offenen Standort. **Schwach:** $\sum_j x_{ij}\le |J|\,y_i$ statt $x_{ij}\le y_i$; die LP-Lösung setzt $y_i=\frac1{|J|}\sum_jx_{ij}$ und zahlt nur den gelieferten Bruchteil der Fixkosten.

**Dual.** Für die starke LP ergibt sich (nach Elimination der Multiplikatoren von $x_{ij}\le y_i$):
$$\max\sum_j v_j\quad\text{u.d.N.}\quad\sum_j\max(0,v_j-c_{ij})\le f_i\ \ \forall i.$$
**Dual Ascent** (Erlenkotter 1978): Start $v_j=\min_ic_{ij}$; je Kunde $v_j$ anheben bis zum nächsten Knickpunkt $c_{ij}$ oder bis ein Standort knapp wird ($s_i=f_i-\sum_j\max(0,v_j-c_{ij})=0$); ein einmal blockierter Kunde bleibt blockiert (die Restbeträge sinken nur), ein Durchlauf genügt. $\sum_jv_j$ ist eine untere Schranke.
**Komplementarität:** öffnet man knappe Standorte so, dass kein Kunde zu zwei offenen mit $v_j>c_{ij}$ beiträgt, sind die Kosten gleich $\sum_jv_j$ und die Lösung optimal. **Anpassung** (hier vereinfacht): $v_j$ auf den niedrigsten knappen Standort senken, alle anderen erneut anheben, behalten, wenn die Summe steigt.

**Add / Drop / Interchange** sind Nachbarschaftssuchen in $\{0,1\}^{|I|}$ mit den Zügen öffnen, schließen, tauschen; sie halten in einem lokalen Optimum. UFL ist NP-schwer (Reduktion von Set Cover), eine 1,488-Näherung ist bekannt (Li 2013); die Schranken oben gelten für die hier gemessenen Netze.

Implementiert in `ufl_scenario.py` (Netze, Lehrnetze, Zufallsgenerator), `ufl_heuristics.py` (Add, Drop, Interchange), `ufl_dual.py` (Dual Ascent, Anpassung, primale Auswahl), `ufl_exact.py` (HiGHS: Optimum, starke und schwache LP), `ufl_evaluation.py` (Vergleiche, Reihen, Verteilungen).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
