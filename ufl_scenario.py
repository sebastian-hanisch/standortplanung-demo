"""Szenario: Standortplanung ohne Kapazitätsgrenze (Uncapacitated Facility Location): welche Kandidaten-Standorte werden eröffnet, wenn jeder Kunde vom
nächsten offenen Standort bedient wird?

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über `numpy.random`: numpy garantiert keine über
Versionen stabilen Zufallsströme, die CI installiert aber wöchentlich die neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl
auf Windows und Linux dieselben.

Kosten: `f[i]` sind die Fixkosten des Standorts i, `c[i][j]` die Kosten, Kunde j von i aus zu beliefern (Nachfrage mal Entfernung). Gesamtkosten einer
Auswahl = Fixkosten der offenen Standorte + je Kunde der billigste offene Standort.
"""

from dataclasses import dataclass
from math import isqrt

_MASK = (1 << 64) - 1
MAP_W = 100
FIXED_BASE = 1750        # mittlere Fixkosten bei Fixkosten-Faktor 100 %


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


@dataclass(frozen=True)
class UFL:
    kind: str            # "map" (Karte), "ties" (gleichstandsreich), "teaching" (Lehrnetz von Hand)
    site_names: tuple    # Anzeigename je Kandidat
    cust_names: tuple    # Anzeigename je Kunde
    site_pos: tuple      # ((x, y), ...) oder () ohne Karte
    cust_pos: tuple
    demand: tuple        # Nachfrage je Kunde (leer ohne Karte)
    f: tuple             # Fixkosten je Kandidat
    c: tuple             # c[i][j]

    @property
    def m(self):
        """Anzahl Kandidaten."""
        return len(self.f)

    @property
    def n(self):
        """Anzahl Kunden."""
        return len(self.cust_names)

    @property
    def has_map(self):
        return bool(self.site_pos)


def distance(a, b):
    """Euklidische Entfernung in Zehntel-Einheiten, ganzzahlig (abgerundet)."""
    return isqrt(100 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _names(prefix, count):
    return tuple(f"{prefix} {k + 1}" for k in range(count))


def generate_map(n_sites, n_customers, fixed_pct, seed):
    """Kandidaten und Kunden gleichverteilt auf einer Karte 0..99; Nachfrage 1..9; Fixkosten 60 bis 140 % von FIXED_BASE, skaliert mit `fixed_pct`."""
    rng = SplitMix64(seed)
    sites = tuple((rng.below(MAP_W), rng.below(MAP_W)) for _ in range(n_sites))
    custs = tuple((rng.below(MAP_W), rng.below(MAP_W)) for _ in range(n_customers))
    demand = tuple(1 + rng.below(9) for _ in range(n_customers))
    f = tuple(FIXED_BASE * (60 + rng.below(81)) // 100 * fixed_pct // 100 for _ in range(n_sites))
    c = tuple(tuple(demand[j] * distance(sites[i], custs[j]) for j in range(n_customers)) for i in range(n_sites))
    return UFL("map", _names("Standort", n_sites), _names("Kunde", n_customers), sites, custs, demand, f, c)


def generate_ties(n_sites, n_customers, fixed_pct, seed):
    """Gleichstandsreiche Instanz ohne Karte: Zuordnungskosten 1..5, Fixkosten 3 oder 4 (skaliert mit `fixed_pct`). Viele gleich gute Auswahlen."""
    rng = SplitMix64(seed)
    c = tuple(tuple(1 + rng.below(5) for _ in range(n_customers)) for _ in range(n_sites))
    f = tuple((3 + rng.below(2)) * fixed_pct // 100 for _ in range(n_sites))
    return UFL("ties", _names("Standort", n_sites), _names("Kunde", n_customers), (), (), (), f, c)


def teaching(name):
    """Lehrnetze von Hand: kleine Instanzen, an denen sich ein Effekt nachrechnen lässt."""
    if name not in TEACHING:
        raise KeyError(name)
    sites, custs, f, c = TEACHING[name]
    return UFL("teaching", tuple(sites), tuple(custs), (), (), (), tuple(f), tuple(tuple(r) for r in c))


def _t(m, n, f, c):
    return tuple(f"S{i + 1}" for i in range(m)), tuple(f"K{j + 1}" for j in range(n)), f, c


# Lehrnetze (per Brute-Force-Suche über kleine Zufallsnetze gefunden, hier fest eingetragen; tests/test_heuristics.py rechnet jede Aussage nach).
TEACHING = {
    "add_trap": _t(3, 4, (3, 6, 2), ((7, 2, 6, 2), (2, 1, 3, 6), (4, 5, 3, 7))),
    "drop_trap": _t(3, 4, (5, 4, 2), ((2, 5, 3, 6), (1, 3, 6, 9), (6, 8, 8, 6))),
    "stuck": _t(3, 4, (9, 5, 2), ((4, 5, 2, 1), (5, 7, 1, 3), (3, 9, 9, 2))),
    "both_trap": _t(4, 5, (10, 3, 10, 7), ((2, 2, 9, 7, 3), (7, 8, 4, 3, 8), (1, 5, 8, 3, 7), (7, 3, 8, 3, 1))),
    "lp_frac": _t(3, 4, (6, 5, 8), ((6, 9, 4, 3), (8, 1, 7, 4), (8, 2, 2, 5))),
    "proven": _t(3, 4, (2, 6, 10), ((8, 6, 4, 9), (4, 9, 9, 3), (1, 8, 3, 6))),
}
