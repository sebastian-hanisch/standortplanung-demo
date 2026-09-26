# Standortplanung – warum ist die Schranke hier fast exakt? – Streamlit-Demo

*(noch nicht deployed)*

Wurzel der **Standortplanungs-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" (Erweiterung E3 der Netzwerkfluss-Planung):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Modell – die **Standortplanung ohne Kapazitätsgrenze** (Uncapacitated Facility Location, UFL) – an einem wachsenden Beispiel.
Welche von mehreren Kandidaten-Standorten soll man eröffnen, wenn jeder Kunde vom nächsten offenen Standort beliefert wird? Jeder Standort kostet Fixkosten, jede Belieferung Nachfrage mal Entfernung.

Es ist dieselbe Ja/Nein-Entscheidung wie im [fixkosten-netzdesign-demo](https://github.com/sebastian-hanisch/fixkosten-netzdesign-demo) (Stück 10 der Netzwerkfluss-Linie), und dort war die LP-Schranke **schwach** (im Mittel 70,7 % des Optimums). Hier ist sie **fast exakt**: die LP-Relaxation trifft das Optimum in 38 von 40 Kartennetzen.
Die Demo zeigt, **warum** (die Kopplung x ≤ y je Kunde statt einer Summe), wie nah die einfachen Heuristiken **Add**, **Drop** und **Interchange** kommen, wie **Dual Ascent** (Erlenkotter 1978) eine Schranke ganz ohne LP-Löser liefert – und **wann** UFL trotzdem schwer wird (Gleichstandsnetze).

**Einordnung in die Reihe (die Kanten des Graphen):** Kontrast zu Stück 10 (gleiche Fixkosten-Entscheidung, dort teilen sich Güter Kapazitäten und die Schranke ist schwach, hier nicht); Verwandte ohne eigenes Stück: [kmeans-demo](https://github.com/sebastian-hanisch/kmeans-demo) (Depotwahl, aber ohne Fixkosten: k-Means; wird die Zahl der Standorte vorgegeben und sind die Fixkosten 0, ist das p-Median = k-Medoids mit Entfernungen, dafür gibt es hier kein eigenes Stück),
`ems_demo` (Rettungsdienst-Standorte mit Verfügbarkeit, Fall-Demo) und [projektauswahl-demo](https://github.com/sebastian-hanisch/projektauswahl-demo) (dort die Negativkontrolle „eine von mehreren Anlagen genügt“ – genau das ist UFL).
```
fixkosten-netzdesign-demo (Stück 10: schwache Schranke, geteilte Kapazität)             [gebaut, Kontrast]
standortplanung-demo (unkapazitiert: starke Kopplung, Schranke fast exakt)               [dieses Stück]
  ├─ kapazitierte Standortplanung + Lagrange-Relaxation                                  [geplant]
  └─ p-Hub-Median (Hub-Standortplanung)                                                  [geplant]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Lehrnetze von Hand, Beispielnetze über ihre Seeds, Verteilungen über feste Netze (Seeds ab 100000: 40 Netze für die Verteilung, 20 je Wert der Fixkosten-Reihe, 5 je Größe). Standard: 25 Kandidaten, 50 Kunden, Fixkosten-Faktor 100 %, Seed 1.
Kosten der Heuristiken und des Dual Ascent sind ganzzahlig und auf allen Plattformen dieselben (eigener Zufallsstrom, reine Ganzzahl-Arithmetik); Werte der LP sind Gleitkommazahlen (Vergleich mit Toleranz), und gezählt wird nur, was der LP-**Wert** hergibt („gleich dem Optimum“), nie, welche Ecke der Löser bei mehreren gleich guten Lösungen wählt. Aufwand zählt in bewerteten Nachbarn bzw. Kantenoperationen, nie in Sekunden.

**Standardnetz.** Das Optimum kostet **39 967** und öffnet **8** Standorte. Starke LP **100,0 %**, schwache LP **63,6 %**. Add endet bei 41 872 (**4,8 %** darüber, 9 Züge, 205 bewertete Nachbarn), Drop (17 Züge, 297) und Interchange (Start = Add, 2 Züge, 491) treffen das Optimum. Dual Ascent erreicht 34 252 (**85,7 %**), mit Anpassung 39 744 (**99,4 %**); die Auswahl aus den knappen Standorten kostet 41 464 (Lücke 1 720, nicht bewiesen), mit anschließendem Interchange 39 967.
Die Reihenfolge der Kunden im Dual Ascent ändert die Schranke (Aufstieg / mit Anpassung): in Reihenfolge 34 252 / 39 744, rückwärts 37 253 / 39 944, nächste Kunden zuerst 32 870 / 33 924, entfernteste zuerst 37 781 / 39 568.

**Die starke Schranke ist fast exakt, die schwache nicht.** 40 Kartennetze (25 × 50, Fixkosten 100 %): starke LP im Mittel **99,99 %** des Optimums (schlechtestes Netz 99,65 %), der LP-Wert ist in **38 von 40** Netzen gleich dem Optimum; die **schwache LP** (Σ_j x_ij ≤ n·y_i) im Mittel **61,0 %** (schlechtestes Netz 53,7 %, bestes 69,3 %). Das Optimum öffnet 7 bis 10 Standorte (Mittel 8,7).
Warum: die starke Formulierung verlangt für jeden Kunden den *ganzen* Standort (x_ij ≤ y_i), die Fixkosten werden fast voll bezahlt; die schwache lässt einen Standort so weit offen sein, wie er Kunden beliefert, und zahlt nur diesen Bruchteil – in Stück 10 wirkt die starke Kopplung kaum, weil dort Kapazitäten geteilt werden.
Über die Fixkosten (Faktor 25 / 50 / 100 / 200 / 400 %, 20 Netze je Wert): offene Standorte im Mittel 13,55 / 11,35 / 8,65 / 6,5 / 4,4; **starke LP 100,0 %** (LP-Wert = Optimum in 20 / 20 / 19 / 19 / 19 von 20 Netzen); **schwache LP 81,1 / 71,4 / 61,1 / 52,1 / 46,1 %** – sie fällt mit den Fixkosten, genau wie die schwache Schranke in Stück 10.

**Heuristiken.** Abstand zum Optimum über die 40 Kartennetze: **Add** im Mittel 1,87 % (Median 1,01 %, schlechtestes Netz 7,96 %, exakt in 6 Netzen), **Drop** 0,11 % (Median 0,00 %, schlechtestes 0,84 %, exakt in 27), **Interchange** 0,11 % (0,00 %; 1,37 %; exakt in 32). Drop ist in **32** Netzen besser als Add und in einem schlechter. Über die Fixkosten wird Add schlechter (0,58 / 1,09 / 1,58 / 2,28 / 3,85 %), Drop bleibt näher am Optimum (0,02 / 0,10 / 0,11 / 0,55 / 0,99 %), Interchange auch (0,02 / 0,00 / 0,18 / 0,22 / 0,31 %).
Auf den Karten heißt das: die Heuristiken sind hier fast so gut wie die Exakte – die Schranke wird gebraucht, um das zu **wissen**.

**Dual Ascent.** Die Kundenwerte v_j werden angehoben, bis ein Standort „voll bezahlt“ (knapp) ist; ihre Summe ist eine untere Schranke, ganz ohne LP-Löser. **Ohne Anpassung** erreicht sie über 40 Kartennetze im Mittel nur **87,8 %** (schlechtestes Netz 76,7 %) und fällt mit den Fixkosten von 96,8 auf 75,3 % (Faktor 25 bis 400 %); **mit Anpassung** (Kundenwert auf den niedrigsten knappen Standort senken, alle anderen erneut anheben, behalten, wenn die Summe steigt; vereinfacht gegenüber Erlenkotters DUALOC) im Mittel **99,4 %** (schlechtestes 97,3 %, in 30 Netzen mindestens 99 %) und über die Fixkosten 99,9 / 99,6 / 99,3 / 99,1 / 99,3 %.
Die **Auswahl aus den knappen Standorten** (Komplementarität: keine zwei offenen Standorte, zu denen derselbe Kunde beiträgt) liegt im Mittel **4,0 %** über dem Optimum und ist nur in **4** Netzen bewiesen optimal; mit anschließendem Interchange 0,02 % (schlechtestes Netz 0,41 %, exakt in 35).

**Gleichstandsnetze** (30 × 30, ganzzahlige Kosten 1 bis 5, Fixkosten 3 oder 4, 40 Netze): hier wird UFL schwer. Starke LP im Mittel **97,3 %** (schlechtestes 95,3 %), der LP-Wert ist nur in **1** von 40 Netzen gleich dem Optimum; schwache LP 69,9 %. Add 3,07 % (Median 2,17, schlechtestes 10,64, exakt in 9), Drop 5,02 % (4,26; 13,33; exakt in 2), Interchange 1,79 % (2,04; 8,51; exakt in 18): **Add ist in 20 Netzen besser als Drop, in 8 schlechter** – umgekehrt wie auf der Karte.
Dual Ascent mit Anpassung erreicht 93,6 % (schlechtestes 89,1 %, ohne Anpassung 82,1 %), bewiesen in keinem Netz; die Auswahl aus den knappen Standorten liegt 26,7 % über dem Optimum, mit Interchange 1,32 % (schlechtestes 6,25 %, exakt in 22). Über die Fixkosten (25 bis 400 %, 20 Netze): starke LP 99,9 / 98,6 / 97,5 / 96,7 / 96,7 % (LP-Wert = Optimum in 19 / 3 / 1 / 1 / 2 Netzen), schwache LP 97,6 / 82,6 / 70,2 / 61,7 / 56,4 %, Dual Ascent mit Anpassung 99,8 / 95,2 / 93,3 / 91,7 / 85,9 %. Bei niedrigen Fixkosten ist auch dieses Netz leicht.

**Größenreihe** (5 Kartennetze je Größe, 10 × 20 / 20 × 50 / 30 × 75 / 40 × 100): offene Standorte im Mittel 4,0 / 7,6 / 10,6 / 13,0; starke LP 100,0 %, gleich dem Optimum in 5 / 5 / 5 / 4 von 5 Netzen; schwache LP 76,4 / 66,3 / 62,6 / 59,2 %. Bewertete Nachbarn: Add 40 / 144 / 290 / 484, Drop 49 / 183 / 414 / 739, Interchange 52 / 184 / 516 / 1 486; Abstand von Interchange 0,00 / 0,12 / 0,01 / 0,28 %. Der Abstand bleibt klein, der Aufwand wächst mit der Zahl der Kandidaten (Interchange am stärksten, es prüft Tauschzüge über alle Paare).

**Lehrnetze von Hand** (3 Standorte, 4 Kunden; per Brute-Force-Suche gefunden, im Test nachgerechnet):
- **Add-Falle:** Add öffnet zuerst S2 (der billigste Einzelstandort, Kosten 17), das Optimum 16 ist {S1, S3}; Drop und Interchange finden es. Der Dual Ascent beweist es: Schranke 16, Auswahl 16.
- **Drop-Falle:** Drop schließt sich in {S2, S3} (22), das Optimum 21 ist S1 allein, Add findet es. Dual Ascent 19, mit Anpassung 20: die Lücke von 1 bleibt.
- **Interchange steckt fest:** Add öffnet nur S1 (21); von dort hilft kein einzelnes Öffnen, Schließen oder Tauschen, das Optimum {S2, S3} kostet 20 und ist zwei Standorte weit weg. Drop findet es, der Dual Ascent beweist es (20 = 20).
- **Beide Fallen** (4 Standorte, 5 Kunden): Add und Drop enden bei 28 in {S2, S4}, Interchange tauscht S4 gegen S1 und trifft das Optimum 27 ({S1, S2}).
- **LP gebrochen:** das Optimum ist S2 allein (25), die starke LP kommt nur auf **24,5** (98,0 %), die schwache auf **18,25** (73,0 %); Dual Ascent 21, mit Anpassung 24.
- **Dual Ascent beweist das Optimum:** ohne Anpassung 23, mit Anpassung 25 = Optimum = die Kosten der knappen Standorte {S1, S2}; Add und Interchange enden dort bei 28, Drop bei 25.

**Voreinstellungen.** *Hohe Fixkosten* (Faktor 400 %): Optimum 63 970 mit 5 offenen Standorten, schwache LP 47,7 %, starke 100,0 %, Add 3,9 % darüber, Drop 1,9 %, Interchange exakt; Dual Ascent ohne Anpassung 67,1 %, mit Anpassung 96,8 % (5 Durchläufe). *Gleichstandsnetz* (30 × 30): Optimum 47 mit 4 Standorten, starke LP 95,8 %, schwache 70,3 %, Add 48, Drop 49, Interchange 47, Dual Ascent mit Anpassung 91,5 %. *Großes Netz* (40 × 100): Optimum 74 264 mit 14 Standorten, schwache LP 58,0 %, starke 100,0 %, Add 0,9 % darüber, Drop 0,3 %, Interchange exakt, Dual Ascent mit Anpassung 98,8 %.

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Bau standen mehrere Vermutungen im Plan (Modellprüfung der Erweiterung E3). Gemessen:

- **„Dual Ascent liefert eine Schranke nahe der LP“ – ohne Anpassung widerlegt.** 87,8 % im Mittel, mit den Fixkosten fallend bis 75,3 %; erst der Anpassungsschritt hebt sie auf 99,4 %. Die erste Fassung des Prototyps (nur der Aufstieg) hätte die Demo schwach aussehen lassen, ohne dass das Verfahren schwach ist.
- **„Die Auswahl aus den knappen Standorten ist fast optimal und beweist das Optimum“ – widerlegt.** 4,0 % im Mittel über dem Optimum (26,7 % auf Gleichstandsnetzen), bewiesen in 4 von 40 Kartennetzen; erst ein Interchange danach bringt sie auf 0,02 %.
- **„Drop ist besser als Add“ – nur auf Karten.** Dort in 32 von 40 Netzen (1,87 gegen 0,11 %); auf Gleichstandsnetzen kehrt es sich um (Add 3,07 %, Drop 5,02 %). Kein Verfahren gewinnt überall.
- **„Interchange findet immer das Optimum“ – nein.** Im Lehrnetz „steckt fest“ bleibt er bei 21 (Optimum 20); auf Karten trifft er in 32 von 40, auf Gleichstandsnetzen in 18 von 40 Netzen.
- **„UFL ist mit der LP schwer“ – nicht auf Zufallsdaten.** Der LP-Wert ist in 38 von 40 Kartennetzen gleich dem Optimum; schwer wird es nur bei Gleichständen (LP 97,3 %, in 1 von 40 Netzen gleich dem Optimum). Die schwache Formulierung dagegen liegt bei 61 %: die *Formulierung* macht den Unterschied, nicht das Modell.
- **„Die Reihenfolge der Kunden ist egal“ – nein.** Im Standardnetz schwankt die Schranke ohne Anpassung zwischen 32 870 und 37 781, mit Anpassung zwischen 33 924 und 39 944; „nächste Kunden zuerst“ ist am schlechtesten.
- **Bestätigt:** die starke LP trifft das Optimum fast immer (38 von 40); die schwache fällt mit den Fixkosten (81 auf 46 %); die Heuristiken sind auf Karten fast exakt.

## Grenzen (was die Demo nicht zeigt)

Unkapazitiert (mit Kapazitäten und Single-Sourcing ist die Schranke wieder schwach; das ist das nächste Stück der Linie mit Lagrange-Relaxation), Kosten linear in Entfernung und Nachfrage (Luftlinie in Zehntel-Einheiten, ganzzahlig), eine Ebene (kein Location-Routing), keine feste Standortzahl (p-Median), Dual Ascent mit vereinfachter Anpassung (nicht Erlenkotters volle DUALOC-Anpassung), erzeugte Netze ohne Fremddaten; die Gleichstandsnetze sind eine Konstruktion, kein Praxisfall.
Die Einheit „bewertete Nachbarn“ ist eine Zählung, keine Uhr; sie gewichtet jede Bewertung gleich, obwohl eine Bewertung mit der Zahl der offenen Standorte teurer wird.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Selbst probieren, Zug für Zug, die Schranke, Experimente auf Abruf |
| `ufl_scenario.py` | Netze (Karte, Gleichstand), Lehrnetze, SplitMix64-Zufallsstrom (Kopie aus den Vorgängern) |
| `ufl_heuristics.py` | Add, Drop, Interchange mit Zug-Verlauf |
| `ufl_dual.py` | Dual Ascent, Anpassung, primale Auswahl aus den knappen Standorten |
| `ufl_exact.py` | HiGHS: Optimum (MILP), starke und schwache LP, Brute Force für Kleinstnetze |
| `ufl_evaluation.py`, `ufl_visualization.py` | Vergleiche, Reihen, Verteilungen; Karte, Kostenmatrix, Balken, Reihen |
| `ufl_presets.py`, `ufl_constants.py` | Presets, Permalink, Regler-Grenzen, feste Seeds |
| `tests/` | 361 Tests: Szenario, Heuristiken (Lehrnetze, Brute Force), Dual (von Hand, Zulässigkeit, Maximalität), Exakt, Presets, Zahlen (`test_claims.py`), App |

Lokal starten: `pip install -r requirements.txt`, dann `streamlit run app.py`; Tests: `pip install -r requirements-dev.txt`, dann `python -m pytest tests`.
