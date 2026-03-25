# `run_bn_toy.py` — Vollständige Erklärung für Einsteiger (Deutsch)

> **Ziel dieses Dokuments:** Jede einzelne Zeile von `run_bn_toy.py` und
> `gnn/backtracking_network.py` wird so erklärt, dass jemand ohne
> Vorkenntnisse in Machine Learning oder PyTorch den Code vollständig
> verstehen kann. Analogien, Visualisierungen und schrittweise Erklärungen
> sind enthalten.

---

## 0. Das große Bild — Was macht dieses Skript überhaupt?

Stell dir eine Grippewelle in einer kleinen Stadt vor. Am Ende der Welle weißt
du, welche Menschen krank waren, welche sich erholt haben und welche noch
anfällig sind. Die Frage ist: **Wer war Patient Zero?**

Dieses Skript trainiert ein neuronales Netz, das genau das lernt: Aus einem
Schnappschuss des Endzustands einer Epidemie (wer ist krank, wer gesund, wer
genesen?) und der Kontaktstruktur (wer trifft wen, wann?) soll der
Ausgangspunkt der Epidemie identifiziert werden.

```
Analogie — Tinte im Fluss:
  Stell dir vor, jemand kippt Tinte in einen Fluss mit vielen
  Verzweigungen und Kanälen. Du siehst NUR das Endergebnis:
  welche Kanäle gefärbt sind. Das neuronale Netz lernt,
  rückwärts zu verfolgen, WO die Tinte eingekippt wurde.

  Das Schwierige: Das Ergebnis ist zufällig! Manchmal breitet
  sich die Tinte weit aus, manchmal stirbt sie früh ab.
  Das Modell muss mit dieser Unsicherheit umgehen.
```

### Der Datensatz: `toy_example_holme.csv`

Die echte Datei sieht so aus:
```
1 2 1
1 2 2
2 3 4
3 4 4
2 3 5
3 4 5
2 4 6
3 4 7
```

Das sind (u, v, t)-Tripel: Knoten u hat Knoten v zum Zeitpunkt t getroffen.
Daraus ergibt sich folgendes temporales Netzwerk (nach Umnummerierung auf 0-Basis):

```
Zeitachse →   t=0  t=1  t=2  t=3  t=4  t=5  t=6

Knoten 0 ↔ 1:  ─    █    █    ─    ─    ─    ─
Knoten 1 ↔ 2:  ─    ─    ─    █    ─    █    ─
Knoten 2 ↔ 3:  ─    ─    ─    █    ─    █    ─
Knoten 1 ↔ 3:  ─    ─    ─    ─    ─    ─    █
Knoten 2 ↔ 3:  ─    ─    ─    ─    ─    ─    █

█ = Kontakt findet statt
─ = kein Kontakt
```

### Der Gesamtablauf in 7 Schritten:

```
┌──────────────────────────────────────────────────────────────┐
│  1. CSV laden → NetworkX-Graph (wer trifft wen, wann?)       │
│       ↓                                                      │
│  2. C-Simulator: 5000 × N Epidemien simulieren               │
│       ↓                                                      │
│  3. Graph → PyTorch-Tensoren (edge_index, edge_attr)         │
│       ↓                                                      │
│  4. Simulationen → Trainings-Tensoren (X, y)                 │
│       ↓                                                      │
│  5. BacktrackingNetwork erstellen                            │
│       ↓                                                      │
│  6. Trainieren (Forward → Loss → Backward → Update)          │
│       ↓                                                      │
│  7. Evaluieren (Top-K, Mean Rank, ...)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. Imports und Pfad-Setup (Zeilen 24–52)

### 1.1 Standard-Bibliotheken

```python
from __future__ import annotations
```
Dieser Import aktiviert einen moderneren Parsing-Modus für Type Hints in
Python. Ohne ihn müsste man z.B. `List[int]` (mit Großbuchstaben, aus
`typing`) schreiben. Mit ihm kann man einfach `list[int]` schreiben. Es
ändert **nichts** am tatsächlichen Verhalten des Programms zur Laufzeit.

```python
import argparse
```
`argparse` ist Pythons Standard-Bibliothek, um Kommandozeilen-Argumente zu
verarbeiten. Statt Werte im Code zu ändern, kann man beim Aufruf schreiben:
```bash
python run_bn_toy.py --beta 0.5 --epochs 200
```
`argparse` liest das automatisch ein und gibt es als strukturiertes Objekt
zurück.

```python
import os
```
`os` ist das Interface zu Betriebssystem-Funktionen: Dateipfade zusammenbauen
(`os.path.join`), Verzeichnisse erstellen (`os.makedirs`), das aktuelle
Verzeichnis wechseln (`os.chdir`), und so weiter. Plattform-unabhängig (Linux,
Mac, Windows).

```python
import random
```
Pythons eingebauter Zufallsgenerator. Wird hier für `random.shuffle` (Batch-
Reihenfolge mischen) und `random.seed` (Reproduzierbarkeit) genutzt. Wichtig:
Dies ist ein **anderer** Zufallsgenerator als NumPy's und PyTorch's — alle drei
müssen separat geseedet werden.

```python
import sys
```
`sys.path` ist die Liste der Verzeichnisse, in denen Python nach Modulen sucht.
Durch `sys.path.insert(0, ROOT)` fügen wir das Projektverzeichnis ganz vorne
ein, damit `from gnn.backtracking_network import ...` funktioniert.

```python
from dataclasses import dataclass, field
```
`@dataclass` ist ein Python-Dekorator, der eine Klasse automatisch mit einem
`__init__`-Konstruktor, `__repr__` (Textdarstellung) und `__eq__`
(Gleichheitsvergleich) ausstattet. Spart viel Boilerplate-Code.

### 1.2 Wissenschaftliche Bibliotheken

```python
import numpy as np
```
NumPy (Numerical Python) ist die Grundlage fast aller wissenschaftlicher
Python-Pakete. Ein NumPy-Array ist wie eine Excel-Tabelle, aber:
- Viel schneller (implementiert in C)
- Unterstützt mehrdimensionale Arrays (Matrizen, Tensoren)
- Hunderte eingebaute mathematische Operationen

```
Analogie: NumPy-Array vs. Python-Liste
  Python-Liste:  [1, 2, 3, 4]           → flexibel, langsam
  NumPy-Array:   np.array([1, 2, 3, 4]) → starr (gleicher Typ), sehr schnell

  Bei 1 Million Elementen ist NumPy ~100x schneller.
```

```python
import torch
import torch.nn.functional as F
```
PyTorch ist das zentrale Framework für Deep Learning in diesem Projekt. Der
Unterschied zu NumPy: PyTorch-Tensoren können auf der GPU berechnet werden und
PyTorch merkt sich automatisch, wie jede Berechnung zustande kam, um
Gradienten berechnen zu können (Automatic Differentiation).

`torch.nn.functional` (kurz `F`) enthält mathematische Funktionen, die in
neuronalen Netzen häufig gebraucht werden: Verlustfunktionen, Aktivierungen,
Normalisierungen. Der Unterschied zu `torch.nn`:
- `torch.nn`: Objekte mit lernbaren Parametern (z.B. `nn.Linear`)
- `torch.nn.functional`: Reine Funktionen ohne eigenen Zustand (z.B. `F.relu`)

```python
import networkx as nx
```
NetworkX ist eine Python-Bibliothek für Graphen. Ein `nx.Graph` speichert
Knoten und Kanten als Python-Dictionaries und erlaubt das Anfügen beliebiger
Attribute (z.B. `G[u][v]["times"] = [1, 2, 4]`). Langsamer als z.B. igraph,
aber sehr benutzerfreundlich und gut für explorativen Einsatz.

```python
import wandb
```
Weights & Biases (W&B) ist ein Experiment-Tracking-Tool. Es loggt automatisch:
- Hyperparameter (Lernrate, Batch-Größe, ...)
- Trainingskurven (Loss pro Epoche)
- Metriken (Top-1 Score, Mean Rank, ...)
- Tabellen und Diagramme

Alles ist live im Browser unter `wandb.ai` abrufbar. Sehr nützlich, um
verschiedene Runs zu vergleichen.

```python
from sklearn.model_selection import train_test_split
```
scikit-learn ist die Standard-Bibliothek für klassisches ML. Hier wird nur
`train_test_split` genutzt: teilt Daten in Trainings- und Validierungsmengen
auf, mit optionaler Stratifizierung (gleiche Klassenverteilung in beiden Teilen).

### 1.3 Pfad-Setup (Zeilen 41–52)

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
```

```
Schritt für Schritt:

  __file__
    = "/Users/dariush/.../project-thesis-agents/toy-example/run_bn_toy.py"

  os.path.abspath(__file__)
    = absoluter Pfad (falls er relativ war, wird er aufgelöst)
    = "/Users/dariush/.../project-thesis-agents/toy-example/run_bn_toy.py"

  os.path.dirname(...)
    = Verzeichnis der Datei
    = "/Users/dariush/.../project-thesis-agents/toy-example"

  os.path.dirname(os.path.dirname(...))
    = Verzeichnis eine Ebene höher
    = "/Users/dariush/.../project-thesis-agents"    ← ROOT

  sys.path.insert(0, ROOT)
    Fügt ROOT an Position 0 (ganz vorne!) in Python's Suchpfad ein.
    Ohne das würde "from gnn.backtracking_network import ..." mit
    ModuleNotFoundError scheitern, weil Python nicht weiß, wo "gnn/" ist.
```

```
Verzeichnisstruktur:
  project-thesis-agents/         ← ROOT
  ├── toy-example/
  │   ├── run_bn_toy.py          ← diese Datei
  │   └── toy_example_holme.csv
  ├── gnn/
  │   └── backtracking_network.py
  ├── tsir/
  │   └── read_run.py
  └── eval/
      ├── ranks.py
      └── scores.py
```

```python
DATA_DIR = os.path.join(ROOT, "toy-example", "data")
```
`os.path.join` baut einen plattform-kompatiblen Pfad. Auf Windows wäre das
`ROOT\toy-example\data`, auf Linux/Mac `ROOT/toy-example/data`. Immer
`os.path.join` verwenden, nie manuell mit `/` oder `\` konkatenieren!

---

## 2. Konfiguration mit `@dataclass` (Zeilen 59–112)

### 2.1 Was ist `@dataclass`?

```python
@dataclass
class Config:
    csv_path: str   = "toy-example/toy_example_holme.csv"
    beta:     float = 0.30
    ...
```

Ein `@dataclass` ist eine Python-Klasse, bei der man nur die Felder und ihre
Standardwerte hinschreiben muss. Python generiert automatisch:

```python
# Das würde Python AUTOMATISCH erzeugen:
def __init__(self, csv_path="toy-example/toy_example_holme.csv",
             beta=0.30, ...):
    self.csv_path = csv_path
    self.beta = beta
    ...

def __repr__(self):
    return f"Config(csv_path={self.csv_path!r}, beta={self.beta!r}, ...)"
```

### 2.2 Alle Parameter im Detail

#### Netzwerk-Parameter

```python
csv_path: str  = "toy-example/toy_example_holme.csv"
directed: bool = False
```
- `csv_path`: Pfad zur Datei mit dem Kontaktnetzwerk.
- `directed`: Ist das Netzwerk **gerichtet**? Bei `False` bedeutet ein Kontakt
  zwischen A und B, dass A B infizieren KANN, aber auch B kann A infizieren.
  Bei `True` (DiGraph) würde ein Kontakt A→B nur bedeuten, dass A B infizieren
  kann (nicht umgekehrt). Für Krankheitsübertragung ist `False` meist realistischer.

#### SIR-Parameter

Das SIR-Modell ist das klassischste Epidemie-Modell:

```
Zustandsübergänge:

  S ──[β × Kontakt mit I]──→ I ──[μ pro Zeitschritt]──→ R

  S (Susceptible) = Anfällig, noch nicht krank
  I (Infectious)  = Infiziert, kann andere anstecken
  R (Recovered)   = Genesen, immun (kann nicht mehr infiziert werden)

  β = Wahrscheinlichkeit, dass ein S-Knoten bei einem Kontakt
      mit einem I-Knoten infiziert wird (pro Kontakt)

  μ = Wahrscheinlichkeit, dass ein I-Knoten in einem Zeitschritt
      genest (wird zu R)
```

```python
beta:   float = 0.30   # 30% Infektionswahrscheinlichkeit pro Kontakt
mu:     float = 0.20   # 20% Genesungswahrscheinlichkeit pro Zeitschritt
n_runs: int   = 5000   # 5000 Simulationen pro Quellknoten
```

Warum 5000 Runs? Weil eine einzelne SIR-Simulation zufällig ist:
```
Gleicher Quellknoten, drei verschiedene Runs:
  Run 1: Knoten 0 infiziert → Epidemie stirbt sofort (Pech)
  Run 2: Knoten 0 infiziert → Mittelgroße Ausbreitung
  Run 3: Knoten 0 infiziert → Fast alle infiziert (Glück)

Mit 5000 Runs mitteln sich die Zufälligkeiten heraus und wir bekommen
eine statistische Verteilung, die das Modell lernen kann.
```

#### Modell-Architektur

```python
hidden_dim:  int = 32   # Breite des Netzes (Embedding-Dimension D)
num_layers:  int = 6    # Anzahl der Graph-Faltungsschichten L
```

```
Visualisierung des Netzes (vereinfacht):

  Eingabe:  [B, N, 3]  ← B Beispiele, N Knoten, 3 SIR-Features

  Projektion: jeder Knoten: 3 → 32  (p_v, eine lineare Schicht)

  Schicht 1: Jeder Knoten aggregiert Nachrichten von direkten Nachbarn
             Informationsreichweite: 1 Hop
  Schicht 2: Jeder Knoten aggregiert von Nachbarn DER Nachbarn
             Informationsreichweite: 2 Hops
  ...
  Schicht 6: Informationsreichweite: 6 Hops
             In einem kleinen Netzwerk (4 Knoten): jeder kennt alles

  Ausgabe:  [B, N, 1] → [B, N]  Log-Wahrscheinlichkeit pro Knoten
```

Warum `hidden_dim=32`? Mehr Dimensionen = mehr Ausdruckskraft, aber auch mehr
Parameter und Overfitting-Gefahr. 32 ist ein guter Startwert für ein kleines
Spielbeispiel-Netzwerk.

#### Optimierungsparameter

```python
lr:           float = 1e-3   # = 0.001  Lernrate
weight_decay: float = 1e-4   # = 0.0001 L2-Regularisierung
```

**Lernrate** — wie groß ist jeder Lernschritt?

```
Visualisierung (1D-Verlustlandschaft):

  Loss
   │    ╭──╮           ╭──╮
   │   ╱    ╲         ╱    ╲
   │  ╱      ╲       ╱      ╲
   │ ╱        ╲_____╱        ╲____
   │                ↑ Minimum (tiefster Punkt)
   └──────────────────────────────→ Parameter W

  Zu große lr (0.5):
    ●──────────────────────────●   (springt über das Tal!)
    Problem: Modell lernt nicht, pendelt wild hin und her.

  Zu kleine lr (0.000001):
    ●●●●●●●●●●... (Babyschritte, braucht ewig)
    Problem: Training dauert Stunden/Tage.

  Gute lr (0.001):
    ●───●──●─●●●  (nähert sich dem Minimum effizient)
```

**Weight Decay** (L2-Regularisierung):

Bei jedem Update wird ein kleiner Teil der Gewichte "weggestraft":
```
W_neu = W_alt - lr × gradient - lr × weight_decay × W_alt
                                └── "Vergiss ein bisschen"
```
Das verhindert, dass einzelne Gewichte sehr groß werden. Große Gewichte
führen oft zu Overfitting: Das Modell "memoriert" die Trainingsdaten statt
das zugrundeliegende Muster zu lernen.

```python
batch_size: int   = 128
epochs:     int   = 500
early_stop: int   = 30
test_size:  float = 0.20
```

**Batch-Training** — warum nicht alle Daten auf einmal?

```
Szenario: 20.000 Trainingsbeispiele

Option A — Stochastic Gradient Descent (batch_size=1):
  Berechne Loss für 1 Beispiel → Update
  Berechne Loss für 1 Beispiel → Update
  ... × 20.000 = eine Epoche
  ✓ Viele Updates, kann aus lokalen Minima entkommen
  ✗ Sehr laut/zufällig (jedes Beispiel kann einen anderen Gradienten geben)
  ✗ Keine GPU-Parallelisierung

Option B — Batch Gradient Descent (batch_size=20000):
  Berechne Loss für ALLE 20.000 Beispiele → 1 Update
  ... = eine Epoche
  ✓ Stabiler Gradient (echter Gradient)
  ✗ 1 Update pro Epoche — sehr langsam
  ✗ Braucht viel GPU-Speicher

Option C — Mini-Batch (batch_size=128):  ← hier verwendet
  Berechne Loss für 128 Beispiele → Update
  Berechne Loss für nächste 128 → Update
  ... × 156 Schritte = eine Epoche (20000/128 ≈ 156)
  ✓ Gute Balance: stabiler Gradient + viele Updates + GPU-Parallelisierung
  ✓ 128 ist typischer Wert, der in GPU-Speicher passt
```

```python
seed: int = 42
```
Der Seed für Zufallszahlen. Mit demselben Seed bekommt man identische
Ergebnisse, egal wann und wo man das Skript ausführt. Das ist für
wissenschaftliche Reproduzierbarkeit essenziell. 42 ist eine populäre Wahl
(Referenz zu "The Hitchhiker's Guide to the Galaxy").

### 2.3 `parse_config()` im Detail

```python
def parse_config() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--beta", type=float, default=cfg.beta)
    # ... alle weiteren Parameter
    args = p.parse_args()
    return Config(**vars(args))
```

**`argparse.ArgumentParser`**: Erstellt einen Parser. `description=__doc__`
setzt den Hilfe-Text auf den Docstring der Datei (der ganz oben stehende
dreifache Anführungszeichen-Text).

**`p.add_argument("--beta", type=float, default=cfg.beta)`**: Registriert
`--beta` als optionales Argument. `type=float` konvertiert den String von der
Kommandozeile in einen float. `default=cfg.beta` wird genutzt wenn kein Wert
angegeben wird.

**`p.parse_args()`**: Liest `sys.argv` (die tatsächlichen Kommandozeilen-
Argumente) und gibt ein Namespace-Objekt zurück.

**`vars(args)`**: Konvertiert das Namespace-Objekt in ein Dictionary:
`{"beta": 0.5, "epochs": 200, ...}`

**`Config(**vars(args))`**: Erstellt ein Config-Objekt aus dem Dictionary.
`**dict` ist Python's Unpacking-Syntax:
```python
Config(**{"beta": 0.5, "epochs": 200})
# entspricht:
Config(beta=0.5, epochs=200)
```

---

## 3. Temporales Netzwerk laden (Zeilen 119–175)

### 3.1 Was ist ein Temporales Netzwerk?

```
Normaler Graph (statisch):
  A --- B       "A und B sind verbunden" (ohne Zeitinformation)
  |     |
  C --- D

Temporaler Graph:
  A ---[t=1,t=2]--- B
  |                  |
  C ---[t=4,t=5]--- D
  |
  [t=6] ── B

  Jede Kante trägt eine LISTE von Zeitpunkten, zu denen der Kontakt stattfand.
```

Für Epidemien ist das entscheidend: Eine Infektion kann nur über eine Kante
übertragen werden, wenn der Kontakt **nach** der Infektion des Senders
stattfindet. Ohne Zeitinformation würde man falsche Ausbreitungswege annehmen.

### 3.2 Die Funktion `load_temporal_network` Schritt für Schritt

```python
def load_temporal_network(csv_path: str, directed: bool) -> nx.Graph:
```

**Typ-Annotation**: `-> nx.Graph` sagt Python (und dem Leser), was die Funktion
zurückgibt. Das ist nur eine Dokumentations-Hilfe — Python prüft das zur Laufzeit
nicht. Trotzdem sehr nützlich für IDEs und Code-Verständnis.

#### Zeilen 139–146: Datei einlesen

```python
edges_raw: list[tuple[int, int, int]] = []
with open(csv_path) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        u, v, t = map(int, line.split())
        edges_raw.append((u, v, t))
```

**`with open(csv_path) as fh:`**
Der `with`-Kontext-Manager (Context Manager) öffnet die Datei und stellt
sicher, dass sie **automatisch geschlossen** wird, wenn der Block verlassen
wird — auch bei Fehlern. `fh` ist das File-Handle (Datei-Objekt).

**`line.strip()`**
Entfernt Whitespace (Leerzeichen, Tabs, Newlines) am Anfang und Ende.
```
"  1 2 3\n".strip() = "1 2 3"
```

**`line.startswith("#")`**
Überspringt Kommentarzeilen, die mit `#` beginnen — Konvention für
wissenschaftliche Datendateien.

**`map(int, line.split())`**
`"1 2 3".split()` → `["1", "2", "3"]` (String aufteilen)
`map(int, ["1","2","3"])` → Wendet `int()` auf jedes Element an
`u, v, t = ...` → Tuple-Unpacking: drei Variablen bekommen die drei Werte

Mit den echten Daten aus `toy_example_holme.csv`:
```
Datei:     "1 2 1\n1 2 2\n2 3 4\n..."
             ↓ nach Parsing:
edges_raw = [(1,2,1), (1,2,2), (2,3,4), (3,4,4),
             (2,3,5), (3,4,5), (2,4,6), (3,4,7)]
```

#### Zeilen 148–150: Knoten auf 0-Basis umnummerieren

```python
node_ids = sorted({n for u, v, _ in edges_raw for n in (u, v)})
node_map = {n: i for i, n in enumerate(node_ids)}
```

**Set Comprehension** `{n for u, v, _ in edges_raw for n in (u, v)}`:
Erstellt eine Menge (Set) aller vorkommenden Knoten-IDs. Sets haben keine
Duplikate — jede Knoten-ID erscheint genau einmal.
```
edges_raw enthält Knoten: 1, 2, 3, 4 (aus den Original-Daten)
{n ...} = {1, 2, 3, 4}
sorted(...) = [1, 2, 3, 4]
```

**Dict Comprehension** `{n: i for i, n in enumerate(node_ids)}`:
`enumerate([1,2,3,4])` → `[(0,1), (1,2), (2,3), (3,4)]`
Das ergibt: `{1: 0, 2: 1, 3: 2, 4: 3}`

Warum umnummerieren? PyTorch-Tensoren für Graphen brauchen
**zusammenhängende Indizes von 0 bis N-1**. Wenn die Originaldaten die Knoten
5, 10, 15 nummerieren, würde man einen Tensor der Größe 16 brauchen, obwohl
nur 3 Knoten existieren. Mit `node_map = {5:0, 10:1, 15:2}` vermeidet man das.

#### Zeilen 153: Zeitachse auf 0-Basis bringen

```python
t_min = min(t for _, _, t in edges_raw)
```
Generator-Ausdruck, der über alle Zeitstempel iteriert und das Minimum findet.
Später wird von jedem `t_raw` `t_min` abgezogen. Wenn die Daten bei t=1
beginnen (wie hier), werden alle Zeiten um 1 verschoben: t=1 → t=0, t=2 → t=1, usw.

#### Zeilen 155–169: NetworkX-Graph aufbauen

```python
G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
G.add_nodes_from(range(len(node_ids)))
```
- `nx.DiGraph()`: Gerichteter Graph (Kante A→B ist nicht dasselbe wie B→A)
- `nx.Graph()`: Ungerichteter Graph (Kante A-B gilt in beide Richtungen)
- `G.add_nodes_from(range(4))`: Fügt Knoten 0, 1, 2, 3 hinzu, ohne Kanten

```python
for u_raw, v_raw, t_raw in edges_raw:
    u = node_map[u_raw]      # Umrechnen: 1→0, 2→1, 3→2, 4→3
    v = node_map[v_raw]
    t = t_raw - t_min        # Zeitachse normalisieren (0-basiert)

    if G.has_edge(u, v):
        G[u][v]["times"].append(t)   # Kante existiert: Zeit hinzufügen
    else:
        G.add_edge(u, v, times=[t]) # Neue Kante mit Zeitliste anlegen
```

Nach der Verarbeitung von `toy_example_holme.csv`:
```
Kante 0↔1: times=[0, 1]     (t=1,t=2 aus Datei minus t_min=1)
Kante 1↔2: times=[3, 4]     (t=4,t=5 aus Datei)
Kante 2↔3: times=[3, 4, 6]  (t=4,t=5,t=7 aus Datei)
Kante 1↔3: times=[5]        (t=6 aus Datei)
```

```python
for _, _, data in G.edges(data=True):
    data["times"] = sorted(set(data["times"]))
```
`set()` entfernt Duplikate (falls der gleiche Kontakt mehrfach in der Datei
steht). `sorted()` sortiert die Zeiten aufsteigend. `data=True` bedeutet, dass
die Iteration auch das Attribut-Dictionary der Kante zurückgibt.

#### Zeilen 171–173: Graph-Metadaten speichern

```python
G.graph["n_nodes"]  = len(node_ids)
G.graph["t_max"]    = max(t for _, _, t in edges_raw) - t_min
G.graph["directed"] = directed
```
`G.graph[...]` ist ein Dictionary für Graph-weite Attribute in NetworkX
(im Gegensatz zu Knoten- oder Kanten-Attributen).

---

## 4. BN-Eingabetensoren bauen (Zeilen 182–222)

### 4.1 Warum müssen wir den Graphen in Tensoren umwandeln?

NetworkX ist komfortabel zum Aufbauen und Analysieren von Graphen, aber
PyTorch kann damit nicht direkt arbeiten. PyTorch braucht **Tensoren** — also
numerische Arrays in einer bestimmten Form.

Graph Neural Networks arbeiten intern mit zwei fundamentalen Tensoren:

```
1. edge_index [2, E]:  "Wer ist mit wem verbunden?"

   Format (COO = Coordinate Format):
     edge_index[0] = [Quelle_1, Quelle_2, ..., Quelle_E]   ← Zeile 0
     edge_index[1] = [Ziel_1,   Ziel_2,   ..., Ziel_E  ]   ← Zeile 1

   Für unser Beispiel-Netzwerk (E=8 gerichtete Kanten):
     edge_index[0] = [0, 1, 1, 2, 2, 3, 1, 3]  ← Quellen
     edge_index[1] = [1, 0, 2, 1, 3, 2, 3, 1]  ← Ziele

2. edge_attr [E, T]:   "Was wissen wir über jede Kante?"
   Hier: binäres Aktivierungsmuster (aktiv oder nicht, pro Zeitschritt)

   edge_attr[e, t] = 1.0, wenn Kante e zum Zeitpunkt t aktiv war
                   = 0.0, sonst
```

### 4.2 Die Funktion im Detail

```python
t_max    = G.graph["t_max"]
directed = G.graph["directed"]
T        = t_max + 1          # T Zeitschritte: 0, 1, 2, ..., t_max
```

`T = t_max + 1`: Wenn `t_max = 6`, dann gibt es 7 Zeitschritte (0 bis 6).
Wichtig für die Größe des Aktivierungsvektors `act`.

```python
src_list:  list[int]          = []
dst_list:  list[int]          = []
attr_list: list[torch.Tensor] = []
```
Drei Listen, in die wir Kante für Kante die Daten sammeln, bevor wir am Ende
alles zu Tensoren zusammenfügen.

```python
for u, v, data in G.edges(data=True):
    act = torch.zeros(T)          # Nullvektor der Länge T
    for t in data["times"]:
        act[t] = 1.0              # 1.0 setzen wo die Kante aktiv war
```

**`torch.zeros(T)`**: Erstellt einen Tensor der Länge T, gefüllt mit Nullen.
```
T=7:  act = [0., 0., 0., 0., 0., 0., 0.]

Kante 0↔1, times=[0,1]:
  act[0] = 1.0
  act[1] = 1.0
  act = [1., 1., 0., 0., 0., 0., 0.]

Kante 1↔2, times=[3,4]:
  act = [0., 0., 0., 1., 1., 0., 0.]
```

```python
    # Vorwärtskante u → v
    src_list.append(u)
    dst_list.append(v)
    attr_list.append(act)

    if not directed:
        # Rückwärtskante v → u (gleiche Aktivierungszeiten)
        src_list.append(v)
        dst_list.append(u)
        attr_list.append(act.clone())
```

Warum die Rückwärtskante? Bei einem ungerichteten Graphen ist ein Kontakt
zwischen A und B symmetrisch: A kann B infizieren UND B kann A infizieren. Im
GNN-Message-Passing muss diese Information in beide Richtungen fließen. Also
fügen wir für jede physikalische Kante zwei gerichtete Kanten ein.

**`act.clone()`**: Erstellt eine unabhängige Kopie des Aktivierungsvektors.

```
Warum nicht einfach act nochmal anhängen?

In Python sind Objekt-Referenzen der Standard:
  x = [1, 2, 3]
  y = x           # y zeigt auf DASSELBE Objekt wie x!
  y[0] = 99
  print(x)        # → [99, 2, 3]  ← x wurde auch verändert!

  Mit .clone():
  y = x.clone()   # y ist ein unabhängiges Objekt
  y[0] = 99
  print(x)        # → [1, 2, 3]   ← x ist unverändert

Falls später jemand die edge_attr der Rückwärtskante verändert,
soll das nicht die Vorwärtskante beeinflussen.
```

```python
edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
edge_attr  = torch.stack(attr_list, dim=0)   # [E, T]
```

**`dtype=torch.long`**: Knoten-Indizes müssen Ganzzahlen sein (long = 64-Bit-
Integer). Tensoren für Aktivierungen sind `float32` (32-Bit-Gleitkommazahlen).

**`torch.stack(list, dim=0)`**: Stapelt eine Liste von 1D-Tensoren zu einer 2D-
Matrix.

```
attr_list = [
  tensor([1., 1., 0., 0., 0., 0., 0.]),   # Kante 0→1
  tensor([1., 1., 0., 0., 0., 0., 0.]),   # Kante 1→0
  tensor([0., 0., 0., 1., 1., 0., 0.]),   # Kante 1→2
  ...
]

torch.stack(attr_list, dim=0):
  [[1., 1., 0., 0., 0., 0., 0.],
   [1., 1., 0., 0., 0., 0., 0.],
   [0., 0., 0., 1., 1., 0., 0.],
   ...]
  Shape: [E, T] = [8, 7]

dim=0 bedeutet: "Stapele entlang einer NEUEN ersten Dimension"
  Jeder Tensor [T] wird zu einer Zeile in der Matrix [E, T].
```

---

## 5. Trainingsdaten aus SIR-Simulationen (Zeilen 228–254)

### 5.1 Was kommt aus dem SIR-Simulator?

Der C-Simulator hat für **jeden** der 4 Knoten als mögliche Quelle
**5000 Simulationen** durchgeführt und das Ergebnis (den Endzustand aller
Knoten) als Binärdateien gespeichert:

```
ground_truth_S.bin: Shape nach Laden und Reshape = [4, 5000, 4]
ground_truth_I.bin: Shape = [4, 5000, 4]
ground_truth_R.bin: Shape = [4, 5000, 4]

truth_S[s][r][n] = 1, wenn Knoten n im Run r (mit Quelle s) am Ende
                      noch SUSCEPTIBLE (anfällig) ist
truth_I[s][r][n] = 1, wenn Knoten n INFECTIOUS (infiziert) ist
truth_R[s][r][n] = 1, wenn Knoten n RECOVERED (genesen) ist

Immer: truth_S[s][r][n] + truth_I[s][r][n] + truth_R[s][r][n] = 1
       (jeder Knoten ist genau in einem der drei Zustände)
```

Konkretes Beispiel (Quelle=Knoten 0, Run 0):
```
truth_S[0][0] = [0, 0, 1, 1]   ← Knoten 2, 3 noch gesund
truth_I[0][0] = [0, 1, 0, 0]   ← Knoten 1 ist noch infiziert
truth_R[0][0] = [1, 0, 0, 0]   ← Knoten 0 (Quelle!) ist genesen

Das ist der Endzustand: Die Epidemie hat sich von Knoten 0 auf 1 ausgebreitet.
Knoten 2 und 3 wurden nicht erreicht.
```

### 5.2 `build_training_data` im Detail

```python
def build_training_data(
    truth_S: np.ndarray,  # [N, N_RUNS, N]
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    n_nodes: int,
    n_runs:  int,
) -> tuple[torch.Tensor, torch.Tensor]:
```

```python
X = torch.tensor(
    np.stack([truth_S, truth_I, truth_R], axis=-1),
    dtype=torch.float32,
)  # Shape: [N, N_RUNS, N, 3]
```

**`np.stack([truth_S, truth_I, truth_R], axis=-1)`**:

`axis=-1` bedeutet "letzte Achse". Drei Arrays der Shape `[4, 5000, 4]`
werden zu einem Array der Shape `[4, 5000, 4, 3]` gestapelt:

```
Vorher (für Knoten n=0 in Simulation r=0, Quelle s=0):
  truth_S[0][0][0] = 0   (Knoten 0 ist nicht susceptible)
  truth_I[0][0][0] = 0   (Knoten 0 ist nicht infectious)
  truth_R[0][0][0] = 1   (Knoten 0 ist recovered)

Nachher (X[0][0][0]):
  X[0][0][0] = [0, 0, 1]   ← "Susceptible=0, Infectious=0, Recovered=1"

Das ist One-Hot-Encoding:
  Susceptible = [1, 0, 0]
  Infectious  = [0, 1, 0]
  Recovered   = [0, 0, 1]
```

**Warum One-Hot-Encoding?**
```
Wenn wir die Zustände als Zahlen kodieren würden: S=0, I=1, R=2

Das Problem: Das impliziert eine Ordnung und Abstände!
  Das Netz würde annehmen: |I - S| = 1, |R - I| = 1, |R - S| = 2
  Das ist medizinisch/mathematisch nicht sinnvoll.

Mit One-Hot-Encoding:
  [1,0,0], [0,1,0], [0,0,1]
  Alle drei Zustände haben den gleichen "Abstand" zueinander (√2).
  Das Netz lernt: "Ein Knoten ist in einem von drei diskreten Zuständen,
  ohne Hierarchie."

Weiteres Beispiel: Farben (rot, grün, blau) sollte man auch
nie als 0,1,2 kodieren — das würde falsche Ähnlichkeiten implizieren.
```

**`dtype=torch.float32`**: Neuronale Netze arbeiten fast immer mit 32-Bit-
Gleitkommazahlen. `float64` wäre genauer, aber doppelt so viel Speicher und
kaum merklich besser für ML.

```python
y = torch.tensor(
    np.repeat(np.arange(n_nodes), n_runs),
    dtype=torch.long,
)  # Shape: [N * N_RUNS]
```

**`np.arange(4)`**: Erstellt `[0, 1, 2, 3]`
**`np.repeat([0,1,2,3], 5000)`**: Wiederholt jeden Wert 5000 Mal:
`[0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3]` (4×5000 = 20.000 Werte)

```
y[0] = 0        ← Das Label für X[0][0] (Quelle=0, Run 0) ist 0
y[1] = 0        ← Das Label für X[0][1] (Quelle=0, Run 1) ist 0
...
y[4999] = 0     ← Das Label für X[0][4999] ist 0
y[5000] = 1     ← Das Label für X[1][0] (Quelle=1, Run 0) ist 1
...
y[19999] = 3
```

`dtype=torch.long`: Labels müssen Ganzzahlen sein — sie sind Klassen-Indizes,
keine kontinuierlichen Werte.

---

## 6. Das BacktrackingNetwork — Architektur und Forward Pass

### 6.1 Was ist ein PyTorch-Modul?

```python
class BacktrackingNetwork(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers):
        super().__init__()
        ...

    def forward(self, x, edge_index, edge_attr):
        ...
        return log_probs
```

Jedes neuronale Netz in PyTorch erbt von `torch.nn.Module`. Das ist der
Basisvertrag:
- `__init__`: Hier werden alle **lernbaren Schichten** definiert und als
  Attribute gespeichert. PyTorch erkennt automatisch alle `nn.Module`-
  Attribute und verfolgt ihre Parameter.
- `forward(...)`: Definiert den **Berechnungsweg** (Forward Pass). Wenn man
  `model(x, ...)` aufruft, wird intern `model.forward(x, ...)` ausgeführt —
  plus das Aufzeichnen des Computational Graphs für Backpropagation.

**`super().__init__()`**: Ruft den Konstruktor der Elternklasse
(`torch.nn.Module`) auf. Das initialisiert interne PyTorch-Strukturen. Ohne
diesen Aufruf würde PyTorch keine Parameter finden und Training würde nicht
funktionieren.

### 6.2 Die Initialisierungsprojektionen

```python
self.p_v = Sequential(Linear(node_feat_dim, hidden_dim), ReLU())
self.p_e = Sequential(Linear(edge_feat_dim, hidden_dim), ReLU())
```

**`nn.Linear(in, out)`**: Eine vollständig verbundene Schicht (Fully Connected
Layer). Berechnet `output = W × input + b`, wobei `W` eine `[out, in]`-Matrix
und `b` ein Bias-Vektor der Länge `out` ist. Alle `W` und `b` sind lernbare
Parameter.

```
Linear(3, 32): Eingabe hat 3 Features, Ausgabe hat 32 Features
  W: [32 × 3] = 96 Gewichte
  b: [32]      = 32 Biases
  Gesamt: 128 lernbare Parameter pro Linear-Schicht

  Beispiel (ein Knoten):
    Eingabe: [0, 0, 1]  (Zustand: Recovered)
    W × input + b = [0.3, -0.1, 0.7, ..., 0.2]  (32 Zahlen)
```

**`nn.ReLU()`**: Rectified Linear Unit — die beliebteste Aktivierungsfunktion:
```
ReLU(x) = max(0, x)

Visualisierung:
  Ausgabe
    │        /
    │       /
    │      /
  0 │_____/
    └──────── Eingabe
         0

Warum Aktivierungsfunktionen?
  Ohne sie wäre das ganze Netz nur eine riesige Matrixmultiplikation:
  W₃ × (W₂ × (W₁ × x)) = (W₃ × W₂ × W₁) × x = W_ges × x

  Das könnte man als eine einzige lineare Schicht ausdrücken!
  Aktivierungsfunktionen fügen NON-LINEARITÄT ein → das Netz kann
  komplexe, nicht-lineare Muster lernen.
```

**`nn.Sequential(Linear(...), ReLU())`**: Verknüpft Schichten hintereinander.
`Sequential(A, B)(x)` berechnet `B(A(x))`.

```
p_v (Node-Projektion): 3 → 32
  Jeder Knoten hat 3 Features (S/I/R) → wird auf 32-dimensionalen
  "Hidden Space" projiziert.

p_e (Edge-Projektion): T → 32
  Jede Kante hat T Features (Aktivierungsmuster) → wird auf 32-dimensionalen
  "Hidden Space" projiziert.
```

### 6.3 Die Faltungsschichten

```python
self.convs = ModuleList([BNConvLayer(hidden_dim) for _ in range(num_layers)])
```

**`nn.ModuleList`**: Eine Liste von PyTorch-Modulen. Wichtig: Eine normale
Python-Liste (`[BNConvLayer(...), ...]`) würde von PyTorch **nicht** als
Parameter erkannt! `ModuleList` registriert alle enthaltenen Module korrekt.

```
[BNConvLayer(32) for _ in range(6)]
= [BNConvLayer_1, BNConvLayer_2, ..., BNConvLayer_6]

Jede Schicht hat ihre eigenen, unabhängigen lernbaren Gewichte.
```

### 6.4 Eine `BNConvLayer` im Detail

```python
class BNConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.f_e = Sequential(Linear(2 * hidden_dim, hidden_dim), ReLU())
        self.f_v = Sequential(Linear(hidden_dim, hidden_dim), ReLU())
```

`f_e` (Edge Update): Eingabe ist die Konkatenation von
`[edge_hidden, source_node_hidden]` = `[D, D]` → `[2D]`. Ausgabe: `[D]`.

`f_v` (Node Self-Transform): Eingabe `[D]`, Ausgabe `[D]`.

```python
def forward(self, h, g, edge_index):
    src, dst = edge_index[0], edge_index[1]
    B, N, D = h.shape
```

`h`: Node Hidden States, Shape `[B, N, D]`
`g`: Edge Hidden States, Shape `[B, E, D]`
`edge_index`: `[[src1, src2, ...], [dst1, dst2, ...]]`

**Schritt 1 — Edge Update (Gleichung 3 aus dem Paper):**

```python
h_src = h[:, src, :]                              # [B, E, D]
g_new = self.f_e(torch.cat([g, h_src], dim=-1))   # [B, E, D]
```

**`h[:, src, :]`**: Fancy Indexing — für jede Kante (src, dst) holen wir die
Hidden State des **Quell-Knotens** (`src`).

```
h hat Shape [B, N, D] = [128, 4, 32]
src = [0, 1, 1, 2, 2, 3, 1, 3]  (Quellknoten jeder Kante)

h[:, src, :] = für jedes Batch-Element: nimm Zeilen 0,1,1,2,2,3,1,3 aus h
             → Shape [128, 8, 32]  (8 Kanten, je 32 Features)
```

**`torch.cat([g, h_src], dim=-1)`**: Konkateniert entlang der letzten Dimension:
```
g:     [B, E, D] = [128, 8, 32]
h_src: [B, E, D] = [128, 8, 32]

torch.cat(..., dim=-1):
  [B, E, 2D] = [128, 8, 64]

Jede Kante bekommt jetzt KOMBINIERTE Information:
  "Wie sah diese Kante vorher aus?" (aus g)
  "Was weiß der Sender gerade?" (aus h_src)
```

**Schritt 2 — Node Update (Gleichung 4 aus dem Paper):**

```python
dst_idx = dst.view(1, -1, 1).expand(B, -1, D)   # [B, E, D]
agg = h.new_zeros(B, N, D)                        # [B, N, D]
agg.scatter_add_(1, dst_idx, g_new)               # akkumuliere bei dst
h_new = F.relu(self.f_v(h) + agg)
```

**`scatter_add_`**: Das ist einer der komplexesten Teile. Erklärt:

```
Ziel: Summiere für jeden Ziel-Knoten (dst) alle eingehenden Kanten-Nachrichten.

agg = h.new_zeros(B, N, D)  → Leeres Akkumulierungs-Array, Shape [B, N, D]

scatter_add_(dim=1, index=dst_idx, src=g_new):
  "Addiere g_new[b, e, :] zu agg[b, dst[e], :] für alle b, e"

Konkret (1 Batch-Element, vereinfacht):
  g_new (Kanten-Updates):
    Kante 0→1: [0.3, 0.7, ...]
    Kante 1→0: [0.2, 0.4, ...]
    Kante 1→2: [0.5, 0.1, ...]
    Kante 2→3: [0.8, 0.2, ...]

  dst = [1, 0, 2, 3]  ← Jede Kante geht zu diesem Zielknoten

  agg nach scatter_add_:
    agg[0] = g_new[Kante 1→0] = [0.2, 0.4, ...]    ← Knoten 0 empfängt von Kante 1→0
    agg[1] = g_new[Kante 0→1] = [0.3, 0.7, ...]    ← Knoten 1 empfängt von Kante 0→1
    agg[2] = g_new[Kante 1→2] = [0.5, 0.1, ...]    ← Knoten 2 empfängt von Kante 1→2
    agg[3] = g_new[Kante 2→3] = [0.8, 0.2, ...]    ← Knoten 3 empfängt von Kante 2→3

  Hätte Knoten 1 ZWEI eingehende Kanten (0→1 und 2→1):
    agg[1] = g_new[Kante 0→1] + g_new[Kante 2→1]   ← summiert!

Das ist "Message Passing": Nachrichten von allen Nachbarn werden gesammelt.

Analogie: Jeder Knoten ist eine Postbox. Alle Nachbarn schicken einen Brief.
scatter_add_ legt alle Briefe in die richtige Postbox.
```

**`dst.view(1, -1, 1).expand(B, -1, D)`**: Dieser Ausdruck ist rein technisch
nötig, um `scatter_add_` korrekt aufzurufen (die Indizes müssen dieselbe Shape
wie der Quell-Tensor haben):
- `dst.view(1, -1, 1)`: Reshapet `[E]` → `[1, E, 1]`
- `.expand(B, -1, D)`: Expandiert zu `[B, E, D]` durch Broadcasting

```python
h_new = F.relu(self.f_v(h) + agg)
```
Die neue Knotenrepräsentation ist: ReLU der Summe aus
- `f_v(h)`: dem selbst-transformierten alten Hidden State
- `agg`: den aggregierten Nachrichten der Nachbarn

### 6.5 Expert Knowledge — Gleichung 6 aus dem Paper

```python
susceptible_mask = x[..., 0].bool()               # [B, N]
scores = scores.masked_fill(susceptible_mask, float('-inf'))
```

```
x[..., 0]: Das erste Feature jedes Knotens (= Susceptible-Indikator)
  x[b][n][0] = 1.0, wenn Knoten n in Beispiel b SUSCEPTIBLE ist
             = 0.0, sonst

.bool(): Konvertiert zu True/False
  [1.0, 0.0, 0.0, 1.0] → [True, False, False, True]

masked_fill(mask, -inf):
  Überall wo mask=True → Score wird -∞

Warum? Ein susceptibler Knoten war am Ende noch anfällig.
Das heißt: Er KONNTE die Quelle nicht sein!
(Die Quelle wurde infiziert und ist danach entweder I oder R.)

-∞ nach dem Log-Softmax:
  log_softmax(-∞) = log(exp(-∞) / Σ) = log(0 / Σ) = -∞
  exp(-∞) = 0  → Wahrscheinlichkeit 0 für susceptible Knoten

Das Netz ERZWINGT also korrekte Physik: Ein gesunder Knoten
kann nie die Quelle gewesen sein.
```

### 6.6 Log-Softmax

```python
log_probs = F.log_softmax(scores, dim=-1)   # [B, N]
```

**Softmax** wandelt beliebige Zahlen ("Scores") in eine Wahrscheinlichkeitsverteilung:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

Eigenschaften:
  - Alle Ausgaben sind zwischen 0 und 1
  - Alle Ausgaben summieren sich zu 1 (= echte Wahrscheinlichkeitsverteilung)
  - Höherer Score → höhere Wahrscheinlichkeit (exponentiell)

Beispiel (4 Knoten, Scores):
  scores = [-1.0, 2.0, 0.5, -inf]
            (S ist -inf, weil er susceptible ist)

  exp(scores) = [0.368, 7.389, 1.649, 0.0]
  Summe = 9.406

  softmax = [0.039, 0.785, 0.175, 0.0]
             ↑       ↑       ↑     ↑
           3.9%   78.5%  17.5%  0.0%

  log_softmax = log(softmax) = [-3.24, -0.24, -1.74, -inf]
```

**Warum LOG-Softmax?**
1. **Numerische Stabilität**: Wahrscheinlichkeiten können 1e-100 oder kleiner
   sein. Ihr Logarithmus (-230) ist gut handhabbar.
2. **Die NLL-Loss-Funktion erwartet Log-Wahrscheinlichkeiten**: Sie ist
   mathematisch konsistenter damit.
3. **Kein Informationsverlust**: `log` ist eine monotone Funktion — die
   Rangordnung bleibt erhalten. Der Knoten mit der höchsten Wahrscheinlichkeit
   hat auch den höchsten Log-Wert.

---

## 7. Trainingsschleife im Detail (Zeilen 261–357)

### 7.1 Train/Validation Split

```python
idx_pairs = [(s, r) for s in range(n_nodes) for r in range(n_runs)]
labels    = [s for s, _ in idx_pairs]
```

```
idx_pairs = [(0,0), (0,1), ..., (0,4999),
             (1,0), (1,1), ..., (1,4999),
             (2,0), ...,
             (3,0), ..., (3,4999)]
= 20.000 (Quelle, Run)-Paare

labels = [0, 0, ..., 0, 1, 1, ..., 1, 2, ..., 3, ..., 3]
       = die Quellknoten-IDs, ein Label pro Paar
```

```python
train_idx, val_idx = train_test_split(
    idx_pairs,
    test_size    = cfg.test_size,     # 0.20 → 20%
    random_state = cfg.seed,          # Reproduzierbarkeit
    stratify     = labels,
)
```

**Stratifizierter Split**: Ohne Stratifizierung könnte durch Zufall ein Knoten
mehr im Training landen als ein anderer — das verzerrt das Training.

```
Ohne stratify (Zufallssplit, Pech gehabt):
  Training: 4500 Runs von Quelle 0, 3000 von 1, 4000 von 2, 4500 von 3
  → Modell sieht Quelle 1 viel seltener! Unfair.

Mit stratify=labels:
  Training: genau 4000 Runs von jeder Quelle (80% × 5000)
  Validation: genau 1000 Runs von jeder Quelle (20% × 5000)
  → Faire Verteilung!
```

### 7.2 Der Adam-Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
)
```

**Adam** (Adaptive Moment Estimation) ist der Standard-Optimizer für Deep
Learning. Er ist eine Erweiterung des einfachen Gradient Descent:

```
Einfaches Gradient Descent:
  W_neu = W_alt - lr × ∂L/∂W

Problem 1 — Unterschiedliche Gradienten-Skalen:
  Manche Parameter haben Gradienten ~100, andere ~0.001.
  Mit einer einheitlichen lr lernen beide suboptimal.

Problem 2 — Richtungsänderungen:
  Wenn der Gradient ständig die Richtung wechselt, macht
  einfaches GD zig-zag-Bewegungen statt direkt zum Minimum.

Adam löst beides:
  1. Momentum (1. Moment):
     m = β₁ × m_prev + (1-β₁) × gradient
     "Merke dir einen gewichteten Durchschnitt der Gradienten"
     Wie eine Kugel, die beim Rollen Schwung aufbaut.

  2. Adaptiver Schritt (2. Moment):
     v = β₂ × v_prev + (1-β₂) × gradient²
     "Merke dir die quadratischen Gradienten"
     Parameter mit großen Gradienten bekommen kleine Updates,
     Parameter mit kleinen Gradienten bekommen große Updates.

  Update:
     W_neu = W_alt - lr × m / (√v + ε)

  Standardwerte: β₁=0.9, β₂=0.999, ε=1e-8
```

**`model.parameters()`**: Generator, der alle lernbaren Parameter des Modells
zurückgibt — einschließlich aller Schichten in verschachtelten Modulen (wie die
`BNConvLayer` in `ModuleList`). Das ist ein großer Vorteil der `nn.Module`-
Architektur: PyTorch findet automatisch alle Parameter.

### 7.3 Die Epochenschleife

```python
for epoch in range(cfg.epochs):
    model.train()
    random.shuffle(train_idx)
    epoch_loss = 0.0
```

**`random.shuffle(train_idx)`**: Mische die Trainingsbeispiele bei jeder Epoche
neu durch. Warum?

```
Ohne Shuffle:
  Epoche 1: [Quelle-0-Runs, Quelle-1-Runs, Quelle-2-Runs, Quelle-3-Runs]
  → Das Modell sieht immer erst alle Runs von Quelle 0, dann Quelle 1, ...
  → Es kann "vergessen" was es über frühere Quellen gelernt hat,
    wenn es lange nur Quelle 3 sieht (Katastrophales Vergessen)

Mit Shuffle:
  Epoche 1: [(Quelle 2, Run 847), (Quelle 0, Run 3), (Quelle 3, Run 1204), ...]
  → Zufällige Reihenfolge → gleichmäßiges, stabileres Lernen
```

#### Der Mini-Batch-Schritt:

```python
for b in range(0, len(train_idx), cfg.batch_size):
    batch = train_idx[b : b + cfg.batch_size]
```

`range(0, 16000, 128)` → `0, 128, 256, ..., 15872` (125 Batches)
`train_idx[b : b+128]` → Slice der Liste: 128 (source, run)-Paare

```python
x_b = torch.stack([X[s, r] for s, r in batch]).to(device)
```

**List Comprehension + Stack**: Für jedes (s,r)-Paar im Batch holen wir
`X[s, r]` — das ist ein Tensor der Shape `[N, 3]` (alle N Knoten, 3 SIR-
Features). `torch.stack([...])` stapelt 128 solche Tensoren zu `[128, N, 3]`.

`.to(device)` verschiebt den Tensor auf die GPU (falls vorhanden). Das muss
**nach** dem Erstellen passieren, da die Daten zuerst im CPU-Speicher
entstehen.

```python
y_b = torch.tensor(
    [y[s * n_runs + r].item() for s, r in batch], device=device
)
```

**`y[s * n_runs + r]`**: Berechnet den flachen Index in y.

```
y hat Shape [N * N_RUNS] = [20.000]:
  y[0..4999]    → Quellknoten 0
  y[5000..9999] → Quellknoten 1
  y[10000..14999] → Quellknoten 2
  y[15000..19999] → Quellknoten 3

Für (s=2, r=300): y[2 * 5000 + 300] = y[10300] = 2
```

**`.item()`**: Extrahiert einen Python-Skalar aus einem 0D-Tensor:
```python
tensor(2).item() == 2   # True — jetzt ein normaler Python-int
```

**`device=device` im Tensor-Konstruktor**: Erstellt den Tensor direkt auf dem
Ziel-Gerät. Effizienter als `.to(device)` danach.

#### Der Core-Loop: Die 5 unverzichtbaren Schritte

```python
optimizer.zero_grad()                               # 1
log_probs = model(x_b, edge_index, edge_attr)       # 2
loss = F.nll_loss(log_probs, y_b, reduction="sum")  # 3
loss.backward()                                     # 4
optimizer.step()                                    # 5
epoch_loss += loss.item()
```

---

### TIEFER EINBLICK: Was passiert bei `loss.backward()`?

PyTorch baut beim Forward Pass automatisch einen **Computational Graph**:

```
                 ┌─────────────────────────────────┐
                 │         COMPUTATIONAL GRAPH      │
                 │                                  │
  x_b ──────────→ p_v (Linear+ReLU) ─────────┐    │
                 │                            ↓    │
  edge_attr ────→ p_e (Linear+ReLU) ─────→ Layer1 ─→ Layer2 ─→ ... ─→ Layer6
                 │                                              │
                 │                                         final (Linear)
                 │                                              │
                 │                                     masked_fill(-inf)
                 │                                              │
                 │                                      log_softmax
                 │                                              │
                 │                                          log_probs
                 │                                              │
  y_b ──────────→──────────────────────────────────────→ nll_loss
                 │                                              │
                 │                                            LOSS
                 └─────────────────────────────────────────────┘

Jede Operation speichert:
  - Ihre Eingaben (für den Backward Pass)
  - Die Rechenvorschrift (wie leite ich ab?)
```

**`loss.backward()`** startet die **Backpropagation**:

```
Der Algorithmus rechnet von rechts nach links (Chain Rule):

  ∂Loss/∂Loss = 1   (Startpunkt)

  ∂Loss/∂log_probs = ∂NLL/∂log_probs
  (= -1 für den korrekten Knoten, 0 für alle anderen)

  ∂Loss/∂scores = ∂log_softmax/∂scores × ∂Loss/∂log_probs

  ∂Loss/∂h_L = ∂final_linear/∂h_L × ∂Loss/∂scores

  ∂Loss/∂W_final = h_L^T × ∂Loss/∂scores
  (= Gradient der letzten Schicht)

  ... und so weiter, rückwärts durch alle 6 Schichten

  Am Ende: Jeder Parameter W hat seinen Gradienten gespeichert in W.grad:
    final.weight.grad  → Wie soll W_final geändert werden?
    conv[0].f_v[0].weight.grad → ...
    conv[1].f_e[0].weight.grad → ...
    ... (für alle ~15.000 Parameter)
```

**Warum Chain Rule?**

```
Mathematisch: Wenn y = f(g(x)), dann:
  dy/dx = (dy/dg) × (dg/dx)

In unserem Netz:
  Loss = nll(log_softmax(linear(relu(linear(scatter_add(...))))))

  ∂Loss/∂W₁ = ∂Loss/∂W₆ × ∂W₆/∂W₅ × ∂W₅/∂W₄ × ... × ∂W₂/∂W₁

  PyTorch berechnet das automatisch — man muss nichts von Hand ableiten!
```

---

### 7.4 Validierung: `torch.no_grad()` im Detail

```python
model.eval()

with torch.no_grad():
    x_v   = torch.stack([X[s, r] for s, r in val_idx]).to(device)
    y_v   = torch.tensor([...], device=device)
    val_log  = model(x_v, edge_index, edge_attr)
    val_loss = F.nll_loss(val_log, y_v, reduction="sum").item()
```

**Was macht `torch.no_grad()`?**

Normalerweise erstellt PyTorch beim Forward Pass für jeden Tensor ein
"requires_grad"-Flag und speichert den Berechnungspfad. Das kostet:
- Zusätzlichen **Speicher** (für den Computational Graph)
- Zusätzliche **Rechenzeit** (für das Aufzeichnen)

Beim Validieren wollen wir KEINE Gradienten berechnen. `no_grad()` deaktiviert
das Aufzeichnen vollständig:

```
MIT torch.no_grad():
  x_v → (forward) → log_probs → loss
                                  ↑
                         Kein Graph gespeichert!
                         ~2× schneller, ~2× weniger Speicher

OHNE torch.no_grad():
  x_v → W₁ → ReLU → W₂ → ReLU → ... → loss
  Alles gespeichert für potenzielle backward()
```

**`model.eval()` vs `torch.no_grad()`**: Das sind ZWEI verschiedene Dinge!
- `model.eval()`: Ändert das **Verhalten** bestimmter Schichten (Dropout,
  BatchNorm). Schaltet keine Gradientenberechnung ab.
- `torch.no_grad()`: Schaltet Gradientenberechnung ab. Ändert keine Schicht-Behavior.

Man braucht oft **beide**:
```python
model.eval()
with torch.no_grad():
    predictions = model(x)
```

### 7.5 Early Stopping im Detail

```python
best_val  = float("inf")   # Bester bisheriger Validation-Loss (Start: unendlich)
patience  = 0              # Wie viele Epochen ohne Verbesserung?

# ... innerhalb der Epochenschleife:
if vl < best_val:
    best_val = vl
    patience = 0
else:
    patience += 1
    if patience >= cfg.early_stop:
        print(f"  Early stopping at epoch {epoch + 1}")
        break
```

```
Visualisierung Loss-Kurven:

  Loss
  │
  │ ╲
  │  ╲ train_loss
  │   ╲___
  │       ╲___
  │           ╲___
  │               ╲___ val_loss
  │                   ╲___
  │                       ╲── Minimum!  ← best_val hier speichern
  │                            ‾‾‾‾‾‾‾‾ val_loss steigt wieder (Overfitting!)
  │
  └──────────────────────────────────────── Epochen
                              ↑              ↑
                         Minimum        patience=30 → STOP!

float("inf"): Python's Darstellung von ∞. Jede echte Zahl ist kleiner,
also wird der erste Validation-Loss immer als "Verbesserung" gewertet.

Warum nicht Modell-Gewichte speichern?
  Im echten Training würde man hier den besten Modell-Checkpoint speichern
  (torch.save(model.state_dict(), "best_model.pt")).
  Im Toy-Beispiel wird das nicht gemacht — es geht nur darum zu stoppen.
```

### 7.6 Loss normalisieren

```python
tl = epoch_loss / len(train_idx)     # Durchschnittlicher Loss pro Beispiel
vl = val_loss   / len(val_idx)
```

Mit `reduction="sum"` hat `F.nll_loss` alle Losses summiert. Teilen durch die
Anzahl der Beispiele gibt den **durchschnittlichen Loss** — unabhängig von der
Batch-Größe und Datensatz-Größe vergleichbar.

```
Beispiel:
  epoch_loss nach einer ganzen Epoche = 12.400 (Summe über 16.000 Beispiele)
  tl = 12.400 / 16.000 = 0.775 pro Beispiel

  Wenn das Modell perfekt wäre: Loss ≈ 0
  Zufälliges Modell (4 Klassen): Loss ≈ log(4) ≈ 1.386 pro Beispiel
  Unser Modell: 0.775 < 1.386 → besser als Zufall!
```

---

## 8. Evaluation (Zeilen 365–465)

### 8.1 Die Selektionsmaske `sel`

```python
truth_S_flat = truth_S.reshape(-1, n_nodes)   # [N*N_RUNS, N]
sel = (1 - truth_S_flat).sum(axis=1) >= 2
```

**`truth_S.reshape(-1, n_nodes)`**: `-1` bedeutet "berechne diese Dimension
automatisch". `[4, 5000, 4].reshape(-1, 4)` → `[20000, 4]`.

```
1 - truth_S_flat: Für jeden Knoten: 1 wenn er NICHT susceptible ist (= I oder R)

(1 - truth_S_flat).sum(axis=1): Pro Simulation: Wie viele Knoten sind I oder R?

>= 2: Mindestens 2 Knoten betroffen (der Quellknoten selbst + mindestens 1 weiterer)

Warum filtern?
  Triviale Epidemien bieten keine Information:
    - Nur 1 Beteiligter (der Quellknoten selbst): Alle Quellen sehen gleich aus!
    - Das Modell kann nicht unterscheiden und sollte nicht bewertet werden.

  "Non-trivial outbreak": Mindestens 2 Knoten wurden betroffen.
```

### 8.2 Inferenz über alle Samples

```python
probs = np.zeros((n_total, n_nodes), dtype=np.float32)

with torch.no_grad():
    for s in range(n_nodes):
        x_s   = X[s].to(device)                    # [N_RUNS, N, 3]
        log_p = model(x_s, edge_index, edge_attr)   # [N_RUNS, N]
        probs[s * n_runs : (s + 1) * n_runs] = (
            log_p.exp().cpu().numpy()
        )
```

**`log_p.exp()`**: Konvertiert Log-Wahrscheinlichkeiten zurück in echte
Wahrscheinlichkeiten. `exp(log(p)) = p`.

```
log_p für einen Run: [-0.15, -2.30, -3.91, -inf]
exp(log_p):          [ 0.86,  0.10,  0.02,  0.00]

Summe ≈ 1.0 (echte Wahrscheinlichkeitsverteilung)
```

**`.cpu().numpy()`**:
- `.cpu()`: Falls der Tensor auf der GPU ist, hole ihn zurück auf die CPU.
  NumPy kann nur mit CPU-Speicher arbeiten.
- `.numpy()`: Konvertiert den PyTorch-Tensor in ein NumPy-Array. Beide teilen
  sich danach oft den gleichen Speicher (keine Kopie!).

```
Warum NumPy und nicht Tensor für probs?
  Die eval-Hilfsfunktionen (compute_expected_ranks, etc.) erwarten NumPy-Arrays.
  Zudem sind viele wissenschaftliche Tools in NumPy verfügbar.
```

### 8.3 Rang-Metriken

```python
ranks = compute_expected_ranks(probs, n_nodes=n_nodes, n_runs=n_runs)
```

```
probs[i][j] = Wahrscheinlichkeit, dass Knoten j die Quelle
              von Simulation i ist

compute_expected_ranks berechnet für jede Simulation i:
  "Welcher Rang hat der WAHRE Quellknoten?"

  Simulation 5: wahrer Quellknoten = 1
  probs[5] = [0.10, 0.70, 0.15, 0.05]
              ↑      ↑      ↑     ↑
           Rang 3  Rang 1  Rang 2 Rang 4

  ranks[5] = 1  ← Quellknoten 1 ist auf Platz 1! ✓

  Simulation 8: wahrer Quellknoten = 0
  probs[8] = [0.05, 0.80, 0.10, 0.05]
              ↑
           Rang 4  ← schlecht! Modell liegt falsch

  ranks[8] = 4
```

**Top-K Score**:
```
top_k_score(ranks, sel, k=1):
  Anteil der Simulationen (nur non-trivial), bei denen der wahre
  Quellknoten auf Rang 1 (= höchste Wahrscheinlichkeit) liegt.

  Bei 4 Knoten: Zufalls-Baseline = 25% (1/4)
  Gutes Modell: 60-80%

top_k_score(ranks, sel, k=3):
  Anteil, bei dem der wahre Knoten unter den Top-3 ist.
  Bei 4 Knoten: Zufalls-Baseline = 75% (3/4)
```

**Rank Score** (normalisiert):
```
rank_score = 1/rank (invertierter Rang)

  rank=1 → rank_score = 1.0  (beste Vorhersage)
  rank=2 → rank_score = 0.5
  rank=4 → rank_score = 0.25

  Durchschnitt über alle non-trivial Simulationen.
  Zufalls-Baseline: 1/N = 0.25 (bei 4 Knoten)
```

**Mean Rank**:
```
Durchschnittlicher Rang des wahren Quellknotens.
  Zufalls-Baseline: (N+1)/2 = 2.5  (bei 4 Knoten)
  Perfektes Modell: 1.0
  Je KLEINER, desto besser.
```

### 8.4 W&B Tabellen und Visualisierungen

```python
wandb.log({
    "results/per_source_table": wandb.Table(
        columns=["source_node", "n_valid", "top1", "top3", "rank_score", "mean_rank"],
        data=per_source_rows,
    )
})
```

`wandb.Table` erstellt eine interaktive Tabelle im W&B-Dashboard. Man kann
dort sortieren, filtern und die Daten exportieren. Gut um z.B. zu sehen: "Für
welchen Quellknoten funktioniert das Modell am schlechtesten?"

```python
wandb.log({
    "results/rank_histogram": wandb.Histogram(
        ranks[sel].tolist(), num_bins=n_nodes
    )
})
```

Ein Histogramm der Rang-Verteilung:
```
Idealer Fall (Modell ist gut):

  Häufigkeit
    │████
    │████
    │████ █
    │████ █ █ █
    └──────────── Rang
      1  2  3  4

  → Die meisten Simulationen haben Rang 1 (wahre Quelle zuerst)

Zufälliges Modell (Modell ist schlecht):
  Häufigkeit
    │████ ████ ████ ████
    │████ ████ ████ ████
    └──────────────────── Rang
      1     2    3    4

  → Gleichmäßig verteilt
```

---

## 9. Die `main()`-Funktion: Alles zusammen

### 9.1 Reproduzierbarkeit sicherstellen

```python
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
```

Warum **vier** verschiedene Seeds?

```
Python, NumPy, PyTorch (CPU) und PyTorch (GPU) haben vollständig
unabhängige Zufallsgeneratoren. Jeder muss separat gesetzt werden:

  random.seed(42):        Python's built-in random module
  np.random.seed(42):     NumPy's Zufallsgenerator
  torch.manual_seed(42):  PyTorch auf der CPU
  torch.cuda.manual_seed_all(42): PyTorch auf ALLEN GPUs

  Wenn man nur random.seed(42) setzt:
    np.random.choice([...]) → trotzdem zufällig
    torch.randn(3, 3) → trotzdem zufällig

Mit allen vier: Gleicher Seed = identische Ergebnisse, egal auf
welchem Computer und wann der Code ausgeführt wird.

Wichtig für Wissenschaft: Kollegen müssen deine Experimente
reproduzieren können!
```

### 9.2 Der SIR-Simulator

```python
H_cread = make_c_readable_from_networkx(G, t_max=t_max, directed=cfg.directed)
seed_gt = random.getrandbits(64)

R0, avg_os, sd, _ = tsir_run(
    H_cread,
    beta   = cfg.beta,
    mu     = cfg.mu,
    start_t = 0,
    end_t   = t_max,
    n      = cfg.n_runs,
    seed   = seed_gt,
    path   = f"{DATA_DIR}/ground_truth_{{}}.bin",
    log    = f"{DATA_DIR}/ground_truth.txt",
)
```

Der Simulator ist in **C** implementiert, nicht in Python. Warum?

```
Python ist ca. 50-100× langsamer als C für Schleifen.

5000 Runs × 4 Knoten × bis zu T=7 Zeitschritte = 140.000 Simulationsschritte

In Python: ~10 Sekunden
In C:      ~0.1 Sekunden

`make_c_readable_from_networkx` wandelt den Python-NetworkX-Graph in
C-kompatible Datenstrukturen um (Arrays statt Dictionaries).
`tsir_run` ruft dann den kompilierten C-Code über Python's ctypes-Interface auf.
```

**`f"{DATA_DIR}/ground_truth_{{}}.bin"`**: Das `{{}}` ist escaped — in einem
f-String wird `{{` zu einem literalen `{`. Das ergibt:
`"/Users/.../data/ground_truth_{}.bin"`
Der Simulator selbst füllt `{}` dann mit 'S', 'I', 'R' aus.

**`R0`** (Basic Reproduction Number):
```
R0 < 1:  Im Schnitt steckt jeder Infizierte weniger als 1 anderen an
         → Epidemie stirbt aus
R0 = 1:  Grenzfall (kritische Phase)
R0 > 1:  Im Schnitt steckt jeder Infizierte mehr als 1 anderen an
         → Epidemie breitet sich aus

Bei β=0.3, µ=0.2 im Holme-Toy-Netzwerk: R0 ≈ 1.5
```

### 9.3 Laden der Binärdaten

```python
truth_S, truth_I, truth_R = (
    np.fromfile(f"{DATA_DIR}/ground_truth_{s}.bin", dtype=np.int8)
      .reshape(n_nodes, cfg.n_runs, n_nodes)
    for s in "SIR"
)
```

**`np.fromfile(..., dtype=np.int8)`**: Liest rohe Bytes aus einer Datei. `int8`
= 8-Bit-Ganzzahl, Werte von -128 bis 127 (hier 0 oder 1). Sehr kompakt:
4 × 5000 × 4 = 80.000 Bytes = 78 KB pro Datei.

**`.reshape(4, 5000, 4)`**: Interpretiert das flache Array als 3D-Array.

**Generator Expression mit Tuple Unpacking**:
```python
truth_S, truth_I, truth_R = (
    np.fromfile(...).reshape(...)
    for s in "SIR"    # iteriert über die Zeichen 'S', 'I', 'R'
)
```
Das ist eine kompakte Schreibweise für:
```python
truth_S = np.fromfile(f"...ground_truth_S.bin", dtype=np.int8).reshape(...)
truth_I = np.fromfile(f"...ground_truth_I.bin", dtype=np.int8).reshape(...)
truth_R = np.fromfile(f"...ground_truth_R.bin", dtype=np.int8).reshape(...)
```

### 9.4 W&B Setup und Logging

```python
run = wandb.init(
    project = cfg.wandb_project,
    entity  = cfg.wandb_entity or None,
    config  = cfg.as_dict(),
    tags    = ["backtracking-network", "toy-example", "holme"],
)
print(f"\nW&B run: {run.url}\n")
```

**`wandb.init()`** startet einen neuen "Run" in W&B:
- `project`: Alle Runs des gleichen Projekts erscheinen zusammen im Dashboard
- `entity`: Team oder Benutzer-Account
- `config`: Hyperparameter werden als Tabelle gespeichert — gut zum Vergleich
  verschiedener Runs
- `tags`: Schlagwörter zum Filtern und Organisieren

```python
wandb.summary["network/n_nodes"] = n_nodes
wandb.log({"epoch": epoch + 1, "train/loss": tl, "val/loss": vl})
```

Der Unterschied:
- `wandb.summary[...]`: Einzel-Werte, die den gesamten Run charakterisieren
  (Netzwerkgröße, finale Metriken). Werden prominent in der Übersicht angezeigt.
- `wandb.log({...})`: Pro-Schritt-Werte (Loss pro Epoche, Metriken). Werden als
  Zeitreihen-Diagramme dargestellt.

---

## 10. Zusammenfassung: Vollständiger Datenfluss

```
toy_example_holme.csv
      │
      ▼
load_temporal_network()
  → NetworkX-Graph G
      │                         │
      ▼                         ▼
build_bn_inputs(G)        make_c_readable() + tsir_run()
  → edge_index [2, E]         │
  → edge_attr  [E, T]         ▼
  → T (int)            ground_truth_S/I/R.bin
                              │
                              ▼
                        truth_S [N, N_RUNS, N]
                        truth_I [N, N_RUNS, N]
                        truth_R [N, N_RUNS, N]
                              │
                              ▼
                        build_training_data()
                          → X [N, N_RUNS, N, 3]
                          → y [N * N_RUNS]
                              │
          ┌───────────────────┘
          │
          ▼
    BacktrackingNetwork(
      node_feat_dim=3,
      edge_feat_dim=T,
      hidden_dim=32,
      num_layers=6
    )
          │
          ▼
       train()
      ┌──────────────────────────────────┐
      │  for epoch in range(500):        │
      │    for batch in train_batches:   │
      │      1. zero_grad()              │
      │      2. log_p = model(x, e, ea)  │
      │      3. loss = nll_loss(log_p,y) │
      │      4. loss.backward()          │
      │      5. optimizer.step()         │
      │    validate()                    │
      │    early_stopping_check()        │
      └──────────────────────────────────┘
          │
          ▼
       evaluate()
         → ranks [N * N_RUNS]
         → top1, top3, rank_score, mean_rank
         → wandb Tables und Histogramme
          │
          ▼
    Ausgabe:
    ┌────────────────────────────────────────────────────┐
    │ Method        Top-1    Top-3  Rank score  Mean rank │
    │ ─────────────────────────────────────────────────  │
    │ Random         25.0%    75.0%       0.250       2.50│
    │ BN             68.3%    95.1%       0.742       1.38│
    └────────────────────────────────────────────────────┘
```

---

## 11. Referenz-Glossar

### PyTorch-Kernkonzepte

| Begriff | Erklärung |
|---------|-----------|
| `torch.Tensor` | Mehrdimensionales Array, kann auf GPU laufen, kann Gradienten haben |
| `dtype=torch.float32` | 32-Bit-Gleitkommazahl (Standard für ML) |
| `dtype=torch.long` | 64-Bit-Ganzzahl (für Indizes und Labels) |
| `.to(device)` | Verschiebt Tensor/Modell auf CPU oder CUDA-GPU |
| `torch.no_grad()` | Deaktiviert Gradientenberechnung (Validierung/Inferenz) |
| `model.train()` | Setzt Trainings-Modus (Dropout aktiv, BatchNorm nutzt Batch-Stats) |
| `model.eval()` | Setzt Evaluierungs-Modus (Dropout deaktiv, BatchNorm nutzt gespeicherte Stats) |
| `optimizer.zero_grad()` | Setzt alle `.grad`-Attribute auf Null (vor jedem Batch) |
| `loss.backward()` | Berechnet Gradienten durch Backpropagation (Chain Rule) |
| `optimizer.step()` | Aktualisiert Parameter basierend auf `.grad` (nach backward) |
| `tensor.item()` | Extrahiert Python-Skalar aus 0D-Tensor |
| `tensor.cpu()` | Kopiert Tensor von GPU in CPU-Speicher |
| `tensor.numpy()` | Konvertiert PyTorch-Tensor zu NumPy-Array (CPU only) |
| `tensor.clone()` | Erstellt unabhängige Kopie eines Tensors |
| `torch.stack(list)` | Stapelt Liste von Tensoren zu einem neuen Tensor |
| `torch.cat(list, dim)` | Konkateniert Tensoren entlang bestehender Dimension |
| `tensor.reshape(...)` | Ändert die Shape ohne Daten zu kopieren |
| `tensor.unsqueeze(dim)` | Fügt eine neue Dimension der Größe 1 ein |
| `tensor.squeeze(dim)` | Entfernt Dimensionen der Größe 1 |
| `tensor.expand(...)` | Wiederholt Tensor entlang Size-1-Dimensionen (zero-copy) |
| `scatter_add_` | Akkumuliert Tensor-Werte an indizierten Positionen (Message Passing) |
| `masked_fill(mask, val)` | Setzt Werte an `True`-Positionen auf `val` |

### PyTorch nn-Module

| Modul | Was es tut |
|-------|-----------|
| `nn.Linear(in, out)` | `W × x + b`, vollverbundene Schicht, `W:[out,in]`, `b:[out]` |
| `nn.ReLU()` | `max(0, x)`, Aktivierungsfunktion |
| `nn.Sequential(A, B, ...)` | Verknüpft Module in Reihe: `B(A(x))` |
| `nn.ModuleList([m1, m2, ...])` | Liste von Modulen (Parameter werden registriert!) |

### Verlustfunktionen

| Funktion | Erklärung |
|----------|-----------|
| `F.nll_loss(log_p, y)` | Negative Log-Likelihood; erwartet Log-Wahrscheinlichkeiten |
| `F.log_softmax(x, dim)` | Numerisch stabile Kombination aus log und softmax |
| `F.cross_entropy(x, y)` | Entspricht `nll_loss(log_softmax(x), y)` — kurzform |

### SIR-Epidemiologie

| Begriff | Bedeutung |
|---------|-----------|
| S (Susceptible) | Anfällig, noch nicht infiziert |
| I (Infectious) | Infiziert, kann andere anstecken |
| R (Recovered) | Genesen, immun |
| β (beta) | Infektionswahrscheinlichkeit pro Kontakt |
| μ (mu) | Genesungswahrscheinlichkeit pro Zeitschritt |
| R₀ | Basic Reproduction Number: Ø Infektionen pro Infiziertem |

### Bewertungsmetriken

| Metrik | Formel | Zufalls-Baseline (4 Knoten) |
|--------|--------|------------------------------|
| Top-1 | P(Rang=1) | 25% (1/N) |
| Top-3 | P(Rang≤3) | 75% (3/N) |
| Rank Score | Ø(1/Rang) | 25% (1/N) |
| Mean Rank | Ø(Rang) | 2.5 ((N+1)/2) |

---

*Dieses Dokument erklärt `toy-example/run_bn_toy.py` und
`gnn/backtracking_network.py` vollständig für ML-Einsteiger.*
