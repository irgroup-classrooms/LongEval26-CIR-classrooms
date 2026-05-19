# LongEval-Sci Retrieval Pipeline: Multi-Stage & Citation Boosting

Dieses Repository enthält den Prototyp einer Retrieval-Architektur zur Verarbeitung der `LongEval-Sci`-Kollektion. Das System ist als [Marimo](https://marimo.io/)-Notebook implementiert und kombiniert probabilistische Modelle wie BM25 mit modernem Feature Engineering, inklusive *Conditional Recency Boosting* und *Citation-Impact-Gewichtung*.

## Datensätze & Ressourcen
Die Evaluation nutzt folgende Quellen:
* **Korpus & Queries**: `longeval-sci-2026/snapshot-1/train/raw` (via `ir_datasets`).
* **Zitationsdaten**: Externe Daten zur Berechnung des *Temporal Impact Boosts* und *Citation Binary Boosts*.

## Konfiguration & Pfadverwaltung
Zur Reproduktion der Ergebnisse müssen die lokalen Pfade im Skript angepasst werden:

1. **`open_citations.csv`**: Diese Datei muss im Root-Verzeichnis (parallel zum `.py`-Skript) abgelegt werden. Sie wird mittels `polars` eingelesen und benötigt die Spalten `creation` (Datum) sowie `cited_doc_id` (Mapping auf `docno`).
2. **PyTerrier Index-Pfad**: Der Pfad zum Index ist derzeit hartcodiert. Suchen Sie im Skript nach `index_path=` bzw. `pt.IndexFactory.of(...)` und passen Sie den Pfad (`C:/Projektdings/dis18-2025/tutorials/index`) an Ihre lokale Umgebung an.

## Installation & Setup

### Voraussetzungen
Das System erfordert eine Java-Installation (JRE/JDK 11+).

### Abhängigkeiten
Installieren Sie die benötigten Python-Bibliotheken:
```bash
pip install marimo python-terrier ir_datasets pandas polars matplotlib tqdm
