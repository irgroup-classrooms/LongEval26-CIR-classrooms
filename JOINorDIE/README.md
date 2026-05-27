# DIS18 – LongEval-USim: Next Query Prediction (Team JOINorDIE)

Dieses Projekt nimmt am **LongEval-USim Track** (CLEF 2026) teil. Ziel ist die Vorhersage der nächsten Suchanfrage einer Nutzerin / eines Nutzers auf Basis ihrer bisherigen Suchsession. Es werden zwei Prompt-Varianten (A und B) mit drei Query-Modi (`first`, `last`, `all`) und drei Dokument-Varianten (`rel`, `non_rel`, `contrastive`) kombiniert – insgesamt 18 Runs.

---

## 📁 Projektstruktur

| Datei / Ordner | Beschreibung |
|---|---|
| `00_Data_loading.ipynb` | Installiert `ir-datasets-longeval` und zeigt das Datenladen aus dem offiziellen LongEval-Datensatz |
| `01_indexierung.ipynb` | Baut den Pyterrier-BM25-Index aus dem LongEval-SCI-Korpus |
| `02_sessions.ipynb` | Lädt Nutzersessions aus dem Datensatz und speichert sie als `data/sessions.csv` |
| `03_retrieval_BM25.ipynb` | BM25-Retrieval pro Session: extrahiert pseudo-relevante und nicht-relevante Dokumente, speichert `data/retrieval_bm25_topic.csv` |
| `04_tfidf.ipynb` | Berechnet TF-IDF-Schlüsselbegriffe aus den abgerufenen Dokumenten und baut Evidence-Strings für die Topic-Generierung, speichert `data/evidence_topic_generation.csv` |
| `05_LLM_Topic_Generierung_openai.ipynb` | Generiert TREC-Topics (title, description, narrative) per GPT-4o-mini und speichert die Runs als JSONL in `runfiles/` |
| `05_LLM_Topic_Generierung_ollama.ipynb` | Generiert TREC-Topics (title, description, narrative) per ollama|
| `08_task3_usim.ipynb` | *(Entwicklungsnotebook)* Prototyp für einen einzelnen Task-3-Run – diente zur interaktiven Entwicklung der Prompt-Logik |
| `08_task3_usim_18runs.ipynb` | *(Entwicklungsnotebook)* Notebook-Version für alle 18 Runs; wurde schrittweise zu `run_all_18.py` ausgebaut |
| `09_task3_evaluation.ipynb` | *(Entwicklungsnotebook)* Interaktive Evaluation der 18 Runs; produktive Version ist `rerun_evaluation.py` |
| `run_all_18.py` | **Finale Version** – vollautomatische Ausführung aller 18 Runs (hervorgegangen aus `08_task3_usim_18runs.ipynb`) |
| `fix_snapshot3_A_runs.py` | Repariert snapshot-3 in allen 9 Prompt-A-ZIPs via Raw-ZIP-Parsing + API-Neugenerierung |
| `rerun_evaluation.py` | **Finale Version** – wertet alle 18 Runs neu aus (hervorgegangen aus `09_task3_evaluation.ipynb`) |
| `task3_prep_tira.py` | Entpackt alle 18 ZIPs in `tira_uploads/<run>/` für den TIRA-Upload |
| `data/` | Zwischenergebnisse: Sessions, Retrieval, Evidence, Topics, Evaluationsergebnisse |
| `runfiles/` | JSONL-Runfiles für Task 1/2 (Topic-basiertes Retrieval) |
| `runfiles_task3/` | ZIP-Abgaben für Task 3 (18 Runs) |
| `runs/` | Finale JSONL-Kopien der Task-1/2-Runs |
| `tira_uploads/` | Entpackte JSON-Dateien je Run, bereit für TIRA |
| `Index_Longeval_SCI_snapshot3/` | Pyterrier-BM25-Index (binär, nicht versioniert) |
| `.env` | API-Schlüssel (nicht committen!) |

---

## 📋 Voraussetzungen / Dependencies

```
pip install pyterrier pandas numpy scikit-learn openai python-dotenv
pip install sentence-transformers matplotlib
pip install git+https://github.com/clef-longeval/ir-datasets-longeval
```

Außerdem wird ein **OpenAI-API-Key** benötigt. Dieser wird in einer `.env`-Datei im Projektordner hinterlegt:

```
OPENAI_API_KEY=sk-...
```

---

## ⚙️ Ausführungsreihenfolge

### Task 2 – Topic-basiertes Retrieval (Notebooks)

1. `00_Data_loading.ipynb` – Datensatz einmalig laden und prüfen
2. `01_indexierung.ipynb` – BM25-Index aufbauen (nur einmal nötig)
3. `02_sessions.ipynb` – Sessions extrahieren → `data/sessions.csv`
4. `03_retrieval_BM25.ipynb` – Retrieval durchführen → `data/retrieval_bm25_topic.csv`
5. `04_tfidf.ipynb` – TF-IDF-Evidence berechnen → `data/evidence_topic_generation.csv`
6. `05_LLM_Topic_Generierung_openai.ipynb` – Topics per LLM generieren (3× für `rel`, `non_rel`, `contrastive`) → `runfiles/*.jsonl`
7. `05_LLM_Topic_Generierung_ollama.ipynb` – Topics per LLM generieren (3× für `rel`, `non_rel`, `contrastive`) → `runfiles/*.jsonl` - llama 3.1:8b 

### Task 3 – Next Query Prediction

> Die Notebooks `08_task3_usim.ipynb` und `08_task3_usim_18runs.ipynb` dokumentieren die Entwicklungsschritte.
> Für die eigentliche Ausführung die finalen Python-Skripte verwenden:

```bash
python run_all_18.py          # Alle 18 Runs generieren
python rerun_evaluation.py    # Evaluation auswerten
python task3_prep_tira.py     # Für TIRA-Upload vorbereiten
```

Bei beschädigten snapshot-3-ZIPs (fehlende Central Directory):

```bash
python fix_snapshot3_A_runs.py
```

---

## 🔧 Verwendung

**Einzelnen Task-3-Run interaktiv ausführen:**
Öffne `08_task3_usim_18runs.ipynb` und passe in Cell 3 (`RUN_CONFIG`) die vier Parameter an:
- `query_mode`: `"first"` | `"last"` | `"all"`
- `prompt_variant`: `"A"` | `"B"`
- `doc_variant`: `"rel"` | `"non_rel"` | `"contrastive"`
- `run_name`: eindeutiger Name für den Output-ZIP

**Evaluation ansehen:**
Die Ergebnisse stehen in `data/task3_evaluation_results.csv` und als Chart in `data/task3_evaluation_chart.png`.
