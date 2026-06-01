# Split n' Simulate
### LongEval 2026 — Task 3: User Simulation

Dieses Projekt ist im Rahmen des DIS18 Projektseminars an der TH Köln entstanden.
Ziel war es, einen User Simulator zu entwickeln der für gegebene Suchsitzungen die nächste Suchanfrage vorhersagt.

---

## Idee

Wir klassifizieren jede Session in eine von 8 Personas (z.B. Student Researcher, Academic Researcher, Policy Analyst) und lassen dann ein LLM aus der Perspektive dieser Persona 5 mögliche nächste Queries vorhersagen. Die 5 Predictions werden anschließend nach ihrer Ähnlichkeit zur letzten bekannten Query gerankt.

**Modell:** llama3.2 (lokal via Ollama)

---

## Voraussetzungen

- Python 3.8+
- [Ollama](https://ollama.com) mit llama3.2 installiert und laufend
- Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

> **Hinweis:** Die Dateipfade in den Skripten sind absolut angegeben und müssen vor der Ausführung an den eigenen Rechner angepasst werden. Durch das ganze Löschen und Umsortieren fürs Repo passt es nicht mal bei mir. :D

---

## Ablauf

Für jeden Datensatz (Train, 06-08, 09-11) gibt es jeweils eine eigene Simulation-, Fix-, Evaluation- und Submission-Datei.

**Beispiel für den Traindatensatz (snapshot-1):**

```bash
# 1. Simulation starten
python simulation/longeval_simulation_train.py

# 2. CSV bereinigen (Duplikate & Dokument-IDs entfernen)
python fix/longeval_fix.py

# 3. Evaluation & Ranking
python evaluation/longeval_evaluation.py

# 4. Submission erstellen
python submission/longeval_submission.py --submit --snapshot 1
```

Für die Testdatensätze entsprechend:
- `simulation/longeval_simulation_06_08.py` → `fix/longeval_fix_06_08.py` → `evaluation/longeval_evaluation_06_08.py` → `submission/longeval_submission_06_08.py --submit`
- `simulation/longeval_simulation_09_11.py` → `fix/longeval_fix_09_11.py` → `evaluation/longeval_evaluation_09_11.py` → `submission/longeval_submission_09_11.py --submit`

---

## Projektstruktur

```
Split-n-Simulate/
├── README.md
├── requirements.txt
│
├── data/                          # Originale Datensätze von LongEval
│   ├── task3_longeval_usim-sessions-train.csv
│   ├── task3_longeval_usim-sessions-06-08_2025.csv
│   └── task3_longeval_usim-sessions-09-11_2025.csv
│
├── simulation/                    # Simulation-Skripte
│   ├── longeval_simulation_train.py
│   ├── longeval_simulation_06_08.py
│   └── longeval_simulation_09_11.py
│
├── fix/                           # Bereinigung (Duplikate & Dokument IDs)
│   ├── longeval_fix.py
│   ├── longeval_fix_06_08.py
│   └── longeval_fix_09_11.py
│
├── evaluation/                    # Evaluation & Ranking
│   ├── longeval_evaluation.py
│   ├── longeval_evaluation_06_08.py
│   └── longeval_evaluation_09_11.py
│
├── submission/                    # Submission Skripte
│   ├── longeval_submission.py
│   ├── longeval_submission_06_08.py
│   └── longeval_submission_09_11.py
│
├── snapshot/                      # Finale Submissions für TIRA
│   ├── snapshot-1.json
│   ├── snapshot-2.json
│   └── snapshot-3.json
│
├── results/                       # Simulation & Evaluationsergebnisse
│   ├── longeval_simulation_all.csv
│   ├── longeval_simulation_06_08.csv
│   ├── longeval_simulation_09_11.csv
│   ├── longeval_evaluation_results.csv
│   ├── longeval_evaluation_results_06_08.csv
│   └── longeval_evaluation_results_09_11.csv
│
└── cache/                         # Persona Cache (automatisch erstellt)
    ├── session_personas_cache.json
    ├── session_personas_cache_06_08.json
    └── session_personas_cache_09_11.json
```

---

## Team

Kübra & Emirhan — TH Köln, DIS18, 2026