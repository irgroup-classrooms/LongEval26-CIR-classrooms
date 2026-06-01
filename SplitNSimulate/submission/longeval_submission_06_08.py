# LongEval Submission Generator — Snapshot 2 (06-08 2025)
# Kübra & Emirhan, 2026
# ============================================================
#
# Was macht dieses Skript?
# ------------------------
# Es erstellt die finalen Submission Dateien für TIRA.
#
# Schritt 1 — Duplikate bereinigen (--clean):
#   Sessions mit doppelten Predictions werden aus der CSV
#   entfernt. Danach muss longeval_simulation_train.py nochmal
#   gestartet werden, es füllt nur die fehlenden Sessions
#   neu auf.
#
# Schritt 2 — Submission erstellen (--submit):
#   Liest das Ranking aus der Evaluation CSV (dort wurde
#   bereits nach Cosine Similarity sortiert) und schreibt
#   die snapshot-X.jsonl Datei im LongEval Format.
#
# Submission Format (eine einzige JSON Datei):
#   {
#     "meta": {"team_name": "...", "run_name": "...", "description": "..."},
#     "0": ["pred1", "pred2", "pred3", "pred4", "pred5"],
#     "1": ["pred1", "pred2", "pred3", "pred4", "pred5"],
#     ...
#   }
#
# Verwendung:
# -----------
# Schritt 1: python longeval_submission.py --clean
#            → Entfernt Duplikat Sessions aus der CSV
#            → Danach: python longeval_simulation_train.py
#
# Schritt 2: python longeval_evaluation.py
#            → Berechnet Similarity + Ranking
#
# Schritt 3: python longeval_submission.py --submit --snapshot 1
#            → Erstellt snapshot-1.jsonl (Train Datensatz)
#
# Für Test Datensätze später:
#   python longeval_submission.py --submit --snapshot 2
#   python longeval_submission.py --submit --snapshot 3
# ============================================================

import pandas as pd
import json
import argparse
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

SIMULATION_FILE = Path("longeval_simulation_06_08.csv")
EVALUATION_FILE = Path("longeval_evaluation_results_06_08.csv")
OUTPUT_DIR      = Path("submissions")

# Team Informationen für die Submission Meta Daten
TEAM_NAME   = "Split n' Simulate"
RUN_NAME    = "Split-n-Simulate-run2"
DESCRIPTION = (
    "Persona  based user simulation using llama3.2. "
    "Each session is classified into one of 8 personas via a single session LLM classifier. "
    "The LLM then simulates the persona to predict the next query, "
    "using clicked results as primary context (clicks first approach). "
    "5 candidate queries are generated per session and ranked by cosine similarity."
)

# ── Duplikate finden ──────────────────────────────────────────────────────────

def find_duplicate_sessions(df: pd.DataFrame) -> list:
    """
    Gibt alle Session-IDs zurück die mindestens eine doppelte Prediction haben.
    Duplikate werden von RDS (Rank-Diversity Score) hart bestraft.
    """
    dup_sessions = []
    for session_id, group in df.groupby("session_id"):
        preds = group["predicted_next"].tolist()
        if len(preds) != len(set(p.strip().lower() for p in preds)):
            dup_sessions.append(session_id)
    return dup_sessions

# ── Schritt 1: Duplikate aus CSV entfernen ────────────────────────────────────

def clean_duplicates():
    """
    Entfernt Sessions mit doppelten Predictions aus der Simulation-CSV.
    Danach kann longeval_simulation_train.py die fehlenden Sessions neu generieren.
    """
    if not SIMULATION_FILE.exists():
        print(f"Fehler: {SIMULATION_FILE} nicht gefunden.")
        return

    df = pd.read_csv(SIMULATION_FILE)
    print(f"CSV geladen: {len(df)} Zeilen, {df['session_id'].nunique()} Sessions")

    dup_sessions = find_duplicate_sessions(df)

    if not dup_sessions:
        print("\nKeine Duplikate gefunden — CSV ist sauber!")
        print("Weiter mit: python longeval_evaluation.py")
        return

    print(f"\n{len(dup_sessions)} Sessions mit Duplikaten gefunden:")
    for sid in dup_sessions:
        group = df[df['session_id'] == sid]
        preds = group["predicted_next"].tolist()
        dups  = [p for p in set(preds) if preds.count(p) > 1]
        print(f"  Session {sid}: {dups}")

    # Duplikat Sessions aus der CSV entfernen
    df_clean = df[~df["session_id"].isin(dup_sessions)]
    df_clean.to_csv(SIMULATION_FILE, index=False)

    print(f"\nEntfernt:     {len(dup_sessions)} Sessions ({len(dup_sessions) * 5} Zeilen)")
    print(f"Verbleibend:  {df_clean['session_id'].nunique()} Sessions")
    print(f"\nNächster Schritt:")
    print(f"  python longeval_simulation_train.py")
    print(f"  → Generiert nur die {len(dup_sessions)} fehlenden Sessions neu")

# ── Schritt 2: Submission JSON erstellen ──────────────────────────────────────

def create_submission(snapshot: int):
    """
    Erstellt die snapshot-X.jsonl Datei im offiziellen LongEval Format.
    Das Ranking (Rank 1 = beste Prediction) kommt aus der Evaluation CSV.

    snapshot 1 → Training-Datensatz (snapshot-1.jsonl)
    snapshot 2 → Test-Datensatz 1   (snapshot-2.jsonl)
    snapshot 3 → Test-Datensatz 2   (snapshot-3.jsonl)
    """
    snapshot_info = {
        1: "Training-Datensatz",
        2: "Test-Datensatz 1 (task3_longeval_usim-sessions-06-08_2025.csv)",
        3: "Test-Datensatz 2 (task3_longeval_usim-sessions-09-11_2025.csv)",
    }

    if not SIMULATION_FILE.exists():
        print(f"Fehler: {SIMULATION_FILE} nicht gefunden.")
        return

    if not EVALUATION_FILE.exists():
        print(f"Fehler: {EVALUATION_FILE} nicht gefunden.")
        print("Bitte zuerst longeval_evaluation.py ausführen.")
        return

    sim     = pd.read_csv(SIMULATION_FILE)
    eval_df = pd.read_csv(EVALUATION_FILE)

    # Nochmal auf Duplikate prüfen, Sicherheitsnetz
    sim_clean    = sim[sim["predicted_next"] != "ERROR"]
    dup_sessions = find_duplicate_sessions(sim_clean)
    if dup_sessions:
        print(f"Warnung: {len(dup_sessions)} Sessions haben noch Duplikate.")
        print("Bitte zuerst --clean ausführen und Simulation neu starten.")
        return

    # Output Ordner erstellen
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"snapshot-{snapshot}.jsonl"

    print(f"Erstelle {output_file} ({snapshot_info[snapshot]})...")
    print(f"Sessions: {eval_df['session_id'].nunique()}")
    print()

    # Submission als ein einziges JSON Objekt aufbauen
    # Format: {"meta": {...}, "0": [...], "1": [...], ...}
    submission = {
        "meta": {
            "team_name":   TEAM_NAME,
            "run_name":    RUN_NAME,
            "description": DESCRIPTION,
        }
    }

    for _, row in eval_df.iterrows():
        session_id = str(int(row["session_id"]))

        # Ranked predictions aus der Evaluation CSV lesen
        # Die Evaluation hat sie bereits nach Cosine Similarity sortiert
        ranked_preds = [
            p.strip()
            for p in str(row["ranked_predictions"]).split(" | ")
        ]

        # Duplikate als letztes Sicherheitsnetz entfernen
        seen         = []
        unique_preds = []
        for p in ranked_preds:
            if p.lower() not in seen:
                seen.append(p.lower())
                unique_preds.append(p)

        submission[session_id] = unique_preds

    # Als eine einzige JSON Datei speichern (mit .jsonl Endung wie LongEval verlangt)
    # Sessions nach ID sortiert
    ordered = {"meta": submission["meta"]}
    for sid in sorted(
        [k for k in submission if k != "meta"],
        key=lambda x: int(x)
    ):
        ordered[sid] = submission[sid]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    print(f"Fertig! Submission gespeichert: {output_file}")
    print(f"Sessions:    {len(ordered) - 1}")
    print(f"Format:      {len(list(ordered.values())[1])} Predictions pro Session (Rank 1 zuerst)")
    print()

    # Vorschau der ersten 3 Sessions
    print("Vorschau (erste 3 Sessions):")
    session_items = [(k, v) for k, v in ordered.items() if k != "meta"]
    for sid, preds in session_items[:3]:
        print(f"  Session {sid}:")
        for j, pred in enumerate(preds, 1):
            print(f"    Rank {j}: {pred}")
    print()
    print("Nächster Schritt:")
    print(f"  Alle drei snapshot-Dateien als ZIP auf TIRA hochladen.")

# ── Argument Parser ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongEval Submission Generator — erstellt snapshot-X.jsonl für TIRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kompletter Ablauf:

  1. Duplikate bereinigen:
     python longeval_submission.py --clean

  2. Simulation neu starten (füllt fehlende Sessions auf):
     python longeval_simulation_train.py

  3. Evaluation mit Ranking:
     python longeval_evaluation.py

  4. Submission erstellen:
     python longeval_submission.py --submit --snapshot 1

  5. Für Test-Datensätze (nach Download):
     python longeval_submission.py --submit --snapshot 2
     python longeval_submission.py --submit --snapshot 3

  6. Alle drei Dateien als ZIP auf TIRA hochladen.
        """
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Duplikat-Sessions aus der Simulation-CSV entfernen"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="snapshot-X.jsonl Submission erstellen"
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Snapshot-Nummer: 1=Train, 2=Test1, 3=Test2 (default: 1)"
    )

    args = parser.parse_args()

    if args.clean:
        clean_duplicates()
    elif args.submit:
        create_submission(snapshot=args.snapshot)
    else:
        parser.print_help()
