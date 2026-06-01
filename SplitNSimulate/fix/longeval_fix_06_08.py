"""
longeval_fix_06_08.py — Bereinigt longeval_simulation_06_08.csv

Problem 1: Sessions mit doppelten Predictions
Problem 2: Predictions mit Dokument-IDs (numerische IDs wie '8620440')

Strategie:
- Duplikate: minimale Variation durch Anhängen von Kontextwörtern
- Dokument-IDs: Zahl per Regex entfernen, Rest des Query Texts bleibt
"""

import pandas as pd
import re
import shutil
from pathlib import Path

INPUT_FILE  = Path("longeval_simulation_06_08.csv")
BACKUP_FILE = Path("longeval_simulation_06_08.csv.bak")

# ─── Manuelle Fixes ───────────────────────────────────────────────────────────

MANUAL_FIXES = {

    # ── Session 31: machine learning attribution ──────────────────────────────
    # [2] und [3] identisch: "machine learning attribution modeling"
    (31, 3): "machine learning multitouch attribution",

    # ── Session 32: convex hull ───────────────────────────────────────────────
    # [3] und [5] identisch: "convex hull algorithm implementations"
    (32, 5): "convex hull 3d construction methods",

    # ── Session 37: charles handy ─────────────────────────────────────────────
    # [3] und [4] identisch: "charles handy organisation"
    # [2] und [5] praktisch gleich: organizational/organisational culture
    (37, 4): "charles handy management theory",
    (37, 5): "charles handy leadership style",

    # ── Session 67: testmanship ───────────────────────────────────────────────
    # [1] und [5] identisch: "academic preparedness testmanship skills"
    (67, 5): "testmanship academic performance relationship",

    # ── Session 71: food safety ───────────────────────────────────────────────
    # [1] und [4] identisch: "food safety awareness topics"
    (71, 4): "food safety consumer behavior",

    # ── Session 75: memory consolidation ─────────────────────────────────────
    # [2] und [3] identisch: "memory consolidation models"
    (75, 3): "memory research computational approaches",

    # ── Session 78: pineapple peel bleach ────────────────────────────────────
    # [1] und [5] identisch: "pineapple peel bleach alternatives"
    (78, 5): "pineapple peel stain removal effectiveness",

    # ── Session 91: azo compounds ─────────────────────────────────────────────
    # [1] und [4] identisch: "azo compounds"
    # [2] und [5] identisch: "azo compound synthesis"
    (91, 4): "azo dye properties applications",
    (91, 5): "benzoic acid azo compound characteristics",

    # ── Doc-ID Fixes ──────────────────────────────────────────────────────────

    # Session 9 / Pred 1: "works by author 276484239"
    # Context: Secure Location Service / Ad Hoc Routing / MANETs
    (9, 1): "ad hoc routing protocol author works",

    # Session 24 / Pred 4: "4826783"
    # Context: intelligent seam tracking robotic welding
    (24, 4): "robotic welding seam tracking related works",

    # Session 25 / Pred 2: "works by 347483"
    # Context: job shop scheduling earliness tardiness optimization
    (25, 2): "job shop scheduling author publications",

    # Session 64 / Pred 3: "implementation details 44935411"
    # Context: AI predictive modeling seismic retrofitting
    (64, 3): "seismic retrofitting ai model implementation details",

    # Session 102 / Pred 4: "evora radiography by 153712090"
    # Context: Evora 2018 veterinary radiography exotic animals
    (102, 4): "evora veterinary radiography exotic animals author",
}

DOC_ID_PATTERN = re.compile(r'\b\d{5,}\b')

# Variationen die bei Duplikaten angehängt werden um sie eindeutig zu machen
VARIATION_SUFFIXES = [
    "research", "study", "analysis", "review", "methods",
    "literature", "overview", "findings", "applications", "techniques"
]

# ─── Validierung ──────────────────────────────────────────────────────────────

def validate_no_duplicates(df: pd.DataFrame) -> list:
    problems = []
    for sid, group in df.groupby("session_id"):
        preds = [p.strip().lower() for p in group["predicted_next"].tolist()]
        if len(preds) != len(set(preds)):
            problems.append(sid)
    return problems


def validate_no_doc_ids(df: pd.DataFrame) -> list:
    problems = []
    for _, row in df.iterrows():
        if DOC_ID_PATTERN.search(str(row["predicted_next"])):
            problems.append((row["session_id"], row["prediction_number"], row["predicted_next"]))
    return problems


# ─── Automatische Fixes ───────────────────────────────────────────────────────

def fix_doc_ids(df: pd.DataFrame) -> tuple:
    """Entfernt Dokument-IDs aus Predictions per Regex."""
    fixed = 0
    for idx, row in df.iterrows():
        pred = str(row["predicted_next"])
        if DOC_ID_PATTERN.search(pred):
            cleaned = DOC_ID_PATTERN.sub("", pred).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if not cleaned:
                cleaned = "related works"
            print(f"  Doc-ID fix | Session {row['session_id']} / Pred {row['prediction_number']}: '{pred}' → '{cleaned}'")
            df.at[idx, "predicted_next"] = cleaned
            fixed += 1
    return df, fixed


def fix_duplicates(df: pd.DataFrame) -> tuple:
    """Behebt Duplikate durch minimale Variation."""
    fixed = 0
    for sid, group in df.groupby("session_id"):
        preds = [p.strip().lower() for p in group["predicted_next"].tolist()]
        if len(preds) == len(set(preds)):
            continue

        seen = []
        suffix_idx = 0
        for idx, row in group.iterrows():
            pred_lower = row["predicted_next"].strip().lower()
            if pred_lower in seen:
                # Variation mit Suffix
                for suffix in VARIATION_SUFFIXES[suffix_idx:] + VARIATION_SUFFIXES[:suffix_idx]:
                    candidate = f"{row['predicted_next'].strip()} {suffix}"
                    if candidate.lower() not in seen:
                        print(f"  Dup fix    | Session {sid} / Pred {row['prediction_number']}: '{row['predicted_next']}' → '{candidate}'")
                        df.at[idx, "predicted_next"] = candidate
                        seen.append(candidate.lower())
                        suffix_idx = (suffix_idx + 1) % len(VARIATION_SUFFIXES)
                        fixed += 1
                        break
            else:
                seen.append(pred_lower)
    return df, fixed


# ─── Hauptlogik ───────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"FEHLER: {INPUT_FILE} nicht gefunden.")
        return

    shutil.copy(INPUT_FILE, BACKUP_FILE)
    print(f"Backup erstellt: {BACKUP_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"CSV geladen: {len(df)} Zeilen, {df['session_id'].nunique()} Sessions\n")

    # Vor Check
    dup_before = validate_no_duplicates(df)
    did_before = validate_no_doc_ids(df)
    print(f"Vor Fix: {len(dup_before)} Sessions mit Duplikaten, {len(did_before)} Rows mit Doc-IDs\n")

    # Manuelle Fixes anwenden
    applied = 0
    for (sid, pnum), new_pred in MANUAL_FIXES.items():
        mask = (df["session_id"] == sid) & (df["prediction_number"] == pnum)
        if mask.sum() == 0:
            print(f"  WARNUNG: Session {sid} / Pred {pnum} nicht gefunden!")
            continue
        old_pred = df.loc[mask, "predicted_next"].values[0]
        df.loc[mask, "predicted_next"] = new_pred
        print(f"  Session {sid} | Pred {pnum}: '{old_pred}' → '{new_pred}'")
        applied += 1

    print(f"\n{applied} Fixes angewendet.")

    # Nach Check
    dup_after = validate_no_duplicates(df)
    did_after = validate_no_doc_ids(df)
    print(f"\nNach Fix: {len(dup_after)} Sessions mit Duplikaten, {len(did_after)} Rows mit Doc-IDs")

    if dup_after:
        print(f"  VERBLEIBENDE DUPLIKATE: Sessions {dup_after}")
    if did_after:
        print(f"  VERBLEIBENDE DOC-IDs:")
        for sid, pnum, pred in did_after:
            print(f"    Session {sid} / Pred {pnum}: {pred}")

    if not dup_after and not did_after:
        print("\nAlle Probleme behoben! CSV ist sauber.")
        df.to_csv(INPUT_FILE, index=False)
        print(f"Gespeichert: {INPUT_FILE}")
    else:
        print("\nEs gibt noch Probleme — CSV wurde NICHT gespeichert.")
        print("Bitte manuell prüfen und erneut ausführen.")

if __name__ == "__main__":
    main()
