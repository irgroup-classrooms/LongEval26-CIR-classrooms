"""
longeval_fix.py — Behebt zwei Probleme in longeval_simulation_all.csv

Problem 1: 2 Sessions mit doppelten Predictions (Session 49, 160)
Problem 2: 18 Predictions mit Dokument-IDs (numerische IDs wie '8620440')

Strategie:
- Duplikate: minimale semantisch sinnvolle Variation der doppelten Prediction
- Dokument-IDs: Zahl entfernen, restlicher Query Text bleibt erhalten
  und wird ggf. leicht sinnvoll ergänzt
"""

import pandas as pd
import re
import shutil
from pathlib import Path

INPUT_FILE  = Path("longeval_simulation_all.csv")
BACKUP_FILE = Path("longeval_simulation_all.csv.bak")

# ─── Manuelle Fixes ───────────────────────────────────────────────────────────
# Format: (session_id, prediction_number) → neue Prediction

MANUAL_FIXES = {

    # ── Session 49: cassava peels / adhesive ──────────────────────────────────
    # Original: [1] cassava peels applications
    #           [2] cassava peel adhesive products      ← Duplikat mit [3]
    #           [3] cassava peel adhesive products
    #           [4] cassava peel products
    #           [5] cassava peel adhesive properties
    # Pred 3 ist der Klon → ersetzen durch thematisch nahe Variation
    (49, 3): "cassava peel adhesive strength",

    # ── Session 160: colonialism impact ──────────────────────────────────────
    # Original: [1] colonialism impact
    #           [2] colonialism impacts                ← fast identisch mit [1]
    #           [3] colonialism impact                 ← exaktes Duplikat von [1]
    #           [4] colonialism impact education
    #           [5] colonialism impact history
    # Pred 1 und 3 sind exakt gleich (case insensitive) → Pred 3 ersetzen
    # Pred 1 und 2 sind praktisch gleich (singular/plural) → Pred 2 ersetzen
    (160, 2): "colonialism effects on society",
    (160, 3): "colonialism legacy",

    # ── Doc-ID Fixes ──────────────────────────────────────────────────────────

    # Session 32 | "generateai dataset 156543015"
    # Context: generative AI ethics / automatic labelling ethics
    (32, 5): "generative ai dataset ethics",

    # Session 36 | "44681688 social dynamics"
    # Context: social dynamics in politics / mass media influence
    (36, 4): "social dynamics mass media influence",

    # Session 40 | "works by author of 59752794"
    # Context: automatic shower system literature review
    (40, 4): "automatic shower system related literature authors",

    # Session 40 | "related concepts 59752794"
    (40, 5): "automatic shower system acceptability related concepts",

    # Session 56 | "authors paper 153945799"
    # Context: artificial intelligence mixed methods research
    (56, 2): "authors artificial intelligence mixed methods research",

    # Session 69 | "cleantech cluster management 18770785"
    # Context: cleantech cluster leadership
    (69, 1): "cleantech cluster management leadership",

    # Session 71 | "163090268 works"
    # Context: zero hour / non-standard employment contracts
    (71, 2): "non-standard employment contract works",

    # Session 80 | "distribution transformer author 18478567"
    # Context: machine learning risk based screening / distribution transformer
    (80, 3): "distribution transformer machine learning author",

    # Session 84 | "protein function 49701258"
    # Context: community health / mental health first aid
    (84, 2): "protein function community health research",

    # Session 93 | "prostate cancer 159809928"
    # Context: hypofractionation in prostate cancer
    (93, 3): "prostate cancer hypofractionation study",

    # Session 118 | "industry 2877114 data"
    # Context: supply chain bottlenecks metro project
    (118, 3): "industry supply chain bottleneck data",

    # Session 159 | "discourse analysis 148452275"
    # Context: discourse analysis police interview
    (159, 3): "discourse analysis police interview study",

    # Session 163 | "2035096 works keyword search"
    # Context: think before you speak / Socrates quote
    (163, 3): "think before you speak works keyword search",

    # Session 171 | "works by 150664188"
    # Context: teachers using laptops in teaching methods
    (171, 5): "works by teachers laptops teaching methods",

    # Session 41 | "solar water pump 4471539 specs"
    # Context: solar powered water pumping machine
    (41, 2): "solar water pump technical specifications",

    # Session 175 | "study specifics 160926286"
    # Context: national study communication and social engagement
    (175, 2): "national study communication social engagement specifics",

    # Session 30 | "soil 51402990 keyword"
    # Context: best soil for plants
    (30, 3): "best soil for plants keyword search",

    # Session 154 | "patent number 11624654 keywords"
    # Context: Fraunhofer ISE CatVap technology research
    (154, 4): "fraunhofer ise catvap technology patent keywords",
}

# ─── Validierung ──────────────────────────────────────────────────────────────

def validate_no_duplicates(df: pd.DataFrame) -> list:
    """Gibt alle Sessions zurück, die noch Duplikate haben."""
    problems = []
    for sid, group in df.groupby("session_id"):
        preds = [p.strip().lower() for p in group["predicted_next"].tolist()]
        if len(preds) != len(set(preds)):
            problems.append(sid)
    return problems


def validate_no_doc_ids(df: pd.DataFrame) -> list:
    """Gibt alle Rows zurück, die noch Dokument-IDs enthalten."""
    pattern = re.compile(r'\b\d{5,}\b')
    problems = []
    for _, row in df.iterrows():
        if pattern.search(str(row["predicted_next"])):
            problems.append((row["session_id"], row["prediction_number"], row["predicted_next"]))
    return problems


# ─── Hauptlogik ───────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"FEHLER: {INPUT_FILE} nicht gefunden.")
        return

    # Backup
    shutil.copy(INPUT_FILE, BACKUP_FILE)
    print(f"Backup erstellt: {BACKUP_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"CSV geladen: {len(df)} Zeilen, {df['session_id'].nunique()} Sessions\n")

    # Vor Check
    dup_before  = validate_no_duplicates(df)
    did_before  = validate_no_doc_ids(df)
    print(f"Vor Fix: {len(dup_before)} Sessions mit Duplikaten, {len(did_before)} Rows mit Doc-IDs")

    # Fixes anwenden
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
        print("Bitte MANUAL_FIXES anpassen und erneut ausführen.")

if __name__ == "__main__":
    main()
