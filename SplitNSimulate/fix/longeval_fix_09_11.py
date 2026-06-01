"""
longeval_fix_09_11.py — Bereinigt longeval_simulation_09_11.csv

Problem 1: 15 Sessions mit doppelten Predictions
Problem 2: 11 Predictions mit Dokument-IDs

Strategie:
- Duplikate: minimale semantisch sinnvolle Variation
- Dokument-IDs: Zahl entfernen, thematisch passende Ergänzung
"""

import pandas as pd
import re
import shutil
from pathlib import Path

INPUT_FILE  = Path("longeval_simulation_09_11.csv")
BACKUP_FILE = Path("longeval_simulation_09_11.csv.bak")

# ─── Manuelle Fixes ───────────────────────────────────────────────────────────

MANUAL_FIXES = {

    # ── Session 4: stress personality (3x identisch + Pred 4) ────────────────
    # [1][2][3][4] alle "stress personality traits"
    (4, 2): "stress and personality disorders",
    (4, 3): "stress coping personality types",
    (4, 4): "extraversion stress resilience research",

    # ── Session 24: household fire extinguisher ───────────────────────────────
    # [4][5] identisch: "household fire extinguisher agents"
    (24, 5): "baking soda chemical fire suppression",

    # ── Session 30: lemon shoe polish authors ─────────────────────────────────
    # [3][5] identisch: "lemon shoe polish authors"
    (30, 5): "lemon shoe polish effectiveness study",

    # ── Session 65: phase contrast optimization ───────────────────────────────
    # [2][5] identisch: "phase contrast optimization techniques"
    (65, 5): "cellphone microscope phase contrast methods",

    # ── Session 69: food waste reduction (3x identisch + Pred 4) ────────────
    # [1][2][4][5] alle "food waste reduction strategies"
    (69, 2): "food cooperative community health impact",
    (69, 4): "food coop impact local economy",
    (69, 5): "rural food system sustainability",

    # ── Session 70: hcv treatment options ────────────────────────────────────
    # [4][5] identisch: "hcv treatment options"
    (70, 5): "hepatitis c virus clinical management",

    # ── Session 93: spinal tuberculosis treatment ─────────────────────────────
    # [2][4] identisch: "spinal tuberculosis treatment options"
    (93, 4): "tuberculosis bone infection therapy",

    # ── Session 108: energy efficient wireless sensor network ─────────────────
    # [3][5] identisch: "energy efficient wireless sensor network protocols"
    (108, 5): "multihop wireless mac protocol design",

    # ── Session 124: vr boids optimization ───────────────────────────────────
    # [1][3] identisch: "vr boids optimization"
    (124, 3): "boids flocking behavior virtual reality",

    # ── Session 125: tennyson poetry analysis ────────────────────────────────
    # [2][4] identisch: "tennyson poetry analysis"
    (125, 4): "tennyson victorian literature themes",

    # ── Session 132: cross layer optimization ────────────────────────────────
    # [1][5] identisch: "cross layer optimization algorithms"
    (132, 5): "voip cross layer wifi performance",

    # ── Session 137: rammed earth insulation cold climates ───────────────────
    # [1][4] identisch: "rammed earth insulation for cold climates"
    (137, 4): "rammed earth thermal mass properties",

    # ── Session 146: visual design principles ────────────────────────────────
    # [1][4] identisch: "visual design principles"
    (146, 4): "visual identity brand guidelines",

    # ── Session 159: asp optimization techniques (3x identisch) ──────────────
    # [2][3][4] alle "asp optimization techniques"
    (159, 3): "answer set programming solving methods",
    (159, 4): "asp constraint satisfaction approaches",

    # ── Session 160: quantum decoherence ─────────────────────────────────────
    # [2][3] identisch: "quantum decoherence"
    (160, 3): "quantum to classical transition mechanisms",

    # ── Doc-ID Fixes ──────────────────────────────────────────────────────────

    # Session 43 / Pred 5: "author of 8365390"
    # Context: law and magic corcos
    (43, 5): "law and magic corcos author works",

    # Session 84 / Pred 2: "type 2 diabetes 48105747"
    # Context: pill burden type 2 diabetes germany
    (84, 2): "type 2 diabetes pill burden management",

    # Session 84 / Pred 3: "mechanism of action 48105747"
    # Context: pill burden type 2 diabetes germany
    (84, 3): "diabetes medication mechanism of action",

    # Session 105 / Pred 2: "compound 50063342"
    # Context: epidemiology in epilepsy
    (105, 2): "epilepsy compound drug treatment",

    # Session 105 / Pred 3: "properties 50063342"
    # Context: epidemiology in epilepsy
    (105, 3): "epilepsy occupational epidemiology properties",

    # Session 130 / Pred 2: "research papers by 4200554"
    # Context: software engineering measures expert opinion
    (130, 2): "software engineering measures author research",

    # Session 131 / Pred 1: "research papers by author 2449049"
    # Context: typed graph transformation systems
    (131, 1): "graph transformation systems author publications",

    # Session 144 / Pred 4: "author name 18502656"
    # Context: role of ai in society / AI and art
    (144, 4): "ai society impact author research",

    # Session 158 / Pred 1: "logic programming by 9832307"
    # Context: logic programming as a service
    (158, 1): "logic programming as a service author",

    # Session 165 / Pred 5: "tissue regeneration biomaterials 83610966"
    # Context: smart biomaterials for tissue regeneration
    (165, 5): "smart biomaterials tissue regeneration applications",

    # Session 168 / Pred 1: "author 123194 sustainable energy"
    # Context: biofuels environmental impacts / socioeconomic impacts
    (168, 1): "biofuels sustainable energy author research",
}

# ─── Validierung ──────────────────────────────────────────────────────────────

def validate_no_duplicates(df: pd.DataFrame) -> list:
    problems = []
    for sid, group in df.groupby("session_id"):
        preds = [p.strip().lower() for p in group["predicted_next"].tolist()]
        if len(preds) != len(set(preds)):
            problems.append(sid)
    return problems


def validate_no_doc_ids(df: pd.DataFrame) -> list:
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

    shutil.copy(INPUT_FILE, BACKUP_FILE)
    print(f"Backup erstellt: {BACKUP_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"CSV geladen: {len(df)} Zeilen, {df['session_id'].nunique()} Sessions\n")

    dup_before = validate_no_duplicates(df)
    did_before = validate_no_doc_ids(df)
    print(f"Vor Fix: {len(dup_before)} Sessions mit Duplikaten, {len(did_before)} Rows mit Doc-IDs\n")

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
