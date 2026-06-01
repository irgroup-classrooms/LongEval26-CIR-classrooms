# LongEval Evaluation — Snapshot 3 (09-11 2025)
# Kübra & Emirhan, 2026
# ============================================================
#
# Änderungen gegenüber longeval_evaluation.py:
# --------------------------------------------
# - Input:  longeval_simulation_09_11.csv
# - Output: longeval_evaluation_results_09_11.csv
# - Kein Groundtruth verfügbar (Testdatensatz) →
#   Ranking basiert auf Cosine Similarity zur letzten
#   bekannten Query der Session (= letzte Query in der CSV,
#   die dem Modell als Kontext vorlag)
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Sentence-BERT wird nur geladen wenn es installiert ist,
# falls nicht, läuft das Skript trotzdem mit Cosine Only.
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

# Wo liegt der Output aus der Simulation?
INPUT_FILE  = Path("longeval_simulation_09_11.csv")

# Wo sollen die Ergebnisse gespeichert werden?
OUTPUT_FILE = Path("longeval_evaluation_results_09_11.csv")

# Welches Sentence-BERT Modell soll verwendet werden?
# "all-MiniLM-L6-v2" ist klein, schnell und gut für kurze Texte wie Suchanfragen.
# Wird beim ersten Aufruf automatisch heruntergeladen (~80 MB).
SBERT_MODEL = "all-MiniLM-L6-v2"

# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("LONGEVAL EVALUATION")
print("=" * 60)

# Zuerst prüfen wir ob die Datei überhaupt existiert
if not INPUT_FILE.exists():
    print(f"Error: {INPUT_FILE} not found!")
    print("Please run longeval_simulation_train.py first.")
    exit()

# CSV laden
df = pd.read_csv(INPUT_FILE)
print(f"Rows loaded:     {len(df)}")
print(f"Sessions:        {df['session_id'].nunique()}")
print(f"Predictions:     {df['prediction_number'].nunique()} per session")
print()

# Kein Groundtruth im Testdatensatz, wir nehmen die letzte bekannte Query
# aus session_queries als Referenz für das Ranking.
# session_queries enthält alle Queries die dem Modell bekannt waren,
# getrennt durch " | ". Die letzte davon ist unser Ranking Anker.
if "session_queries" not in df.columns:
    print("Error: column 'session_queries' not found!")
    exit()

df["reference_query"] = df["session_queries"].apply(
    lambda x: str(x).split(" | ")[-1].strip()
)
print("Reference for ranking: last known query per session (no groundtruth available)")
print()

# ERROR Zeilen rausfiltern,die können wir nicht auswerten
errors_before = len(df)
df = df[df["predicted_next"] != "ERROR"]
error_count = errors_before - len(df)
if error_count > 0:
    print(f"Note: {error_count} ERROR entries were skipped.")
    print()

# ── Sentence-BERT laden ───────────────────────────────────────────────────────
#
# Sentence-BERT wandelt Texte in semantische Vektoren um.
# Das Modell versteht Bedeutung, nicht nur Wörter.
# Wird einmalig geladen und dann für alle Sessions verwendet.

sbert_model = None
if SBERT_AVAILABLE:
    print("Loading Sentence-BERT model (may download ~80 MB on first run)...")
    try:
        sbert_model = SentenceTransformer(SBERT_MODEL)
        print(f"Sentence-BERT loaded: {SBERT_MODEL}")
    except Exception as e:
        print(f"Warning: Could not load Sentence-BERT: {e}")
        print("Continuing with Cosine Similarity only.")
    print()
else:
    print("Note: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    print("Continuing with Cosine Similarity only.")
    print()

# ── Similarity berechnen ──────────────────────────────────────────────────────
#
# Wie funktioniert Cosine Similarity?
# ------------------------------------
# Stell dir vor, jedes Wort ist eine Dimension im Raum.
# Jede Query wird als Vektor (Pfeil) in diesem Raum dargestellt.
# Die Cosine Similarity misst den Winkel zwischen zwei Pfeilen:
# - Gleiche Richtung (0°)  = Wert 1.0 = sehr ähnlich
# - Andere Richtung  (90°) = Wert 0.0 = nicht ähnlich
#
# Wie funktioniert Sentence-BERT?
# --------------------------------
# Statt einzelner Wörter werden ganze Sätze als Vektoren kodiert.
# Das Modell wurde darauf trainiert ähnliche Bedeutungen
# nah beieinander im Vektorraum zu platzieren, egal ob
# die genauen Wörter übereinstimmen oder nicht.

print("Calculating similarities for all predictions...")
print("(This may take a moment)")
print()

results = []

# Session für Session durchgehen
for session_id, group in df.groupby("session_id"):

    # Letzte bekannte Query als Ranking Referenz (kein Groundtruth im Testdatensatz)
    reference   = group["reference_query"].iloc[0]
    persona     = group["persona"].iloc[0]
    predictions = group["predicted_next"].tolist()

    similarities = []

    for pred_num, prediction in enumerate(predictions, 1):

        # ── Cosine Similarity (TF-IDF) ────────────────────────────────────
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([reference, prediction])
            cosine_score = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )[0][0]
        except Exception:
            cosine_score = 0.0

        # ── Sentence-BERT Similarity ──────────────────────────────────────
        # Nur berechnen wenn das Modell geladen werden konnte
        sbert_score = None
        if sbert_model is not None:
            try:
                embeddings  = sbert_model.encode([reference, prediction])
                sbert_score = float(
                    util.cos_sim(embeddings[0], embeddings[1])[0][0]
                )
                sbert_score = round(sbert_score, 4)
            except Exception:
                sbert_score = 0.0

        similarities.append({
            "prediction_number": pred_num,
            "prediction":        prediction,
            "cosine":            round(cosine_score, 4),
            "sbert":             sbert_score,
        })

    # ── Ranking der 5 Predictions ─────────────────────────────────────────
    # Die Predictions werden nach Cosine Similarity zur letzten bekannten
    # Query sortiert: Rank 1 = ähnlichste Prediction, Rank 5 = unähnlichste.
    ranked = sorted(similarities, key=lambda x: x["cosine"], reverse=True)
    for rank, s in enumerate(ranked, 1):
        s["rank"] = rank

    best_cosine = ranked[0]

    avg_cosine = round(
        sum(s["cosine"] for s in similarities) / len(similarities), 4
    )

    row = {
        "session_id":          session_id,
        "persona":             persona,
        "reference_query":     reference,
        "best_prediction":     best_cosine["prediction"],
        "best_pred_number":    best_cosine["prediction_number"],
        "best_similarity":     best_cosine["cosine"],
        "avg_similarity":      avg_cosine,
        # Predictions in Submission Reihenfolge (Rank 1 zuerst)
        "ranked_predictions":  " | ".join(s["prediction"] for s in ranked),
        "all_similarities":    " | ".join(
            f"P{s['prediction_number']}={s['cosine']}(Rank{s['rank']})"
            for s in similarities
        ),
    }

    # Sentence-BERT Werte hinzufügen wenn verfügbar
    if sbert_model is not None:
        best_sbert  = max(similarities, key=lambda x: x["sbert"] or 0)
        avg_sbert   = round(
            sum((s["sbert"] or 0) for s in similarities) / len(similarities), 4
        )
        row["best_sbert_similarity"] = best_sbert["sbert"]
        row["avg_sbert_similarity"]  = avg_sbert
        row["all_sbert_similarities"] = " | ".join(
            f"P{s['prediction_number']}={s['sbert']}" for s in similarities
        )

    results.append(row)

# ── Ergebnisse speichern ──────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved: {OUTPUT_FILE}")
print()

# ── Gesamtstatistik ───────────────────────────────────────────────────────────

total_sessions = len(results_df)
overall_best   = results_df["best_similarity"].mean()
overall_avg    = results_df["avg_similarity"].mean()

print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)
print(f"Sessions evaluated:           {total_sessions}")
print()
print("── Cosine Similarity (TF-IDF) — offizielle LongEval Metrik ──")
print(f"  Avg best similarity:        {overall_best:.4f}")
print(f"  Avg similarity across all:  {overall_avg:.4f}")

if "best_sbert_similarity" in results_df.columns:
    sbert_best = results_df["best_sbert_similarity"].mean()
    sbert_avg  = results_df["avg_sbert_similarity"].mean()
    print()
    print("── Sentence-BERT — semantische Ähnlichkeit ──")
    print(f"  Avg best similarity:        {sbert_best:.4f}")
    print(f"  Avg similarity across all:  {sbert_avg:.4f}")
    print()
    # Zeigt wie viel besser Sentence-BERT die Predictions bewertet
    diff = sbert_best - overall_best
    print(f"  Sentence-BERT liegt {diff:+.4f} über Cosine Similarity")
    print(f"  → Das zeigt wie viele Predictions semantisch korrekt sind,")
    print(f"    aber andere Wörter als der Groundtruth verwenden.")
print()

# Einordnung
print("Assessment (Cosine):")
if overall_best >= 0.7:
    print("  -> Great! Predictions frequently match the last known query.")
elif overall_best >= 0.4:
    print("  -> Good. Predictions are thematically close to the last known query.")
elif overall_best >= 0.2:
    print("  -> Okay. There are some thematic overlaps.")
else:
    print("  -> Needs improvement. Predictions often differ from the last known query.")
print()

# ── Welche Prediction-Nummer trifft am häufigsten? ────────────────────────────
#
# Interessante Frage: Ist Prediction 1 immer die beste,
# oder trifft manchmal auch Prediction 4 oder 5 am besten?

print("=" * 60)
print("WHICH PREDICTION NUMBER WINS MOST OFTEN?")
print("=" * 60)
print("(Shows whether the first or a later prediction fits best)")
print()

distribution = results_df["best_pred_number"].value_counts().sort_index()
for num, count in distribution.items():
    bar = "█" * int(count * 20 / total_sessions)
    print(f"  Prediction {num}:  {count:>3} sessions  {bar}")
print()

# ── Ergebnisse pro Persona ────────────────────────────────────────────────────
#
# Für welche Persona funktioniert die Simulation am besten?

print("=" * 60)
print("RESULTS PER PERSONA")
print("=" * 60)
print()

agg_cols = {
    "sessions":  ("session_id", "count"),
    "avg_best":  ("best_similarity", "mean"),
    "avg_all":   ("avg_similarity", "mean"),
}
if "best_sbert_similarity" in results_df.columns:
    agg_cols["avg_sbert"] = ("best_sbert_similarity", "mean")

persona_stats = results_df.groupby("persona").agg(**agg_cols).sort_values(
    "avg_best", ascending=False
)

for persona, row in persona_stats.iterrows():
    bar = "█" * int(row["avg_best"] * 20)
    line = (f"  {persona:<42}  Sessions: {int(row['sessions']):>3}  |  "
            f"Cosine: {row['avg_best']:.4f}")
    if "avg_sbert" in row:
        line += f"  |  SBERT: {row['avg_sbert']:.4f}"
    line += f"  {bar}"
    print(line)
print()

# ── Top 5 beste Sessions ──────────────────────────────────────────────────────

print("=" * 60)
print("TOP 5 - BEST SESSIONS")
print("=" * 60)
print("(Sessions where the prediction was closest to the last known query)")
print()

for _, row in results_df.nlargest(5, "best_similarity").iterrows():
    print(f"  Session {int(row['session_id'])} ({row['persona']})")
    print(f"    Last known query:      {row['reference_query']}")
    print(f"    Best prediction:  {row['best_prediction']}")
    print(f"    Cosine:           {row['best_similarity']}", end="")
    if "best_sbert_similarity" in results_df.columns:
        print(f"  |  SBERT: {row['best_sbert_similarity']}", end="")
    print()
    print()

# ── Bottom 5 schwierigste Sessions ───────────────────────────────────────────

print("=" * 60)
print("BOTTOM 5 - HARDEST SESSIONS")
print("=" * 60)
print("(Sessions where the prediction was furthest from the last known query)")
print()

for _, row in results_df.nsmallest(5, "best_similarity").iterrows():
    print(f"  Session {int(row['session_id'])} ({row['persona']})")
    print(f"    Last known query:      {row['reference_query']}")
    print(f"    Best prediction:  {row['best_prediction']}")
    print(f"    Cosine:           {row['best_similarity']}", end="")
    if "best_sbert_similarity" in results_df.columns:
        print(f"  |  SBERT: {row['best_sbert_similarity']}", end="")
    print()
    print()

print("=" * 60)
print(f"Done! All results saved in: {OUTPUT_FILE}")
print("=" * 60)

# ── Installationshinweis ──────────────────────────────────────────────────────
#
# Falls Sentence-BERT noch nicht installiert ist:
# pip install sentence-transformers
# Das Modell (~80 MB) wird beim ersten Aufruf automatisch heruntergeladen.

if not SBERT_AVAILABLE:
    print()
    print("To enable Sentence-BERT similarity:")
    print("  pip install sentence-transformers")
    print("  Then re-run this script.")
