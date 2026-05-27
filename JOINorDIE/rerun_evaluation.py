"""
rerun_evaluation.py – Wertet alle 18 Task-3-Runs neu aus.
Gleiche Logik wie Notebook 09, aber als einfaches Skript.

Starten mit:  python rerun_evaluation.py
Ausgabe:      data/task3_evaluation_results.csv  (wird überschrieben)
              data/task3_evaluation_chart.png    (wird überschrieben)
"""

import os, json, zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR     = Path(__file__).parent
RUNFILES_DIR = BASE_DIR / 'runfiles_task3'

SNAPSHOT_CSVs = {
    'snapshot-1.jsonl': BASE_DIR / 'predetermined_queries_Task_A_test.csv',
    'snapshot-2.jsonl': BASE_DIR / 'task3_longeval_usim-sessions-06-08_2025.csv',
    'snapshot-3.jsonl': BASE_DIR / 'task3_longeval_usim-sessions-09-11_2025.csv',
}

# ── Ground Truth laden ─────────────────────────────────────────────────────────
def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['idx','user','session_id','query','timestamp',
                  'retrieved_docs','session_hash','interactions']
    sessions = defaultdict(list)
    for _, row in df.sort_values('timestamp').iterrows():
        sessions[str(row['session_id'])].append(str(row['query']))
    return {sid: qs[-1] for sid, qs in sessions.items() if len(qs) > 1}

print('Lade Ground Truth...')
ground_truth = {}
for snap_name, csv_path in SNAPSHOT_CSVs.items():
    gt = load_ground_truth(csv_path)
    ground_truth[snap_name] = gt
    print(f'  {snap_name}: {len(gt)} Sessions')

# ── SBERT laden ────────────────────────────────────────────────────────────────
print('\nLade SBERT-Modell (all-MiniLM-L6-v2)...')
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    USE_SBERT = True
    print('  SBERT geladen ✓')
except ImportError:
    USE_SBERT = False
    print('  WARNUNG: sentence-transformers nicht installiert – nur Cosine wird berechnet.')
    print('  Installation: pip install sentence-transformers')

# ── Metriken ───────────────────────────────────────────────────────────────────
def compute_metrics(predicted_list, gt_list):
    if not predicted_list:
        return 0.0, 0.0, 0.0, 0.0
    # TF-IDF Cosine
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(predicted_list + gt_list)
    n = len(predicted_list)
    cos_scores = [cosine_similarity(tfidf[i], tfidf[n+i])[0][0] for i in range(n)]
    # SBERT
    if USE_SBERT:
        pe = sbert_model.encode(predicted_list, convert_to_numpy=True, show_progress_bar=False)
        ge = sbert_model.encode(gt_list,        convert_to_numpy=True, show_progress_bar=False)
        sbert_scores = [float(cosine_similarity([pe[i]], [ge[i]])[0][0]) for i in range(n)]
    else:
        sbert_scores = [0.0] * n
    return (
        round(float(np.mean(cos_scores)),    4),
        round(float(np.std(cos_scores)),     4),
        round(float(np.mean(sbert_scores)),  4),
        round(float(np.std(sbert_scores)),   4),
    )

# ── Runs einlesen und evaluieren ───────────────────────────────────────────────
def load_run(zip_path):
    result = {}
    with zipfile.ZipFile(zip_path) as zf:
        for snap_name in ['snapshot-1.jsonl','snapshot-2.jsonl','snapshot-3.jsonl']:
            if snap_name not in zf.namelist(): continue
            data = json.loads(zf.read(snap_name))
            result[snap_name] = {
                str(sid): str(val[0])
                for sid, val in data.items()
                if sid != 'meta' and isinstance(val, list) and val
            }
    return result

zip_files = sorted(RUNFILES_DIR.glob('task3_submission_*.zip'))
print(f'\nGefundene Runs: {len(zip_files)}')

results = []
for zip_path in zip_files:
    run_name = zip_path.stem.replace('task3_submission_', '')
    print(f'  Evaluiere {run_name} ...', end=' ', flush=True)
    run_data = load_run(zip_path)
    all_pred, all_gt = [], []
    for snap_name, gt_dict in ground_truth.items():
        snap_preds = run_data.get(snap_name, {})
        for sid, gt_q in gt_dict.items():
            if sid in snap_preds:
                all_pred.append(snap_preds[sid])
                all_gt.append(gt_q)
    cm, cs, sm, ss = compute_metrics(all_pred, all_gt)
    results.append({'Run': run_name, 'Matches': len(all_pred),
                    'Cosine Mean': cm, 'Cosine Std': cs,
                    'SBERT Mean': sm, 'SBERT Std': ss})
    print(f'Cosine={cm:.4f}, SBERT={sm:.4f}  ({len(all_pred)} Sessions)')

results_df = pd.DataFrame(results).sort_values('SBERT Mean', ascending=False).reset_index(drop=True)

# ── Speichern ──────────────────────────────────────────────────────────────────
out_csv = BASE_DIR / 'data' / 'task3_evaluation_results.csv'
results_df.to_csv(out_csv, index=False)
print(f'\nGespeichert: {out_csv}')

# ── Zusammenfassung ────────────────────────────────────────────────────────────
results_df['Prompt'] = results_df['Run'].str.extract(r'_(A|B)_')
results_df['Mode']   = results_df['Run'].str.extract(r'openai_(first|last|all)_')

print('\n── Mittelwerte nach Prompt ──')
print(results_df.groupby('Prompt')[['Cosine Mean','SBERT Mean']].mean().round(4))

print('\n── Mittelwerte nach Query-Modus ──')
print(results_df.groupby('Mode')[['Cosine Mean','SBERT Mean']].mean().round(4))

# ── Chart ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    df_plot   = results_df.sort_values('SBERT Mean', ascending=True)
    short_names = df_plot['Run'].str.replace('openai_', '', regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(short_names, df_plot['Cosine Mean'], xerr=df_plot['Cosine Std'],
                 color='#2E5496', alpha=0.85, capsize=4)
    axes[0].set_xlabel('Cosine Mean')
    axes[0].set_title('Cosine-Ähnlichkeit (TF-IDF) – 18 Runs')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].barh(short_names, df_plot['SBERT Mean'], xerr=df_plot['SBERT Std'],
                 color='#C00000', alpha=0.85, capsize=4)
    axes[1].set_xlabel('SBERT Mean')
    axes[1].set_title('SBERT-Ähnlichkeit – 18 Runs')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    chart_path = BASE_DIR / 'data' / 'task3_evaluation_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart gespeichert: {chart_path}')
except Exception as e:
    print(f'Chart konnte nicht erstellt werden: {e}')

print('\nFertig! Bitte CSV-Werte für Berichts-Update bereitstellen.')
