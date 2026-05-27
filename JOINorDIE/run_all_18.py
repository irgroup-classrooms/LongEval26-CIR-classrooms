"""
run_all_18.py – Alle 18 Task-3-Runs automatisch nacheinander ausführen.

Starten mit:  python run_all_18.py
Abbruch:      Strg+C  (bereits fertige Runs bleiben gespeichert)

Reihenfolge: erst alle 9 Prompt-A-Runs (kein doc_lookup nötig),
             dann alle 9 Prompt-B-Runs (doc_lookup wird einmalig geladen).

Snapshot-Strategie:
  - snapshot-1 wird bei jedem Run NEU generiert (neuer offizieller Trainingsdatensatz)
  - snapshot-2 & -3 werden aus dem bestehenden ZIP übernommen, falls vorhanden
    (spart API-Kosten, da diese Snapshots bereits korrekt generiert wurden)
"""

import os, json, time, re, zipfile, ast
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

# ── Setup ─────────────────────────────────────────────────────────────────────
for folder in [Path(__file__).parent, *Path(__file__).parent.parents]:
    if (folder / '.env').exists():
        load_dotenv(folder / '.env', override=True)
        break

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip().strip('"').strip("'")
if not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY fehlt! Bitte in .env Datei eintragen.')
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / 'runfiles_task3'
OUTPUT_DIR.mkdir(exist_ok=True)

# snapshot-1 = neuer offizieller Trainingsdatensatz
# snapshot-2 & -3 = Test-Snapshots (werden aus bestehendem ZIP übernommen)
SNAPSHOT_FILES = {
    'snapshot-1.jsonl': str(BASE_DIR / 'task3_longeval_usim-sessions-train.csv'),
    'snapshot-2.jsonl': str(BASE_DIR / 'task3_longeval_usim-sessions-06-08_2025.csv'),
    'snapshot-3.jsonl': str(BASE_DIR / 'task3_longeval_usim-sessions-09-11_2025.csv'),
}

# Prüfen ob snapshot-1 CSV vorhanden ist (das ist das einzige was wir neu brauchen)
snap1_csv = SNAPSHOT_FILES['snapshot-1.jsonl']
if not Path(snap1_csv).exists():
    print(f'FEHLER – Datei fehlt: {snap1_csv}')
    print('Bitte task3_longeval_usim-sessions-train.csv in diesen Ordner kopieren.')
    raise SystemExit(1)

MODEL        = 'gpt-4o-mini'
TEAM_NAME    = 'JOINorDIE'
DESCRIPTION  = 'LLM-based next query prediction using OpenAI GPT-4o-mini.'
MAX_DOC_CHARS = 600
N_REL_DOCS   = 3
N_NONREL_DOCS = 3

# ── 18 Run-Konfigurationen ────────────────────────────────────────────────────
# Erst alle Prompt-A-Runs (kein doc_lookup), dann alle Prompt-B-Runs
ALL_RUNS = [
    # Prompt A
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_first_A_rel'},
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_first_A_non_rel'},
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_first_A_contrastive'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_all_A_rel'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_all_A_non_rel'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_all_A_contrastive'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_last_A_rel'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_last_A_non_rel'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_last_A_contrastive'},
    # Prompt B
    {'query_mode': 'first', 'prompt_variant': 'B', 'doc_variant': 'rel',         'run_name': 'openai_first_B_rel'},
    {'query_mode': 'first', 'prompt_variant': 'B', 'doc_variant': 'non_rel',     'run_name': 'openai_first_B_non_rel'},
    {'query_mode': 'first', 'prompt_variant': 'B', 'doc_variant': 'contrastive', 'run_name': 'openai_first_B_contrastive'},
    {'query_mode': 'all',   'prompt_variant': 'B', 'doc_variant': 'rel',         'run_name': 'openai_all_B_rel'},
    {'query_mode': 'all',   'prompt_variant': 'B', 'doc_variant': 'non_rel',     'run_name': 'openai_all_B_non_rel'},
    {'query_mode': 'all',   'prompt_variant': 'B', 'doc_variant': 'contrastive', 'run_name': 'openai_all_B_contrastive'},
    {'query_mode': 'last',  'prompt_variant': 'B', 'doc_variant': 'rel',         'run_name': 'openai_last_B_rel'},
    {'query_mode': 'last',  'prompt_variant': 'B', 'doc_variant': 'non_rel',     'run_name': 'openai_last_B_non_rel'},
    {'query_mode': 'last',  'prompt_variant': 'B', 'doc_variant': 'contrastive', 'run_name': 'openai_last_B_contrastive'},
]

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def parse_docnos(val):
    try:
        return [str(x) for x in ast.literal_eval(str(val))]
    except:
        return []

def parse_interactions(val):
    try:
        items = ast.literal_eval(str(val))
        return [str(item[0]) for item in items if isinstance(item, (list, tuple))]
    except:
        return []

def load_sessions(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['idx','user','session_id','query','timestamp',
                  'retrieved_docs','session_hash','interactions']
    sessions = defaultdict(lambda: {'queries': [], 'rel': set(), 'nonrel': set()})
    for _, row in df.sort_values('timestamp').iterrows():
        sid = str(row['session_id'])
        sessions[sid]['queries'].append(str(row['query']))
        retrieved  = parse_docnos(row['retrieved_docs'])
        interacted = set(parse_interactions(row['interactions']))
        sessions[sid]['rel'].update(interacted)
        sessions[sid]['nonrel'].update(d for d in retrieved if d not in interacted)
    return {
        sid: {'queries':       d['queries'],
              'rel_docnos':    list(d['rel']),
              'nonrel_docnos': list(d['nonrel'])}
        for sid, d in sessions.items()
    }

def get_doc_text(docno, doc_lookup):
    d = doc_lookup.get(str(docno), {})
    t = (d.get('title', '') or '').strip()
    a = (d.get('abstract', '') or '').strip()
    text = ('Title: ' + t + '\nAbstract: ' + a) if (t or a) else ('Document ID: ' + docno)
    return text[:MAX_DOC_CHARS]

def get_context(queries, mode):
    ctx = queries[:-1] if len(queries) > 1 else queries
    if mode == 'first': return ctx[0] if ctx else queries[0]
    if mode == 'last':  return ctx[-1] if ctx else queries[0]
    return '\n'.join(ctx)

# ── Topic-Generierung (on-the-fly, Abgabe-Ansatz) ────────────────────────────
topic_cache = {}

def get_topic(sid, all_queries):
    if sid in topic_cache:
        return topic_cache[sid]
    queries_text = '\n'.join(all_queries[:-1] if len(all_queries) > 1 else all_queries)
    prompt = (
        'You are an expert information retrieval analyst. '
        'Create a TREC-style search topic from these search queries. '
        'Return ONLY valid JSON with keys: title, description, narrative.\n\n'
        'Search queries: ' + queries_text
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3, max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                t = str(data.get('title', queries_text[:30])).strip()
                d = str(data.get('description', '')).strip()
                n = str(data.get('narrative', '')).strip()
                if d and n:
                    topic = {'title': t, 'description': d, 'narrative': n, 'tfidf_terms': ''}
                    topic_cache[sid] = topic
                    return topic
        except Exception:
            time.sleep(2 ** attempt)
    fallback = {'title': all_queries[0][:30], 'description': 'Research about ' + all_queries[0],
                'narrative': 'N/A', 'tfidf_terms': ''}
    topic_cache[sid] = fallback
    return fallback

# ── Prompt-Funktionen ─────────────────────────────────────────────────────────
DOC_VARIANT_INSTRUCTION_A = {
    'rel': (
        'The researcher has already found some RELEVANT documents on this topic. '
        'Predict the 5 most likely next search queries that go DEEPER or MORE SPECIFIC '
        'into the topic — the researcher wants to explore further details or subtopics.'
    ),
    'non_rel': (
        'The researcher has NOT yet found useful documents — the retrieved results were NOT relevant. '
        'Predict the 5 most likely next search queries the researcher would use to '
        'REFORMULATE or try ALTERNATIVE angles to find what they are looking for.'
    ),
    'contrastive': (
        'Based on the research topic below, predict the 5 most likely next search queries '
        'that this researcher would naturally search for next.'
    ),
}

def build_prompt_A(sid, queries_text, all_queries, doc_variant):
    topic       = get_topic(sid, all_queries)
    tfidf       = topic.get('tfidf_terms', '').strip() or 'N/A'
    instruction = DOC_VARIANT_INSTRUCTION_A.get(doc_variant, DOC_VARIANT_INSTRUCTION_A['contrastive'])
    time.sleep(0.2)
    system = (
        'You are a precise research assistant. '
        'Your sole task is to generate follow-up search queries strictly based on the provided topic context. '
        'Rules:\n'
        '- Use ONLY information explicitly present in the context below.\n'
        '- Do NOT invent new topics, assumptions, or unrelated content.\n'
        '- Be specific, neutral, and concise.\n'
        '- Output ONLY valid JSON - no intro, no explanation, no commentary.'
    )
    user = (
        instruction + '\n\n'
        'TOPIC CONTEXT:\n'
        'Title:            ' + topic['title'] + '\n'
        'Description:      ' + topic['description'] + '\n'
        'Narrative:        ' + topic['narrative'] + '\n'
        'Previous Queries: ' + queries_text + '\n'
        'Key Terms:        ' + tfidf + '\n\n'
        'Output Format (strictly):\n'
        'Return ONLY valid JSON with key "queries": array of exactly 5 strings.\n'
        'No extra text, no markdown - only the JSON object.'
    )
    return system + '\n\n' + user

def build_prompt_B(queries_text, rel_docnos, nonrel_docnos, doc_variant, doc_lookup):
    lines = [
        'Given some user queries, relevant and not relevant documents,',
        'predict the 5 most likely next search queries the user will type.',
        '',
        'Queries',
        'A person has typed these queries into a search engine:',
        queries_text,
        '',
    ]
    if doc_variant in ('rel', 'contrastive'):
        rel_block = ''
        for i, d in enumerate(rel_docnos[:N_REL_DOCS], 1):
            rel_block += f'[Document {i}]\n' + get_doc_text(d, doc_lookup) + '\n\n'
        lines += [
            '--- BEGIN RELEVANT DOCUMENTS CONTENT ---',
            rel_block.strip() or '(no clicked documents in this session)',
            '--- END RELEVANT DOCUMENTS CONTENT ---', '',
        ]
    if doc_variant in ('non_rel', 'contrastive'):
        nonrel_block = ''
        for i, d in enumerate(nonrel_docnos[:N_NONREL_DOCS], 1):
            nonrel_block += f'[Document {i}]\n' + get_doc_text(d, doc_lookup) + '\n\n'
        lines += [
            '--- BEGIN NOT RELEVANT DOCUMENTS CONTENT ---',
            nonrel_block.strip() or '(no skipped documents available)',
            '--- END NOT RELEVANT DOCUMENTS CONTENT ---', '',
        ]
    lines += [
        'Output Format and Structure:',
        'Return ONLY valid JSON with key "queries": array of exactly 5 strings.',
        'Rank from most to least likely. Keep queries concise and on-topic.',
        'No extra text, no markdown - only the JSON object.',
    ]
    return '\n'.join(lines)

def generate_next_queries(sid, sdata, cfg, doc_lookup, retries=3):
    queries_text = get_context(sdata['queries'], cfg['query_mode'])
    if cfg['prompt_variant'] == 'A':
        prompt = build_prompt_A(sid, queries_text, sdata['queries'], cfg['doc_variant'])
    else:
        prompt = build_prompt_B(queries_text, sdata['rel_docnos'],
                                sdata['nonrel_docnos'], cfg['doc_variant'], doc_lookup)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.5, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                qs = data.get('queries', [])
                if isinstance(qs, list) and len(qs) >= 1:
                    return [str(q).strip() for q in qs[:5]]
        except Exception as e:
            print(f'    Attempt {attempt+1} failed: {e}')
            time.sleep(2 ** attempt)
    last = sdata['queries'][-2] if len(sdata['queries']) > 1 else sdata['queries'][0]
    return [last, last+' review', last+' study', last+' analysis', last+' overview']

# ── Einen einzelnen Run ausführen ─────────────────────────────────────────────
def run_single(cfg, doc_lookup):
    run_name = cfg['run_name']
    out_zip  = OUTPUT_DIR / f'task3_submission_{run_name}.zip'

    print(f'\n{"─"*60}')
    print(f'  RUN: {run_name}')
    print(f'  query_mode={cfg["query_mode"]} | prompt={cfg["prompt_variant"]} | doc_variant={cfg["doc_variant"]}')
    print(f'{"─"*60}')

    snapshot_results = {}
    t_start = time.time()

    # ── Schritt 1: Snapshot-1 immer neu generieren ────────────────────────────
    snap1_name = 'snapshot-1.jsonl'
    print(f'\n  === {snap1_name} (neu mit offiziellem Trainingsdatensatz) ===')
    sessions = load_sessions(SNAPSHOT_FILES[snap1_name])
    run_data = {'meta': {
        'team_name':   TEAM_NAME,
        'description': DESCRIPTION,
        'run_name':    run_name,
        'query_mode':  cfg['query_mode'],
        'prompt':      cfg['prompt_variant'],
        'doc_variant': cfg['doc_variant'],
    }}
    for j, (sid, sdata) in enumerate(sessions.items(), 1):
        print(f'    [{j}/{len(sessions)}] {sdata["queries"][0][:55]}')
        run_data[sid] = generate_next_queries(sid, sdata, cfg, doc_lookup)
        time.sleep(0.3)
    snapshot_results[snap1_name] = run_data
    print(f'  -> {len(sessions)} Sessions fertig.')

    # ── Schritt 2: Snapshot-2 & -3 aus bestehendem ZIP übernehmen ────────────
    if out_zip.exists():
        print(f'\n  Übernehme snapshot-2 & -3 aus bestehendem ZIP...')
        with zipfile.ZipFile(out_zip, 'r') as zf:
            zip_files = zf.namelist()
            for snap_name in ['snapshot-2.jsonl', 'snapshot-3.jsonl']:
                # Suche auch nach .json (ohne l) für ältere ZIPs
                snap_name_alt = snap_name.replace('.jsonl', '.json')
                found = snap_name if snap_name in zip_files else (snap_name_alt if snap_name_alt in zip_files else None)
                if found:
                    data = json.loads(zf.read(found))
                    sids = [k for k in data if k != 'meta']
                    snapshot_results[snap_name] = data  # immer als .jsonl speichern
                    print(f'    {found} → {snap_name}: {len(sids)} Sessions übernommen ✓')
                else:
                    print(f'    WARNUNG: {snap_name} nicht im ZIP – wird neu generiert')
                    _generate_snapshot(snap_name, cfg, doc_lookup, snapshot_results)
    else:
        print(f'\n  Kein bestehender ZIP – generiere snapshot-2 & -3 ebenfalls neu...')
        for snap_name in ['snapshot-2.jsonl', 'snapshot-3.jsonl']:
            _generate_snapshot(snap_name, cfg, doc_lookup, snapshot_results)

    # ── ZIP speichern ─────────────────────────────────────────────────────────
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for snap_name, data in snapshot_results.items():
            zf.writestr(snap_name, json.dumps(data, ensure_ascii=False, indent=2))

    # Validierung
    errors = []
    with zipfile.ZipFile(out_zip, 'r') as zf:
        for name in zf.namelist():
            data = json.loads(zf.read(name))
            for key, val in data.items():
                if key == 'meta': continue
                if not isinstance(val, list) or len(val) < 1:
                    errors.append(f'{name} / Session {key}')

    elapsed = (time.time() - t_start) / 60
    if errors:
        print(f'  ✗ FEHLER in {len(errors)} Sessions: {errors[:3]}')
        return False
    else:
        counts = {k: len(v)-1 for k, v in snapshot_results.items()}
        print(f'  ✓ {run_name} gespeichert ({elapsed:.1f} Min) | Sessions: {counts}')
        return True


def _generate_snapshot(snap_name, cfg, doc_lookup, snapshot_results):
    """Hilfsfunktion: einen Snapshot neu generieren und in snapshot_results eintragen."""
    csv_path = SNAPSHOT_FILES[snap_name]
    print(f'\n  === {snap_name} ===')
    sessions = load_sessions(csv_path)
    run_data = {'meta': {
        'team_name':   TEAM_NAME,
        'description': DESCRIPTION,
        'run_name':    cfg['run_name'],
        'query_mode':  cfg['query_mode'],
        'prompt':      cfg['prompt_variant'],
        'doc_variant': cfg['doc_variant'],
    }}
    for j, (sid, sdata) in enumerate(sessions.items(), 1):
        print(f'    [{j}/{len(sessions)}] {sdata["queries"][0][:55]}')
        run_data[sid] = generate_next_queries(sid, sdata, cfg, doc_lookup)
        time.sleep(0.3)
    snapshot_results[snap_name] = run_data
    print(f'  -> {len(sessions)} Sessions fertig.')


# ── Hauptprogramm ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    start_total = time.time()
    print('=' * 60)
    print(' Task-3 – Alle 18 Runs (snapshot-1 neu generieren)')
    print('=' * 60)
    print(f'Ausgabe-Ordner: {OUTPUT_DIR}')
    print(f'Snapshot-1 CSV: {SNAPSHOT_FILES["snapshot-1.jsonl"]}')
    print('Snapshot-2 & -3 werden aus bestehendem ZIP übernommen.')
    print()

    results = {}

    # ── Phase 1: Alle Prompt-A-Runs (kein doc_lookup) ────────────────────────
    runs_A = [r for r in ALL_RUNS if r['prompt_variant'] == 'A']
    print(f'{"="*60}')
    print(f' PHASE 1: {len(runs_A)} Prompt-A-Runs (kein doc_lookup)')
    print(f'{"="*60}')
    for i, cfg in enumerate(runs_A, 1):
        print(f'\n[{i}/{len(runs_A)} A-Runs]')
        results[cfg['run_name']] = run_single(cfg, doc_lookup={})

    # ── Phase 2: doc_lookup einmalig laden für Prompt-B-Runs ─────────────────
    runs_B = [r for r in ALL_RUNS if r['prompt_variant'] == 'B']
    print(f'\n{"="*60}')
    print(f' PHASE 2: doc_lookup laden für {len(runs_B)} Prompt-B-Runs')
    print(f'{"="*60}')
    print('Lade Dokument-Lookup (~5-10 Min)...')
    from ir_datasets_longeval import load as ld_load
    doc_lookup = {}
    dataset = ld_load('longeval-sci-2026/snapshot-1/train')
    for i, doc in enumerate(dataset.docs_iter()):
        doc_lookup[str(doc.doc_id)] = {
            'title':    getattr(doc, 'title', '') or '',
            'abstract': getattr(doc, 'abstract', '') or ''
        }
        if i % 300000 == 0 and i > 0:
            print(f'  {i:,} Docs geladen...')
    print(f'Fertig: {len(doc_lookup):,} Docs.\n')

    for i, cfg in enumerate(runs_B, 1):
        print(f'\n[{i}/{len(runs_B)} B-Runs]')
        results[cfg['run_name']] = run_single(cfg, doc_lookup)

    # ── Abschlussbericht ──────────────────────────────────────────────────────
    elapsed_total = (time.time() - start_total) / 60
    ok  = [k for k, v in results.items() if v]
    err = [k for k, v in results.items() if not v]

    print(f'\n{"="*60}')
    print(f' FERTIG – {elapsed_total:.0f} Minuten gesamt')
    print(f'{"="*60}')
    print(f'  Erfolgreich: {len(ok)} Runs')
    if err:
        print(f'  Fehlerhaft:  {len(err)} Runs → {err}')

    all_zips = sorted(OUTPUT_DIR.glob('task3_submission_openai_*.zip'))
    print(f'\n  ZIPs in {OUTPUT_DIR.name}/: {len(all_zips)}/18')
    for z in all_zips:
        print(f'    ✓ {z.name}')
    print()
