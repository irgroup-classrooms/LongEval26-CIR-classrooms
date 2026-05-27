"""
fix_snapshot3_A_runs.py – Korrigiert snapshot-3 in allen 9 Prompt-A-Runs.

VERSION 2: Liest snapshot-1 & snapshot-2 direkt aus den beschädigten ZIPs
(via Raw-Byte-Parsing, da Central Directory fehlt). Nur snapshot-3 wird
neu per API generiert.

Starten mit:  python fix_snapshot3_A_runs.py
Abbruch:      Strg+C  (bereits fertige Runs bleiben gespeichert)
"""

import os, json, time, re, zipfile, zlib, struct, ast
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

# ── Setup ──────────────────────────────────────────────────────────────────────
for folder in [Path(__file__).parent, *Path(__file__).parent.parents]:
    if (folder / '.env').exists():
        load_dotenv(folder / '.env', override=True)
        break

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip().strip('"').strip("'")
if not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY fehlt! Bitte in .env Datei eintragen.')
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'runfiles_task3'

SNAPSHOT3_CSV = str(BASE_DIR / 'task3_longeval_usim-sessions-09-11_2025.csv')
if not Path(SNAPSHOT3_CSV).exists():
    raise SystemExit(f'FEHLER – Datei fehlt: {SNAPSHOT3_CSV}')

MODEL         = 'gpt-4o-mini'
TEAM_NAME     = 'JOINorDIE'
DESCRIPTION   = 'LLM-based next query prediction using OpenAI GPT-4o-mini.'

A_RUNS = [
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_first_A_rel'},
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_first_A_non_rel'},
    {'query_mode': 'first', 'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_first_A_contrastive'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_all_A_rel'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_all_A_non_rel'},
    {'query_mode': 'all',   'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_all_A_contrastive'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'rel',         'run_name': 'openai_last_A_rel'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'non_rel',     'run_name': 'openai_last_A_non_rel'},
    {'query_mode': 'last',  'prompt_variant': 'A', 'doc_variant': 'contrastive', 'run_name': 'openai_last_A_contrastive'},
]

# ── Raw-ZIP-Reader (funktioniert auch ohne Central Directory) ──────────────────
def read_zip_raw(zip_path):
    """Liest alle vollständig komprimierten Einträge aus einem ZIP,
    auch wenn das Central Directory fehlt (beschädigte Datei)."""
    with open(zip_path, 'rb') as f:
        data = f.read()
    entries = {}
    offset = 0
    while offset < len(data) - 4:
        if data[offset:offset+4] == b'PK\x03\x04':
            compressed_size   = struct.unpack_from('<I', data, offset+18)[0]
            filename_len      = struct.unpack_from('<H', data, offset+26)[0]
            extra_len         = struct.unpack_from('<H', data, offset+28)[0]
            filename          = data[offset+30:offset+30+filename_len].decode('utf-8')
            data_start        = offset + 30 + filename_len + extra_len
            data_end          = data_start + compressed_size
            compressed_data   = data[data_start:data_end]
            try:
                raw = zlib.decompress(compressed_data, -15)
                entries[filename] = json.loads(raw)
            except Exception:
                pass  # Abgeschnittener Eintrag – überspringen
            offset = data_end
        else:
            offset += 1
    return entries

# ── Session-Loader & Kontext ───────────────────────────────────────────────────
def parse_docnos(val):
    try:    return [str(x) for x in ast.literal_eval(str(val))]
    except: return []

def parse_interactions(val):
    try:
        items = ast.literal_eval(str(val))
        return [str(item[0]) for item in items if isinstance(item, (list, tuple))]
    except: return []

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

def get_context(queries, mode):
    ctx = queries[:-1] if len(queries) > 1 else queries
    if mode == 'first': return ctx[0] if ctx else queries[0]
    if mode == 'last':  return ctx[-1] if ctx else queries[0]
    return '\n'.join(ctx)

# ── Topic-Cache (geteilt über alle 9 Runs) ────────────────────────────────────
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
                d = json.loads(match.group())
                t = str(d.get('title', queries_text[:30])).strip()
                desc = str(d.get('description', '')).strip()
                n = str(d.get('narrative', '')).strip()
                if desc and n:
                    topic = {'title': t, 'description': desc, 'narrative': n, 'tfidf_terms': ''}
                    topic_cache[sid] = topic
                    return topic
        except Exception:
            time.sleep(2 ** attempt)
    fallback = {'title': all_queries[0][:30], 'description': 'Research about ' + all_queries[0],
                'narrative': 'N/A', 'tfidf_terms': ''}
    topic_cache[sid] = fallback
    return fallback

# ── Prompt A ───────────────────────────────────────────────────────────────────
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

def generate_next_queries(sid, sdata, cfg, retries=3):
    queries_text = get_context(sdata['queries'], cfg['query_mode'])
    prompt = build_prompt_A(sid, queries_text, sdata['queries'], cfg['doc_variant'])
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
                d = json.loads(match.group())
                qs = d.get('queries', [])
                if isinstance(qs, list) and len(qs) >= 1:
                    return [str(q).strip() for q in qs[:5]]
        except Exception as e:
            print(f'    Attempt {attempt+1} failed: {e}')
            time.sleep(2 ** attempt)
    last = sdata['queries'][-2] if len(sdata['queries']) > 1 else sdata['queries'][0]
    return [last, last+' review', last+' study', last+' analysis', last+' overview']

# ── Einen Run fixen ────────────────────────────────────────────────────────────
def fix_run(cfg, sessions_snap3):
    run_name = cfg['run_name']
    zip_path = OUTPUT_DIR / f'task3_submission_{run_name}.zip'

    print(f'\n{"─"*60}')
    print(f'  FIX: {run_name}')
    print(f'  query_mode={cfg["query_mode"]} | doc_variant={cfg["doc_variant"]}')
    print(f'{"─"*60}')

    if not zip_path.exists():
        print(f'  FEHLER: ZIP nicht gefunden: {zip_path}')
        return False

    # Lade snapshot-1 & snapshot-2 via Raw-Byte-Parser (funktioniert auch bei kaputten ZIPs)
    print(f'  Lese snapshot-1 & snapshot-2 aus beschädigtem ZIP...')
    existing = read_zip_raw(str(zip_path))

    for snap in ['snapshot-1.jsonl', 'snapshot-2.jsonl']:
        if snap in existing:
            sids = [k for k in existing[snap] if k != 'meta']
            print(f'  {snap}: {len(sids)} Sessions gelesen ✓')
        else:
            print(f'  FEHLER: {snap} nicht lesbar!')
            return False

    # Generiere snapshot-3 NEU
    print(f'\n  === snapshot-3.jsonl NEU generieren ({len(sessions_snap3)} Sessions) ===')
    new_snap3 = {'meta': {
        'team_name':   TEAM_NAME,
        'description': DESCRIPTION,
        'run_name':    run_name,
        'query_mode':  cfg['query_mode'],
        'prompt':      cfg['prompt_variant'],
        'doc_variant': cfg['doc_variant'],
    }}

    t_start = time.time()
    for j, (sid, sdata) in enumerate(sessions_snap3.items(), 1):
        print(f'    [{j}/{len(sessions_snap3)}] {sdata["queries"][0][:55]}')
        new_snap3[sid] = generate_next_queries(sid, sdata, cfg)
        time.sleep(0.3)

    elapsed = (time.time() - t_start) / 60
    print(f'  -> {len(sessions_snap3)} Sessions generiert ({elapsed:.1f} Min)')
    print(f'  Session 0: {new_snap3.get("0", ["???"])[0][:65]}')

    # Schreibe neues gültiges ZIP
    all_data = {
        'snapshot-1.jsonl': existing['snapshot-1.jsonl'],
        'snapshot-2.jsonl': existing['snapshot-2.jsonl'],
        'snapshot-3.jsonl': new_snap3,
    }
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for snap_name, data in all_data.items():
            zf.writestr(snap_name, json.dumps(data, ensure_ascii=False, indent=2))

    # Validierung
    errors = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            data = json.loads(zf.read(name))
            for key, val in data.items():
                if key == 'meta': continue
                if not isinstance(val, list) or len(val) < 1:
                    errors.append(f'{name}/Session {key}')

    counts = {k: len(v)-1 for k, v in all_data.items()}
    if errors:
        print(f'  ✗ FEHLER in {len(errors)} Sessions: {errors[:3]}')
        return False
    else:
        print(f'  ✓ {run_name}.zip gespeichert | Sessions: {counts}')
        return True


# ── Hauptprogramm ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print(' Fix snapshot-3 – Alle 9 Prompt-A-Runs (Version 2)')
    print('=' * 60)
    print(f'Snapshot-3 CSV: {SNAPSHOT3_CSV}')
    print()

    print('Lade Snapshot-3 Sessions...')
    sessions_snap3 = load_sessions(SNAPSHOT3_CSV)
    print(f'  {len(sessions_snap3)} Sessions geladen.')
    print(f'  Session 0: {sessions_snap3["0"]["queries"][:2]}')
    print()

    results = {}
    for i, cfg in enumerate(A_RUNS, 1):
        print(f'\n[{i}/{len(A_RUNS)}]')
        results[cfg['run_name']] = fix_run(cfg, sessions_snap3)

    ok  = [k for k, v in results.items() if v]
    err = [k for k, v in results.items() if not v]

    print(f'\n{"="*60}')
    print(f' FERTIG')
    print(f'{"="*60}')
    print(f'  Erfolgreich: {len(ok)} Runs')
    if err:
        print(f'  Fehlerhaft:  {len(err)} Runs → {err}')
    print()
    print('Nächste Schritte:')
    print('  Die 9 aktualisierten ZIPs in runfiles_task3/ bei TIRA neu einreichen.')
    print()
