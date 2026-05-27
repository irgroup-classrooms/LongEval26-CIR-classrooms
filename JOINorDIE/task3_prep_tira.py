"""
task3_prep_tira.py - Entpackt alle 18 Task-3-ZIPs fuer den TIRA-Upload.

Liest jede task3_submission_<run>.zip aus runfiles_task3/ und schreibt
die enthaltenen JSONL-Snapshots als einzelne .json-Dateien nach tira_uploads/<run>/.
TIRA erwartet pro Run einen Ordner mit snapshot-1.json, snapshot-2.json, snapshot-3.json.

Starten mit:  python task3_prep_tira.py
"""

import zipfile, json
from pathlib import Path

runs = [
    'openai_first_A_rel', 'openai_first_A_non_rel', 'openai_first_A_contrastive',
    'openai_first_B_rel', 'openai_first_B_non_rel', 'openai_first_B_contrastive',
    'openai_all_A_rel',   'openai_all_A_non_rel',   'openai_all_A_contrastive',
    'openai_all_B_rel',   'openai_all_B_non_rel',   'openai_all_B_contrastive',
    'openai_last_A_rel',  'openai_last_A_non_rel',  'openai_last_A_contrastive',
    'openai_last_B_rel',  'openai_last_B_non_rel',  'openai_last_B_contrastive',
]

for r in runs:
    zip_path = Path(f'runfiles_task3/task3_submission_{r}.zip')
    out_dir  = Path(f'tira_uploads/{r}')
    if not zip_path.exists():
        print(f'FEHLT: {zip_path}')
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            data = json.loads(zf.read(name))
            out_name = name.replace('.jsonl', '.json')
            out_dir.joinpath(out_name).write_text(
                json.dumps(data, ensure_ascii=True, indent=2), encoding='utf-8'
            )
    print(f'OK: {r}')

print('Fertig!')
