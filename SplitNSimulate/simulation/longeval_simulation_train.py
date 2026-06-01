# LongEval Session Simulation
# Kübra & Emirhan, 2026
# ============================================================
#
# Was macht dieses Skript?
# ------------------------
# Es simuliert 5 mögliche nächste Queries pro Session,
# basierend auf der Session Historie, den Suchergebnissen
# (SERP) und den angeklickten Ergebnissen.
#
# Phase 1 – Persona Klassifizierung
#   Jede Session wird einzeln klassifiziert, hier kein Batching mehr.
#   Das vermeidet ID Mapping Fehler die beim Batching
#   entstehen können (z.B. Session 26 landete in falscher Persona).
#   Der Classifier bekommt ALLE Queries einer Session,
#   also auch die letzte (Groundtruth). Das ist okay,
#   weil wir hier nur die Persona bestimmen, nicht die
#   nächste Query vorhersagen.
#   Das Ergebnis wird im Cache gespeichert.
#   → session_personas_cache.json
#
# Phase 2 – Predictions generieren
#   Pro Session werden 5 verschiedene Predictions erstellt.
#   Hier wird die letzte Query (natürlich als Groundtruth betrachtet) NICHT an das Modell übergeben, also keine Sorge, Andreas :D
#   Das Skript ist resume fähig: bereits fertige Sessions
#   werden übersprungen, unvollständige vervollständigt.
#   → longeval_simulation_all.csv
# ============================================================

import pandas as pd
import ollama
import json
import csv
import urllib.request
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME              = "llama3.2"      
CLASSIFIER_MODEL        = "llama3.2"     # für experimentelle Klassifizierung.... beide Modelle sind am gleich geblieben, damit die Klassifizierung zuverlässiger wird (phi3 hatte zu viele Halluzinationen bei der Persona Erkennung)
OLLAMA_URL              = "http://localhost:11434/api/generate"
PREDICTIONS_PER_SESSION = 5
CACHE_FILE              = Path("session_personas_cache.json")
OUTPUT_FILE             = Path("longeval_simulation_all.csv")

# ── Persona definitions ───────────────────────────────────────────────────────
#
# Drei Bausteine:
# PERSONAS                        → alle gültigen Persona Namen (für Validierung)
# PERSONA_DESCRIPTIONS            → Rollenbeschreibung für den Prediction Prompt.
#                                   Wichtig: kein Philippines Bias, keine zu engen
#                                   Annahmen, sonst halluziniert das Modell wieder Details
#                                   die gar nicht in der Session stehen.
# PERSONA_CLASSIFIER_DESCRIPTIONS → Erkennungsmerkmale für den Classifier Prompt.
#                                   Nicht nur WAS gesucht wird, sondern auch WIE:
#                                   Schreibstil, Suchverhalten, Formulierungsmuster.

PERSONAS = [
    "Student Researcher",
    "Academic Researcher",
    "Policy & Security Analyst",
    "Sustainability & Climate Researcher",
    "Healthcare & Life Sciences Researcher",
    "Business & Economics Researcher",
    "Engineering & Tech Developer",
    "Unknown",
]

PERSONA_DESCRIPTIONS = {
    "Student Researcher": (
        "Student Researcher",
        "You are a student or educator searching for academic content — either for a thesis, "
        "research paper, or teaching material. You look for literature reviews (RRL), "
        "academic performance studies, and topic-specific educational content. "
        "Your queries are exploratory and follow the structure of an academic paper."
    ),
    "Academic Researcher": (
        "Academic Researcher",
        "You are a specialist researcher working on a very narrow scientific or technical topic. "
        "You use precise terminology, cite authors and years, and look for peer-reviewed papers, "
        "specific methods, and datasets. Your queries are highly specific and stay within one niche."
    ),
    "Policy & Security Analyst": (
        "Policy & Security Analyst",
        "You are an analyst researching geopolitics, security, military affairs, or energy policy. "
        "You track political figures, organizations, and policy developments. "
        "Your queries combine specific names with policy domains and are factual and investigative."
    ),
    "Sustainability & Climate Researcher": (
        "Sustainability & Climate Researcher",
        "You are a researcher or practitioner focused on renewable energy, climate change, "
        "or environmental sustainability. You search for technical solutions, policy frameworks, "
        "and sustainability strategies across energy, waste, and biodiversity topics."
    ),
    "Healthcare & Life Sciences Researcher": (
        "Healthcare & Life Sciences Researcher",
        "You are a researcher, clinician, or public health professional searching for medical "
        "or biological information. You use precise clinical or scientific terminology and look "
        "for disease information, treatment protocols, compounds, and clinical studies."
    ),
    "Business & Economics Researcher": (
        "Business & Economics Researcher",
        "You are a business analyst, economist, or manager researching business operations, "
        "financial planning, supply chains, or economic trends. Your queries are practical "
        "and results-oriented, focused on real-world business problems."
    ),
    "Engineering & Tech Developer": (
        "Engineering & Tech Developer",
        "You are a developer or engineer working on technical systems — software, hardware, "
        "infrastructure, or data engineering. Your queries are technical and implementation-focused, "
        "using exact tool names, methods, and system components."
    ),
    "Unknown": (
        "General Researcher",
        "You are a researcher with a specific but uncommon topic that does not fit a clear domain. "
        "Follow the exact topic and direction of the search history without adding assumptions "
        "about the user's background or location."
    ),
}

# Erkennungsmerkmale für den Classifier.
# Wir beschreiben nicht nur WAS gesucht wird, sondern auch WIE:
# Schreibstil, Formulierungsmuster, typische Fehler, Suchverhalten.
PERSONA_CLASSIFIER_DESCRIPTIONS = """\
- Student Researcher:
  WHO: Students or educators searching for academic content — theses, research papers,
  or teaching material. The most common persona in this dataset.
  HOW THEY SEARCH:
  - Queries often contain typos, incomplete sentences, or abrupt endings
  - Copy-paste long titles directly from papers or textbooks
  - Add "local studies", "foreign studies", "local literature", "RRL" to queries
  - Use phrases like "effects of X on Y", "impact of X on academic performance"
  - Reference school levels: "Grade 9", "Grade 10", "SHS", "HUMSS", "senior high school"
  - Topics: social media & students, mobile games, sleep, classroom behavior,
    teacher effectiveness, inclusive education, LGBTQ in schools, commuters,
    thesis writing, academic performance across subjects
  NOTE: Queries about teachers and education topics belong here, NOT in Healthcare.
  NOTE: Do NOT assign this persona just because queries are vague or short.

- Academic Researcher:
  WHO: Specialist scientists or PhD-level researchers on very narrow topics.
  HOW THEY SEARCH:
  - Extremely precise queries with domain-specific terminology
  - Include author names + publication years (e.g. "Galloway 1989")
  - Use exact method or model names: LSTM, CNN, APSIM, ARIMA, XGBoost
  - Queries stay in a very narrow niche across the whole session
  - No typos, very structured and formal phrasing
  - Topics: computational models, data analysis methods, niche biology, engineering science

- Policy & Security Analyst:
  WHO: Analysts tracking geopolitics, security, energy policy, or military affairs.
  HOW THEY SEARCH:
  - Combine specific names of people or organizations with policy domains
  - Examples: "Erdogan oil sector", "private military company Chief of Staff",
    "Biden foreign policy", "Trump unilateralism", "drone warfare think tank"
  - Track relationships between political figures and industries or organizations
  - Queries are factual and investigative, not academic in style
  - Topics: foreign policy, drone warfare, think tanks, military organizations,
    oil exploration, geopolitical influence, security services
  IMPORTANT: Sessions about political figures and their policies belong here,
  even if the queries are short or seem vague.

- Sustainability & Climate Researcher:
  WHO: Researchers or practitioners in clean energy, climate, or environment.
  HOW THEY SEARCH:
  - Mix of engineering terms and policy language
  - Often include years like "2030" as targets or deadlines
  - Topics: solar panels, biofuels, MPPT, flow batteries, carbon reduction,
    decarbonization, circular economy, waste management, biodiversity,
    ecological carrying capacity, rural sustainability, climate adaptation
  NOTE: Sustainability topics that overlap with engineering (e.g. solar pump systems,
  biofuel optimization) belong here, not in Engineering.

- Healthcare & Life Sciences Researcher:
  WHO: Clinicians, nurses, public health workers, biology researchers.
  HOW THEY SEARCH:
  - Use medical or biological terminology precisely
  - Search for specific diseases, treatments, compounds, or organisms
  - Include Latin species names, drug names, or clinical terms
  - Topics: cancer, pharmacogenetics, caregiving, plant medicine, animal genetics,
    physiotherapy, HIV, community health, nanomedicine, clinical studies
  NOT: education, schools, or teacher topics — those are Student Researcher.
  NOT: general mental health or social behavior without a clinical angle.

- Business & Economics Researcher:
  WHO: Business analysts, economists, entrepreneurs, or managers.
  HOW THEY SEARCH:
  - Practical and results-oriented queries
  - Focus on management problems, financial metrics, or market dynamics
  - Topics: cash flow, debt, supply chain, corporate tax, accounting, inflation,
    marketing strategies, customs, logistics, financial literacy, SME operations,
    social media marketing, consumer behavior from a business angle

- Engineering & Tech Developer:
  WHO: Software developers, hardware engineers, or technical architects.
  HOW THEY SEARCH:
  - Highly technical and implementation-focused queries
  - Use exact technical terms: "embedded systems", "model-based testing",
    "IoT", "blockchain", "cloud", "load balancing", "debug trace"
  - Topics: software testing, cloud architecture, IoT security, drilling optimization,
    flow measurement, tunnel construction, spacecraft scheduling, data structures,
    processor efficiency, machine learning engineering (not research)

- Unknown:
  Use ONLY when the session clearly does not fit any of the above.
  Examples: cultural history, linguistics, literature, philosophy, architecture,
  religion, archaeology, very mixed topics with no clear domain.
  Do NOT use Unknown for political topics (-> Policy & Security Analyst),
  do NOT use Unknown for vague or short queries alone.\
"""

# ── Ollama check ──────────────────────────────────────────────────────────────

def check_ollama():
    # Prüfen ob Ollama läuft und beide Modelle verfügbar sind
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data   = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]

        for required in [MODEL_NAME, CLASSIFIER_MODEL]:
            short = required.split(":")[0]
            match = [m for m in models if m.startswith(short)]
            if not match:
                print(f"Warning: model '{required}' not found. Available: {', '.join(models)}")
                raise SystemExit(1)
            print(f"Ollama OK — {('classifier' if required == CLASSIFIER_MODEL else 'prediction')} model: {match[0]}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: Ollama not reachable. Please run 'ollama serve'.\n{e}")
        raise SystemExit(1)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    column_names = [
        "query_sequence", "user_id", "session_id",
        "query_text", "timestamp", "returned_results",
        "query_id", "clicks",
    ]
    return pd.read_csv(filepath, header=None, names=column_names)

# ── Phase 1: Single Session Classifier ───────────────────────────────────────
#
# Jede Session wird einzeln klassifiziert, kein Batch mapping mehr.
# Das war, wie es scheint, der Hauptgrund für falsche Zuordnungen wie Session 26:
# Im Batch Modus musste das Modell lokale IDs (1–10) auf echte
# Session IDs mappen, dabei passierten Verwechslungen.
# Jetzt bekommt das Modell genau eine Session und gibt genau
# eine Antwort zurück. Kein Mapping, kein Fehlerrisiko.

def extract_json(raw: str):
    # Markdown fences entfernen falls das Modell sie trotzdem schreibt
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                raw = part
                break
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    return json.loads(raw)


def classify_single_session(session_id: int, queries: list) -> str:
    """
    Klassifiziert eine einzelne Session und gibt den Persona String zurück.
    Bei Fehler wird "Unknown" zurückgegeben.
    """
    all_queries = " | ".join(queries)

    prompt = f"""\
Classify the following search session with EXACTLY one persona.

Available personas:
{PERSONA_CLASSIFIER_DESCRIPTIONS}

Rules:
- Respond ONLY with a JSON object, no other text, no markdown fences.
- Format: {{"persona": "Student Researcher"}}
- Use only the exact persona names from the list above.
- Choose "Unknown" only if the session truly does not fit any other persona.

Session queries (all queries separated by " | "):
{all_queries}

JSON response:"""

    for attempt in range(3):
        try:
            payload = json.dumps({
                "model":   CLASSIFIER_MODEL,   
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 64},
            }).encode()
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            raw     = data.get("response", "").strip()
            result  = extract_json(raw)
            persona = result.get("persona", "Unknown").strip()

            # Validierung, nur bekannte Persona Namen akzeptieren
            if persona not in PERSONAS:
                # Fuzzy Match: enthält die Antwort einen bekannten Namen?
                for p in PERSONAS:
                    if p.lower() in persona.lower():
                        persona = p
                        break
                else:
                    persona = "Unknown"

            return persona

        except Exception as e:
            wait = 2 ** attempt
            print(f"\n      [Attempt {attempt+1}/3 failed: {e} — waiting {wait}s]", end="")
            if attempt < 2:
                time.sleep(wait)

    print(f"\n      [All attempts failed -> Unknown]", end="")
    return "Unknown"


def classify_all_sessions(all_sessions: dict) -> dict:
    """
    Klassifiziert alle Sessions einzeln und speichert den Cache nach jeder Session.
    Bereits klassifizierte Sessions werden übersprungen, absturzsicher.
    """
    # Vorhandenen Cache laden
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Cache loaded: {len(cache)} sessions already classified.")

    todo = [sid for sid in all_sessions if str(sid) not in cache]
    print(f"  Still to classify: {len(todo)} sessions\n")

    for idx, session_id in enumerate(todo, 1):
        queries     = all_sessions[session_id]
        queries_str = " | ".join(queries)
        print(f"  [{idx:>3}/{len(todo)}] Session {session_id}: "
              f"{queries_str[:80]}{'...' if len(queries_str) > 80 else ''}", end="")

        persona = classify_single_session(session_id, queries)
        print(f"  -> {persona}")

        cache[str(session_id)] = persona

        # Nach jeder Session speichern, bei Absturz geht nichts verloren
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    print(f"\n  Classification done -> {CACHE_FILE}")
    return {int(k): v for k, v in cache.items()}

# ── Phase 2: Generate predictions ────────────────────────────────────────────

def extract_clicked_titles(clicks_raw) -> list:
    """
    Extrahiert die Titel der angeklickten Ergebnisse aus dem Clicks Feld.
    Gibt eine leere Liste zurück wenn keine Clicks vorhanden sind.
    """
    import ast
    if not clicks_raw or str(clicks_raw).strip() in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(str(clicks_raw))
        if isinstance(parsed, list):
            titles = []
            for item in parsed:
                if isinstance(item, dict):
                    # Titel aus dem Dict holen — je nach Struktur
                    title = item.get("title") or item.get("name") or str(item)
                elif isinstance(item, str):
                    title = item
                else:
                    title = str(item)
                if title:
                    titles.append(title.strip())
            return titles
    except Exception:
        # Falls das Parsen scheitert, den Rohtext zurückgeben
        raw = str(clicks_raw).strip()
        if len(raw) > 2:
            return [raw]
    return []


def build_context(serp, clicks_raw, query_label="Current") -> str:
    """
    Baut den Kontext Block für den Prediction Prompt.
    Strategie: Clicks zuerst, die zeigen was den Nutzer wirklich interessiert hat.
    Fallback auf Top-10 SERP wenn keine Clicks vorhanden sind (trifft auf ~1% der Sessions zu).
    """
    clicked = extract_clicked_titles(clicks_raw)

    if clicked:
        # Normalfall: nur die angeklickten Ergebnisse zeigen
        titles = "\n".join(f"  - {t}" for t in clicked)
        return f"{query_label} — clicked results:\n{titles}"
    else:
        # Fallback: Top-10 aus der SERP (kein SERP rauschen für das kleine Modell)
        serp_str = str(serp).strip()
        if serp_str and serp_str != "nan":
            # Nur die ersten 10 Einträge nehmen falls es eine Liste ist
            try:
                import ast
                parsed = ast.literal_eval(serp_str)
                if isinstance(parsed, list):
                    top10 = parsed[:10]
                    titles = "\n".join(
                        f"  - {item.get('title', str(item)) if isinstance(item, dict) else str(item)}"
                        for item in top10
                    )
                    return f"{query_label} — top search results (no clicks):\n{titles}"
            except Exception:
                pass
            # Rohtext kürzen falls Parsen scheitert
            return f"{query_label} — search results (no clicks):\n  {serp_str[:300]}"
        return f"{query_label} — no results available."


def create_prediction_prompt(persona_role, persona_desc, history_list,
                             current_serp, current_clicks,
                             previous_serps=None, previous_clicks=None):
    # Bisherige Queries als nummerierte Liste formatieren
    formatted_history = "\n".join(
        [f"Query {i+1}: {q}" for i, q in enumerate(history_list)]
    )

    # Kontext aus früheren Suchergebnissen, ebenfalls clicks first
    context_parts = []
    if previous_serps and previous_clicks:
        for i, (serp, clicks) in enumerate(zip(previous_serps, previous_clicks)):
            context_parts.append(
                build_context(serp, clicks, query_label=f"Previous Query {i+1}")
            )
    context = "\n\n".join(context_parts) if context_parts else "No previous search context."

    # Aktueller Kontext, clicks first
    current_context = build_context(current_serp, current_clicks, query_label="Current query")

    return f"""
### ROLE ###
You are a simulator acting as: {persona_role}.
{persona_desc}

### PREVIOUS SEARCH CONTEXT ###
{context}

### CURRENT SESSION HISTORY ###
{formatted_history}

### WHAT THE USER CLICKED ###
{current_context}

### INSTRUCTION ###
Based on your persona, the search history, and what the user clicked,
predict what the **next logical query** would be in this research session.

Consider:
- What the user found useful based on what they clicked
- What gaps remain in their understanding
- How their research question is evolving
- Whether they need to narrow down, broaden, or pivot their search

**IMPORTANT:** Your predicted_query must be:
- Maximum 8 words
- Plain search keywords only — no full sentences, no questions
- No punctuation except spaces
- No boolean operators (AND, OR, NOT), quotation marks, or parentheses

Return ONLY a valid JSON object:
{{
  "reasoning": "Brief analysis of what the user clicked and what they would search next",
  "predicted_query": "short keyword search query here"
}}
"""


def get_prediction(prompt: str, retries: int = 3) -> dict:
    # Prediction vom Modell holen, mit automatischen Wiederholungsversuchen.
    # llama3.2 gibt zuverlässiges JSON zurück, der Parser bleibt trotzdem
    # robust für den Fall dass Markdown vom Modell hinzugefügt wird oder es andere Formatierungsabweichungen gibt.
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.7},
            )
            raw = response["message"]["content"]

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Markdown-Fences entfernen falls vorhanden
                clean = raw.strip()
                if "```" in clean:
                    for part in clean.split("```"):
                        part = part.strip().lstrip("json").strip()
                        if part.startswith("{"):
                            clean = part
                            break
                start = clean.find("{")
                end   = clean.rfind("}")
                if start != -1 and end != -1:
                    clean = clean[start:end+1]
                result = json.loads(clean)

            predicted_q = result.get("predicted_query", "ERROR")

            if not predicted_q or predicted_q == "ERROR":
                raise ValueError(f"No valid predicted_query in: {result}")

            return {
                "predicted_query": str(predicted_q).strip(),
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            print(f"\n      [Prediction attempt {attempt+1}/{retries} failed: {e}]", end="")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    print(f"\n      [All attempts failed]")
    return {"predicted_query": "ERROR", "reasoning": "max retries exceeded"}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_simulation(filepath: str):
    print("=" * 70)
    print("LONGEVAL SIMULATION  —  all sessions")
    print(f"Classifier: {CLASSIFIER_MODEL}  |  Predictions: {MODEL_NAME}  |  {PREDICTIONS_PER_SESSION} predictions/session")
    print("=" * 70 + "\n")

    check_ollama()

    df = load_data(filepath)

    # Sessions aufbauen, gruppiert nach session_id
    all_sessions = {}   # {session_id: [alle queries inkl. groundtruth]}
    session_data = {}   # {session_id: {queries, serps, clicks}}

    for session_id, group in df.groupby("session_id"):
        queries = group.sort_values("query_sequence")["query_text"].tolist()
        # Sessions mit nur einer Query überspringen, kein Kontext vorhanden
        if len(queries) < 2:
            continue
        all_sessions[session_id] = queries
        session_data[session_id] = {
            "queries": queries,
            "serps":   group.sort_values("query_sequence")["returned_results"].tolist(),
            "clicks":  group.sort_values("query_sequence")["clicks"].tolist(),
        }

    total_sessions = len(all_sessions)
    print(f"Sessions in dataset:  {total_sessions}")
    print(f"Total predictions:    {total_sessions * PREDICTIONS_PER_SESSION}\n")

    # ── Phase 1: Classify personas ───────────────────────────────────────────
    print("── Phase 1: Persona classification ──")
    persona_map = classify_all_sessions(all_sessions)

    # ── Phase 2: Generate predictions ────────────────────────────────────────
    print("\n── Phase 2: Generate predictions ──")

    # Herausfinden welche Sessions bereits fertige Predictions haben
    done: dict[int, set] = {}
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        # ERROR Zeilen nicht als fertig zählen, werden neu generiert
        valid = existing[existing["predicted_next"] != "ERROR"]
        if len(valid) < len(existing):
            error_count = len(existing) - len(valid)
            print(f"  {error_count} ERROR entries found -> will be regenerated.")
            valid.to_csv(OUTPUT_FILE, index=False)
        for sid, grp in valid.groupby("session_id"):
            done[int(sid)] = set(grp["prediction_number"].tolist())
        complete   = sum(1 for s in done.values() if len(s) >= PREDICTIONS_PER_SESSION)
        incomplete = sum(1 for sid in all_sessions
                         if sid in done and len(done[sid]) < PREDICTIONS_PER_SESSION)
        missing    = sum(1 for sid in all_sessions if sid not in done)
        print(f"  Already complete:   {complete} sessions")
        print(f"  Incomplete:         {incomplete} sessions (will be completed)")
        print(f"  Not started yet:    {missing} sessions\n")

    mode = "a" if OUTPUT_FILE.exists() else "w"
    fieldnames = [
        "session_id", "prediction_number", "persona",
        "session_queries", "groundtruth_query", "predicted_next",
        "reasoning", "session_length", "total_clicks", "lexical_diversity",
    ]

    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        todo_sessions = [
            sid for sid in sorted(all_sessions.keys())
            if len(done.get(sid, set())) < PREDICTIONS_PER_SESSION
        ]
        total_todo = len(todo_sessions)

        for idx, session_id in enumerate(todo_sessions, 1):
            sd      = session_data[session_id]
            queries = sd["queries"]
            serps   = sd["serps"]
            clicks  = sd["clicks"]

            persona    = persona_map.get(session_id, "Unknown")
            role, desc = PERSONA_DESCRIPTIONS[persona]

            # Einfache Session Metriken berechnen
            session_length    = len(queries)
            total_clicks      = sum(1 for c in clicks if c and str(c) != "[]")
            all_words         = " ".join(queries).lower().split()
            lexical_diversity = len(set(all_words)) / len(all_words) if all_words else 0

            already_done  = done.get(session_id, set())
            missing_preds = [
                n for n in range(1, PREDICTIONS_PER_SESSION + 1)
                if n not in already_done
            ]

            print(f"  [{idx}/{total_todo}] Session {session_id} ({role})")
            print(f"    Queries: {queries}")
            print(f"    Generating {len(missing_preds)} prediction(s)...")

            # Für den Prediction Prompt: letzte Query (als Groundtruth) weggelassen, Kontext aus vorherigen Queries + SERP + Clicks
            context_queries   = queries[:-1]
            groundtruth_query = queries[-1]
            context_serp      = serps[-2] if len(serps) > 1 else serps[-1]
            context_clicks    = clicks[-2] if len(clicks) > 1 else clicks[-1]
            previous_serps    = serps[:-2] if len(serps) > 2 else None
            previous_clicks   = clicks[:-2] if len(clicks) > 2 else None

            print(f"    Context:     {context_queries}")
            print(f"    Groundtruth: '{groundtruth_query}' (not shown in prompt)")

            for pred_num in missing_preds:
                prompt       = create_prediction_prompt(
                    role, desc,
                    context_queries,
                    context_serp, context_clicks,
                    previous_serps, previous_clicks,
                )
                response_obj = get_prediction(prompt)
                predicted_q  = response_obj.get("predicted_query", "ERROR")
                reasoning    = response_obj.get("reasoning", "")

                print(f"    Prediction {pred_num}: '{predicted_q}'")

                writer.writerow({
                    "session_id":        session_id,
                    "prediction_number": pred_num,
                    "persona":           role,
                    "session_queries":   " | ".join(queries),
                    "groundtruth_query": groundtruth_query,
                    "predicted_next":    predicted_q,
                    "reasoning":         reasoning,
                    "session_length":    session_length,
                    "total_clicks":      total_clicks,
                    "lexical_diversity": round(lexical_diversity, 4),
                })

            out_f.flush()

    # Abschluss Statistik ausgeben
    final = pd.read_csv(OUTPUT_FILE)
    print(f"\n{'=' * 70}")
    print(f"DONE — {OUTPUT_FILE}")
    print(f"  {final['session_id'].nunique()} sessions x "
          f"{PREDICTIONS_PER_SESSION} = {len(final)} rows")
    print("\nPersona distribution:")
    dist = final.groupby("persona")["session_id"].nunique().sort_values(ascending=False)
    for persona, count in dist.items():
        bar = "█" * int(count * 25 / total_sessions)
        print(f"  {persona:<42} {count:>3}  {bar}")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation(
        "../data/task3_longeval_usim-sessions-train.csv"
    )
