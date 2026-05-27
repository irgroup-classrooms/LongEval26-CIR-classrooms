import json
import re
import pandas as pd
import ollama

# We still use your parser just to get the clean, chronological data easily
from persona_NLP import process_personas

MODEL_NAME = "llama3"

def create_cot_top5_prompt(history):
    """Creates a baseline prompt using History and Hidden Chain-of-Thought."""

    # Format history cleanly
    history_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(history)])

    # The prompt demands reasoning, but forces absolute strictness on the final list
    return f"""You are an AI simulating a human search engine user.

--- CONTEXT ---
Here is the user's search history for the session so far (in chronological order):
{history_str}

--- TASK ---
Based ONLY on this history, predict the very next search query this user would type.
You must predict exactly 5 diverse query candidates that remain semantically similar to the original intent. Rank them in descending order of confidence.

You MUST think step-by-step. First, analyze the logical progression of the search history. Then, generate the 5 queries.
Format your output EXACTLY like this:
Reasoning: [1-2 sentences explaining your logic]
Queries:
1. [First predicted query ONLY. NO explanations. NO confidence scores.]
2. [Second predicted query ONLY. NO explanations. NO confidence scores.]
3. [Third predicted query ONLY. NO explanations. NO confidence scores.]
4. [Fourth predicted query ONLY. NO explanations. NO confidence scores.]
5. [Fifth predicted query ONLY. NO explanations. NO confidence scores.]

CRITICAL RULE: Do NOT include quotes, parentheses, confidence levels, or reasoning inside the numbered list. The list must contain ONLY the raw search text."""

def get_ollama_prediction_list(prompt_text):
    """Calls Llama 3, lets it think, but only returns the final 5 queries."""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt_text}
        ])
        raw_text = response['message']['content'].strip()

        queries_text = raw_text

        # 1. SPLIT THE TEXT: We separate the reasoning from the final list
        if "Queries:" in raw_text:
            parts = raw_text.split("Queries:")
            # We completely ignore parts[0] (the reasoning) and only look at parts[1]
            queries_text = parts[1].strip()

        # 2. STRICT REGEX EXTRACTION: Grab only the numbered list
        queries = []
        for line in queries_text.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.*)', line)
            if match:
                # Clean up any stray quotes just in case the AI disobeys
                clean_query = match.group(1).strip(' "\'')
                # Extra safety: remove trailing parentheses if the AI still tries to add confidence scores
                clean_query = re.sub(r'\s*\(.*?\)$', '', clean_query)
                queries.append(clean_query)

        # 3. FALLBACK: Ensure exactly 5 items so TIRA JSON doesn't crash
        while len(queries) < 5:
            fallback = queries[0] if queries else "fallback query"
            queries.append(fallback)

        return queries[:5] # Notice we ONLY return the queries!
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ["error"] * 5

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    filepath = 'data/task3_longeval_usim-sessions-09-11_2025.csv'

    print("Loading data via persona_NLP.py (Ignoring Personas)...")
    df_raw, session_personas = process_personas(filepath)

    # Prepare the TIRA JSON structure
    submission_data = {
        "meta": {
            "team_name": "Promptly",
            "description": "CoT Baseline: History + Chain of Thought. Predicting top 5. [09-11-2025 test set]",
            "run_name": "baseline-cot-snapshot-14"
        }
    }

    print("\nStarting HIDDEN CoT BASELINE TOP-5 Simulation...\n" + "="*50)

    for session_id in session_personas.keys():

        # Get chronological search history for this session
        session_queries = df_raw[df_raw['session_id'] == session_id]['query_text'].tolist()

        if len(session_queries) < 2:
            continue

        history_to_show = session_queries

        print(f"Session {int(session_id)} | CoT BASELINE (Hidden Reasoning)")
        print(f"   Input History: {history_to_show}")

        # Create Prompt & Predict
        prompt = create_cot_top5_prompt(history_to_show)

        # The reasoning is happening, but we only get the clean list back!
        predictions_list = get_ollama_prediction_list(prompt)

        # Clean Terminal Output! No more massive blocks of text.
        print(f"   Predicted Top 5:")
        for idx, q in enumerate(predictions_list):
            print(f"     {idx+1}. {q}")
        print("=" * 50)

        # Integer ID Fix: Convert 44.0 to "44" so TIRA doesn't crash
        clean_id = str(int(float(session_id)))
        submission_data[clean_id] = predictions_list

    # Save the final JSON
    output_filename = 'snapshot-14.jsonl'
    with open(output_filename, 'w', encoding='utf-8') as f:
        for key, value in submission_data.items():
            f.write(json.dumps({key: value}) + '\n')

    print(f"\nSimulation complete. Results formatted and saved to '{output_filename}'.")
