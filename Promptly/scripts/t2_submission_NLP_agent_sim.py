import json
import re
import pandas as pd
import ollama

# Import your math and tagging logic from the other file
from persona_NLP import process_personas

MODEL_NAME = "llama3"

def create_dynamic_top5_prompt(base_persona, tags, history):
    """Creates a highly specific persona prompt WITHOUT Chain-of-Thought."""
    tag_string = f"[{', '.join(tags)}]" if tags else "[Standard]"

    # Format history cleanly for the LLM
    history_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(history)])

    # The prompt includes the Glossary, but asks ONLY for the final list
    return f"""You are an AI simulating a human search engine user.
Your base behavior is: {base_persona}.
Keep your simulated next query strictly aligned with the following behavioral tags: {tag_string}.

--- TAG GLOSSARY ---
* [Conversational]: Write the query as a full sentence or natural question.
* [Keyword-focused]: Use very short, punchy noun phrases (1-3 words).
* [Deep Explorer]: You are digging deep into a topic; make lateral jumps to related sub-topics.
* [Quick Fact-Finder]: You just want a fast answer; keep it highly specific.
* [Frustrated]: You are annoyed because your past queries failed. Drastically change your vocabulary or simplify.
* [Successful]: You found good results previously. Carefully refine your current path.
* [Rapid Skimmer]: You didn't read the previous results. Act impatient and slightly tweak your last query.
* [Deep Reader]: You spent a long time reading an article. Your next query should be highly educated and specific based on the previous topic.

--- CONTEXT ---
Here is the user's search history for the session so far (in chronological order):
{history_str}

--- TASK ---
Based on this history and your specific persona traits, predict the very next search query this user would type.
You must predict exactly 5 diverse query candidates that remain semantically similar to the original intent but reflect your persona. Rank them in descending order of confidence.

Format your output EXACTLY as a numbered list from 1 to 5, like this:
1. [First predicted query ONLY. NO explanations. NO confidence scores.]
2. [Second predicted query ONLY. NO explanations. NO confidence scores.]
3. [Third predicted query ONLY. NO explanations. NO confidence scores.]
4. [Fourth predicted query ONLY. NO explanations. NO confidence scores.]
5. [Fifth predicted query ONLY. NO explanations. NO confidence scores.]

CRITICAL RULE: Do NOT include quotes, parentheses, confidence levels, reasoning, or introductory text. The list must contain ONLY the raw search text."""

def get_ollama_prediction_list(prompt_text):
    """Calls Llama 3 and strictly extracts the 5 queries directly."""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt_text}
        ])
        raw_text = response['message']['content'].strip()

        # STRICT REGEX EXTRACTION: Prevent spillover!
        queries = []
        for line in raw_text.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.*)', line)
            if match:
                # Clean up stray quotes
                clean_query = match.group(1).strip(' "\'')
                # Extra safety: remove trailing parentheses if the AI still tries to add confidence scores
                clean_query = re.sub(r'\s*\(.*?\)$', '', clean_query)
                queries.append(clean_query)

        # FALLBACK: Ensure exactly 5 items so JSON doesn't break
        while len(queries) < 5:
            fallback = queries[0] if queries else "fallback query"
            queries.append(fallback)

        return queries[:5] # Strictly return 5
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ["error"] * 5

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    filepath = 'data/task3_longeval_usim-sessions-09-11_2025.csv'

    print("Loading data and generating personas via persona_NLP.py...")
    df_raw, session_personas = process_personas(filepath)

    # Prepare the JSON structure mandated by TIRA guidelines
    submission_data = {
        "meta": {
            "team_name": "Promptly",
            "description": "Persona Simulation (NLP Tags) NO CoT. Predicting top 5. [09-11-2025 test set]",
            "run_name": "nlp-persona-history-only-snapshot-16"
        }
    }

    print("\nStarting Persona TOP-5 Simulation (NO CoT)...\n" + "="*50)

    for session_id, persona_data in session_personas.items():
        base_persona = persona_data['base']
        tags = persona_data['tags']

        # Get chronological search history for this session
        session_queries = df_raw[df_raw['session_id'] == session_id]['query_text'].tolist()

        if len(session_queries) < 2:
            continue

        history_to_show = session_queries

        print(f"Session {int(session_id)} | {base_persona} {tags}")
        print(f"   Input History: {history_to_show}")

        # Create Prompt & Predict
        prompt = create_dynamic_top5_prompt(base_persona, tags, history_to_show)
        predictions_list = get_ollama_prediction_list(prompt)

        print(f"   Predicted Top 5:")
        for idx, q in enumerate(predictions_list):
            print(f"     {idx+1}. {q}")
        print("-" * 50)

        # Integer Fix
        clean_id = str(int(float(session_id)))
        submission_data[clean_id] = predictions_list

    # Save the final JSON file
    output_filename = 'snapshot-16.jsonl'
    with open(output_filename, 'w', encoding='utf-8') as f:
        for key, value in submission_data.items():
            f.write(json.dumps({key: value}) + '\n')

    print(f"\nSimulation complete. Results formatted and saved to '{output_filename}'.")
