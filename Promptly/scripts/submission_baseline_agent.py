import json
import re
import ollama

# We still use your parser just to get the clean, chronological data easily
from persona_NLP import process_personas

MODEL_NAME = "llama3" 

def create_top5_prompt(history):
    """Creates a prompt asking for 5 diverse query candidates."""
    
    # Format history cleanly
    history_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(history)])
    
    return f"""You are simulating a human search engine user.

Here is the user's search history for the session so far (in chronological order):
{history_str}

Based on this history, predict the very next search query this user would type. 
You must predict exactly 5 diverse query candidates that remain semantically similar to the original intent. Rank them in descending order of confidence.

Format your output EXACTLY as a numbered list from 1 to 5, like this:
1. [First predicted query]
2. [Second predicted query]
3. [Third predicted query]
4. [Fourth predicted query]
5. [Fifth predicted query]

Return ONLY the numbered list. Do not include quotes, reasoning, or introductory text."""

def get_ollama_prediction_list(prompt_text):
    """Calls Llama 3 and parses the response into a list of exactly 5 strings."""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt_text}
        ])
        raw_text = response['message']['content'].strip()
        
        # Use Regex to find lines that start with a number (e.g., "1. query")
        queries = []
        for line in raw_text.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.*)', line)
            if match:
                # Strip out any random quotes the LLM might have added
                clean_query = match.group(1).strip(' "\'')
                queries.append(clean_query)
        
        # Fallback: Ensure we ALWAYS have exactly 5 items so the JSON doesn't break
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
    filepath = 'data/task3_longeval_usim-sessions-train.csv'
    
    print("Loading data via persona_NLP.py (Ignoring Personas)...")
    df_raw, session_personas = process_personas(filepath)
    
    # Prepare the JSON structure mandated by the guidelines
    submission_data = {
        "meta": {
            "team_name": "Promptly",
            "description": "Simple Baseline: History only, predicting top 5 candidates.",
            "run_name": "simple-baseline-history-only-snapshot-1"
        }
    }
    
    print("\nStarting TOP-5 BASELINE Simulation...\n" + "="*50)

    for session_id in session_personas.keys():
        
        # Get chronological search history for this session
        session_queries = df_raw[df_raw['session_id'] == session_id]['query_text'].tolist()
        
        # Adhere to guidelines: withhold the last query as ground truth
        if len(session_queries) < 2:
            continue
            
        history_to_show = session_queries[:-1]
        
        print(f"Session {int(session_id)}")
        print(f"   Input History: {history_to_show}")
        
        # Create Prompt & Predict
        prompt = create_top5_prompt(history_to_show)
        predictions_list = get_ollama_prediction_list(prompt)
        
        print(f"   Predicted Top 5:")
        for idx, q in enumerate(predictions_list):
            print(f"     {idx+1}. {q}")
        print("-" * 50)
        
        # Add to the JSON dictionary (Keys must be strings)
        submission_data[str(int(session_id))] = predictions_list
    
    # Save the final JSON file according to the naming convention
    output_filename = 'snapshot-1.jsonl'
    with open(output_filename, 'w', encoding='utf-8') as f:
        for key, value in submission_data.items():
            f.write(json.dumps({key: value}) + '\n')
        
    print(f"\nSimulation complete. Results formatted and saved to '{output_filename}'.")