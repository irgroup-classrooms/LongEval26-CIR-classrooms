# CIR at LongEval 2026: Ad-Hoc Scientific Retrieval, Topic Extraction From Query Logs, and User Simulation

[![Venue: LongEval 2026](https://img.shields.io/badge/Venue-LongEval%2026-blue.svg)](https://clef-longeval.github.io/)


> **CIR at LongEval 2026: Ad-Hoc Scientific Retrieval, Topic Extraction From Query Logs, and User Simulation**
> Can Bakirci, Alexander Brückner, Joshua Matthew Christian, Nadja Kahsai Debrezion, Carmelo Heinrich Antonino Di Pino, Abdelilah Imam, Kübra Kartal, Aaron Benjamin Klatt, Robin Klinkhammer, Yannick Mählmann, Lennard Michel, Emirhan Sahin, Leon Damian Thies, Vico Timmer, Andreas Kruff, Jüri Keller.
> *CLEF 2026 Working Notes*.


# Overview of Prompts Used in Experiments

This section provides an overview of the prompts used across different experimental setups for the CLEF Workshop paper. The referenced files contain the full prompt implementations.

---

## 1. Prompts – Team Promptly

The prompts developed by Team Promptly are divided into two main categories:

- Baseline approaches  
- NLP Tag–based approaches  

The NLP Tag–based prompts additionally include:
- A **TAG glossary** defining how tags should be interpreted and used
- **Persona information**
- **NLP tag annotations**

Across both categories, different reasoning strategies are applied:
- No reasoning  
- Chain-of-Thought (CoT)  
- Self-Refinement  

### 1.1 Baseline Prompts

- Baseline Prompt  
  [Promptly/scripts/submission_baseline_agent.py](Promptly/scripts/submission_baseline_agent.py)

- Baseline Prompt + CoT  
  [Promptly/scripts/submission_baseline_cot.py](Promptly/scripts/submission_baseline_cot.py)

- Baseline Prompt + Self-Refinement  
  [Promptly/scripts/submission_baseline_refine_cot.py](Promptly/scripts/submission_baseline_refine_cot.py)

### 1.2 NLP Tag Prompts

- NLP Tag Prompt  
  [Promptly/scripts/submission_NLP_agent_sim.py](Promptly/scripts/submission_NLP_agent_sim.py)

- NLP Tag Prompt + CoT  
  [Promptly/scripts/submission_NLP_agent_cot_sim.py](Promptly/scripts/submission_NLP_agent_cot_sim.py)

- NLP Tag Prompt + Self-Refinement  
  [Promptly/scripts/submission_NLP_agent_refine_cot_sim.py](Promptly/scripts/submission_NLP_agent_refine_cot_sim.py)

---

## 2. Prompts – Team Split ’n Simulate

This team uses two main prompt types:

### 2.1 Persona Classification Prompt

Used to classify the persona of a given session.

- Function: `classify_single_session`
- File: [SplitNSimulate/simulation/longeval_simulation_train.py](SplitNSimulate/simulation/longeval_simulation_train.py)

This prompt includes detailed descriptions of predefined user personas, focusing on query behavior patterns and user characteristics.

### 2.2 Next-Query Prediction Prompt

Used to predict the next query in a session.

- Function: `create_prediction_prompt`  
- File: [SplitNSimulate/simulation/longeval_simulation_train.py](SplitNSimulate/simulation/longeval_simulation_train.py)

This prompt incorporates:
- Assigned persona  
- Persona description  
- Additional session context  

---

## 3. Prompts – Team JOINorDIE

Team JOINorDIE uses different prompt designs depending on the task.

---

### 3.1 Task 2: Topic Generation

For Task 2, two different prompt implementations were used to generate TREC-style topics. The prompt design varies depending on the underlying LLM backend:

- **Ollama-based prompt**  
  [JOINorDIE/05_LLM_Topic_Generierung_Ollama.ipynb](JOINorDIE/05_LLM_Topic_Generierung_Ollama.ipynb)

- **OpenAI-based prompt**  
  [JOINorDIE/05_LLM_Topic_Generierung_openai.ipynb](JOINorDIE/05_LLM_Topic_Generierung_openai.ipynb)

Both versions utilize additional contextual evidence in the form of TF-IDF-based terms.  
This information is provided via the variable `evidence` and can be found here:

- Evidence dataset:  
  [JOINorDIE/data/evidence_topic_generation.csv](JOINorDIE/data/evidence_topic_generation.csv)

---

### 3.2 Task 3: Next-Query Prediction

For Task 3, the prompt design is based on session-level query prediction. The following components are used:

- Topic generation from session history  
  - Implemented in: `get_topic()`

- Next-query prediction prompts  
  - Implemented in:
    - `build_prompt_A()`
    - `build_prompt_B()`

All implementations can be found in:

- [JOINorDIE/run_all_18.py](JOINorDIE/run_all_18.py)
