# CLEF LongEval Task 3: Next-Query Prediction (Team: Promptly)

**Author:** Joshua Matthew Christian, Nadja Kahsai Debrezion  
**Institution:** TH Köln

## 📌 Project Context
This repository contains the code implementations for Task 3 of the CLEF LongEval competition. The objective of this task is to predict a user's next search query based on their chronological search history within a single session. 

The core research question of this project is to evaluate whether complex LLM prompt engineering and reasoning techniques outperform simple, history-based heuristics for short-text search prediction. 

## 🛠️ Implemented Architectures
The codebase contains the execution scripts for 6 distinct experimental runs, testing different levels of prompt complexity:

1. **Simple Baseline:** Pure history-based heuristic prediction.
2. **Baseline + Hidden Chain-of-Thought (CoT):** Forcing the LLM to use a reasoning scratchpad before outputting predictions.
3. **Baseline + Hidden CoT + Self-Refinement:** Adding a self-critique loop to the reasoning process.
4. **NLP Persona Agent:** Using a deterministic decision tree to assign behavioral tags (e.g., `[Conversational]`, `[Deep Explorer]`) based on session metrics.
5. **NLP Persona + Hidden CoT**
6. **NLP Persona + Hidden CoT + Self-Refinement**

## 📊 Data & Evaluation Note
The models generated in these scripts were designed to be evaluated against the `usim-sessions-train` dataset. The evaluation strategy isolates the absolute final query of each chronological session to use as the hidden ground-truth target, measuring performance via ROUGE-L (lexical accuracy) and BERTScore (semantic intent).