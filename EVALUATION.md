# Evaluation Pipeline for Question Answering

This document explains how the evaluation workflow in `task_eval/evaluate_qa.py` orchestrates model inference, per-question scoring, and aggregated reporting for question answering (QA) over generated conversations.


## High-Level Overview

The driver script `task_eval/evaluate_qa.py` performs the following sequence:

1. Parses CLI arguments that describe the model to run, input dataset, and output destinations.
2. Configures API keys or model pipelines for OpenAI, Anthropic, Gemini, or HuggingFace models.
3. Loads conversation samples from the data file and optionally resumes from a partially completed output file.
4. Dispatches to a model-specific helper (`get_gpt_answers`, `get_claude_answers`, `get_gemini_answers`, or `get_hf_answers`) to obtain predictions for every QA pair.
5. Scores each QA using `task_eval.evaluation.eval_question_answering`, appending metrics such as F1 and (optionally) recall.
6. Persists updated predictions, then runs `task_eval.evaluation_stats.analyze_aggr_acc` to generate aggregated accuracy statistics and memory-depth breakdowns (or recall summaries for RAG runs).


## Command-Line Interface

Run the script from the repository root:

```bash
python task_eval/evaluate_qa.py \
  --model gpt-3.5-turbo \
  --data-file data/locomo10.json \
  --out-file outputs/gpt-3.5-turbo.json
```

Supported arguments:

| Flag | Required | Description |
| --- | --- | --- |
| `--model` | ✅ | Model identifier; determines which helper executes and what credentials are needed. |
| `--data-file` | ✅ | Path to the conversation + QA JSON dataset. |
| `--out-file` | ✅ | Destination JSON file for predictions and per-QA metrics. Intermediate directories are created automatically. |
| `--batch-size` |  | Number of QA pairs processed together. For RAG mode, this must be 1. |
| `--use-rag` |  | Enable retrieval-augmented generation. Requires `--rag-mode`, `--emb-dir`, `--retriever`, and optionally `--top-k`. |
| `--rag-mode` |  | One of `summary`, `dialog`, or `observation`; selects which embedding index to use. |
| `--emb-dir` |  | Directory containing (or storing) serialized embedding databases. |
| `--top-k` |  | Number of retrieved contexts to attach in RAG mode (default 5). |
| `--retriever` |  | Embedding model name passed to `task_eval.rag_utils.get_embeddings` (default `contriever`). |
| `--use-4bit` |  | When set, loads HuggingFace models with 4-bit quantization. |
| `--overwrite` |  | Force regeneration of predictions even if they already exist in the output file. |


## Data Model Expectations

Each entry in `--data-file` is a dictionary shaped roughly as follows:

```json
{
  "sample_id": "conv-26",
  "conversation": {
    "session_1": [
      {
        "dia_id": "D1:1",
        "speaker": "Caroline",
        "text": "I just joined an activist group...",
        "blip_caption": "Caroline holding a protest sign"
      }
      // … additional dialog turns and sessions …
    ],
    "session_1_date_time": "20 July, 2023 20:56"
    // … optional session summaries, observations, etc. …
  },
  "qa": [
    {
      "question": "When did Melanie attend the concert?",
      "answer": "14 August, 2023",
      "evidence": ["D11:3"],
      "category": 2
    }
    // … more QA entries …
  ]
}
```

Important fields:

- `sample_id` must be unique across the dataset and is the key used to merge with existing outputs.
- `conversation` stores chronological dialog sessions; helpers construct prompts from these sessions (with optional BLIP image captions).
- `qa` is a list of question objects that include:
  - `question`: natural language query.
  - `answer`: reference answer string (or comma-separated list for multi-answer questions).
  - `evidence`: dialog turn identifiers (`D<session>:<turn>`) pointing to supporting context.
  - `category`: numeric tag that determines the evaluation strategy (see *Scoring Logic*).


## Execution Flow

1. **Argument parsing** ensures required flags are present, then logs the selected model for traceability.
2. **Credential setup**:
   - Models containing `gpt` call `global_methods.set_openai_key()`, expecting `OPENAI_API_KEY`.
   - `claude` models call `set_anthropic_key()` (placeholder for Anthropic configuration).
   - `gemini` models call `set_gemini_key()` (expects `GOOGLE_API_KEY`) and instantiate `google.generativeai.GenerativeModel`.
   - HuggingFace identifiers containing `gemma`, `llama`, or `mistral` load a pipeline via `task_eval.hf_llm_utils.init_hf_model` (requires `HF_TOKEN` and optional `--use-4bit` quantization).
3. **Input & resume handling**:
   - Loads the dataset JSON into `samples`.
   - Determines `prediction_key` and `model_key` strings (which embed RAG settings if applicable) to avoid collisions between runs.
   - If `--out-file` exists, it is read into `out_samples` for resuming; predictions are skipped unless `--overwrite` is passed.
4. **Per-sample processing**:
   - Each sample’s QA list is cloned into `out_data` (preserving prior predictions when resuming).
   - Model dispatch:
     - `get_gpt_answers` (OpenAI) batches questions, builds prompts with conversation context, and records generated answers plus optional retrieved context IDs.
     - `get_claude_answers` and `get_gemini_answers` follow analogous patterns for their providers.
     - `get_hf_answers` runs local transformers pipelines, managing chat templates and 4-bit quantized loading when requested.
5. **Scoring**:
   - `eval_question_answering` traverses the QA list, computing per-question F1 (exact-match style) according to category:
     - Category 1 (multi-hop): splits composite answers and averages per-component F1.
     - Categories 2–4: single-hop variants evaluated with token-level F1.
     - Category 5 (adversarial): verifies that the model abstains when information is absent.
   - Scores are stored under `<model_key>_f1`. When RAG is enabled and recall is returned, `<model_key>_recall` captures supporting-context coverage.
6. **Persistence & stats**:
   - Updated `out_samples` are written with `json.dump(..., indent=2)` to ensure readability.
   - `analyze_aggr_acc` computes category-level aggregates, memory-depth curves, and (for RAG) recall summaries, saving them to `--out-file` with `_stats.json` suffix.


## Output Artifacts

`--out-file` is a JSON array where each element mirrors the input sample but includes model predictions and metrics. Example excerpt:

```json
{
  "sample_id": "conv-26",
  "qa": [
    {
      "question": "When did Melanie attend the concert?",
      "answer": "14 August, 2023",
      "evidence": ["D11:3"],
      "category": 2,
      "gpt-3.5-turbo_prediction": "August 14, 2023",
      "gpt-3.5-turbo_f1": 1.0
    },
    {
      "question": "What motivates Melanie's pottery hobby?",
      "answer": "It helps her express her feelings.",
      "evidence": ["D12:5"],
      "category": 4,
      "gpt-3.5-turbo_prediction": "expressing her feelings",
      "gpt-3.5-turbo_f1": 1.0
    }
  ]
}
```

Aggregated statistics are emitted to `--out-file` with `_stats.json` appended (e.g., `outputs/gpt-3.5-turbo_stats.json`). This file records:

- `category_counts`: how many questions per category were evaluated.
- `cum_accuracy_by_category`: summed accuracy scores (divide by counts to get mean F1).
- `category_counts_by_memory` and `cum_accuracy_by_category_by_memory`: bucketed by farthest evidence distance (only when not using RAG).
- `recall_by_category`: average supporting-evidence recall when RAG is active.
- `context_length_counts`: distribution of temporal span between supporting evidence in multi-hop questions.


## Retrieval-Augmented Generation (RAG)

Enabling `--use-rag` alters the flow:

- Only batch size 1 is supported to simplify context retrieval.
- `task_eval.gpt_utils.prepare_for_rag` loads (or synthesizes) embedding databases based on `--rag-mode`:
  - `summary`: expects per-session summary embeddings.
  - `dialog`: builds turn-level embeddings on demand if missing.
  - `observation`: targets observational memory embeddings.
- `get_rag_context` retrieves the top `--top-k` contexts, returns both human-readable snippets and dialog IDs. Retrieved IDs are stored under `<prediction_key>_context` for downstream analysis.
- `eval_question_answering` computes recall based on whether the retrieved context covers the annotated evidence list.
- `analyze_aggr_acc` switches to recall-by-category reporting instead of memory-depth buckets.

Make sure the embedding directory is writable (for dialog mode) and contains the required pre-computed `.pkl` files for summary or observation modes.


## Model-Specific Requirements

| Model Type | Helper | Credentials / Setup | Notes |
| --- | --- | --- | --- |
| OpenAI GPT (`gpt-3.5`, `gpt-4`, etc.) | `task_eval.gpt_utils.get_gpt_answers` | `OPENAI_API_KEY` | Supports batching and RAG. Token budgets handled via `tiktoken`. |
| Anthropic Claude (`claude-3-*`) | `task_eval.claude_utils.get_claude_answers` | `ANTHROPIC_API_KEY` | Uses `global_methods.run_claude` under the hood. |
| Google Gemini (`gemini-pro-1.0`, etc.) | `task_eval.gemini_utils.get_gemini_answers` | `GOOGLE_API_KEY` | Instantiates `genai.GenerativeModel`; see `global_methods.run_gemini`. |
| HuggingFace (LLaMA, Mistral, Gemma) | `task_eval.hf_llm_utils.get_hf_answers` | `HF_TOKEN` | Optional `--use-4bit` to lower memory. Only batch size 1 implemented. |


## Scoring Logic Details

`eval_question_answering` applies tailored metrics per category:

- **Category 1 – Multi-hop reasoning:** Splits comma-separated answers and averages the maximum token-level F1 over candidate pairings.
- **Category 2 – Temporal:** Emphasizes date normalization, using F1 between normalized tokens.
- **Category 3 – Disambiguation:** Applies single-answer F1 after trimming semicolon-separated metadata.
- **Category 4 – Open-domain:** Uses token-level F1 against the canonical answer.
- **Category 5 – Adversarial / Unanswerable:** Rewards abstention by checking for strings like “no information available”.

All F1 scores are rounded to three decimal places before being stored. When recall data are available (RAG mode with evidence annotations), scores are rounded similarly.


## Extending the Pipeline

- **Adding a new model provider:** Implement `get_<provider>_answers` that matches the signature of existing helpers and update the dispatch logic in `evaluate_qa.py`. Ensure you set any necessary API keys or initialize pipelines before entering the sample loop.
- **Adding new metrics:** Modify `task_eval.evaluation.eval_question_answering` to compute additional metrics, then push them into each QA record inside the processing loop. Remember to extend `analyze_aggr_acc` if the aggregate report must include the new metrics.
- **Custom RAG retrievers:** Update `task_eval.gpt_utils.prepare_for_rag` to recognize new `--retriever` names and integrate the corresponding embedding backends.


## Troubleshooting

- **Missing credentials:** Verify environment variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`) are exported before running the script.
- **Resuming runs:** If you need to re-evaluate only unfinished QA pairs, omit `--overwrite`; existing predictions are preserved. Use `--overwrite` to regenerate everything.
- **Output validation:** After execution, inspect the `_stats.json` file for category coverage and aggregate accuracy. Unexpected zeros often indicate that predictions were skipped or metrics were not written.
- **RAG embedding errors:** Ensure precomputed embedding `.pkl` files exist for `summary` and `observation` modes. For `dialog` mode the script will synthesize and cache embeddings; confirm the directory is writable.


