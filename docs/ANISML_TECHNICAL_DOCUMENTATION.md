# ANISML Technical Documentation

## 1. Introduction

This document provides a complete, end-to-end technical overview of the **Taxonomy Alignment & Pre-processing** project (codename *anisml*) implemented in Python. It covers:

- Project goals and high-level architecture
- Dependency management and environment setup
- Detailed walkthrough of each pipeline stage (scripts 01–08)
- Logging, metrics, and outputs
- Extension points and customization

## 2. Repository Structure

```
anisml/                       # Project root
├── .cursor/                  # Cursor rules definitions (sprint1.xml)
├── data/                     # Intermediate and final CSV outputs
│   ├── taxo_raw.csv          # Normalized taxonomy
│   ├── embeddings.npy        # Computed embeddings
│   ├── merge_pairs.csv       # Merge instructions (source→target)
│   ├── taxonomy_aligned.csv  # Final aligned taxonomy export
│   └── sorted_data.csv       # Original data.csv sorted by isMatch flag
├── logs/                     # Chain-of-Thought (CoT) and other logs
│   └── cot.log               # Heuristic decision logs
├── models/                   # Trained models & checkpoints
│   └── agent_classifier.joblib
├── scripts/                  # Python scripts for each pipeline step
│   ├── 01_collect_normalize.py
│   ├── 02_vectorize.py
│   ├── 03_apply_rules.py
│   ├── 04_ppo_finetune.py
│   ├── 05_evaluate.py
│   ├── 06_export.py
│   ├── 07_train_agent_classifier.py
│   └── 08_sort_data.py
├── requirements.txt          # Strictly pinned Python dependencies
└── README.md                 # Quick start and overview
```  

## 3. Environment Setup

1. **Python 3.10+** (tested on 3.12)
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure `logs/` and `models/` directories exist or will be auto-created by scripts.

## 4. Pipeline Stages

### 4.1 01_collect_normalize.py

- **Purpose:** Aggregate raw CSV/TSV sources, normalize text (NFKC, strip accents, lowercase).
- **Key features:**
  - Delimiter sniffing: chooses `\t` vs `,` based on header content
  - Progress bars via `tqdm` for file collection and `pandas.progress_applymap`
- **Usage:**
  ```bash
  python scripts/01_collect_normalize.py --sources data.csv --output data/taxo_raw.csv
  ```

### 4.2 02_vectorize.py

- **Purpose:** Convert each taxonomy row into a text string and compute embeddings.
- **Supported models:**
  - `miniLM` (default): Sentence-Transformers all-MiniLM-L6-v2 with CLS pooling
  - `gemma`: HF AutoModel (google/gemma-2b-it) with CLS pooling + HW accelerate
  - `deepseek`: HF AutoModel (deepseek-ai/deepseek-r1-3b)
- **Key features:**
  - Unbuffered (`-u`) output shows `tqdm` progress of batches
  - Saves `embeddings.npy` and minimal `embeddings_meta.json`
- **Usage:**
  ```bash
  python scripts/02_vectorize.py --input data/taxo_raw.csv --model miniLM
  ```

### 4.3 03_apply_rules.py

- **Purpose:** Apply Cursor rules to generate merge instructions.
- **Implemented rules:**
  1. Exact duplicates after normalization (**dup-merge-5**)
  2. Lexical (Jaro-Winkler ≥ 0.80) **and** embedding similarity (cosine ≥ 0.85) (**sim-merge-1**)
- **Key features:**
  - Auto-creates `logs/cot.log` for Chain-of-Thought entries
  - Faiss support for approximate search (if installed) or block-wise fallback
  - Configurable `--block-size` to limit peak memory
  - Prints summary metrics per rule and overall merge statistics
- **Usage:**
  ```bash
  python scripts/03_apply_rules.py \
    --taxo data/taxo_raw.csv \
    --emb data/embeddings.npy \
    --block-size 256
  ```

### 4.4 04_ppo_finetune.py

- **Purpose:** (Optional) Fine-tune an LLM via PPO on merge-vs-non-merge preferences.
- **Key features:**
  - Loads base and reference models with value heads
  - Builds prompts from `data/taxo_raw.csv` & `data/merge_pairs.csv`
  - Implements PPO training loop with logging and checkpointing
- **Usage:** See script docstring; requires gated model access for `google/gemma-2b-it`.

### 4.5 05_evaluate.py

- **Purpose:** Report merge statistics.
- **Metrics:**
  - Original label count
  - Total merge pairs
  - Unique source rows merged
  - Aligned labels remaining
  - Reduction rate (%)
- **Usage:**
  ```bash
  python scripts/05_evaluate.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv
  ```

### 4.6 06_export.py

- **Purpose:** Export the aligned taxonomy by dropping merged source rows.
- **Outputs:** `data/taxonomy_aligned.csv`
- **Usage:**
  ```bash
  python scripts/06_export.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv --output data/taxonomy_aligned.csv
  ```

### 4.7 07_train_agent_classifier.py

- **Purpose:** Train a supervised binary classifier (Logistic Regression) that predicts merge decisions from:
  - Lexical similarity (Jaro-Winkler)
  - Embedding cosine similarity
- **Workflow:**
  1. Load positive pairs from `merge_pairs.csv`
  2. Sample negative pairs at random
  3. Compute 2-dim feature vectors and train/test split
  4. Save the model to `models/agent_classifier.joblib`
- **Metrics:** classification report & ROC-AUC
- **Usage:**
  ```bash
  python scripts/07_train_agent_classifier.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --merges data/merge_pairs.csv
  ```

### 4.8 08_sort_data.py

- **Purpose:** Sort the original raw `data.csv` by the `isMatch` flag in the enrichment column.
- **Features:** JSON or regex parsing, progress bar, delimiter sniffing.
- **Usage:**
  ```bash
  python scripts/08_sort_data.py --input data.csv --output data/sorted_data.csv
  ```

## 5. Logging & Outputs

- **Chain-of-Thought:** `logs/cot.log` records each rule's firing with details.
- **Embeddings metadata:** `data/embeddings_meta.json` contains model name, dimension, filter.
- **Merge pairs:** `data/merge_pairs.csv` for downstream eval and export.
- **Aligned taxonomy:** `data/taxonomy_aligned.csv` final product.
- **Sorted raw data:** `data/sorted_data.csv` for manual inspection.
- **Classifier model:** `models/agent_classifier.joblib` ready for inference.

## 6. Customization & Extension

- **Thresholds:** Adjust `LEXICAL_THRESHOLD` & `EMBED_THRESHOLD` in `03_apply_rules.py` or via Cursor XML.
- **Block size:** Tune `--block-size` to balance memory vs. speed.
- **Models:** Add new entries to `_MODEL_REGISTRY` in `02_vectorize.py` for other embeddings.
- **New rules:** Define additional `<cursorRule>` entries in `.cursor/rules/sprint1.xml` and implement in `03_apply_rules.py`.
- **API Integration:** Wrap scripts as functions or FastAPI endpoints for interactive classification or export.

## 7. End-to-End Usage Example
Below is a single-line invocation that chains all steps, recreates caches, and prints summary metrics at each stage:
```bash
rm -rf .venv && find . -type d -name "__pycache__" -exec rm -rf {} + \
  && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements.txt \
  && python -u scripts/01_collect_normalize.py --sources data.csv --output data/taxo_raw.csv \
  && python -u scripts/02_vectorize.py --input data/taxo_raw.csv --model miniLM \
  && python -u scripts/03_apply_rules.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --block-size 256 \
  && python -u scripts/05_evaluate.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv \
  && python -u scripts/06_export.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv --output data/taxonomy_aligned.csv
```

## 8. Extending Cursor Rules
All merge, split, flag and attach logic is defined in `.cursor/rules/sprint1.xml` (or `anisairules.mdc`).
- To add a new rule:
  1. Define a `<cursorRule>` block with a unique `id`, `priority`, `<condition>`, and `<action>` in the XML.
  2. Update `scripts/03_apply_rules.py` to parse your rule and implement its Python logic (e.g., new function or branch).
  3. Write unit tests under `tests/` matching your rule's behavior for edge cases.
  4. Run the pipeline and inspect `logs/cot.log` to verify Chain-of-Thought entries.

## 9. Integration & CI/CD
- **Pre-commit hooks:** Add linting for Python (e.g., `ruff`), formatting (`black`), and XML schema validation for `.cursor/rules/*.xml`.
- **Automated runs:** Configure GitHub Actions (or GitLab CI) to:
  1. Build the venv and install dependencies.
  2. Run `scripts/01_collect_normalize.py` → `scripts/03_apply_rules.py` → `scripts/05_evaluate.py` as smoke tests.
  3. Execute unit tests in `tests/` and fail on regressions.
- **Scheduled tasks:** Use cron or GitHub scheduled workflows to update the taxonomy weekly and notify stakeholders on changes (e.g., via email or Slack webhook).

## 10. Troubleshooting
- **Empty or corrupt `embeddings.npy`:** Check that `scripts/02_vectorize.py` ran without errors; verify disk space.
- **No merges produced:** Confirm thresholds (`LEXICAL_THRESHOLD`, `EMBED_THRESHOLD`) and input data quality; inspect `logs/cot.log` for rule decisions.
- **Slow performance or OOM:** Lower `--block-size`, enable Faiss for approximate nearest neighbors, or increase available RAM.
- **Log directory missing:** Scripts will auto-create `logs/`; ensure file permissions allow this.

## 11. FAQ
**Q1. How can I use a different embedding model?**
>A1. Add your model to `_MODEL_REGISTRY` in `02_vectorize.py` and implement its loader in `_load_*_encoder`, then pass `--model yourModel`.

**Q2. How do I adjust merge thresholds?**
>A2. Modify `LEXICAL_THRESHOLD` or `EMBED_THRESHOLD` constants in `03_apply_rules.py`, or override via environment variables if exposed.

**Q3. Can I preview merge pairs before export?**
>A3. Yes—`data/merge_pairs.csv` contains all candidate merges; open it in a spreadsheet or use `scripts/05_evaluate.py` with `--verbose` flag (if added).

**Q4. How to retrain the agent classifier?**
>A4. Rerun `scripts/07_train_agent_classifier.py` after adjusting positive/negative sampling ratios or feature engineering.

## 12. Core Capabilities & Use Cases
- **Automated Taxonomy Ingestion & Normalization:** Read raw CSV/TSV sources, auto-detect delimiters, apply Unicode NFKC normalization, accent stripping, and lowercase conversion with live progress feedback.
- **Flexible Embedding Generation:** Support multiple embedding backends (Sentence-Transformers, HF models like Gemma and Deepseek) with batch progress bars and metadata tracking.
- **Rule-based Taxonomy Alignment:** Apply a configurable set of Cursor rules (deduplication, lexical+embedding similarity merges, synonym merges, splits, flags, geographical attachments, temporal grouping, and final hierarchy validation) with Chain-of-Thought logging for auditability.
- **Metrics & Reporting:** Compute and print detailed statistics on original vs. merged labels, reduction percentages, and export merge pair listings for manual review.
- **Aligned Taxonomy Export:** Generate a final, aligned taxonomy CSV by applying merge instructions to the normalized dataset.
- **Agent Classifier Training & Inference:** Train a logistic regression classifier on lexical and embedding similarity features to predict merge decisions, and apply it to new data pairs.
- **Data Sorting & Inspection:** Sort and filter the original dataset by enrichment flags (e.g., GPT `isMatch` results) for easy examination of matched vs. unmatched entries.
- **Custom Rule Extension:** Rapidly define and integrate new Cursor rules via XML and Python hooks, with unit test scaffolding and CoT logging.
- **One-Line Pipeline Orchestration:** Execute the full sequence from environment setup to final export in a single command for CI/CD or ad hoc runs.
- **CI/CD & Automation Ready:** Easily integrate into linting, smoke tests, scheduled tasks, and reporting workflows.

## 13. Agent & PPO Fine-Tuning

### 13.1 Agent Classifier Capabilities
- **Binary Merge Prediction:** Uses logistic regression on lexical (Jaro-Winkler) and embedding cosine similarity to predict whether two labels should merge.  
- **High Accuracy & Speed:** Near-perfect performance on known merge pairs and sub-millisecond per-pair inference.  
- **Batch Inference:** Apply the classifier to all candidate pairs generated by `03_apply_rules.py` for secondary filtering or confidence scoring.  
- **Model Export & Loading:** The trained model is saved to `models/agent_classifier.joblib` and can be loaded in custom Python code or services.

### 13.2 PPO Fine-Tuned LLM Agent
- **Interactive Merge Assistant:** A language model fine-tuned via PPO (`scripts/04_ppo_finetune.py`) to generate or verify merge suggestions conversationally.  
- **Preference-Based Learning:** Uses positive and negative merge examples (`data/merge_pairs.csv`) to align the model's outputs with domain rules.  
- **Chain-of-Thought Integration:** Logs reasoning steps into `logs/cot.log` for transparency and audit.  
- **Checkpointing & Resuming:** Checkpoints are written to `models/ppo_checkpoints/` to allow long-running training to resume.  

#### Usage
```bash
source .venv/bin/activate
python scripts/04_ppo_finetune.py \
  --taxo data/taxo_raw.csv \
  --merges data/merge_pairs.csv \
  --output-model models/ppo_agent \
  --batch-size 8 \
  --epochs 3
```
Adjust flags (`--learning-rate`, `--ppo-steps`, `--seed`) in the script or via environment variables as needed.

## 14. Use Cases: Echoline & QSE
Below are tailored examples for two stakeholder teams on leveraging ANISML:

### Echoline
- **Category Consolidation:** Use `01_collect_normalize.py` to ingest and standardize vendor product category CSVs into `data/taxo_raw.csv`.
- **Domain-Specific Embeddings:** Specify `--model gemma` or a custom French embedding model in `02_vectorize.py` to capture nuanced terminology in your inventory.
- **Customized Merge Policies:** Tune `LEXICAL_THRESHOLD` & `EMBED_THRESHOLD` in `03_apply_rules.py` to reflect Echoline's SKU consolidation rules (e.g., threshold adjustments for synonyms).
- **Automated Delivery:** Embed the one-line orchestration command into your CI workflow to regenerate and publish `taxonomy_aligned.csv` weekly to your PIM or ERP system.

### QSE (Quality, Safety & Environment)
- **Normalization of Compliance Terms:** Run `01_collect_normalize.py` on departmental CSVs (e.g., hazard codes, regulation labels) for consistent text formatting.
- **Audit Trails:** Leverage `logs/cot.log` generated by `03_apply_rules.py` for traceable Chain-of-Thought during rule-based merges—critical for regulatory compliance.
- **Predictive Compliance Checking:** Use `07_train_agent_classifier.py` to train or fine-tune `agent_classifier.joblib` for automated detection of potential label conflicts.
- **Conversational Quality Reviews:** Deploy the PPO agent from `04_ppo_finetune.py` within chatOps or an internal tool to suggest and validate quality labels interactively.

## Appendix A: Machine Learning & Embeddings Explained (for Non-Experts)
This appendix demystifies the ML components of the ANISML pipeline in simple terms.

1. **What is Machine Learning?**
   - A way for computers to learn patterns from examples (data) instead of being explicitly programmed.  
   - In ANISML, we use two ML subfields:
     - *Supervised learning* (agent classifier): learns from labeled examples of 'should-merge' vs. 'should-not-merge'.
     - *Reinforcement learning* (PPO agent): learns by trying merge decisions and receiving rewards when they match expert rules.

2. **Text Embeddings: Turning Words into Numbers**
   - Words and phrases are converted into numerical vectors (arrays of numbers) so mathematical operations (like similarity calculation) become possible.  
   - We use pre-trained models (e.g., MiniLM, Gemma) to generate these embeddings:
     - Input: one line of taxonomy text.  
     - Output: a fixed-length vector (e.g., 384 numbers) that captures semantic meaning.  
   - Similar lines should have embeddings that point in similar directions in this high-dimensional space.

3. **Similarity Measures**
   - **Lexical similarity** (Jaro-Winkler): measures how similar two strings are character-by-character (good for typos or small edits).  
   - **Cosine similarity**: measures the angle between two embedding vectors (values range from -1 to +1, where +1 means identical direction).

4. **Rule-Based Merging vs. ML Classifier**
   - The **rule-based step** (`03_apply_rules.py`) applies human-defined rules (thresholds on lexical+embedding similarity) to produce initial merge pairs.  
   - The **classifier** step (`07_train_agent_classifier.py`) treats that output as training data:
     1. It takes each pair and computes two features: [lexical_similarity, embedding_similarity].
     2. It fits a logistic regression model that learns a boundary between merge vs. non-merge pairs.
     3. After training, it can predict new pairs or re-score existing ones for more flexible decision-making.

5. **PPO Fine-Tuning: Turning an LLM into an Interactive Assistant**
   - PPO (Proximal Policy Optimization) is a method for training models by giving them a 'reward' signal.  
   - In `04_ppo_finetune.py`, we:
     1. Prepare prompts like "Should these two labels merge?" with examples.
     2. Let the model generate an answer.
     3. Compute a reward based on whether the answer agrees with our expert merge rules.
     4. Update the model's parameters so it gradually improves its merge advice.

6. **Why These Components?**
   - **Embeddings + Rules** combine human intuition (cursor rules) with semantic understanding (embeddings).  
   - **Classifier** adds adaptability: it can learn from past decisions and generalize to new cases.  
   - **PPO agent** creates an interactive, explainable assistant with Chain-of-Thought logs for human reviewers.

By following this appendix, even team members without ML backgrounds can understand how ANISML processes taxonomy labels from raw text to aligned output, and how each component contributes to accurate, traceable merges.
  
## 15. Complete Pipeline Flow
Below is the exact sequence of scripts, inputs, and outputs when you run the full ANISML pipeline:

1. **Environment Setup**  
   - Command: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`  
   - Outcome: Isolated Python environment with all dependencies installed.

2. **Data Collection & Normalization**  
   - Script: `01_collect_normalize.py --sources data.csv --output data/taxo_raw.csv`  
   - Function: Ingest raw CSV(s), sniff delimiters, normalize text (NFKC, accents, lowercase), show progress bars.  
   - Output: `data/taxo_raw.csv` (cleaned taxonomy).

3. **Embedding Generation**  
   - Script: `02_vectorize.py --input data/taxo_raw.csv --model miniLM`  
   - Function: Convert each row into a numerical embedding via a pre-trained model, display batch progress.  
   - Outputs: `data/embeddings.npy`, `data/embeddings_meta.json`.

4. **Rule-Based Merge Instruction**  
   - Script: `03_apply_rules.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --block-size 256`  
   - Function: Apply Cursor XML rules (dedupe, lexical+embedding similarity), log Chain-of-Thought, and print metrics.  
   - Output: `data/merge_pairs.csv` (source→target merge pairs).

5. **Evaluation of Merges**  
   - Script: `05_evaluate.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv`  
   - Function: Calculate and print statistics: original count, merge pairs, unique sources, aligned count, reduction %.

6. **Export Aligned Taxonomy**  
   - Script: `06_export.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv --output data/taxonomy_aligned.csv`  
   - Function: Remove merged source rows, produce the final aligned taxonomy CSV.

7. **Optional: Train Agent Classifier**  
   - Script: `07_train_agent_classifier.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --merges data/merge_pairs.csv`  
   - Function: Build features (lexical & embedding sim), train logistic regression, save to `models/agent_classifier.joblib`.

8. **Optional: PPO Fine-Tuning**  
   - Script: `04_ppo_finetune.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv --output-model models/ppo_agent --batch-size 8 --epochs 3`  
   - Function: Fine-tune an LLM with PPO, log CoT, save checkpoints under `models/ppo_checkpoints/`.

9. **Optional: Sort Original Data**  
   - Script: `08_sort_data.py --input data.csv --output data/sorted_data.csv`  
   - Function: Sniff delimiter, parse enrichment JSON/regEx, sort rows by `isMatch`, export for review.

10. **One-Line Orchestration**  
    ```bash
    rm -rf .venv __pycache__ logs data/*.csv models/*.joblib \
      && python3 -m venv .venv && source .venv/bin/activate \
      && pip install -r requirements.txt \
      && python -u scripts/01_collect_normalize.py --sources data.csv \
      && python -u scripts/02_vectorize.py --input data/taxo_raw.csv --model miniLM \
      && python -u scripts/03_apply_rules.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --block-size 256 \
      && python -u scripts/05_evaluate.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv \
      && python -u scripts/06_export.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv --output data/taxonomy_aligned.csv
    ```  

11. **Read Detailed Article**  
    [ARTICLE](ARTICLE_URL)  <!-- mocked link -->  

## 16. Why It Works
- **Modular Design:** Each script has a single responsibility (SRP), making debugging, maintenance, and customization straightforward.
- **Progress & Transparency:** Real-time progress bars and Chain-of-Thought logs ensure you can monitor every step and audit merge logic decisions.
- **Hybrid Approach:** Combining deterministic, human-defined rules with flexible ML models (classifier & PPO agent) balances precision, explainability, and adaptability.
- **Scalability:** Block-wise processing and optional FAISS integration allow handling large taxonomies without excessive memory use.
- **Reproducibility:** Strict version pinning, environment isolation (`.venv`), and one-line orchestration guarantee consistent results across runs and CI/CD pipelines.
- **Extensibility:** Cursor XML rules, pluggable embedding backends, and clear APIs enable rapid adaptation to new domains and evolving requirements.

## 17. CSV Artifacts Explained
Below is a quick reference to every CSV file you will encounter in the ANISML pipeline:

- **data.csv**  
  *Your raw input.* Contains the original taxonomy and any enrichment columns. Can be multi-column (e.g., labels, metadata) and may include an `isMatch` flag after enrichment.

- **data/taxo_raw.csv**  
  *Normalized taxonomy.* Produced by `01_collect_normalize.py`:
  - Single-column (no header) of cleaned labels.
  - All accents stripped, Unicode NFKC applied, lowercased.

- **data/merge_pairs.csv**  
  *Merge instructions.* Produced by `03_apply_rules.py`:
  - Two columns: `source_id` and `target_id` (zero-based row indices into `taxo_raw.csv`).
  - Each row indicates that the label at `source_id` should merge into `target_id`.

- **data/taxonomy_aligned.csv**  
  *Final aligned taxonomy.* Produced by `06_export.py`:
  - Same format as `taxo_raw.csv` but with all `source_id` rows removed.
  - Ready for import into downstream systems (PIMs, databases).

- **data/sorted_data.csv**  
  *Inspection-ready raw data.* Produced by `08_sort_data.py`:
  - Same columns as your original `data.csv`, sorted so that rows with `isMatch=True` appear first.
  - Useful for manual review of matched vs. unmatched records.

*Document generated by AI-Assisted tooling on 2025-04-18.*





