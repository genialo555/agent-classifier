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
│   ├── 08_sort_data.py
│   ├── 09_export_rules.py
│   ├── 10_group_compartments.py
│   ├── 11_topic_grouping.py
│   ├── 12_classify_group.py
│   ├── 13_classify_all_groups.py
│   ├── 14_export_to_excel.py
│   ├── 15_assign_records.py
│   ├── 16_create_thematic_excel.py
│   ├── 17_customize_thematic_excel.py
│   ├── 18_visualize_themes.py
│   ├── 19_group_by_veille_type.py
│   ├── 20_visualize_veille_types.py
│   └── 21_classify_new_article.py
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

### 4.9 09_export_rules.py

- **Purpose:** Export Cursor rules from XML to a CSV summary.
- **Features:** XML parsing, rule summary with priorities and descriptions.
- **Usage:**
  ```bash
  python scripts/09_export_rules.py --rules .cursor/rules/sprint1.xml --output data/rules_summary.csv
  ```

### 4.10 10_group_compartments.py

- **Purpose:** Group taxonomy entries into compartments based on merge pairs.
- **Features:** Builds and exports groupings based on target canonical labels.
- **Usage:**
  ```bash
  python scripts/10_group_compartments.py --taxo data/taxo_raw.csv --merges data/merge_pairs.csv
  ```

### 4.11 11_topic_grouping.py

- **Purpose:** Group taxonomy labels by semantic similarity to a seed topic.
- **Features:** Finds and exports all labels related to a specific seed subject.
- **Usage:**
  ```bash
  python scripts/11_topic_grouping.py --subject "santé sécurité au travail" --threshold 0.80
  ```

### 4.12 12_classify_group.py

- **Purpose:** Use the trained agent classifier to group taxonomy labels around a seed.
- **Features:** Combines lexical and embedding similarity with ML classification.
- **Usage:**
  ```bash
  python scripts/12_classify_group.py --subject "santé sécurité travail" --prob-threshold 0.7
  ```

### 4.13 13_classify_all_groups.py

- **Purpose:** Process all canonical seeds to create comprehensive classification groups.
- **Features:** Batch classification of entries against all seed subjects.
- **Usage:**
  ```bash
  python scripts/13_classify_all_groups.py --model models/agent_classifier.joblib
  ```

### 4.14 14_export_to_excel.py

- **Purpose:** Combine compartments and classification groups into an Excel workbook.
- **Features:** Multi-sheet Excel export with summary statistics.
- **Usage:**
  ```bash
  python scripts/14_export_to_excel.py --compartments data/compartments.csv --classifier_groups data/classifier_all_groups.csv
  ```

### 4.15 15_assign_records.py

- **Purpose:** Assign each record to taxonomy labels using trained classifier.
- **Features:** Predicts best matches for records based on title similarity.
- **Usage:**
  ```bash
  python scripts/15_assign_records.py --records data/sorted_data.csv --title-col 1
  ```

### 4.16 16_create_thematic_excel.py

- **Purpose:** Automatically identify major themes in taxonomy and generate Excel with one sheet per theme.
- **Features:** 
  - Uses K-means clustering on embeddings to detect thematic groups
  - Automatically names themes based on frequent words in each cluster
  - Creates one Excel sheet per theme with summary statistics
- **Usage:**
  ```bash
  python scripts/16_create_thematic_excel.py --taxo data/taxo_raw.csv --emb data/embeddings.npy --n-themes 15
  ```

### 4.17 17_customize_thematic_excel.py

- **Purpose:** Customize the automatically generated thematic Excel with better names and manual reassignments.
- **Features:**
  - Allows renaming themes via `theme_mapping.csv` 
  - Supports manual reassignments of entries between themes via `reassignments.csv`
  - Generates templates for customization on first run
- **Usage:**
  ```bash
  python scripts/17_customize_thematic_excel.py --input data/thematic_taxonomy.xlsx --output data/thematic_taxonomy_custom.xlsx
  ```

### 4.18 18_visualize_themes.py

- **Purpose:** Generate visualizations of taxonomy distribution across themes.
- **Features:**
  - Creates bar charts showing entry counts per theme
  - Generates pie charts for proportion visualization
  - Provides metrics on theme distribution and balance
- **Usage:**
  ```bash
  python scripts/18_visualize_themes.py --input data/thematic_taxonomy_custom.xlsx --top 15
  ```

### 4.19 19_group_by_veille_type.py

- **Purpose:** Group entries by their type de veille and create Excel workbook with one sheet per type.
- **Features:**
  - Analyzes original data.csv to extract "type de veille" classifications
  - Automatically detects the typedeveille column with normalization
  - Creates one Excel sheet per veille type with all corresponding entries
- **Usage:**
  ```bash
  python scripts/19_group_by_veille_type.py --input data.csv --output data/veille_types.xlsx
  ```

### 4.20 20_visualize_veille_types.py

- **Purpose:** Generate visualizations of taxonomy distribution across veille types.
- **Features:**
  - Creates horizontal bar charts for better readability of long type names
  - Generates treemap visualizations (requires `squarify` package)
  - Displays count and percentage metrics for each type
- **Usage:**
  ```bash
  python scripts/20_visualize_veille_types.py --input data/veille_types.xlsx --top 20
  ```

### 4.21 21_classify_new_article.py

- **Purpose:** Classify a new article into the most appropriate theme and type de veille.
- **Features:**
  - Extracts keywords and key phrases from article text
  - Creates a composite embedding that captures the article's essence
  - Compares with existing taxonomy themes and veille types 
  - Identifies best matches with confidence scores
  - Generates visualizations of classification results
- **Usage:**
  ```bash
  # Classify from a file
  python scripts/21_classify_new_article.py --input path/to/article.txt
  
  # Or directly from text
  python scripts/21_classify_new_article.py --text "Text of the article to classify"
  ```

## 5. Logging & Outputs

- **Chain-of-Thought:** `logs/cot.log` records each rule's firing with details.
- **Embeddings metadata:** `data/embeddings_meta.json` contains model name, dimension, filter.
- **Merge pairs:** `data/merge_pairs.csv` for downstream eval and export.
- **Aligned taxonomy:** `data/taxonomy_aligned.csv` final product.
- **Sorted raw data:** `data/sorted_data.csv` for manual inspection.
- **Classifier model:** `models/agent_classifier.joblib` ready for inference.
- **Thematic Excel:** `data/thematic_taxonomy.xlsx` with automatically detected themes.
- **Customized themes:** `data/thematic_taxonomy_custom.xlsx` with renamed themes and reassignments.
- **Veille types:** `data/veille_types.xlsx` with entries organized by type de veille.
- **Visualizations:** Various PNG files for distribution analysis.
- **Article classifications:** `data/article_classification.xlsx` contains analysis of classified articles.

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
- **Thematic Organization:** Automatically cluster and organize taxonomy entries into thematic groups using unsupervised learning.
- **Type de Veille Classification:** Group entries by their declared type de veille categories for domain-specific analysis.
- **Interactive Customization:** Provide tools for users to rename themes and reassign entries between categories.
- **Visual Analytics:** Generate informative visualizations of taxonomy distribution across themes and types.
- **Article Classification:** Automatically analyze and classify new articles by theme and type de veille based on their content.

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

- **data/theme_mapping.csv**  
  *Theme renaming configuration.* Created by `17_customize_thematic_excel.py`:
  - Maps from auto-generated theme names to user-friendly display names.
  - Editable template for customizing theme organization.

- **data/reassignments.csv**  
  *Manual entry reassignments.* Created by `17_customize_thematic_excel.py`:
  - Specifies entries to move from one theme to another.
  - Format: label_id, label, old_theme, new_theme.

## 18. Excel Artifacts Explained
The pipeline generates several Excel workbooks for different organizational purposes:

- **data/taxonomy_groups.xlsx**  
  *Basic taxonomy grouping.* Produced by `14_export_to_excel.py`:
  - Contains compartments from merge operations
  - Groups based on classifier results

- **data/record_classification.xlsx**  
  *Record assignments.* Produced by `15_assign_records.py`:
  - Links each record to taxonomy classifications
  - Includes confidence scores from the classifier

- **data/thematic_taxonomy.xlsx**  
  *Automatic thematic organization.* Produced by `16_create_thematic_excel.py`:
  - Creates data-driven themes using K-means clustering
  - One sheet per detected theme with all related entries
  - Summary sheet with theme statistics and examples

- **data/thematic_taxonomy_custom.xlsx**  
  *Customized thematic organization.* Produced by `17_customize_thematic_excel.py`:
  - Incorporates user-defined theme names and reassignments
  - Same structure as thematic_taxonomy.xlsx but with user improvements

- **data/veille_types.xlsx**  
  *Type de veille organization.* Produced by `19_group_by_veille_type.py`:
  - Organizes entries by their declared type de veille
  - One sheet per veille type with all related entries
  - Summary sheet with distribution statistics

## 19. Visualization Outputs
The pipeline generates several visualization files:

- **data/theme_distribution.png**  
  *Theme distribution chart.* Produced by `18_visualize_themes.py`:
  - Bar chart showing counts of entries per theme
  - Sorted by size for easy analysis

- **data/theme_distribution_pie.png**  
  *Theme proportion visualization.* Produced by `18_visualize_themes.py`:
  - Pie chart showing relative sizes of themes

- **data/veille_types_distribution.png**  
  *Veille type distribution.* Produced by `20_visualize_veille_types.py`:
  - Horizontal bar chart optimized for long type names
  - Includes both counts and percentages

- **data/veille_types_distribution_treemap.png**  
  *Veille type treemap.* Produced by `20_visualize_veille_types.py`:
  - Hierarchical visualization of type distribution
  - Area proportional to entry counts

- **data/article_classification.png**  
  *Article classification results.* Produced by `21_classify_new_article.py`:
  - Bar charts showing best matching themes and types
  - Top 5 results with confidence scores

## 20. Article Classification Process

The ANISML pipeline includes a sophisticated article classification system that can analyze and categorize new content automatically:

### 20.1 Approach & Algorithm

1. **Text Analysis**
   - Extracts relevant keywords using frequency analysis and stopword filtering
   - Identifies key phrases that best represent the article's focus
   - Creates a composite embedding that balances full text, keywords, and key phrases

2. **Similarity Computation**
   - Compares the article embedding against all thematic categories
   - Measures similarity with all veille types
   - Calculates weighted scores based on multiple similarity metrics

3. **Confidence Scoring**
   - Applies weighted formulas to determine the most likely classifications
   - Prioritizes content similarity over metadata matching
   - Provides confidence scores for all potential matches

### 20.2 Practical Usage

Researchers and analysts can use the article classification capability for:

- **Content Organization:** Automatically sort incoming articles into thematic folders
- **Trend Monitoring:** Track which themes and types receive the most new content over time
- **Recommendation Systems:** Suggest related content based on theme similarities
- **Automatic Tagging:** Apply taxonomy labels to articles for improved searchability

### 20.3 Performance

The classifier shows particularly good results when:
- Articles have clear thematic focuses (rather than mixing many topics)
- Content has sufficient length (500+ characters)
- The taxonomy is comprehensive and properly organized

### 20.4 Output Analysis

The classification results are provided in multiple formats:
- Terminal output with best matches and confidence scores
- Excel workbook with detailed analysis
- Visualization showing top matches with relative confidence levels

This allows both quick assessment of classification accuracy and detailed investigation of classifier reasoning when needed.







