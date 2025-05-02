# Taxonomy Alignment & Pre-processing – Sprint 1

> **Status:** 🚧 _in active development_ (see roadmap below)

## 1. Project Overview
This repository hosts the code and configuration required to **merge, clean and align multiple hierarchical taxonomies** (~4 k nodes each, ≤ 6 levels deep) prior to fine-tuning Gemma-3 .1B-it / DeepSeek-R1 3B with PPO.  
The workflow is orchestrated through **Cursor rules** (see `.cursor/rules/sprint1.xml`) and a series of Python scripts in `scripts/`.

Key objectives of **Sprint 1**:
1. Collect & normalise raw taxonomy files (Unicode NFKC, accents stripped, lower-case).
2. Vectorise every node with an encoder-only LLM.
3. Apply deterministic merge/split rules (lexical, embedding, regex).
4. Prepare a dataset of merge vs. split preferences for PPO fine-tuning.
5. Produce an aligned master taxonomy (`taxonomy_aligned.csv`) with ≥ 65 % correct merges.

## 2. Repository Structure
```text
.
├── data/                       # Raw & intermediate taxonomy CSVs
│   ├── taxo_a.csv
│   └── taxo_b.csv
├── scripts/                    # Processing & evaluation pipeline
│   ├── 01_collect_normalize.py
│   ├── 02_vectorize.py
│   ├── 03_apply_rules.py
│   ├── 04_ppo_finetune.py
│   ├── 05_evaluate.py
│   └── 06_export.py
├── .cursor/
│   └── rules/
│       └── sprint1.xml         # Cursor rules executed in this sprint
├── requirements.txt            # Strictly pinned Python dependencies
└── README.md                   # You are here
```

## 3. Quick Start
```bash
# 0. Clone & enter the repo
$ git clone <repo-url> && cd <repo-dir>

# 1. Install dependencies (Python ≥ 3.10, CUDA 11+ recommended)
$ python -m pip install -r requirements.txt

# 2. Run the pipeline end-to-end on sample data
$ python scripts/01_collect_normalize.py --sources data/*.csv
$ python scripts/02_vectorize.py --model gemma
$ python scripts/03_apply_rules.py
$ python scripts/05_evaluate.py
$ python scripts/06_export.py
```

> **Tip :** Each script supports a `--dry-run` flag so CI can run fast on PRs.

## 4. Roadmap
| Sprint | Milestone | Target date | Status |
| ------ | --------- | ----------- | ------ |
| **1**  | Repo skeleton + CI green | 2025-04-19 | ✅ Done |
|        | Collect & normalise scripts | 2025-04-22 | ⏳ In progress |
|        | Vectorisation (embeddings) | 2025-04-24 | ⏳ |
|        | Rule engine (exact dup + JW+cosine) | 2025-04-26 | ⏳ |
|        | Evaluation & export | 2025-04-28 | ⏳ |
|        | Monthly GitHub Actions workflow | 2025-04-29 | ⏳ |
| **2**  | Multilingual model support (LaBSE/XLM-R) | 2025-05 | 🔜 |
|        | Incremental taxonomy diffs & changelogs | 2025-05 | 🔜 |
| **3**  | Active learning loop with human-in-the-loop UI | 2025-06 | 🔜 |

Legend: ✅ done  ⏳ ongoing  🔜 planned

## 5. Installation Details
The project uses **strict version pinning** for full reproducibility.  
See [`requirements.txt`](requirements.txt) for exact versions.

```text
transformers>=4.51,<5
sentence-transformers>=2.7,<4
accelerate>=0.29,<0.30
trl>=0.8,<0.9
scikit-learn>=1.4,<1.7
pandas>=2.2,<2.3
text-unidecode==1.3
jaro-winkler>=2,<3
```

## 6. Contributing
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification and enforce Ruff/Black in CI.  
Please open an issue to discuss significant changes before submitting a PR.

## 7. License
Specify license here (e.g., MIT).  
© 2025 Your Organisation.

---
*This README follows best-practice guidance from [GitHub Docs – About READMEs](https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes).* 