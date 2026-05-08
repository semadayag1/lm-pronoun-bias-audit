# When Pronouns Change the Comment

Replication materials for the preprint:

**When Pronouns Change the Comment: Auditing Gendered Patterns in LLM Writing Feedback**

This repository contains the dataset, code, paper source, and supporting reports for a controlled audit study of whether three commercial large language models produce different writing feedback when the only varied input is a student's stated pronouns.

The study collected 1,500 model responses across a 2 x 3 x 5 design: two pronoun conditions, three models, five student-authored stimulus essays, and 50 repeated generations per cell. The essays were written for this project as controlled elicitation materials. After Benjamini-Hochberg FDR correction, eight statistically significant differences survived; seven of eight remained significant in mixed-effects models with essay as a random intercept.

The paper is available at `paper/paper.pdf`. The LaTeX source and bibliography are in the same folder. The replication package is archived on Zenodo at https://doi.org/10.5281/zenodo.20081530.

## Repository Structure

```text
llm-pronoun-bias-audit/
|-- README.md
|-- CITATION.cff
|-- DATA_USE.md
|-- LICENSE
|-- LICENSE-CODE
|-- PREPRINT_METADATA.md
|-- paper/
|   |-- paper.pdf
|   |-- paper.tex
|   `-- references.bib
|-- code/
|   |-- requirements.txt
|   |-- llm_bias_experiment.py
|   |-- analyze_bias.py
|   |-- deep_analysis.py
|   |-- fdr_correction.py
|   `-- robustness_checks.py
|-- data/
|   |-- bias_experiment.sqlite
|   `-- bias_analysis_features.csv
|-- essays/
|   |-- essay_1_outrage.txt
|   |-- essay_2_pressure.txt
|   |-- essay_3_sat.txt
|   |-- essay_4_algorithm.txt
|   `-- essay_5_vulnerable.txt
`-- reports/
    |-- bias_analysis_report.txt
    |-- ngram_bias_report.txt
    `-- Outlier_Dossier.md
```

## Reproducibility

### Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
python -m nltk.downloader vader_lexicon punkt
```

### Re-run the Analysis

The released dataset of 1,500 model responses is included in `data/bias_experiment.sqlite`. To reproduce the statistical analyses without re-running the LLM API calls:

```bash
cd code
python analyze_bias.py
python deep_analysis.py
python fdr_correction.py
python robustness_checks.py
```

The scripts read from `../data/bias_experiment.sqlite` and `../data/bias_analysis_features.csv`, then write reports to stdout and to `../reports/`.

### Re-run the Experiment

Re-collecting model responses requires an OpenRouter account and will incur API costs.

1. Create `code/.env` from `code/.env.example`.
2. Add an OpenRouter API key.
3. Run:

```bash
cd code
python llm_bias_experiment.py
```

The runner requests responses from OpenAI GPT-5.2, Anthropic Claude Sonnet 4.5, and Google Gemini 3 Pro Preview through the OpenRouter API. The script caps concurrent requests at 50 and uses exponential backoff for rate limits and server errors.

Model behavior and provider routing can change over time. A later re-run may not reproduce the exact responses reported in the preprint.

## Data Schema

`data/bias_experiment.sqlite` contains one table, `results`.

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Auto-incremented primary key. |
| `timestamp` | DATETIME | Timestamp inserted by SQLite when the row was written. |
| `model_name` | TEXT | Requested model identifier. |
| `essay_id` | TEXT | Stimulus essay identifier, such as `essay_1_outrage`. |
| `condition` | TEXT | Pronoun condition, either `He/Him` or `She/Her`. |
| `feedback_text` | TEXT | Full LLM-generated writing feedback. |

`data/bias_analysis_features.csv` contains one row per response and adds derived text features used in the analysis:

```text
id, timestamp, model_name, essay_id, condition, feedback_text,
word_count, char_count, ttr, sentiment_compound, sentiment_pos,
sentiment_neg, warmth_count, competence_count, hedging_count,
directive_count, warmth_per100, competence_per100, hedging_per100,
directive_per100
```

The SQLite table does not include an explicit replication index. Repeated generations are represented by repeated rows for the same `(model_name, essay_id, condition)` cell.

## Ethics and Reuse

The stimulus essays were written by first-year students for this project and were released with the writers' verbal permission. Student names, course identifiers, institutional identifiers, and other direct identifiers are not linked to any released artifact.

Even though the essays were project-specific and de-identified, they are still student-authored writing. Treat the essays and model responses as sensitive educational data. See `DATA_USE.md` for reuse limits.

## License

Paper, data, essays, and reports are released under the Creative Commons Attribution-NonCommercial 4.0 International License. Code is released under the MIT License. See `LICENSE` and `LICENSE-CODE`.

## Citation

Please cite the preprint if you use these materials. A citation file is included in `CITATION.cff`.

## Contact

Inquiries about the dataset, pipeline, or methodology may be directed to Sophia Madayag at the email address listed in the paper.
