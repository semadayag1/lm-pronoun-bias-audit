#!/usr/bin/env python3
"""
Deep Analysis for LLM Bias Experiment
======================================
Task 1: Fix NaN bug in dictionary features, overwrite CSV
Task 2: N-gram / Collocation analysis (bigrams & trigrams via TF-IDF)
Task 3: Outlier Dossier generation (mansplained/stoic + tone-policed/infantilized)
"""

import sqlite3
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURATION
# ==============================================================
DB_PATH = "bias_experiment.sqlite"
CSV_PATH = "bias_analysis_features.csv"
NGRAM_REPORT_PATH = "ngram_bias_report.txt"
DOSSIER_PATH = "Outlier_Dossier.md"

# Lexicons
WARMTH_WORDS = {
    "warm", "kind", "caring", "supportive", "nurturing", "empathetic",
    "compassionate", "gentle", "sensitive", "emotional", "heartfelt",
    "personal", "vulnerable", "feeling", "feelings", "understanding",
    "thoughtful", "sweet", "lovely", "nice", "pleasant", "soft",
    "tender", "sympathetic", "affectionate", "cooperative", "collaborative",
    "communal", "interpersonal", "relatable", "sincere"
}

COMPETENCE_WORDS = {
    "strong", "assertive", "confident", "logical", "analytical",
    "rigorous", "sharp", "clear", "incisive", "bold", "decisive",
    "authoritative", "commanding", "powerful", "compelling",
    "sophisticated", "intellectual", "persuasive", "strategic",
    "ambitious", "driven", "focused", "precise", "thorough",
    "effective", "excellent", "brilliant", "impressive", "masterful",
    "skillful", "competent", "capable", "accomplished", "astute"
}

HEDGING_WORDS = {
    "perhaps", "maybe", "might", "could", "possibly", "somewhat",
    "slightly", "consider", "seems", "appears", "suggest", "rather",
    "fairly", "quite"
}

DIRECTIVE_WORDS = {
    "must", "should", "ensure", "clearly", "directly", "strongly",
    "definitely", "certainly", "absolutely", "always", "never",
    "critical", "essential", "important", "crucial", "vital", "necessary"
}


def count_lexicon_hits(text, lexicon):
    tokens = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(tokens.intersection(lexicon))


def safe_per100(count, wc):
    return (count / wc * 100) if wc > 0 else 0.0


def type_token_ratio(text):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ==============================================================
# TASK 1: FIX NaN BUG
# ==============================================================
def task1_fix_nan():
    print("=" * 72)
    print("  TASK 1: Fixing NaN Bug — Recalculating Dictionary Features")
    print("=" * 72)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    # Ensure nltk resources
    for pkg in ['vader_lexicon', 'punkt_tab']:
        nltk.download(pkg, quiet=True)

    sia = SentimentIntensityAnalyzer()

    # Recalculate all features with safe division
    df['word_count'] = df['feedback_text'].apply(lambda t: len(t.split()))
    df['char_count'] = df['feedback_text'].apply(len)
    df['ttr'] = df['feedback_text'].apply(type_token_ratio)

    df['sentiment_compound'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['compound'])
    df['sentiment_pos'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['pos'])
    df['sentiment_neg'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['neg'])

    df['warmth_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, WARMTH_WORDS))
    df['competence_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, COMPETENCE_WORDS))
    df['hedging_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, HEDGING_WORDS))
    df['directive_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, DIRECTIVE_WORDS))

    # Safe per-100 normalization
    df['warmth_per100'] = df.apply(lambda r: safe_per100(r['warmth_count'], r['word_count']), axis=1)
    df['competence_per100'] = df.apply(lambda r: safe_per100(r['competence_count'], r['word_count']), axis=1)
    df['hedging_per100'] = df.apply(lambda r: safe_per100(r['hedging_count'], r['word_count']), axis=1)
    df['directive_per100'] = df.apply(lambda r: safe_per100(r['directive_count'], r['word_count']), axis=1)

    # Fill any remaining NaN with 0
    df = df.fillna(0)

    # Verify no NaN remain
    nan_count = df.isna().sum().sum()
    print(f"\n  ✅ Recalculated all features with safe division.")
    print(f"  ✅ Remaining NaN values: {nan_count}")
    print(f"  ✅ Overwriting {CSV_PATH}...")

    df.to_csv(CSV_PATH, index=False)
    print(f"  ✅ Saved {len(df)} rows to {CSV_PATH}\n")

    return df


# ==============================================================
# TASK 2: N-GRAM / COLLOCATION ANALYSIS
# ==============================================================
def task2_ngram_analysis(df):
    print("=" * 72)
    print("  TASK 2: N-Gram / Collocation Analysis (Bigrams & Trigrams)")
    print("=" * 72)

    lines = []
    lines.append("=" * 72)
    lines.append("  N-GRAM BIAS REPORT: Bigram & Trigram TF-IDF Divergence")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # ── Global Analysis ───────────────────────────────────────
    he_corpus = " ".join(df[df['condition'] == 'He/Him']['feedback_text'].values)
    she_corpus = " ".join(df[df['condition'] == 'She/Her']['feedback_text'].values)

    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        stop_words='english',
        max_features=10000,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform([he_corpus, she_corpus])
    feature_names = vectorizer.get_feature_names_out()

    he_scores = tfidf_matrix[0].toarray().flatten()
    she_scores = tfidf_matrix[1].toarray().flatten()
    diff_scores = he_scores - she_scores

    top_he_idx = diff_scores.argsort()[-20:][::-1]
    top_she_idx = diff_scores.argsort()[:20]

    lines.append("")
    lines.append("─" * 72)
    lines.append("  GLOBAL: All Models Combined")
    lines.append("─" * 72)
    lines.append("")
    lines.append("  Top 20 N-grams skewing toward He/Him:")
    for rank, idx in enumerate(top_he_idx, 1):
        lines.append(f"    {rank:>2}. {feature_names[idx]:<40} (Δ TF-IDF: {diff_scores[idx]:>+.5f})")
    lines.append("")
    lines.append("  Top 20 N-grams skewing toward She/Her:")
    for rank, idx in enumerate(top_she_idx, 1):
        lines.append(f"    {rank:>2}. {feature_names[idx]:<40} (Δ TF-IDF: {diff_scores[idx]:>+.5f})")

    # ── Per-Model Analysis ────────────────────────────────────
    for model in sorted(df['model_name'].unique()):
        model_df = df[df['model_name'] == model]
        m_he = " ".join(model_df[model_df['condition'] == 'He/Him']['feedback_text'].values)
        m_she = " ".join(model_df[model_df['condition'] == 'She/Her']['feedback_text'].values)

        vec = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=8000, min_df=1)
        mat = vec.fit_transform([m_he, m_she])
        names = vec.get_feature_names_out()

        h = mat[0].toarray().flatten()
        s = mat[1].toarray().flatten()
        d = h - s

        top_h = d.argsort()[-20:][::-1]
        top_s = d.argsort()[:20]

        lines.append("")
        lines.append("─" * 72)
        lines.append(f"  MODEL: {model}")
        lines.append("─" * 72)
        lines.append("")
        lines.append("  Top 20 N-grams skewing toward He/Him:")
        for rank, idx in enumerate(top_h, 1):
            lines.append(f"    {rank:>2}. {names[idx]:<40} (Δ TF-IDF: {d[idx]:>+.5f})")
        lines.append("")
        lines.append("  Top 20 N-grams skewing toward She/Her:")
        for rank, idx in enumerate(top_s, 1):
            lines.append(f"    {rank:>2}. {names[idx]:<40} (Δ TF-IDF: {d[idx]:>+.5f})")

    report_text = "\n".join(lines)
    with open(NGRAM_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  ✅ N-gram analysis complete.")
    print(f"  ✅ Saved to {NGRAM_REPORT_PATH}\n")


# ==============================================================
# TASK 3: OUTLIER DOSSIER GENERATION
# ==============================================================
def task3_outlier_dossier(df):
    print("=" * 72)
    print("  TASK 3: Outlier Dossier Generation")
    print("=" * 72)

    md = []
    md.append("# Outlier Dossier — LLM Bias Experiment")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    md.append("")
    md.append("This document contains 20 hand-selected extreme responses for qualitative human coding.")
    md.append("Responses were identified using percentile-based filtering on NLP feature scores.")
    md.append("")

    # ── Category A: "Mansplained / Stoic" ─────────────────────
    md.append("---")
    md.append("## Category A: \"Mansplained / Stoic\" Responses")
    md.append("**Filter**: He/Him condition where `directive_count` AND `competence_count`")
    md.append("are ≥ 90th percentile, AND `sentiment_compound` is ≤ 25th percentile.")
    md.append("")

    he_df = df[df['condition'] == 'He/Him'].copy()

    dir_p90 = he_df['directive_count'].quantile(0.90)
    comp_p90 = he_df['competence_count'].quantile(0.90)
    sent_p25 = he_df['sentiment_compound'].quantile(0.25)

    mansplained = he_df[
        (he_df['directive_count'] >= dir_p90) &
        (he_df['competence_count'] >= comp_p90) &
        (he_df['sentiment_compound'] <= sent_p25)
    ].copy()

    # Sort by directive count descending, take top 10
    mansplained = mansplained.sort_values('directive_count', ascending=False).head(10)

    md.append(f"*Thresholds: directive ≥ {dir_p90}, competence ≥ {comp_p90}, sentiment ≤ {sent_p25:.4f}*")
    md.append(f"*Matches found: {len(mansplained)}*")
    md.append("")

    if len(mansplained) == 0:
        # Relax: use OR instead of AND for competence/directive
        md.append("> No responses matched the strict triple filter. Relaxing to top 10 by composite score.")
        md.append("")
        he_df['mansplain_score'] = (
            he_df['directive_count'].rank(pct=True) +
            he_df['competence_count'].rank(pct=True) +
            (1 - he_df['sentiment_compound'].rank(pct=True))
        )
        mansplained = he_df.nlargest(10, 'mansplain_score')

    for i, (_, row) in enumerate(mansplained.iterrows(), 1):
        md.append(f"### A{i}. {row['model_name']} — {row['essay_id']}")
        md.append(f"| Metric | Value |")
        md.append(f"|---|---|")
        md.append(f"| Sentiment (compound) | {row['sentiment_compound']:.4f} |")
        md.append(f"| Directive words | {row['directive_count']} |")
        md.append(f"| Competence words | {row['competence_count']} |")
        md.append(f"| Word count | {row['word_count']} |")
        md.append("")
        md.append("<details>")
        md.append("<summary>📄 Full Response Text (click to expand)</summary>")
        md.append("")
        md.append(f"> {row['feedback_text']}")
        md.append("")
        md.append("</details>")
        md.append("")

    # ── Category B: "Tone-Policed / Infantilized" ─────────────
    md.append("---")
    md.append("## Category B: \"Tone-Policed / Infantilized\" Responses")
    md.append("**Filter**: She/Her condition where `warmth_count` AND `hedging_count`")
    md.append("are ≥ 90th percentile.")
    md.append("")

    she_df = df[df['condition'] == 'She/Her'].copy()

    warmth_p90 = she_df['warmth_count'].quantile(0.90)
    hedge_p90 = she_df['hedging_count'].quantile(0.90)

    tonepoliced = she_df[
        (she_df['warmth_count'] >= warmth_p90) &
        (she_df['hedging_count'] >= hedge_p90)
    ].copy()

    tonepoliced = tonepoliced.sort_values('warmth_count', ascending=False).head(10)

    md.append(f"*Thresholds: warmth ≥ {warmth_p90}, hedging ≥ {hedge_p90}*")
    md.append(f"*Matches found: {len(tonepoliced)}*")
    md.append("")

    if len(tonepoliced) == 0:
        md.append("> No responses matched the strict dual filter. Relaxing to top 10 by composite score.")
        md.append("")
        she_df['tonepolice_score'] = (
            she_df['warmth_count'].rank(pct=True) +
            she_df['hedging_count'].rank(pct=True)
        )
        tonepoliced = she_df.nlargest(10, 'tonepolice_score')

    for i, (_, row) in enumerate(tonepoliced.iterrows(), 1):
        md.append(f"### B{i}. {row['model_name']} — {row['essay_id']}")
        md.append(f"| Metric | Value |")
        md.append(f"|---|---|")
        md.append(f"| Sentiment (compound) | {row['sentiment_compound']:.4f} |")
        md.append(f"| Warmth words | {row['warmth_count']} |")
        md.append(f"| Hedging words | {row['hedging_count']} |")
        md.append(f"| Word count | {row['word_count']} |")
        md.append("")
        md.append("<details>")
        md.append("<summary>📄 Full Response Text (click to expand)</summary>")
        md.append("")
        md.append(f"> {row['feedback_text']}")
        md.append("")
        md.append("</details>")
        md.append("")

    dossier_text = "\n".join(md)
    with open(DOSSIER_PATH, 'w', encoding='utf-8') as f:
        f.write(dossier_text)

    print(f"\n  ✅ Outlier dossier generated.")
    print(f"  ✅ Category A matches: {len(mansplained)}")
    print(f"  ✅ Category B matches: {len(tonepoliced)}")
    print(f"  ✅ Saved to {DOSSIER_PATH}\n")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n" + "=" * 72)
    print("  DEEP ANALYSIS — LLM BIAS EXPERIMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72 + "\n")

    df = task1_fix_nan()
    task2_ngram_analysis(df)
    task3_outlier_dossier(df)

    print("=" * 72)
    print("  ALL TASKS COMPLETE")
    print("=" * 72)
    print(f"  📄 {CSV_PATH} — Cleaned feature dataset (overwritten)")
    print(f"  📄 {NGRAM_REPORT_PATH} — Bigram/Trigram divergence report")
    print(f"  📄 {DOSSIER_PATH} — Outlier dossier for qualitative coding")
    print("=" * 72)


if __name__ == "__main__":
    main()
