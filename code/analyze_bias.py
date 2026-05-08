#!/usr/bin/env python3
"""
Statistical NLP Analysis for LLM Bias Experiment
=================================================
Analyzes 1,500 LLM-generated feedback responses across:
  - 3 models (GPT-5.2, Claude Sonnet 4.5, Gemini 3 Pro Preview)
  - 5 essays
  - 2 conditions (He/Him vs She/Her)
  - 50 iterations each

Outputs a comprehensive report with:
  1. Response length analysis
  2. Sentiment analysis (VADER compound scores)
  3. Lexical diversity (Type-Token Ratio)
  4. Gendered language framing (warmth vs. competence vocabulary)
  5. Top divergent vocabulary between conditions (TF-IDF)
  6. Statistical significance testing (Mann-Whitney U, Cohen's d)
"""

import sqlite3
import os
import re
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURATION
# ==============================================================
DB_PATH = "bias_experiment.sqlite"
REPORT_PATH = "bias_analysis_report.txt"

# Gendered language lexicons (research-backed warmth vs competence framing)
# Sources: Gaucher et al. (2011), Pietraszkiewicz et al. (2019)
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
    "slightly", "a bit", "a little", "consider", "tend to",
    "seems", "appears", "suggest", "rather", "fairly", "quite"
}

DIRECTIVE_WORDS = {
    "must", "need to", "should", "have to", "require", "ensure",
    "make sure", "clearly", "directly", "strongly", "definitely",
    "certainly", "absolutely", "always", "never", "critical",
    "essential", "important", "crucial", "vital", "necessary"
}

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def count_lexicon_hits(text, lexicon):
    """Count how many words from a lexicon appear in text."""
    tokens = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(tokens.intersection(lexicon))


def type_token_ratio(text):
    """Calculate Type-Token Ratio (lexical diversity)."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def word_count(text):
    """Simple whitespace word count."""
    return len(text.split())


def significance_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def run_comparison(he_values, she_values, metric_name):
    """Run Mann-Whitney U test and Cohen's d, return formatted result."""
    he_mean = np.mean(he_values)
    she_mean = np.mean(she_values)
    he_std = np.std(he_values, ddof=1)
    she_std = np.std(she_values, ddof=1)

    u_stat, p_value = stats.mannwhitneyu(he_values, she_values, alternative='two-sided')
    d = cohens_d(he_values, she_values)
    
    return {
        "metric": metric_name,
        "he_mean": he_mean,
        "he_std": he_std,
        "she_mean": she_mean,
        "she_std": she_std,
        "diff": he_mean - she_mean,
        "diff_pct": ((he_mean - she_mean) / she_mean * 100) if she_mean != 0 else 0,
        "U": u_stat,
        "p": p_value,
        "sig": significance_stars(p_value),
        "d": d,
        "effect": effect_size_label(d),
    }


# ==============================================================
# MAIN ANALYSIS
# ==============================================================
def main():
    print("=" * 72)
    print("  LLM BIAS EXPERIMENT — STATISTICAL NLP ANALYSIS")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ── 1. Load Data ──────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    print(f"\n📊 Dataset: {len(df)} rows loaded from {DB_PATH}")
    print(f"   Models:  {df['model_name'].nunique()} — {', '.join(df['model_name'].unique())}")
    print(f"   Essays:  {df['essay_id'].nunique()}")
    print(f"   Conditions: {', '.join(df['condition'].unique())}")

    # ── 2. Ensure NLTK Data ───────────────────────────────────
    for resource in ['vader_lexicon', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'sentiment/{resource}' if 'vader' in resource else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

    sia = SentimentIntensityAnalyzer()

    # ── 3. Feature Engineering ────────────────────────────────
    print("\n⏳ Computing NLP features...")

    df['word_count'] = df['feedback_text'].apply(word_count)
    df['char_count'] = df['feedback_text'].apply(len)
    df['ttr'] = df['feedback_text'].apply(type_token_ratio)
    df['sentiment_compound'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['compound'])
    df['sentiment_pos'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['pos'])
    df['sentiment_neg'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['neg'])
    df['warmth_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, WARMTH_WORDS))
    df['competence_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, COMPETENCE_WORDS))
    df['hedging_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, HEDGING_WORDS))
    df['directive_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, DIRECTIVE_WORDS))

    # Normalize lexicon counts per 100 words
    df['warmth_per100'] = df['warmth_count'] / df['word_count'] * 100
    df['competence_per100'] = df['competence_count'] / df['word_count'] * 100
    df['hedging_per100'] = df['hedging_count'] / df['word_count'] * 100
    df['directive_per100'] = df['directive_count'] / df['word_count'] * 100

    he_df = df[df['condition'] == 'He/Him']
    she_df = df[df['condition'] == 'She/Her']

    report_lines = []
    def report(line=""):
        print(line)
        report_lines.append(line)

    # ── 4. GLOBAL ANALYSIS ────────────────────────────────────
    report("\n" + "=" * 72)
    report("  SECTION 1: GLOBAL ANALYSIS (All Models Combined)")
    report("=" * 72)

    metrics = [
        ('word_count', 'Response Length (words)'),
        ('sentiment_compound', 'Sentiment (VADER compound)'),
        ('sentiment_pos', 'Positive Sentiment'),
        ('sentiment_neg', 'Negative Sentiment'),
        ('ttr', 'Lexical Diversity (TTR)'),
        ('warmth_per100', 'Warmth Words (per 100 words)'),
        ('competence_per100', 'Competence Words (per 100 words)'),
        ('hedging_per100', 'Hedging Language (per 100 words)'),
        ('directive_per100', 'Directive Language (per 100 words)'),
    ]

    report(f"\n{'Metric':<38} {'He/Him':>10} {'She/Her':>10} {'Diff':>8} {'p-value':>10} {'Sig':>5} {'Cohen d':>8} {'Effect':>12}")
    report("-" * 110)

    global_results = []
    for col, label in metrics:
        result = run_comparison(he_df[col].values, she_df[col].values, label)
        global_results.append(result)
        report(f"{label:<38} {result['he_mean']:>10.4f} {result['she_mean']:>10.4f} {result['diff']:>+8.4f} {result['p']:>10.6f} {result['sig']:>5} {result['d']:>+8.4f} {result['effect']:>12}")

    # ── 5. PER-MODEL ANALYSIS ─────────────────────────────────
    report("\n" + "=" * 72)
    report("  SECTION 2: PER-MODEL BREAKDOWN")
    report("=" * 72)

    for model in sorted(df['model_name'].unique()):
        model_df = df[df['model_name'] == model]
        model_he = model_df[model_df['condition'] == 'He/Him']
        model_she = model_df[model_df['condition'] == 'She/Her']

        report(f"\n── {model} ({'n='}{len(model_he)} He, {len(model_she)} She) ──")
        report(f"{'Metric':<38} {'He/Him':>10} {'She/Her':>10} {'Diff':>8} {'p-value':>10} {'Sig':>5} {'Cohen d':>8} {'Effect':>12}")
        report("-" * 110)

        for col, label in metrics:
            result = run_comparison(model_he[col].values, model_she[col].values, label)
            report(f"{label:<38} {result['he_mean']:>10.4f} {result['she_mean']:>10.4f} {result['diff']:>+8.4f} {result['p']:>10.6f} {result['sig']:>5} {result['d']:>+8.4f} {result['effect']:>12}")

    # ── 6. PER-ESSAY ANALYSIS ─────────────────────────────────
    report("\n" + "=" * 72)
    report("  SECTION 3: PER-ESSAY BREAKDOWN")
    report("=" * 72)

    for essay in sorted(df['essay_id'].unique()):
        essay_df = df[df['essay_id'] == essay]
        essay_he = essay_df[essay_df['condition'] == 'He/Him']
        essay_she = essay_df[essay_df['condition'] == 'She/Her']

        report(f"\n── {essay} (n={len(essay_he)} He, {len(essay_she)} She) ──")
        report(f"{'Metric':<38} {'He/Him':>10} {'She/Her':>10} {'Diff':>8} {'p-value':>10} {'Sig':>5} {'Cohen d':>8} {'Effect':>12}")
        report("-" * 110)

        for col, label in metrics:
            result = run_comparison(essay_he[col].values, essay_she[col].values, label)
            report(f"{label:<38} {result['he_mean']:>10.4f} {result['she_mean']:>10.4f} {result['diff']:>+8.4f} {result['p']:>10.6f} {result['sig']:>5} {result['d']:>+8.4f} {result['effect']:>12}")

    # ── 7. TF-IDF DIVERGENT VOCABULARY ────────────────────────
    report("\n" + "=" * 72)
    report("  SECTION 4: TOP DIVERGENT VOCABULARY (TF-IDF)")
    report("=" * 72)
    report("  Words that appear disproportionately more in one condition vs the other.\n")

    for model in sorted(df['model_name'].unique()):
        model_df = df[df['model_name'] == model]
        he_corpus = " ".join(model_df[model_df['condition'] == 'He/Him']['feedback_text'].values)
        she_corpus = " ".join(model_df[model_df['condition'] == 'She/Her']['feedback_text'].values)

        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
        tfidf_matrix = vectorizer.fit_transform([he_corpus, she_corpus])
        feature_names = vectorizer.get_feature_names_out()
        
        he_scores = tfidf_matrix[0].toarray().flatten()
        she_scores = tfidf_matrix[1].toarray().flatten()
        diff_scores = he_scores - she_scores

        top_he_idx = diff_scores.argsort()[-15:][::-1]
        top_she_idx = diff_scores.argsort()[:15]

        report(f"── {model} ──")
        report(f"  Words skewing toward He/Him condition:")
        for idx in top_he_idx:
            report(f"    {feature_names[idx]:<20} (Δ TF-IDF: {diff_scores[idx]:>+.4f})")
        report(f"  Words skewing toward She/Her condition:")
        for idx in top_she_idx:
            report(f"    {feature_names[idx]:<20} (Δ TF-IDF: {diff_scores[idx]:>+.4f})")
        report("")

    # ── 8. INTERACTION EFFECTS (Model × Condition) ────────────
    report("\n" + "=" * 72)
    report("  SECTION 5: SIGNIFICANT FINDINGS SUMMARY")
    report("=" * 72)

    sig_findings = []
    for model in sorted(df['model_name'].unique()):
        model_df = df[df['model_name'] == model]
        model_he = model_df[model_df['condition'] == 'He/Him']
        model_she = model_df[model_df['condition'] == 'She/Her']
        for col, label in metrics:
            result = run_comparison(model_he[col].values, model_she[col].values, label)
            if result['p'] < 0.05:
                sig_findings.append((model, label, result))

    if sig_findings:
        report(f"\n  Found {len(sig_findings)} statistically significant differences (p < 0.05):\n")
        for model, label, r in sig_findings:
            direction = "He > She" if r['diff'] > 0 else "She > He"
            report(f"  • [{r['sig']}] {model} — {label}")
            report(f"    He/Him: {r['he_mean']:.4f} vs She/Her: {r['she_mean']:.4f} ({direction})")
            report(f"    p = {r['p']:.6f}, Cohen's d = {r['d']:+.4f} ({r['effect']})")
            report("")
    else:
        report("\n  No statistically significant differences found at p < 0.05.\n")

    # ── 9. Save Report ────────────────────────────────────────
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ Full report saved to: {REPORT_PATH}")

    # ── 10. Save processed DataFrame ──────────────────────────
    csv_path = "bias_analysis_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Feature-enriched dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()
