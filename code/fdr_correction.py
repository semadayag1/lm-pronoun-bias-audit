#!/usr/bin/env python3
"""
FDR Correction Script
=====================
Applies Benjamini-Hochberg correction to all per-model statistical tests
from the bias experiment and outputs corrected p-values.
"""
import sqlite3
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DB_PATH = "bias_experiment.sqlite"

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
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return 0.0
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0.0

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    sia = SentimentIntensityAnalyzer()
    df['word_count'] = df['feedback_text'].apply(lambda t: len(t.split()))
    df['ttr'] = df['feedback_text'].apply(type_token_ratio)
    df['sentiment_compound'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['compound'])
    df['sentiment_pos'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['pos'])
    df['sentiment_neg'] = df['feedback_text'].apply(lambda t: sia.polarity_scores(t)['neg'])
    df['warmth_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, WARMTH_WORDS))
    df['competence_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, COMPETENCE_WORDS))
    df['hedging_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, HEDGING_WORDS))
    df['directive_count'] = df['feedback_text'].apply(lambda t: count_lexicon_hits(t, DIRECTIVE_WORDS))
    df['warmth_per100'] = df.apply(lambda r: safe_per100(r['warmth_count'], r['word_count']), axis=1)
    df['competence_per100'] = df.apply(lambda r: safe_per100(r['competence_count'], r['word_count']), axis=1)
    df['hedging_per100'] = df.apply(lambda r: safe_per100(r['hedging_count'], r['word_count']), axis=1)
    df['directive_per100'] = df.apply(lambda r: safe_per100(r['directive_count'], r['word_count']), axis=1)
    df = df.fillna(0)

    metrics = [
        ('word_count', 'Response Length'),
        ('sentiment_compound', 'Sentiment (compound)'),
        ('sentiment_pos', 'Positive Sentiment'),
        ('sentiment_neg', 'Negative Sentiment'),
        ('ttr', 'Lexical Diversity (TTR)'),
        ('warmth_per100', 'Warmth/100w'),
        ('competence_per100', 'Competence/100w'),
        ('hedging_per100', 'Hedging/100w'),
        ('directive_per100', 'Directive/100w'),
    ]

    # ── Collect ALL per-model tests (the ones we report in the paper) ──
    all_tests = []

    for model in sorted(df['model_name'].unique()):
        mdf = df[df['model_name'] == model]
        he = mdf[mdf['condition'] == 'He/Him']
        she = mdf[mdf['condition'] == 'She/Her']
        for col, label in metrics:
            u, p = stats.mannwhitneyu(he[col].values, she[col].values, alternative='two-sided')
            d = cohens_d(he[col].values, she[col].values)
            all_tests.append({
                'model': model, 'metric': label, 'col': col,
                'he_mean': np.mean(he[col].values),
                'she_mean': np.mean(she[col].values),
                'U': u, 'p_raw': p, 'd': d
            })

    # ── Also collect global tests ──
    he_all = df[df['condition'] == 'He/Him']
    she_all = df[df['condition'] == 'She/Her']
    global_tests = []
    for col, label in metrics:
        u, p = stats.mannwhitneyu(he_all[col].values, she_all[col].values, alternative='two-sided')
        d = cohens_d(he_all[col].values, she_all[col].values)
        global_tests.append({
            'model': 'GLOBAL', 'metric': label, 'col': col,
            'he_mean': np.mean(he_all[col].values),
            'she_mean': np.mean(she_all[col].values),
            'U': u, 'p_raw': p, 'd': d
        })

    # ── Apply BH correction to per-model tests (27 tests) ──
    raw_ps = [t['p_raw'] for t in all_tests]
    reject, corrected_ps, _, _ = multipletests(raw_ps, method='fdr_bh')
    for i, t in enumerate(all_tests):
        t['p_corrected'] = corrected_ps[i]
        t['sig_corrected'] = reject[i]

    # ── Apply BH correction to global tests (9 tests) ──
    raw_ps_g = [t['p_raw'] for t in global_tests]
    reject_g, corrected_ps_g, _, _ = multipletests(raw_ps_g, method='fdr_bh')
    for i, t in enumerate(global_tests):
        t['p_corrected'] = corrected_ps_g[i]
        t['sig_corrected'] = reject_g[i]

    # ── Print results ──
    def sig_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "n.s."

    print("=" * 100)
    print("  GLOBAL ANALYSIS (BH-corrected, 9 tests)")
    print("=" * 100)
    print(f"{'Metric':<25} {'He':>8} {'She':>8} {'Δ':>8} {'p_raw':>10} {'p_BH':>10} {'Sig':>5} {'d':>7}")
    print("-" * 100)
    for t in global_tests:
        diff = t['he_mean'] - t['she_mean']
        print(f"{t['metric']:<25} {t['he_mean']:>8.4f} {t['she_mean']:>8.4f} {diff:>+8.4f} {t['p_raw']:>10.6f} {t['p_corrected']:>10.6f} {sig_stars(t['p_corrected']):>5} {t['d']:>+7.3f}")

    print()
    print("=" * 100)
    print("  PER-MODEL ANALYSIS (BH-corrected across all 27 per-model tests)")
    print("=" * 100)

    for model in sorted(df['model_name'].unique()):
        model_tests = [t for t in all_tests if t['model'] == model]
        print(f"\n── {model} ──")
        print(f"{'Metric':<25} {'He':>8} {'She':>8} {'Δ':>8} {'p_raw':>10} {'p_BH':>10} {'Sig':>5} {'d':>7}")
        print("-" * 100)
        for t in model_tests:
            diff = t['he_mean'] - t['she_mean']
            print(f"{t['metric']:<25} {t['he_mean']:>8.4f} {t['she_mean']:>8.4f} {diff:>+8.4f} {t['p_raw']:>10.6f} {t['p_corrected']:>10.6f} {sig_stars(t['p_corrected']):>5} {t['d']:>+7.3f}")

    # ── Summary of what survives ──
    survivors = [t for t in all_tests if t['sig_corrected']]
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY: {len(survivors)} of 27 per-model tests survive BH correction at α=0.05")
    print(f"{'=' * 100}")
    for t in survivors:
        direction = "He > She" if t['he_mean'] > t['she_mean'] else "She > He"
        print(f"  ✓ {t['model']} — {t['metric']} ({direction}) p_BH={t['p_corrected']:.6f}, d={t['d']:+.3f}")

    dropped = [t for t in all_tests if t['p_raw'] < 0.05 and not t['sig_corrected']]
    if dropped:
        print(f"\n  Dropped after correction ({len(dropped)}):")
        for t in dropped:
            print(f"  ✗ {t['model']} — {t['metric']} p_raw={t['p_raw']:.6f} → p_BH={t['p_corrected']:.6f}")

if __name__ == "__main__":
    main()
