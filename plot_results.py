"""Regenerate comparison figures from completed experiment CSV files."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path(__file__).parent / "results"
NAMES = {"tabm":"TabM", "tabpfn":"TabPFN", "logistic":"Logistinė regresija", "rule":"Rizikos taisyklė"}

def main():
    sns.set_theme(style="whitegrid")
    raw = pd.read_csv(OUT / "results_by_seed.csv")
    one = raw[raw.fpr_target == .01].copy()
    one["Metodas"] = one.method.map(NAMES)
    order = ["Logistinė regresija", "TabM", "TabPFN", "Rizikos taisyklė"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(one, x="Metodas", y="recall", order=order, errorbar="sd", capsize=.08, ax=ax, palette="deep")
    ax.set(xlabel="", ylabel="Recall, FPR tikslas 1 %", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(OUT / "recall_comparison.png", dpi=180); plt.close(fig)

    summary = pd.read_csv(OUT / "results_summary.csv")
    one = summary[summary.fpr_target == .01].copy(); one["Metodas"] = one.method.map(NAMES)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(one, x="Metodas", y="ap_mean", order=order, ax=ax, palette="deep")
    ax.set(xlabel="", ylabel="Vidutinis AP", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(OUT / "ap_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(one, x="Metodas", y="false_alerts_per_1000_mean", order=order, ax=ax, palette="deep")
    ax.set(xlabel="", ylabel="Klaidingi perspėjimai 1 000 teisėtų svetainių")
    fig.tight_layout(); fig.savefig(OUT / "false_alerts_comparison.png", dpi=180); plt.close(fig)

if __name__ == "__main__": main()
