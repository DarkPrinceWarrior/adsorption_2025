#!/usr/bin/env python3
"""Generate all figures for the paper from real pipeline artifacts.

Run from repo root:
    PYTHONPATH=src .venv/bin/python docs/paper/make_figures.py
Outputs PNG (300 dpi) into docs/paper/figures/.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "docs", "paper", "figures")
DATA = os.path.join(ROOT, "data", "SEC_SYN_with_features_enriched.csv")
MODELS = os.path.join(ROOT, "artifacts", "forward_models")
TABPFN = os.path.join(ROOT, "artifacts", "forward_models_tabpfn3")
LOGO = os.path.join(ROOT, "artifacts", "forward_logo")

TARGETS = ["E0, кДж/моль", "х0, нм", "Sme, м2/г"]
TARGET_TEX = {"E0, кДж/моль": "$E_0$, кДж/моль", "х0, нм": "$x_0$, нм", "Sme, м2/г": "$S_{me}$, м$^2$/г"}
TARGET_SHORT = {"E0, кДж/моль": "$E_0$", "х0, нм": "$x_0$", "Sme, м2/г": "$S_{me}$"}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

C = {
    "blue": "#2E5E8C",
    "lightblue": "#7FA8C9",
    "red": "#B5443C",
    "orange": "#D98841",
    "green": "#4C7F4C",
    "gray": "#666666",
    "lightgray": "#BBBBBB",
}
TARGET_COLORS = {TARGETS[0]: C["blue"], TARGETS[1]: C["green"], TARGETS[2]: C["red"]}


def safe_target(t: str) -> str:
    return t.replace("/", "_").replace(" ", "_")


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))
    plt.close(fig)
    print("saved", name)


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


# ----------------------------------------------------------------- Fig 1: pipeline scheme
def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, lines, fc="#EDF2F7", ec=C["blue"], title_c="#1A365D",
            title_fs=10.0, line_fs=8.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h - 0.33, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=title_c, zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.70 - i * 0.32, line, ha="center", va="center",
                    fontsize=line_fs, color="#222222", zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, color=C["gray"], zorder=1))

    # top row: data flow (3 boxes)
    box(0.20, 4.55, 3.30, 1.75, "Датасет синтезов",
        ["380 образцов, 17 систем", "9 параметров синтеза", "СЭХ из изотерм N$_2$ (77 K)"])
    box(4.10, 4.55, 3.30, 1.75, "Обогащение дескрипторами",
        ["стехиометрия, концентрации,", "дескрипторы металла, лиганда,", "растворителя; скрытая вода"])
    box(8.00, 4.55, 3.30, 1.75, "Валидация данных",
        ["физико-химические проверки:", "температуры, T кипения,", "стехиометрия, соотношения СЭХ"])
    arrow(3.50, 5.42, 4.10, 5.42)
    arrow(7.40, 5.42, 8.00, 5.42)

    # middle row: forward model + validation + challenger
    box(0.20, 2.25, 3.85, 1.75, "Прямая модель: синтез → СЭХ",
        ["ансамбль CatBoost (5 × 3 таргета)", "отбор признаков внутри фолда", "конформные интервалы MAPIE"],
        fc="#E8F0E8", ec=C["green"], title_c="#1E3B1E")
    box(4.65, 2.25, 3.40, 1.75, "Валидация модели",
        ["групповая CV по рецептурам", "химический холдаут", "LOGO-CV + y-рандомизация"],
        fc="#FDF6EC", ec=C["orange"], title_c="#6B4A12")
    box(8.65, 2.25, 2.65, 1.75, "Сравнение: TabPFN-3",
        ["табличный foundation-", "трансформер, единый", "протокол оценки"], fc="#F2F2F2", ec=C["gray"])
    arrow(2.10, 4.55, 2.10, 4.00)
    arrow(4.05, 3.12, 4.65, 3.12)
    arrow(8.05, 3.12, 8.65, 3.12)

    # bottom row: inverse design
    box(0.20, 0.0, 5.55, 1.75, "Обратный дизайн (BoFire)",
        ["qParEGO + qLogNEI, целевые СЭХ", "физико-химические ограничения,", "проекция в допустимую область"],
        fc="#EAE4F0", ec="#5B4A78", title_c="#3B2D52")
    box(6.35, 0.0, 4.95, 1.75, "Список рекомендуемых рецептур",
        ["металл, лиганд, растворитель,", "массы, объём, T-режимы +", "интервалы предсказаний, ранг"],
        fc="#F7F0EA", ec=C["red"], title_c="#6B2420")
    arrow(2.10, 2.25, 2.10, 1.75)
    arrow(5.75, 0.87, 6.35, 0.87)

    save(fig, "fig1_pipeline")


# ----------------------------------------------------------------- Fig 2: dataset overview
def fig2_dataset() -> None:
    df = pd.read_csv(DATA)
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34)

    # (a) group sizes
    ax = fig.add_subplot(gs[0, :2])
    groups = (df["Металл"].astype(str) + "|" + df["Лиганд"].astype(str)).value_counts()
    colors = [C["blue"] if v >= 10 else (C["orange"] if v >= 3 else C["red"]) for v in groups.values]
    ax.bar(range(len(groups)), groups.values, color=colors, width=0.72)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.index, rotation=45, ha="right", fontsize=8.6)
    ax.set_ylabel("Число образцов")
    for i, v in enumerate(groups.values):
        ax.text(i, v + 2.0, str(v), ha="center", fontsize=8)
    ax.set_ylim(0, groups.max() * 1.12)
    panel_label(ax, "(а)")

    # (b) E0 vs x0 + Dubinin-Stoeckli
    ax = fig.add_subplot(gs[0, 2])
    e0 = pd.to_numeric(df[TARGETS[0]], errors="coerce")
    x0 = pd.to_numeric(df[TARGETS[1]], errors="coerce")
    ax.scatter(e0, x0, s=16, alpha=0.55, c=C["blue"], edgecolors="white", linewidths=0.4)
    grid_e0 = np.linspace(max(e0.min(), 8), e0.max(), 200)
    ax.plot(grid_e0, 12.0 / grid_e0, "--", color=C["red"], linewidth=1.6, label="$x_0 = 12/E_0$")
    ax.set_xlabel(TARGET_TEX[TARGETS[0]])
    ax.set_ylabel(TARGET_TEX[TARGETS[1]])
    ax.legend(frameon=False)
    panel_label(ax, "(б)")

    # (c,d,e) target histograms
    for i, (t, lab) in enumerate(zip(TARGETS, ["(в)", "(г)", "(д)"])):
        ax = fig.add_subplot(gs[1, i])
        vals = pd.to_numeric(df[t], errors="coerce").dropna()
        ax.hist(vals, bins=32, color=TARGET_COLORS[t], alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.axvline(vals.median(), color="#222222", linestyle="--", linewidth=1.1)
        ax.text(0.97, 0.92, f"медиана = {vals.median():.3g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.6)
        if t == TARGETS[2]:
            ax.text(0.97, 0.80, f"нулей: {(vals == 0).sum()}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.6, color=C["red"])
        ax.set_xlabel(TARGET_TEX[t])
        ax.set_ylabel("Число образцов")
        panel_label(ax, lab)

    save(fig, "fig2_dataset")


# ----------------------------------------------------------------- Fig 3: parity plots
def fig3_parity() -> None:
    metrics = json.load(open(os.path.join(MODELS, "metrics.json")))
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.1))
    for ax, t, lab in zip(axes, TARGETS, ["(а)", "(б)", "(в)"]):
        preds = pd.read_csv(os.path.join(MODELS, f"predictions_{safe_target(t)}.csv"))
        y, yhat = preds["y_actual"].to_numpy(), preds["y_oof"].to_numpy()
        lo, hi = preds["y_lo"].to_numpy(), preds["y_hi"].to_numpy()
        center = (lo + hi) / 2.0
        ax.errorbar(y, center, yerr=np.vstack([center - lo, hi - center]), fmt="none",
                    ecolor=C["lightgray"], elinewidth=0.7, alpha=0.5, zorder=1)
        ax.scatter(y, yhat, s=20, c=TARGET_COLORS[t], alpha=0.65, edgecolors="white",
                   linewidths=0.4, zorder=2)
        lim = [min(y.min(), lo.min()), max(y.max(), hi.max())]
        pad = (lim[1] - lim[0]) * 0.04
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, "k--", linewidth=1.0, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        m = metrics[t]
        ax.set_title(f"{TARGET_TEX[t]}\n$R^2$={m['R2_oof']:.2f}, RMSE={m['RMSE_oof']:.3g}, MAE={m['MAE_oof']:.3g}")
        ax.set_xlabel("Эксперимент")
        ax.set_ylabel("Предсказание (OOF)")
        panel_label(ax, lab)
    fig.tight_layout()
    save(fig, "fig3_parity")


# ----------------------------------------------------------------- Fig 4: UQ
def fig4_uq() -> None:
    metrics = json.load(open(os.path.join(MODELS, "metrics.json")))
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))

    # (a) coverage
    ax = axes[0]
    cov = [metrics[t]["interval_coverage"] for t in TARGETS]
    ax.bar(range(3), cov, color=[TARGET_COLORS[t] for t in TARGETS], width=0.6, alpha=0.85)
    ax.axhline(0.9, linestyle="--", color="#222222", linewidth=1.2)
    ax.text(2.42, 0.905, "цель 0.90", fontsize=8.6, va="bottom", ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
    ax.set_ylim(0.0, 1.07)
    ax.set_ylabel("Эмпирическое покрытие")
    for i, v in enumerate(cov):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.6)
    panel_label(ax, "(а)")

    # (b) normalized width
    ax = axes[1]
    wn = [metrics[t]["interval_width_normalized"] for t in TARGETS]
    ax.bar(range(3), wn, color=[TARGET_COLORS[t] for t in TARGETS], width=0.6, alpha=0.85)
    ax.set_xticks(range(3)); ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
    ax.set_ylabel("Ширина интервала, $\\times\\sigma$ таргета")
    for i, v in enumerate(wn):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.6)
    ax.set_ylim(0, max(wn) * 1.18)
    panel_label(ax, "(б)")

    # (c) rejection curves
    ax = axes[2]
    for t in TARGETS:
        preds = pd.read_csv(os.path.join(MODELS, f"predictions_{safe_target(t)}.csv"))
        err = np.abs(preds["y_actual"] - preds["y_oof"]).to_numpy()
        width = preds["interval_width"].to_numpy()
        order = np.argsort(width)
        err_sorted = err[order]
        fracs = np.linspace(0.2, 1.0, 33)
        rmse0 = float(np.sqrt(np.mean(err ** 2)))
        curve = [np.sqrt(np.mean(err_sorted[: max(2, int(f * len(err_sorted))) ] ** 2)) / rmse0 for f in fracs]
        ax.plot(fracs * 100, curve, linewidth=1.8, color=TARGET_COLORS[t], label=TARGET_SHORT[t])
    ax.axhline(1.0, linestyle=":", color="#888888", linewidth=1.0)
    ax.set_xlabel("Доля принятых предсказаний, %\n(по возрастанию ширины интервала)")
    ax.set_ylabel("RMSE / RMSE$_{полн}$")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "(в)")

    fig.tight_layout()
    save(fig, "fig4_uq")


# ----------------------------------------------------------------- Fig 5: generalization
def fig5_generalization() -> None:
    metrics = json.load(open(os.path.join(MODELS, "metrics.json")))
    logo = json.load(open(os.path.join(LOGO, "logo_metrics.json")))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.05, 1.3]})

    # (a) OOF vs LOGO vs null
    ax = axes[0]
    x = np.arange(3)
    w = 0.26
    oof = [metrics[t]["R2_oof"] for t in TARGETS]
    pooled = [logo["targets"][t]["R2_pooled"] for t in TARGETS]
    null = [logo["targets"][t]["y_scramble_null_R2_mean"] for t in TARGETS]
    ax.bar(x - w, oof, w, color=C["blue"], alpha=0.88, label="Внутренняя CV (OOF)")
    ax.bar(x, pooled, w, color=C["red"], alpha=0.88, label="LOGO (новая химия)")
    ax.bar(x + w, null, w, color=C["lightgray"], label="y-scrambling (null)")
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(x); ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
    ax.set_ylabel("$R^2$")
    for xi, v in zip(x - w, oof):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.2)
    for xi, v in zip(x, pooled):
        ax.text(xi, v - 0.10, f"{v:.2f}", ha="center", fontsize=8.2)
    for xi, v in zip(x + w, null):
        ax.text(xi, v - 0.10, f"{v:.2f}", ha="center", fontsize=8.2)
    ax.set_ylim(-1.05, 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=8.6)
    panel_label(ax, "(а)")

    # (b) per-group LOGO R2 vs group size (E0)
    ax = axes[1]
    markers = {"E0, кДж/моль": "o", "х0, нм": "s", "Sme, м2/г": "^"}
    for t in TARGETS:
        per_group = logo["targets"][t]["per_group"]
        ns, r2s = [], []
        for g, vals in per_group.items():
            if vals["R2"] is None or vals["n"] < 2:
                continue
            ns.append(vals["n"])
            r2s.append(max(vals["R2"], -8.0))
        ax.scatter(ns, r2s, s=42, alpha=0.75, c=TARGET_COLORS[t], marker=markers[t],
                   edgecolors="white", linewidths=0.5, label=TARGET_SHORT[t])
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Размер удерживаемой группы Металл|Лиганд (log)")
    ax.set_ylabel("$R^2$ на группе (обрезано до −8)")
    ax.set_ylim(-8.8, 1.6)
    ax.legend(frameon=False, loc="upper right", ncol=3, columnspacing=1.0)
    panel_label(ax, "(б)")

    fig.tight_layout()
    save(fig, "fig5_generalization")


# ----------------------------------------------------------------- Fig 6: feature importance
def fig6_importance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, t, lab in zip(axes, TARGETS, ["(а)", "(б)", "(в)"]):
        imps, names = [], None
        for k in range(5):
            path = os.path.join(MODELS, f"catboost_{safe_target(t)}_ens{k}.cbm")
            model = CatBoostRegressor()
            model.load_model(path)
            imps.append(model.get_feature_importance())
            names = list(model.feature_names_)
        imp = np.mean(imps, axis=0)
        std = np.std(imps, axis=0)
        order = np.argsort(imp)[-10:]
        ax.barh(range(len(order)), imp[order], xerr=std[order], color=TARGET_COLORS[t],
                alpha=0.85, error_kw={"elinewidth": 0.9, "ecolor": "#444444"})
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8.2)
        ax.set_xlabel("Важность признака, %")
        ax.set_title(TARGET_TEX[t])
        panel_label(ax, lab)
    fig.tight_layout()
    save(fig, "fig6_importance")


# ----------------------------------------------------------------- Fig 7: CatBoost vs TabPFN
def fig7_backends() -> None:
    cb = json.load(open(os.path.join(MODELS, "metrics.json")))
    tp = json.load(open(os.path.join(TABPFN, "metrics.json")))
    logo = json.load(open(os.path.join(LOGO, "logo_metrics.json")))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(3)
    w = 0.22
    ax.bar(x - 1.0 * w, [cb[t]["R2_oof"] for t in TARGETS], w, color=C["blue"], alpha=0.9,
           label="CatBoost — внутренняя CV")
    ax.bar(x, [tp[t]["R2_oof"] for t in TARGETS], w, color=C["lightblue"], alpha=0.9,
           label="TabPFN-3 — внутренняя CV")
    ax.bar(x + 1.0 * w, [logo["targets"][t]["R2_pooled"] for t in TARGETS], w, color=C["red"],
           alpha=0.85, label="CatBoost — LOGO (новая химия)")
    ax.axhline(0, color="#222222", linewidth=0.9)
    for xs, vals in [(x - w, [cb[t]["R2_oof"] for t in TARGETS]),
                     (x, [tp[t]["R2_oof"] for t in TARGETS]),
                     (x + w, [logo["targets"][t]["R2_pooled"] for t in TARGETS])]:
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.03 if v > 0 else v - 0.10, f"{v:.2f}", ha="center", fontsize=8.0)
    ax.set_xticks(x); ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-1.0, 1.05)
    ax.legend(frameon=False, fontsize=8.8, loc="lower left")
    save(fig, "fig7_backends")


# ----------------------------------------------------------------- Fig 8: inverse design
def fig8_inverse() -> None:
    all_path = os.path.join(ROOT, "docs", "paper", "predictions_bofire_paper_all.csv")
    short_path = os.path.join(ROOT, "docs", "paper", "predictions_bofire_paper.csv")
    if not (os.path.exists(all_path) and os.path.exists(short_path)):
        print("fig8 skipped: BoFire run not finished")
        return
    allr = pd.read_csv(all_path)
    short = pd.read_csv(short_path)
    targets = {"E0, кДж/моль": 15.0, "х0, нм": 0.5, "Sme, м2/г": 100.0}

    fig = plt.figure(figsize=(12.3, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.36)

    # (a) convergence
    ax = fig.add_subplot(gs[0, 0])
    feas = allr.sort_values("search_rank")
    score = feas["score"].where(feas["feasible"], np.nan)
    best = score.copy()
    running = np.inf
    bvals = []
    for s in score:
        if np.isfinite(s) and s < running:
            running = s
        bvals.append(running if np.isfinite(running) else np.nan)
    ax.plot(feas["search_rank"], score, ".", color=C["lightblue"], markersize=4, alpha=0.6,
            label="кандидаты (feasible)")
    ax.plot(feas["search_rank"], bvals, "-", color=C["red"], linewidth=1.8, label="лучший score")
    ax.set_yscale("log")
    ax.set_xlabel("Итерация поиска")
    ax.set_ylabel("Целевая функция (score)")
    ax.legend(frameon=False, fontsize=8.6)
    panel_label(ax, "(а)")

    # (b,c) shortlist predictions vs targets
    for j, (t, lab) in enumerate(zip(["E0, кДж/моль", "Sme, м2/г"], ["(б)", "(в)"])):
        ax = fig.add_subplot(gs[0, j + 1])
        n = len(short)
        mean = short[f"Pred_{t}"]
        lo = short[f"Pred_{t}_lo"]
        hi = short[f"Pred_{t}_hi"]
        ax.errorbar(range(1, n + 1), mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="o",
                    markersize=5, color=TARGET_COLORS[t], ecolor=C["lightgray"], elinewidth=1.2,
                    capsize=2.5)
        ax.axhline(targets[t], linestyle="--", color="#222222", linewidth=1.2)
        ax.text(n + 0.3, targets[t], "цель", fontsize=8.6, va="bottom", ha="right")
        ax.set_xlabel("Ранг кандидата в шортлисте")
        ax.set_ylabel(TARGET_TEX[t])
        ax.set_xticks(range(1, n + 1, 2))
        panel_label(ax, lab)

    save(fig, "fig8_inverse")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_pipeline()
    fig2_dataset()
    fig3_parity()
    fig4_uq()
    fig5_generalization()
    fig6_importance()
    fig7_backends()
    fig8_inverse()
    print("done")
