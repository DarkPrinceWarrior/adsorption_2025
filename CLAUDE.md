# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tooling

Code navigation uses three MCP servers, each with one job — do not duplicate them:

- **fff** — locate files and literal text (strings, comments, log messages).
  Use fff instead of shell `find`/`grep`/`rg`. One bare identifier per query;
  after two greps, read the code.
- **codegraph** — structural questions over a tree-sitter symbol graph.
  `codegraph_context "<task>"` is the primary tool (entry points + related
  symbols + code in one call). Also `codegraph_search` (symbol by name —
  prefer over `fff grep`), `codegraph_callers`/`codegraph_callees`,
  `codegraph_impact` (blast radius before a refactor), `codegraph_node`,
  `codegraph_explore` (deeper architecture/module exploration after
  `codegraph_search` or `codegraph_context` surfaces concrete symbol/file
  names; in CodeGraph 0.8+ source sections include line numbers for direct
  `file:line` citations).
  Trust its results — full AST parse; do not re-verify with grep.
- **serena** — LSP-precise symbol navigation and the only tool that *edits*
  at symbol level (`find_symbol`, `get_symbols_overview`,
  `find_referencing_symbols`, `replace_symbol_body`, `insert_*`,
  `rename_symbol`, `safe_delete_symbol`). Prefer over reading whole files.
  Runs with `--project-from-cwd`; no manual activation needed.

Cycle: locate (fff / `codegraph_search`) → understand (`codegraph_context`,
then one precise `codegraph_explore` for deep architecture questions) → assess
risk (`codegraph_impact`) → read and edit (serena) → verify.

**codegraph index sync** — the MCP server auto-syncs (~2 s debounce), but keep
it fresh explicitly: run `codegraph status` at the start of a session and
`codegraph sync` if it reports pending changes, or after any bulk change the
watcher may miss (`git pull`, branch switch, mass file/artifact generation). If
a codegraph answer contradicts the file, the index is stale — `codegraph sync`
and re-query. Do not query the index in the same turn as an edit (~500 ms lag).
On slow or WSL `/mnt/*` filesystems, use `codegraph serve --mcp --no-watch`
and rely on explicit `codegraph sync` or CodeGraph-installed git hooks.

Other MCP: **context7** for version-sensitive library docs (CatBoost, MAPIE,
BoFire, BoTorch, BayBE, scikit-learn, RDKit, Optuna — prefer over web search);
**tavily** for general web search and external dataset discovery; **playwright**
for browser smoke-checks when relevant.

Verified local tool versions on 2026-06-09: `fff-mcp 0.9.3`, `Serena 1.5.4.dev0`,
and `codegraph 0.9.9`.

## Memory (Honcho)

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.

Current Honcho MCP tools expose peer cards, conclusions, chat over peer
representations, and dream scheduling: `get_peer_card`, `set_peer_card`,
`list_conclusions`, `create_conclusions`, `chat`, `schedule_dream`. Use those
current names rather than older `search`/`create_conclusion` notes.

Separate confirmed facts from inference. Treat files and command outputs as
confirmed; treat Honcho memory and architectural guesses as inference unless
verified locally.

## Project Purpose

**adsorb_synthesis** — an inverse-design framework for porous metal-organic
frameworks (MOFs). It finds synthesis conditions that yield a material with
prescribed structural-energetic characteristics. Three target properties:

- `E0` [kJ/mol] — characteristic adsorption energy
- `x0` [nm] — characteristic pore half-width
- `Sme` [m²/g] — mesopore specific surface area

End-to-end pipeline: CSV synthesis dataset → descriptor enrichment → forward
model (CatBoost + MAPIE) → UQ + external chemistry holdout → target-oriented
inverse design (native BoFire). Production stack is CatBoost + MAPIE forward and
native BoFire inverse; wave 2 challengers (TabPFN, BoTorch, BayBE, direct
inverse) share one benchmark and comparative report. See `wave2_decision.md` and
`README.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

The package lives under `src/`, so set `PYTHONPATH=src` for scripts and tests.

### Run tests
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

### Enrich descriptors (Step 0)
```bash
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv
```

### Tune hyperparameters (Step 1, optional)
```bash
PYTHONPATH=src python scripts/tune_hyperparams.py \
    --data data/SEC_SYN_with_features_enriched.csv --trials 80
```
Result: `artifacts/best_hyperparams.json` + a snippet for `config.py`.

### Train production forward model (Step 2)
```bash
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost --validation-mode warn
```
Result in `artifacts/forward_models/`: CatBoost ensemble (5 members × 3 targets),
`metrics.json`, `uncertainty_calibrators.joblib` (MAPIE), and `predictions_*.csv`
with OOF predictions + `y_lo`/`y_hi` intervals.

### Validate UQ (Step 3, internal CV diagnostic)
```bash
PYTHONPATH=src python scripts/validate_uncertainty.py
```
Result: `artifacts/plots/uncertainty_rejection_plots.png`.

### External chemistry holdout (Step 3b)
```bash
PYTHONPATH=src python scripts/evaluate_forward_holdout.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --output-dir artifacts/forward_holdout --backend all
```
Result: `split_manifest.json` + per-backend holdout metrics/predictions.

### Production inverse design (Step 4, native BoFire)
```bash
PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 --x0 0.5 --Sme 100.0 \
    --trials 300 --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv
```
Result: a diverse shortlist CSV with synthesis conditions, predicted properties,
`Pred_*_lo/hi` intervals, `feasible`, `constraint_reasons`, `score`, and
`search_rank`. Historical fallback: `scripts/run_bofire_optuna_legacy.py`.

### Wave 2 comparative suite (Step 5, optional)
```bash
PYTHONPATH=src python scripts/run_wave2_suite.py --mode full \
    --data data/SEC_SYN_with_features_enriched.csv \
    --catboost-models artifacts/forward_models \
    --E0 15 --x0 0.5 --Sme 100 \
    --run-dir artifacts/wave2_runs/selection_run
```
Computes TabPFN / external holdout / BayBE / direct inverse if missing, runs
`benchmark_wave2.py`, and builds a comparative report via `generate_wave2_report.py`.

No linting, formatting, or test runner is otherwise configured. After changes,
run the affected script directly to check for import/runtime errors.

## Server workflow (a100)

Heavy runs execute on the a100 server; the local checkout is for code editing and
MCP navigation. Keep **local ⇄ GitHub ⇄ server on the SAME commit** — no commit may
diverge. Write code locally → commit → `git push` → `git pull` on the server. Never
edit code directly on the server.

- **SSH:** `ssh a100` (LAN/office) or `ssh a100-remote` (jump host `jump-37`, any
  network); both in `~/.ssh/config`. GitHub SSH auth works from the server.
- **Path:** `/root/projects/adsorb_synthesis`, cloned from
  `git@github.com:DarkPrinceWarrior/adsorption_2025.git` (branch `Bayesian-Optimization`).
- **Env (uv):** `uv venv --python 3.13 .venv` + `uv pip install -r requirements.txt`.
  Verified server stack: uv 0.11.8, Python 3.13.5, torch 2.11.0+cu130, CUDA 13.0,
  6× A100-40GB. GPU0 is busy (~8.4 GB) → use `CUDA_VISIBLE_DEVICES=1..5` for GPU jobs.
- **Run heavy jobs on the server in `tmux`**, then copy results back to local `artifacts/`:
  ```bash
  ssh a100 'tmux new -d -s adsorb_logo \
    "cd /root/projects/adsorb_synthesis && PYTHONPATH=src .venv/bin/python \
     scripts/evaluate_chemistry_logo.py --permutations 3 2>&1 | tee logo.log"'
  # train_forward_model.py / feature_stability.py / evaluate_forward_holdout.py likewise
  scp -r a100:/root/projects/adsorb_synthesis/artifacts/forward_logo artifacts/
  ```
- **TabPFN-3 on the server** needs a Prior Labs `TABPFN_TOKEN` in `~/.cache/tabpfn/auth_token`.
- `artifacts/`, `.venv/`, `.codegraph/` stay server-local (gitignored).

## Architecture

### Data flow
```
data/SEC_SYN_with_features.csv
  → scripts/enrich_descriptors.py
  → data/SEC_SYN_with_features_enriched.csv
  → scripts/train_forward_model.py            (CatBoost ensemble + MAPIE)
  → artifacts/forward_models/{metrics.json, uncertainty_calibrators.joblib, predictions_*.csv}
  → scripts/validate_uncertainty.py + scripts/evaluate_forward_holdout.py
  → scripts/run_bofire_opt.py
  → artifacts/predictions_bofire*.csv
```

### Library: `src/adsorb_synthesis/`
The shared package every script imports from. Key modules:
- `constants.py` — targets (`E0`, `x0`, `Sme`), feature lists, physics bounds,
  random seed; single source of truth for config.
- `config.py` — per-target tuned CatBoost hyperparameters.
- `data_processing.py` — descriptor generation and feature engineering;
  DataFrame mutators take an `inplace` flag.
- `data_validation.py` — `validate_synthesis_data` with `warn`/`strict` modes
  (temperature order, boiling point, stoichiometry).
- `feature_selection.py` — domain + statistical selection (correlation |r|>0.85,
  VIF>10, permutation importance), run per outer fold (no leakage).
- `forward_modeling.py` — CatBoost ensemble + MAPIE cross-conformal intervals.
- `holdout_evaluation.py` — deterministic `Metal|Ligand` chemistry split + metrics.
- `inverse_optimization.py` — target-oriented inverse-design optimizer logic.
- `molar_masses.py` — reagent molar-mass lookups.
- `physics_losses.py` — physics-informed constraints and penalties.

### Forward model + UQ
Fold-local CatBoost pipeline (outer CV without feature-selection leakage) plus
cross-conformal `MAPIE` intervals on the same CV scheme. `predictions_<target>.csv`
carry `y_actual`, `y_oof`, `y_prod_mean`, `y_lo`, `y_hi`, `interval_width`.
`validate_uncertainty.py` is the internal coverage/rejection diagnostic;
`evaluate_forward_holdout.py` is the external chemistry-split evaluation.

### Inverse design
Production inverse stage is target-oriented native **BoFire** optimization
(`run_bofire_opt.py`) with explicit constraints (temperature order, boiling
point, stoichiometry, `E0` bounds), emitting a ranked diverse shortlist with
prediction intervals and diagnostics. `run_botorch_mobo.py` and
`run_baybe_campaign.py` are research backends; `train_inverse_direct.py` is a
benchmark baseline only.

### Wave 2 status
- **CatBoost** — current production default forward backend.
- **TabPFN** — `TabPFN-3` (`tabpfn>=8.0.0`, explicit `ModelVersion.V3`) now matches CatBoost on
  internal CV (OOF R² ~0.74–0.81) but both fail the external chemistry holdout (negative R²);
  still a challenger (no interval UQ). v3 weights need a Prior Labs `TABPFN_TOKEN`
  (ux.priorlabs.ai) + license acceptance, not just an HF token.
- **BoFire** — current production inverse default (native strategy loop).
- **BoTorch** — working research backend, still behind BoFire on `best_score_pool`.
- **BayBE** — working campaign backend for the low-data loop, not the best optimizer.
- **Direct inverse** — useful benchmark, not the recommended recipe-search path.
- **Wave 3 finding** — the main remaining bottleneck is chemistry generalization
  of the data, not the orchestration pipeline.

## Conventions

- `from __future__ import annotations` at the top of every module
- Type hints throughout; union types via `X | Y` (Python 3.10+)
- `@dataclass` for data-holding structs
- Concise module-level docstrings are fine; names self-documenting, no
  per-function docstrings by default
- Datasets/predictions in CSV; metrics/config/tuning in JSON; fitted models and
  conformal calibrators in joblib
- Feature selection runs inside each outer CV fold — preserve the no-leakage boundary

## Output locations
- Enriched dataset: `data/SEC_SYN_with_features_enriched.csv`
- Forward models + metrics: `artifacts/forward_models/` (models, `metrics.json`,
  `uncertainty_calibrators.joblib`, `predictions_*.csv`)
- UQ plots: `artifacts/plots/uncertainty_rejection_plots.png`
- External holdout: `artifacts/forward_holdout/`
- Inverse shortlist: `artifacts/predictions_bofire*.csv`
- Wave 2 runs/reports: `artifacts/wave2_runs/`

`artifacts/` is gitignored — do not commit its contents.
