# Repository instructions

## Memory and context

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.
Current Honcho MCP tools expose peer cards, conclusions, chat over peer
representations, and dream scheduling (`get_peer_card`, `set_peer_card`,
`list_conclusions`, `create_conclusions`, `chat`, `schedule_dream`). Use those
current names rather than older `search`/`create_conclusion` notes.

Separate confirmed facts from inference. Treat facts from files and command
outputs as confirmed; treat remembered context and architectural guesses as
inference unless verified locally.

## Tooling rules

Code navigation uses three MCP servers with a strict division of labour:
`fff` to locate, `codegraph` to understand structure, `serena` to read a
symbol precisely and edit it. Do not duplicate them — each owns one job.

Verified local tool versions on 2026-06-09: `fff-mcp 0.9.3`, `Serena 1.5.4.dev0`,
and `codegraph 0.9.9`.

### fff — locate files and literal text

For any file search or grep in the current git repository, use fff first.
Do not use shell `find`, `grep`, or `rg` when fff can express the query.

Use fff for:

- finding files by name or pattern;
- searching literal text — strings, comments, log messages;
- discovering entry points, scripts, library modules, and tests;
- narrowing the area to inspect before using codegraph or Serena.

Search one bare identifier per query; after two grep calls, read the code
instead of grepping variations.

### codegraph — structural questions over the symbol graph

`codegraph` is a tree-sitter knowledge graph (SQLite) of every symbol, edge,
and file. Use it for structural questions, not literal text:

- `codegraph_context "<task>"` — PRIMARY: entry points + related symbols +
  code in one call. Start here for any feature, bug, or unfamiliar area.
- `codegraph_search` — find a symbol by name (kind + signature + location);
  prefer this over `fff grep` when looking up a symbol by name.
- `codegraph_callers` / `codegraph_callees` — who calls / what is called.
- `codegraph_impact <symbol>` — blast radius before a refactor.
- `codegraph_node` — a symbol's source / signature / docstring.
- `codegraph_explore` — deeper architecture/module exploration. Use it after
  `codegraph_search` or `codegraph_context` has surfaced concrete symbol or
  file names; prefer one precise explore call over a grep/read loop. In
  CodeGraph 0.8+, explore source sections include line numbers for direct
  `file:line` citations.
- `codegraph_files` / `codegraph_status` — directory layout / index health.

Trust codegraph results — they come from a full AST parse; do not re-verify
with grep. Do not query the index in the same turn as a file edit — the
watcher debounces ~500 ms behind writes.

### serena — symbolic navigation and symbol-level edits

After fff/codegraph identify the relevant area, use Serena for LSP-precise
navigation and symbol-level edits — Serena is the only one of the three that
edits code:

- `get_symbols_overview`;
- `find_symbol`;
- `find_referencing_symbols` — LSP-accurate references; final check before
  `rename_symbol` (use `codegraph_impact` for the quick blast-radius estimate);
- `replace_symbol_body`;
- `insert_before_symbol`;
- `insert_after_symbol`;
- `rename_symbol`;
- `safe_delete_symbol`.

Prefer Serena tools over reading or rewriting full source files when symbolic
tools are sufficient. Serena runs with `--project-from-cwd`; do not ask to
manually activate the repository when Serena is already available.

### codegraph index sync

The MCP server watches the project and auto-syncs the graph (~2 s debounce).
Still, keep the index fresh explicitly:

- at the start of a work session, run `codegraph status` — if it reports
  pending changes, run `codegraph sync`;
- after any bulk external change the watcher may have missed — `git pull`,
  branch switch, mass file/artifact generation — run `codegraph sync` before
  relying on codegraph answers;
- if a codegraph result contradicts what you see in a file, the index is
  stale: `codegraph sync` and re-query.
- if the workspace is on a slow or WSL `/mnt/*` filesystem and watcher startup
  is a problem, run MCP with `codegraph serve --mcp --no-watch` and rely on
  explicit `codegraph sync` or CodeGraph-installed git hooks.

Standard cycle: locate (`fff` / `codegraph_search`) → understand
(`codegraph_context`, then `codegraph_explore` for deep architecture questions)
→ assess risk (`codegraph_impact`) → read and edit (`serena`) → verify (run the
affected script or the focused tests).

Use Context7 before relying on memory for version-sensitive library behavior,
especially CatBoost, MAPIE, BoFire, BoTorch/GPyTorch, BayBE, scikit-learn,
Optuna, RDKit, pandas, NumPy, or pytest.

Use Tavily MCP tools for fresh external information: releases, changelogs,
documentation gaps, external datasets, comparisons, or web research.

Use a skill when the task matches its description.

## Project purpose

`adsorb_synthesis` is an inverse-design framework for porous metal-organic
frameworks (MOFs). It solves the materials-science problem of finding synthesis
conditions that yield a material with prescribed structural-energetic
characteristics. Three target properties are modelled:

- `E0` [kJ/mol] — characteristic adsorption energy;
- `x0` [nm] — characteristic pore half-width;
- `Sme` [m²/g] — mesopore specific surface area.

The end-to-end research pipeline is:

```text
CSV synthesis dataset -> descriptor enrichment -> forward model (CatBoost + MAPIE)
  -> UQ + external chemistry holdout -> target-oriented inverse design (BoFire)
```

The production path is CatBoost + MAPIE for forward modelling and native BoFire
for inverse optimization. Wave 2 research challengers (TabPFN, BoTorch, BayBE,
direct inverse) share a single benchmark and comparative report. The detailed
decision is recorded in `wave2_decision.md`; see `README.md` for the full
methodology.

## Repository areas

Core package `src/adsorb_synthesis/`:

- `config.py` — model configuration (per-target tuned CatBoost hyperparameters);
- `constants.py` — targets, features, physics bounds, lookup constants
  (single source of truth for config);
- `data_processing.py` — feature engineering and descriptor generation
  (DataFrame mutators support an `inplace` flag);
- `data_validation.py` — synthesis constraint validation (`warn`/`strict`);
- `feature_selection.py` — domain + statistical feature selection (correlation,
  VIF, permutation importance), run per outer fold to avoid leakage;
- `forward_modeling.py` — forward model helpers (CatBoost ensemble + MAPIE);
- `holdout_evaluation.py` — deterministic chemistry-split holdout and metrics;
- `inverse_optimization.py` — inverse-design optimizer logic;
- `molar_masses.py` — molar-mass lookups for reagents;
- `physics_losses.py` — physics-informed constraints and penalties.

Other areas:

- `scripts/` — runnable workflow, benchmark, ingestion, and report scripts;
- `tests/` — pytest suite (unit + regression);
- `data/` — experimental and external synthesis datasets (CSV/XLS);
- `artifacts/` — generated models, metrics, plots, and optimization results
  (gitignored — do not commit their contents).

Active production keys: forward backend `catboost`; inverse backend
`run_bofire_opt.py` (native BoFire). `run_bofire_optuna_legacy.py` and
`run_bayes_opt.py` are historical fallbacks; do not reintroduce removed
experiment families unless a new research task explicitly starts one.

## Key commands

Set up the environment (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite (the package lives under `src/`, so set `PYTHONPATH`):

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Main workflow entry points:

```bash
# Step 0: enrich descriptors (RDKit + coordination chemistry)
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv

# Step 1 (optional): Optuna HP tuning with fold-local feature selection
PYTHONPATH=src python scripts/tune_hyperparams.py \
    --data data/SEC_SYN_with_features_enriched.csv --trials 80

# Step 2: train production forward model (CatBoost ensemble + MAPIE)
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost --validation-mode warn

# Step 3: internal UQ calibration diagnostic (rejection plots + coverage)
PYTHONPATH=src python scripts/validate_uncertainty.py

# Step 3b: external chemistry holdout (deterministic Metal|Ligand split)
PYTHONPATH=src python scripts/evaluate_forward_holdout.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --output-dir artifacts/forward_holdout --backend all

# Step 4: production inverse design (native BoFire) — target SEC -> shortlist
PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 --x0 0.5 --Sme 100.0 \
    --trials 300 --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv

# Step 5 (optional): wave 2 comparative suite
PYTHONPATH=src python scripts/run_wave2_suite.py --mode full \
    --data data/SEC_SYN_with_features_enriched.csv \
    --catboost-models artifacts/forward_models \
    --E0 15 --x0 0.5 --Sme 100 \
    --run-dir artifacts/wave2_runs/selection_run
```

No global lint/format runner is configured. After changes, run the affected
script directly to check imports/runtime, plus the relevant tests.

## Coding conventions

- Use `from __future__ import annotations` at the top of Python modules.
- Use type hints throughout; prefer `X | Y` union syntax (Python 3.10+).
- Use `@dataclass` for data-holding structs.
- Concise module-level docstrings are fine; keep names self-documenting and do
  not add per-function docstrings by default.
- Datasets and predictions are CSV; metrics/config/tuning outputs are JSON;
  fitted models and conformal calibrators are joblib (`.joblib`).
- Feature selection runs separately inside each outer CV fold — preserve this
  no-leakage boundary.
- DataFrame mutators expose an `inplace` flag (`inplace=True` default); use
  `inplace=False` when a safe copy is needed.

## Scope discipline

Before changing code, briefly state what was found, what will change, and why.

Prefer minimal, localized edits over broad refactors.

Do not rename files, move modules, change public interfaces, install
dependencies, edit secrets, or touch generated outputs unless explicitly
required.

Do not commit contents of `artifacts/` (models, predictions, plots, results).

When finishing a code task: run the focused relevant tests when possible, run
the full suite for broad/shared changes, check `git status --short`, and report
only task-related changes. Do not revert unrelated dirty work.
