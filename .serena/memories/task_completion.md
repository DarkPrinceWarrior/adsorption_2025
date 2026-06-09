# Task Completion

1. Run focused relevant tests; for broad/shared changes run the full suite:
   `PYTHONPATH=src python -m pytest tests/ -v`.
2. For scripts that generate artifacts, confirm the exact script path and CLI from
   `README.md` or `--help` before launching a long run.
3. `git status --short`; report only task-related changes; do not revert unrelated dirty work.
4. No lint/format/type-check command exists — preserve the existing file style unless
   explicitly asked to introduce tooling.
5. After bulk or file-generating changes, run `codegraph sync` (or `codegraph status`)
   to refresh the index before relying on codegraph answers.
