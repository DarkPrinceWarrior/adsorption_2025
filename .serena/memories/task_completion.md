# Task Completion Checklist

Before finishing code changes:
1. Run the relevant focused tests when possible.
2. For broad or shared changes, run the full test suite:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```
3. If changing scripts that generate artifacts, verify the specific script path and CLI invocation from `README.md` or `--help` before running a long job.
4. Check `git status --short` and only report changes related to the task.
5. Do not revert unrelated dirty work. During onboarding, `.serena/` appeared as untracked after project activation.

No separate lint/format command is documented in this repository. If formatting is needed, preserve the style already present in the file unless the user explicitly asks to introduce a formatter.