"""The three ways a run stops without producing an approved resume.

Escalation is an output, not an exception path. Each exit writes a run folder the user can
act on directly: a `STATUS` file naming the failure, an `ESCALATION.md` explaining it in
plain language, and whatever artifacts are relevant.

The PRD insists overflow and compile error stay visibly distinct, because the fixes differ
— overflow means cut content, a compile error means edit content. Collapsing them into one
"it didn't work" would hide that.

Nothing here calls a model. The severity note on an overflow is computed from the page
count, deliberately: asking an LLM how bad the overflow is would be a guess about a number
already known exactly.
"""

from __future__ import annotations

from pathlib import Path

from suitup.models import CompileResult, Critique, RunResult, RunStatus


def _write(run_dir: Path, status: RunStatus, body: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STATUS").write_text(status.value + "\n")
    (run_dir / "ESCALATION.md").write_text(body)


def overflow_severity(page_count: int) -> str:
    """Plain-language reading of how far over one page a draft ran.

    Computed from the page count in code, never from a model call.
    """
    if page_count <= 1:
        return "Fits on one page."
    if page_count == 2:
        return (
            "Just over one page. This usually means wording is slightly long rather than "
            "that too much was selected — tightening two or three bullets is often enough. "
            "Check the last listing on the page first."
        )
    return (
        f"Meaningfully over at {page_count} pages. This is a selection problem, not a "
        f"wording one: too many listings or bullets were included. Revisit the Listing "
        f"Selector's choices and the count caps in models.yaml before rerunning."
    )


def content_fail(
    run_dir: Path,
    critique: Critique,
    rounds: int,
    draft_path: Path | None = None,
) -> RunResult:
    """The Critic never cleared, and the drafting loop ran out of rounds."""
    issues = (
        "\n".join(
            f"- **{issue.severity.value}** ({issue.category})"
            + (
                f" in `{issue.listing_id}`" + (f" slot {issue.slot}" if issue.slot else "")
                if issue.listing_id
                else ""
            )
            + f": {issue.detail}"
            for issue in critique.issues
        )
        or "- (the critic reported no itemised issues)"
    )

    body = f"""# Escalation: content did not pass review

The draft was revised {rounds} time(s) and the critic still reports unresolved issues.
Stopping here rather than burning more rounds on a draft that is not converging.

## Outstanding issues

{issues}

## What to do

Content problems are usually a library problem, not a drafting one. In order of likelihood:

1. A **groundedness** issue means the draft claimed something the library does not support.
   That content should not ship. Check whether the underlying variant is actually missing.
2. A **repetition** or **redundant emphasis** issue that keeps recurring usually means two
   variants genuinely overlap. Fix it in `library/listings/` and the problem stops coming
   back — `pytest tests/test_library_integrity.py` will confirm.
3. **Coverage** issues mean the posting wants something you have not written a bullet for
   yet. That is a real gap, and worth authoring a variant for rather than working around.

{"The draft is at `" + draft_path.name + "` in this folder." if draft_path else ""}
"""
    _write(run_dir, RunStatus.CONTENT_FAIL, body)
    return RunResult(
        status=RunStatus.CONTENT_FAIL,
        run_dir=str(run_dir),
        low_coverage_warning=critique.low_coverage_warning,
        message=f"critic did not pass after {rounds} round(s)",
    )


def compile_error(run_dir: Path, result: CompileResult) -> RunResult:
    """LaTeX failed. A different problem from overflow, and it gets a different exit."""
    body = f"""# Escalation: LaTeX failed to compile

The document could not be built, so there is no PDF and no page count. **This is not an
overflow** — nothing needs to be cut. Something in the document is malformed.

## Errors

```
{result.error_log or "(no error output captured)"}
```

## What to do

1. The most common cause is an unescaped special character (`%`, `&`, `_`, `#`, `$`) in a
   **generated** bullet. Library content is authored LaTeX and passes through untouched;
   only generated text is escaped, so a bug there shows up exactly like this.
2. A missing `.sty` file means a LaTeX package is not installed rather than anything being
   wrong with the resume. The error above says which one.
3. `{Path(result.tex_path).name}` and the full `.log` are in this folder. Compiling by hand
   in this directory reproduces the failure exactly:

   ```
   cd {run_dir}
   xelatex -interaction=nonstopmode {Path(result.tex_path).name}
   ```
"""
    _write(run_dir, RunStatus.COMPILE_ERROR, body)
    return RunResult(
        status=RunStatus.COMPILE_ERROR,
        run_dir=str(run_dir),
        message="latex compilation failed",
    )


def overflow(
    run_dir: Path,
    result: CompileResult,
    unaddressed: list[str] | None = None,
    low_coverage: bool = False,
) -> RunResult:
    """It compiled, but not onto one page. No automatic trimming — that is the user's call."""
    pages = result.page_count or 0
    unaddressed_block = ""
    if unaddressed:
        listed = "\n".join(f"- {req}" for req in unaddressed)
        unaddressed_block = (
            f"\n## Requirements with no matching listing\n\n{listed}\n\n"
            f"Worth knowing before you cut: these are already uncovered.\n"
        )

    body = f"""# Escalation: resume is {pages} pages

The document compiled cleanly — this is purely a length problem.

## Severity

{overflow_severity(pages)}

## What to do

No automatic trimming was attempted, deliberately. Cutting content is a judgment about
what matters for this posting, and the MVP hands it back rather than guessing.

- `{Path(result.tex_path).name}` and `draft.yaml` are in this folder. Editing the `.tex`
  and recompiling is the fastest way to see the effect of a cut.
- To make the change stick for future runs, lower `caps` in `models.yaml` or drop a slot
  from the template in `library/templates/`.
- Recompile with:

  ```
  cd {run_dir}
  xelatex -interaction=nonstopmode {Path(result.tex_path).name}
  ```
{unaddressed_block}"""
    _write(run_dir, RunStatus.OVERFLOW, body)
    return RunResult(
        status=RunStatus.OVERFLOW,
        run_dir=str(run_dir),
        page_count=pages,
        unaddressed_requirements=unaddressed or [],
        low_coverage_warning=low_coverage,
        message=f"compiled to {pages} pages",
    )
