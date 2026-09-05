"""Compile LaTeX and count pages. The only authority on whether a resume fits.

The PRD is deliberate that there is no length estimation anywhere else in the MVP: word
counts and character budgets are guesses, and this is the measurement. Phase 1 confirmed
why — one template fit at 20 bullets while another overflowed at 20, because its bullets
ran longer.

Single pass, no retry loop. A failure here escalates to the user with the `.tex` in hand.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

from suitup.models import CompileResult

# xelatex over pdflatex: it is what the source resume was built with, and it handles the
# unicode that reaches the page through generated bullets without extra packages.
ENGINE = "xelatex"
TIMEOUT_SECONDS = 120

# LaTeX errors start with "! ". Everything else in the log is noise for our purposes.
_ERROR_LINE = re.compile(r"^!.*$", re.MULTILINE)
_MISSING_FILE = re.compile(r"! LaTeX Error: File `([^']+)' not found")


class CompilerNotFound(RuntimeError):
    """The LaTeX engine is not installed."""


def engine_available(engine: str = ENGINE) -> bool:
    return shutil.which(engine) is not None


def extract_errors(log: str, limit: int = 8) -> str:
    """Pull the useful lines out of a LaTeX log.

    A xelatex log is thousands of lines; the escalation should hand the user the handful
    that identify the problem, with the full log still on disk beside it.
    """
    errors = _ERROR_LINE.findall(log)

    missing = _MISSING_FILE.search(log)
    if missing:
        package = missing.group(1)
        errors.append(
            f"[hint] {package} is not installed. On BasicTeX: "
            f"tlmgr --usermode install --repository "
            f"https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final "
            f"<package>  (note fullpage.sty ships inside `preprint`)"
        )

    if not errors:
        return "\n".join(log.splitlines()[-limit:])
    return "\n".join(errors[:limit])


def page_count(pdf_path: Path) -> int:
    """Pages in the produced PDF.

    Uses pypdf rather than `pdfinfo`, which needs Poppler and is not installed here.
    """
    return len(PdfReader(pdf_path).pages)


def compile_tex(
    tex: str,
    workdir: Path,
    stem: str = "resume",
    engine: str = ENGINE,
) -> CompileResult:
    """Write `tex` into `workdir` and compile it once.

    Artifacts stay in `workdir` on both success and failure — the `.tex` and `.log` are
    exactly what an escalation hands over.
    """
    if not engine_available(engine):
        raise CompilerNotFound(
            f"{engine} not found on PATH. Install MacTeX or BasicTeX, then re-run."
        )

    workdir.mkdir(parents=True, exist_ok=True)
    tex_path = workdir / f"{stem}.tex"
    tex_path.write_text(tex)

    try:
        process = subprocess.run(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",  # a generated bullet must never be able to run a command
                f"{stem}.tex",
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        log = process.stdout + process.stderr
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        # Almost always a LaTeX error that opened an interactive prompt despite
        # nonstopmode. Treat as a compile failure rather than hanging the run.
        return CompileResult(
            ok=False,
            tex_path=str(tex_path),
            error_log=f"{engine} timed out after {TIMEOUT_SECONDS}s",
        )

    log_path = workdir / f"{stem}.log"
    if log_path.exists():
        log = log_path.read_text(errors="replace")

    pdf_path = workdir / f"{stem}.pdf"
    if returncode != 0 or not pdf_path.exists():
        return CompileResult(ok=False, tex_path=str(tex_path), error_log=extract_errors(log))

    return CompileResult(
        ok=True,
        page_count=page_count(pdf_path),
        tex_path=str(tex_path),
        pdf_path=str(pdf_path),
    )
