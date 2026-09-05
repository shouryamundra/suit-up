"""Turn a draft into LaTeX.

Two things here are easy to get wrong.

**Escaping is selective, not universal.** Library variants are hand-authored LaTeX: they
contain `\\texttimes{}`, `$\\to$`, `\\textbf{...}`, and `30\\%`. Escaping those would
render backslashes on the page and break the math. So library content passes through
verbatim, and only *generated* bullets — untrusted model output — are escaped. That is
also the PRD's edge case about a malformed generated bullet causing a compile error, and
this is where it is prevented.

**Jinja needs LaTeX-safe delimiters.** `{{ }}` and `{% %}` collide with LaTeX braces, so
the environment uses `((* *))`, `((( )))`, and `((= =))`. The template on disk is written
against these; changing them here silently breaks it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from suitup.library.loader import Library, Profile
from suitup.models import BulletSource, Draft, ListingType

# LaTeX specials, and the unicode a model routinely emits raw. Both maps are merged into
# one regex below and applied in a SINGLE pass. Chained str.replace calls are wrong here:
# escaping `{` after `\` rewrites the braces of the `\textbackslash{}` the previous step
# just produced, yielding `\textbackslash\{\}`. One pass never re-examines its own output.
_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]

_UNICODE = {
    "—": "---",
    "–": "--",
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "…": r"\ldots{}",
    "×": r"\texttimes{}",
    "→": r"$\to$",
    "±": r"\(\pm\)",
    " ": " ",
}


_SUBSTITUTIONS = {**dict(_ESCAPES), **_UNICODE}
_SUBSTITUTION_RE = re.compile("|".join(re.escape(k) for k in _SUBSTITUTIONS))


def escape_latex(text: str) -> str:
    """Make arbitrary text safe to drop into a LaTeX document.

    Applied to generated bullets only. Never apply it to library content, which is
    authored LaTeX and would be mangled.
    """
    escaped = _SUBSTITUTION_RE.sub(lambda m: _SUBSTITUTIONS[m.group()], text)
    return re.sub(r"\s+", " ", escaped).strip()


def make_environment(latex_dir: Path) -> Environment:
    """Jinja configured so bare braces stay LaTeX's."""
    return Environment(
        loader=FileSystemLoader(latex_dir),
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((=",
        comment_end_string="=))",
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
        undefined=StrictUndefined,
    )


def _bullet_text(bullet) -> str:
    """Library content verbatim; generated content escaped."""
    if bullet.source is BulletSource.GENERATED:
        return escape_latex(bullet.text)
    return " ".join(bullet.text.split())


def build_context(draft: Draft, library: Library, profile: Profile | None = None) -> dict[str, Any]:
    """Group the draft's ordered listings into the sections the template renders.

    The draft carries display order; this only sorts listings into their sections, keeping
    the draft's relative order inside each.
    """
    context: dict[str, Any] = {
        "profile": (profile or library.profile).model_dump(),
        "education": None,
        "experience": [],
        "projects": [],
        "skills": [],
        "achievements": [],
    }

    for draft_listing in draft.listings:
        listing = library.listing(draft_listing.listing_id)
        texts = [_bullet_text(b) for b in draft_listing.bullets]
        if not texts:
            continue

        if listing.type is ListingType.EDUCATION:
            context["education"] = {
                "org": listing.context.org,
                "location": listing.context.location or "",
                "title": listing.title,
                "dates": listing.context.dates,
                "bullets": texts,
            }
        elif listing.type is ListingType.EXPERIENCE:
            context["experience"].append(
                {
                    "org": listing.context.org,
                    "location": listing.context.location or "",
                    "title": listing.title,
                    "dates": listing.context.dates,
                    "bullets": texts,
                }
            )
        elif listing.type is ListingType.PROJECT:
            context["projects"].append(
                {
                    "title": listing.title,
                    "stack": listing.context.stack if listing.context else None,
                    "dates": listing.context.dates if listing.context else "",
                    "bullets": texts,
                }
            )
        elif listing.type is ListingType.SKILLS:
            context["skills"] = [
                {"label": _label(listing, b), "text": text}
                for b, text in zip(draft_listing.bullets, texts)
            ]
        elif listing.type is ListingType.ACHIEVEMENTS:
            context["achievements"] = texts

    return context


def _label(listing, draft_bullet) -> str:
    """The category heading for a skills row: variant label wins, then slot label."""
    bullet = listing.bullet(draft_bullet.slot)
    if draft_bullet.variant_id:
        for variant in bullet.variants:
            if variant.variant_id == draft_bullet.variant_id and variant.label:
                return variant.label
    return bullet.label or ""


def render(
    draft: Draft,
    library: Library,
    latex_dir: Path,
    template_name: str = "resume.tex.j2",
    profile: Profile | None = None,
) -> str:
    """Render the draft to a complete LaTeX document."""
    env = make_environment(latex_dir)
    template = env.get_template(template_name)
    return template.render(**build_context(draft, library, profile))
