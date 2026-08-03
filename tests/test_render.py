"""LaTeX escaping, rendering, compiling, and the golden end-to-end test."""

from __future__ import annotations

import pytest

from suitup.escalate import compile_error, content_fail, overflow, overflow_severity
from suitup.library.compose import draft_from_template
from suitup.models import (
    BulletSource,
    CompileResult,
    Critique,
    CritiqueIssue,
    Draft,
    DraftBullet,
    DraftListing,
    GenerationMode,
    RunStatus,
    Severity,
)
from suitup.render.compiler import compile_tex, engine_available, extract_errors, page_count
from suitup.render.formatter import build_context, escape_latex, render

requires_latex = pytest.mark.skipif(
    not engine_available(), reason="xelatex not installed on this machine"
)


class TestEscapeLatex:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("50% faster", r"50\% faster"),
            ("R&D team", r"R\&D team"),
            ("cost $5", r"cost \$5"),
            ("issue #12", r"issue \#12"),
            ("snake_case", r"snake\_case"),
            ("a{b}c", r"a\{b\}c"),
        ],
    )
    def test_escapes_latex_specials(self, raw, expected):
        assert escape_latex(raw) == expected

    def test_backslash_is_escaped_without_re_escaping_the_result(self):
        # The backslash must be handled first, or the replacements that introduce
        # backslashes would themselves be escaped.
        assert escape_latex(r"a\b") == r"a\textbackslash{}b"

    def test_no_special_survives_unescaped(self):
        # The property that actually matters: every %, &, #, _, {, } in the output must be
        # preceded by a backslash. Removing escaped pairs first means anything left is bare.
        from suitup.render.formatter import _SUBSTITUTIONS

        out = escape_latex("100% of R&D spend on #1 priority_items {scoped} ~ $5 ^ 2")
        residue = out
        # Longest first, so `\textasciitilde{}` is removed before its own `{` and `}`.
        for replacement in sorted(_SUBSTITUTIONS.values(), key=len, reverse=True):
            residue = residue.replace(replacement, "")
        assert not set(residue) & set("%&#_{}$\\"), f"unescaped specials survived in {out!r}"

    def test_escaping_is_idempotent_in_shape(self):
        # Escaping twice must not double-escape into gibberish — a regression here would
        # silently corrupt any bullet that passed through the formatter more than once.
        once = escape_latex("50% of R&D")
        assert once == r"50\% of R\&D"
        assert escape_latex(once) == r"50\textbackslash{}\% of R\textbackslash{}\&D"

    def test_converts_typographic_unicode(self):
        assert escape_latex("it’s — really…") == r"it's --- really\ldots{}"

    def test_converts_symbols_a_model_might_emit(self):
        assert (
            escape_latex("3× faster, 9ms → 6ms, ±10cm")
            == r"3\texttimes{} faster, 9ms $\to$ 6ms, \(\pm\)10cm"
        )

    def test_collapses_whitespace(self):
        assert escape_latex("a  \n  b") == "a b"


class TestSelectiveEscaping:
    """Library content is authored LaTeX; only generated bullets are escaped."""

    def _draft(self, source, text, **kwargs):
        return Draft(
            listings=[
                DraftListing(
                    listing_id="aruw_lead_swe_2025",
                    bullets=[DraftBullet(slot=1, source=source, text=text, **kwargs)],
                )
            ]
        )

    def test_library_latex_passes_through_untouched(self, library):
        # \texttimes{} and 30\% are correct LaTeX already. Escaping them would print
        # backslashes on the page.
        draft = self._draft(BulletSource.ESSENTIAL, r"Scaled 3\texttimes{} at 30\% usage")
        rendered = build_context(draft, library)["experience"][0]["bullets"][0]
        assert rendered == r"Scaled 3\texttimes{} at 30\% usage"

    def test_generated_content_is_escaped(self, library):
        # This is the PRD's edge case: an unescaped % from a model breaks compilation.
        draft = self._draft(
            BulletSource.GENERATED,
            "Improved throughput 30% for R&D",
            generation_mode=GenerationMode.NOVEL,
        )
        rendered = build_context(draft, library)["experience"][0]["bullets"][0]
        assert rendered == r"Improved throughput 30\% for R\&D"


class TestBuildContext:
    def test_routes_listings_into_their_sections(self, library):
        context = build_context(draft_from_template(library, "systems_quant"), library)
        assert context["education"] is not None
        assert len(context["experience"]) == 2
        assert len(context["projects"]) == 2
        assert context["skills"] and context["achievements"]

    def test_skills_rows_carry_their_variant_label(self, library):
        context = build_context(draft_from_template(library, "systems_quant"), library)
        labels = [row["label"] for row in context["skills"]]
        assert "Languages" in labels
        assert "Systems and Concurrency" in labels

    def test_projects_carry_their_stack_line(self, library):
        context = build_context(draft_from_template(library, "systems_quant"), library)
        assert any(p["stack"] for p in context["projects"])

    def test_profile_reaches_the_header(self, library):
        context = build_context(draft_from_template(library, "systems_quant"), library)
        assert context["profile"]["name"] == library.profile.name
        assert context["profile"]["links"]


class TestRender:
    @pytest.mark.parametrize("template_id", ["systems_quant", "ml_robotics", "swe_infra"])
    def test_renders_a_complete_document(self, library, paths, template_id):
        tex = render(draft_from_template(library, template_id), library, paths.latex_dir)
        assert tex.lstrip().startswith("%") or "\\documentclass" in tex
        assert "\\begin{document}" in tex
        assert "\\end{document}" in tex

    def test_no_unrendered_jinja_delimiters_survive(self, library, paths):
        tex = render(draft_from_template(library, "systems_quant"), library, paths.latex_dir)
        for delimiter in ("((*", "*))", "(((", ")))"):
            assert delimiter not in tex, f"{delimiter} left unrendered"

    def test_content_actually_reaches_the_document(self, library, paths):
        tex = render(draft_from_template(library, "systems_quant"), library, paths.latex_dir)
        assert "RealNetworks" in tex
        assert "Monte Carlo" in tex
        assert library.profile.name in tex


class TestCompilerHelpers:
    def test_extracts_error_lines_from_a_log(self):
        log = "noise\n! LaTeX Error: Something bad.\nmore noise\n! Emergency stop.\n"
        errors = extract_errors(log)
        assert "! LaTeX Error: Something bad." in errors
        assert "noise" not in errors

    def test_missing_package_gets_an_install_hint(self):
        # BasicTeX is missing several packages this template needs; the hint saves the
        # next person the same hour.
        log = "! LaTeX Error: File `fullpage.sty' not found.\n"
        assert "preprint" in extract_errors(log)

    def test_falls_back_to_the_log_tail_when_nothing_matches(self):
        assert "final line" in extract_errors("some\nlines\nfinal line")


@requires_latex
class TestCompile:
    def test_compiles_valid_latex_and_counts_pages(self, tmp_path):
        tex = r"\documentclass{article}\begin{document}hello\end{document}"
        result = compile_tex(tex, tmp_path)
        assert result.ok
        assert result.page_count == 1
        assert page_count(tmp_path / "resume.pdf") == 1

    def test_counts_multiple_pages(self, tmp_path):
        tex = r"\documentclass{article}\begin{document}one\newpage two\end{document}"
        assert compile_tex(tex, tmp_path).page_count == 2

    def test_invalid_latex_fails_with_a_readable_error(self, tmp_path):
        result = compile_tex(r"\documentclass{article}\begin{document}\undefinedmacro", tmp_path)
        assert not result.ok
        assert result.page_count is None
        assert result.error_log

    def test_artifacts_survive_a_failure_for_the_escalation(self, tmp_path):
        # The escalation hands the user the .tex; it has to still be there.
        compile_tex(r"\documentclass{article}\begin{document}\undefinedmacro", tmp_path)
        assert (tmp_path / "resume.tex").exists()


@requires_latex
class TestGoldenEndToEnd:
    """The test that proves the deterministic core works, with no LLM anywhere."""

    @pytest.mark.parametrize("template_id", ["systems_quant", "ml_robotics", "swe_infra"])
    def test_every_template_compiles_to_exactly_one_page(
        self, library, paths, tmp_path, template_id
    ):
        draft = draft_from_template(library, template_id)
        tex = render(draft, library, paths.latex_dir)
        result = compile_tex(tex, tmp_path / template_id)

        assert result.ok, f"{template_id} failed to compile:\n{result.error_log}"
        assert result.page_count == 1, (
            f"{template_id} rendered {result.page_count} pages. Each template reconstructs "
            f"a resume that already fit on one page, so this is a regression."
        )

    def test_a_generated_bullet_with_specials_still_compiles(self, library, paths, tmp_path):
        # The PRD's compile-error edge case, exercised end to end: raw model output with
        # LaTeX specials must not be able to break the build.
        draft = draft_from_template(library, "systems_quant")
        draft.listings[1].bullets.append(
            DraftBullet(
                slot=99,
                source=BulletSource.GENERATED,
                text="Cut cost 50% for R&D_team #1 {scope} ~ $5 ^ 2 \\oops",
                generation_mode=GenerationMode.NOVEL,
            )
        )
        result = compile_tex(render(draft, library, paths.latex_dir), tmp_path)
        assert result.ok, f"escaping failed to protect the build:\n{result.error_log}"


class TestEscalations:
    def test_overflow_severity_distinguishes_wording_from_selection(self):
        # The PRD wants these read differently: two pages is usually wording, more is a
        # selection problem.
        assert "wording" in overflow_severity(2)
        assert "selection problem" in overflow_severity(4)

    def test_overflow_writes_a_status_and_a_note(self, tmp_path):
        result = CompileResult(ok=True, page_count=2, tex_path=str(tmp_path / "r.tex"))
        run = overflow(tmp_path, result, unaddressed=["Kubernetes experience"])
        assert run.status is RunStatus.OVERFLOW
        assert (tmp_path / "STATUS").read_text().strip() == "overflow"
        body = (tmp_path / "ESCALATION.md").read_text()
        assert "2 pages" in body
        assert "Kubernetes experience" in body

    def test_compile_error_is_distinct_from_overflow(self, tmp_path):
        # Different failures need different fixes: cut content vs edit content.
        result = CompileResult(ok=False, tex_path=str(tmp_path / "r.tex"), error_log="! boom")
        run = compile_error(tmp_path, result)
        assert run.status is RunStatus.COMPILE_ERROR
        body = (tmp_path / "ESCALATION.md").read_text()
        assert "not an\noverflow" in body or "not an overflow" in body.replace("\n", " ")
        assert "! boom" in body

    def test_content_fail_lists_the_outstanding_issues(self, tmp_path):
        critique = Critique(
            passed=False,
            issues=[
                CritiqueIssue(
                    category="groundedness",
                    severity=Severity.MAJOR,
                    detail="claims a 40% gain the sources do not support",
                    listing_id="aruw_lead_swe_2025",
                    slot=3,
                )
            ],
        )
        run = content_fail(tmp_path, critique, rounds=3)
        assert run.status is RunStatus.CONTENT_FAIL
        body = (tmp_path / "ESCALATION.md").read_text()
        assert "groundedness" in body
        assert "aruw_lead_swe_2025" in body
        assert "40% gain" in body

    def test_low_coverage_warning_propagates(self, tmp_path):
        critique = Critique(passed=False, low_coverage_warning=True)
        assert content_fail(tmp_path, critique, rounds=3).low_coverage_warning
