# Suit-Up MVP — Build Plan

## Context

`PRD.md` describes a resume-tailoring agent: given a job posting, assemble a one-page
resume from a personal library of pre-approved bullet variants, generating new content
only for real gaps. The repo is currently empty — `PRD.md` and a two-line `README.md`.

This plan turns that PRD into a buildable MVP. It differs from the PRD in four ways,
each a decision made during planning:

1. **Trimmed agent spine.** 13 agents → 6 LLM nodes + 4 code nodes. Template Matcher
   folds into Listing Selector; Sequencer folds into Drafter; Ingester becomes
   semi-manual. Fewer moving parts, fewer calls per run, same output contract.
2. **No importer CLI.** The library bootstrap is a one-time job done by hand (by me,
   from the user's `.tex` files) rather than a shipped feature. All the resumes share
   one LaTeX template, so this is grunt work, not a parsing problem worth automating.
3. **Provider swappability is a headline feature, not an implementation detail.** Any
   node can run on Mistral, Gemini, or Ollama by editing one line of config.
4. **Deterministic core is built and verified before any LLM node exists.**

Intended outcome: `suitup run posting.txt` produces a verified one-page PDF plus a
structured YAML draft, or a clearly-labelled escalation explaining why it could not.

---

## Decisions locked during planning

| Question | Decision |
|---|---|
| Scope | Trimmed spine (see §2) |
| Interface | CLI only |
| Providers | Mistral + Gemini + Ollama, all swappable per node |
| Default routing | Gemini drafts, Mistral critiques, Ollama for cheap structured work |
| Company-tailored resumes | Seed `configurations.json` entries + tagged variants — **not** templates |
| Library bootstrap | Manual, done by me from user's `.tex` files |
| Variant merging | Auto-group by listing + text overlap, flag near-duplicates for review |
| Count caps | Derived from the user's actual one-page resumes, written to config |
| Timeline format | Phases with session estimates, no calendar dates |

**Security note:** a Mistral API key was pasted into the planning conversation and must
be rotated before use. `.env` is gitignored in Phase 0, commit 1. No key is ever written
to a tracked file.

---

## 1. Stack

- **Python 3.11** + **poetry** (both already installed).
- **LangGraph** for orchestration — per PRD. Kept as a *thin wiring layer*: every agent
  is a plain `(State) -> dict` function importable and testable without the graph.
- **Pydantic v2** for every schema. Structured LLM output validates against these or the
  call retries.
- **Typer** for the CLI, **Jinja2** for LaTeX rendering, **pypdf** for page count
  (`pdfinfo` is not installed; pypdf avoids a Poppler dependency).
- **xelatex** (present at `/Library/TeX/texbin/xelatex`) for compilation.
- **pytest** for tests.

---

## 2. The trimmed spine

```
posting.txt
  → JD Analyst        (LLM)   → requirements.json
  → Listing Selector  (LLM)   → chosen listings + unaddressed_requirements
        (template matching folded in: templates are scored lexically in code,
         the LLM only decides deltas from the best-scoring template)
  → Retriever         (CODE)  → scored candidates per flexible slot
  → Bullet Selector   (LLM)   → selections.json + gap list
  → Drafter           (LLM)   → draft + ordering + generated bullets
        (Sequencer folded in — ordering is decided in the same pass that
         polishes for coherence, since both are holistic judgments)
  → Critic            (LLM)   → critique.json
  → Supervisor        (CODE)  → loop back to Drafter (max 3) or proceed
  → Formatter         (CODE)  → .tex
  → Compiler          (CODE)  → PDF + page count
  → Config Checker    (CODE)  → approved | needs_review
```

**Cut from the PRD MVP, with reasons:**

- *Template Matcher as its own LLM node* — lexical template scoring in code is enough to
  pick a starting point; the judgment that needs an LLM is the delta decision, which
  Listing Selector already makes.
- *Sequencer as its own node* — an extra call and an extra state hop to decide something
  the Drafter must already reason about.
- *Ingester as an automated node* — needs real run data to be useful. MVP writes
  `pending_variants.yaml` and archives the run; promoting variants stays a human edit.
- *Automated overflow condensing, vector search, length pre-estimation* — already PRD
  non-goals.

---

## 3. Provider abstraction (the headline piece)

All three providers expose **OpenAI-compatible HTTP endpoints**, so this is one adapter,
not three SDK integrations:

| Provider | Base URL | Structured-output tier |
|---|---|---|
| Mistral | `https://api.mistral.ai/v1` | native json_schema |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | native json_schema |
| Ollama | `http://localhost:11434/v1` | json_object only (weaker) |

**`suitup/llm/provider.py`** — one `StructuredLLM` class built on the `openai` SDK,
parameterized by `(base_url, api_key_env, model, schema_tier)`. Single public method:

```python
def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel
```

It requests the strongest structured mode the provider supports, validates the response
against the Pydantic model, and on `ValidationError` retries up to twice with the
validation error appended to the prompt. Third failure raises `StructuredOutputError`,
which the Supervisor turns into a labelled escalation rather than a stack trace. This
retry wrapper is the reliability backbone — Ollama's `json_object` tier in particular
will need it.

**`models.yaml`** — the swappability surface. One line per node:

```yaml
defaults:
  providers:
    mistral: { base_url: "https://api.mistral.ai/v1",  api_key_env: MISTRAL_API_KEY, tier: schema }
    gemini:  { base_url: "https://generativelanguage.googleapis.com/v1beta/openai/", api_key_env: GEMINI_API_KEY, tier: schema }
    ollama:  { base_url: "http://localhost:11434/v1",  api_key_env: null, tier: json_object }

roles:
  jd_analyst:       { provider: mistral, model: mistral-small-latest }
  listing_selector: { provider: mistral, model: mistral-small-latest }
  bullet_selector:  { provider: mistral, model: mistral-small-latest }
  drafter:          { provider: gemini,  model: gemini-3.6 }       # exact id confirmed at build time
  critic:           { provider: mistral, model: mistral-large-latest }
  tagger:           { provider: ollama,  model: llama3.1 }
```

Agents receive an injected `StructuredLLM` and never name a provider. Swapping a node is
a one-line YAML edit, or at runtime:

```
suitup run posting.txt --model drafter=ollama:llama3.1
suitup models                 # print the active role → provider:model table
suitup models --check         # ping every configured provider, report reachability
```

Drafter and Critic are asserted at config-load time to be **different models**, per the
PRD's blind-critique requirement. Cross-vendor by default (Gemini drafts, Mistral
critiques) makes that requirement real rather than nominal.

---

## 4. Repository layout

```
suitup/
  cli.py                 # typer: run, approve, models, library check
  config.py              # pydantic-settings; caps, paths, models.yaml loader
  models.py              # Listing, Bullet, Variant, Requirement, Selection, Draft, RunResult
  library/
    loader.py            # YAML → objects, validation
    index.py             # flattened index.json, mtime-based rebuild
    configs.py           # composition keys, configurations.json read/write
  llm/
    provider.py          # StructuredLLM + retry wrapper
    router.py            # models.yaml → role-bound provider instances
  agents/
    jd_analyst.py  listing_selector.py  retriever.py
    bullet_selector.py   drafter.py     critic.py
  render/
    formatter.py         # draft → .tex (Jinja2 + user's preamble), LaTeX escaping
    compiler.py          # xelatex subprocess, pypdf page count
  graph.py               # LangGraph wiring + Supervisor routing
  escalate.py            # the three labelled exits

library/                 # data, git-versioned, human-editable
  listings/*.yaml   templates/*.yaml
  configurations.json    pending_variants.yaml    index.json (generated)

templates/latex/resume.tex.j2    # derived from the user's own preamble
runs/<date>-<slug>/              # tex, pdf, draft.yaml, critique.json, STATUS
tests/
```

---

## 5. Design points that need care

**Composition key** (`library/configs.py`) — deterministic string
`listing_id|slot1:essential|slot2:v1_cv_metrics|...`, slots sorted numerically. Whole-resume
key is the ordered list of listing keys. Exactness is the entire value of the cache, so
this gets tested against reordering, missing slots, and unicode.

**Retriever scoring** (`agents/retriever.py`, pure code, no LLM) — for each flexible
variant, score `Σ requirement.weight × match(requirement, variant)`. Exact-phrase hit on
normalized text (lowercased, punctuation stripped, whitespace collapsed) scores full
weight; token-subset hit scores partial. Essential bullets bypass scoring entirely and
are always included. Every score is written to the run folder — this is the PRD's
auditability requirement, and everything downstream depends on it being right, so it
gets the heaviest unit-test coverage in the project.

**LaTeX escaping** (`render/formatter.py`) — `% & _ # $ ~ ^ \` and friends escaped in all
bullet text before rendering. PRD edge case: a malformed generated bullet causing a
compile error is a *different failure* from overflow. Deterministic and unit-tested.

**Three labelled exits** (`escalate.py`) — escalation is a first-class output, not an
exception path. Each writes the run folder with a `STATUS` file and a human-readable
`ESCALATION.md`:
- `CONTENT_FAIL` — Critic exhausted 3 rounds without strict improvement.
- `COMPILE_ERROR` — xelatex failed; hands over `.tex` + the error log.
- `OVERFLOW` — compiled but >1 page; hands over `.tex` + `draft.yaml` + page count +
  severity note (computed from page count in code, **not** another LLM call).

**Supervisor** is control logic, not a model call. Loop back to Drafter only if
iterations remain *and* critique severity strictly improved; otherwise `CONTENT_FAIL`.

---

## 6. Testing strategy

- **Pure-code nodes** — retriever scoring, composition keys, LaTeX escaping, index
  rebuild, page counting. Real unit tests, written first (TDD). This is where correctness
  actually lives.
- **LLM nodes** — tested against a `FakeProvider` returning canned schema-valid
  responses. Asserts the node's contract (state in → state out), never LLM quality.
- **Golden end-to-end** — fake provider + **real xelatex compile**, asserting a one-page
  PDF. This is the test that proves the system works.
- **Provider conformance** — one live-network test per provider, marked and skipped by
  default, run manually when adding or changing a model.

---

## 7. Phases

Estimates are working sessions, not calendar time.

**Phase 0 — Scaffold (1 session)**
Poetry project, `.gitignore` with `.env` first, Pydantic schemas in `models.py`, config
loader, `models.yaml`, CLI skeleton with `suitup models`.
*Done when:* `suitup models --check` reports reachability for all three providers.

**Phase 1 — Library bootstrap (1–2 sessions, gated on user handing over `.tex` files)**
I manually convert the 4–5 resumes into `library/listings/*.yaml`: group bullets into
listings, auto-group near-identical bullets across resumes into variants of one slot,
flag close-but-not-identical pairs for the user to adjudicate, tag keywords, and mark
`essential` vs `flexible` (user reviews these — the PRD says this call is human, not
inferred). Extract 3–4 career-direction `templates/`. Write company-tailored resumes into
`configurations.json` as pre-approved compositions with provenance. Extract the shared
LaTeX preamble into `templates/latex/resume.tex.j2`. Measure the observed listing/bullet
counts and write them into config as the caps.
*Done when:* the library round-trips and the user confirms the essential/flexible calls.

**Phase 2 — Deterministic core (1–2 sessions)**
Loader, index, composition keys, Retriever scoring, Formatter, Compiler, escaping,
escalation exits. Full test coverage. **Zero LLM code.**
*Done when:* a template's listing set renders and compiles to a verified one-page PDF,
and the golden end-to-end test passes. This is the riskiest deterministic path and it
validates the entire data model before a single token is spent.

**Phase 3 — Provider layer + structured nodes (1–2 sessions)**
`StructuredLLM` + retry wrapper + router. JD Analyst, Listing Selector, Bullet Selector.
*Done when:* `posting.txt` → a validated selection set, and every node's provider can be
swapped from the CLI without touching code.

**Phase 4 — Drafter, Critic, graph (1–2 sessions)**
Drafter with ordering + gap handling (synthesis from existing slot variants attempted
before novel generation, tagged `generation_mode`). Critic with its own rubric and
persona. Supervisor loop. LangGraph wiring.
*Done when:* a full run completes end-to-end on a real posting and produces a PDF.

**Phase 5 — Cache, archive, polish (1 session)**
Config Checker, run archive, output YAML per PRD §11, `suitup approve`,
`pending_variants.yaml` proposals.
*Done when:* rerunning the same posting hits the cache and skips review.

**Phase 6 — Shakedown (ongoing, not a milestone)**
Real postings. Tune caps and the low-coverage threshold. Answer the PRD's open questions
with data: is the count cap good enough, is Ollama adequate for narrow nodes, how often
do runs actually overflow.

Total to a working MVP: **6–9 sessions**, with Phase 1 blocked on the resume files.

---

## 8. Verification

- `pytest` — unit + golden end-to-end (includes a real xelatex compile).
- `suitup models --check` — all three providers reachable, drafter ≠ critic enforced.
- `suitup run tests/fixtures/posting_ml.txt` — inspect `runs/<date>-*/`: `draft.yaml`
  marks each bullet `essential`/`selected`/`generated`, `page_count` is 1, scores are
  auditable in the run folder.
- Swap check: `suitup run ... --model drafter=ollama:llama3.1` completes without code
  changes — proves the abstraction.
- Cache check: rerun the same posting, confirm Config Checker reports an exact match and
  skips review.
- Escalation check: force each of the three exits (bad LaTeX in a bullet, an
  over-stuffed template, a Critic that never improves) and confirm each is distinctly
  labelled.

---

## 9. Open questions carried forward

Deliberately unanswered until there is run data — these are the PRD's own open questions
and MVP is not the place to guess at them:

- Count cap per listing type vs one global cap.
- The numeric threshold for the low-coverage warning.
- Whether Ollama is good enough for the narrow structured nodes.
- Whether automated condensing is worth building, and if so what it should cut.

## 10. First action on approval

Copy this plan to `docs/BUILD_PLAN.md` and commit it alongside `PRD.md`, then start
Phase 0. Phase 1 begins as soon as the `.tex` resumes are handed over — feel free to drop
them in at any point; the work reorders around whenever they arrive.
