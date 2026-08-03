"""Suit-Up command line interface."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from suitup.config import ConfigError, load_config, parse_override, set_role_binding
from suitup.llm.provider import StructuredLLM
from suitup.llm.router import Router

app = typer.Typer(
    name="suitup",
    help="Tailor a one-page resume from a library of pre-approved bullet variants.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help="Inspect and change which model runs which pipeline role.",
    # Bare `suitup models` prints the routing table via the callback rather than help.
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


ModelOpt = typer.Option(
    None,
    "--model",
    "-m",
    help="Override one role, e.g. drafter=ollama:llama3.1. Repeatable.",
    metavar="ROLE=PROVIDER:MODEL",
)
ProfileOpt = typer.Option(
    None,
    "--profile",
    "-p",
    help="Apply a named routing from models.yaml, e.g. local or mistral-only.",
)


def _load(overrides: list[str] | None = None, profile: str | None = None):
    try:
        return load_config(overrides=overrides, profile=profile)
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@models_app.callback()
def models_default(
    ctx: typer.Context,
    check: bool = typer.Option(
        False, "--check", help="Ping each provider and verify the routed model is offered."
    ),
    model: list[str] = ModelOpt,
    profile: str = ProfileOpt,
) -> None:
    """Show the active role -> provider:model routing."""
    if ctx.invoked_subcommand is not None:
        return
    _show_routing(check=check, model=model, profile=profile)


def _show_routing(check: bool, model: list[str] | None, profile: str | None) -> None:
    config = _load(model, profile)
    router = Router(config)

    title = "Role routing" + (f"  (profile: {profile})" if profile else "")
    table = Table(title=title, header_style="bold")
    table.add_column("Role")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Temp", justify="right")
    table.add_column("Key")
    if check:
        table.add_column("Status")

    exit_code = 0
    for row in router.describe():
        key_cell = "[green]set[/green]" if row.has_key else "[red]missing[/red]"
        cells = [row.role, row.provider, row.model, f"{row.temperature:g}", key_cell]

        if not check:
            table.add_row(*cells)
            continue

        if not row.has_key:
            table.add_row(*cells, "[red]skipped, no key[/red]")
            exit_code = 1
            continue

        ok, detail = router.for_role(row.role).ping()
        table.add_row(*cells, f"[green]{detail}[/green]" if ok else f"[red]{detail}[/red]")
        if not ok:
            exit_code = 1

    console.print(table)

    drafter, critic = config.roles["drafter"], config.roles["critic"]
    if drafter == critic:
        console.print(
            f"\n[yellow]Drafter and critic share {drafter} — the critique is not blind.[/yellow]"
        )
    else:
        console.print(
            f"\nDrafter [bold]{drafter}[/bold] and critic [bold]{critic}[/bold] differ, "
            f"so the critique is blind."
        )

    if check and exit_code:
        console.print("\n[yellow]Some roles are not ready. Fix the rows marked in red.[/yellow]")
    raise typer.Exit(code=exit_code)


@models_app.command("set")
def models_set(
    assignment: str = typer.Argument(
        ..., help="role=provider:model, e.g. drafter=openrouter:anthropic/claude-sonnet-4.5"
    ),
) -> None:
    """Permanently route a role to a provider and model, editing models.yaml in place."""
    try:
        role, binding = parse_override(assignment)
        path = set_role_binding(role, binding)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"[green]{role}[/green] -> [bold]{binding}[/bold]  (written to {path.name})")

    # Re-validate: the edit may have collided with the blind-critique rule, and it is
    # better to say so now than to fail on the next run.
    try:
        load_config()
    except ConfigError as exc:
        console.print(f"\n[yellow]Warning:[/yellow] {exc}")
        raise typer.Exit(code=1) from exc


@models_app.command("available")
def models_available(
    provider: str = typer.Argument(..., help="Provider name from models.yaml."),
    grep: str = typer.Option(None, "--grep", "-g", help="Only show ids containing this substring."),
    limit: int = typer.Option(60, "--limit", "-n", help="Maximum ids to print."),
) -> None:
    """List the model ids a provider actually offers, so names are checked not guessed."""
    config = _load()
    if provider not in config.providers:
        console.print(
            f"[red]Unknown provider {provider!r}.[/red] "
            f"Defined: {', '.join(sorted(config.providers))}"
        )
        raise typer.Exit(code=2)

    spec = config.providers[provider]
    if not spec.has_key():
        console.print(f"[red]{spec.api_key_env} is not set; cannot query {provider}.[/red]")
        raise typer.Exit(code=2)

    probe = StructuredLLM(role="probe", provider=spec, model="unused")
    try:
        ids = sorted(m.id for m in probe._client.models.list().data)
    except Exception as exc:  # noqa: BLE001 - the raw failure is the useful message
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if grep:
        ids = [i for i in ids if grep.lower() in i.lower()]
    if not ids:
        console.print("[yellow]No models matched.[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]{provider}[/bold]: {len(ids)} model(s)" + (f" matching {grep!r}" if grep else "")
    )
    for model_id in ids[:limit]:
        console.print(f"  {model_id}")
    if len(ids) > limit:
        console.print(f"  [dim]... and {len(ids) - limit} more (raise --limit)[/dim]")


app.add_typer(models_app, name="models")


@app.command("run")
def run_cmd(
    posting: typer.FileText = typer.Argument(..., help="Path to the job posting, plain text."),
    model: list[str] = ModelOpt,
    profile: str = ProfileOpt,
) -> None:
    """Tailor a resume for a job posting. (Wired up in phase 4.)"""
    _load(model, profile)
    console.print("[yellow]`run` is not wired up yet — the pipeline lands in phase 4.[/yellow]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
