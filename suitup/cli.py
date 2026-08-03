"""Suit-Up command line interface."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from suitup.config import ConfigError, load_config
from suitup.llm.router import Router

app = typer.Typer(
    name="suitup",
    help="Tailor a one-page resume from a library of pre-approved bullet variants.",
    no_args_is_help=True,
    add_completion=False,
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


@app.command("models")
def models_cmd(
    check: bool = typer.Option(
        False, "--check", help="Ping each provider and verify the routed model is offered."
    ),
    model: list[str] = ModelOpt,
    profile: str = ProfileOpt,
) -> None:
    """Show the active role -> provider:model routing."""
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
