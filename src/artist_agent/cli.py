#!/usr/bin/env python3
"""Artist Agent CLI -- one command, one image: pick a direction, compose a prompt,
generate via Draw Things, self-critique, write a portable item note, fold the
outcome back into the interests file."""
from datetime import datetime
from typing import Annotated

import typer
from local_first_common.cli import (
    dry_run_option,
    no_llm_option,
    resolve_dry_run,
    resolve_provider,
)
from local_first_common.config import get_setting
from local_first_common.providers import PROVIDERS
from local_first_common.tracking import register_tool, timed_run

from .core import (
    DRAW_THINGS_CLI_PATH,
    DRAW_THINGS_MODEL,
    DRAW_THINGS_REMOTE_PORT,
    DRAW_THINGS_REMOTE_URL,
    Item,
    images_dir,
    interests_path,
    item_stem,
    items_dir,
    render_item_note,
    render_signal_note,
    resolve_ai_artist_dir,
    signal_stem,
    signals_dir,
)
from .drawthings import DrawThingsError, generate_image
from .interests import parse_interests, pick_direction, record_outcome, render_interests
from .prompt import compose_prompt
from .scoring import SelfScore, self_score
from .signals import fetch_current_event

_TOOL_NAME = "artist-agent"
_TOOL = register_tool(_TOOL_NAME)

app = typer.Typer(add_completion=False)


@app.command()
def generate(
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="LLM backend for prompt composition")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override the prompt-composition model")
    ] = None,
    vision_provider: Annotated[
        str | None,
        typer.Option("--vision-provider", help="LLM backend for self-critique (must support vision)"),
    ] = None,
    vision_model: Annotated[
        str | None, typer.Option("--vision-model", help="Override the self-critique model")
    ] = None,
    ai_artist_dir: Annotated[
        str | None, typer.Option("--dir", "-d", help="Root folder for images/items/interests")
    ] = None,
    dry_run: Annotated[bool, dry_run_option()] = False,
    no_llm: Annotated[bool, no_llm_option()] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Print the composed prompt and self-critique")
    ] = False,
) -> None:
    """Generate one image end-to-end and record it as a new item."""
    provider = get_setting(_TOOL_NAME, "provider", cli_val=provider, default="local")
    model = get_setting(
        _TOOL_NAME, "model", cli_val=model, default="llama3.2:3b" if provider in ("local", "ollama") else None
    )
    # Vision defaults to anthropic regardless of the text provider's default: self-critique
    # needs a vision-capable model, and no local model here has one installed.
    vision_provider = get_setting(_TOOL_NAME, "vision_provider", cli_val=vision_provider, default="anthropic")
    vision_model = get_setting(_TOOL_NAME, "vision_model", cli_val=vision_model, default=None)

    dry_run = resolve_dry_run(dry_run, no_llm)

    base = resolve_ai_artist_dir(ai_artist_dir)
    i_path = interests_path(base)
    if not i_path.exists():
        typer.echo(f"Error: no interests file found at {i_path}", err=True)
        raise typer.Exit(1)

    preamble, interests = parse_interests(i_path.read_text(encoding="utf-8"))
    if not interests:
        typer.echo(f"Error: {i_path} has no parseable interests", err=True)
        raise typer.Exit(1)

    chosen = pick_direction(interests)
    typer.echo(f"Direction: {chosen.title}")

    current_event = None
    if chosen.signal == "content-discovery":
        current_event = fetch_current_event()
        if current_event and verbose:
            typer.echo(f"Current-event signal: {current_event.title}")

    try:
        llm_provider = resolve_provider(PROVIDERS, provider, model, no_llm=no_llm)
    except Exception as e:  # noqa: BLE001 - top-level CLI boundary: report cleanly and exit, don't show a raw traceback
        typer.echo(f"Error initializing provider '{provider}': {e}", err=True)
        raise typer.Exit(1)

    with timed_run(_TOOL_NAME, getattr(llm_provider, "model", None)) as run:
        image_prompt = compose_prompt(
            llm_provider,
            chosen.description,
            current_event=current_event.title if current_event else None,
        )
        run.item_count = 1
    if verbose:
        typer.echo(f"Prompt: {image_prompt}")

    if dry_run:
        typer.echo("\nDry run -- stopping before generation.")
        raise typer.Exit(0)

    model = chosen.model or DRAW_THINGS_MODEL
    settings = dict(chosen.settings)
    try:
        image_bytes = generate_image(
            image_prompt,
            model,
            settings,
            cli_path=DRAW_THINGS_CLI_PATH,
            remote_url=DRAW_THINGS_REMOTE_URL,
            remote_port=DRAW_THINGS_REMOTE_PORT,
        )
    except DrawThingsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    now = datetime.now().astimezone()

    vision_llm = None
    try:
        vision_llm = resolve_provider(PROVIDERS, vision_provider, vision_model, no_llm=no_llm)
        critique = self_score(vision_llm, chosen.description, image_prompt, image_bytes)
    except Exception as e:  # noqa: BLE001 - self-scoring failure shouldn't discard a real generation
        typer.echo(f"Warning: self-scoring failed, recording without a score: {e}", err=True)
        critique = SelfScore(score=0.0, notes=f"self-scoring failed: {e}")
    if verbose:
        typer.echo(f"Self-critique ({critique.score:.2f}): {critique.notes}")

    models_used = {
        "prompt": f"{provider}/{llm_provider.model}",
        "image": model,
        "critique": f"{vision_provider}/{vision_llm.model}" if vision_llm else f"{vision_provider} (failed)",
    }

    item = Item(
        generated_at=now,
        prompt=image_prompt,
        interest_title=chosen.title,
        settings=settings,
        image_filename="",  # set below once the stem is known
        self_score=critique.score,
        self_score_notes=critique.notes,
        models=models_used,
        current_event_title=current_event.title if current_event else None,
        current_event_source_url=current_event.source_url if current_event else None,
    )
    stem = item_stem(item)
    item.image_filename = f"{stem}.png"

    img_dir, it_dir = images_dir(base), items_dir(base)
    img_dir.mkdir(parents=True, exist_ok=True)
    it_dir.mkdir(parents=True, exist_ok=True)

    (img_dir / item.image_filename).write_bytes(image_bytes)
    note_path = it_dir / f"{stem}.md"
    note_path.write_text(render_item_note(item, f"../images/{item.image_filename}"), encoding="utf-8")

    if current_event:
        sig_dir = signals_dir(base)
        sig_dir.mkdir(parents=True, exist_ok=True)
        sig_path = sig_dir / f"{signal_stem(current_event.title, now)}.md"
        sig_path.write_text(
            render_signal_note(current_event.title, current_event.source_url, now), encoding="utf-8"
        )

    updated_interests = record_outcome(interests, chosen.title, critique.score)
    i_path.write_text(render_interests(updated_interests, preamble=preamble), encoding="utf-8")

    typer.echo(f"Image:  {img_dir / item.image_filename}")
    typer.echo(f"Item:   {note_path}")
    typer.echo(f"\nDone. self_score={critique.score:.2f}")


if __name__ == "__main__":
    app()
