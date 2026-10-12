#!/usr/bin/env python3
"""Artist Agent CLI -- one command, one piece: a studio session decides the theme, idea,
palette, medium, technique and prompt; Draw Things renders it; the critic scores it against
the intention and names the lesson; the piece is written as a portable item note and the
outcome folded back into the artist's themes."""

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
from local_first_common.heartbeat import heartbeat
from local_first_common.providers import PROVIDERS
from local_first_common.tracking import register_tool

from .catalog import load_catalog
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
from .feedback import apply_scores, collect, load_config, recent_work, track_record
from .interests import parse_interests, record_outcome, render_interests
from .practice import is_complete, load_practice, save_practice
from .reflect import apply as apply_revisions
from .reflect import record as record_reflection
from .reflect import revise, revise_practice, unconsumed
from .scoring import SelfScore, self_score
from .signals import fetch_current_event
from .studio import RECENT_PIECES, plan, resolve

_TOOL_NAME = "artist-agent"
_TOOL = register_tool(_TOOL_NAME)

app = typer.Typer(add_completion=False)


@app.command()
def generate(
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="LLM backend for the studio session")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override the studio-session model")] = None,
    vision_provider: Annotated[
        str | None,
        typer.Option("--vision-provider", help="LLM backend for self-critique (must support vision)"),
    ] = None,
    vision_model: Annotated[str | None, typer.Option("--vision-model", help="Override the self-critique model")] = None,
    ai_artist_dir: Annotated[
        str | None, typer.Option("--dir", "-d", help="Root folder for images/items/interests")
    ] = None,
    dry_run: Annotated[bool, dry_run_option()] = False,
    no_llm: Annotated[bool, no_llm_option()] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print the intention, prompt and self-critique")] = False,
) -> None:
    """Make one piece end-to-end and record it as a new item."""
    # The studio session makes every creative decision of the day in one structured call; a 3B
    # local model could not (2026-10-12), so it defaults to the same provider the critic uses.
    provider = str(get_setting(_TOOL_NAME, "provider", cli_val=provider, default="anthropic"))
    model = get_setting(
        _TOOL_NAME, "model", cli_val=model, default="llama3.2:3b" if provider in ("local", "ollama") else None
    )
    # Vision defaults to anthropic regardless of the text provider's default: self-critique
    # needs a vision-capable model, and no local model here has one installed.
    vision_provider = str(get_setting(_TOOL_NAME, "vision_provider", cli_val=vision_provider, default="anthropic"))
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

    artist = load_config(base)
    all_feedback = collect(artist, base)
    if artist.learns_from == "jamal":
        # Scores come from Jamal's ratings, recomputed each run since ratings arrive days later.
        interests = apply_scores(interests, all_feedback)
        if not dry_run:
            i_path.write_text(render_interests(interests, preamble=preamble), encoding="utf-8")
    record = track_record(all_feedback)
    recent = recent_work(artist, base, RECENT_PIECES)
    practice = load_practice(base)
    catalog = load_catalog(_TOOL_NAME)

    # Best effort, never a reason to fail the day: the studio may use it or not.
    current_event = fetch_current_event()
    if current_event and verbose:
        typer.echo(f"Current-event signal: {current_event.title}")

    try:
        llm_provider = resolve_provider(PROVIDERS, provider, model, no_llm=no_llm, tool_name=_TOOL_NAME)
    except Exception as e:  # noqa: BLE001 - top-level CLI boundary: report cleanly and exit, don't show a raw traceback
        typer.echo(f"Error initializing provider '{provider}': {e}", err=True)
        raise typer.Exit(1) from None

    # Heartbeat after each stage so process-doctor judges progress by the beat, not by
    # CPU: image generation and the vision critique are long, mostly-idle waits. Replaces
    # the 600 s kill-watchdog that artist-agent-run carried from 2026-09-13 to 2026-10-04.
    heartbeat()
    llm_provider.item_count = 1
    now = datetime.now().astimezone()
    intention = plan(
        llm_provider,
        practice,
        interests,
        record,
        recent,
        artist.learns_from,
        catalog,
        current_event=current_event.title if current_event else None,
        today=now.strftime("%A %Y-%m-%d"),
    )
    choice = resolve(intention, interests, catalog, record, DRAW_THINGS_MODEL)
    heartbeat()
    # The theme on record is the matching direction's exact title, so scores stay attributable;
    # a one-off idea keeps the title the artist gave it.
    theme_title = choice.interest.title if choice.interest else intention.theme.strip()
    uses_signal = bool(current_event and choice.interest and choice.interest.signal == "content-discovery")
    typer.echo(f"Theme:  {theme_title}" + ("" if choice.interest else " (one-off)"))
    typer.echo(
        f"Mode:   {intention.mode} -- {intention.hypothesis}" if intention.hypothesis else f"Mode:   {intention.mode}"
    )
    typer.echo(f"Medium: {choice.model} ({choice.how}); " + ", ".join(f"{k} {v}" for k, v in choice.settings.items()))
    if verbose:
        typer.echo(f"Idea:   {intention.idea}")
        typer.echo(f"Palette: {intention.palette} | feel: {intention.feel}")
        if intention.follows:
            typer.echo(f"Follows: {intention.follows}")
        typer.echo(f"Why:    {intention.why}")
        typer.echo(f"Prompt: {intention.prompt}")
    image_prompt = intention.prompt.strip() or intention.idea.strip()

    if dry_run:
        typer.echo("\nDry run -- stopping before generation.")
        raise typer.Exit(0)

    try:
        image_bytes = generate_image(
            image_prompt,
            choice.model,
            choice.settings,
            cli_path=DRAW_THINGS_CLI_PATH,
            remote_url=DRAW_THINGS_REMOTE_URL,
            remote_port=DRAW_THINGS_REMOTE_PORT,
        )
    except DrawThingsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    heartbeat()

    theme_description = choice.interest.description if choice.interest else intention.idea
    vision_llm = None
    try:
        vision_llm = resolve_provider(PROVIDERS, vision_provider, vision_model, no_llm=no_llm, tool_name=_TOOL_NAME)
        critique = self_score(
            vision_llm, theme_description, image_prompt, image_bytes, intention.record(), choice.settings
        )
    except Exception as e:  # noqa: BLE001 - self-scoring failure shouldn't discard a real generation
        typer.echo(f"Warning: self-scoring failed, recording without a score: {e}", err=True)
        critique = SelfScore(score=None, notes=f"self-scoring failed: {e}")
    heartbeat()
    if verbose and critique.score is not None:
        typer.echo(f"Self-critique ({critique.score:.2f}): {critique.notes}")
        typer.echo(f"Lesson: {critique.lesson}")

    models_used = {
        "prompt": f"{provider}/{llm_provider.model}",
        "image": choice.model,
        "critique": f"{vision_provider}/{vision_llm.model}" if vision_llm else f"{vision_provider} (failed)",
    }

    item = Item(
        generated_at=now,
        prompt=image_prompt,
        interest_title=theme_title,
        settings=choice.settings,
        image_filename="",  # set below once the stem is known
        self_score=critique.score,
        self_score_notes=critique.notes,
        models=models_used,
        intention=intention.record(),
        lesson=critique.lesson,
        current_event_title=current_event.title if uses_signal else None,
        current_event_source_url=current_event.source_url if uses_signal else None,
        artist=artist.name,
    )
    stem = item_stem(item)
    item.image_filename = f"{stem}.png"

    img_dir, it_dir = images_dir(base), items_dir(base)
    img_dir.mkdir(parents=True, exist_ok=True)
    it_dir.mkdir(parents=True, exist_ok=True)

    (img_dir / item.image_filename).write_bytes(image_bytes)
    note_path = it_dir / f"{stem}.md"
    note_path.write_text(render_item_note(item, f"../images/{item.image_filename}"), encoding="utf-8")

    if uses_signal and current_event:
        sig_dir = signals_dir(base)
        sig_dir.mkdir(parents=True, exist_ok=True)
        sig_path = sig_dir / f"{signal_stem(current_event.title, now)}.md"
        sig_path.write_text(render_signal_note(current_event.title, current_event.source_url, now), encoding="utf-8")

    if artist.learns_from == "self" and critique.score is not None and choice.interest:
        updated_interests = record_outcome(interests, choice.interest.title, critique.score)
        i_path.write_text(render_interests(updated_interests, preamble=preamble), encoding="utf-8")

    typer.echo(f"Image:  {img_dir / item.image_filename}")
    typer.echo(f"Item:   {note_path}")
    typer.echo(f"\nDone. self_score={'none (critic failed)' if critique.score is None else f'{critique.score:.2f}'}")


@app.command()
def reflect(
    ai_artist_dir: Annotated[
        str | None, typer.Option("--dir", "-d", help="Root folder for images/items/interests")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="LLM for rewriting the directions")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model for rewriting the directions")] = None,
    min_new: Annotated[int, typer.Option("--min-new", help="Skip unless this many new pieces of feedback exist")] = 3,
    dry_run: Annotated[bool, dry_run_option()] = False,
) -> None:
    """Rewrite this artist's themes and its practice file from the feedback it learns from."""
    provider = str(get_setting(_TOOL_NAME, "reflect_provider", cli_val=provider, default="claude-code"))
    model = get_setting(_TOOL_NAME, "reflect_model", cli_val=model, default="sonnet")
    base = resolve_ai_artist_dir(ai_artist_dir)
    i_path = interests_path(base)
    preamble, interests = parse_interests(i_path.read_text(encoding="utf-8"))
    artist = load_config(base)
    all_feedback = collect(artist, base)
    new = unconsumed(base, all_feedback)
    if len(new) < min_new:
        typer.echo(f"{artist.name}: {len(new)} new piece(s) of feedback, waiting for {min_new}.")
        return
    heartbeat()  # two long LLM calls follow; see generate() for why process-doctor needs this
    llm = resolve_provider(PROVIDERS, provider, model, fallback=False, tool_name=_TOOL_NAME)
    catalog = load_catalog(_TOOL_NAME)
    record = track_record(all_feedback)
    reflection = revise(llm, interests, new, artist.learns_from, catalog, DRAW_THINGS_MODEL, record)
    revised, log = apply_revisions(interests, reflection, new, catalog)
    heartbeat()
    practice_before = load_practice(base)
    practice_rev = revise_practice(llm, practice_before, revised, new, artist.learns_from, record)
    if is_complete(practice_rev.practice):
        log.append("- **practice**: " + " ".join(practice_rev.change.split()))
        practice_after = practice_rev.practice
    else:
        log.append("- **practice**: rewrite dropped a section, kept the previous file")
        practice_after = None
    heartbeat()
    typer.echo(f"{artist.name} (learns from {artist.learns_from}), {len(new)} new piece(s) of feedback:")
    typer.echo("\n".join(log) or "- no changes")
    if dry_run:
        return
    snapshot = record_reflection(
        base, preamble, interests, revised, log, new, datetime.now().astimezone(), practice_before
    )
    i_path.write_text(render_interests(revised, preamble=preamble), encoding="utf-8")
    if practice_after is not None:
        save_practice(base, practice_after)
    typer.echo(f"Previous directions archived: {snapshot}")


if __name__ == "__main__":
    app()
