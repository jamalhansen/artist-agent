# artist-agent

A generative artist with a daily practice. Each morning a studio session decides what to make
and how (theme, idea, palette, medium, technique, size, study or piece, and what it expects to
learn), Draw Things renders it, a vision model critiques it against that intention and names
the lesson, and the piece is written as a portable markdown item beside its image. On Sundays
the artist rewrites its themes and its practice file from the feedback it learns from.

Two artists run from this code (`~/bin/artist-agent-run`, 06:00 daily): `~/iCloud/ai-artist`
learns only from its own critic, `~/iCloud/ai-artist-mentored` only from Jamal's star ratings
(`calib art rate`, `/rate-art`). See each folder's `_state/artist.toml`.

## Installation
```bash
uv sync
```

## Usage
```bash
uv run artist-agent generate --dir /path/to/art-root [--verbose] [--dry-run]
uv run artist-agent reflect  --dir /path/to/art-root [--min-new 3] [--dry-run]
```

Standard flags: `--dry-run`, `--no-llm`, `--provider`/`--model` (the studio session),
`--vision-provider`/`--vision-model` (the critic, must support vision), `--verbose`.

## What an artist keeps

```
_state/interests.md   themes: title, score line, description; optional model/setting pins
_state/practice.md    the artist's account of its style, palettes, media, themes, stretch, lessons
_state/artist.toml    name, who it learns from, which folders it reads ratings from
_state/history/       every previous interests.md and practice.md, plus changelog.md
items/*.md            one note per piece; images/*.png beside them
```

Each item records `interest` (the theme), `intention` (theme, idea, palette, feel, medium,
mode, hypothesis, follows, why), `prompt`, `settings` (everything passed to draw-things-cli,
seed included), `models` (studio, image, critic), `self_score` and `lesson`, and the
`human_score` that `calib art rate` fills in.

## The studio session (`studio.py`)

One structured call. The artist reads its practice file, its themes with their score and
per-medium record, its last 14 pieces (what it tried, how it was rated, what it learned; an
unrated piece still shows what was tried), the media catalog, and an optional real-world
signal, and returns an `Intention`. Rules it works under: learn from the feedback, don't
repeat yesterday's theme and medium unless deliberately returning to a piece (`follows`),
make about one piece in three a study that tries something from "Stretching toward", keep
the themes' "avoid" rules.

`resolve()` then meets the constraints: a theme Jamal pinned with `model:` wins over the
artist's medium; a medium not in the catalog falls back to a record-weighted draw; the
technique is clamped to the medium's room; a theme's own setting lines go on top; a seed
is chosen so the piece can be re-run.

## Media and technique (`catalog.py`)

`~/.config/local-first/artist-agent.toml`, `[[image_models]]`, one entry per checkpoint in
`DRAWTHINGS_MODELS_DIR`:

```toml
[[image_models]]
file = "flux_1_schnell_q8p.ckpt"
name = "FLUX.1 schnell 8-bit"
notes = "What it does well, in the artist's terms."
steps = 4            # known-good defaults the artist starts from
cfg = 4.5
width = 1024
height = 576
sampler = "Euler A Trailing"
clip_skip = 2
steps_range = [1, 8]           # the room: clamped
cfg_range = [1.0, 6.0]
samplers = ["Euler A Trailing", "DPM++ 2M Trailing"]   # only these accepted
sizes = ["1024x576", "1024x1024", "576x1024"]          # only these accepted
minutes = 2.5
```

The same file carries `provider`/`vision_provider` for the studio session and the critic.

## Reflection (`reflect.py`, Sundays)

With at least `--min-new` pieces of new feedback, the artist rewrites each theme's
description from the feedback on it (it may move a theme's core idea, add one theme or
retire one, 3 to 5 themes, titles never change; it may pin one theme to a medium or set
its technique when the record is clear), then rewrites `practice.md`: style, palettes it
likes, what each medium does in its hands, themes in its own words, what to stretch toward
next week, the ten most useful lessons. A rewrite that drops a section is refused and the
previous file kept. Both files are archived under `_state/history/` first.
