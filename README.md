# artist-agent

Daily generative-art agent: composes a prompt from an evolving interests file, generates the image via Draw Things, self-critiques it with a vision model, writes a portable markdown item, and folds the outcome back into the interests file so the next run's prompt has learned something.

## Installation
```bash
uv sync
```

## Usage
```bash
uv run artist-agent --dir /path/to/art-root
```

Reads `interests.md` from `--dir` (default: current directory), picks a direction, composes a prompt, generates via Draw Things, and self-critiques with a vision-capable model. The critique and outcome are recorded back into the interests file, and a new item note is written alongside the generated image.

Standard flags: `--dry-run`, `--no-llm`, `--provider`/`--model` (prompt composition), `--vision-provider`/`--vision-model` (self-critique, must support vision), `--verbose`.

## Image models and settings

Each direction in an artist's `_state/interests.md` may pin an image model and settings under
its score line:

```
## Synesthetic wireframe
score: 0.35 (3 generations)
model: flux_1_schnell_q8p.ckpt
steps: 4
cfg: 4.5
sampler: Euler A Trailing
```

Keys: `model`, `steps`, `cfg`, `width`, `height`, `seed`, `sampler` (a Draw Things sampler name),
`clip_skip`. The catalog in `~/.config/local-first/artist-agent.toml` (`[[image_models]]`, one
entry per checkpoint in `DRAWTHINGS_MODELS_DIR`) carries each model's default settings; a
direction's own setting lines go on top of them. A seed is chosen every run. Whatever was passed
to `draw-things-cli` is recorded on the piece as `settings:`, beside `models:` (prompt writer,
image model, critic).

**Daily model draw.** A direction without a `model:` line draws its image model each day from the
catalog, weighted by that direction's mean score on each model over every piece so far (the
artist's own feedback source: critic scores for self-taught, Jamal's stars for mentored). An
untried model weighs as much as the best tried one, so new checkpoints get real trials. A
`model:` line pins the direction to one model instead.

On Sundays `artist-agent reflect` sees, per direction, the track record by model, what it renders
with, the model and settings behind every new piece of feedback, and the catalog. It may pin one
direction to a model when the record is clear, change settings when the feedback points at the
renderer rather than the brief, move a brief's core idea, add one direction, or retire one
(3 to 5 directions, titles never change).
