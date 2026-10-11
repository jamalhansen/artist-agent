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
`clip_skip`. A direction without settings takes the defaults for its model from the catalog in
`~/.config/local-first/artist-agent.toml` (`[[image_models]]`, one entry per checkpoint in
`DRAWTHINGS_MODELS_DIR`). A seed is chosen every run. Whatever was passed to `draw-things-cli`
is recorded on the piece as `settings:`, beside `models:` (prompt writer, image model, critic).

On Sundays `artist-agent reflect` sees each direction's model and settings, the same for every
piece of feedback, and the catalog; it may move one direction to another catalog model or change
its settings when the feedback points at the renderer rather than the brief. It may also move a
brief's core idea, add one direction, or retire one (3 to 5 directions, titles never change).
