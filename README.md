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
