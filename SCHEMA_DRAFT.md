---
status: draft, not implemented
created: 2026-09-07
---

# Artist agent — preference-state schema (draft)

Groundwork only, per tonight's brainstorm. Mirrors `content-discovery-agent`'s
`store.py` pattern deliberately — same shape of problem (score a generated
item, let a human periodically say kept/dismissed, use that history to judge
future runs), already proven out this session.

## `items` table

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `prompt` | TEXT | full prompt sent to the generator |
| `prompt_source` | TEXT | e.g. `"image-generation-md:duckdb-speed-07"` if seeded from the existing library, or `"generated"` if the agent varied/composed it itself |
| `settings` | TEXT | JSON blob — model, steps, cfg, sampler, seed, resolution, whatever Draw Things' API accepts |
| `theme` | TEXT | content category, matching Image Generation.md's groups (DuckDB/speed, SQL, Python, AI/MCP, workflows/career) |
| `image_path` | TEXT | where the generated file landed |
| `self_score` | REAL | LLM's own rubric score, 0.0-1.0 |
| `self_score_notes` | TEXT | brief LLM rationale — palette adherence, composition, genericness flag |
| `human_score` | REAL | NULL until reviewed |
| `human_notes` | TEXT | NULL until reviewed |
| `status` | TEXT | `new` \| `reviewed` — mirrors content-discovery-agent's new/kept/dismissed, but scored not binary |
| `generated_at` | TEXT | ISO date |
| `reviewed_at` | TEXT | NULL until reviewed |

## Deliberately not doing yet

- No separate "insights" table. Per tonight's brainstorm: promoted, generalizable
  findings ("gold-on-dark-grey outscores emerald-on-dark for AI/MCP") go to a
  markdown note somewhere reviewable, not a database row — same split Contexta
  draws between raw logs and promoted notes. Don't build that path until there's
  at least one real finding worth promoting.
- No fixture/theme-rotation scheduler. First version picks from the existing
  Image Generation.md library manually or via a simple round-robin — don't
  design a scheduling system before there's a single real generation logged.

## Open question for when this becomes real

`self_score` vs `human_score` disagreement is supposed to be a signal (per
the brainstorm) — worth a `disagreement` view/query once there's enough rows
to make it meaningful, not before.
