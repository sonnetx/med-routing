# med-routing — agent notes

Operational gotchas and architecture pointers for future agents. Implementation details are in the code; this file is for things that aren't obvious from reading source.

## Stack quick map

- App: FastAPI on `:8000` (or `:8001` if `docker-compose.override.yml` is present — see below)
- Prometheus on `:9090`, scrapes the app every 5s
- Grafana on `:3000`, anonymous viewer on, dashboard auto-loads
- Persistence under `./data/`: SQLite (`med_routing.db`), datasets (`datasets/*.json`), completion cache (`completion_cache.json`), judge cache (`judge_cache.json`)
- Audit log under `./audit/decisions-YYYY-MM-DD.jsonl`
- Demo CSVs under `./demo_data/`

## Prometheus dies silently — always probe directly

`docker compose ps` will report Prometheus as `Up Xh` even when the process inside is dead. Symptom: Grafana panels return `400 Bad Request` with `status_source=downstream` in Grafana logs.

**Always check Prometheus directly first:**
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9090/-/ready
# 200 = alive; 000 = container is dead despite what `docker compose ps` claims
```

If dead and a normal `docker compose up -d prometheus` fails with `endpoint with name med-routing-prometheus already exists`, the network has a stale endpoint:

```bash
docker compose down                       # stops everything; volumes survive
docker network rm med-routing_default     # may need to be retried
docker compose up -d
```

## NLI install is ~2GB; skip it unless you need it

`pip install -e ".[nli]"` pulls torch + CUDA libs (~2GB). The Dockerfile installs base deps only. The `semantic_entropy` router (Kuhn et al. NLI clustering) is gated on `ENABLE_NLI=true` and silently skips registration if the import fails — so default startup is fast and lean. The `semantic_entropy_embed` router is the torch-free alternative (uses OpenAI embeddings for clustering).

## Persisted caches survive container restarts

Three files under `./data/` (bind-mounted volume):
- `completion_cache.json` — every weak/strong/sampler completion. Reused across eval runs. Saved after each `/v1/datasets/{id}/evaluate` and on shutdown.
- `judge_cache.json` — gpt-4o-mini judge verdicts, keyed by sha1(question + predicted answer)
- `datasets/{id}.json` — uploaded datasets and their last eval reports

Cold eval ≈ 4-5 min for 25 questions. Warm replay (post-restart) ≈ 19s. **Pre-warm before any live demo.**

## /v1/datasets workflow — two paths, deterministic IDs

- **Upload with `ground_truth` column populated** → deterministic content-hash ID; re-uploading same CSV finds the same dataset and its saved report. Use this path for the demo "instant report" UX.
- **Upload without `ground_truth`** → calls `/v1/datasets/{id}/generate` (faked, loads cached Opus answers from `demo_data/clinical_eval_25.opus.json` keyed by qid). For the demo's "show me the generation flow", pass `?fresh=1` to bypass dedup so the generate steps render every click.

The eval sweep (`/v1/datasets/{id}/evaluate`) pre-computes weak/strong/sampler/judge once per question, then sweeps thresholds purely in-memory. Two routers ship by default in the UI: `self_reported` and `semantic_entropy_embed`.

## Grafana datasource UID footgun

`grafana/provisioning/datasources/datasource.yml` declares `uid: prometheus`. Grafana's provisioning is **create-if-missing for the UID** — if Grafana booted *before* this line existed, it'll have a different auto-generated UID, and every dashboard panel (which references `uid: prometheus`) will return "datasource not found." Fix: `docker compose up -d --force-recreate grafana` (the in-container `/var/lib/grafana` is wiped, provisioning re-runs cleanly; bind-mounted dashboards are unaffected).

## docker-compose.override.yml is local-machine state

It remaps `8001:8000` (because the original author has another process on `:8000`) and bind-mounts `./src/med_routing` and `./demo_data` for live source iteration without rebuilding. **Delete this file if you don't need either** — the remap will force you to use `:8001` and the bind mount can confuse builds.

## Where things actually run during the demo

1. `/` — single-question cascade demo (existing)
2. `/datasets` — company-facing eval workflow (new): upload CSV → cost estimate → generate (faked) → eval sweep → Pareto chart with recommendation
3. `:3000` — Grafana, dashboard `med-routing — Uncertainty Cascade`
4. `:9090` — Prometheus

## Routers currently registered

`self_reported`, `self_consistency`, `predictive_entropy`, `semantic_entropy_embed`, `auto`. Optional behind env flags: `semantic_entropy` (NLI), `routellm`, `learned`. The `auto` router dispatches by prompt format (MCQ vs free-form) to a sub-router.

## Demo storyline that uses this codebase

- News hook: bromism case (Annals of Internal Medicine, Aug 2025 — patient followed ChatGPT advice to use sodium bromide as a salt substitute, was hospitalized)
- Live demo 1: the bromism question on `/`, cascade catches it
- Live demo 2: `/datasets` with the cached demo dataset, Pareto chart appears in <1s (instant if persistence + dedup ID works)
- Closing: cumulative savings + audit/FHIR + per-customer eval as the value-prop sandwich
