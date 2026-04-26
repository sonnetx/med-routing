# med-routing

**Uncertainty-based LLM cascade for medical Q&A, with Prometheus + Grafana observability.**

A small model answers first; if it's uncertain, the request is escalated to a stronger model. Four uncertainty signals are implemented and compared on MedMCQA:

| Router | Signal | Cost beyond the weak call |
|---|---|---|
| `self_reported` | Ask the weak model "0–100 confidence?" | ~5 tokens |
| `predictive_entropy` | Token-logprob entropy on the answer letter | $0 |
| `self_consistency` | Sample N=5; `1 - modal_count/N` | 5× weak |
| `semantic_entropy` | Sample N=5; bidirectional NLI clustering (DeBERTa-v3-MNLI); entropy over clusters | 5× weak + NLI |
| `routellm` (baseline) | Pre-call learned classifier on the prompt; score = strong-wins probability | RouteLLM forward pass + embedding |

Each request returns a normal OpenAI-compatible response plus `X-Med-Router`, `X-Uncertainty`, `X-Escalated`, `X-Model-Used` headers and a `med_routing` block in the JSON body.

## Architecture

```
client ──► FastAPI /v1/chat/completions ──► Cascade ──► weak model (GPT-4o-mini)
                       │                           │
                       │                           ▼
                       │                       UncertaintyRouter (1 of 4)
                       │                           │
                       │                  score < threshold? ── return weak
                       │                           │ else
                       │                           ▼
                       │                       strong model (GPT-4o)
                       │
                       └──► /metrics ──► Prometheus ──► Grafana
```

## Quickstart

```bash
cp .env.example .env          # add OPENAI_API_KEY
docker compose up --build
```

- Server: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (anonymous viewer; admin/admin to edit) — dashboard "med-routing — Uncertainty Cascade" auto-loads.

Run an eval:

```bash
pip install -e ".[dev]"
python -m med_routing.eval.runner --router self_consistency --n 200
```

This streams 200 MedMCQA validation questions through the cascade, writes per-question rows to `runs/<router>_<ts>.jsonl`, and pushes rolling accuracy to the `medr_accuracy` gauge so Grafana updates live.

## Routers

Set per-request via the `router` field or `X-Router` header. Available names:

- `self_reported`
- `predictive_entropy`
- `self_consistency`
- `semantic_entropy` (only when `ENABLE_NLI=true` and the `nli` extra is installed)
- `routellm` (only when `ENABLE_ROUTELLM=true` and the `routellm` extra is installed) — used as the baseline to beat. Note: in-cascade it still pays the weak-call cost, since the cascade always runs the weak model before scoring; for a true cost-fair RouteLLM baseline, run it as a pre-call short-circuit in eval.

## Configuration

See [.env.example](.env.example). Per-router thresholds are independent — calibrate them on a held-out slice of MedMCQA. Defaults are reasonable but the self-reported one in particular needs tuning since GPT-4o-mini's stated confidence is heavily skewed high.

## Tests

```bash
pytest -q
```

Covers each router, the cache, and the cascade end-to-end with a stubbed OpenAI client.

## References

- Kuhn, Gal, Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Chuang et al. *Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization*. 2025. [arXiv:2502.04428](https://arxiv.org/abs/2502.04428)
- [RouteLLM](https://github.com/lm-sys/RouteLLM) — wired in as the `routellm` baseline router (pretrained `mf` matrix-factorization classifier). The current four uncertainty routers should beat it on MedMCQA since RouteLLM's pretrained checkpoints are trained on general Chatbot Arena preferences, not medical content.
- [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) — Apache 2.0 medical MCQ benchmark.




https://docs.google.com/spreadsheets/d/1J-SINNkb0gIwcNnNawpRijTXEADaWx9WpyPQI-X_GCE/edit?gid=1988170623#gid=1988170623
https://docs.google.com/document/d/1_UCyD2x3T1gNEinnOai3XbupErI4rZlK5uwPiWxRU9U/edit?tab=t.0
https://github.com/sonnetx/med-routing
https://platform.openai.com/settings/organization/usage
https://app-production-c4cc.up.railway.app/
https://app-production-c4cc.up.railway.app/datasets
https://docs.google.com/document/d/1_UCyD2x3T1gNEinnOai3XbupErI4rZlK5uwPiWxRU9U/edit?tab=t.0
https://grafana-production-a828.up.railway.app/d/med-routing-eval/med-routing-e28094-eval-dashboard?from=now-30m&to=now
https://grafana-production-a828.up.railway.app/d/med-routing/med-routing-e28094-uncertainty-cascade?from=now-30m&to=now
https://docs.google.com/presentation/d/14kMAqIn4M5pXx2Bs9EuiykycznvLdmgEwrc_bN_l81Y/edit?slide=id.g3d963976b8a_1_0#slide=id.g3d963976b8a_1_0
