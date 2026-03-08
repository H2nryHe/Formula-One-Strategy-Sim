# TASK_BOARD — F1 Real-Time Strategy Simulator

> Guiding principle: **Replay-first** (historical) to validate the full loop; then add live.

## Stage 0 — Repo bootstrap (0.5–1 day)
- [ ] Initialize Python project (`pyproject.toml`, ruff, pytest)
- [ ] Add skeleton modules under `src/f1sim/*`
- [ ] Add CI (lint + tests)
- [ ] Add `.gitignore` to exclude large/raw data

**Exit criteria**
- CI passes on empty/smoke tests
- `python -m f1sim` runs a hello pipeline

---

## Stage 1 — Data ingest (1–3 days)
### FastF1 connector (replay)
- [ ] Implement `FastF1Connector.load_session(year, gp, session)`
- [ ] Normalize into canonical tables: laps, events, weather (as available)
- [ ] Write to SQLite with schema from `PROJECT_SPEC.md`

### OpenF1 connector (optional early)
- [ ] Implement `OpenF1Connector` with rate limiting + retries
- [ ] Map OpenF1 entities → canonical schema

**Exit criteria**
- One race ingested into SQL with consistent keys
- Basic sanity checks (laps monotonic, no negative lap times)

---

## Stage 2 — Race state engine (2–4 days)
- [ ] Define `RaceState` dataclass (order, gaps, tyre info, track status, weather)
- [ ] Implement `StateEngine.step(tick)` that:
  - applies lap updates
  - resolves pit in/out and tyre changes
  - applies track status transitions (SC/VSC)
- [ ] Add derived features:
  - stint length, tyre age, recent pace trend
  - traffic indicator (gap to car ahead/behind)

**Exit criteria**
- Replay reproduces plausible race order and pit events
- Unit tests for pit/SC transitions

---

## Stage 3 — Baseline models (3–6 days)
### Pace + degradation
- [ ] Implement baseline pace model (simple regression / rolling median per driver)
- [ ] Implement degradation curve per compound (piecewise linear or spline-lite)
- [ ] Add traffic penalty heuristic (gap < threshold)

### Opponent pit policy (simple)
- [ ] Estimate pit probability from features:
  - tyre age, recent pace drop, SC state, weather
- [ ] Fallback heuristic if model not trained

**Exit criteria**
- Model can predict next-lap lap time within reasonable error (replay)
- Pit probability outputs are non-degenerate (not all 0/1)

---

## Stage 4 — Strategy search + simulator (4–8 days)
- [ ] Define action space and constraints
- [ ] Implement rollout simulator:
  - horizon H laps
  - scenarios for SC persistence + weather transition
- [ ] Score candidate plans (expected delta time, position gain probability, risk)
- [ ] Return top‑K plans with structured outputs

**Exit criteria**
- For a chosen “target car”, recommendations update each lap
- Top plan changes sensibly under SC / rain signals

---

## Stage 5 — Explainability layer (1–3 days)
- [ ] Add reason codes:
  - `SC_WINDOW`, `UNDERCUT_THREAT`, `TYRE_CLIFF`, `RAIN_RISK`, `TRAFFIC_PENALTY`
- [ ] Generate explanations from feature thresholds + contribution heuristics
- [ ] Include counterfactual: compare `PIT_NOW` vs `STAY_OUT`

**Exit criteria**
- Each plan has ≥2 human-readable reasons
- Reasons match obvious race situations in replay

---

## Stage 6 — Serving + UI (2–5 days)
- [ ] FastAPI endpoints:
  - `/state`, `/recommendations`, `/replay/{session_id}/tick`
- [ ] Minimal dashboard (Streamlit ok):
  - state table, top‑K plans, timeline chart
- [ ] Add “replay control” (next lap, autoplay)

**Exit criteria**
- End-to-end demo: open UI → run replay → see live updating suggestions

---

## Stage 7 — Evaluation harness (2–5 days)
- [ ] Implement evaluation scripts:
  - pit window hit rate
  - decision quality via counterfactual simulation
  - calibration curves for probabilities
- [ ] Add report artifact generation (markdown + plots)

**Exit criteria**
- `python -m f1sim.eval --session ...` produces metrics + plots
- Baselines included (always stay out, naive pit-at-tyre-age)

---

## Stage 8 — v1 hardening (optional)
- [ ] Caching, rate limits, and fault tolerance for live connector
- [ ] Better opponent modeling (team style embeddings)
- [ ] More robust uncertainty modeling (bootstrap / Bayesian-lite)
- [ ] Documentation: architecture diagram + API docs

