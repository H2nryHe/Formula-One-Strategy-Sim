# TASK_BOARD — F1 Strategy Simulator (Replay-First, v2)

> v2 focus: **historical replay only**, strict interfaces, and **evaluation-first** so model upgrades are comparable.

---

## Stage 0 — Repo bootstrap (0.5–1 day)
- [ ] Python project skeleton (`pyproject.toml`, ruff, pytest)
- [ ] Repo structure under `src/f1sim/*`
- [ ] CI: lint + tests
- [ ] Add docs: `PROJECT_SPEC.md`, `ASSUMPTIONS_AND_LIMITS.md`, `MODEL_INTERFACE_SPEC.md`, `EVAL_PLAN.md`

**Exit criteria**
- CI passes
- `python -m f1sim` runs a smoke pipeline

---

## Stage 1 — Historical ingest to canonical schema (1–3 days)
- [ ] Implement `FastF1Connector.load_session(...)`
- [ ] Normalize to canonical tables: laps, events, weather (where available)
- [ ] Persist to SQLite

**Exit criteria**
- One race ingested with consistent primary keys
- Sanity checks pass (laps monotonic; no negative times)

---

## Stage 2 — Replay State Engine + cleaning (2–4 days)
- [ ] Implement `RaceState` / `CarState` per `MODEL_INTERFACE_SPEC.md`
- [ ] Implement `StateEngine.step(lap_end_tick)`
- [ ] Implement cleaning rules:
  - mark/exclude inlap/outlap
  - mark SC/VSC laps
  - traffic proxy flags (gap thresholds)
- [ ] Derived features: tyre_age, stint_id, recent pace trend

**Exit criteria**
- Replay reproduces plausible order and pit transitions
- Unit tests for pit/track-status transitions

---

## Stage 3 — Evaluation harness (minimal) comes early (2–4 days)
> Even with crude models, lock the evaluation protocol now.

- [ ] Implement race-based train/test split tooling
- [ ] Behavioral metrics scaffolding:
  - pit-in-next-W (W=1/3/5) AUROC/Brier + calibration
  - timing error for pit lap
- [ ] Decision quality scaffolding:
  - counterfactual Δtime vs baselines (stay out / tyre-age rule / copy leader)
  - summary stats (mean/median/P10/P90)

**Exit criteria**
- `python -m f1sim.eval --session ...` produces a metrics JSON + markdown summary
- Baselines are emitted in the same `Plan` schema

---

## Stage 4 — Baseline models v0 (3–6 days)
- [ ] `PaceModelV0`: rolling median + linear adjustments (traffic/SC/weather)
- [ ] `PitPolicyModelV0`: logistic regression or rules (opponents)
- [ ] `DegradationModelV0`: placeholder heuristics (piecewise/constant)
- [ ] `ScenarioModelV0`: SC persistence scenarios (simple) + optional weather drift

**Exit criteria**
- Models produce non-degenerate outputs across multiple races
- Behavioral metrics above naive baselines on at least some races

---

## Stage 5 — Strategy search + top‑K output (4–8 days)
- [ ] Implement action space + constraints
- [ ] Rollout simulator (horizon H, N scenarios)
- [ ] Scoring: E[Δtime], P(gain≥1), risk sigma/tail
- [ ] Emit stable `RecommendationBundle` schema

**Exit criteria**
- Top‑K plans update each lap for a target driver
- Recommendations shift sensibly under SC vs green segments

---

## Stage 6 — Explainability layer (1–3 days)
- [ ] Reason codes + evidence:
  - SC_WINDOW, UNDERCUT_THREAT, TYRE_CLIFF (stub), TRAFFIC_PENALTY, RAIN_RISK (optional)
- [ ] Counterfactual compare best plan vs STAY_OUT and PIT_NEXT_LAP

**Exit criteria**
- Each plan has ≥2 explanations (except degenerate states)
- Explanations match obvious race situations in replay

---

## Stage 7 — Model Zoo upgrades + comparison (iterative)
> Add models one by one, keeping interfaces and eval fixed.

- [ ] Upgrade A: GAM / hierarchical regression (pace and/or pit policy)
- [ ] Upgrade B: GBDT residual correction / pit probability
- [ ] Upgrade C: State-space tyre degradation (post-MVP)
- [ ] Upgrade D: Sequence model for opponent behavior (optional)
- [ ] Upgrade E: Search upgrade (MCTS/DP/robust optimization)

**Comparison requirements**
- Same train/test split
- Same evaluation tick set
- Same scenario seeds (or seed grid)
- Produce a single comparison report:
  - behavioral + decision-quality metrics
  - runtime/latency per tick

**Exit criteria**
- `python -m f1sim.compare --models ...` generates a table + plots + markdown report

---

## Stage 8 — Optional UI demo (2–5 days)
- [ ] Replay dashboard (Streamlit or web):
  - state table + top‑K plans + confidence
  - timeline of events/pits/track status

**Exit criteria**
- End-to-end demo video/gif: run replay → see strategy suggestions

