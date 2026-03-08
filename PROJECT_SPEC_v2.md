# PROJECT_SPEC — F1 Strategy Simulator (Replay-First, v2)

## 0) One-liner
A **historical (replay-first)** race strategy engine that ingests lap-by-lap timing + events + weather, maintains a live replay `RaceState`, and outputs **top‑K robust strategy recommendations** with uncertainty and explanations.

> This v2 spec aligns with:
- `ASSUMPTIONS_AND_LIMITS.md` (task definition + limits)
- `MODEL_INTERFACE_SPEC.md` (pluggable model contracts)

---

## 1) Scope & Non-goals

### In-scope (MVP / v0)
- **Historical-only** replay engine (no real-time data requirement).
- Canonical `RaceState` updated on **lap-end ticks**.
- Strategy recommendations for a target car:
  - pit timing (next 1–3 laps + horizon windows)
  - tyre choice (dry compounds; inter/wet optional if weather signal exists)
  - simple undercut/overcut reasoning when supported by gaps/traffic proxies
- **Explainable output**: reason codes + evidence + counterfactual comparisons.
- Two-layer evaluation: **behavioral** + **decision quality**.

### Non-goals (MVP / v0)
- Team-only telemetry (tyre temps, ERS modes, damage diagnostics, radio intent).
- Perfect incident/SC prediction.
- Full-fidelity physics (overtaking models, pit congestion, double-stacks) beyond coarse approximations.

---

## 2) Strategy Positioning (What we are optimizing)
We produce **public-data optimal/robust strategy suggestions** under explicit assumptions.

- The full assumptions and limitations are specified in `ASSUMPTIONS_AND_LIMITS.md`.
- “Optimal/robust” means: maximize expected outcome under our simulator and penalize uncertainty (variance/tail risk) via scenario-based rollouts.

---

## 3) Inputs (Historical Data Only)

### Primary historical source (recommended)
- **FastF1** for session loading and replay datasets (laps, telemetry where available, weather where available).

> Always respect data source Terms/ToS.

### Required minimum fields
- Session metadata: year, circuit, session type, start time, lap count
- Per-car per-lap:
  - lap number, lap time (sector times optional)
  - position, gap/interval
  - pit in/out (or pit stop detection)
  - tyre compound, derived tyre age (laps on tyre)
  - track status marker where available (green/SC/VSC/yellow)
- Weather snapshot (if available): air/track temp, humidity, rainfall proxy

### Static circuit parameters (hand-maintained, MVP)
- pit loss time estimate (ms) — per circuit if possible, else global constant

---

## 4) System Overview

### 4.1 Modules
1. **Ingest**
   - `FastF1Connector` loads sessions and normalizes to canonical tables.
2. **State Engine**
   - `StateEngine.step()` updates `RaceState` per lap tick:
     - order, gaps, pit transitions, tyre age, track status, weather snapshot
3. **Feature Builder**
   - derives stable features with a versioned schema (`feature_schema_version`).
4. **Models (Pluggable)**
   - `PaceModel`: expected lap time distribution
   - `PitPolicyModel`: pit probability for opponents (and behavioral eval)
   - `DegradationModel`: **stub/heuristic in MVP**, upgraded post-MVP
   - `ScenarioModel`: SC/weather uncertainty scenarios
5. **Strategy Search**
   - enumerates candidate actions and evaluates via rollouts; returns top‑K plans.
6. **Explainability Layer**
   - reason codes + evidence + counterfactuals
7. **Evaluation Harness**
   - two-layer evaluation per `EVAL_PLAN.md`

> All interfaces must follow `MODEL_INTERFACE_SPEC.md`.

---

## 5) Baseline (MVP) Modeling Choices

### 5.1 Baseline PaceModel (explainable)
Recommended MVP baseline:
- Rolling median/mean of recent **clean laps** per driver within stint
- Plus linear/additive adjustments for:
  - track status (SC/VSC multipliers)
  - traffic penalty (gap thresholds)
  - weather proxy (if available)

This may be implemented as:
- simple heuristics + linear regression, or
- a small GAM-like additive model (still explainable)

### 5.2 Baseline PitPolicyModel
- Logistic regression or transparent rules:
  - tyre age, recent pace drop, SC state, weather proxy, track position/gaps

### 5.3 DegradationModel (MVP: placeholder)
- MVP uses:
  - piecewise linear or quadratic degradation-by-compound heuristics, or
  - constant “soft drift” assumptions
- **Post-MVP**: replace with a state-space tyre degradation model (Bayesian) while keeping the same interface.

---

## 6) Strategy Search (MVP)
- Action space (minimum):
  - `STAY_OUT`
  - `PIT_TO_SOFT/MEDIUM/HARD`
  - optionally `PIT_TO_INTER/WET` when rain risk triggers
- Horizon: H laps (e.g., 8–15)
- Scenarios: sample SC persistence / weather transitions (if modeled)
- Score each plan:
  - expected delta time vs baseline
  - probability of gaining places (proxy)
  - risk (sigma / tail penalty)

Outputs are in the fixed `RecommendationBundle` schema.

---

## 7) Outputs
### 7.1 Programmatic output
- `RecommendationBundle` with:
  - top‑K plans (metrics + uncertainty quantiles)
  - reason codes + evidence
  - counterfactual comparisons

### 7.2 Optional UI (later)
- minimal replay dashboard showing:
  - race state + top‑K plans + confidence
  - timeline of track status and pit events

---

## 8) Data Storage
- SQLite for MVP; PostgreSQL optional for v1.
- Canonical tables:
  - sessions, cars, laps, events, weather
  - strategy_calls (store outputs + model_versions + assumptions_hash)

---

## 9) Engineering Standards
- Deterministic replay (seeded scenarios)
- Reproducible training/eval splits (race-based)
- CI: ruff + pytest (+ optional mypy)
- Small “silver fixtures” for tests

---

## 10) Milestones (Replay-first)
- M0: ingest + replay state engine works on 1 race
- M1: baseline top‑K recommendations produced each lap with explanations
- M2: evaluation harness (behavior + decision quality) produces metrics
- M3: model zoo upgrades + comparison report

