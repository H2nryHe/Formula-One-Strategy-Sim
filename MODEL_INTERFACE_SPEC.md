# MODEL_INTERFACE_SPEC — Pluggable Models & Contracts

This spec defines stable interfaces so you can swap model families (linear → GAM → GBDT → state-space → sequence)
without changing the strategy engine and evaluation harness.

---

## 0) Design Principles
- **Single canonical state** (`RaceState`) is the only source of truth.
- Models consume **features** derived from `RaceState` and output typed objects.
- Decision/search consumes model outputs and produces **Top‑K plans** in a fixed schema.
- Every model must declare:
  - `model_name`, `model_version`, `trained_on`, and `feature_schema_version`.

---

## 1) Tick Semantics (Replay-first)
### 1.1 Tick granularity
Default: **lap-based tick**.
- Tick `t` corresponds to the end of lap `L` (after timing update and events applied).
- Recommendations produced for actions taken before lap `L+1`.

Optional later: time-based tick (not required for historical MVP).

### 1.2 Deterministic replay
Given the same session + config + seeds, the engine must produce identical recommendations.

---

## 2) Canonical Data Structures

### 2.1 RaceState (minimum fields)
`RaceState` must include:
- `session_id`, `lap`
- `track_status`: enum {GREEN, YELLOW, VSC, SC, RED}
- `weather`: snapshot dict (air_c, track_c, humidity, rainfall, wind_ms) optional keys allowed
- `cars`: mapping `driver_id -> CarState`

### 2.2 CarState (minimum fields)
- `driver_id`, `team`
- `position`
- `gap_to_leader_ms`, `interval_ahead_ms`, `interval_behind_ms`
- `tyre_compound`: enum {SOFT, MEDIUM, HARD, INTER, WET, UNKNOWN}
- `tyre_age_laps`
- `stint_id` (increments on each pit stop)
- `recent_lap_times_ms`: last N clean laps (after filters)
- `is_pitting` / `pit_in` / `pit_out` flags (as available)

### 2.3 FeatureFrame (model input)
All models consume a feature dictionary or dataframe row with:
- stable key names
- explicit missing value policy
- `feature_schema_version` string

---

## 3) Model Interfaces

### 3.1 PaceModel
Purpose: predict expected lap time distribution under current conditions.

**Interface**
- `predict_lap_time(state: RaceState, driver_id: str, *, horizon_lap: int = 1) -> LapTimePred`

**Output: LapTimePred**
- `mean_ms: float`
- `sigma_ms: float` (or other uncertainty descriptor)
- `components: dict[str, float]` (optional; for explainability)
  - e.g., base, degr, traffic, track_status, weather

**Notes**
- MVP baseline can be: rolling median + linear adjustments.
- Upgrades: GAM, GBDT residual, state-space latent pace.

---

### 3.2 DegradationModel
Purpose: predict tyre-related pace delta as a function of tyre age, compound, and conditions.

**Interface**
- `predict_delta(compound: str, tyre_age_laps: int, ctx: dict) -> DegradationPred`

**Output: DegradationPred**
- `delta_mean_ms: float` (expected added time due to degradation)
- `delta_sigma_ms: float`
- `cliff_risk: float` (optional, 0–1)

**Notes**
- MVP: piecewise linear / quadratic by compound.
- Later: Bayesian state-space (latent degradation rate + uncertainty).

---

### 3.3 PitPolicyModel (Opponent + optionally self-behavior)
Purpose: predict pit probabilities for opponents (and optionally for “behavioral” evaluation).

**Interface**
- `predict_pit_prob(state: RaceState, driver_id: str, *, window_laps: int = 1) -> PitProbPred`

**Output: PitProbPred**
- `p_pit_in_window: float`
- `p_compound: dict[str, float]` (optional; conditional distribution)
- `calibration_meta: dict` (optional)

**Notes**
- MVP: logistic regression or rules.
- Upgrade: GBDT, sequence model, or game-aware policy.

---

### 3.4 ScenarioModel (Uncertainty generator)
Purpose: generate scenario trajectories for SC persistence and weather transition.

**Interface**
- `sample_scenarios(state: RaceState, *, horizon_laps: int, n: int, seed: int) -> list[Scenario]`

**Scenario minimal fields**
- per-lap track status path (or transition times)
- per-lap weather path (or deltas)
- optional incident shocks (disabled by default)

---

## 4) Strategy Search Interface

### 4.1 Action space
Canonical action enum:
- `STAY_OUT`
- `PIT_TO_SOFT`, `PIT_TO_MEDIUM`, `PIT_TO_HARD`
- `PIT_TO_INTER`, `PIT_TO_WET` (enabled when weather risk is active)

### 4.2 Search contract
**Interface**
- `recommend(state: RaceState, target_driver: str, *, horizon_laps: int, top_k: int, seed: int) -> RecommendationBundle`

Search may implement:
- limited rollout enumeration
- MCTS
- dynamic programming approximation
- robust optimization variants

### 4.3 Output schema (must be stable)
**RecommendationBundle**
- `session_id`, `lap`, `target_driver`
- `generated_at_ts` (replay time)
- `top_k: list[Plan]`
- `baselines: dict[str, PlanComparison]` (optional)
- `assumptions_hash` (string) — ties back to ASSUMPTIONS_AND_LIMITS
- `model_versions: dict[str, str]` — pace/degr/pit/scenario/search

**Plan**
- `plan_id`
- `actions`: list[{"at_lap": int, "action": str}]
- `metrics`:
  - `delta_time_mean_ms`
  - `delta_time_p10_ms`, `delta_time_p50_ms`, `delta_time_p90_ms`
  - `p_gain_pos_ge_1`
  - `risk_sigma_ms`
- `explanations`: list[{"code": str, "text": str, "evidence": dict}]
- `counterfactuals`: dict[str, dict]  # e.g. compare vs STAY_OUT

**Hard requirements**
- Always return exactly `top_k` plans if feasible; else return fewer with a warning.
- Explanations must include ≥2 reason codes unless the state is degenerate (e.g., red flag).

---

## 5) Explainability Interface

### 5.1 Reason code generation
**Interface**
- `explain(state: RaceState, target_driver: str, plan: Plan, diagnostics: dict) -> list[Explanation]`

**Explanation**
- `code`: string
- `text`: short human explanation
- `evidence`: structured evidence (e.g., gap, tyre_age, SC flag)

### 5.2 Minimum evidence fields by code (examples)
- `SC_WINDOW`: `{"track_status": "SC", "pit_loss_ms": ..., "field_bunched": true}`
- `UNDERCUT_THREAT`: `{"opponent": "...", "gap_ms": ..., "opponent_tyre_age": ...}`
- `TYRE_CLIFF`: `{"tyre_age": ..., "compound": "...", "cliff_risk": ...}`
- `RAIN_RISK`: `{"rainfall": ..., "track_temp": ..., "p_rain_next_10m": ...}` (if available)

---

## 6) Baselines (for evaluation harness)
All evaluation runs must compute:
- `STAY_OUT` baseline
- `RULE_TYRE_AGE` baseline (configurable thresholds)
- `COPY_LEADER` or `COPY_NEAREST` baseline (configurable)

Baselines must be emitted in the same `Plan` schema.

---

## 7) Versioning & Compatibility
- `feature_schema_version`: bump when feature names/meaning change
- `assumptions_hash`: hash of the active assumptions config
- Model serialization must record:
  - training races
  - preprocessing steps
  - seed
  - commit SHA (optional)

---

## 8) Performance Budgets (Replay MVP targets)
- Per tick (single target driver):
  - `top_k=3`, `horizon=8`, `n_scenarios=100` should run within a reasonable time on a laptop.
- If too slow:
  - reduce scenarios
  - use analytic approximations
  - cache opponent policy outputs
