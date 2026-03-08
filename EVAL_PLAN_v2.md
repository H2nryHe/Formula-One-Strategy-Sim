# EVAL_PLAN — F1 Strategy Simulator (Replay-First, v2)

## 0) Evaluation philosophy
Strategy evaluation must be explicit about assumptions and objectives.
- The task definition and limits are in `ASSUMPTIONS_AND_LIMITS.md`.
- We evaluate in **two layers**:
  1) **Behavioral prediction** (what happened)
  2) **Decision quality** (what helps under our simulator)

These are different and must not be conflated.

---

## 1) Data Splits (avoid leakage)
- Split by **race/session** (not by lap):
  - Train: a set of races
  - Test: disjoint races
- Optional: hold out circuits to test generalization.

Every run logs:
- `session_ids_train`, `session_ids_test`
- `feature_schema_version`
- `assumptions_hash`
- `model_versions`
- simulation seeds

---

## 2) Tick set (when we evaluate)
Default: lap-end ticks.
Recommended evaluation tick filters:
- exclude red flag segments
- exclude inlap/outlap for pace metrics
- compute regime buckets (SC/VSC vs green; wet vs dry if signal exists)

---

## 3) Behavioral Metrics (Layer 1)
### 3.1 Pit window prediction
Task: predict `P(pit within next W laps)` for W ∈ {1,3,5}
- AUROC, AUPRC
- Brier score
- Reliability / calibration (binning)

### 3.2 Pit timing error (conditional)
For actual pit events:
- `|predicted_best_pit_lap - actual_pit_lap|`
Report median and P90.

### 3.3 Compound class prediction (conditional)
If a pit occurs:
- accuracy for {dry vs inter/wet}
Optional: accuracy among {soft/medium/hard} when labels are reliable.

---

## 4) Decision Quality Metrics (Layer 2)
Decision quality is assessed by **counterfactual simulation** using the engine.

### 4.1 Baselines (must include)
All evaluation runs compute baselines in the same `Plan` schema:
- `STAY_OUT`
- `RULE_TYRE_AGE` (configurable thresholds per compound)
- `COPY_LEADER` or `COPY_NEAREST` (configurable)

### 4.2 Counterfactual expected gain
At each tick for selected drivers (or all drivers):
- Evaluate recommended plan `a*` vs baselines:
  - `E[Δtime_ms]` over horizon H (or to race end if feasible)
Report:
- mean/median/P10/P90
- % ticks where `E[Δtime] > 0`

### 4.3 Risk / tail behavior
Report:
- `Std[Δtime]` or `risk_sigma_ms`
- worst decile outcome (CVaR-lite)
- `P(gain ≥ 1)` and `P(lose ≥ 1)` (position proxy)

### 4.4 Regime breakdown
Compute metrics by regime buckets:
- Green vs SC/VSC
- Early stint vs late stint
- Traffic-heavy midpack vs clean air
- Wet transition segments (if available)

---

## 5) Fair model comparison protocol (required for Model Zoo)
To compare models A vs B:
- Same train/test races
- Same evaluation ticks
- Same scenario generator configuration and seed grid
- Same action space and horizon H (unless explicitly ablated)
- Same output schema

Also report:
- runtime per tick (p50/p95)
- memory usage (optional)

---

## 6) Ablations (prove what matters)
- Remove traffic features
- Remove track status features
- Remove weather features
- Replace opponent model with fixed policy
- Reduce scenario count N (speed/quality tradeoff)

---

## 7) Qualitative case studies (portfolio-grade)
Pick 3–5 “strategy drama” races and produce:
- timeline plots (track status, pits)
- recommended top‑K vs baselines
- narrative explanations (“why”)

---

## 8) MVP success criteria (replay-only)
You can claim MVP success when:
- Behavioral:
  - pit-in-next-3 AUROC meaningfully above naive baselines
  - probabilities show reasonable calibration
- Decision quality:
  - positive `E[Δtime]` vs `STAY_OUT` on average
  - tails are controlled (few catastrophic outcomes)
- Engineering:
  - deterministic replay and logged assumptions/model versions
  - stable `RecommendationBundle` schema

