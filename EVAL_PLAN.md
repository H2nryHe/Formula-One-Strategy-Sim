# EVAL_PLAN — F1 Real-Time Strategy Simulator

## 0) Why evaluation is hard
Real teams do not always choose “globally optimal” actions (hidden info, constraints, risk appetite). So evaluate in **two layers**:
1) **Behavioral prediction**: did we predict what actually happened?
2) **Decision quality**: would our action improve outcome under a reasonable simulator?

You need both, and you must be explicit about the assumptions.

---

## 1) Datasets & Splits
### Replay dataset
- Use multiple seasons/circuits if possible.
- Split by **race** (not random laps) to avoid leakage:
  - Train: races A
  - Test: races B (unseen)

### Unit test fixtures
- Tiny “silver” session snapshots (1–3 laps) included in repo.

---

## 2) Behavioral Metrics (predicting real calls)
### 2.1 Pit window hit rate
- Define a window: “pit within next W laps” (W=1,3,5)
- Metric: accuracy / F1 / AUROC for `P(pit in next W laps)`

### 2.2 Timing error
- For actual pit events, measure:
  - `|predicted_best_pit_lap - actual_pit_lap|`
- Report median and P90.

### 2.3 Compound choice accuracy (conditional)
- Condition on “a pit happened”
- Accuracy of predicted compound class (dry vs inter/wet, and soft/med/hard if applicable)

### 2.4 Calibration
- Reliability diagram / Brier score for pit probability
- Goal: probabilities should be meaningful (0.7 should happen ~70% of the time)

---

## 3) Decision Quality Metrics (counterfactual)
This is the key “strategy simulator” evaluation.

### 3.1 Counterfactual delta time
At decision ticks (each lap for each car or selected cars):
- Compare recommended action `a*` against baselines:
  - `STAY_OUT`
  - `PIT_AT_TYRE_AGE_T`
  - “Copy leader” heuristic
- Use your simulator to estimate:
  - `E[Δtime]` over horizon H laps and/or to race end (if feasible)
- Report:
  - mean, median, P10/P90
  - % ticks with positive expected gain

### 3.2 Position gain probability
- Under simulator scenarios:
  - `P(gain ≥ 1 place)` and `P(lose ≥ 1 place)`
- Summarize risk tradeoff.

### 3.3 Robustness / Stress tests
Evaluate on specific regimes:
- SC/VSC segments
- Wet-to-dry transitions
- High degradation circuits
- Traffic-heavy mid-pack

Report metrics per regime.

---

## 4) Ablations (prove value)
- Remove weather features → measure degradation in wet regimes
- Remove opponent model (fixed policy) → measure change in decision quality
- Remove traffic features → measure undercut/overcut performance

---

## 5) Human sanity checks (qualitative)
Pick 3–5 famous races with known strategy drama and produce:
- timeline plots
- model recommendations vs reality
- “why” explanations

A good portfolio demo is often **one** strong case study, clearly narrated.

---

## 6) Reproducibility checklist
- Deterministic seeds for rollouts
- Versioned model artifacts (`model_version`)
- Frozen dependency lock
- Exact session IDs recorded for evaluation runs

---

## 7) Success Criteria (MVP)
You can claim MVP success if:
- Behavioral:
  - pit-in-next-3-laps AUROC significantly above naive baseline
  - probabilities are calibrated (reasonable Brier score; no extreme miscalibration)
- Decision quality:
  - positive expected Δtime on average vs stay-out baseline
  - not overly risky (variance controlled; few catastrophic tails)
- Demo:
  - replay UI shows stable top‑K plans with sensible explanations

