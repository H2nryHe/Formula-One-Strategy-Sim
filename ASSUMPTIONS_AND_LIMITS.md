# ASSUMPTIONS_AND_LIMITS — F1 Strategy Simulator (Public-Data Optimal/Robust)

This document defines **what the system assumes is true** and **what it explicitly does not model**.  
All evaluation claims and model comparisons must be interpreted under these assumptions.

---

## 1) Data Availability Assumptions (Public Signals Only)

### 1.1 Observable inputs (allowed)
From public timing/event feeds and derived features:
- Per-car lap timing: lap time and (if available) sector times
- Position, gaps/intervals
- Pit in/out events; stint segmentation; tyre compound (and derived tyre age in laps)
- Track status / race control: SC, VSC, red flag, yellow flags (as available)
- Weather snapshot: air/track temperature, humidity, rainfall proxy, wind (if available)
- Static circuit parameters (hand-maintained):
  - pit lane loss (approx), race distance, typical degradation class (optional)

### 1.2 Unobservable internal signals (not allowed)
We do **not** use team-only telemetry:
- Tyre carcass/surface temperature, pressures, brake temps
- Power unit modes / ERS deployment maps
- Real-time damage / mechanical health indicators
- Radio intent, strategy calls, fuel targets, lift-and-coast instructions
- Exact tyre compound allocation constraints beyond what is inferable publicly

**Implication:** recommendations represent a **best-effort strategy** given publicly visible information.

---

## 2) Environment & Physics Simplifications

### 2.1 Lap time decomposition
We treat observed lap time as:
- baseline pace (driver/car/session)
- + tyre degradation effect (later: modeled explicitly)
- + traffic penalty (proxy via gaps)
- + track status effect (SC/VSC/yellow)
- + weather effect (coarse)
- + noise/outliers

### 2.2 Fuel effect (simplified)
Fuel burn-off is **not** explicitly measured. MVP assumes:
- either absorbed into a smooth lap-in-race trend
- or captured as a “stint position / lap index” feature

### 2.3 Pit lane cost (approximate)
Pit loss is modeled as:
- a fixed constant per circuit (preferred), or
- a global constant for MVP

Does not include stochastic variance from:
- unsafe releases / holds
- slow tyre changes
- pit entry/exit traffic

### 2.4 Cautions & interruptions
SC/VSC and red flags are modeled as:
- regime switches with coarse lap-time multipliers or deterministic rules
- optional stochastic persistence scenarios for rollouts

We do not attempt perfect prediction of:
- incidents
- safety car deployments
- red flags

---

## 3) Opponent Modeling Assumptions

### 3.1 Public-policy assumption
Opponents’ decisions depend only on public variables (same feature set).

### 3.2 Policy families by stage
- MVP: heuristics or logistic pit probability
- v1+: learned model (GBDT / sequence model) or game-aware policy
- optional: team-style embedding to capture “aggressive vs conservative” tendencies

### 3.3 Independence simplification (MVP)
MVP may assume opponents act independently conditional on features.
Later stages may add:
- coupling via pit lane congestion / double stacks
- explicit cover/response behavior

---

## 4) Objective Function Assumptions (What “Optimal/Robust” Means)

### 4.1 Primary objective
Recommendations optimize **expected** outcome under the simulator:
- minimize expected cumulative time to horizon H (or to race end if feasible)
- optionally maximize expected position/points proxy

### 4.2 Risk-aware utility (robustness)
We incorporate robustness via:
- variance penalty: `E[Δtime] - λ * Std[Δtime]`
- tail penalty: penalize worst decile outcome (CVaR-lite)
- scenario-based uncertainty: SC persistence and weather transition uncertainty

### 4.3 Constraints (rules)
We follow only what we can encode from public rules for the target season, at minimum:
- tyre compound validity (dry vs inter/wet)
- minimum pit separation (technical constraint)
Season-specific constraints (e.g., mandatory compound usage) are optional and must be explicitly enabled.

---

## 5) Explainability Requirements

Every recommendation must expose:
- **Top‑K plans** with structured metrics: `E[Δtime]`, `P(gain≥1)`, risk
- **Reason codes** mapped to human explanations:
  - `SC_WINDOW`, `UNDERCUT_THREAT`, `OVERcut_OPPORTUNITY`, `TYRE_CLIFF`,
    `RAIN_RISK`, `TRAFFIC_PENALTY`, `TRACK_POSITION`, `PIT_CONGESTION` (if modeled)
- **Counterfactual**: compare best plan vs `STAY_OUT` and/or `PIT_NEXT_LAP`

Explainability may be rule-based (thresholds) even if the scorer is complex.

---

## 6) Evaluation Interpretation Limits

### 6.1 Two-layer evaluation
- Behavioral: “did we predict what happened?”
- Decision quality: “would our action help under our simulator?”

These are different and must not be conflated.

### 6.2 No claim of team-grade superiority
We do not claim the system matches or beats real team strategy decisions,
because:
- teams have private information
- teams optimize broader objectives (points, teammate coordination, reliability)

### 6.3 Reproducibility
Every evaluation run must log:
- dataset/session IDs
- model version/hash
- simulation seed(s)
- configuration of assumptions toggles (pit loss constants, SC scenarios, etc.)

---

## 7) Known Failure Modes (Expected)
- Sudden car damage/mechanical issues causing pace loss (unmodeled)
- Weather micro-variations not captured by track-level snapshots
- Highly strategic interactions (e.g., multi-car blocking, team orders) not captured
- Pit lane congestion/double stack timing if not explicitly modeled
- Track position value spikes (e.g., Monaco) underestimated without circuit-specific overtaking models

---

## 8) Planned Relaxations (Roadmap)
- Add Bayesian/state-space tyre degradation with uncertainty
- Add opponent coupling and “cover” behavior
- Add pit lane congestion and double-stack modeling
- Add circuit-specific overtaking difficulty / position value functions
