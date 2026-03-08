# PROJECT_SPEC — F1 Real-Time Strategy Simulator (MVP → v1)

## 0) One-liner
A real-time (or replay) race strategy engine that ingests lap-by-lap signals (timing, events, weather), maintains a live race state, and outputs top‑K pit/tyre strategy recommendations with uncertainty + explanations.

## 1) Scope & Non-goals
### In-scope (MVP)
- **Replay-first** engine using historical sessions (race + quali optional).
- Strategy suggestions for:
  - Pit timing (next 1–3 laps and horizon windows)
  - Tyre compound choice (dry + intermediate/wet as available)
  - Under/overcut style decisions when relevant
- **Explainable output**: why a recommendation is made (traffic, tyre age, SC window, weather).
- **Dashboard/API**: serve current state + recommendations.

### Non-goals (MVP)
- “Team-grade” proprietary telemetry (tyre surface temps, brake temps, radio intent).
- Perfect incident/SC prediction.
- Full car performance model (power unit modes, ERS deployment) beyond coarse proxies.

## 2) Product Personas
- **Analyst/Streamer**: wants plausible “what should they do now?” suggestions during replay/live.
- **Quant/Engineer**: wants a reproducible backtest + evaluation harness.

## 3) Inputs (Data)
### Primary sources (recommended for MVP)
- **Historical timing/telemetry**: FastF1 (Python) for replay datasets.
- **Live / quasi-live**: OpenF1 (HTTP API) for timing/events/weather (depends on availability & your usage policy).

> NOTE: Always respect data source Terms/ToS. MVP can be “personal/portfolio/research” mode; commercial use requires proper licensing.

### Required fields (minimum)
- Session metadata: year, circuit, session type, start time, lap count
- Per-car:
  - lap number, lap time, sector times (if available)
  - position, interval/gap, pit in/out, tyre compound, tyre age (laps on tyre)
  - status flags (DNF, in garage)
- Race control events:
  - SC/VSC/Red flag/Yellow flags, retirements, penalties (if available)
- Weather:
  - track temp, air temp, humidity, rainfall (or a proxy), wind speed (optional)
- Track features (static):
  - pit loss time, DRS zones (optional), circuit class (street/permanent), typical degradation class

## 4) System Overview
### 4.1 Architecture (modules)
1. **Ingest**
   - connectors: `fastf1_connector`, `openf1_connector`
   - produces: normalized event stream + time-series tables

2. **State Engine**
   - maintains a canonical `RaceState` updated per tick (lap boundary or time-based tick)
   - resolves: pit events, SC phases, tyre changes, order, gaps, derived signals

3. **Models**
   - **Pace model**: predict expected lap time under clean air, degradation, traffic loss
   - **Degradation model**: tyre age → pace delta; separate by compound and conditions
   - **Pit decision model**:
     - next-lap pit probability (baseline + features) and/or
     - expected value comparison of candidate actions via simulation

4. **Strategy Search / Simulator**
   - enumerates candidate strategies (top‑K) over a horizon H laps (e.g., 5–15)
   - evaluates via Monte Carlo / scenario rollouts:
     - scenarios: SC persistence, weather transitions, opponents’ pit behavior

5. **Explainability**
   - attach “reasons” to each recommendation:
     - SC window open, undercut threat, tyre cliff approaching, rain risk, traffic penalty

6. **Serving Layer**
   - REST endpoints (FastAPI recommended)
   - dashboard (Streamlit or simple web UI)

### 4.2 Data Model (storage)
- Use SQL (SQLite for MVP, PostgreSQL for v1).
- Core tables:
  - `sessions(session_id, year, circuit, session_type, start_ts, laps, source)`
  - `cars(driver_id, team, car_number, season)`
  - `laps(session_id, driver_id, lap, lap_time_ms, s1_ms, s2_ms, s3_ms, position, gap_ms, interval_ms, pit_in, pit_out, tyre, tyre_age, track_status)`
  - `events(session_id, ts, lap, type, payload_json)`
  - `weather(session_id, ts, lap, air_c, track_c, humidity, rainfall, wind_ms, payload_json)`
  - `strategy_calls(session_id, ts, lap, driver_id, topk_json, model_version, notes)`

## 5) Core Algorithms (MVP choices)
### 5.1 Baseline pace + degradation
- Start with a **hierarchical regression**:
  - `lap_time = base(driver, stint) + degr(compound, tyre_age) + traffic(gap) + sc_adjust + noise`
- Keep it explainable and robust; upgrade later.

### 5.2 Candidate action space
- For a target car, at each decision point:
  - `STAY_OUT`
  - `PIT_TO_SOFT/MED/HARD`
  - `PIT_TO_INTER/WET` (if weather signal suggests)
- Add constraints:
  - minimum pit separation, mandatory tyre rules (if modeling a season with such rules)
  - avoid illegal compound sequences if applicable

### 5.3 Search
- MVP: limited-depth rollout:
  - evaluate actions over next `H` laps (e.g., 8)
  - opponents follow a simple policy model (pit probability baseline)
- Score each plan by:
  - expected total time delta vs baseline
  - expected position delta / probability of gaining places
  - risk penalties (variance, SC uncertainty, rain uncertainty)

## 6) Outputs
### 6.1 API payload (per lap or tick)
- `state_summary`:
  - lap, leader, gaps, track status (green/SC/VSC), weather snapshot
- `recommendations` (top‑K):
  - plan_id, action_now, optional action_later
  - `E[delta_time_ms]`, `P(gain_position>=1)`, `risk_sigma`
  - `explanations`: list of reason codes + human text
  - `counterfactuals`: what if stay out vs pit now

### 6.2 UI
- “Strategy panel”:
  - best plan + 2 alternatives
  - confidence bar, key drivers (traffic, degradation, SC, weather)
- “State timeline”:
  - SC/VSC phases, pit stops, weather trend

## 7) Evaluation Plan (high-level)
- See `EVAL_PLAN.md` for metrics + methodology:
  - event-conditional accuracy (pit window hit rate)
  - decision quality via counterfactual simulation
  - calibration of probabilities

## 8) Repo Structure (suggested)
```
f1-strategy-sim/
  README.md
  PROJECT_SPEC.md
  TASK_BOARD.md
  EVAL_PLAN.md
  pyproject.toml
  src/
    f1sim/
      ingest/
      state/
      models/
      sim/
      api/
      ui/
      utils/
  tests/
  data/
    raw/          # ignored
    snapshots/    # ignored or small fixtures
  notebooks/
  docs/
```

## 9) Engineering Standards
- Reproducibility: pinned dependencies, deterministic seeds for simulations
- CI: lint (ruff), type-check (mypy optional), unit tests (pytest)
- Data fixtures: small, legal-to-ship “silver” snapshots for tests

## 10) Milestones
- M0: replay engine works on 1 race end-to-end
- M1: top‑K recommendations stable + explainable
- M2: evaluation harness with baseline metrics
- M3: live/quasi-live connector + dashboard

