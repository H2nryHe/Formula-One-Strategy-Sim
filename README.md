# Formula One Strategy Sim

`f1sim` is a replay-first Formula 1 race strategy simulator. The current MVP can ingest historical sessions into SQLite, replay lap-end race state, evaluate baseline models, and produce deterministic top-K rollout recommendations with structured explanations.

## Latest Version

Current version: `v0.8`

### What's new in v0.8
- Improved recommendation stability and reduced pathological default actions.
- Fixed backward compatibility for older SQLite databases without `team_calls`.
- Corrected actual-action reconstruction logic for replay evaluation.
- Tightened offline evaluation consistency between recommended action and historical action labels.

See `CHANGELOG.md` for full details.

## Scope

This repository is currently limited to **historical replay only**. The intended engine consumes lap-end timing, events, and weather snapshots from public data sources, maintains a canonical `RaceState`, and will later produce top-K strategy recommendations with explanations.

The repository explicitly aligns with:

- [PROJECT_SPEC_v2.md](PROJECT_SPEC_v2.md)
- [ASSUMPTIONS_AND_LIMITS.md](ASSUMPTIONS_AND_LIMITS.md)
- [MODEL_INTERFACE_SPEC.md](MODEL_INTERFACE_SPEC.md)

## Rules Supported

- Non-Monaco FIA dry-tyre compliance for historical replay
- If no `INTER` or `WET` is used, the driver must use at least two distinct dry compounds
- If `INTER` or `WET` is used at any point, the two-dry-compound requirement is waived

## Ground Truth Team Calls

This repository can derive a public-timing ground truth view of what teams actually did:

- `PIT` vs `STAY_OUT` at each lap
- the tyre compound chosen after each pit stop
- optional track-status context at the pit lap

This is not radio or intent parsing. It is timing-derived instrumentation only.

Convention used everywhere in code and evaluation:

- a pit lap is the lap where `pit_in == True`
- `compound_after` is the next observed tyre compound for that driver after that pit-in lap

## Current package layout

- `src/f1sim/`: replay state, ingest, models, rollout search, evaluation, and explanation logic
- `tests/`: synthetic offline test coverage
- `.github/workflows/ci.yml`: GitHub Actions for `ruff` and `pytest`

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m f1sim
```

You can also run the installed smoke entrypoint directly:

```bash
f1sim-smoke
```

## What MVP Can Do

- Replay-only lap-end race-state reconstruction from canonical SQLite tables
- Explainable baseline models for pace, pit policy, degradation, and scenarios
- Rollout-based top-K recommendations with reason codes and counterfactuals
- Offline evaluation outputs in JSON and markdown
- Read-only Streamlit demo over replayed `RaceState` and `RecommendationBundle` outputs

## Demo

Run the MVP demo against a previously ingested historical SQLite database:

```bash
python3 -m pip install -e .
streamlit run app.py
```

The demo is deterministic for a fixed lap and seed. It caches session rows, replayed lap-end states, and
per-lap recommendation payloads in process so moving the lap slider stays responsive after the first load.

Suggested capture workflow:

- Run `streamlit run app.py`
- Open the demo in your browser and select a stable lap/seed
- Capture a screenshot with your OS screenshot tool
- For a short GIF, record a lap-slider interaction with a lightweight recorder such as Kap or Licecap

## Example Recommendation Snippet

```json
{
  "plan_id": "STAY_OUT",
  "actions": [],
  "explanations": [
    {
      "code": "SC_WINDOW",
      "text": "Neutralized conditions shrink the effective pit-loss window.",
      "evidence": {
        "track_status": "SC",
        "n_scenarios": 8,
        "pit_loss_ms": 12000.0
      }
    },
    {
      "code": "TRACK_POSITION",
      "text": "Track position and current gaps frame the near-term action value.",
      "evidence": {
        "position": 2,
        "gap_to_leader_ms": 3200.0,
        "interval_ahead_ms": 3200.0,
        "action": "STAY_OUT"
      }
    }
  ],
  "counterfactuals": {
    "vs_STAY_OUT": {
      "reference_plan_id": "STAY_OUT",
      "reference_action": "STAY_OUT",
      "delta_time_mean_ms": 0.0,
      "delta_time_p50_ms": 0.0
    },
    "vs_PIT_NEXT_LAP": {
      "reference_plan_id": "PIT_TO_HARD",
      "reference_action": "PIT_TO_HARD",
      "delta_time_mean_ms": 16043.0859375,
      "delta_time_p50_ms": 16032.125
    }
  }
}
```
