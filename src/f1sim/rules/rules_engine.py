"""FIA dry-tyre usage rule helpers for replay-time strategy search."""

from __future__ import annotations

from f1sim.contracts import Plan
from f1sim.state import RaceState, TyreCompound

DRY_COMPOUNDS = {
    TyreCompound.SOFT.value,
    TyreCompound.MEDIUM.value,
    TyreCompound.HARD.value,
}
WET_COMPOUNDS = {
    TyreCompound.INTER.value,
    TyreCompound.WET.value,
}
ALL_ACTIONS = (
    "STAY_OUT",
    "PIT_TO_SOFT",
    "PIT_TO_MEDIUM",
    "PIT_TO_HARD",
    "PIT_TO_INTER",
    "PIT_TO_WET",
)


def is_two_dry_required(state: RaceState, driver_id: str) -> bool:
    return not state.cars[driver_id].used_wet


def compounds_used_count(state: RaceState, driver_id: str) -> int:
    return len(state.cars[driver_id].used_dry_compounds)


def plan_satisfies_rules(
    state: RaceState,
    driver_id: str,
    plan: Plan,
    race_total_laps: int,
) -> bool:
    used_dry_compounds, used_wet, last_planned_lap = _project_usage(state, driver_id, plan)
    if used_wet or len(used_dry_compounds) >= 2:
        return True
    return last_planned_lap < race_total_laps


def action_mask(
    state: RaceState,
    driver_id: str,
    lap: int,
    race_total_laps: int,
    *,
    deadline_laps: int = 12,
) -> dict[str, bool]:
    mask = {action: True for action in ALL_ACTIONS}
    if not is_two_dry_required(state, driver_id):
        return mask

    car = state.cars[driver_id]
    dry_count = len(car.used_dry_compounds)
    laps_remaining = max(0, race_total_laps - lap)
    if dry_count >= 2:
        return mask

    if dry_count == 1 and laps_remaining <= deadline_laps:
        for action in ALL_ACTIONS:
            planned_compound = _action_compound(action)
            if action == "STAY_OUT":
                mask[action] = False
            elif planned_compound in WET_COMPOUNDS:
                mask[action] = True
            elif planned_compound in DRY_COMPOUNDS:
                mask[action] = planned_compound not in car.used_dry_compounds
            else:
                mask[action] = False
    elif dry_count < 1 and laps_remaining <= 0:
        mask["STAY_OUT"] = False
    return mask


def _project_usage(
    state: RaceState,
    driver_id: str,
    plan: Plan,
) -> tuple[set[str], bool, int]:
    car = state.cars[driver_id]
    used_dry_compounds = set(car.used_dry_compounds)
    used_wet = car.used_wet
    last_planned_lap = state.lap
    for action in sorted(plan.actions, key=_action_lap):
        action_name = str(action["action"])
        planned_compound = _action_compound(action_name)
        if planned_compound in DRY_COMPOUNDS:
            used_dry_compounds.add(planned_compound)
        elif planned_compound in WET_COMPOUNDS:
            used_wet = True
        last_planned_lap = max(last_planned_lap, _action_lap(action))
    return used_dry_compounds, used_wet, last_planned_lap


def _action_lap(action: dict[str, int | str]) -> int:
    return int(action.get("at_lap", 0))


def _action_compound(action: str) -> str | None:
    if not action.startswith("PIT_TO_"):
        return None
    return action.removeprefix("PIT_TO_")
