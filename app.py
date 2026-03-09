"""Read-only Streamlit demo for the replay-first F1 strategy MVP."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from f1sim.ui import (
    build_demo_payload,
    build_key_moments,
    build_race_situation_panel,
    build_timeline_rows,
    list_drivers,
    list_sessions,
    max_lap_for_session,
)

DEFAULT_DB_PATH = "data/interim/f1sim.sqlite"
DEFAULT_SEED = 7
DEFAULT_TOP_K = 3
DEFAULT_HORIZON_LAPS = 8
DEFAULT_N_SCENARIOS = 16


def main() -> None:
    st.set_page_config(
        page_title="F1 Strategy Sim MVP Demo",
        page_icon="F1",
        layout="wide",
    )
    st.title("Formula One Strategy Sim")
    st.caption(
        "Replay-only historical demo. The UI is read-only and uses a deterministic rollout seed."
    )

    with st.sidebar:
        st.header("Controls")
        db_path = st.text_input("SQLite DB path", value=DEFAULT_DB_PATH)
        try:
            sessions = list_sessions(db_path)
        except Exception as exc:
            st.error(f"Failed to load sessions: {exc}")
            return
        if not sessions:
            st.error("No sessions found in the SQLite database.")
            return

        session_info = _render_session_selector(sessions)
        state_max_lap = max_lap_for_session(db_path, session_info.session_id)
        selected_driver = _render_driver_selector(db_path, session_info.session_id)
        seed = st.number_input("Seed", min_value=0, value=DEFAULT_SEED, step=1)
        lap = _render_lap_controls(session_info.session_id, state_max_lap)
        key_moments = build_key_moments(
            db_path=db_path,
            session_id=session_info.session_id,
            driver_id=selected_driver,
            seed=int(seed),
            top_k=DEFAULT_TOP_K,
            horizon_laps=DEFAULT_HORIZON_LAPS,
            n_scenarios=DEFAULT_N_SCENARIOS,
        )
        _render_key_moments(key_moments)

    race_state, recommendations = build_demo_payload(
        db_path=db_path,
        session_id=session_info.session_id,
        driver_id=selected_driver,
        lap=lap,
        seed=int(seed),
        top_k=DEFAULT_TOP_K,
        horizon_laps=DEFAULT_HORIZON_LAPS,
        n_scenarios=DEFAULT_N_SCENARIOS,
    )
    situation_panel = build_race_situation_panel(
        db_path=db_path,
        session_id=session_info.session_id,
        lap=lap,
    )
    timeline_rows = build_timeline_rows(
        db_path=db_path,
        session_id=session_info.session_id,
        driver_id=selected_driver,
    )

    summary_left, summary_right = st.columns([2, 1])
    with summary_left:
        st.subheader("Replay State")
        st.caption(
            f"Session `{race_state.session_id}` lap {race_state.lap} | "
            f"track status `{race_state.track_status.value}` | target `{selected_driver}`"
        )
    with summary_right:
        st.metric("Top recommendation", recommendations.top_k[0].plan_id)
        st.metric("Scenario seed", int(seed))

    top10_col, strategy_col, situation_col = st.columns([1.0, 1.0, 0.9], gap="large")
    with top10_col:
        st.markdown("#### Top 10")
        st.dataframe(
            pd.DataFrame(_top10_rows(race_state)),
            width="stretch",
            hide_index=True,
        )
    with strategy_col:
        st.markdown("#### Strategy Panel")
        _render_pred_vs_actual(recommendations)
        for idx, plan in enumerate(recommendations.top_k, start=1):
            with st.container(border=True):
                st.markdown(f"**#{idx} {plan.plan_id}**")
                action_text = ", ".join(
                    f"L{int(action['at_lap'])}: {action['action']}" for action in plan.actions
                )
                st.write(f"Actions: {action_text or 'STAY_OUT'}")
                metrics = plan.metrics
                metric_cols = st.columns(5)
                metric_cols[0].metric("E[delta_time] ms", f"{metrics.delta_time_mean_ms:,.0f}")
                metric_cols[1].metric(
                    "P10 / P50",
                    _format_pair(metrics.delta_time_p10_ms, metrics.delta_time_p50_ms),
                )
                metric_cols[2].metric("P90", f"{metrics.delta_time_p90_ms:,.0f}")
                metric_cols[3].metric("Risk sigma", f"{metrics.risk_sigma_ms:,.0f}")
                metric_cols[4].metric("P(gain>=1)", f"{metrics.p_gain_pos_ge_1:.2f}")
                st.caption("Reasons")
                for explanation in plan.explanations:
                    st.write(
                        f"`{explanation.code}` {explanation.text} "
                        f"({_short_evidence(explanation.evidence)})"
                    )
                if plan.contributions:
                    st.caption("Contributions vs STAY_OUT")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"component": key, "delta_ms": value}
                                for key, value in plan.contributions.items()
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
    with situation_col:
        st.markdown("#### Race-wide Situation")
        _render_situation_list("Undercut window candidates", situation_panel["undercut_candidates"])
        _render_situation_list("Tyre risk ranking", situation_panel["tyre_risk_ranking"])
        _render_situation_list("Traffic hotspots", situation_panel["traffic_hotspots"])

    st.markdown("#### Timeline")
    st.dataframe(
        pd.DataFrame(timeline_rows),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Caching: session rows/states and per-lap recommendation payloads are memoized "
        "in-process."
    )


def _render_session_selector(sessions: tuple[Any, ...]) -> Any:
    selector_mode = st.radio(
        "Session selector",
        options=("session_id", "year/gp/session"),
        horizontal=True,
    )
    if selector_mode == "session_id":
        labels = [session.label for session in sessions]
        selected_label = st.selectbox("Session", options=labels, index=0)
        return next(session for session in sessions if session.label == selected_label)

    years = sorted({session.year for session in sessions}, reverse=True)
    year = st.selectbox("Year", options=years, index=0)
    gps = sorted({session.gp for session in sessions if session.year == year})
    gp = st.selectbox("Grand Prix", options=gps, index=0)
    session_types = [
        session.session_type for session in sessions if session.year == year and session.gp == gp
    ]
    session_type = st.selectbox("Session type", options=session_types, index=0)
    return next(
        session
        for session in sessions
        if session.year == year and session.gp == gp and session.session_type == session_type
    )


def _render_driver_selector(db_path: str, session_id: str) -> str:
    drivers = list(list_drivers(db_path, session_id))
    return st.selectbox("Target driver", options=drivers, index=0)


def _render_lap_controls(session_id: str, max_lap: int) -> int:
    if "demo_lap" not in st.session_state:
        st.session_state.demo_lap = 1
    if st.session_state.get("demo_session_id") != session_id:
        st.session_state.demo_session_id = session_id
        st.session_state.demo_lap = 1

    back_col, forward_col = st.columns(2)
    with back_col:
        if st.button("Step back", width="stretch"):
            st.session_state.demo_lap = max(1, int(st.session_state.demo_lap) - 1)
    with forward_col:
        if st.button("Step forward", width="stretch"):
            st.session_state.demo_lap = min(max_lap, int(st.session_state.demo_lap) + 1)

    st.session_state.demo_lap = st.slider(
        "Lap",
        min_value=1,
        max_value=max_lap,
        value=min(int(st.session_state.demo_lap), max_lap),
    )
    return int(st.session_state.demo_lap)


def _top10_rows(race_state: Any) -> list[dict[str, object]]:
    cars = [
        car
        for car in sorted(race_state.cars.values(), key=lambda car: (car.position, car.driver_id))
        if 1 <= int(car.position) < 900
    ][:10]
    return [
        {
            "position": car.position,
            "driver": car.driver_id,
            "gap_to_leader": _format_ms(car.gap_to_leader_ms),
            "tyre": car.tyre_compound.value,
            "tyre_age": car.tyre_age_laps,
            "last_lap_time": _format_ms(car.last_lap_time_ms),
        }
        for car in cars
    ]


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}"


def _format_pair(first: float, second: float) -> str:
    return f"{first:,.0f} / {second:,.0f}"


def _short_evidence(evidence: dict[str, object]) -> str:
    parts = [f"{key}={value}" for key, value in evidence.items()]
    return ", ".join(parts[:3])


def _render_pred_vs_actual(recommendations: Any) -> None:
    st.caption("Pred vs Actual")
    ground_truth = recommendations.ground_truth
    actual_cols = st.columns(2)
    actual_cols[0].metric("Actual action", str(ground_truth.get("actual_action", "-")))
    actual_cols[1].metric(
        "Actual compound after",
        str(ground_truth.get("actual_compound_after") or "-"),
    )
    pit_timeline = ground_truth.get("pit_timeline", [])
    st.write(
        "Pit timeline:",
        ", ".join(
            f"L{item['lap']} {item['compound_before']}->{item['compound_after']}"
            for item in pit_timeline
        )
        or "none",
    )


def _render_situation_list(title: str, rows: list[dict[str, object]]) -> None:
    st.caption(title)
    if not rows:
        st.write("No items")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_key_moments(key_moments: tuple[dict[str, object], ...]) -> None:
    st.caption("Key moments")
    if not key_moments:
        st.write("No jump targets")
        return
    for moment in key_moments:
        label = f"Lap {moment['lap']}: {moment['summary']}"
        if st.button(label, width="stretch", key=f"jump_{moment['lap']}"):
            st.session_state.demo_lap = int(moment["lap"])


if __name__ == "__main__":
    main()
