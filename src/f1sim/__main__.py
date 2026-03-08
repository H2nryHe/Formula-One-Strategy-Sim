from f1sim.assumptions import ASSUMPTIONS_VERSION, default_assumptions_hash
from f1sim.replay import build_bootstrap_state


def main() -> None:
    """Run a small replay-only smoke check for the Stage 0 scaffold."""
    state = build_bootstrap_state()
    print("f1sim Stage 0 smoke")
    print(f"session={state.session_id} lap={state.lap} cars={len(state.cars)}")
    print(f"track_status={state.track_status} assumptions={ASSUMPTIONS_VERSION}")
    print(f"assumptions_hash={default_assumptions_hash()}")


if __name__ == "__main__":
    main()
