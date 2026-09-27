"""Legacy audit entry point for normalized target actions and terminal masking.

Feasible physical bounds are computed only by the environment. Target critics
consume the normalized target actor output, so replay needs no motor-map or
future-driving-cycle data for a Bellman update.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from src.agent.test_ddpg_feasible_action_path import NormalizedDDPGTests
    names = ["test_current_and_target_critics_receive_normalized_actions",
             "test_all_terminal_targets_equal_rewards_without_network_evaluation",
             "test_learning_never_constructs_physics_or_maps_actions"]
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(NormalizedDDPGTests(name) for name in names)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
