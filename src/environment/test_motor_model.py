from motor_model import DualMotorModel


def print_result(name, result):
    print(f"\n{name}")
    print("-" * 60)

    for key, value in result.items():
        print(f"{key:35s}: {value}")


def main():

    motor = DualMotorModel()

    print("\n" + "=" * 60)
    print("DUAL-MOTOR MODEL — TORQUE DISTRIBUTION VALIDATION")
    print("=" * 60)

    wheel_torque = 1000.0
    velocity = 50.0

    # ---------------------------------------------------------
    # CASE 1 — Motor 1 only
    # ---------------------------------------------------------

    result_0 = motor.calculate_operating_point(
        velocity_kmh=velocity,
        wheel_torque_nm=wheel_torque,
        sigma_tor=0.0,
    )

    print_result(
        "CASE 1 — SIGMA = 0.0 (MOTOR 2 ONLY)",
        result_0,
    )

    # ---------------------------------------------------------
    # CASE 2 — 50/50
    # ---------------------------------------------------------

    result_05 = motor.calculate_operating_point(
        velocity_kmh=velocity,
        wheel_torque_nm=wheel_torque,
        sigma_tor=0.5,
    )

    print_result(
        "CASE 2 — SIGMA = 0.5 (50/50)",
        result_05,
    )

    # ---------------------------------------------------------
    # CASE 3 — Motor 1 only
    # ---------------------------------------------------------

    result_1 = motor.calculate_operating_point(
        velocity_kmh=velocity,
        wheel_torque_nm=wheel_torque,
        sigma_tor=1.0,
    )

    print_result(
        "CASE 3 — SIGMA = 1.0 (MOTOR 1 ONLY)",
        result_1,
    )

    # ---------------------------------------------------------
    # CASE 4 — Deliberately infeasible torque
    # ---------------------------------------------------------

    infeasible = motor.calculate_operating_point(
        velocity_kmh=50.0,
        wheel_torque_nm=3000.0,
        sigma_tor=0.5,
    )

    print_result(
        "CASE 4 — DELIBERATELY HIGH TORQUE",
        infeasible,
    )

    # ---------------------------------------------------------
    # AUTOMATED CHECKS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUTOMATED CHECKS")
    print("=" * 60)

    # sigma = 0
    assert result_0["motor1_torque_nm"] == 0.0
    assert result_0["motor2_torque_nm"] > 0

    print("✓ σ=0 assigns all wheel torque to Motor 2")

    # sigma = 0.5
    assert abs(
        result_05["wheel_torque_motor1_nm"]
        - result_05["wheel_torque_motor2_nm"]
    ) < 1e-12

    print("✓ σ=0.5 gives equal wheel-torque contribution")

    # sigma = 1
    assert result_1["motor1_torque_nm"] > 0
    assert result_1["motor2_torque_nm"] == 0.0

    print("✓ σ=1 assigns all wheel torque to Motor 1")

    # Torque conservation
    for result in [result_0, result_05, result_1]:
        reconstructed = (
            result["wheel_torque_motor1_nm"]
            + result["wheel_torque_motor2_nm"]
        )

        assert abs(reconstructed - wheel_torque) < 1e-9

    print("✓ Wheel torque is conserved for all distributions")

    # Deliberately infeasible case
    assert not infeasible["overall_feasible"]

    print("✓ Infeasible high-demand operating point is detected")

    print("\n🎉 TORQUE DISTRIBUTION VALIDATION PASSED")


if __name__ == "__main__":
    main()