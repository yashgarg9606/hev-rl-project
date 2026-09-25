from pathlib import Path

from motor_efficiency_map import MotorEfficiencyMap


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "motor_maps"
)


def main():

    motor1 = MotorEfficiencyMap(
        MAP_DIR / "motor1_efficiency_contours.csv"
    )

    motor2 = MotorEfficiencyMap(
        MAP_DIR / "motor2_efficiency_contours.csv"
    )

    print("=" * 70)
    print("MOTOR EFFICIENCY MAP — VALIDATION")
    print("=" * 70)

    print("\nMotor 1 data points:")
    print(len(motor1.data))

    print("\nMotor 2 data points:")
    print(len(motor2.data))

    print("\nEfficiency levels in Motor 1:")
    print(
        sorted(
            motor1.data[
                "efficiency_percent"
            ].unique()
        )
    )

    print("\nEfficiency levels in Motor 2:")
    print(
        sorted(
            motor2.data[
                "efficiency_percent"
            ].unique()
        )
    )

    print("\nTest operating points")
    print("-" * 70)

    test_points = [
        (500.0, 50.0),
        (1000.0, 50.0),
        (3000.0, 50.0),
        (5000.0, 50.0),
        (6000.0, 20.0),
        (8000.0, 15.0),
    ]

    for speed, torque in test_points:

        eta1 = motor1.get_efficiency(
            speed,
            torque,
        )

        eta2 = motor2.get_efficiency(
            speed,
            torque * 155.0 / 180.0,
        )

        print(
            f"{speed:7.0f} rpm | "
            f"M1: {eta1:6.2f}% | "
            f"M2: {eta2:6.2f}%"
        )

        assert 51.0 <= eta1 <= 95.0
        assert 51.0 <= eta2 <= 95.0

    print("\nBoundary checks")
    print("-" * 70)

    eta_boundary = motor1.get_efficiency(
        8000.0,
        15.0,
    )

    assert 51.0 <= eta_boundary <= 95.0

    print(
        "✓ Motor 1 accepts valid operating point inside reconstructed map"
    )

    try:

        motor1.get_efficiency(
            5000.0,
            -10.0,
        )

    except ValueError:

        print(
            "✓ Negative torque is explicitly rejected"
        )

    else:

        raise AssertionError(
            "Negative torque should be rejected"
        )

    print("\n" + "=" * 70)
    print("✓ MOTOR EFFICIENCY MAP VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()