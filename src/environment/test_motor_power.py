"""
Validation tests for the motor electrical-power model.
"""

from pathlib import Path

from motor_efficiency_map import MotorEfficiencyMap
from motor_power import MotorPowerModel


def main():

    project_root = Path(__file__).resolve().parents[2]

    motor1_map_path = (
        project_root
        / "data"
        / "processed"
        / "motor_maps"
        / "motor1_efficiency_contours.csv"
    )

    motor2_map_path = (
        project_root
        / "data"
        / "processed"
        / "motor_maps"
        / "motor2_efficiency_contours.csv"
    )

    motor1_map = MotorEfficiencyMap(
        motor1_map_path
    )

    motor2_map = MotorEfficiencyMap(
        motor2_map_path
    )

    motor1 = MotorPowerModel(motor1_map)
    motor2 = MotorPowerModel(motor2_map)

    print("=" * 70)
    print("MOTOR ELECTRICAL POWER — VALIDATION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Test 1: Motoring
    # ----------------------------------------------------------

    print("\nTest 1 — Motoring")
    print("-" * 70)

    result = motor1.calculate(
        speed_rpm=5000.0,
        torque_nm=100.0,
    )

    print(
        f"Speed:                 "
        f"{result.speed_rpm:.2f} rpm"
    )

    print(
        f"Torque:                "
        f"{result.torque_nm:.2f} Nm"
    )

    print(
        f"Angular speed:         "
        f"{result.angular_speed_rad_s:.4f} rad/s"
    )

    print(
        f"Mechanical power:     "
        f"{result.mechanical_power_kw:.4f} kW"
    )

    print(
        f"Efficiency:            "
        f"{result.efficiency_percent:.2f}%"
    )

    print(
        f"Electrical power:      "
        f"{result.electrical_power_kw:.4f} kW"
    )

    assert result.mechanical_power_w > 0.0
    assert result.electrical_power_w > 0.0
    assert (
        result.electrical_power_w
        > result.mechanical_power_w
    )

    print("✓ Motoring power calculation is physically consistent")

    # ----------------------------------------------------------
    # Test 2: Regeneration
    # ----------------------------------------------------------

    print("\nTest 2 — Regenerative braking")
    print("-" * 70)

    result = motor1.calculate(
        speed_rpm=5000.0,
        torque_nm=-100.0,
    )

    print(
        f"Speed:                 "
        f"{result.speed_rpm:.2f} rpm"
    )

    print(
        f"Torque:                "
        f"{result.torque_nm:.2f} Nm"
    )

    print(
        f"Mechanical power:     "
        f"{result.mechanical_power_kw:.4f} kW"
    )

    print(
        f"Efficiency:            "
        f"{result.efficiency_percent:.2f}%"
    )

    print(
        f"Electrical power:      "
        f"{result.electrical_power_kw:.4f} kW"
    )

    assert result.mechanical_power_w < 0.0
    assert result.electrical_power_w < 0.0

    # During regeneration, the magnitude of returned
    # electrical power must be lower than mechanical input.
    assert (
        abs(result.electrical_power_w)
        < abs(result.mechanical_power_w)
    )

    print(
        "✓ Regenerative power calculation is "
        "physically consistent"
    )

    # ----------------------------------------------------------
    # Test 3: Zero torque
    # ----------------------------------------------------------

    print("\nTest 3 — Zero torque")
    print("-" * 70)

    result = motor1.calculate(
        speed_rpm=5000.0,
        torque_nm=0.0,
    )

    print(
        f"Mechanical power:     "
        f"{result.mechanical_power_kw:.4f} kW"
    )

    print(
        f"Electrical power:      "
        f"{result.electrical_power_kw:.4f} kW"
    )

    assert result.mechanical_power_w == 0.0
    assert result.electrical_power_w == 0.0

    print("✓ Zero-torque case handled correctly")

    # ----------------------------------------------------------
    # Test 4: Motor 2
    # ----------------------------------------------------------

    print("\nTest 4 — Motor 2")
    print("-" * 70)

    result = motor2.calculate(
        speed_rpm=5000.0,
        torque_nm=80.0,
    )

    print(
        f"Efficiency:            "
        f"{result.efficiency_percent:.2f}%"
    )

    print(
        f"Mechanical power:      "
        f"{result.mechanical_power_kw:.4f} kW"
    )

    print(
        f"Electrical power:      "
        f"{result.electrical_power_kw:.4f} kW"
    )

    assert result.electrical_power_w > 0.0

    print("✓ Motor 2 power calculation works")

    print("\n" + "=" * 70)
    print("✓ MOTOR ELECTRICAL POWER VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()