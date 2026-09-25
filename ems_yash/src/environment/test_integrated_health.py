"""
Phase 2D + Phase 2E integration validation.

Validates that the integrated powertrain correctly propagates:
    vehicle dynamics
        -> dual motors
        -> motor electrical power
        -> battery electrical power
        -> battery state
        -> battery health
        -> motor health
"""

from pathlib import Path

from integrated_powertrain import (
    IntegratedPowertrain,
    default_motor_map_paths,
)


def build_powertrain() -> IntegratedPowertrain:
    """Build the integrated powertrain using the processed motor maps."""

    project_root = Path(__file__).resolve().parents[2]

    motor1_map, motor2_map = default_motor_map_paths(project_root)

    return IntegratedPowertrain(
        motor1_map_path=motor1_map,
        motor2_map_path=motor2_map,
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 2D + 2E — INTEGRATED POWERTRAIN HEALTH VALIDATION")
    print("=" * 70)

    powertrain = build_powertrain()

    # ---------------------------------------------------------------
    # TEST 1 — Initial health state
    # ---------------------------------------------------------------

    print("\nTEST 1 — Initial health state")

    health = powertrain.health_model.get_health_state()

    print(f"Battery SOH:    {health['battery_soh']:.12f}")
    print(f"Motor 1 SOH:    {health['motor1_soh']:.12f}")
    print(f"Motor 2 SOH:    {health['motor2_soh']:.12f}")

    assert health["battery_soh"] == 1.0
    assert health["motor1_soh"] == 1.0
    assert health["motor2_soh"] == 1.0

    print("✓ All components initially at SOH = 1.0")

    # ---------------------------------------------------------------
    # TEST 2 — One normal driving timestep
    # ---------------------------------------------------------------

    print("\nTEST 2 — 50 km/h steady cruising")

    result = powertrain.step(
        velocity_kmh=50.0,
        target_velocity_kmh=50.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Battery power:      {result.battery_power_kw:.6f} kW")
    print(f"Battery current:    {result.battery_current_a:.6f} A")
    print(f"Battery SOC:        {result.soc:.12f}")

    print(f"Battery SOH:        {result.battery_soh:.12f}")
    print(f"Motor 1 SOH:        {result.motor1_soh:.12f}")
    print(f"Motor 2 SOH:        {result.motor2_soh:.12f}")

    assert result.battery_power_kw < 0.0
    assert result.battery_current_a < 0.0

    assert 0.0 < result.battery_soh < 1.0
    assert 0.0 < result.motor1_soh < 1.0
    assert 0.0 < result.motor2_soh < 1.0

    print("✓ Battery health decreased during discharge")
    print("✓ Motor 1 health decreased")
    print("✓ Motor 2 health decreased")

    # ---------------------------------------------------------------
    # TEST 3 — Health changes are physically directional
    # ---------------------------------------------------------------

    print("\nTEST 3 — Health degradation direction")

    health_after_step = powertrain.health_model.get_health_state()

    assert health_after_step["battery_soh"] == result.battery_soh
    assert health_after_step["motor1_soh"] == result.motor1_soh
    assert health_after_step["motor2_soh"] == result.motor2_soh

    assert result.battery_soh < 1.0
    assert result.motor1_soh < 1.0
    assert result.motor2_soh < 1.0

    print("✓ Returned SOH matches internal health model")
    print("✓ All active components show degradation")

    # ---------------------------------------------------------------
    # TEST 4 — Second timestep accumulates degradation
    # ---------------------------------------------------------------

    print("\nTEST 4 — Health accumulation over multiple timesteps")

    previous_battery_soh = result.battery_soh
    previous_motor1_soh = result.motor1_soh
    previous_motor2_soh = result.motor2_soh

    result_2 = powertrain.step(
        velocity_kmh=50.0,
        target_velocity_kmh=50.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Battery SOH:        {result_2.battery_soh:.12f}")
    print(f"Motor 1 SOH:        {result_2.motor1_soh:.12f}")
    print(f"Motor 2 SOH:        {result_2.motor2_soh:.12f}")

    assert result_2.battery_soh < previous_battery_soh
    assert result_2.motor1_soh < previous_motor1_soh
    assert result_2.motor2_soh < previous_motor2_soh

    print("✓ Battery degradation accumulates")
    print("✓ Motor 1 degradation accumulates")
    print("✓ Motor 2 degradation accumulates")

    # ---------------------------------------------------------------
    # TEST 5 — SOH remains physically bounded
    # ---------------------------------------------------------------

    print("\nTEST 5 — SOH bounds")

    for name, value in [
        ("battery_soh", result_2.battery_soh),
        ("motor1_soh", result_2.motor1_soh),
        ("motor2_soh", result_2.motor2_soh),
    ]:
        print(f"{name:15s}: {value:.12f}")
        assert 0.0 <= value <= 1.0

    print("✓ All SOH values remain within [0, 1]")

    # ---------------------------------------------------------------
    # TEST 6 — Motor power and battery power conservation
    # ---------------------------------------------------------------

    print("\nTEST 6 — Electrical power conservation")

    conservation_error = (
        result_2.motor_total_electrical_power_kw
        + result_2.battery_power_kw
    )

    print(
        "P_motor,total + P_battery = "
        f"{conservation_error:.12e} kW"
    )

    assert abs(conservation_error) < 1e-10

    print("✓ Electrical power conservation preserved")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 2D + 2E INTEGRATED HEALTH VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()