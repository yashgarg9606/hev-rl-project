from pathlib import Path

from integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOTOR1_MAP, MOTOR2_MAP = default_motor_map_paths(
    PROJECT_ROOT
)


def main():
    plant = IntegratedPowertrain(
        motor1_map_path=MOTOR1_MAP,
        motor2_map_path=MOTOR2_MAP,
        parameters=IntegratedPowertrainParameters(
            initial_soc=0.6,
            battery_temperature_c=25.0,
        ),
    )

    print("=" * 75)
    print("PHASE 2D — INTEGRATED POWERTRAIN VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------------
    # Test 1: steady cruising
    # --------------------------------------------------------------

    print("\nTEST 1 — 50 km/h steady cruising")
    print("-" * 75)

    result = plant.step(
        velocity_kmh=50.0,
        target_velocity_kmh=50.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Wheel torque:             {result.wheel_torque_nm:.6f} Nm")
    print(
        f"M1 torque / speed:        "
        f"{result.motor1_torque_nm:.6f} Nm / "
        f"{result.motor1_speed_rpm:.3f} rpm"
    )
    print(
        f"M2 torque / speed:        "
        f"{result.motor2_torque_nm:.6f} Nm / "
        f"{result.motor2_speed_rpm:.3f} rpm"
    )
    print(
        f"M1 efficiency / power:    "
        f"{result.motor1_efficiency_percent:.3f}% / "
        f"{result.motor1_electrical_power_kw:.6f} kW"
    )
    print(
        f"M2 efficiency / power:    "
        f"{result.motor2_efficiency_percent:.3f}% / "
        f"{result.motor2_electrical_power_kw:.6f} kW"
    )
    print(
        f"Total motor power:        "
        f"{result.motor_total_electrical_power_kw:.6f} kW"
    )
    print(
        f"Battery power:            "
        f"{result.battery_power_kw:.6f} kW"
    )
    print(
        f"Battery current:          "
        f"{result.battery_current_a:.6f} A"
    )
    print(
        f"Battery voltage:           "
        f"{result.battery_terminal_voltage_v:.6f} V"
    )
    print(f"SOC after step:            {result.soc:.9f}")

    assert result.overall_feasible
    assert result.motor_total_electrical_power_kw > 0.0
    assert result.battery_power_kw < 0.0
    assert result.battery_current_a < 0.0
    assert result.soc < 0.6

    # --------------------------------------------------------------
    # Test 2: acceleration
    # --------------------------------------------------------------

    print("\nTEST 2 — 50 -> 60 km/h acceleration")
    print("-" * 75)

    result = plant.step(
        velocity_kmh=50.0,
        target_velocity_kmh=60.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Acceleration:             {result.acceleration_mps2:.6f} m/s^2")
    print(f"Wheel torque:             {result.wheel_torque_nm:.6f} Nm")
    print(f"Total motor power:        {result.motor_total_electrical_power_kw:.6f} kW")
    print(f"Battery power:            {result.battery_power_kw:.6f} kW")
    print(f"Battery current:          {result.battery_current_a:.6f} A")
    print(f"SOC:                      {result.soc:.9f}")

    assert result.overall_feasible
    assert result.wheel_torque_nm > 0.0
    assert result.battery_power_kw < 0.0
    assert result.battery_current_a < 0.0

    # --------------------------------------------------------------
    # Test 3: energy conservation
    # --------------------------------------------------------------

    print("\nTEST 3 — Electrical power sign/conservation")
    print("-" * 75)

    error_kw = (
        result.motor_total_electrical_power_kw
        + result.battery_power_kw
    )

    print(
        f"P_motor,total + P_battery = {error_kw:.12e} kW"
    )

    assert abs(error_kw) < 1e-12

    # --------------------------------------------------------------
    # Test 4: regeneration
    # --------------------------------------------------------------

    print("\nTEST 4 — 60 -> 50 km/h regenerative braking")
    print("-" * 75)

    result = plant.step(
        velocity_kmh=60.0,
        target_velocity_kmh=50.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Acceleration:             {result.acceleration_mps2:.6f} m/s^2")
    print(f"Wheel torque:             {result.wheel_torque_nm:.6f} Nm")
    print(f"Total motor power:        {result.motor_total_electrical_power_kw:.6f} kW")
    print(f"Battery power:            {result.battery_power_kw:.6f} kW")
    print(f"Battery current:          {result.battery_current_a:.6f} A")
    print(f"SOC:                      {result.soc:.9f}")

    assert result.overall_feasible
    assert result.wheel_torque_nm < 0.0
    assert result.motor_total_electrical_power_kw < 0.0
    assert result.battery_power_kw > 0.0
    assert result.battery_current_a > 0.0

    print("\n" + "=" * 75)
    print("✓ PHASE 2D INTEGRATED POWERTRAIN VALIDATION PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()
