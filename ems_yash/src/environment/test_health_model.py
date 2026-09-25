"""
Phase 2E — Health / Degradation Model Validation
"""

import math

from health_model import (
    BatteryHealthParameters,
    HealthDegradationModel,
    MotorHealthParameters,
)


def main():

    print("=" * 75)
    print("PHASE 2E — HEALTH / DEGRADATION MODEL VALIDATION")
    print("=" * 75)

    model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(
            nominal_capacity_ah=72.0,
        ),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=28.0,
            rated_efficiency=0.98,
            life_hours=30_000.0,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=24.0,
            rated_efficiency=0.98,
            life_hours=30_000.0,
        ),
    )

    # --------------------------------------------------------------
    # 1. Initial health
    # --------------------------------------------------------------

    print("\nINITIAL HEALTH")
    print("-" * 75)

    health = model.get_health_state()

    print(f"Battery SOH:             {health['battery_soh']:.9f}")
    print(f"Motor 1 SOH:             {health['motor1_soh']:.9f}")
    print(f"Motor 2 SOH:             {health['motor2_soh']:.9f}")

    assert health["battery_soh"] == 1.0
    assert health["motor1_soh"] == 1.0
    assert health["motor2_soh"] == 1.0

    print("✓ Initial SOH values correct")

    # --------------------------------------------------------------
    # 2. Battery aging parameters
    # --------------------------------------------------------------

    print("\nBATTERY AGING PARAMETERS")
    print("-" * 75)

    c_rate = 2.0
    temperature_c = 25.0

    b = model.b_factor(c_rate)
    ea = model.activation_energy(c_rate)
    ah_eol = model.end_of_life_amp_hours(
        c_rate=c_rate,
        temperature_c=temperature_c,
    )
    n_eol = model.eol_cycles(
        c_rate=c_rate,
        temperature_c=temperature_c,
    )

    print(f"C-rate:                  {c_rate:.3f} C")
    print(f"B(c):                    {b:.3f}")
    print(f"Ea(c):                   {ea:.3f} J/mol")
    print(f"Ah to EOL:               {ah_eol:.3f} Ah")
    print(f"Cycles to EOL:            {n_eol:.3f}")

    assert math.isclose(
        b,
        21681.0,
        rel_tol=1e-12,
    )

    assert math.isclose(
        ea,
        30959.4,
        rel_tol=1e-12,
    )

    assert ah_eol > 0.0
    assert n_eol > 0.0

    print("✓ Battery aging parameters validated")

    # --------------------------------------------------------------
    # 3. Zero-current battery aging
    # --------------------------------------------------------------

    print("\nZERO-CURRENT BATTERY TEST")
    print("-" * 75)

    delta_soh = model.battery_soh_change(
        current_a=0.0,
        temperature_c=25.0,
        dt_s=1.0,
    )

    assert delta_soh == 0.0

    print(f"SOH change at zero current: {delta_soh:.12e}")
    print("✓ Zero-current aging correctly zero")

    # --------------------------------------------------------------
    # 4. Charge/discharge symmetry
    # --------------------------------------------------------------

    print("\nCHARGE / DISCHARGE SYMMETRY")
    print("-" * 75)

    discharge_delta = model.battery_soh_change(
        current_a=-20.0,
        temperature_c=25.0,
        dt_s=1.0,
    )

    charge_delta = model.battery_soh_change(
        current_a=20.0,
        temperature_c=25.0,
        dt_s=1.0,
    )

    print(
        f"Discharge ΔSOH:         "
        f"{discharge_delta:.12e}"
    )

    print(
        f"Charge ΔSOH:            "
        f"{charge_delta:.12e}"
    )

    assert discharge_delta < 0.0
    assert charge_delta < 0.0

    assert math.isclose(
        discharge_delta,
        charge_delta,
        rel_tol=1e-12,
    )

    print("✓ Charge/discharge aging uses |I| correctly")

    # --------------------------------------------------------------
    # 5. Battery update
    # --------------------------------------------------------------

    print("\nBATTERY UPDATE")
    print("-" * 75)

    old_soh = model.battery_state.soh

    delta = model.update_battery(
        current_a=-20.0,
        temperature_c=25.0,
        dt_s=1.0,
    )

    new_soh = model.battery_state.soh

    print(f"SOH before:              {old_soh:.12f}")
    print(f"ΔSOH:                    {delta:.12e}")
    print(f"SOH after:               {new_soh:.12f}")
    print(
        "Cumulative Ah:           "
        f"{model.battery_state.cumulative_ah_throughput:.9f}"
    )

    assert new_soh < old_soh
    assert (
        model.battery_state.cumulative_ah_throughput
        > 0.0
    )

    print("✓ Battery SOH update correct")

    # --------------------------------------------------------------
    # 6. Motor lifetime loss
    # --------------------------------------------------------------

    print("\nMOTOR LIFETIME LOSS")
    print("-" * 75)

    m1_lifetime_loss = (
        model.motor_lifetime_energy_loss_kwh(
            model.motor1_parameters
        )
    )

    m2_lifetime_loss = (
        model.motor_lifetime_energy_loss_kwh(
            model.motor2_parameters
        )
    )

    print(
        f"Motor 1 lifetime loss:  "
        f"{m1_lifetime_loss:.6f} kWh"
    )

    print(
        f"Motor 2 lifetime loss:  "
        f"{m2_lifetime_loss:.6f} kWh"
    )

    assert math.isclose(
        m1_lifetime_loss,
        (
            (1.0 / 0.98 - 1.0)
            * 28.0
            * 30_000.0
        ),
        rel_tol=1e-12,
    )

    assert math.isclose(
        m2_lifetime_loss,
        (
            (1.0 / 0.98 - 1.0)
            * 24.0
            * 30_000.0
        ),
        rel_tol=1e-12,
    )

    print("✓ Motor lifetime loss equations validated")

    # --------------------------------------------------------------
    # 7. Motor one-step degradation
    # --------------------------------------------------------------

    print("\nMOTOR ONE-STEP DEGRADATION")
    print("-" * 75)

    m1_delta = model.update_motor1(
        mechanical_power_kw=10.0,
        efficiency=0.90,
        dt_s=1.0,
    )

    m2_delta = model.update_motor2(
        mechanical_power_kw=10.0,
        efficiency=0.90,
        dt_s=1.0,
    )

    print(
        f"Motor 1 ΔSOH:           "
        f"{m1_delta:.12e}"
    )

    print(
        f"Motor 2 ΔSOH:           "
        f"{m2_delta:.12e}"
    )

    assert m1_delta < 0.0
    assert m2_delta < 0.0

    assert (
        model.motor1_state.soh < 1.0
    )

    assert (
        model.motor2_state.soh < 1.0
    )

    print("✓ Motor SOH decreases with energy loss")

    # --------------------------------------------------------------
    # 8. Zero motor power
    # --------------------------------------------------------------

    print("\nZERO MOTOR POWER")
    print("-" * 75)

    zero_loss = model.motor_step_energy_loss_kwh(
        mechanical_power_kw=0.0,
        efficiency=0.90,
        dt_s=1.0,
    )

    assert zero_loss == 0.0

    print(
        f"Zero-power loss:        "
        f"{zero_loss:.12e} kWh"
    )

    print("✓ Zero motor power causes zero degradation")

    # --------------------------------------------------------------
    # 9. Health bounds
    # --------------------------------------------------------------

    print("\nHEALTH BOUNDS")
    print("-" * 75)

    health = model.get_health_state()

    for name, value in health.items():

        print(
            f"{name:24s}: {value:.12f}"
        )

        assert 0.0 <= value <= 1.0

    print("✓ All SOH values remain physically bounded")

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n" + "=" * 75)
    print("✓ PHASE 2E HEALTH / DEGRADATION VALIDATION PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()