from battery_model import BatteryElectricalModel, BatteryElectricalParameters, BatteryState


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def main():
    battery = BatteryElectricalModel(
        parameters=BatteryElectricalParameters(
            nominal_capacity_ah=72.0,
            nominal_ocv_v=255.5,
            ohmic_resistance_ohm=0.0031,
        ),
        initial_state=BatteryState(soc=0.6),
    )

    print("BATTERY ELECTRICAL MODEL — VALIDATION")
    print("-" * 55)

    # Published/reference-lineage parameter check at 25 degC, SOC=0.5.
    p_d = battery.electrical_parameters(
        temperature_c=25.0, current_a=-20.0, soc=0.5
    )
    p_c = battery.electrical_parameters(
        temperature_c=25.0, current_a=20.0, soc=0.5
    )

    print("25 C, SOC=0.50, discharge:")
    print(f"  R1 = {p_d['r1_ohm']:.9f} ohm")
    print(f"  R2 = {p_d['r2_ohm']:.9f} ohm")
    print(f"  C1 = {p_d['c1_f']:.6f} F")
    print(f"  C2 = {p_d['c2_f']:.6f} F")

    print("25 C, SOC=0.50, charge:")
    print(f"  R1 = {p_c['r1_ohm']:.9f} ohm")
    print(f"  R2 = {p_c['r2_ohm']:.9f} ohm")
    print(f"  C1 = {p_c['c1_f']:.6f} F")
    print(f"  C2 = {p_c['c2_f']:.6f} F")

    assert p_d["direction"] == "discharge"
    assert p_c["direction"] == "charge"
    assert p_d["r1_ohm"] > 0
    assert p_d["r2_ohm"] > 0
    assert p_d["c1_f"] > 0
    assert p_d["c2_f"] > 0

    # Wu sign convention: negative current discharges.
    v0 = battery.terminal_voltage(current_a=-20.0, temperature_c=25.0)
    p0 = battery.battery_power(current_a=-20.0, temperature_c=25.0)
    print(f"Initial discharge voltage: {v0:.6f} V")
    print(f"Initial discharge power:   {p0/1000:.6f} kW")
    assert p0 < 0

    # One-second SOC update.
    soc0 = battery.state.soc
    battery.update(current_a=-20.0, temperature_c=25.0, dt_s=1.0)
    expected_soc = soc0 - 20.0 / (3600.0 * 72.0)
    approx(battery.state.soc, expected_soc, tol=1e-12)
    print(f"SOC after 1 s at -20 A:    {battery.state.soc:.9f}")

    # Positive current increases SOC.
    soc1 = battery.state.soc
    battery.update(current_a=20.0, temperature_c=25.0, dt_s=1.0)
    assert battery.state.soc > soc1
    print(f"SOC after 1 s at +20 A:    {battery.state.soc:.9f}")

    # Power -> current inversion.
    target_power = -5000.0
    current = battery.current_from_battery_power(
        battery_power_w=target_power,
        temperature_c=25.0,
    )
    recovered_power = battery.battery_power(
        current_a=current,
        temperature_c=25.0,
    )
    print(f"5 kW discharge current:    {current:.6f} A")
    print(f"Recovered battery power:    {recovered_power:.6f} W")
    approx(recovered_power, target_power, tol=1e-6)

    # Temperature bounds are deliberate: do not silently extrapolate.
    try:
        battery.electrical_parameters(temperature_c=10.0, current_a=-10.0)
    except ValueError:
        print("✓ Out-of-range temperature rejected")
    else:
        raise AssertionError("Temperature extrapolation was not rejected.")

    print("-" * 55)
    print("✓ BATTERY ELECTRICAL MODEL VALIDATION PASSED")


if __name__ == "__main__":
    main()
