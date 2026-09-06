from vehicle_dynamics import VehicleDynamics


def run_case(vehicle, name, current_speed, target_speed):
    result = vehicle.calculate_required_force(
        velocity_kmh=current_speed,
        target_velocity_kmh=target_speed,
        slope_rad=0.0,
        dt=1.0,
    )

    torque = vehicle.calculate_wheel_torque(
        velocity_kmh=current_speed,
        target_velocity_kmh=target_speed,
        slope_rad=0.0,
        dt=1.0,
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Speed              : {current_speed:.1f} -> {target_speed:.1f} km/h")
    print(f"Acceleration       : {result['acceleration_mps2']:.6f} m/s²")
    print(f"Inertial force     : {result['force_inertia_n']:.6f} N")
    print(f"Rolling force      : {result['force_rolling_n']:.6f} N")
    print(f"Aerodynamic force  : {result['force_aero_n']:.6f} N")
    print(f"Total force        : {result['force_total_n']:.6f} N")
    print(f"Wheel torque       : {torque:.6f} Nm")

    return result, torque


def main():
    vehicle = VehicleDynamics()

    print("\n" + "=" * 60)
    print("VEHICLE DYNAMICS — THREE-CASE VALIDATION")
    print("=" * 60)

    # Case A: constant speed
    constant, torque_constant = run_case(
        vehicle,
        "CASE A — CONSTANT SPEED",
        50.0,
        50.0,
    )

    # Case B: acceleration
    accel, torque_accel = run_case(
        vehicle,
        "CASE B — ACCELERATION",
        50.0,
        60.0,
    )

    # Case C: braking
    braking, torque_braking = run_case(
        vehicle,
        "CASE C — BRAKING",
        60.0,
        50.0,
    )

    print("\n" + "=" * 60)
    print("AUTOMATED CHECKS")
    print("=" * 60)

    # Constant speed
    assert abs(constant["acceleration_mps2"]) < 1e-12
    assert torque_constant > 0
    print("✓ Constant speed: acceleration = 0 and road-load torque > 0")

    # Acceleration
    assert accel["acceleration_mps2"] > 0
    assert torque_accel > torque_constant
    print("✓ Acceleration: positive acceleration and increased torque demand")

    # Braking
    assert braking["acceleration_mps2"] < 0
    assert torque_braking < torque_constant
    print("✓ Braking: negative acceleration and reduced torque demand")

    if torque_braking < 0:
        print("✓ Braking torque is negative -> regenerative braking is possible")
    else:
        print("ℹ Braking torque remains positive because road loads exceed deceleration demand")

    print("\n🎉 ALL VEHICLE DYNAMICS TESTS PASSED")


if __name__ == "__main__":
    main()