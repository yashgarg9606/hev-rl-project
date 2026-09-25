"""
Diagnose motor efficiency map interpolation failures during Phase 3B experiments.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path

from environment.motor_efficiency_map import MotorEfficiencyMap
from environment.motor_model import DualMotorModel

def check_map_coverage():
    """Check the actual coverage of the motor efficiency maps."""
    print("="*60)
    print("MOTOR EFFICIENCY MAP COVERAGE ANALYSIS")
    print("="*60)
    print()

    # Load the contour data
    motor1_df = pd.read_csv('data/processed/motor_maps/motor1_efficiency_contours.csv')
    motor2_df = pd.read_csv('data/processed/motor_maps/motor2_efficiency_contours.csv')

    for name, df in [('Motor 1', motor1_df), ('Motor 2', motor2_df)]:
        print(f"{name}:")
        print(f"  Speed range: {df['speed_rpm'].min():.1f} - {df['speed_rpm'].max():.1f} rpm")
        print(f"  Torque range: {df['torque_nm'].min():.1f} - {df['torque_nm'].max():.1f} Nm")
        print(f"  Efficiency range: {df['efficiency_percent'].min():.1f} - {df['efficiency_percent'].max():.1f} %")
        print(f"  Data points: {len(df)}")

        # Check for zero-speed points
        zero_speed = df[df['speed_rpm'] < 1.0]
        print(f"  Points with speed < 1 rpm: {len(zero_speed)}")
        if len(zero_speed) > 0:
            print(f"    Min torque at near-zero speed: {zero_speed['torque_nm'].min():.2f} Nm")

        # Check high-speed coverage
        high_speed = df[df['speed_rpm'] > 10000]
        print(f"  Points with speed > 10000 rpm: {len(high_speed)}")
        if len(high_speed) > 0:
            print(f"    Torque range at high speed: {high_speed['torque_nm'].min():.2f} - {high_speed['torque_nm'].max():.2f} Nm")

        print()

def test_zero_torque_operation():
    """Test whether zero-torque operation (coasting) is supported."""
    print("="*60)
    print("ZERO-TORQUE OPERATION TEST")
    print("="*60)
    print()

    motor1_map = MotorEfficiencyMap('data/processed/motor_maps/motor1_efficiency_contours.csv')
    motor2_map = MotorEfficiencyMap('data/processed/motor_maps/motor2_efficiency_contours.csv')

    test_speeds = [50, 500, 1000, 5000, 10000, 11000]

    print("Testing zero-torque (coasting) at various speeds:")
    print()

    for speed in test_speeds:
        try:
            eff1 = motor1_map.get_efficiency(speed, 0.0)
            eff2 = motor2_map.get_efficiency(speed, 0.0)
            print(f"  {speed:5.0f} rpm, 0 Nm: Motor1={eff1:.1f}%, Motor2={eff2:.1f}% [OK]")
        except ValueError as e:
            print(f"  {speed:5.0f} rpm, 0 Nm: FAIL - outside map")
    print()

def test_typical_driving_cycle():
    """Test motor operating points for a typical driving cycle."""
    print("="*60)
    print("TYPICAL DRIVING CYCLE OPERATING POINTS")
    print("="*60)
    print()

    motor1_map = MotorEfficiencyMap('data/processed/motor_maps/motor1_efficiency_contours.csv')
    motor2_map = MotorEfficiencyMap('data/processed/motor_maps/motor2_efficiency_contours.csv')

    motor_model = DualMotorModel()

    # Typical UDDS-like cycle points (velocity in km/h, wheel torque demand in Nm)
    cycle_points = [
        (0, 0, "Idle / stopped"),
        (0, 50, "Launch from standstill"),
        (10, 100, "Low-speed acceleration"),
        (30, 80, "Urban driving"),
        (50, 60, "Suburban driving"),
        (70, 40, "Highway cruise"),
        (90, 30, "High-speed cruise"),
        (50, -20, "Regenerative braking"),
    ]

    failures = []

    print(f"Testing {len(cycle_points)} typical operating points:")
    print()

    for v_kmh, wheel_torque_nm, desc in cycle_points:
        # Test with equal torque split (sigma = 0.5)
        sigma_tor = 0.5

        # Calculate motor speeds
        motor1_speed = motor_model.calculate_motor_speed(v_kmh, motor_model.params.motor1.gear_ratio)
        motor2_speed = motor_model.calculate_motor_speed(v_kmh, motor_model.params.motor2.gear_ratio)

        # Calculate motor torques
        torque_dist = motor_model.distribute_torque(wheel_torque_nm, sigma_tor)
        motor1_torque = torque_dist['motor1_torque_nm']
        motor2_torque = torque_dist['motor2_torque_nm']

        # Test efficiency lookup
        status = []

        try:
            if motor1_torque >= 0:
                eff1 = motor1_map.get_efficiency(motor1_speed, motor1_torque)
                status.append(f"M1: OK ({eff1:.1f}%)")
            else:
                status.append("M1: SKIP (regen)")
        except ValueError:
            status.append("M1: FAIL")
            failures.append((desc, 1, motor1_speed, motor1_torque))

        try:
            if motor2_torque >= 0:
                eff2 = motor2_map.get_efficiency(motor2_speed, motor2_torque)
                status.append(f"M2: OK ({eff2:.1f}%)")
            else:
                status.append("M2: SKIP (regen)")
        except ValueError:
            status.append("M2: FAIL")
            failures.append((desc, 2, motor2_speed, motor2_torque))

        status_str = ", ".join(status)
        print(f"{desc:25s} ({v_kmh:3.0f} km/h, {wheel_torque_nm:4.0f} Nm)")
        print(f"  Motor 1: {motor1_speed:6.1f} rpm, {motor1_torque:6.2f} Nm")
        print(f"  Motor 2: {motor2_speed:6.1f} rpm, {motor2_torque:6.2f} Nm")
        print(f"  Status: {status_str}")
        print()

    if failures:
        print("="*60)
        print(f"SUMMARY: {len(failures)} FAILURES DETECTED")
        print("="*60)
        print()
        for desc, motor_idx, speed, torque in failures:
            print(f"  {desc}: Motor {motor_idx} at {speed:.1f} rpm, {torque:.2f} Nm")
        print()
    else:
        print("="*60)
        print("SUMMARY: ALL TESTS PASSED")
        print("="*60)
        print()

    return failures

def main():
    print("\n")
    print("PHASE 3B MOTOR EFFICIENCY MAP DIAGNOSTIC")
    print("="*60)
    print()

    check_map_coverage()
    test_zero_torque_operation()
    failures = test_typical_driving_cycle()

    print("="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

    if failures:
        print(f"\nFound {len(failures)} operating points outside the map.")
        print("These points need to be handled before running Phase 3B experiments.")
    else:
        print("\nAll typical operating points are within the map coverage.")

if __name__ == '__main__':
    main()
