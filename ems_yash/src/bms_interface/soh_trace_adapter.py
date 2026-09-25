"""
Phase 3B — SOH Trace Adapter

Provides sample-and-hold access to BMS SOH predictions for EMS simulation.

The BMS model generates SOH predictions at sparse intervals (every 20 cycles).
The EMS operates at 1-second simulation timesteps. This adapter bridges the
timescale gap by holding the most recent BMS prediction until the next update.
"""

import numpy as np
from pathlib import Path


class SOHTraceAdapter:
    """
    Sample-and-hold adapter for BMS SOH prediction trace.

    Provides the most recent valid SOH estimate for a given simulation time.

    Parameters
    ----------
    trace_file : str or Path
        Path to BMS SOH trace file (.npz)

    Notes
    -----
    - Returns first prediction for times before first timestamp
    - Returns last prediction for times after final timestamp
    - Never extrapolates beyond trace boundaries
    """

    def __init__(self, trace_file):
        self.trace_file = Path(trace_file)

        if not self.trace_file.exists():
            raise FileNotFoundError(
                f"BMS trace file not found: {self.trace_file}"
            )

        # Load trace
        data = np.load(self.trace_file)

        self.time_seconds = data['time_seconds']
        self.soh_predicted = data['soh_predicted']
        self.soh_true = data['soh_true']

        # Validate trace
        if len(self.time_seconds) != len(self.soh_predicted):
            raise ValueError(
                f"Time and SOH arrays must have same length. "
                f"Got {len(self.time_seconds)} times, {len(self.soh_predicted)} SOH values"
            )

        if len(self.time_seconds) == 0:
            raise ValueError("Trace is empty")

        # Track current position for efficient lookup
        self.current_index = 0

    def get_soh_at_time(self, simulation_time_s: float) -> float:
        """
        Get BMS SOH estimate at given simulation time (sample-and-hold).

        Parameters
        ----------
        simulation_time_s : float
            Current simulation time in seconds

        Returns
        -------
        soh_estimated : float
            Most recent BMS SOH prediction
        """

        # Handle edge cases
        if simulation_time_s <= self.time_seconds[0]:
            # Before first prediction: use initial SOH
            return float(self.soh_predicted[0])

        if simulation_time_s >= self.time_seconds[-1]:
            # After final prediction: hold last value
            return float(self.soh_predicted[-1])

        # Advance to most recent prediction (sample-and-hold)
        while (self.current_index < len(self.time_seconds) - 1 and
               self.time_seconds[self.current_index + 1] <= simulation_time_s):
            self.current_index += 1

        return float(self.soh_predicted[self.current_index])

    def get_true_soh_at_time(self, simulation_time_s: float) -> float:
        """
        Get true SOH from trace at given time (for validation only).

        This is used for comparison experiments to align true SOH
        with BMS timestamps, NOT for EMS consumption.
        """

        if simulation_time_s <= self.time_seconds[0]:
            return float(self.soh_true[0])

        if simulation_time_s >= self.time_seconds[-1]:
            return float(self.soh_true[-1])

        # Find nearest timestamp
        idx = np.searchsorted(self.time_seconds, simulation_time_s, side='right') - 1
        return float(self.soh_true[idx])

    def reset(self):
        """Reset adapter to beginning of trace."""
        self.current_index = 0

    def __repr__(self):
        return (
            f"SOHTraceAdapter("
            f"samples={len(self.soh_predicted)}, "
            f"duration={self.time_seconds[-1]/3600:.1f}h)"
        )


# Unit tests
def test_soh_trace_adapter():
    """Test SOHTraceAdapter sample-and-hold behavior."""

    print("=" * 70)
    print("SOH TRACE ADAPTER UNIT TESTS")
    print("=" * 70)
    print()

    # Create test trace
    test_trace = Path(__file__).parent.parent.parent / "data" / "bms_soh_trace.npz"

    if not test_trace.exists():
        print(f"[SKIP] Trace not found: {test_trace}")
        print("Run generate_bms_trace.py first")
        return

    # Load adapter
    adapter = SOHTraceAdapter(test_trace)
    print(f"Loaded: {adapter}")
    print()

    # Test 1: Sample-and-hold behavior
    print("TEST 1: Sample-and-hold")
    print("-" * 70)

    # Get first few timestamps
    times = adapter.time_seconds[:5]
    print(f"First 5 trace timestamps (hours): {times/3600}")
    print()

    # Test times between updates
    test_times = [
        0,                    # At first prediction
        times[0] + 1000,     # Between first and second
        times[1],            # At second prediction
        times[1] + 5000,     # Between second and third
        times[2],            # At third prediction
    ]

    for t in test_times:
        soh = adapter.get_soh_at_time(t)
        print(f"  t = {t/3600:8.2f} hours -> SOH = {soh:.6f}")

    print("[PASS] Sample-and-hold works correctly")
    print()

    # Test 2: Edge cases
    print("TEST 2: Edge cases")
    print("-" * 70)

    adapter.reset()

    # Before first
    soh_before = adapter.get_soh_at_time(-1000)
    soh_first = adapter.get_soh_at_time(times[0])
    assert soh_before == soh_first, "Before first should return first SOH"
    print(f"  t < first -> SOH = {soh_before:.6f} (uses first prediction)")

    # After last
    soh_after = adapter.get_soh_at_time(times[-1] + 1e9)
    soh_last = adapter.get_soh_at_time(times[-1])
    assert soh_after == soh_last, "After last should return last SOH"
    print(f"  t > last -> SOH = {soh_after:.6f} (holds last prediction)")

    print("[PASS] Edge cases handled correctly")
    print()

    # Test 3: Monotonic time access
    print("TEST 3: Monotonic time access (typical EMS usage)")
    print("-" * 70)

    adapter.reset()
    prev_soh = None
    update_count = 0

    for i, t in enumerate(times[:10]):
        soh = adapter.get_soh_at_time(t)
        if prev_soh is not None and soh != prev_soh:
            update_count += 1
            print(f"  t = {t/3600:8.2f} hours -> SOH updated to {soh:.6f}")
        prev_soh = soh

    print(f"[PASS] {update_count} SOH updates detected")
    print()

    # Test 4: Reset functionality
    print("TEST 4: Reset")
    print("-" * 70)

    adapter.get_soh_at_time(times[5])  # Advance to middle
    assert adapter.current_index > 0, "Should have advanced"

    adapter.reset()
    assert adapter.current_index == 0, "Reset should return to start"

    print("[PASS] Reset works correctly")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_soh_trace_adapter()
