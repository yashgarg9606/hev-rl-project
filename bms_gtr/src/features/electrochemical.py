import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


V_MIN = 2.5
V_MAX = 4.2
N_POINTS = 300
SG_WINDOW = 15
SG_POLYORDER = 3


def smooth_signal(signal):
    signal = np.asarray(signal, dtype=np.float64).squeeze()

    if len(signal) < SG_WINDOW:
        return signal

    return savgol_filter(
        signal,
        SG_WINDOW,
        SG_POLYORDER
    )


def interpolate_voltage_domain(voltage, signal):
    voltage = np.asarray(voltage, dtype=np.float64).squeeze()
    signal = np.asarray(signal, dtype=np.float64).squeeze()

    mask = np.isfinite(voltage) & np.isfinite(signal)

    voltage = voltage[mask]
    signal = signal[mask]

    if len(voltage) < 2:
        raise ValueError("Not enough points for interpolation.")

    order = np.argsort(voltage)

    voltage = voltage[order]
    signal = signal[order]

    voltage, unique_indices = np.unique(
        voltage,
        return_index=True
    )

    signal = signal[unique_indices]

    grid = np.linspace(
        V_MIN,
        V_MAX,
        N_POINTS
    )

    valid = (
        (grid >= voltage.min())
        & (grid <= voltage.max())
    )

    if valid.sum() < 2:
        raise ValueError("Voltage trace does not cover enough of the target grid.")

    interpolator = interp1d(
        voltage,
        signal,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    result = interpolator(grid)

    return grid, result


def calculate_derivative(y, x):
    return np.gradient(y, x)


def extract_features(discharge_cycle):
    voltage = np.asarray(
        discharge_cycle["Voltage_measured"],
        dtype=np.float64
    ).squeeze()

    current = np.asarray(
        discharge_cycle["Current_measured"],
        dtype=np.float64
    ).squeeze()

    time = np.asarray(
        discharge_cycle["Time"],
        dtype=np.float64
    ).squeeze()

    voltage_smooth = smooth_signal(voltage)
    current_smooth = smooth_signal(current)

    dt = np.gradient(time)

    dt[dt == 0] = np.nan

    capacity_from_current = np.cumsum(
        np.nan_to_num(current_smooth) * dt
    ) / 3600.0

    _, current_interp = interpolate_voltage_domain(
        voltage_smooth,
        current_smooth
    )

    _, capacity_interp = interpolate_voltage_domain(
        voltage_smooth,
        capacity_from_current
    )

    voltage_grid = np.linspace(
        V_MIN,
        V_MAX,
        N_POINTS
    )

    dv_dq = calculate_derivative(
        voltage_grid,
        capacity_interp
    )

    dq_dv = calculate_derivative(
        capacity_interp,
        voltage_grid
    )

    di_dv = calculate_derivative(
        current_interp,
        voltage_grid
    )

    features = np.stack(
        [
            dq_dv,
            dv_dq,
            di_dv
        ],
        axis=0
    )

    return voltage_grid, features