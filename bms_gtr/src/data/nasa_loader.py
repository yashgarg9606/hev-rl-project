from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_battery(mat_path):
    mat_path = Path(mat_path)

    data = loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=True
    )

    battery_name = mat_path.stem
    battery = data[battery_name]

    cycles = np.atleast_1d(battery.cycle)

    records = []

    for cycle_index, cycle in enumerate(cycles):
        cycle_type = str(cycle.type)

        record = {
            "cycle_index": cycle_index,
            "type": cycle_type,
            "ambient_temperature": float(cycle.ambient_temperature),
            "time": cycle.time
        }

        if hasattr(cycle, "data"):
            cycle_data = cycle.data

            for field in [
                "Voltage_measured",
                "Current_measured",
                "Temperature_measured",
                "Current_charge",
                "Voltage_charge",
                "Time",
                "Capacity",
                "Sense_current",
                "Battery_current",
                "Current_ratio",
                "Battery_impedance",
                "Rectified_impedance",
                "Re",
                "Rct"
            ]:
                if hasattr(cycle_data, field):
                    value = getattr(cycle_data, field)

                    if isinstance(value, np.ndarray):
                        value = np.asarray(value).squeeze()

                    record[field] = value

        records.append(record)

    return records


def get_discharge_cycles(records):
    return [
        record
        for record in records
        if record["type"].lower() == "discharge"
        and "Capacity" in record
    ]


def get_charge_cycles(records):
    return [
        record
        for record in records
        if record["type"].lower() == "charge"
    ]


def get_impedance_cycles(records):
    return [
        record
        for record in records
        if record["type"].lower() == "impedance"
    ]


def calculate_soh(discharge_cycles):
    capacities = np.array(
        [float(np.asarray(c["Capacity"]).squeeze()) for c in discharge_cycles],
        dtype=np.float64
    )

    valid = np.isfinite(capacities) & (capacities > 0)

    capacities = capacities[valid]

    if len(capacities) == 0:
        raise ValueError("No valid discharge capacities found.")

    bol_capacity = capacities[0]

    soh = capacities / bol_capacity

    return capacities, soh