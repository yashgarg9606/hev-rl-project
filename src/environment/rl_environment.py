"""
Phase 3D — Gymnasium RL Environment

Wraps the validated integrated powertrain as a Gymnasium-compatible
reinforcement-learning environment.

State:
    [velocity_kmh, wheel_torque_nm, SOC, battery_SOH,
     motor1_SOH, motor2_SOH]

Action:
    [sigma_tor]

where sigma_tor is the fraction of wheel torque assigned to Motor 1.

Important:
    The target Wu et al. paper defines the RL state as:
        [v, Td, SOC, SOH, SOHM1, SOHM2]

    Here:
        v      -> velocity_kmh
        Td     -> wheel_torque_nm

Reward is intentionally not implemented in this phase.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)


class EnergyManagementEnv(gym.Env):
    """
    Gymnasium environment for dual-motor BEV energy management.

    One environment step corresponds to one driving-cycle timestep.

    Observation:
        [velocity_kmh,
         wheel_torque_nm,
         SOC,
         battery_SOH,
         motor1_SOH,
         motor2_SOH]

    Action:
        [sigma_tor], bounded to [0, 1].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cycle: np.ndarray,
        project_root: str | Path | None = None,
        powertrain_parameters: IntegratedPowertrainParameters | None = None,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # Validate driving-cycle input
        # --------------------------------------------------------------

        cycle = np.asarray(cycle, dtype=np.float64)

        if cycle.ndim != 2:
            raise ValueError(
                "Driving cycle must be a 2-D array."
            )

        if cycle.shape[1] != 3:
            raise ValueError(
                "Driving cycle must have 3 columns: "
                "[time_s, velocity_kmh, slope_rad]."
            )

        if cycle.shape[0] < 2:
            raise ValueError(
                "Driving cycle must contain at least 2 timesteps."
            )

        if not np.all(np.isfinite(cycle)):
            raise ValueError(
                "Driving cycle contains non-finite values."
            )

        time_s = cycle[:, 0]

        if not np.all(np.diff(time_s) > 0):
            raise ValueError(
                "Driving-cycle time values must be strictly increasing."
            )

        self.cycle = cycle

        # --------------------------------------------------------------
        # Project root and powertrain
        # --------------------------------------------------------------

        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        project_root = Path(project_root)

        motor1_map, motor2_map = default_motor_map_paths(
            project_root
        )

        self.powertrain_parameters = (
            powertrain_parameters
            or IntegratedPowertrainParameters()
        )

        self.powertrain = IntegratedPowertrain(
            motor1_map_path=motor1_map,
            motor2_map_path=motor2_map,
            parameters=self.powertrain_parameters,
        )

        # --------------------------------------------------------------
        # Environment dimensions
        # --------------------------------------------------------------

        self.state_dim = 6
        self.action_dim = 1

        # --------------------------------------------------------------
        # Action space
        # --------------------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Observation space
        #
        # Physical bounds are intentionally broad for velocity and
        # torque. SOC and SOH are physically bounded to [0, 1].
        # --------------------------------------------------------------

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,       # velocity [km/h]
                    -5000.0,   # wheel torque [Nm]
                    0.0,       # battery SOC
                    0.0,       # battery SOH
                    0.0,       # motor 1 SOH
                    0.0,       # motor 2 SOH
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    200.0,     # velocity [km/h]
                    5000.0,    # wheel torque [Nm]
                    1.0,       # battery SOC
                    1.0,       # battery SOH
                    1.0,       # motor 1 SOH
                    1.0,       # motor 2 SOH
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------------

        self.current_index = 0
        self.current_velocity_kmh = 0.0
        self.current_state = np.zeros(
            self.state_dim,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(
        self,
        velocity_kmh: float,
        wheel_torque_nm: float,
        soc: float,
        battery_soh: float,
        motor1_soh: float,
        motor2_soh: float,
    ) -> np.ndarray:
        """
        Construct the six-dimensional RL observation.
        """

        state = np.array(
            [
                velocity_kmh,
                wheel_torque_nm,
                soc,
                battery_soh,
                motor1_soh,
                motor2_soh,
            ],
            dtype=np.float32,
        )

        if not self.observation_space.contains(state):
            raise ValueError(
                f"Constructed state is outside observation space: {state}"
            )

        return state

    # ------------------------------------------------------------------
    # Gymnasium reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """
        Reset the environment to the beginning of the driving cycle.
        """

        super().reset(seed=seed)

        # Reset physical powertrain state.
        self.powertrain = IntegratedPowertrain(
            motor1_map_path=default_motor_map_paths(
                Path(__file__).resolve().parents[2]
            )[0],
            motor2_map_path=default_motor_map_paths(
                Path(__file__).resolve().parents[2]
            )[1],
            parameters=self.powertrain_parameters,
        )

        self.current_index = 0

        velocity_kmh = float(self.cycle[0, 1])

        # At reset we need the wheel torque associated with the first
        # transition of the driving cycle. This makes Td available in
        # the initial state exactly as required by the paper.
        target_velocity_kmh = float(self.cycle[1, 1])
        slope_rad = float(self.cycle[0, 2])

        vehicle_result = self.powertrain.vehicle.calculate_required_force(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=float(self.cycle[1, 0] - self.cycle[0, 0]),
        )

        wheel_torque_nm = self.powertrain.vehicle.calculate_wheel_torque(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=float(self.cycle[1, 0] - self.cycle[0, 0]),
        )

        health = self.powertrain.health_model.get_health_state()

        self.current_velocity_kmh = velocity_kmh

        self.current_state = self._build_state(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            soc=self.powertrain.battery.state.soc,
            battery_soh=health["battery_soh"],
            motor1_soh=health["motor1_soh"],
            motor2_soh=health["motor2_soh"],
        )

        info = {
            "cycle_index": self.current_index,
            "time_s": float(self.cycle[0, 0]),
            "velocity_kmh": velocity_kmh,
            "wheel_torque_nm": wheel_torque_nm,
        }

        return self.current_state.copy(), info

    # ------------------------------------------------------------------
    # Gymnasium step
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Advance the environment by one driving-cycle timestep.

        Reward is temporarily set to zero in Phase 3D.2.
        """

        # --------------------------------------------------------------
        # Validate action
        # --------------------------------------------------------------

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (1,):
            raise ValueError(
                f"Action must have shape (1,), got {action.shape}."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                "Action contains non-finite values."
            )

        if not self.action_space.contains(action):
            raise ValueError(
                f"Action outside [0, 1]: {action}"
            )

        sigma_tor = float(action[0])

        # --------------------------------------------------------------
        # Check episode boundary
        # --------------------------------------------------------------

        if self.current_index >= len(self.cycle) - 1:
            raise RuntimeError(
                "Episode has already terminated. Call reset()."
            )

        # --------------------------------------------------------------
        # Current and target driving-cycle point
        # --------------------------------------------------------------

        current_index = self.current_index
        next_index = current_index + 1

        velocity_kmh = float(self.cycle[current_index, 1])
        target_velocity_kmh = float(self.cycle[next_index, 1])

        slope_rad = float(self.cycle[current_index, 2])

        dt_s = float(
            self.cycle[next_index, 0]
            - self.cycle[current_index, 0]
        )

        # --------------------------------------------------------------
        # Integrated physical plant
        # --------------------------------------------------------------

        result = self.powertrain.step(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            sigma_tor=sigma_tor,
            slope_rad=slope_rad,
            dt_s=dt_s,
        )

        # --------------------------------------------------------------
        # Advance cycle
        # --------------------------------------------------------------

        self.current_index = next_index
        self.current_velocity_kmh = target_velocity_kmh

        # --------------------------------------------------------------
        # Construct next state
        # --------------------------------------------------------------

        if self.current_index < len(self.cycle) - 1:
            next_target_velocity_kmh = float(
                self.cycle[self.current_index + 1, 1]
            )

            next_slope_rad = float(
                self.cycle[self.current_index, 2]
            )

            next_dt_s = float(
                self.cycle[self.current_index + 1, 0]
                - self.cycle[self.current_index, 0]
            )

            next_wheel_torque_nm = (
                self.powertrain.vehicle.calculate_wheel_torque(
                    velocity_kmh=target_velocity_kmh,
                    target_velocity_kmh=next_target_velocity_kmh,
                    slope_rad=next_slope_rad,
                    dt=next_dt_s,
                )
            )
        else:
            # No future transition exists. The torque component is
            # retained from the final physical step.
            next_wheel_torque_nm = result.wheel_torque_nm

        next_state = self._build_state(
            velocity_kmh=target_velocity_kmh,
            wheel_torque_nm=next_wheel_torque_nm,
            soc=result.soc,
            battery_soh=result.battery_soh,
            motor1_soh=result.motor1_soh,
            motor2_soh=result.motor2_soh,
        )

        self.current_state = next_state

        # --------------------------------------------------------------
        # Episode termination
        # --------------------------------------------------------------

        terminated = self.current_index >= len(self.cycle) - 1

        truncated = False

        # --------------------------------------------------------------
        # Reward placeholder
        #
        # Phase 3D.4 will replace this with the validated reward model.
        # --------------------------------------------------------------

        reward = 0.0

        # --------------------------------------------------------------
        # Diagnostic information
        # --------------------------------------------------------------

        info = {
            "cycle_index": self.current_index,
            "time_s": float(self.cycle[self.current_index, 0]),
            "sigma_tor": sigma_tor,
            "wheel_torque_nm": result.wheel_torque_nm,
            "battery_power_kw": result.battery_power_kw,
            "battery_current_a": result.battery_current_a,
            "battery_terminal_voltage_v": (
                result.battery_terminal_voltage_v
            ),
            "soc": result.soc,
            "battery_soh": result.battery_soh,
            "motor1_soh": result.motor1_soh,
            "motor2_soh": result.motor2_soh,
            "motor1_feasible": result.motor1_feasible,
            "motor2_feasible": result.motor2_feasible,
            "overall_feasible": result.overall_feasible,
        }

        return (
            next_state.copy(),
            reward,
            terminated,
            truncated,
            info,
        )