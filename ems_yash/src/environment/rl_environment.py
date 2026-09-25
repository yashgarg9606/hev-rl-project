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

Target Wu et al. (2024) state:
    [v, Td, SOC, SOH, SOHM1, SOHM2]

Implementation mapping:
    v       -> velocity_kmh
    Td      -> wheel_torque_nm

Reward:
    Implemented using the validated Wu et al. (2024) reward model.

Important numerical detail:
    The Gymnasium observation is float32, but reward calculation keeps
    SOC/SOH values internally at float64 precision so that very small
    motor-health changes are not lost through float32 quantization.
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
from .reward import WuReward
from .constraints import PhysicalConstraintChecker
from src.agent.feasible_action_mapper import map_action


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

        self.project_root = Path(project_root)

        (
            self.motor1_map_path,
            self.motor2_map_path,
        ) = default_motor_map_paths(self.project_root)

        self.powertrain_parameters = (
            powertrain_parameters
            or IntegratedPowertrainParameters()
        )

        self.powertrain = self._create_powertrain()

        # --------------------------------------------------------------
        # Reward model
        # --------------------------------------------------------------

        self.reward_model = WuReward()
        self.constraint_checker = PhysicalConstraintChecker()

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
        # These are intentionally broad observation bounds.
        # The paper's stricter physical constraints will be handled
        # separately from the Gymnasium observation-space definition.
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

        # --------------------------------------------------------------
        # High-precision state used exclusively for reward calculation
        #
        # This intentionally remains float64.
        #
        # Layout:
        #     [velocity, wheel_torque, SOC, battery_SOH,
        #      motor1_SOH, motor2_SOH]
        # --------------------------------------------------------------

        self._reward_state = np.zeros(
            self.state_dim,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Powertrain construction
    # ------------------------------------------------------------------

    def _create_powertrain(self) -> IntegratedPowertrain:
        """
        Construct a fresh integrated powertrain.

        A new instance is used at the beginning of every episode so
        battery SOC/polarization and component health return to their
        initial conditions.
        """

        return IntegratedPowertrain(
            motor1_map_path=self.motor1_map_path,
            motor2_map_path=self.motor2_map_path,
            parameters=self.powertrain_parameters,
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

        The observation is intentionally float32 because this is the
        tensor representation consumed by the RL agent.
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
                f"Constructed state is outside observation space: "
                f"{state}"
            )

        return state

    # ------------------------------------------------------------------
    # High-precision reward state
    # ------------------------------------------------------------------

    def _build_reward_state(
        self,
        velocity_kmh: float,
        wheel_torque_nm: float,
        soc: float,
        battery_soh: float,
        motor1_soh: float,
        motor2_soh: float,
    ) -> np.ndarray:
        """
        Construct the high-precision state used by the reward model.

        Unlike the Gymnasium observation, this state is retained as
        float64 so that very small SOH changes are preserved.
        """

        return np.array(
            [
                velocity_kmh,
                wheel_torque_nm,
                soc,
                battery_soh,
                motor1_soh,
                motor2_soh,
            ],
            dtype=np.float64,
        )

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

        # --------------------------------------------------------------
        # Reset physical powertrain state
        # --------------------------------------------------------------

        self.powertrain = self._create_powertrain()

        self.current_index = 0
        self.simulation_time_s = 0.0  # Phase 3B: track simulation time for BMS trace

        # --------------------------------------------------------------
        # Initial driving-cycle point
        # --------------------------------------------------------------

        velocity_kmh = float(self.cycle[0, 1])

        # Td in the RL state is mapped to the wheel torque associated
        # with the next driving-cycle transition.

        target_velocity_kmh = float(self.cycle[1, 1])

        slope_rad = float(self.cycle[0, 2])

        dt_s = float(
            self.cycle[1, 0] - self.cycle[0, 0]
        )

        wheel_torque_nm = (
            self.powertrain.vehicle.calculate_wheel_torque(
                velocity_kmh=velocity_kmh,
                target_velocity_kmh=target_velocity_kmh,
                slope_rad=slope_rad,
                dt=dt_s,
            )
        )

        # --------------------------------------------------------------
        # Initial health state
        # --------------------------------------------------------------

        # Phase 3A: Get health state through BMS interface
        health = self.powertrain.health_interface.get_health_state(self.simulation_time_s)

        soc = float(self.powertrain.battery.state.soc)

        battery_soh = float(health["battery_soh"])
        motor1_soh = float(health["motor1_soh"])
        motor2_soh = float(health["motor2_soh"])

        self.current_velocity_kmh = velocity_kmh

        # --------------------------------------------------------------
        # Construct observation
        # --------------------------------------------------------------

        self.current_state = self._build_state(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            soc=soc,
            battery_soh=battery_soh,
            motor1_soh=motor1_soh,
            motor2_soh=motor2_soh,
        )

        # --------------------------------------------------------------
        # Initialize high-precision reward state
        # --------------------------------------------------------------

        self._reward_state = self._build_reward_state(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            soc=soc,
            battery_soh=battery_soh,
            motor1_soh=motor1_soh,
            motor2_soh=motor2_soh,
        )

        # --------------------------------------------------------------
        # Reset information
        # --------------------------------------------------------------

        info = {
            "cycle_index": self.current_index,
            "time_s": float(self.cycle[0, 0]),
            "velocity_kmh": velocity_kmh,
            "wheel_torque_nm": wheel_torque_nm,
            "soc": soc,
            "battery_soh": battery_soh,
            "motor1_soh": motor1_soh,
            "motor2_soh": motor2_soh,
        }

        return self.current_state.copy(), info

    # ------------------------------------------------------------------
    # Gymnasium step
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Feasibility-aware action mapping
    # ------------------------------------------------------------------

    def _calculate_sigma_min(
        self,
        velocity_kmh: float,
        wheel_torque_nm: float,
    ) -> float:
        """
        Calculate the minimum feasible physical torque split.

        The RL Actor outputs a normalized action a ∈ [0, 1].
        This method determines the lower feasible physical boundary
        sigma_min for the current transition.

        The six-dimensional RL state is unchanged.

        sigma_min is derived only from the current transition's physical
        operating point and does not become an observation variable.
        """

        sigma_grid = np.linspace(
            0.0,
            1.0,
            10001,
            dtype=np.float64,
        )

        for sigma in sigma_grid:

            motor_operating_point = (
                self.powertrain.motors.calculate_operating_point(
                    velocity_kmh=velocity_kmh,
                    wheel_torque_nm=wheel_torque_nm,
                    sigma_tor=float(sigma),
                )
            )

            if motor_operating_point["overall_feasible"]:
                return float(sigma)

        raise ValueError(
            "No feasible sigma_tor exists for the current transition: "
            f"velocity={velocity_kmh:.6f} km/h, "
            f"wheel_torque={wheel_torque_nm:.6f} Nm"
        )

    def step(self, action):
        """
        Advance the environment by one driving-cycle timestep.

        The physical plant is advanced first. The Wu et al. reward is
        then calculated from the high-precision pre-step and post-step
        SOC/SOH values.
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

        normalized_action = float(action[0])

        # --------------------------------------------------------------
        # Check episode boundary
        # --------------------------------------------------------------

        if self.current_index >= len(self.cycle) - 1:
            raise RuntimeError(
                "Episode has already terminated. Call reset()."
            )

        # --------------------------------------------------------------
        # Preserve pre-step reward state
        # --------------------------------------------------------------

        reward_state_before = self._reward_state.copy()

        # --------------------------------------------------------------
        # Current and target driving-cycle point
        # --------------------------------------------------------------

        current_index = self.current_index
        next_index = current_index + 1

        velocity_kmh = float(
            self.cycle[current_index, 1]
        )

        target_velocity_kmh = float(
            self.cycle[next_index, 1]
        )

        slope_rad = float(
            self.cycle[current_index, 2]
        )

        dt_s = float(
            self.cycle[next_index, 0]
            - self.cycle[current_index, 0]
        )

        # --------------------------------------------------------------
        # Calculate wheel torque for action mapping
        # --------------------------------------------------------------

        wheel_torque_nm = (
            self.powertrain.vehicle.calculate_wheel_torque(
                velocity_kmh=velocity_kmh,
                target_velocity_kmh=target_velocity_kmh,
                slope_rad=slope_rad,
                dt=dt_s,
            )
        )

        # --------------------------------------------------------------
        # Feasible action mapping
        #
        # FIXED PRE-EXISTING BUG: Original code used sigma_tor before
        # defining it. The action mapping was missing.
        # --------------------------------------------------------------

        sigma_min = self._calculate_sigma_min(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
        )

        sigma_tor = map_action(
            actor_action=normalized_action,
            sigma_min=sigma_min,
        )

        # --------------------------------------------------------------
        # Physical constraint check BEFORE powertrain simulation
        #
        # The integrated powertrain intentionally rejects infeasible
        # motor operating points. Therefore the RL environment checks
        # the motor operating point first so that an infeasible action
        # terminates the episode in a controlled manner instead of
        # raising an exception.
        # --------------------------------------------------------------

        motor_operating_point = (
            self.powertrain.motors.calculate_operating_point(
                velocity_kmh=velocity_kmh,
                wheel_torque_nm=wheel_torque_nm,
                sigma_tor=sigma_tor,
            )
        )

        # Get current health state for constraint checking
        current_health = self.powertrain.health_interface.get_health_state(self.simulation_time_s)

        pre_step_constraint_result = (
            self.constraint_checker.check(
                soc=float(self.powertrain.battery.state.soc),
                battery_soh=float(current_health["battery_soh"]),
                motor1_soh=float(current_health["motor1_soh"]),
                motor2_soh=float(current_health["motor2_soh"]),
                motor1_speed_rpm=float(
                    motor_operating_point["motor1_speed_rpm"]
                ),
                motor2_speed_rpm=float(
                    motor_operating_point["motor2_speed_rpm"]
                ),
                motor1_torque_nm=float(
                    motor_operating_point["motor1_torque_nm"]
                ),
                motor2_torque_nm=float(
                    motor_operating_point["motor2_torque_nm"]
                ),
            )
        )

        if not pre_step_constraint_result.overall_valid:

            self.current_index = next_index
            self.current_velocity_kmh = target_velocity_kmh

            terminated = True
            truncated = False

            info = {
                "cycle_index": self.current_index,
                "time_s": float(
                    self.cycle[self.current_index, 0]
                ),
                "normalized_action": normalized_action,
                "sigma_min": sigma_min,
                "sigma_tor": sigma_tor,

                "constraint_violation": True,
                "violated_constraints": (
                    pre_step_constraint_result.violated_constraints
                ),

                "constraint_stage": "pre_step",

                "motor1_speed_rpm": float(
                    motor_operating_point["motor1_speed_rpm"]
                ),
                "motor2_speed_rpm": float(
                    motor_operating_point["motor2_speed_rpm"]
                ),
                "motor1_torque_nm": float(
                    motor_operating_point["motor1_torque_nm"]
                ),
                "motor2_torque_nm": float(
                    motor_operating_point["motor2_torque_nm"]
                ),

                "soc": float(
                    self.powertrain.battery.state.soc
                ),
                "battery_soh": float(current_health["battery_soh"]),
                "motor1_soh": float(current_health["motor1_soh"]),
                "motor2_soh": float(current_health["motor2_soh"]),
            }

            return (
                self.current_state.copy(),
                0.0,
                terminated,
                truncated,
                info,
            )


        result = self.powertrain.step(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            sigma_tor=sigma_tor,
            slope_rad=slope_rad,
            dt_s=dt_s,
        )

        # Increment simulation time after powertrain step
        self.simulation_time_s += dt_s

        # --------------------------------------------------------------
        # Physical constraint check AFTER powertrain simulation
        # --------------------------------------------------------------

        post_step_constraint_result = (
            self.constraint_checker.check(
                soc=float(result.soc),
                battery_soh=float(result.battery_soh),
                motor1_soh=float(result.motor1_soh),
                motor2_soh=float(result.motor2_soh),
                motor1_speed_rpm=float(result.motor1_speed_rpm),
                motor2_speed_rpm=float(result.motor2_speed_rpm),
                motor1_torque_nm=float(result.motor1_torque_nm),
                motor2_torque_nm=float(result.motor2_torque_nm),
            )
        )

        # --------------------------------------------------------------
        # Advance cycle
        # --------------------------------------------------------------

        self.current_index = next_index

        self.current_velocity_kmh = target_velocity_kmh

        # --------------------------------------------------------------
        # Construct next state's demand torque
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

            # No future transition exists. Retain the torque from
            # the final physical transition.
            next_wheel_torque_nm = result.wheel_torque_nm

        # --------------------------------------------------------------
        # Build high-precision post-step reward state
        # --------------------------------------------------------------

        reward_state_after = self._build_reward_state(
            velocity_kmh=target_velocity_kmh,
            wheel_torque_nm=next_wheel_torque_nm,
            soc=float(result.soc),
            battery_soh=float(result.battery_soh),
            motor1_soh=float(result.motor1_soh),
            motor2_soh=float(result.motor2_soh),
        )

        # --------------------------------------------------------------
        # Calculate Wu et al. reward
        # --------------------------------------------------------------

        reward = float(
            self.reward_model.compute(
                reward_state_before,
                reward_state_after,
            )
        )

        # Update high-precision reward state only after the reward has
        # been calculated.
        self._reward_state = reward_state_after

        # --------------------------------------------------------------
        # Construct next observation
        # --------------------------------------------------------------

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

        terminated = (
            self.current_index >= len(self.cycle) - 1
            or not post_step_constraint_result.overall_valid
        )

        truncated = False

        # --------------------------------------------------------------
        # Reward diagnostics
        #
        # Wu reward uses:
        #
        #   ΔSOC   = SOC_before   - SOC_after
        #   ΔSOH   = SOH_before   - SOH_after
        #   ΔSOHM1 = SOHM1_before - SOHM1_after
        #   ΔSOHM2 = SOHM2_before - SOHM2_after
        #
        # This preserves the validated signed convention of the
        # reward implementation.
        # --------------------------------------------------------------

        delta_soc = float(
            reward_state_before[2]
            - reward_state_after[2]
        )

        delta_soh = float(
            reward_state_before[3]
            - reward_state_after[3]
        )

        delta_sohm1 = float(
            reward_state_before[4]
            - reward_state_after[4]
        )

        delta_sohm2 = float(
            reward_state_before[5]
            - reward_state_after[5]
        )

        # --------------------------------------------------------------
        # Diagnostic information
        # --------------------------------------------------------------

        info = {
            "cycle_index": self.current_index,
            "time_s": float(
                self.cycle[self.current_index, 0]
            ),
            "constraint_violation": (
                not post_step_constraint_result.overall_valid
            ),
            "violated_constraints": (
                post_step_constraint_result.violated_constraints
            ),
            "constraint_stage": (
                "post_step"
                if not post_step_constraint_result.overall_valid
                else None
            ),
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

            # Reward
            "reward": reward,

            # Reward deltas
            "delta_soc_reward": delta_soc,
            "delta_soh_reward": delta_soh,
            "delta_sohm1_reward": delta_sohm1,
            "delta_sohm2_reward": delta_sohm2,
        }

        return (
            next_state.copy(),
            reward,
            terminated,
            truncated,
            info,
        )