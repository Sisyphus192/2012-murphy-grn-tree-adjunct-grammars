"""Cart-Pole physics model.

Implements the inverted pendulum dynamics using the equations
from Section 5.1 of the paper (Equation 4).
"""

import math
from dataclasses import dataclass, field

# Canonical cart-pole state-variable ranges (paper sec. 4 / Whitley generalisation
# grid). Shared by input encoding, random initial states, and the generalisation
# grid so all three agree. Angles/rates are in radians (the integrator's units).
X_RANGE = (-2.4, 2.4)  # m
X_DOT_RANGE = (-1.0, 1.0)  # m/s
THETA_RANGE = (-math.radians(12), math.radians(12))  # rad (+/-12 deg)
THETA_DOT_RANGE = (-math.radians(1.5), math.radians(1.5))  # rad/s (+/-1.5 deg/s)


@dataclass
class CartPoleState:
    """State of the cart-pole system.

    Attributes:
        x: Cart position (meters), bounds: [-2.4, 2.4]
        x_dot: Cart velocity (m/s)
        theta: Pole angle (radians), bounds: [-12°, 12°] = [-0.2094, 0.2094]
        theta_dot: Pole angular velocity (rad/s)
    """

    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0


@dataclass
class CartPoleParams:
    """Physical parameters for the cart-pole system.

    Attributes:
        gravity: Gravitational acceleration (m/s²), default 9.8
        cart_mass: Mass of the cart (kg), default 1.0
        pole_mass: Mass of the pole (kg), default 0.1
        pole_half_length: Half the pole length (m), default 0.5
        force_magnitude: Maximum force magnitude (N), default 10.0
        x_bounds: Cart position bounds (m), default (-2.4, 2.4)
        theta_bounds: Pole angle bounds (rad), default (-12°, 12°)
    """

    gravity: float = 9.8  # Positive gravity for inverted pendulum
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    force_magnitude: float = 10.0
    x_bounds: tuple[float, float] = field(default_factory=lambda: (-2.4, 2.4))
    theta_bounds: tuple[float, float] = field(
        default_factory=lambda: (-math.radians(12), math.radians(12))
    )


class CartPole:
    """Cart-pole physics simulation.

    Implements the physics model from Equation 4:
    θ̈_t = [g*sin(θ_t) - cos(θ_t) * (F_t + m*l*θ̇_t²*sin(θ_t))/(m_c + m)]
           / [l * (4/3 - m*cos²(θ_t)/(m_c + m))]

    ẍ_t = [F_t + m*l*(θ̇_t²*sin(θ_t) - θ̈_t*cos(θ_t))] / (m_c + m)

    Per the paper (p. 383): "The GRN is then iterated 2000 times, corresponding
    to 0.2s of simulated time for the cart-pole model." This means each control
    cycle (one call to step()) applies a constant force for 0.2s of physics time.

    ``dt`` is the **control interval** (how long the force is held); ``substeps``
    is how many Euler integration sub-steps that interval is split into. The two
    are decoupled because a single 0.2s Euler step is wildly inaccurate (it
    overshoots the pole angle by ~13deg in one step). The paper-faithful regime is
    ``dt=0.2, substeps=10`` (force held 0.2s, physics integrated at 0.02s — the
    classic Barto cart-pole step). ``substeps=1`` is a single Euler step of size
    ``dt``.

    Attributes:
        state: Current state of the system
        params: Physical parameters
        dt: Control interval in seconds (force held constant for this long)
        substeps: Euler sub-steps per control interval (each of size dt/substeps)
    """

    def __init__(
        self,
        state: CartPoleState | None = None,
        params: CartPoleParams | None = None,
        dt: float = 0.2,  # control interval in seconds (per paper p. 383)
        substeps: int = 1,  # Euler sub-steps per control interval
    ) -> None:
        """Initialize the cart-pole system.

        Args:
            state: Initial state (default: all zeros)
            params: Physical parameters (default: standard values)
            dt: Control interval in seconds (the force is held constant for this
                duration; 0.2s per paper p. 383).
            substeps: Number of Euler integration sub-steps per control interval,
                each of size dt/substeps. Default 1 (a single Euler step of dt);
                use 10 for the paper-faithful 0.2s/0.02s regime.
        """
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        # Copy state values to prevent mutation of the original object
        if state is not None:
            self.state = CartPoleState(
                x=state.x,
                x_dot=state.x_dot,
                theta=state.theta,
                theta_dot=state.theta_dot,
            )
        else:
            self.state = CartPoleState()
        self.params = params or CartPoleParams()
        self.dt = dt
        self.substeps = substeps

    def step(self, alpha: float) -> bool:
        """Advance the simulation by one control cycle (dt seconds).

        The control decision ``alpha`` is held constant for the whole control
        interval ``dt``; the physics is integrated in ``substeps`` semi-implicit
        Euler sub-steps of size ``dt/substeps``. Bounds are checked after every
        sub-step, so a mid-interval failure (the pole leaving ±12deg or the cart
        leaving ±2.4m) is caught immediately rather than only at the boundary.

        Args:
            alpha: Normalized force in [-1.0, 1.0], applied for the entire interval

        Returns:
            True if the state is valid through the whole interval, False if it
            entered a failure state at any sub-step.
        """
        # Clamp alpha to valid range
        alpha = max(-1.0, min(1.0, alpha))

        # Convert normalized force to actual force (held constant across substeps)
        force = alpha * self.params.force_magnitude

        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            # Compute accelerations using Equation 4 (state-dependent each substep)
            theta_ddot, x_ddot = self._compute_accelerations(force)

            # Semi-implicit Euler integration
            self.state.theta_dot += theta_ddot * sub_dt
            self.state.theta += self.state.theta_dot * sub_dt
            self.state.x_dot += x_ddot * sub_dt
            self.state.x += self.state.x_dot * sub_dt

            if not self.is_valid():
                return False

        return True

    def _compute_accelerations(self, force: float) -> tuple[float, float]:
        """Compute angular and linear accelerations.

        Args:
            force: Applied force in Newtons

        Returns:
            Tuple of (theta_ddot, x_ddot)
        """
        g = self.params.gravity
        mc = self.params.cart_mass
        m = self.params.pole_mass
        half_length = self.params.pole_half_length
        theta = self.state.theta
        theta_dot = self.state.theta_dot

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        total_mass = mc + m

        # Angular acceleration (Equation 4a)
        temp = (force + m * half_length * theta_dot**2 * sin_theta) / total_mass
        numerator = g * sin_theta - cos_theta * temp
        denominator = half_length * (4.0 / 3.0 - m * cos_theta**2 / total_mass)
        theta_ddot = numerator / denominator

        # Linear acceleration (Equation 4b)
        x_ddot = (
            force
            + m * half_length * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)
        ) / total_mass

        return theta_ddot, x_ddot

    def is_valid(self) -> bool:
        """Check if current state is within bounds.

        Returns:
            True if state is valid
        """
        x_min, x_max = self.params.x_bounds
        theta_min, theta_max = self.params.theta_bounds

        if not (x_min <= self.state.x <= x_max):
            return False
        if not (theta_min <= self.state.theta <= theta_max):
            return False

        # Check for numerical instability
        if math.isnan(self.state.x) or math.isnan(self.state.theta):
            return False
        if math.isinf(self.state.x) or math.isinf(self.state.theta):
            return False

        return True

    def reset(self, state: CartPoleState | None = None) -> None:
        """Reset the system to a given state.

        Args:
            state: New state (default: all zeros)
        """
        self.state = state or CartPoleState()
