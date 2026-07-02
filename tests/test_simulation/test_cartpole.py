"""Tests for cart-pole physics."""

import math

import pytest

from grntage.simulation.cartpole import CartPole, CartPoleParams, CartPoleState


class TestCartPoleState:
    """Tests for CartPoleState dataclass."""

    def test_default_state(self) -> None:
        """Test default state is all zeros."""
        state = CartPoleState()
        assert state.x == 0.0
        assert state.x_dot == 0.0
        assert state.theta == 0.0
        assert state.theta_dot == 0.0

    def test_custom_state(self) -> None:
        """Test custom state values."""
        state = CartPoleState(x=1.0, x_dot=0.5, theta=0.1, theta_dot=-0.2)
        assert state.x == 1.0
        assert state.x_dot == 0.5
        assert state.theta == 0.1
        assert state.theta_dot == -0.2


class TestCartPoleParams:
    """Tests for CartPoleParams dataclass."""

    def test_default_params(self) -> None:
        """Test default physical parameters."""
        params = CartPoleParams()
        assert params.gravity == 9.8
        assert params.cart_mass == 1.0
        assert params.pole_mass == 0.1
        assert params.pole_half_length == 0.5
        assert params.force_magnitude == 10.0
        assert params.x_bounds == (-2.4, 2.4)
        assert params.theta_bounds == pytest.approx(
            (-math.radians(12), math.radians(12)), rel=1e-6
        )


class TestCartPole:
    """Tests for CartPole physics simulation."""

    def test_initial_state(self) -> None:
        """Test cart-pole initializes with given state."""
        state = CartPoleState(x=1.0, theta=0.05)
        cp = CartPole(state=state)
        assert cp.state.x == 1.0
        assert cp.state.theta == 0.05

    def test_zero_force_balanced(self) -> None:
        """Test zero force on balanced pole."""
        cp = CartPole()
        is_valid = cp.step(0.0)
        assert is_valid
        # State should change minimally
        assert cp.state.x == pytest.approx(0.0, abs=1e-6)

    def test_positive_force_accelerates_cart(self) -> None:
        """Test positive force accelerates cart right."""
        cp = CartPole()
        cp.step(1.0)  # Max rightward force
        # Cart should have moved right
        assert cp.state.x > 0.0
        assert cp.state.x_dot > 0.0

    def test_negative_force_accelerates_cart(self) -> None:
        """Test negative force accelerates cart left."""
        cp = CartPole()
        cp.step(-1.0)  # Max leftward force
        # Cart should have moved left
        assert cp.state.x < 0.0
        assert cp.state.x_dot < 0.0

    def test_boundary_detection_x(self) -> None:
        """Test failure when cart exceeds x bounds."""
        state = CartPoleState(x=2.5)  # Outside bounds
        cp = CartPole(state=state)
        assert not cp.is_valid()

    def test_boundary_detection_theta(self) -> None:
        """Test failure when pole exceeds angle bounds."""
        state = CartPoleState(theta=math.radians(15))  # Outside ±12°
        cp = CartPole(state=state)
        assert not cp.is_valid()

    def test_reset(self) -> None:
        """Test reset restores default state."""
        cp = CartPole(state=CartPoleState(x=1.0, theta=0.1))
        cp.reset()
        assert cp.state.x == 0.0
        assert cp.state.theta == 0.0

    def test_reset_to_custom_state(self) -> None:
        """Test reset to custom state."""
        cp = CartPole()
        cp.reset(CartPoleState(x=0.5, theta=0.05))
        assert cp.state.x == 0.5
        assert cp.state.theta == 0.05

    def test_pole_falls_without_control(self) -> None:
        """Test pole falls when started off-balance."""
        # Start with significant angle (close to limit)
        state = CartPoleState(theta=0.15)  # ~8.6 degrees
        cp = CartPole(state=state)

        # Run many steps without control (use larger time step for faster fall)
        cp.dt = 0.001  # 1ms per step
        for _ in range(10000):
            if not cp.step(0.0):
                break

        # Pole should have fallen (exceeded 12 degree limit)
        assert not cp.is_valid()

    def test_numerical_stability(self) -> None:
        """Test no NaN or Inf values produced."""
        cp = CartPole()
        for i in range(1000):
            alpha = i % 3 - 1  # Alternating -1, 0, 1
            cp.step(alpha)
            assert not math.isnan(cp.state.x)
            assert not math.isnan(cp.state.theta)
            assert not math.isinf(cp.state.x)
            assert not math.isinf(cp.state.theta)

    def test_alpha_clamping(self) -> None:
        """Test force is clamped to [-1, 1]."""
        cp = CartPole()
        # Apply excessive force
        cp.step(5.0)  # Should be clamped to 1.0
        x_max = cp.state.x

        cp.reset()
        cp.step(1.0)
        assert cp.state.x == pytest.approx(x_max, rel=1e-6)


class TestCartPoleSubstepping:
    """Sub-stepping decouples control interval (dt) from integration accuracy."""

    def test_substeps_default_is_single_euler_step(self) -> None:
        """substeps=1 (default) is exactly one Euler step of size dt."""
        a = CartPole(state=CartPoleState(theta=0.05), dt=0.02)
        b = CartPole(state=CartPoleState(theta=0.05), dt=0.02, substeps=1)
        a.step(1.0)
        b.step(1.0)
        assert a.state.theta == b.state.theta
        assert a.state.theta_dot == b.state.theta_dot

    def test_substeps_match_equivalent_single_steps(self) -> None:
        """One step(dt, substeps=N) == N steps of a substeps=1 sim at dt/N.

        Holding a (gentle, in-bounds) force constant, integrating a 0.2s interval
        in 10 sub-steps equals ten 0.02s single Euler steps.
        """
        coarse = CartPole(state=CartPoleState(theta=0.0), dt=0.2, substeps=10)
        fine = CartPole(state=CartPoleState(theta=0.0), dt=0.02, substeps=1)
        assert coarse.step(0.3) is True  # stays in bounds
        assert all(fine.step(0.3) for _ in range(10))
        assert coarse.state.theta == pytest.approx(fine.state.theta, abs=1e-12)
        assert coarse.state.x == pytest.approx(fine.state.x, abs=1e-12)

    def test_substepping_changes_single_step_result(self) -> None:
        """A single 0.2s Euler step is materially less accurate than a 10x
        sub-stepped 0.2s interval — the reason sub-stepping exists. Compared on a
        gentle force where both stay in bounds (no bounds-check interference)."""
        single = CartPole(state=CartPoleState(theta=0.0), dt=0.2, substeps=1)
        subbed = CartPole(state=CartPoleState(theta=0.0), dt=0.2, substeps=10)
        assert single.step(0.3) is True
        assert subbed.step(0.3) is True
        # The coarse step's angle is off by ~0.07 rad (~4deg) vs the accurate one.
        assert abs(single.state.theta - subbed.state.theta) > 0.05

    def test_per_substep_bounds_failure_halts_at_crossing(self) -> None:
        """Bounds are checked per sub-step, so a mid-interval failure halts at the
        crossing rather than integrating the whole interval.

        From just inside +12deg with no rescuing force, both regimes fail, but the
        sub-stepped sim stops at the sub-step it crosses the bound (~12.1deg) while
        a single coarse 0.2s step blows through to ~19deg. Asserting the
        sub-stepped final angle is far smaller uniquely isolates per-sub-step
        detection from an end-of-interval-only check.
        """
        single = CartPole(
            state=CartPoleState(theta=math.radians(11.9)), dt=0.2, substeps=1
        )
        subbed = CartPole(
            state=CartPoleState(theta=math.radians(11.9)), dt=0.2, substeps=10
        )
        assert single.step(0.0) is False
        assert subbed.step(0.0) is False
        # sub-stepped halts just past 12deg; single-step overshoots to ~19deg.
        assert subbed.state.theta < math.radians(13)
        assert single.state.theta > math.radians(18)
        assert subbed.state.theta < single.state.theta

    def test_substeps_must_be_positive(self) -> None:
        """substeps < 1 is rejected."""
        with pytest.raises(ValueError):
            CartPole(substeps=0)
