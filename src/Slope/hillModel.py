import math
import numpy as np
from scipy.integrate import trapezoid

def deg_to_rad(deg: float) -> float:
    return deg * (math.pi / 180.0)

def hermite_basis(t: float):
    h00 = (2 * t**3 - 3 * t**2 + 1)
    h10 = (t**3 - 2 * t**2 + t)
    h01 = (-2 * t**3 + 3 * t**2)
    h11 = (t**3 - t**2)
    return h00, h10, h01, h11

def hermite_basis_derivative(t: float):
    dh00 = (6 * t**2 - 6 * t)
    dh10 = (3 * t**2 - 4 * t + 1)
    dh01 = (-6 * t**2 + 6 * t)
    dh11 = (3 * t**2 - 2 * t)
    return dh00, dh10, dh01, dh11

def hermite_y(x, x0, x1, y0, y1, dy0, dy1):
    L = (x1 - x0)
    if abs(L) < 1e-12:
        return float(y0)
    t = (x - x0) / L
    h00, h10, h01, h11 = hermite_basis(t)
    return (h00 * y0 +
            h10 * L * dy0 +
            h01 * y1 +
            h11 * L * dy1)

def hermite_dy_dx(x, x0, x1, y0, y1, dy0, dy1):
    L = (x1 - x0)
    if abs(L) < 1e-12:
        return float(dy0)
    t = (x - x0) / L
    dh00, dh10, dh01, dh11 = hermite_basis_derivative(t)
    dy_dt = (dh00 * y0 +
             dh10 * L * dy0 +
             dh01 * y1 +
             dh11 * L * dy1)
    return dy_dt / L

def curve_arclength(x0, x1, dy_fun, n=2000):
    xs = np.linspace(x0, x1, n)
    dys = np.array([dy_fun(float(x)) for x in xs], dtype=float)
    integrand = np.sqrt(1.0 + dys**2)
    return float(trapezoid(integrand, xs))

def _as_abs_x(x_candidate: float, x_ref: float) -> float:
    # If x is not beyond landing start, treat it as offset
    x_candidate = float(x_candidate)
    return x_candidate if x_candidate > x_ref else (x_ref + x_candidate)

class HillModel:
    """
    Requirements:
    1) y=0 at the top of the inrun.
    2) Jump gap at takeoff equals takeoff_height between table and landing.
    3) landing_angle is enforced at x = hill_size (HS); after HS the hill flattens.
    """

    def __init__(self, slope_data):
        self.slope_data = slope_data
        self.name = getattr(slope_data, "name", "unknown")

        self.tot_height = float(slope_data.tot_height)
        self.tower_height = float(slope_data.tower_height)
        self.inrun_length = float(slope_data.inrun_length)
        self.inrun_angle = float(slope_data.inrun_angle)
        self.takeoff_angle = float(slope_data.takeoff_angle)
        self.takeoff_length = float(slope_data.takeoff_length)
        self.takeoff_height = float(slope_data.takeoff_height)
        self.landing_angle = float(slope_data.landing_angle)

        self.hill_size = float(getattr(slope_data, "hill_size", 0.0))
        self.k_point = float(getattr(slope_data, "k_point", 0.0))  # not required here, kept for data completeness

        if self.inrun_length <= 0:
            raise ValueError("inrun_length must be >0 (check DataModel assignment).")
        if self.takeoff_length <= 0:
            raise ValueError("takeoff_length must be > 0.")
        if self.tot_height <= 0:
            raise ValueError("tot_height must be > 0.")
        if self.hill_size <= 0:
            raise ValueError("hill_size must be > 0.")

        # Reference: y=0 at the top of the inrun
        self.y_inrun_top = 0.0

        # Takeoff start at x=0 located at -tower_height
        self.x_takeoff_start = 0.0
        self.y_takeoff_start = -self.tower_height

        # Slopes dy/dx (x downhill, y upward)
        self.m_inrun_start = -math.tan(deg_to_rad(self.inrun_angle))
        self.m_takeoff = math.tan(deg_to_rad(self.takeoff_angle))

        # Curved inrun: Hermite from (x_inrun_start, y=0) to (0, y=-tower_height)
        # with final slope matching the table and total arc length = inrun_length
        self.x_inrun_end = 0.0
        self.y_inrun_end = self.y_takeoff_start
        self.m_inrun_end = self.m_takeoff

        def inrun_y_factory(x0):
            return lambda x: hermite_y(
                x, x0, self.x_inrun_end,
                self.y_inrun_top, self.y_inrun_end,
                self.m_inrun_start, self.m_inrun_end
            )

        def inrun_dy_factory(x0):
            return lambda x: hermite_dy_dx(
                x, x0, self.x_inrun_end,
                self.y_inrun_top, self.y_inrun_end,
                self.m_inrun_start, self.m_inrun_end
            )

        def arclen_for_span(span):
            x0 = -span
            dy_fun = inrun_dy_factory(x0)
            return curve_arclength(x0, 0.0, dy_fun, n=2500)

        # Bisection on horizontal span to match arc length to inrun_length
        lo, hi = 1.0, max(20.0, self.inrun_length * 2.5)
        while arclen_for_span(hi) < self.inrun_length:
            hi *= 1.5
            if hi > 5000:
                break

        for _ in range(70):
            mid = 0.5 * (lo + hi)
            Lmid = arclen_for_span(mid)
            if Lmid < self.inrun_length:
                lo = mid
            else:
                hi = mid

        self.x_inrun_start = -hi
        self.y_inrun_start = self.y_inrun_top

        self._inrun_y = inrun_y_factory(self.x_inrun_start)
        self._inrun_dy = inrun_dy_factory(self.x_inrun_start)

        # Linear takeoff table on [0, takeoff_length]
        self.x_takeoff_end = self.takeoff_length
        self.y_takeoff_end = self.y_takeoff_start + self.m_takeoff * (self.x_takeoff_end - self.x_takeoff_start)

        # Landing start after the required jump gap
        self.x_landing_start = self.x_takeoff_end
        self.y_landing_start = self.y_takeoff_end - self.takeoff_height  # required discontinuity

        # End of hill: total drop = tot_height relative to the top (y_top=0)
        self.y_end = -self.tot_height

        # Enforce that landing_angle holds at x = hill_size (HS)
        self.x_HS = _as_abs_x(self.hill_size, self.x_landing_start)

        # Slope at HS (exactly landing_angle)
        self.m_HS = -math.tan(deg_to_rad(self.landing_angle))

        # Choose an initial landing slope (steeper than HS)
        self.landing_start_angle =0.0
        self.m_landing_start = -math.tan(deg_to_rad(self.landing_start_angle))

        # After HS the hill flattens (less steep)
        self.outrun_angle = max(2.0, min(10.0, 0.18 * self.landing_angle))
        self.m_outrun_end = -math.tan(deg_to_rad(self.outrun_angle))

        # Build segment 1: landing_start -> HS
        def landing1_y(x):
            return hermite_y(
                x, self.x_landing_start, self.x_HS,
                self.y_landing_start, self._y_at_HS,
                self.m_landing_start, self.m_HS
            )

        def landing1_dy(x):
            return hermite_dy_dx(
                x, self.x_landing_start, self.x_HS,
                self.y_landing_start, self._y_at_HS,
                self.m_landing_start, self.m_HS
            )

        # We need y at HS; set it from a baseline (trapezoid slope average)
        dx1 = (self.x_HS - self.x_landing_start)
        if dx1 <= 0:
            raise ValueError("hill_size must be after landing start.")
        self._y_at_HS = self.y_landing_start + 0.5 * (self.m_landing_start + self.m_HS) * dx1

        self._landing1_y = landing1_y
        self._landing1_dy = landing1_dy

        # Choose x_end so that reaching y_end is possible with the flatter outrun slope
        denom = max(1e-6, -self.m_outrun_end)
        dx2 = (self._y_at_HS - self.y_end) / denom
        dx2 = max(20.0, float(dx2))
        self.x_end = self.x_HS + dx2

        # Build segment 2: HS -> end (flattening)
        def landing2_y(x):
            return hermite_y(
                x, self.x_HS, self.x_end,
                self._y_at_HS, self.y_end,
                self.m_HS, self.m_outrun_end
            )

        def landing2_dy(x):
            return hermite_dy_dx(
                x, self.x_HS, self.x_end,
                self._y_at_HS, self.y_end,
                self.m_HS, self.m_outrun_end
            )

        self._landing2_y = landing2_y
        self._landing2_dy = landing2_dy

    def y(self, x: float) -> float:
        x = float(x)

        # Curved inrun (x < 0)
        if x < 0.0:
            return self._inrun_y(x)

        # Takeoff table (0 <= x <= takeoff_end)
        if x <= self.x_takeoff_end:
            return self.y_takeoff_start + self.m_takeoff * x

        # Landing segment 1 (takeoff_end .. HS)
        if x <= self.x_HS:
            return self._landing1_y(x)

        # Landing segment 2 (HS .. x_end)
        if x <= self.x_end:
            return self._landing2_y(x)

        # Beyond x_end: extend along the final outrun tangent
        return self.m_outrun_end * (x - self.x_end) + self.y_end

    def sample(self, x_max: float | None = None, n: int = 2000):
        """
        Sample in two segments to avoid plotting the vertical jump:
        - (x_inrun_start ... x_takeoff_end) = inrun + takeoff table
        - (x_takeoff_end ... x_max) = landing
        """
        if x_max is None:
            x_max = self.x_end

        n1 = max(10, int(n * 0.55))
        n2 = max(10, int(n * 0.45))

        xs1 = np.linspace(float(self.x_inrun_start), float(self.x_takeoff_end), n1)
        ys1 = np.array([self.y(x) for x in xs1], dtype=float)

        xs2 = np.linspace(float(self.x_takeoff_end), float(x_max), n2)
        ys2 = np.array([self.y(x) for x in xs2], dtype=float)

        return (xs1, ys1), (xs2, ys2)
