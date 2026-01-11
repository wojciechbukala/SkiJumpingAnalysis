import math
import numpy as np

def deg_to_rad(deg: float) -> float:
    return deg * (math.pi / 180.0)

def hermite_basis(t: float):
    # h00, h10, h01, h11
    h00 = (2*t**3 - 3*t**2 + 1)
    h10 = (t**3 - 2*t**2 + t)
    h01 = (-2*t**3 + 3*t**2)
    h11 = (t**3 - t**2)
    return h00, h10, h01, h11

def hermite_basis_derivative(t: float):
    # dh00/dt, dh10/dt, dh01/dt, dh11/dt
    dh00 = (6*t**2 - 6*t)
    dh10 = (3*t**2 - 4*t + 1)
    dh01 = (-6*t**2 + 6*t)
    dh11 = (3*t**2 - 2*t)
    return dh00, dh10, dh01, dh11

def hermite_y(x, x0, x1, y0, y1, dy0, dy1):
    L = (x1 - x0)
    t = (x - x0) / L
    h00, h10, h01, h11 = hermite_basis(t)
    return (h00 * y0 +
            h10 * L * dy0 +
            h01 * y1 +
            h11 * L * dy1)

def hermite_dy_dx(x, x0, x1, y0, y1, dy0, dy1):
    L = (x1 - x0)
    t = (x - x0) / L
    dh00, dh10, dh01, dh11 = hermite_basis_derivative(t)
    # dy/dt
    dy_dt = (dh00 * y0 +
             dh10 * L * dy0 +
             dh01 * y1 +
             dh11 * L * dy1)
    return dy_dt / L

def curve_arclength(x0, x1, y_fun, dy_fun, n=2000):
    xs = np.linspace(x0, x1, n)
    dys = np.array([dy_fun(float(x)) for x in xs], dtype=float)
    integrand = np.sqrt(1.0 + dys**2)
    return float(np.trapz(integrand, xs))

class HillModel:
    def __init__(self, slope_data, y0_takeoff_start: float = 0.0):
        self.slope_data = slope_data
        self.name = getattr(slope_data, "name", "unknown")

        # ---- tuoi parametri ----
        self.tot_height     = float(slope_data.tot_height)
        self.tower_height   = float(slope_data.tower_height)
        self.inrun_length   = float(slope_data.inrun_length)
        self.inrun_angle    = float(slope_data.inrun_angle)
        self.takeoff_angle  = float(slope_data.takeoff_angle)
        self.takeoff_length = float(slope_data.takeoff_length)
        self.takeoff_height = float(slope_data.takeoff_height)
        self.landing_angle  = float(slope_data.landing_angle)

        # Scelta: uso blend_length derivato da takeoff_length (così NON è un parametro “magico”)
        self.blend_length = 3.0 * self.takeoff_length  # tipicamente 15–25m su LH

        # Setup coordinate: x=0 inizio takeoff table
        self.y_takeoff_start = float(y0_takeoff_start)
        self.x_takeoff_start = 0.0

        # Quote top inrun e fine collina usando tower_height e tot_height
        self.y_inrun_top = self.y_takeoff_start + self.tower_height
        self.y_end = self.y_inrun_top - self.tot_height  # usa tot_height davvero

        # slopes
        self.m_inrun_start = -math.tan(deg_to_rad(self.inrun_angle))   # pendenza al top
        self.m_takeoff = math.tan(deg_to_rad(self.takeoff_angle))      # pendenza tavolo
        self.m_landing_end = -math.tan(deg_to_rad(self.landing_angle)) # pendenza finale landing

        # ---- INRUN CURVO ----
        # Inrun è una Hermite da (x_top, y_top) a (0, y_takeoff_start), con slope che ruota fino al takeoff_angle.
        # x_top lo scelgo in modo che la lunghezza lungo pista = inrun_length.
        # Risolvo x_top (negativo) con bisezione.
        self.x_inrun_end = 0.0
        self.y_inrun_end = self.y_takeoff_start
        self.m_inrun_end = self.m_takeoff  # così è C1 con il takeoff table

        def inrun_y_factory(x_top):
            return lambda x: hermite_y(x, x_top, self.x_inrun_end,
                                      self.y_inrun_top, self.y_inrun_end,
                                      self.m_inrun_start, self.m_inrun_end)

        def inrun_dy_factory(x_top):
            return lambda x: hermite_dy_dx(x, x_top, self.x_inrun_end,
                                          self.y_inrun_top, self.y_inrun_end,
                                          self.m_inrun_start, self.m_inrun_end)

        def length_for_span(span):
            x_top = -span
            y_fun = inrun_y_factory(x_top)
            dy_fun = inrun_dy_factory(x_top)
            return curve_arclength(x_top, 0.0, y_fun, dy_fun, n=2500)

        # bracket per span (orizzontale): deve essere > 0
        lo, hi = 1.0, max(50.0, self.inrun_length * 3.0)
        while length_for_span(hi) < self.inrun_length:
            hi *= 1.5
            if hi > 5000:
                break

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            Lmid = length_for_span(mid)
            if Lmid < self.inrun_length:
                lo = mid
            else:
                hi = mid

        self.x_inrun_start = -hi
        self.y_inrun_start = self.y_inrun_top

        # salva funzioni inrun
        self._inrun_y = inrun_y_factory(self.x_inrun_start)
        self._inrun_dy = inrun_dy_factory(self.x_inrun_start)

        # ---- TAKEOFF TABLE ----
        self.x_takeoff_end = self.takeoff_length
        self.y_takeoff_end = self.y_takeoff_start + self.m_takeoff * (self.x_takeoff_end - self.x_takeoff_start)

        # ---- LANDING CURVATA (2 segmenti Hermite) ----
        # Punto "mid" dove impongo che la collina sia scesa di takeoff_height.
        self.x_land_mid = self.x_takeoff_end + self.blend_length
        self.y_land_mid = self.y_takeoff_end - self.takeoff_height

        # pendenza “iniziale” landing (dopo la zona takeoff): meno ripida della landing finale
        self.landing_start_angle = max(8.0, 0.30 * self.landing_angle)
        self.m_landing_mid = -math.tan(deg_to_rad(self.landing_start_angle))

        # scelgo x_end coerente con drop fino a y_end e landing_angle finale (stima orizzontale)
        drop = (self.y_land_mid - self.y_end)  # positivo se y_end è più basso
        dx2 = drop / max(1e-6, math.tan(deg_to_rad(self.landing_angle)))
        self.x_end = self.x_land_mid + dx2

        # Segmento 1: [x_takeoff_end, x_land_mid]
        def land1_y(x):
            return hermite_y(x, self.x_takeoff_end, self.x_land_mid,
                             self.y_takeoff_end, self.y_land_mid,
                             self.m_takeoff, self.m_landing_mid)

        def land1_dy(x):
            return hermite_dy_dx(x, self.x_takeoff_end, self.x_land_mid,
                                 self.y_takeoff_end, self.y_land_mid,
                                 self.m_takeoff, self.m_landing_mid)

        # Segmento 2: [x_land_mid, x_end]
        def land2_y(x):
            return hermite_y(x, self.x_land_mid, self.x_end,
                             self.y_land_mid, self.y_end,
                             self.m_landing_mid, self.m_landing_end)

        def land2_dy(x):
            return hermite_dy_dx(x, self.x_land_mid, self.x_end,
                                 self.y_land_mid, self.y_end,
                                 self.m_landing_mid, self.m_landing_end)

        self._land1_y, self._land1_dy = land1_y, land1_dy
        self._land2_y, self._land2_dy = land2_y, land2_dy

    def y(self, x: float) -> float:
        x = float(x)

        # inrun curvo
        if x < 0.0:
            return self._inrun_y(x)

        # takeoff table (lineare)
        if x <= self.x_takeoff_end:
            return self.y_takeoff_start + self.m_takeoff * x

        # landing curvata (due segmenti)
        if x <= self.x_land_mid:
            return self._land1_y(x)
        if x <= self.x_end:
            return self._land2_y(x)

        # estensione oltre x_end: tangente finale
        return self.m_landing_end * (x - self.x_end) + self.y_end

    def sample(self, x_max: float | None = None, n: int = 2000):
        x_min = self.x_inrun_start
        if x_max is None:
            x_max = self.x_end
        xs = np.linspace(float(x_min), float(x_max), int(n))
        ys = np.array([self.y(x) for x in xs], dtype=float)
        return xs, ys
