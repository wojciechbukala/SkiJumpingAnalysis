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
    dy_dt = (dh00 * y0 +
             dh10 * L * dy0 +
             dh01 * y1 +
             dh11 * L * dy1)
    return dy_dt / L

def curve_arclength(x0, x1, dy_fun, n=2000):
    """
    Lunghezza arco = ∫ sqrt(1 + (dy/dx)^2) dx
    """
    xs = np.linspace(x0, x1, n)
    dys = np.array([dy_fun(float(x)) for x in xs], dtype=float)
    integrand = np.sqrt(1.0 + dys**2)
    return float(np.trapz(integrand, xs))


class HillModel:
    """
    Correzioni richieste:
    1) y=0 nel punto più alto della pista (top inrun).
    2) Discontinuità al take-off pari a takeoff_height tra takeoff-table e landing.
    3) Usa tutti i parametri: tot_height, tower_height, inrun_length, inrun_angle,
       takeoff_angle, takeoff_length, takeoff_height, landing_angle.
    """

    def __init__(self, slope_data):
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

        if self.inrun_length <= 0:
            raise ValueError("inrun_length must be > 0 (controlla che DataModel lo assegni).")
        if self.takeoff_length <= 0:
            raise ValueError("takeoff_length must be > 0.")
        if self.tot_height <= 0:
            raise ValueError("tot_height must be > 0.")

        # =========================================================
        # RIFERIMENTO: y=0 al punto più alto (top inrun)
        # =========================================================
        self.y_inrun_top = 0.0

        # Takeoff start (x=0) a quota -tower_height
        self.x_takeoff_start = 0.0
        self.y_takeoff_start = -self.tower_height

        # slopes dy/dx (x verso valle, y verso l'alto)
        self.m_inrun_start  = -math.tan(deg_to_rad(self.inrun_angle))
        self.m_takeoff      =  math.tan(deg_to_rad(self.takeoff_angle))
        self.m_landing_end  = -math.tan(deg_to_rad(self.landing_angle))

        # =========================================================
        # 1) INRUN CURVO: Hermite da (x_inrun_start, y=0) a (0, y=-tower_height)
        #    con pendenza finale = m_takeoff (C1 col tavolo) e
        #    lunghezza lungo curva = inrun_length
        # =========================================================
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

        # bisezione su span orizzontale per far venire l'arco = inrun_length
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

        self._inrun_y  = inrun_y_factory(self.x_inrun_start)
        self._inrun_dy = inrun_dy_factory(self.x_inrun_start)

        # =========================================================
        # 2) TAKEOFF TABLE (lineare): [0, takeoff_length]
        # =========================================================
        self.x_takeoff_end = self.takeoff_length
        self.y_takeoff_end = self.y_takeoff_start + self.m_takeoff * (self.x_takeoff_end - self.x_takeoff_start)

        # =========================================================
        # 3) LANDING con DISCONTINUITÀ al takeoff:
        #    landing_start(x_takeoff_end) = takeoff_end - takeoff_height
        #    landing curvata (Hermite) fino a fine collina
        # =========================================================
        self.x_landing_start = self.x_takeoff_end
        self.y_landing_start = self.y_takeoff_end - self.takeoff_height  # DISCONTINUITÀ richiesta

        # Fine collina: drop totale = tot_height rispetto al top (y_top=0)
        self.y_end = -self.tot_height

        # x_end stimato imponendo pendenza finale ~ landing_angle
        drop = self.y_landing_start - self.y_end
        denom = max(1e-6, math.tan(deg_to_rad(self.landing_angle)))
        self.x_end = self.x_landing_start + drop / denom

        # pendenza iniziale della landing (deterministica dai tuoi dati; più dolce della finale)
        self.landing_start_angle = max(8.0, 0.30 * self.landing_angle)
        self.m_landing_start = -math.tan(deg_to_rad(self.landing_start_angle))

        def landing_y(x):
            return hermite_y(
                x, self.x_landing_start, self.x_end,
                self.y_landing_start, self.y_end,
                self.m_landing_start, self.m_landing_end
            )

        def landing_dy(x):
            return hermite_dy_dx(
                x, self.x_landing_start, self.x_end,
                self.y_landing_start, self.y_end,
                self.m_landing_start, self.m_landing_end
            )

        self._landing_y = landing_y
        self._landing_dy = landing_dy

    def y(self, x: float) -> float:
        x = float(x)

        # inrun curvo (x < 0)
        if x < 0.0:
            return self._inrun_y(x)

        # takeoff table (0 <= x <= takeoff_end)
        if x <= self.x_takeoff_end:
            return self.y_takeoff_start + self.m_takeoff * x

        # landing curvata (x > takeoff_end) con DISCONTINUITÀ al bordo
        if x <= self.x_end:
            return self._landing_y(x)

        # estensione oltre x_end: tangente finale
        return self.m_landing_end * (x - self.x_end) + self.y_end

    def sample(self, x_max: float | None = None, n: int = 2000):
        """
        Campiono separando in 2 segmenti per non disegnare la verticale della discontinuità:
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
