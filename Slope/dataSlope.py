from dataclasses import dataclass
from typing import ClassVar, Dict, Any

@dataclass
class DataModel:
    # profile identity
    name: str = ""

    # slope attributes (aligned with profile keys)
    tot_height: float = 0.0
    tower_height: float = 0.0
    inrun_length: float = 0.0
    inrun_angle: float = 0.0
    takeoff_angle: float = 0.0
    takeoff_length: float = 0.0
    takeoff_height: float = 0.0
    landing_angle: float = 0.0
    hill_size: float = 0.0
    k_point: float = 0.0

    # predefined slope profiles (keys should match the attributes above)
    _PROFILES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "default": {
            "tot_height": 140.0,
            "tower_height": 44.0,
            "inrun_length": 99.0,
            "inrun_angle": 35.0,
            "takeoff_angle": 11.0,
            "takeoff_length": 6.5,
            "takeoff_height": 3.38,
            "landing_angle": 35.5,
            "hill_size":106,
            "k_point": 95,
        },
                "obersdorf": {
            "tot_height": 140.0,
            "tower_height": 44.0,
            "inrun_length": 99.0,
            "inrun_angle": 35.0,
            "takeoff_angle": 11.0,
            "takeoff_length": 6.5,
            "takeoff_height": 3.38,
            "landing_angle": 35.5,
            "hill_size":106,
            "k_point": 95,
        },
    }

    def __init__(self, profile_name: str):
        name = (profile_name or "").strip().lower()
        data = self._PROFILES.get(name)
        if data is None:
            data = self._PROFILES["default"]
            name = "obersdorf"
        self.name = name
        self.tot_height = float(data.get("tot_height", 0.0))
        self.tower_height = float(data.get("tower_height", 0.0))
        self.inrun_angle = float(data.get("inrun_angle", 0.0))
        self.takeoff_angle = float(data.get("takeoff_angle", 0.0))
        self.takeoff_length = float(data.get("takeoff_length", 0.0))
        self.takeoff_height = float(data.get("takeoff_height", 0.0))
        self.landing_angle = float(data.get("landing_angle", 0.0))
        self.inrun_length = float(data.get("inrun_length", 0.0))
        self.hill_size = float(data.get("hill_size", 0.0))
        self.k_point = float(data.get("k_point", 0.0))


    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tot_height": self.tot_height,
            "tower_height": self.tower_height,
            "inrun_length": self.inrun_length,
            "inrun_angle": self.inrun_angle,
            "takeoff_angle": self.takeoff_angle,
            "takeoff_length": self.takeoff_length,
            "takeoff_height": self.takeoff_height,
            "landing_angle": self.landing_angle,
            "hill_size": self.hill_size,
            "k_point": self.k_point,
        }
