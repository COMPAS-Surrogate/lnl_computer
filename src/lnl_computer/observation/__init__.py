import traceback

from .observation import Observation


def load_observation(obs: str) -> Observation:
    try:
        if obs.upper() == "LVK":
            from .lvk_observation import LVKObservation

            return LVKObservation.from_ogc4_data()
        elif obs.endswith(".npz"):
            return Observation.load(fname=obs)
        else:
            raise ValueError(f"Unknown observation: {obs}")
    except Exception as e:
        raise ValueError(
            f"Error loading observation [{obs}]: {e} {traceback.format_exc()}"
        )
