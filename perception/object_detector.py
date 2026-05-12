import time

import mujoco
import numpy as np


class ObjectDetector:
    """
    Simulated detector interface for the MuJoCo prototype.

    The current implementation reads MuJoCo truth and returns it through the
    same interface a camera-based detector would use later. Optional Gaussian
    noise can be enabled for robustness experiments, but defaults to zero so
    existing behavior stays unchanged.
    """

    def __init__(self, noise_std=0.0, rng=None):
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng() if rng is None else rng

    def detect(self, model, data, shelf_position=None):
        start_time = time.perf_counter()

        box_position = self._body_pos(model, data, "box")
        if shelf_position is None:
            shelf_position = self._geom_pos(model, data, "shelf_top_geom")
        else:
            shelf_position = np.asarray(shelf_position, dtype=float).copy()

        result = {
            "box_position": self._with_noise(box_position),
            "shelf_position": self._with_noise(shelf_position),
            "processing_time_s": time.perf_counter() - start_time,
        }
        return result

    def _with_noise(self, position):
        position = np.asarray(position, dtype=float).copy()
        if self.noise_std <= 0.0:
            return position
        return position + self.rng.normal(0.0, self.noise_std, size=position.shape)

    def _body_pos(self, model, data, name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[body_id].copy()

    def _geom_pos(self, model, data, name):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return data.geom_xpos[geom_id].copy()
