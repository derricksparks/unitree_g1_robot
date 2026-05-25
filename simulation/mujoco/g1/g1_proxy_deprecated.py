"""Deprecated proxy-finger grasp API (legacy ``g1_position_actuated`` + proxy sliders).

The **active** dual-arm pipeline uses :file:`assets/g1_dex3_hands_actuated.xml` and
``run_g1_dual_arm_box`` with real Dex3 collision geoms and finger position actuators.

Legacy symbols (still importable for compatibility / gradual removal):

* ``LEFT_PROXY_GEOMS`` / ``RIGHT_PROXY_GEOMS`` — proxy geom name tuples in ``run_g1_dual_arm_box``
* ``_proxy_finger_slide_target`` / ``PROXY_*`` constants — in ``run_g1_grasp_box``
* ``_proxy_geom_ids`` — helper in ``run_g1_dual_arm_box``

Do **not** use proxy overlap or squeeze targets for grasp readiness on the Dex3 path.
"""
