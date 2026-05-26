"""Simple red box detection from MuJoCo camera rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from lucky_bridge.lucky_scene_setup import detect_box_ground_truth


@dataclass(frozen=True)
class RedBoxDetectionResult:
    vision_enabled: bool
    detection_mode: str
    pose_source: str
    box_detected: bool
    detection_success: bool
    box_bbox_px: tuple[int, int, int, int] | None
    box_center_px: tuple[float, float] | None
    box_position_world: tuple[float, float, float] | None
    frame_processing_time_s: float
    frame_processing_time_ok: bool
    camera_source: str
    distance_to_box_m: float | None
    detection_confidence: float
    obstacle_detected: bool
    obstacle_distance_m: float | None
    obstacle_lateral_bias: float
    rgb_fps: float | None
    depth_fps: float | None
    debug_image_path: str | None
    error: str | None = None


def _segment_red_box(rgb: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[float, float] | None, int]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red_mask = (r > 120.0) & (r > 1.5 * g) & (r > 1.5 * b)
    ys, xs = np.where(red_mask)
    if ys.size < 50:
        return None, None, int(ys.size)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    bbox = (x0, y0, x1, y1)
    center_px = (0.5 * float(x0 + x1), 0.5 * float(y0 + y1))
    return bbox, center_px, int(ys.size)


def _annotate_frame(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    label: str,
    panel_lines: list[str] | None = None,
) -> np.ndarray:
    img = np.asarray(rgb, dtype=np.uint8).copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0 = int(np.clip(x0, 0, img.shape[1] - 1))
        x1 = int(np.clip(x1, 0, img.shape[1] - 1))
        y0 = int(np.clip(y0, 0, img.shape[0] - 1))
        y1 = int(np.clip(y1, 0, img.shape[0] - 1))
        img[y0 : y0 + 2, x0 : x1 + 1] = [0, 255, 0]
        img[y1 - 1 : y1 + 1, x0 : x1 + 1] = [0, 255, 0]
        img[y0 : y1 + 1, x0 : x0 + 2] = [0, 255, 0]
        img[y0 : y1 + 1, x1 - 1 : x1 + 1] = [0, 255, 0]
    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        if panel_lines:
            line_h = 22
            panel_h = 8 + line_h * len(panel_lines)
            cv2.rectangle(bgr, (8, 34), (420, 34 + panel_h), (20, 20, 20), -1)
            cv2.rectangle(bgr, (8, 34), (420, 34 + panel_h), (0, 255, 0), 1)
            for i, line in enumerate(panel_lines):
                y = 34 + 18 + i * line_h
                cv2.putText(bgr, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return img


def _choose_camera_from_names(model: mujoco.MjModel, names: tuple[str, ...]) -> int | str | None:
    for name in names:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            return str(name)
    return None


def _choose_camera(model: mujoco.MjModel) -> int | str | None:
    # Prefer mounted robot cameras first, then generic names, else first available.
    preferred = (
        "robot_rgb_camera",
        "head_rgb_camera",
        "head_cam",
        "wrist_cam",
        "front_camera",
        "front",
        "rgb",
        "camera",
        "cam0",
    )
    for name in preferred:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            return str(name)
    if int(model.ncam) > 0:
        return 0
    return None


def _save_debug_image(path: Path, rgb: np.ndarray, bbox: tuple[int, int, int, int] | None) -> str | None:
    img = _annotate_frame(rgb, bbox, "red-box-detection")
    try:
        import imageio.v2 as imageio  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, img)
        return str(path)
    except Exception:
        # Dependency-free fallback: write PPM.
        try:
            out = path if path.suffix.lower() == ".ppm" else path.with_suffix(".ppm")
            out.parent.mkdir(parents=True, exist_ok=True)
            h, w, _ = img.shape
            with out.open("wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
                f.write(img.tobytes())
            return str(out)
        except Exception:
            return None


class RedBoxCameraDetector:
    """Persistent renderer-based detector with optional live preview window."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int = 640,
        height: int = 480,
        show_window: bool = False,
        show_depth_window: bool = False,
        camera_mode: str = "robot_egocentric",
        window_name: str = "G1 Camera Detection",
        depth_window_name: str = "G1 Depth Perception",
    ) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.camera_mode = str(camera_mode)
        self.rgb_camera = _choose_camera_from_names(
            model,
            ("robot_rgb_camera", "head_rgb_camera", "head_cam", "front_camera", "rgb", "camera"),
        )
        self.depth_camera = _choose_camera_from_names(
            model,
            ("robot_depth_camera", "head_depth_camera", "head_cam", "front_camera", "depth", "camera"),
        )
        self.camera = _choose_camera(model)
        torso_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"))
        if torso_bid < 0:
            torso_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link_rev_1_0"))
        self.torso_bid = torso_bid
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.show_window = bool(show_window)
        self.show_depth_window = bool(show_depth_window)
        self.window_name = str(window_name)
        self.depth_window_name = str(depth_window_name)
        self._cv2_available = None
        self._cv2 = None
        self._last_rgb_ts: float | None = None
        self._last_depth_ts: float | None = None
        self._rgb_fps_ema: float | None = None
        self._depth_fps_ema: float | None = None

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
        if self._cv2_available and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            try:
                self._cv2.destroyWindow(self.depth_window_name)
            except Exception:
                pass

    def _show_rgb(self, rgb: np.ndarray, *, enabled: bool, window_name: str) -> None:
        if not enabled:
            return
        if self._cv2_available is None:
            try:
                import cv2  # type: ignore

                self._cv2 = cv2
                self._cv2_available = True
            except Exception:
                self._cv2_available = False
                return
        if not self._cv2_available or self._cv2 is None:
            return
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        self._cv2.imshow(window_name, bgr)
        self._cv2.waitKey(1)

    def _update_fps(self, now_s: float, stream: str) -> float | None:
        if stream == "rgb":
            last = self._last_rgb_ts
            self._last_rgb_ts = now_s
            ema = self._rgb_fps_ema
        else:
            last = self._last_depth_ts
            self._last_depth_ts = now_s
            ema = self._depth_fps_ema
        if last is None:
            return ema
        dt = max(1e-6, float(now_s - last))
        inst = 1.0 / dt
        updated = inst if ema is None else (0.25 * inst + 0.75 * ema)
        if stream == "rgb":
            self._rgb_fps_ema = float(updated)
        else:
            self._depth_fps_ema = float(updated)
        return float(updated)

    def _camera_for_robot_egocentric(self, data: mujoco.MjData) -> mujoco.MjvCamera | int | str:
        if self.torso_bid < 0:
            return self.rgb_camera if self.rgb_camera is not None else self.camera
        torso_pos = np.asarray(data.xpos[self.torso_bid, :3], dtype=float)
        torso_mat = np.asarray(data.xmat[self.torso_bid, :9], dtype=float).reshape(3, 3)
        forward = torso_mat[:, 0]
        up = torso_mat[:, 2]
        forward = forward / max(1e-6, float(np.linalg.norm(forward)))
        up = up / max(1e-6, float(np.linalg.norm(up)))
        eye = torso_pos + torso_mat @ np.asarray([0.02, 0.0, 0.40], dtype=float)
        lookat = eye + 1.2 * forward + np.asarray([0.0, 0.0, -0.18], dtype=float)
        view_dir = lookat - eye
        view_dir = view_dir / max(1e-6, float(np.linalg.norm(view_dir)))
        azimuth = float(np.degrees(np.arctan2(view_dir[1], view_dir[0])))
        elevation = float(np.degrees(np.arctan2(view_dir[2], np.hypot(view_dir[0], view_dir[1]))))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.trackbodyid = -1
        cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
        cam.distance = 1.2
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def _render_depth(self, data: mujoco.MjData, camera_arg: int | str | mujoco.MjvCamera) -> np.ndarray:
        self.renderer.update_scene(data, camera=camera_arg)
        self.renderer.enable_depth_rendering()
        depth = np.asarray(self.renderer.render(), dtype=np.float32)
        self.renderer.disable_depth_rendering()
        return depth

    def _colorize_depth(self, depth: np.ndarray, bbox: tuple[int, int, int, int] | None, label: str) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(d)
        if not np.any(finite):
            vis = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
            return vis
        vals = d[finite]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        hi = max(lo + 1e-6, hi)
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        gray = (255.0 * (1.0 - norm)).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        return _annotate_frame(rgb, bbox, label)

    def _estimate_distance_from_depth(
        self,
        depth: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> float | None:
        d = np.asarray(depth, dtype=np.float32)
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        x0 = int(np.clip(x0, 0, d.shape[1] - 1))
        x1 = int(np.clip(x1, 0, d.shape[1] - 1))
        y0 = int(np.clip(y0, 0, d.shape[0] - 1))
        y1 = int(np.clip(y1, 0, d.shape[0] - 1))
        crop = d[y0 : y1 + 1, x0 : x1 + 1]
        finite = crop[np.isfinite(crop)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def _estimate_obstacle_metrics(
        self,
        depth: np.ndarray,
        box_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[bool, float | None, float]:
        d = np.asarray(depth, dtype=np.float32)
        h, w = int(d.shape[0]), int(d.shape[1])
        y0 = int(0.25 * h)
        y1 = int(0.95 * h)
        x0 = int(0.10 * w)
        x1 = int(0.90 * w)
        roi = np.asarray(d[y0:y1, x0:x1], dtype=np.float32)
        valid = np.isfinite(roi) & (roi > 0.02) & (roi < 4.0)
        if box_bbox is not None:
            bx0, by0, bx1, by1 = box_bbox
            bx0 = int(np.clip(bx0 - x0 - 6, 0, roi.shape[1] - 1))
            bx1 = int(np.clip(bx1 - x0 + 6, 0, roi.shape[1] - 1))
            by0 = int(np.clip(by0 - y0 - 6, 0, roi.shape[0] - 1))
            by1 = int(np.clip(by1 - y0 + 6, 0, roi.shape[0] - 1))
            if bx1 >= bx0 and by1 >= by0:
                valid[by0 : by1 + 1, bx0 : bx1 + 1] = False
        if not np.any(valid):
            return False, None, 0.0
        vals = roi[valid]
        dist = float(np.percentile(vals, 10))
        mid = roi.shape[1] // 2
        left_valid = valid[:, :mid]
        right_valid = valid[:, mid:]
        left_min = float(np.percentile(roi[:, :mid][left_valid], 10)) if np.any(left_valid) else 4.0
        right_min = float(np.percentile(roi[:, mid:][right_valid], 10)) if np.any(right_valid) else 4.0
        # Positive means obstacle is closer on the left side (steer right).
        bias = float(np.clip((right_min - left_min) / max(0.10, max(left_min, right_min)), -1.0, 1.0))
        return bool(dist < 2.5), dist, bias

    def detect(
        self,
        data: mujoco.MjData,
        *,
        save_debug_image: bool = False,
        debug_image_path: str | None = None,
        show_window: bool | None = None,
        show_depth_window: bool | None = None,
        compute_depth_metrics: bool = False,
    ) -> RedBoxDetectionResult:
        t0 = time.perf_counter()
        detection_mode = "camera_color_segmentation"
        pose_source = "mujoco_body_pose_after_visual_detection"
        bbox: tuple[int, int, int, int] | None = None
        center_px: tuple[float, float] | None = None
        world_pos: tuple[float, float, float] | None = None
        debug_path_out: str | None = None
        err: str | None = None
        box_detected = False
        detection_success = False
        rgb = None
        depth: np.ndarray | None = None
        distance_to_box_m: float | None = None
        mask_pixels = 0
        detection_confidence = 0.0
        obstacle_detected = False
        obstacle_distance_m: float | None = None
        obstacle_lateral_bias = 0.0
        rgb_fps: float | None = None
        depth_fps: float | None = None

        camera_source = "fixed_scene_camera"
        try:
            if self.camera_mode == "robot_mounted":
                cam_arg = self.rgb_camera if self.rgb_camera is not None else self.camera
                if isinstance(cam_arg, str):
                    camera_source = f"mounted_named_camera:{cam_arg}"
                elif isinstance(cam_arg, int):
                    camera_source = f"mounted_camera_id:{cam_arg}"
                else:
                    camera_source = "mounted_camera_unavailable_fallback"
            elif self.camera_mode == "robot_egocentric":
                cam_arg = self._camera_for_robot_egocentric(data)
                camera_source = "robot_egocentric_virtual_camera"
            else:
                cam_arg = self.camera
                if isinstance(cam_arg, str):
                    camera_source = f"named_camera:{cam_arg}"
                elif isinstance(cam_arg, int):
                    camera_source = f"camera_id:{cam_arg}"
            self.renderer.update_scene(data, camera=cam_arg)
            rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
            rgb_fps = self._update_fps(time.perf_counter(), "rgb")
            bbox, center_px, mask_pixels = _segment_red_box(rgb)
            detection_confidence = float(np.clip(mask_pixels / max(1.0, 0.08 * rgb.shape[0] * rgb.shape[1]), 0.0, 1.0))
            if bbox is not None:
                gt = detect_box_ground_truth(self.model, data)
                world_pos = tuple(float(x) for x in gt.box_position_world)
                box_detected = True
                detection_success = True
            else:
                detection_mode = "ground_truth_fallback"
                pose_source = "mujoco_ground_truth_fallback"
                gt = detect_box_ground_truth(self.model, data)
                world_pos = tuple(float(x) for x in gt.box_position_world)
                box_detected = bool(gt.box_detected)
                detection_success = bool(gt.box_detected)
            if save_debug_image and rgb is not None:
                out_path = Path(debug_image_path) if debug_image_path else Path("artifacts/debug/red_box_detection.png")
                debug_path_out = _save_debug_image(out_path, rgb, bbox)
        except Exception as exc:
            detection_mode = "ground_truth_fallback"
            pose_source = "mujoco_ground_truth_fallback"
            gt = detect_box_ground_truth(self.model, data)
            world_pos = tuple(float(x) for x in gt.box_position_world)
            box_detected = bool(gt.box_detected)
            detection_success = bool(gt.box_detected)
            err = str(exc)

        show_rgb_now = bool(self.show_window if show_window is None else show_window)
        show_depth_now = bool(self.show_depth_window if show_depth_window is None else show_depth_window)
        need_depth = bool(compute_depth_metrics or show_depth_now)
        if rgb is not None and need_depth:
            if self.camera_mode == "robot_mounted":
                depth_cam_arg: int | str | mujoco.MjvCamera = self.depth_camera if self.depth_camera is not None else cam_arg
            else:
                depth_cam_arg = cam_arg
            depth = self._render_depth(data, depth_cam_arg)
            depth_fps = self._update_fps(time.perf_counter(), "depth")
            distance_to_box_m = self._estimate_distance_from_depth(depth, bbox)
            obstacle_detected, obstacle_distance_m, obstacle_lateral_bias = self._estimate_obstacle_metrics(depth, bbox)

        if rgb is not None and (show_rgb_now or show_depth_now):
            label = f"{detection_mode} detected={box_detected}"
            panel_lines = [
                f"distance_to_box_m: {distance_to_box_m:.3f}" if distance_to_box_m is not None else "distance_to_box_m: n/a",
                f"detection_confidence: {detection_confidence:.2f}",
                f"obstacle_distance_m: {obstacle_distance_m:.3f}" if obstacle_distance_m is not None else "obstacle_distance_m: n/a",
                f"rgb_fps: {rgb_fps:.1f}" if rgb_fps is not None else "rgb_fps: n/a",
                f"depth_fps: {depth_fps:.1f}" if depth_fps is not None else "depth_fps: n/a",
            ]
            if show_depth_now:
                panel_lines[0] = f"distance_to_box_m: {distance_to_box_m:.3f}" if distance_to_box_m is not None else "distance_to_box_m: n/a"
                panel_lines[2] = f"obstacle_distance_m: {obstacle_distance_m:.3f}" if obstacle_distance_m is not None else "obstacle_distance_m: n/a"
                panel_lines[4] = f"depth_fps: {depth_fps:.1f}" if depth_fps is not None else "depth_fps: n/a"
                depth_rgb = self._colorize_depth(depth, bbox, label=f"depth + {label}")
                depth_rgb = _annotate_frame(depth_rgb, bbox, label=f"depth + {label}", panel_lines=panel_lines)
                self._show_rgb(depth_rgb, enabled=True, window_name=self.depth_window_name)
            rgb_annotated = _annotate_frame(rgb, bbox, label, panel_lines=panel_lines)
            if show_rgb_now:
                self._show_rgb(rgb_annotated, enabled=True, window_name=self.window_name)

        dt = float(time.perf_counter() - t0)
        return RedBoxDetectionResult(
            vision_enabled=True,
            detection_mode=detection_mode,
            pose_source=pose_source,
            box_detected=bool(box_detected),
            detection_success=bool(detection_success),
            box_bbox_px=bbox,
            box_center_px=center_px,
            box_position_world=world_pos,
            frame_processing_time_s=dt,
            frame_processing_time_ok=bool(dt <= 1.0),
            camera_source=camera_source,
            distance_to_box_m=distance_to_box_m,
            detection_confidence=float(detection_confidence),
            obstacle_detected=bool(obstacle_detected),
            obstacle_distance_m=obstacle_distance_m,
            obstacle_lateral_bias=float(obstacle_lateral_bias),
            rgb_fps=rgb_fps,
            depth_fps=depth_fps,
            debug_image_path=debug_path_out,
            error=err,
        )


def detect_red_box_from_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int = 640,
    height: int = 480,
    save_debug_image: bool = False,
    debug_image_path: str | None = None,
) -> RedBoxDetectionResult:
    detector = RedBoxCameraDetector(
        model,
        width=int(width),
        height=int(height),
        show_window=False,
        show_depth_window=False,
        camera_mode="robot_mounted",
    )
    try:
        return detector.detect(
            data,
            save_debug_image=bool(save_debug_image),
            debug_image_path=debug_image_path,
            show_window=False,
        )
    finally:
        detector.close()

