from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import yaml
import cv2
import numpy as np
from time import time
from numpy.typing import NDArray

from apriltag_bundle_detector.transform import Rotation, Transform
from pyapriltags import Detector


class TagConfig(TypedDict):
    size: float
    T_bundle_tag: Transform


class AprilTagBundleDetector:
    """Detect configured AprilTag bundles and estimate bundle poses in camera frame.

    Pose convention used throughout this class:
    - `T_bundle_tag` maps points from each tag frame into the bundle frame.
    - Solved bundle poses map points from bundle frame into camera frame
      (`T_camera_bundle`, i.e. bundle -> camera).
    """

    def __init__(
        self,
        yaml_path: str,
        camera_matrix: NDArray[np.floating[Any]],
        distortion_coeffs: Optional[NDArray[np.floating[Any]]] = None,
        reproj_error_thresh: float = 1.0,
        min_tags: int = 1,
        tag_family: Optional[str] = None,
    ) -> None:
        """Initialize detector configuration and camera calibration.

        Args:
            yaml_path: Path to the bundle YAML configuration file.
            camera_matrix: Camera intrinsics matrix (3x3).
            distortion_coeffs: OpenCV distortion coefficients.
            reproj_error_thresh: Per-tag mean reprojection error threshold in pixels.
            min_tags: Minimum visible inlier tags required to localize a bundle.
            tag_family: Optional AprilTag family override.
        """
        self.camera_matrix: NDArray[np.floating[Any]] = camera_matrix
        self.dist_coeffs: Optional[NDArray[np.floating[Any]]] = distortion_coeffs
        self.reproj_error_thresh = reproj_error_thresh
        self.min_tags = min_tags

        self.bundles, resolved_family = self._load_config(
            yaml_path, tag_family
        )
        self.tag_family: str = resolved_family
        self.detector = Detector(families=resolved_family)

        self.last_transforms: Dict[str, Transform] = {
            name: Transform.identity() for name in self.bundles
        }

    # ---------------------------------------------------------

    def __call__(
        self, image: NDArray[np.uint8], debug: bool = False
    ) -> Dict[str, Transform]:
        """Run AprilTag detection and return per-bundle camera-frame poses.

        Args:
            image: Input grayscale image as `uint8` array.
            debug: If `True`, print timing and warning messages.

        Returns:
            `Dict[str, Transform]`: Mapping from bundle name to pose transform
            from bundle coordinates to camera coordinates (`T_camera_bundle`,
            bundle -> camera). If a bundle cannot be localized in the current
            frame, the most recent valid transform for that bundle is returned.
        """
        start = time()
        detections = self.detector.detect(image)
        detection_time = time()
        results: Dict[str, Transform] = {}

        for bundle_name, bundle_tags in self.bundles.items():
            obj_pts, img_pts, tag_slices = self._collect_points(
                detections, bundle_tags
            )

            if len(tag_slices) < self.min_tags:
                if debug: print(f"{time()} Warning: failed to localize bundle '{bundle_name}': tags out of view")
                results[bundle_name] = self.last_transforms[bundle_name]
                continue

            T = self._solve_bundle_pnp(obj_pts, img_pts, tag_slices)

            if T is not None:
                self.last_transforms[bundle_name] = T
            else:
                if debug: print(f"{time()} Warning: failed to localize bundle '{bundle_name}': could not solve PnP")

            results[bundle_name] = self.last_transforms[bundle_name]

        if debug: print(f"Detection done in {((detection_time-start)*1000):.2f}ms full call in {((time()-start)*1000):.2f}ms")

        return results

    # ---------------------------------------------------------

    def _solve_bundle_pnp(
        self,
        obj_pts: NDArray[np.float32],
        img_pts: NDArray[np.float32],
        tag_slices: Sequence[Tuple[int, int]],
    ) -> Optional[Transform]:
        """Solve PnP for the bundle and return `T_camera_bundle` (bundle -> camera)."""
        # Initial solve
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None

        errors = self._per_tag_errors(obj_pts, img_pts, rvec, tvec, tag_slices)

        # Keep tags whose mean error is reasonable
        inlier_mask = [
            err < self.reproj_error_thresh for err in errors
        ]

        if sum(inlier_mask) < self.min_tags:
            return None

        # Rebuild inlier point sets
        obj_in, img_in = [], []
        for keep, (i0, i1) in zip(inlier_mask, tag_slices):
            if not keep:
                continue
            obj_in.append(obj_pts[i0:i1])
            img_in.append(img_pts[i0:i1])

        obj_in = np.vstack(obj_in)
        img_in = np.vstack(img_in)

        # Final solve
        ok, rvec, tvec = cv2.solvePnP(
            obj_in,
            img_in,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None

        R = Rotation.from_rotvec(rvec.ravel())
        return Transform.from_components(tvec.ravel(), R)

    # ---------------------------------------------------------

    def _per_tag_errors(
        self,
        obj_pts: NDArray[np.float32],
        img_pts: NDArray[np.float32],
        rvec: NDArray[np.floating[Any]],
        tvec: NDArray[np.floating[Any]],
        tag_slices: Sequence[Tuple[int, int]],
    ) -> List[float]:
        proj, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        proj = proj.reshape(-1, 2)

        errors: List[float] = []
        for i0, i1 in tag_slices:
            err = np.mean(
                np.linalg.norm(proj[i0:i1] - img_pts[i0:i1], axis=1)
            )
            errors.append(err)

        return errors

    # ---------------------------------------------------------

    def _collect_points(
        self,
        detections: Sequence[Any],
        bundle_tags: Mapping[int, TagConfig],
    ) -> Tuple[
        Optional[NDArray[np.float32]],
        Optional[NDArray[np.float32]],
        List[Tuple[int, int]],
    ]:
        obj_pts: List[NDArray[np.float32]] = []
        img_pts: List[NDArray[np.float32]] = []
        tag_slices: List[Tuple[int, int]] = []

        idx = 0
        for det in detections:
            if det.tag_id not in bundle_tags:
                continue

            tag = bundle_tags[det.tag_id]
            corners_3d = self._tag_corners_3d(tag["size"])

            # Transform tag corners into bundle frame
            T_bundle_tag = tag["T_bundle_tag"]
            corners_bundle = T_bundle_tag.apply(corners_3d).astype(np.float32)

            obj_pts.append(corners_bundle)
            img_pts.append(det.corners.astype(np.float32))

            tag_slices.append((idx, idx + 4))
            idx += 4

        if not obj_pts:
            return None, None, []

        return (
            np.vstack(obj_pts),
            np.vstack(img_pts),
            tag_slices,
        )

    # ---------------------------------------------------------

    def _load_config(
        self, path: str, tag_family: Optional[str]
    ) -> Tuple[Dict[str, Dict[int, TagConfig]], str]:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        resolved_family = data.get("tag_family") or tag_family or "tag36h11"
        bundles: Dict[str, Dict[int, TagConfig]] = {}

        for bundle in data.get("tag_bundles", []):
            tags: Dict[int, TagConfig] = {}
            for tag in bundle["layout"]:
                T_bundle_tag = Transform.from_components(
                    [tag["x"], tag["y"], tag["z"]],
                    Rotation.from_quat(
                        [tag["qx"], tag["qy"], tag["qz"], tag["qw"]]
                    ),
                )
                tags[tag["id"]] = {
                    "size": tag["size"],
                    "T_bundle_tag": T_bundle_tag,
                }

            bundles[bundle["name"]] = tags

        return bundles, resolved_family

    # ---------------------------------------------------------

    @staticmethod
    def _tag_corners_3d(size: float) -> NDArray[np.float32]:
        s = size / 2.0
        return np.array(
            [
                [-s, -s, 0.0],
                [ s, -s, 0.0],
                [ s,  s, 0.0],
                [-s,  s, 0.0],
            ],
            dtype=np.float32,
        )
