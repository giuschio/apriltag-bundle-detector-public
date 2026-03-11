import os
import argparse
import numpy as np

import cv2

from apriltag_bundle_detector.solver import AprilTagBundleDetector

from realsense_utils import SceneCamera


def _draw_axes(image, camera_matrix, dist_coeffs, transform, axis_len, label):
    tvec, rot = transform.as_components()
    rvec = rot.as_rotvec().reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    axes = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float32,
    )
    img_pts, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    origin = tuple(img_pts[0])
    cv2.line(image, origin, tuple(img_pts[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(img_pts[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(img_pts[3]), (255, 0, 0), 2)

    cv2.putText(
        image,
        label,
        (origin[0] + 6, origin[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _estimate_axis_len(detector):
    sizes = []
    for tags in detector.bundles.values():
        for tag in tags.values():
            sizes.append(tag["size"])
    if not sizes:
        return 0.05
    return max(sizes) * 1.5


def _draw_origin_circle(image, camera_matrix, dist_coeffs, transform, radius=6):
    tvec, rot = transform.as_components()
    rvec = rot.as_rotvec().reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    img_pts, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
    center = tuple(img_pts.reshape(-1, 2).astype(int)[0])
    cv2.circle(image, center, radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-display",
        choices=("axes", "circle"),
        default="axes",
        help="Display full axes or just a circle at the reference center.",
    )
    args = parser.parse_args()

    config_path = 'assets/config_cube.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    camera = SceneCamera()
    camera_matrix = camera.color_intrinsics_matrix
    dist_coeffs = camera.dist_coeffs

    detector = AprilTagBundleDetector(config_path, camera_matrix, distortion_coeffs=dist_coeffs)
    axis_len = _estimate_axis_len(detector)

    cv2.namedWindow("apriltag bundle", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.capture(debug=False)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector(gray, debug=False)

            if args.ref_display == "axes":
                for name, transform in results.items():
                    _draw_axes(frame, camera_matrix, dist_coeffs, transform, axis_len, name)
            else:
                for transform in results.values():
                    _draw_origin_circle(frame, camera_matrix, dist_coeffs, transform)

            cv2.imshow("apriltag bundle", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.finalize()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
