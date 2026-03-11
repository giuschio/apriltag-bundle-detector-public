import argparse

import open3d as o3d
import yaml

from apriltag_bundle_detector.transform import Rotation, Transform


def _load_bundles(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    bundles = {}
    for bundle in data.get("tag_bundles", []):
        tags = []
        for tag in bundle["layout"]:
            rot = Rotation.from_quat(
                [tag["qx"], tag["qy"], tag["qz"], tag["qw"]]
            )
            T_bundle_tag = Transform.from_components(
                [tag["x"], tag["y"], tag["z"]],
                rot,
            )
            tags.append(
                {
                    "id": tag["id"],
                    "size": tag["size"],
                    "T_bundle_tag": T_bundle_tag,
                }
            )
        bundles[bundle["name"]] = tags

    return bundles


def _estimate_axis_len(tags):
    sizes = [tag["size"] for tag in tags]
    if not sizes:
        return 0.05
    return max(sizes) * 1.5


def _make_coord_frame(size, transform=None):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if transform is not None:
        frame.transform(transform)
    return frame


def _visualize_with_labels(bundle_name, geometries, labels):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    window = app.create_window(f"Bundle: {bundle_name}", 1024, 768)
    widget = o3d.visualization.gui.SceneWidget()
    widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    for idx, geometry in enumerate(geometries):
        widget.scene.add_geometry(f"geom_{idx}", geometry, material)

    bbox = geometries[0].get_axis_aligned_bounding_box()
    for geometry in geometries[1:]:
        bbox += geometry.get_axis_aligned_bounding_box()

    widget.setup_camera(60.0, bbox, bbox.get_center())

    for text, position in labels:
        widget.add_3d_label(position, text)

    app.run()


def main():
    parser = argparse.ArgumentParser(description="Visualize bundle and tag frames with Open3D.")
    parser.add_argument("--yaml_path", default="assets/config_cube.yaml", help="Path to bundle YAML config.")
    parser.add_argument("--bundle_name", default="cube", help="Bundle name to visualize.")
    args = parser.parse_args()

    bundles = _load_bundles(args.yaml_path)
    if args.bundle_name not in bundles:
        raise ValueError(
            f"Bundle '{args.bundle_name}' not found in {args.yaml_path}."
        )

    tags = bundles[args.bundle_name]
    axis_len = _estimate_axis_len(tags)

    geometries = []
    geometries.append(
        _make_coord_frame(size=axis_len * 1.5)
    )

    labels = []
    for tag in tags:
        T = tag["T_bundle_tag"].as_matrix()
        geometries.append(
            _make_coord_frame(size=tag["size"], transform=T)
        )
        tvec, _ = tag["T_bundle_tag"].as_components()
        labels.append((f"id {tag['id']}", tvec))

    _visualize_with_labels(
        args.bundle_name,
        geometries,
        labels,
    )


if __name__ == "__main__":
    main()
