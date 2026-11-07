"""Prepare the ground-station meshes for the MuJoCo task scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "groud"
VISUAL_DIR = PROJECT_ROOT / "model" / "assets" / "scene" / "visual"
COLLISION_DIR = PROJECT_ROOT / "model" / "assets" / "scene" / "collision"
STORAGE_COLLISION_XML = (
    PROJECT_ROOT / "model" / "energy_storage_collision.xml"
)

SOURCE_MESHES = {
    "exchange_core": SOURCE_ROOT / "装配体1" / "meshes" / "base_link.STL",
    "energy_storage": SOURCE_ROOT / "qq" / "meshes" / "base_link.STL",
    "energy_unit": SOURCE_ROOT / "engery" / "meshes" / "base_link.STL",
}
VISUAL_FACE_LIMITS = {
    "exchange_core": 24_000,
    "energy_storage": 30_000,
    "energy_unit": 15_000,
}
EXCHANGE_CYLINDER_SECTIONS = 64
STORAGE_CYLINDER_COUNT = 6
STORAGE_COLLISION_FACE_THRESHOLD = 100


def simplify_mesh(mesh: trimesh.Trimesh, face_limit: int) -> trimesh.Trimesh:
    """Returns a cleaned mesh with no more than ``face_limit`` triangles."""
    result = mesh.copy()
    result.process(validate=True)
    result.remove_unreferenced_vertices()
    if len(result.faces) > face_limit:
        result = result.simplify_quadric_decimation(face_count=face_limit)
        result.process(validate=True)
        result.remove_unreferenced_vertices()
    return result


def export_mesh(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """Exports one scene mesh and creates its parent directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="stl")


def find_exchange_cylinder(
    components: list[trimesh.Trimesh],
) -> tuple[int, trimesh.Trimesh]:
    """Finds the small X-axis cylinder in the exchange core."""
    matches = []
    for index, component in enumerate(components):
        lower, upper = component.bounds
        size = upper - lower
        is_x_axis_cylinder = (
            0.08 < size[0] < 0.12
            and abs(size[1] - size[2]) < 0.005
            and lower[0] > 0.5
        )
        if is_x_axis_cylinder:
            matches.append((index, component))

    if len(matches) != 1:
        raise ValueError(
            "Expected one X-axis exchange cylinder, "
            f"found {len(matches)}"
        )
    return matches[0]


def make_clean_cylinder(
    source_cylinder: trimesh.Trimesh,
    axis: int,
    output_axis: int,
    center_at_origin: bool,
) -> trimesh.Trimesh:
    """Replaces a detailed CAD cylinder with a lightweight cylinder mesh."""
    lower, upper = source_cylinder.bounds
    size = upper - lower
    center = (lower + upper) / 2.0
    cross_axes = [index for index in range(3) if index != axis]
    radius = max(size[cross_axes]) / 2.0
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=size[axis],
        sections=EXCHANGE_CYLINDER_SECTIONS,
    )
    if output_axis != 2:
        rotation_axis = [0.0, 0.0, 0.0]
        if output_axis == 0:
            rotation_axis[1] = 1.0
        elif output_axis == 1:
            rotation_axis[0] = 1.0
        transform = trimesh.transformations.rotation_matrix(
            np.pi / 2.0,
            rotation_axis,
        )
        cylinder.apply_transform(transform)
    if not center_at_origin:
        cylinder.apply_translation(center)
    return cylinder


def make_local_z_cylinder(
    source_cylinder: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Builds a centered cylinder along the component's true principal axis."""
    vertices = np.asarray(source_cylinder.vertices)
    centroid = vertices.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.cov((vertices - centroid).T)
    )
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projected = (vertices - centroid) @ principal_axis
    length = float(projected.max() - projected.min())
    orthogonal = vertices - centroid
    radial = orthogonal - np.outer(
        orthogonal @ principal_axis,
        principal_axis,
    )
    candidate_radii = np.linalg.norm(radial, axis=1)
    radius = float(np.quantile(candidate_radii, 0.98))
    return trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=EXCHANGE_CYLINDER_SECTIONS,
    )


def prepare_exchange_core(source_mesh: trimesh.Trimesh, force: bool) -> None:
    """Splits and marks the exchange cylinder, then simplifies the body."""
    visual_path = VISUAL_DIR / "exchange_core.stl"
    collision_path = COLLISION_DIR / "exchange_core.stl"
    cylinder_visual_path = VISUAL_DIR / "exchange_cylinder.stl"
    cylinder_collision_path = COLLISION_DIR / "exchange_cylinder.stl"
    output_paths = (
        visual_path,
        collision_path,
        cylinder_visual_path,
        cylinder_collision_path,
    )
    if not force and all(path.is_file() for path in output_paths):
        print("exchange_core: assets already exist, skipped")
        return

    components = source_mesh.split(only_watertight=False)
    cylinder_index, source_cylinder = find_exchange_cylinder(components)
    body_mesh = trimesh.util.concatenate(
        [
            component
            for index, component in enumerate(components)
            if index != cylinder_index
        ]
    )
    body_visual = simplify_mesh(
        body_mesh,
        VISUAL_FACE_LIMITS["exchange_core"],
    )
    body_collision = simplify_mesh(body_mesh, 3_000).convex_hull
    body_collision.process(validate=True)
    cylinder = make_clean_cylinder(
        source_cylinder,
        axis=0,
        output_axis=0,
        center_at_origin=False,
    )

    export_mesh(body_visual, visual_path)
    export_mesh(body_collision, collision_path)
    export_mesh(cylinder, cylinder_visual_path)
    export_mesh(cylinder, cylinder_collision_path)
    print(
        f"exchange_core: body={len(body_visual.faces)} "
        f"cylinder={len(cylinder.faces)} "
        f"(source {len(source_cylinder.faces)})"
    )


def find_storage_cylinders(
    components: list[trimesh.Trimesh],
) -> list[tuple[int, trimesh.Trimesh]]:
    """Finds the six Y-axis cylinders on the storage mechanism."""
    matches = []
    for index, component in enumerate(components):
        lower, upper = component.bounds
        size = upper - lower
        center = (lower + upper) / 2.0
        is_y_axis_cylinder = (
            0.02 < size[1] < 0.12
            and abs(size[0] - size[2]) < 0.004
            and 0.02 < size[0] < 0.04
            and abs(center[2] + 0.3517) < 0.002
        )
        if is_y_axis_cylinder:
            matches.append((index, component))

    if len(matches) != STORAGE_CYLINDER_COUNT:
        raise ValueError(
            f"Expected {STORAGE_CYLINDER_COUNT} storage cylinders, "
            f"found {len(matches)}"
        )
    return sorted(
        matches,
        key=lambda item: (
            float(np.mean(item[1].bounds[:, 1])),
            float(np.mean(item[1].bounds[:, 0])),
        ),
    )


def prepare_energy_storage(
    source_mesh: trimesh.Trimesh,
    force: bool,
) -> None:
    """Splits the six storage cylinders from the main mechanism mesh."""
    visual_path = VISUAL_DIR / "energy_storage.stl"
    collision_path = COLLISION_DIR / "energy_storage.stl"
    cylinder_visual_paths = [
        VISUAL_DIR / f"energy_storage_cylinder_{index:02d}.stl"
        for index in range(1, STORAGE_CYLINDER_COUNT + 1)
    ]
    cylinder_collision_paths = [
        COLLISION_DIR / f"energy_storage_cylinder_{index:02d}.stl"
        for index in range(1, STORAGE_CYLINDER_COUNT + 1)
    ]
    output_paths = (
        visual_path,
        collision_path,
        *cylinder_visual_paths,
        *cylinder_collision_paths,
    )
    if not force and all(path.is_file() for path in output_paths):
        print("energy_storage: assets already exist, skipped")
        return

    components = source_mesh.split(only_watertight=False)
    cylinders = find_storage_cylinders(components)
    cylinder_indices = {index for index, _ in cylinders}
    body_mesh = trimesh.util.concatenate(
        [
            component
            for index, component in enumerate(components)
            if index not in cylinder_indices
        ]
    )
    body_visual = simplify_mesh(
        body_mesh,
        VISUAL_FACE_LIMITS["energy_storage"],
    )
    body_collision = simplify_mesh(body_mesh, 3_000).convex_hull
    body_collision.process(validate=True)
    export_mesh(body_visual, visual_path)
    export_mesh(body_collision, collision_path)

    collision_components = [
        (index, component)
        for index, component in enumerate(components)
        if (
            index not in cylinder_indices
            and len(component.faces) >= STORAGE_COLLISION_FACE_THRESHOLD
        )
    ]
    mesh_entries = []
    body_geom_entries = []
    outer_hull_geom_entries = []
    for collision_index, (_, component) in enumerate(collision_components):
        collision_component = component.convex_hull
        collision_component.process(validate=True)
        filename = f"energy_storage_part_{collision_index:02d}.stl"
        export_mesh(collision_component, COLLISION_DIR / filename)
        mesh_name = f"energy_storage_part_{collision_index:02d}"
        mesh_entries.append(
            f'    <mesh name="{mesh_name}" '
            f'file="scene/collision/{filename}"/>'
        )
        geom_entry = (
            f'      <geom name="{mesh_name}_collision" type="mesh" '
            f'mesh="{mesh_name}" contype="2" conaffinity="1" '
            'condim="3" group="3" rgba="0.15 0.55 0.85 0.18" '
            'friction="0.8 0.02 0.01"/>'
        )
        component_size = component.bounds[1] - component.bounds[0]
        if component_size[0] > 2.0 and component_size[1] > 1.0:
            outer_hull_geom_entries.append(geom_entry)
        else:
            body_geom_entries.append(geom_entry)

    collision_xml = "\n".join(
        [
            "<mujoco>",
            "  <asset>",
            *mesh_entries,
            "  </asset>",
            "  <worldbody>",
            '    <body name="energy_storage_solids" '
            'pos="0 0.95 -0.105" '
            'quat="0 0 0.70710678 0.70710678">',
            *body_geom_entries,
            "    </body>",
            '    <body name="energy_storage_outer_hull" '
            'pos="0 0.95 -0.105" '
            'quat="0 0 0.70710678 0.70710678">',
            *outer_hull_geom_entries,
            "    </body>",
            "  </worldbody>",
            "</mujoco>",
            "",
        ]
    )
    STORAGE_COLLISION_XML.write_text(collision_xml, encoding="utf-8")

    simplified_faces = []
    for cylinder_index in range(STORAGE_CYLINDER_COUNT):
        cylinder = make_local_z_cylinder(cylinders[cylinder_index][1])
        export_mesh(cylinder, cylinder_visual_paths[cylinder_index])
        export_mesh(cylinder, cylinder_collision_paths[cylinder_index])
        simplified_faces.append(len(cylinder.faces))
    print(
        f"energy_storage: body={len(body_visual.faces)} "
        f"cylinders={simplified_faces} "
        f"collision_parts={len(collision_components)}"
    )


def prepare_scene_meshes(force: bool) -> None:
    """Builds simplified visual meshes and collision hulls."""
    for name, source_path in SOURCE_MESHES.items():
        visual_path = VISUAL_DIR / f"{name}.stl"
        collision_path = COLLISION_DIR / f"{name}.stl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source mesh: {source_path}")
        if not force and visual_path.is_file() and collision_path.is_file():
            print(f"{name}: assets already exist, skipped")
            continue

        source_mesh = trimesh.load(
            source_path,
            force="mesh",
            process=True,
        )
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise TypeError(f"{source_path} did not load as a triangle mesh")

        if name == "exchange_core":
            prepare_exchange_core(source_mesh, force)
            continue
        if name == "energy_storage":
            prepare_energy_storage(source_mesh, force)
            continue

        visual_mesh = simplify_mesh(
            source_mesh,
            VISUAL_FACE_LIMITS[name],
        )
        collision_mesh = simplify_mesh(source_mesh, 3_000).convex_hull
        collision_mesh.process(validate=True)
        export_mesh(visual_mesh, visual_path)
        export_mesh(collision_mesh, collision_path)
        print(
            f"{name}: source={len(source_mesh.faces)} "
            f"visual={len(visual_mesh.faces)} "
            f"collision={len(collision_mesh.faces)}"
        )


def parse_args() -> argparse.Namespace:
    """Parses command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate assets even when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the scene mesh preparation."""
    prepare_scene_meshes(parse_args().force)


if __name__ == "__main__":
    main()
