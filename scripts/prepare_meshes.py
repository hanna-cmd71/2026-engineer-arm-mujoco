"""Prepare visual and collision meshes for the MuJoCo model."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "meshes"
VISUAL_DIR = PROJECT_ROOT / "model" / "assets" / "visual"
COLLISION_DIR = PROJECT_ROOT / "model" / "assets" / "collision"

VISUAL_FACE_LIMIT = 30_000
COLLISION_FACE_LIMIT = 2_000
MESH_STEMS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")


def simplify_mesh(mesh: trimesh.Trimesh, face_limit: int) -> trimesh.Trimesh:
    """Returns a simplified mesh with no more than the requested face count."""
    clean_mesh = mesh.copy()
    clean_mesh.process(validate=True)
    clean_mesh.remove_unreferenced_vertices()

    if len(clean_mesh.faces) <= face_limit:
        return clean_mesh

    simplified = clean_mesh.simplify_quadric_decimation(face_count=face_limit)
    simplified.process(validate=True)
    simplified.remove_unreferenced_vertices()
    return simplified


def export_mesh(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="stl")


def prepare_mesh(stem: str, force: bool) -> None:
    source_path = SOURCE_DIR / f"{stem}.STL"
    visual_path = VISUAL_DIR / f"{stem}.stl"
    collision_path = COLLISION_DIR / f"{stem}.stl"

    if not force and visual_path.exists() and collision_path.exists():
        print(f"{stem}: assets already exist, skipped")
        return

    source = trimesh.load(source_path, force="mesh", process=True)
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(f"{source_path} did not load as a triangle mesh")

    visual = simplify_mesh(source, VISUAL_FACE_LIMIT)
    collision_source = simplify_mesh(source, COLLISION_FACE_LIMIT)
    collision = collision_source.convex_hull
    collision.process(validate=True)

    export_mesh(visual, visual_path)
    export_mesh(collision, collision_path)
    print(
        f"{stem}: source={len(source.faces)} visual={len(visual.faces)} "
        f"collision={len(collision.faces)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate assets even if output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for stem in MESH_STEMS:
        prepare_mesh(stem, args.force)


if __name__ == "__main__":
    main()
