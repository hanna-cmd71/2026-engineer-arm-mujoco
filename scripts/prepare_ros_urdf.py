"""Generate a ROS/MoveIt URDF with installed visual and collision assets."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "JGarm.urdf"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "ros2_ws"
    / "src"
    / "jgarm_description"
    / "urdf"
    / "JGarm.ros.urdf"
)


def mesh_uri(mesh, category: str) -> str:
    stem = Path(mesh.get("filename")).stem.lower()
    return f"package://jgarm_description/model/assets/{category}/{stem}.stl"


def main() -> None:
    tree = ElementTree.parse(SOURCE_PATH)
    root = tree.getroot()
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is not None:
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                mesh.set("filename", mesh_uri(mesh, "visual"))
        collision = link.find("collision")
        if collision is not None:
            mesh = collision.find("geometry/mesh")
            if mesh is not None:
                mesh.set("filename", mesh_uri(mesh, "collision"))

    ElementTree.indent(tree, space="  ")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        OUTPUT_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
